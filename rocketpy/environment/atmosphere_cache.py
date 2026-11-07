"""Disk cache for downloaded atmospheric datasets (netCDF profiles and JSON).

Cache root defaults to ``~/.rocketpy_cache/atmosphere``. Override the root with
the ``ROCKETPY_CACHE`` environment variable (the ``atmosphere`` subfolder is
created under it), or disable caching entirely by setting it to one of ``0``,
``off``, ``false``, ``no``, ``none`` or ``disabled``.

OPeNDAP "Best" aggregations are virtual catalogs, not downloadable files. For
Forecast/Ensemble shortcuts this module therefore stores the **location-and-time
profiles** RocketPy extracts after the first successful fetch, as a compact
``.nc`` file, together with every derived attribute ``Environment`` publishes
for that model (date range, grid bounds and the raw interpolation inputs) so a
cache hit reproduces the same object a fresh download would have produced.
Windy responses are stored as ``.json``.

Forecasts are re-issued by their providers on a fixed cycle, so cache entries
expire after ``ROCKETPY_CACHE_TTL`` seconds (default: 6 hours, matching the GFS
cycle). Reanalysis data is immutable and never expires. Set the TTL to ``0`` to
disable expiry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

CACHE_ENV_VAR = "ROCKETPY_CACHE"
CACHE_TTL_ENV_VAR = "ROCKETPY_CACHE_TTL"
DEFAULT_CACHE_ROOT = Path.home() / ".rocketpy_cache"
PROFILE_FORMAT_ATTR = "rocketpy_atmosphere_profiles_v2"
JSON_FORMAT_KEY = "rocketpy_cache_format"
CREATED_AT_ATTR = "rocketpy_cache_created_at"
DEFAULT_CACHE_TTL_SECONDS = 6 * 3600

#: Model kinds whose data never changes once published, so they never expire.
IMMUTABLE_KINDS = frozenset({"reanalysis"})

_DISABLED_VALUES = frozenset({"", "0", "off", "false", "no", "none", "disabled"})
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Raised by ``netCDF4.Dataset`` when a file cannot be opened at all: a damaged,
#: empty or truncated file gives ``OSError``, a missing one ``FileNotFoundError``
#: and an unwritable location ``PermissionError`` (both ``OSError`` subclasses).
#: ``RuntimeError`` is listed because netCDF4 surfaces some HDF5-level failures
#: that way, and the installed version is not pinned.
CACHE_OPEN_ERRORS = (OSError, RuntimeError)

#: Everything the above can raise, plus what reading or writing the contents of
#: an otherwise-openable cache file can raise. Verified against netCDF4 1.7.4:
#: a missing variable raises ``KeyError``; a missing or non-numeric attribute
#: ``AttributeError``; a wrong-length array or an undeclared dimension
#: ``ValueError``; an invalid dtype or attribute type ``TypeError``; and
#: touching a closed dataset ``RuntimeError``.
#:
#: The list is deliberately explicit rather than a bare ``except Exception``: a
#: corrupt cache entry must degrade to a re-download, but a ``NameError`` or any
#: other genuine defect in this module has to keep propagating instead of being
#: silently reported to the user as a cache miss.
CACHE_FILE_ERRORS = CACHE_OPEN_ERRORS + (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)

# Scalar metadata persisted as netCDF attributes, with the caster used on read.
_SCALAR_METADATA = (
    ("atmospheric_model_interval", float),
    ("atmospheric_model_init_lat", float),
    ("atmospheric_model_end_lat", float),
    ("atmospheric_model_init_lon", float),
    ("atmospheric_model_end_lon", float),
    ("lat_index", int),
    ("lon_index", int),
)
_DATE_METADATA = ("atmospheric_model_init_date", "atmospheric_model_end_date")
_PAIR_METADATA = ("lat_array", "lon_array", "time_array")
#: Raw interpolation inputs, shaped ``(raw_level, lat_pair, lon_pair)``.
_CORNER_METADATA = ("geopotentials", "wind_us", "wind_vs", "temperatures")


# ---------------------------------------------------------------------------
# Cache location and policy
# ---------------------------------------------------------------------------


def is_cache_enabled() -> bool:
    """Return False when ``ROCKETPY_CACHE`` opts out of disk caching."""
    raw = os.environ.get(CACHE_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in _DISABLED_VALUES


def get_cache_root() -> Path:
    """Return the root cache directory (honors ``ROCKETPY_CACHE``)."""
    return Path(os.environ.get(CACHE_ENV_VAR) or DEFAULT_CACHE_ROOT).expanduser()


def get_atmosphere_cache_dir() -> Path:
    """Return the atmosphere subdirectory under the cache root."""
    return get_cache_root() / "atmosphere"


def get_cache_ttl() -> float:
    """Return the cache lifetime in seconds (``0`` disables expiry)."""
    raw = os.environ.get(CACHE_TTL_ENV_VAR)
    if raw is None:
        return float(DEFAULT_CACHE_TTL_SECONDS)
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid {CACHE_TTL_ENV_VAR}='{raw}'. "
            f"Using the default of {DEFAULT_CACHE_TTL_SECONDS} seconds.",
            UserWarning,
            stacklevel=2,
        )
        return float(DEFAULT_CACHE_TTL_SECONDS)


def is_entry_expired(created_at, kind) -> bool:
    """Return True when a cache entry written at ``created_at`` is too old.

    Entries whose ``kind`` is in :data:`IMMUTABLE_KINDS` never expire, and a
    TTL of ``0`` disables expiry for every kind.
    """
    if kind in IMMUTABLE_KINDS:
        return False
    ttl = get_cache_ttl()
    if ttl <= 0:
        return False
    try:
        age = time.time() - float(created_at)
    except (TypeError, ValueError):
        return True  # unreadable timestamp: treat as stale and re-fetch
    return age > ttl


def ensure_atmosphere_cache_dir() -> Path | None:
    """Create the atmosphere cache directory.

    Returns
    -------
    pathlib.Path or None
        The directory path, or ``None`` if caching is disabled or the
        directory could not be created.
    """
    if not is_cache_enabled():
        return None
    cache_dir = get_atmosphere_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except OSError as exc:
        warnings.warn(
            f"Could not create atmosphere cache directory '{cache_dir}': {exc}. "
            "Caching disabled for this request.",
            UserWarning,
            stacklevel=2,
        )
        return None


def clear_atmosphere_cache() -> bool:
    """Delete every cached atmosphere file. Returns False on failure."""
    cache_dir = get_atmosphere_cache_dir()
    if not cache_dir.is_dir():
        return True
    try:
        shutil.rmtree(cache_dir)
        return True
    except OSError as exc:
        warnings.warn(
            f"Could not clear atmosphere cache '{cache_dir}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        return False


def sanitize_cache_key(key: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def cache_path_for(key: str, suffix: str) -> Path:
    """Build a cache file path for ``key`` with the given suffix (e.g. ``.nc``)."""
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return get_atmosphere_cache_dir() / f"{sanitize_cache_key(key)}{suffix}"


def build_atmosphere_cache_key(
    kind: str,
    source: str,
    latitude: float,
    longitude: float,
    datetime_date,
    variant: str = "",
) -> str:
    """Build a stable cache key for a Forecast/Ensemble/Windy request.

    ``variant`` distinguishes requests that hit the same source and location
    but decode it differently (a different variable dictionary or pressure
    conversion factor), which would otherwise collide on one file.
    """
    if datetime_date is None:
        date_part = "nodate"
    else:
        date_part = datetime_date.strftime("%Y%m%d%H")
    key = f"{kind}_{source}_{latitude:.4f}_{longitude:.4f}_{date_part}"
    if variant:
        key = f"{key}_{variant}"
    return sanitize_cache_key(key)


def is_remote_url(path_or_url) -> bool:
    """Return True if ``path_or_url`` looks like an HTTP(S)/OPeNDAP URL."""
    if not isinstance(path_or_url, str):
        return False
    lowered = path_or_url.lower()
    return lowered.startswith(("http://", "https://", "dods://"))


# ---------------------------------------------------------------------------
# Raw byte / JSON helpers
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` atomically. Returns False on failure."""
    if ensure_atmosphere_cache_dir() is None:
        return False
    temp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            handle.write(data)
            temp_name = handle.name
        Path(temp_name).replace(path)
        return True
    except OSError as exc:
        warnings.warn(
            f"Failed to write atmosphere cache file '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        _discard(temp_name)
        return False


def _discard(path) -> None:
    """Best-effort removal of a leftover temporary file."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def load_json_cache(path: Path, kind: str = "windy") -> dict | None:
    """Load a JSON cache file, or ``None`` if missing/stale/unreadable."""
    if not is_cache_enabled() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Failed to read cached atmosphere JSON '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if (
        not isinstance(envelope, dict)
        or envelope.get(JSON_FORMAT_KEY) != PROFILE_FORMAT_ATTR
    ):
        return None
    if is_entry_expired(envelope.get(CREATED_AT_ATTR), kind):
        return None
    payload = envelope.get("payload")
    return payload if isinstance(payload, dict) else None


def save_json_cache(path: Path, payload: dict) -> bool:
    """Serialize ``payload`` as JSON to ``path``. Returns False on failure."""
    if not is_cache_enabled():
        return False
    envelope = {
        JSON_FORMAT_KEY: PROFILE_FORMAT_ATTR,
        CREATED_AT_ATTR: time.time(),
        "payload": payload,
    }
    try:
        data = json.dumps(envelope).encode("utf-8")
    except (TypeError, ValueError) as exc:
        warnings.warn(
            f"Failed to serialize atmosphere JSON for cache '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return atomic_write_bytes(path, data)


# ---------------------------------------------------------------------------
# Model metadata persistence
# ---------------------------------------------------------------------------


def _write_metadata(dataset, metadata) -> None:
    """Store the ``Environment`` model metadata on an open netCDF dataset."""
    metadata = metadata or {}
    _write_metadata_attributes(dataset, metadata)
    _write_metadata_arrays(dataset, metadata)


def _write_metadata_attributes(dataset, metadata) -> None:
    """Store the scalar, date and coordinate-pair metadata as attributes."""
    for name, _ in _SCALAR_METADATA:
        value = metadata.get(name)
        if value is not None:
            dataset.setncattr(name, float(value))

    for name in _DATE_METADATA:
        value = metadata.get(name)
        if isinstance(value, datetime):
            dataset.setncattr(name, value.strftime(_DATE_FORMAT))

    for name in _PAIR_METADATA:
        value = metadata.get(name)
        if value is not None:
            dataset.setncattr(name, [float(item) for item in value])


def _write_metadata_arrays(dataset, metadata) -> None:
    """Store the raw interpolation inputs as netCDF variables."""
    raw_levels = metadata.get("levels")
    if raw_levels is None:
        return

    raw_levels = np.asarray(raw_levels)
    dataset.setncattr(
        "levels_integer", int(np.issubdtype(raw_levels.dtype, np.integer))
    )
    dataset.createDimension("raw_level", raw_levels.size)
    dataset.createDimension("lat_pair", 2)
    dataset.createDimension("lon_pair", 2)

    variable = dataset.createVariable("raw_levels", "f8", ("raw_level",))
    variable[:] = np.asarray(raw_levels, dtype=float)

    raw_height = metadata.get("height")
    if raw_height is not None:
        variable = dataset.createVariable("raw_height", "f8", ("raw_level",))
        variable[:] = _filled(raw_height)

    for name in _CORNER_METADATA:
        values = metadata.get(name)
        if values is None:
            continue
        variable = dataset.createVariable(
            name, "f8", ("raw_level", "lat_pair", "lon_pair")
        )
        variable[:] = _filled(values)


def _read_metadata(dataset) -> dict:
    """Rebuild the ``Environment`` model metadata from an open netCDF dataset."""
    metadata = {}

    for name, caster in _SCALAR_METADATA:
        if hasattr(dataset, name):
            metadata[name] = caster(dataset.getncattr(name))

    for name in _DATE_METADATA:
        if hasattr(dataset, name):
            metadata[name] = datetime.strptime(dataset.getncattr(name), _DATE_FORMAT)

    for name in _PAIR_METADATA:
        if hasattr(dataset, name):
            metadata[name] = [
                float(item) for item in np.atleast_1d(dataset.getncattr(name))
            ]

    if "raw_levels" in dataset.variables:
        levels = np.array(dataset.variables["raw_levels"][:], dtype=float)
        if int(getattr(dataset, "levels_integer", 0)):
            levels = levels.astype(np.int64)
        metadata["levels"] = levels

    if "raw_height" in dataset.variables:
        metadata["height"] = np.array(dataset.variables["raw_height"][:], dtype=float)

    for name in _CORNER_METADATA:
        if name in dataset.variables:
            metadata[name] = np.array(dataset.variables[name][:], dtype=float)

    return metadata


def _filled(values):
    """Return a plain float array, replacing any masked entries with NaN."""
    return np.ma.filled(np.ma.asarray(values).astype(float), np.nan)


def _open_for_write(path: Path):
    """Create a temporary netCDF file next to ``path``. Returns (dataset, temp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".nc.tmp"
    ) as handle:
        temp_path = Path(handle.name)
    return netCDF4.Dataset(temp_path, mode="w", format="NETCDF4"), temp_path


def _set_common_attributes(dataset, kind, elevation, max_expected_height) -> None:
    """Write the attributes shared by every cache file format."""
    dataset.setncattr("rocketpy_cache_format", PROFILE_FORMAT_ATTR)
    dataset.setncattr("rocketpy_cache_kind", kind)
    dataset.setncattr(CREATED_AT_ATTR, float(time.time()))
    dataset.setncattr("elevation", float(elevation))
    dataset.setncattr("max_expected_height", float(max_expected_height))


def _open_valid_cache(path: Path, expected_kind=None):
    """Open a cache file, returning ``None`` if absent, foreign or expired."""
    if not is_cache_enabled() or not path.is_file():
        return None
    try:
        dataset = netCDF4.Dataset(path, mode="r")
    except CACHE_OPEN_ERRORS as exc:
        # A cache miss must never be louder than the download it replaces.
        warnings.warn(
            f"Failed to open atmosphere cache '{path}': {exc}. Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None

    fmt = getattr(dataset, "rocketpy_cache_format", None)
    kind = getattr(dataset, "rocketpy_cache_kind", None)
    if fmt != PROFILE_FORMAT_ATTR or (
        expected_kind is not None and kind != expected_kind
    ):
        dataset.close()
        return None
    if is_entry_expired(getattr(dataset, CREATED_AT_ATTR, None), kind):
        dataset.close()
        return None
    return dataset


# ---------------------------------------------------------------------------
# Forecast / Reanalysis profiles
# ---------------------------------------------------------------------------


def write_profile_netcdf(
    path: Path,
    *,
    height,
    pressure,
    temperature,
    wind_u,
    wind_v,
    elevation: float,
    max_expected_height: float,
    kind: str = "forecast",
    metadata=None,
) -> bool:
    """Write extracted atmospheric profiles to a compact local netCDF file."""
    if ensure_atmosphere_cache_dir() is None:
        return False

    columns = (
        ("height", height, "m"),
        ("pressure", pressure, "Pa"),
        ("temperature", temperature, "K"),
        ("wind_u", wind_u, "m s-1"),
        ("wind_v", wind_v, "m s-1"),
    )
    temp_path = None
    try:
        dataset, temp_path = _open_for_write(path)
        try:
            _set_common_attributes(dataset, kind, elevation, max_expected_height)
            dataset.createDimension("level", np.asarray(height, dtype=float).size)
            for name, values, units in columns:
                variable = dataset.createVariable(name, "f8", ("level",))
                variable.units = units
                variable[:] = np.asarray(values, dtype=float)
            _write_metadata(dataset, metadata)
        finally:
            dataset.close()
        temp_path.replace(path)
        return True
    except CACHE_FILE_ERRORS as exc:
        warnings.warn(
            f"Failed to write atmosphere profile cache '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        _discard(temp_path)
        return False


def read_profile_netcdf(path: Path) -> dict | None:
    """Read a profile netCDF written by :func:`write_profile_netcdf`."""
    dataset = _open_valid_cache(path)
    if dataset is None:
        return None
    try:
        profiles = {
            "kind": getattr(dataset, "rocketpy_cache_kind", "forecast"),
            "elevation": float(dataset.getncattr("elevation")),
            "max_expected_height": float(dataset.getncattr("max_expected_height")),
        }
        for name in ("height", "pressure", "temperature", "wind_u", "wind_v"):
            profiles[name] = np.array(dataset.variables[name][:], dtype=float)
        profiles["metadata"] = _read_metadata(dataset)
        return profiles
    except CACHE_FILE_ERRORS as exc:
        warnings.warn(
            f"Failed to read atmosphere profile cache '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
# Ensemble profiles
# ---------------------------------------------------------------------------


def write_ensemble_profile_netcdf(
    path: Path,
    *,
    levels,
    height_ensemble,
    temperature_ensemble,
    wind_u_ensemble,
    wind_v_ensemble,
    elevation: float,
    max_expected_height: float,
    metadata=None,
) -> bool:
    """Write ensemble member profiles to a compact local netCDF file."""
    if ensure_atmosphere_cache_dir() is None:
        return False

    height_ensemble = np.asarray(height_ensemble, dtype=float)
    if height_ensemble.ndim != 2:
        return False
    num_members, num_levels = height_ensemble.shape
    columns = (
        ("height", height_ensemble, "m"),
        ("temperature", temperature_ensemble, "K"),
        ("wind_u", wind_u_ensemble, "m s-1"),
        ("wind_v", wind_v_ensemble, "m s-1"),
    )

    temp_path = None
    try:
        dataset, temp_path = _open_for_write(path)
        try:
            _set_common_attributes(dataset, "ensemble", elevation, max_expected_height)
            dataset.createDimension("member", num_members)
            dataset.createDimension("level", num_levels)

            level_var = dataset.createVariable("level", "f8", ("level",))
            level_var.units = "Pa"
            level_var[:] = np.asarray(levels, dtype=float)

            for name, values, units in columns:
                variable = dataset.createVariable(name, "f8", ("member", "level"))
                variable.units = units
                variable[:] = np.asarray(values, dtype=float)
            _write_metadata(dataset, metadata)
        finally:
            dataset.close()
        temp_path.replace(path)
        return True
    except CACHE_FILE_ERRORS as exc:
        warnings.warn(
            f"Failed to write ensemble atmosphere cache '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        _discard(temp_path)
        return False


def read_ensemble_profile_netcdf(path: Path) -> dict | None:
    """Read an ensemble profile netCDF written by :func:`write_ensemble_profile_netcdf`."""
    dataset = _open_valid_cache(path, expected_kind="ensemble")
    if dataset is None:
        return None
    try:
        profiles = {
            "elevation": float(dataset.getncattr("elevation")),
            "max_expected_height": float(dataset.getncattr("max_expected_height")),
            "levels": np.array(dataset.variables["level"][:], dtype=float),
        }
        for key, name in (
            ("height_ensemble", "height"),
            ("temperature_ensemble", "temperature"),
            ("wind_u_ensemble", "wind_u"),
            ("wind_v_ensemble", "wind_v"),
        ):
            profiles[key] = np.array(dataset.variables[name][:], dtype=float)
        profiles["metadata"] = _read_metadata(dataset)
        return profiles
    except CACHE_FILE_ERRORS as exc:
        warnings.warn(
            f"Failed to read ensemble atmosphere cache '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None
    finally:
        dataset.close()
