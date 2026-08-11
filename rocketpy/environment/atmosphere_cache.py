"""Disk cache for downloaded atmospheric datasets (netCDF profiles and JSON).

Cache root defaults to ``~/.rocketpy_cache/atmosphere``. Override with the
``ROCKETPY_CACHE`` environment variable (the ``atmosphere`` subfolder is
created under that root).

OPeNDAP "Best" aggregations are virtual catalogs, not downloadable files. For
Forecast/Ensemble shortcuts this module therefore stores the **location-and-time
profiles** RocketPy extracts after the first successful fetch, as a compact
``.nc`` file. Subsequent identical requests load those profiles from disk.
Windy responses are stored as ``.json``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from pathlib import Path

import netCDF4
import numpy as np

CACHE_ENV_VAR = "ROCKETPY_CACHE"
DEFAULT_CACHE_ROOT = Path.home() / ".rocketpy_cache"
PROFILE_FORMAT_ATTR = "rocketpy_atmosphere_profiles_v1"


def get_cache_root() -> Path:
    """Return the root cache directory (honors ``ROCKETPY_CACHE``)."""
    return Path(os.environ.get(CACHE_ENV_VAR, DEFAULT_CACHE_ROOT)).expanduser()


def get_atmosphere_cache_dir() -> Path:
    """Return the atmosphere subdirectory under the cache root."""
    return get_cache_root() / "atmosphere"


def ensure_atmosphere_cache_dir() -> Path | None:
    """Create the atmosphere cache directory.

    Returns
    -------
    pathlib.Path or None
        The directory path, or ``None`` if creation failed (caching disabled).
    """
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
) -> str:
    """Build a stable cache key for a Forecast/Ensemble/Windy request."""
    if datetime_date is None:
        date_part = "nodate"
    else:
        date_part = datetime_date.strftime("%Y%m%d%H")
    return sanitize_cache_key(
        f"{kind}_{source}_{latitude:.4f}_{longitude:.4f}_{date_part}"
    )


def is_remote_url(path_or_url) -> bool:
    """Return True if ``path_or_url`` looks like an HTTP(S)/OPeNDAP URL."""
    if not isinstance(path_or_url, str):
        return False
    lowered = path_or_url.lower()
    return lowered.startswith(("http://", "https://", "dods://"))


def atomic_write_bytes(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` atomically. Returns False on failure."""
    cache_dir = ensure_atmosphere_cache_dir()
    if cache_dir is None:
        return False
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
        try:
            Path(temp_name).unlink(missing_ok=True)
        except (OSError, NameError):
            pass
        return False


def load_json_cache(path: Path) -> dict | None:
    """Load a JSON cache file, or ``None`` if missing/unreadable."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Failed to read cached atmosphere JSON '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None


def save_json_cache(path: Path, payload: dict) -> bool:
    """Serialize ``payload`` as JSON to ``path``. Returns False on failure."""
    try:
        data = json.dumps(payload).encode("utf-8")
    except (TypeError, ValueError) as exc:
        warnings.warn(
            f"Failed to serialize atmosphere JSON for cache '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return atomic_write_bytes(path, data)


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
) -> bool:
    """Write extracted atmospheric profiles to a compact local netCDF file."""
    cache_dir = ensure_atmosphere_cache_dir()
    if cache_dir is None:
        return False

    height = np.asarray(height, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    wind_u = np.asarray(wind_u, dtype=float)
    wind_v = np.asarray(wind_v, dtype=float)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, suffix=".nc.tmp"
        ) as handle:
            temp_path = Path(handle.name)

        dataset = netCDF4.Dataset(temp_path, mode="w", format="NETCDF4")
        try:
            dataset.setncattr("rocketpy_cache_format", PROFILE_FORMAT_ATTR)
            dataset.setncattr("rocketpy_cache_kind", kind)
            dataset.setncattr("elevation", float(elevation))
            dataset.setncattr("max_expected_height", float(max_expected_height))

            dataset.createDimension("level", height.size)
            for name, values, units in (
                ("height", height, "m"),
                ("pressure", pressure, "Pa"),
                ("temperature", temperature, "K"),
                ("wind_u", wind_u, "m s-1"),
                ("wind_v", wind_v, "m s-1"),
            ):
                variable = dataset.createVariable(name, "f8", ("level",))
                variable.units = units
                variable[:] = values
        finally:
            dataset.close()

        temp_path.replace(path)
        return True
    except OSError as exc:
        warnings.warn(
            f"Failed to write atmosphere profile cache '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        try:
            temp_path.unlink(missing_ok=True)
        except (OSError, NameError):
            pass
        return False


def read_profile_netcdf(path: Path) -> dict | None:
    """Read a profile netCDF written by :func:`write_profile_netcdf`."""
    if not path.is_file():
        return None
    try:
        dataset = netCDF4.Dataset(path, mode="r")
    except OSError as exc:
        warnings.warn(
            f"Failed to open atmosphere profile cache '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        fmt = getattr(dataset, "rocketpy_cache_format", None)
        if fmt != PROFILE_FORMAT_ATTR:
            warnings.warn(
                f"Ignoring atmosphere cache '{path}' with unknown format '{fmt}'.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return {
            "kind": getattr(dataset, "rocketpy_cache_kind", "forecast"),
            "elevation": float(dataset.getncattr("elevation")),
            "max_expected_height": float(dataset.getncattr("max_expected_height")),
            "height": np.array(dataset.variables["height"][:], dtype=float),
            "pressure": np.array(dataset.variables["pressure"][:], dtype=float),
            "temperature": np.array(dataset.variables["temperature"][:], dtype=float),
            "wind_u": np.array(dataset.variables["wind_u"][:], dtype=float),
            "wind_v": np.array(dataset.variables["wind_v"][:], dtype=float),
        }
    except (AttributeError, KeyError, ValueError, OSError) as exc:
        warnings.warn(
            f"Failed to read atmosphere profile cache '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None
    finally:
        dataset.close()


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
) -> bool:
    """Write ensemble member profiles to a compact local netCDF file."""
    cache_dir = ensure_atmosphere_cache_dir()
    if cache_dir is None:
        return False

    levels = np.asarray(levels, dtype=float)
    height_ensemble = np.asarray(height_ensemble, dtype=float)
    temperature_ensemble = np.asarray(temperature_ensemble, dtype=float)
    wind_u_ensemble = np.asarray(wind_u_ensemble, dtype=float)
    wind_v_ensemble = np.asarray(wind_v_ensemble, dtype=float)
    num_members, num_levels = height_ensemble.shape

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, suffix=".nc.tmp"
        ) as handle:
            temp_path = Path(handle.name)

        dataset = netCDF4.Dataset(temp_path, mode="w", format="NETCDF4")
        try:
            dataset.setncattr("rocketpy_cache_format", PROFILE_FORMAT_ATTR)
            dataset.setncattr("rocketpy_cache_kind", "ensemble")
            dataset.setncattr("elevation", float(elevation))
            dataset.setncattr("max_expected_height", float(max_expected_height))

            dataset.createDimension("member", num_members)
            dataset.createDimension("level", num_levels)

            level_var = dataset.createVariable("level", "f8", ("level",))
            level_var.units = "Pa"
            level_var[:] = levels

            for name, values, units in (
                ("height", height_ensemble, "m"),
                ("temperature", temperature_ensemble, "K"),
                ("wind_u", wind_u_ensemble, "m s-1"),
                ("wind_v", wind_v_ensemble, "m s-1"),
            ):
                variable = dataset.createVariable(name, "f8", ("member", "level"))
                variable.units = units
                variable[:] = values
        finally:
            dataset.close()

        temp_path.replace(path)
        return True
    except OSError as exc:
        warnings.warn(
            f"Failed to write ensemble atmosphere cache '{path}': {exc}.",
            UserWarning,
            stacklevel=2,
        )
        try:
            temp_path.unlink(missing_ok=True)
        except (OSError, NameError):
            pass
        return False


def read_ensemble_profile_netcdf(path: Path) -> dict | None:
    """Read an ensemble profile netCDF written by :func:`write_ensemble_profile_netcdf`."""
    if not path.is_file():
        return None
    try:
        dataset = netCDF4.Dataset(path, mode="r")
    except OSError as exc:
        warnings.warn(
            f"Failed to open ensemble atmosphere cache '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        fmt = getattr(dataset, "rocketpy_cache_format", None)
        kind = getattr(dataset, "rocketpy_cache_kind", None)
        if fmt != PROFILE_FORMAT_ATTR or kind != "ensemble":
            return None
        return {
            "elevation": float(dataset.getncattr("elevation")),
            "max_expected_height": float(dataset.getncattr("max_expected_height")),
            "levels": np.array(dataset.variables["level"][:], dtype=float),
            "height_ensemble": np.array(dataset.variables["height"][:], dtype=float),
            "temperature_ensemble": np.array(
                dataset.variables["temperature"][:], dtype=float
            ),
            "wind_u_ensemble": np.array(dataset.variables["wind_u"][:], dtype=float),
            "wind_v_ensemble": np.array(dataset.variables["wind_v"][:], dtype=float),
        }
    except (AttributeError, KeyError, ValueError, OSError) as exc:
        warnings.warn(
            f"Failed to read ensemble atmosphere cache '{path}': {exc}. "
            "Fetching fresh data.",
            UserWarning,
            stacklevel=2,
        )
        return None
    finally:
        dataset.close()
