"""Best-effort helpers for listing NOAA/NCEP atmosphere datasets on THREDDS.

RocketPy's forecast shortcuts historically hard-coded OPeNDAP URLs (and, for
NOMADS GrADS models, probed calendar times). NOMADS OPeNDAP has been retired,
so discovery is based on the Unidata THREDDS catalogs that already back the
GFS/NAM/RAP/HRRR/AIGFS fetchers.

Limitations
-----------
- Catalog layout can change without notice; treat results as advisory.
- Listing uses HTTP XML catalogs, not NOMADS OpenDAP directory pages.
- ``fetch_latest_noaa_dataset`` opens the resolved OPeNDAP URL via netCDF4
  (same pattern as the existing ``fetch_*_file_return_dataset`` helpers). It
  does not write a local GRIB/NetCDF file to disk.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urljoin, urlparse

import netCDF4
import requests

from rocketpy.environment.fetchers.base import MAX_RETRY_DELAY_SECONDS

THREDDS_ROOT = "https://thredds.ucar.edu/thredds"
THREDDS_CATALOG_ROOT = f"{THREDDS_ROOT}/catalog/"
THREDDS_OPENDAP_ROOT = f"{THREDDS_ROOT}/dodsC/"
FORECAST_MODELS_CATALOG_URL = f"{THREDDS_CATALOG_ROOT}idd/forecastModels.xml"

# Collections used by Environment's latest-model shortcuts.
NOAA_MODEL_COLLECTIONS = {
    "GFS": "grib/NCEP/GFS/Global_0p25deg",
    "NAM": "grib/NCEP/NAM/CONUS_12km",
    "RAP": "grib/NCEP/RAP/CONUS_13km",
    "HRRR": "grib/NCEP/HRRR/CONUS_2p5km",
    "AIGFS": "grib/NCEP/AIGFS/Global_0p25deg",
}

_THREDDS_NS = {
    "thredds": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0",
    "xlink": "http://www.w3.org/1999/xlink",
}
_RUN_TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}_\d{4})")
_DEFAULT_TIMEOUT_SECONDS = 30


def _local_tag(tag: str) -> str:
    """Strip Clark-notation namespace from an ElementTree tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _absolute_catalog_url(href: str, base_url: str) -> str:
    """Resolve a catalogRef href against a catalog URL."""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        parsed = urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}{href}"
    return urljoin(base_url, href)


def _collection_path_from_catalog_url(catalog_url: str) -> str | None:
    """Extract ``grib/NCEP/...`` path from a THREDDS catalog URL, if present."""
    marker = "/catalog/"
    if marker not in catalog_url:
        return None
    path = catalog_url.split(marker, 1)[1]
    if path.endswith("/catalog.xml"):
        path = path[: -len("/catalog.xml")]
    elif path.endswith("catalog.xml"):
        path = path[: -len("catalog.xml")].rstrip("/")
    return path or None


def _fetch_catalog_xml(catalog_url: str, timeout: float = _DEFAULT_TIMEOUT_SECONDS):
    """GET a THREDDS catalog XML document and return its root element."""
    try:
        response = requests.get(catalog_url, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Unable to reach NOAA/THREDDS catalog at {catalog_url}."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            "Unable to list NOAA/THREDDS datasets: "
            f"HTTP {response.status_code} for {catalog_url}."
        )

    try:
        return ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError(
            f"Invalid THREDDS catalog XML received from {catalog_url}."
        ) from exc


def _opendap_url_from_path(url_path: str) -> str:
    """Build an OPeNDAP URL for a THREDDS dataset ``urlPath``."""
    return urljoin(THREDDS_OPENDAP_ROOT, url_path.lstrip("/"))


def _run_sort_key(identifier: str):
    """Sort key that prefers identifiers embedding ``YYYYMMDD_HHMM``."""
    match = _RUN_TIMESTAMP_RE.search(identifier)
    if match:
        try:
            return datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M")
        except ValueError:
            pass
    return datetime.min


def resolve_noaa_collection_path(model_or_path: str) -> str:
    """Map a model shortcut or collection path to a THREDDS collection path.

    Parameters
    ----------
    model_or_path : str
        Shortcut such as ``\"GFS\"`` or a collection path such as
        ``\"grib/NCEP/GFS/Global_0p25deg\"``.

    Returns
    -------
    str
        Collection path without a leading slash.

    Raises
    ------
    ValueError
        If ``model_or_path`` is not a known shortcut and does not look like a
        collection path.
    """
    if not isinstance(model_or_path, str) or not model_or_path.strip():
        raise ValueError("model_or_path must be a non-empty string.")

    key = model_or_path.strip()
    mapped = NOAA_MODEL_COLLECTIONS.get(key.upper())
    if mapped is not None:
        return mapped

    normalized = key.lstrip("/")
    if normalized.startswith("grib/"):
        return normalized.removesuffix("/catalog.xml").rstrip("/")

    raise ValueError(
        f"Unknown NOAA model collection {model_or_path!r}. "
        f"Known shortcuts: {sorted(NOAA_MODEL_COLLECTIONS)}. "
        "Pass a THREDDS collection path such as "
        "'grib/NCEP/GFS/Global_0p25deg' instead."
    )


def collection_catalog_url(model_or_path: str) -> str:
    """Return the THREDDS ``catalog.xml`` URL for a model collection."""
    collection = resolve_noaa_collection_path(model_or_path)
    return f"{THREDDS_CATALOG_ROOT}{collection}/catalog.xml"


def list_noaa_atmosphere_datasets(
    catalog_url: str = FORECAST_MODELS_CATALOG_URL,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> list[dict]:
    """List available NCEP atmosphere model collections from THREDDS.

    Parameters
    ----------
    catalog_url : str, optional
        Root forecast-models catalog URL. Defaults to Unidata's NCEP models
        catalog (the same server family used by RocketPy's GFS/NAM/… fetchers).
    timeout : float, optional
        HTTP timeout in seconds.

    Returns
    -------
    list of dict
        Each entry has ``name``, ``catalog_url``, ``collection_path`` and
        ``opendap_best_url`` (the latter is ``None`` until a collection
        catalog is inspected).
    """
    root = _fetch_catalog_xml(catalog_url, timeout=timeout)
    datasets = []
    seen = set()

    for ref in root.findall(".//thredds:catalogRef", _THREDDS_NS):
        href = ref.get(f"{{{_THREDDS_NS['xlink']}}}href") or ref.get("href")
        if not href:
            continue
        name = (
            ref.get(f"{{{_THREDDS_NS['xlink']}}}title")
            or ref.get("name")
            or ref.get("ID")
            or href
        )
        absolute = _absolute_catalog_url(href, catalog_url)
        collection_path = _collection_path_from_catalog_url(absolute)
        key = (name, absolute)
        if key in seen:
            continue
        seen.add(key)
        datasets.append(
            {
                "name": name,
                "catalog_url": absolute,
                "collection_path": collection_path,
                "opendap_best_url": (
                    _opendap_url_from_path(f"{collection_path}/Best")
                    if collection_path
                    else None
                ),
            }
        )

    if not datasets:
        raise RuntimeError(
            f"No NOAA/THREDDS dataset collections found in {catalog_url}."
        )

    return datasets


def list_noaa_dataset_identifiers(
    model_or_path: str,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    include_aggregates: bool = True,
) -> list[str]:
    """List dataset identifiers available inside a NOAA/NCEP collection.

    Parameters
    ----------
    model_or_path : str
        Shortcut (``\"GFS\"``, ``\"NAM\"``, …) or a THREDDS collection path.
    timeout : float, optional
        HTTP timeout in seconds.
    include_aggregates : bool, optional
        When ``True`` (default), include virtual aggregates such as ``Best``
        and ``TwoD`` alongside individual forecast-run identifiers.

    Returns
    -------
    list of str
        Identifiers suitable for :func:`build_noaa_opendap_url`. Forecast-run
        filenames (``*.grib2``) are sorted newest-first when timestamps can be
        parsed; aggregate names are appended afterward.
    """
    catalog_url = collection_catalog_url(model_or_path)
    root = _fetch_catalog_xml(catalog_url, timeout=timeout)

    runs = []
    aggregates = []

    for element in root.iter():
        tag = _local_tag(element.tag)
        if tag not in {"dataset", "catalogRef"}:
            continue

        url_path = element.get("urlPath")
        name = (
            element.get(f"{{{_THREDDS_NS['xlink']}}}title")
            or element.get("name")
            or element.get("ID")
            or ""
        )

        if url_path in {None, "", "latest.xml"}:
            continue

        identifier = url_path.rsplit("/", 1)[-1]
        if identifier.lower().endswith(".grib2") or _RUN_TIMESTAMP_RE.search(
            identifier
        ):
            runs.append(identifier)
        elif include_aggregates and identifier in {"Best", "TwoD"}:
            aggregates.append(identifier)
        elif include_aggregates and name and "Best" in name and identifier:
            aggregates.append(identifier)

    # Preserve order while dropping duplicates.
    unique_runs = list(dict.fromkeys(runs))
    unique_runs.sort(key=_run_sort_key, reverse=True)
    unique_aggregates = list(dict.fromkeys(aggregates))
    identifiers = unique_runs + unique_aggregates

    if not identifiers:
        raise RuntimeError(
            "No dataset identifiers found in NOAA/THREDDS collection catalog "
            f"{catalog_url}."
        )

    return identifiers


def build_noaa_opendap_url(model_or_path: str, identifier: str) -> str:
    """Build an OPeNDAP URL for a collection dataset identifier."""
    collection = resolve_noaa_collection_path(model_or_path)
    return _opendap_url_from_path(f"{collection}/{identifier}")


def get_latest_noaa_dataset_identifier(
    model_or_path: str,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Return the newest forecast-run identifier for a collection.

    Prefers THREDDS ``latest.xml`` when present. Falls back to the maximum
    timestamp among listed ``*.grib2`` runs, then to the ``Best`` aggregate.

    Parameters
    ----------
    model_or_path : str
        Shortcut or collection path.
    timeout : float, optional
        HTTP timeout in seconds.

    Returns
    -------
    str
        Dataset identifier (for example
        ``\"GFS_Global_0p25deg_20260810_1800.grib2\"`` or ``\"Best\"``).
    """
    collection = resolve_noaa_collection_path(model_or_path)
    latest_catalog_url = f"{THREDDS_CATALOG_ROOT}{collection}/latest.xml"

    try:
        root = _fetch_catalog_xml(latest_catalog_url, timeout=timeout)
    except RuntimeError:
        root = None

    if root is not None:
        for element in root.iter():
            if _local_tag(element.tag) != "dataset":
                continue
            url_path = element.get("urlPath")
            name = element.get("name")
            if url_path and url_path != "latest.xml":
                return url_path.rsplit("/", 1)[-1]
            if name and name.lower().endswith(".grib2"):
                return name

    identifiers = list_noaa_dataset_identifiers(
        model_or_path, timeout=timeout, include_aggregates=True
    )
    for identifier in identifiers:
        if identifier.lower().endswith(".grib2") or _RUN_TIMESTAMP_RE.search(
            identifier
        ):
            return identifier

    if "Best" in identifiers:
        return "Best"

    return identifiers[0]


def get_latest_noaa_opendap_url(
    model_or_path: str = "GFS",
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Resolve the OPeNDAP URL for the latest dataset in a collection."""
    identifier = get_latest_noaa_dataset_identifier(model_or_path, timeout=timeout)
    return build_noaa_opendap_url(model_or_path, identifier)


def fetch_latest_noaa_dataset(
    model_or_path: str = "GFS",
    max_attempts: int = 10,
    base_delay: float = 2,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
):
    """Open the latest NOAA/NCEP dataset for a model via OPeNDAP.

    Parameters
    ----------
    model_or_path : str, optional
        Shortcut such as ``\"GFS\"`` (default) or a THREDDS collection path.
    max_attempts : int, optional
        Maximum netCDF4 open attempts. Default is 10.
    base_delay : float, optional
        Base exponential backoff delay in seconds. Default is 2.
    timeout : float, optional
        HTTP timeout used while resolving the catalog entry.

    Returns
    -------
    netCDF4.Dataset
        Open dataset handle.

    Raises
    ------
    RuntimeError
        If the catalog cannot be resolved or all open attempts fail.
    """
    file_url = get_latest_noaa_opendap_url(model_or_path, timeout=timeout)
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            return netCDF4.Dataset(file_url)
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    raise RuntimeError(
        "Unable to load the latest NOAA/THREDDS weather dataset through " + file_url
    )
