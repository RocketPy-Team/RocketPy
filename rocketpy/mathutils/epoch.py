"""Time representation used by RocketPy simulations.

The :class:`Epoch` class supports both absolute UTC epochs and a relative
simulation origin.  Relative epochs keep date-less legacy simulations valid,
while models that depend on Earth orientation or ephemerides can explicitly
require an absolute epoch.
"""

from __future__ import annotations

import csv
import logging
import math
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger("rocketpy")

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


class EarthOrientationProvider:
    """Cached IERS Earth-orientation and leap-second data.

    Data is loaded explicitly by high-fidelity datum components before a
    simulation starts. If no cache is available, the provider downloads the
    IERS ``finals2000A`` table and the official leap-second table. A failed
    download falls back to zero polar motion/UT1 correction and RocketPy's
    bundled historical leap-second table, with a warning.
    """

    IERS_URL = "https://datacenter.iers.org/data/csv/finals2000A.all.csv"
    LEAP_SECONDS_URL = "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"

    # UTC effective date and TAI-UTC, covering every leap second since 1972.
    _BUILTIN_LEAPS = (
        (datetime(1972, 1, 1, tzinfo=UTC), 10.0),
        (datetime(1972, 7, 1, tzinfo=UTC), 11.0),
        (datetime(1973, 1, 1, tzinfo=UTC), 12.0),
        (datetime(1974, 1, 1, tzinfo=UTC), 13.0),
        (datetime(1975, 1, 1, tzinfo=UTC), 14.0),
        (datetime(1976, 1, 1, tzinfo=UTC), 15.0),
        (datetime(1977, 1, 1, tzinfo=UTC), 16.0),
        (datetime(1978, 1, 1, tzinfo=UTC), 17.0),
        (datetime(1979, 1, 1, tzinfo=UTC), 18.0),
        (datetime(1980, 1, 1, tzinfo=UTC), 19.0),
        (datetime(1981, 7, 1, tzinfo=UTC), 20.0),
        (datetime(1982, 7, 1, tzinfo=UTC), 21.0),
        (datetime(1983, 7, 1, tzinfo=UTC), 22.0),
        (datetime(1985, 7, 1, tzinfo=UTC), 23.0),
        (datetime(1988, 1, 1, tzinfo=UTC), 24.0),
        (datetime(1990, 1, 1, tzinfo=UTC), 25.0),
        (datetime(1991, 1, 1, tzinfo=UTC), 26.0),
        (datetime(1992, 7, 1, tzinfo=UTC), 27.0),
        (datetime(1993, 7, 1, tzinfo=UTC), 28.0),
        (datetime(1994, 7, 1, tzinfo=UTC), 29.0),
        (datetime(1996, 1, 1, tzinfo=UTC), 30.0),
        (datetime(1997, 7, 1, tzinfo=UTC), 31.0),
        (datetime(1999, 1, 1, tzinfo=UTC), 32.0),
        (datetime(2006, 1, 1, tzinfo=UTC), 33.0),
        (datetime(2009, 1, 1, tzinfo=UTC), 34.0),
        (datetime(2012, 7, 1, tzinfo=UTC), 35.0),
        (datetime(2015, 7, 1, tzinfo=UTC), 36.0),
        (datetime(2017, 1, 1, tzinfo=UTC), 37.0),
    )

    def __init__(self, cache_directory=None, automatic_download=True):
        self.cache_directory = Path(
            cache_directory
            or Path(tempfile.gettempdir()).joinpath("rocketpy-earth-orientation")
        )
        self.automatic_download = bool(automatic_download)
        self._eop = None
        self._leaps = np.array(
            [
                [entry.timestamp() / 86400.0 + 40587.0, offset]
                for entry, offset in self._BUILTIN_LEAPS
            ],
            dtype=float,
        )
        self._prepared = False
        self._warned_bounds = set()

    def prepare(self):
        """Load cached data and, when allowed, refresh missing tables."""
        if self._prepared:
            return self
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        eop_path = self.cache_directory / "finals2000A.all.csv"
        leap_path = self.cache_directory / "Leap_Second.dat"
        self._eop = self._load_or_download_eop(eop_path)
        downloaded_leaps = self._load_or_download_leaps(leap_path)
        if downloaded_leaps is not None:
            self._leaps = downloaded_leaps
        self._prepared = True
        return self

    def _download(self, url, destination, label):
        if not self.automatic_download:
            return False
        logger.info("Downloading %s to %s.", label, destination)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(destination)
        except Exception as exc:  # pragma: no cover - network dependent
            partial.unlink(missing_ok=True)
            logger.warning(
                "Could not download %s (%s). Using the available fallback.",
                label,
                exc,
            )
            return False
        logger.info("Finished downloading %s.", label)
        return True

    def _load_or_download_eop(self, path):
        if not path.exists() and not self._download(
            self.IERS_URL, path, "IERS Earth-orientation data"
        ):
            return None
        try:
            rows = []
            with path.open(encoding="utf-8", errors="ignore") as stream:
                reader = csv.reader(stream, delimiter=";")
                next(reader, None)
                for row in reader:
                    try:
                        mjd = float(row[0])
                    except (IndexError, TypeError, ValueError):
                        continue

                    def value(*indices):
                        for index in indices:
                            try:
                                candidate = float(row[index])
                            except (IndexError, TypeError, ValueError):
                                continue
                            if math.isfinite(candidate):
                                return candidate
                        return 0.0

                    # MJD, xp, yp, UT1-UTC, LOD(ms), dX and dY. Prefer
                    # observed Bulletin A columns and fall back to predicted
                    # columns. Raw indices follow finals2000A.all.csv.
                    rows.append(
                        [
                            mjd,
                            value(5, 28),
                            value(7, 29),
                            value(14, 31),
                            value(16),
                            value(23, 35),
                            value(25, 36),
                        ]
                    )
            data = np.asarray(rows, dtype=float)
            if len(data) < 2:
                raise ValueError("table contains fewer than two records")
            return data
        except Exception as exc:  # pragma: no cover - corrupt external cache
            logger.warning(
                "Could not read IERS Earth-orientation cache %s (%s). "
                "Using zero Earth-orientation corrections.",
                path,
                exc,
            )
            return None

    def _load_or_download_leaps(self, path):
        if not path.exists() and not self._download(
            self.LEAP_SECONDS_URL, path, "IERS leap-second data"
        ):
            return None
        try:
            rows = []
            with path.open(encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    fields = line.split()
                    if len(fields) < 5 or not fields[0].replace(".", "").isdigit():
                        continue
                    rows.append([float(fields[0]), float(fields[4])])
            data = np.asarray(rows, dtype=float)
            if not len(data):
                raise ValueError("table contains no records")
            return data
        except Exception as exc:  # pragma: no cover - corrupt external cache
            logger.warning(
                "Could not read leap-second cache %s (%s). "
                "Using RocketPy's bundled leap-second history.",
                path,
                exc,
            )
            return None

    def values(self, mjd_utc):
        """Return ``UT1-UTC, xp, yp, dX, dY, LOD`` in SI/radian units."""
        self.prepare()
        if self._eop is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        mjd = float(mjd_utc)
        table = self._eop
        if mjd <= table[0, 0]:
            row = table[0]
            bound = "before"
        elif mjd >= table[-1, 0]:
            row = table[-1]
            bound = "after"
        else:
            bound = None
        if bound is not None and bound not in self._warned_bounds:
            logger.warning(
                "Epoch MJD %.3f lies outside the IERS table; using its nearest record.",
                mjd,
            )
            self._warned_bounds.add(bound)
        if bound is None:
            upper = int(np.searchsorted(table[:, 0], mjd))
            lower = upper - 1
            weight = (mjd - table[lower, 0]) / (table[upper, 0] - table[lower, 0])
            row = (1.0 - weight) * table[lower] + weight * table[upper]
        return (
            float(row[3]),
            float(row[1]) * ARCSEC_TO_RAD,
            float(row[2]) * ARCSEC_TO_RAD,
            float(row[5]) * ARCSEC_TO_RAD / 1000.0,
            float(row[6]) * ARCSEC_TO_RAD / 1000.0,
            float(row[4]) / 1000.0,
        )

    def tai_minus_utc(self, mjd_utc):
        """Return the applicable historical TAI-UTC offset in seconds."""
        index = np.searchsorted(self._leaps[:, 0], float(mjd_utc), side="right") - 1
        return float(self._leaps[max(0, min(index, len(self._leaps) - 1)), 1])


DEFAULT_EARTH_ORIENTATION = EarthOrientationProvider()


@dataclass(frozen=True)
class Epoch:
    """An absolute UTC instant or a relative simulation time origin.

    Parameters
    ----------
    seconds : float
        For absolute epochs, seconds from the Unix epoch. For relative epochs,
        seconds from an arbitrary simulation origin.
    is_absolute : bool
        Whether ``seconds`` identifies an absolute UTC instant.
    """

    seconds: float
    is_absolute: bool = True

    @classmethod
    def from_datetime(cls, value: datetime) -> "Epoch":
        """Create an absolute epoch from a timezone-aware datetime."""
        if value.tzinfo is None:
            raise ValueError("Absolute epochs require a timezone-aware datetime.")
        return cls(value.astimezone(UTC).timestamp(), is_absolute=True)

    @classmethod
    def relative_origin(cls) -> "Epoch":
        """Create the origin used by simulations without an absolute date."""
        return cls(0.0, is_absolute=False)

    def require_absolute(self, feature: str = "This operation") -> None:
        """Raise when an absolute epoch is required but unavailable."""
        if not self.is_absolute:
            raise ValueError(
                f"{feature} requires an absolute epoch. Set Environment.date "
                "to a timezone-aware datetime."
            )

    def to_datetime(self) -> datetime:
        """Return the corresponding UTC datetime."""
        self.require_absolute("Datetime conversion")
        return datetime.fromtimestamp(self.seconds, tz=UTC)

    @property
    def jd_utc(self) -> float:
        """Julian date in UTC."""
        self.require_absolute("UTC Julian date evaluation")
        return self.seconds / 86400.0 + 2440587.5

    @property
    def mjd_utc(self) -> float:
        """Modified Julian date in UTC."""
        return self.jd_utc - 2400000.5

    def prepare_earth_orientation(self) -> "Epoch":
        """Prepare implicit IERS and leap-second data before propagation."""
        self.require_absolute("Earth-orientation preparation")
        DEFAULT_EARTH_ORIENTATION.prepare()
        return self

    @property
    def jd_ut1(self) -> float:
        """Julian date in UT1 using the active IERS table."""
        delta_ut1, *_ = DEFAULT_EARTH_ORIENTATION.values(self.mjd_utc)
        return self.jd_utc + delta_ut1 / 86400.0

    @property
    def jd_tai(self) -> float:
        """Julian date in International Atomic Time."""
        return self.jd_utc + (
            DEFAULT_EARTH_ORIENTATION.tai_minus_utc(self.mjd_utc) / 86400.0
        )

    @property
    def jd_tt(self) -> float:
        """Approximate Julian date in Terrestrial Time.

        Leap seconds are resolved from the cached IERS table, with a bundled
        historical fallback.
        """
        return self.jd_tai + 32.184 / 86400.0

    @property
    def tt_century(self) -> float:
        """Julian centuries in Terrestrial Time since J2000."""
        return (self.jd_tt - 2451545.0) / 36525.0

    @property
    def jd_tdb(self) -> float:
        """Approximate Julian date in Barycentric Dynamical Time.

        Uses the truncated Fairhead-Bretagnon series given by Montenbruck &
        Gill, *Satellite Orbits* (2000), Eq. 3.118. The approximation is
        periodic, remains below two milliseconds from TT, and is suitable for
        the planetary ephemeris arguments used by RocketPy.
        """
        centuries = self.tt_century
        delta_seconds = (
            0.001657 * math.sin(628.3076 * centuries + 6.2401)
            + 0.000022 * math.sin(575.3385 * centuries + 4.2970)
            + 0.000014 * math.sin(1256.6152 * centuries + 6.1969)
            + 0.000005 * math.sin(606.9777 * centuries + 4.0212)
            + 0.000005 * math.sin(52.9691 * centuries + 0.4444)
            + 0.000002 * math.sin(21.3299 * centuries + 5.5431)
            + 0.000010 * centuries * math.sin(628.3076 * centuries + 4.2490)
        )
        return self.jd_tt + delta_seconds / 86400.0

    @property
    def earth_rotation_angle(self) -> float:
        """IAU Earth rotation angle in radians."""
        self.require_absolute("Earth rotation")
        days_from_j2000 = self.jd_ut1 - 2451545.0
        turns = 0.7790572732640 + 1.00273781191135448 * days_from_j2000
        return (turns % 1.0) * 2.0 * 3.141592653589793

    @property
    def earth_orientation(self) -> np.ndarray:
        """Polar motion and celestial-pole offsets ``xp, yp, dX, dY``."""
        _, xp, yp, dx, dy, _ = DEFAULT_EARTH_ORIENTATION.values(self.mjd_utc)
        result = np.array([xp, yp, dx, dy], dtype=float)
        result.setflags(write=False)
        return result

    @property
    def length_of_day(self) -> float:
        """Observed excess length of day in seconds."""
        *_, lod = DEFAULT_EARTH_ORIENTATION.values(self.mjd_utc)
        return lod

    @property
    def greenwich_mean_sidereal_time(self) -> float:
        """Approximate IAU-1982 Greenwich mean sidereal time in radians."""
        centuries = (self.jd_ut1 - 2451545.0) / 36525.0
        seconds = (
            67310.54841
            + (876600.0 * 3600.0 + 8640184.812866) * centuries
            + 0.093104 * centuries**2
            - 6.2e-6 * centuries**3
        )
        return (seconds % 86400.0) * (2.0 * 3.141592653589793 / 86400.0)

    def __add__(self, seconds: float | timedelta) -> "Epoch":
        if isinstance(seconds, timedelta):
            seconds = seconds.total_seconds()
        return Epoch(self.seconds + float(seconds), self.is_absolute)

    def __sub__(self, other: "Epoch" | float | timedelta):
        if isinstance(other, Epoch):
            if self.is_absolute != other.is_absolute:
                raise ValueError("Cannot subtract absolute and relative epochs.")
            return self.seconds - other.seconds
        if isinstance(other, timedelta):
            other = other.total_seconds()
        return Epoch(self.seconds - float(other), self.is_absolute)
