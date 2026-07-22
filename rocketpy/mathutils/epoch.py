"""Time representation used by RocketPy simulations.

The :class:`Epoch` class supports both absolute UTC epochs and a relative
simulation origin.  Relative epochs keep date-less legacy simulations valid,
while models that depend on Earth orientation or ephemerides can explicitly
require an absolute epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


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

    @property
    def jd_tt(self) -> float:
        """Approximate Julian date in Terrestrial Time.

        RocketPy ships no implicit network-backed leap-second service. The
        current value uses the modern TAI-UTC offset (37 s), appropriate for
        dates from 2017 onward. A data-provider-backed implementation can
        override this through the environment Earth-orientation model.
        """
        return self.jd_utc + (37.0 + 32.184) / 86400.0

    @property
    def jd_tdb(self) -> float:
        """Approximate Julian date in Barycentric Dynamical Time."""
        # Millisecond-level periodic TT/TDB terms are immaterial for the
        # built-in low-fidelity ephemerides. High-fidelity providers receive
        # the Epoch and may perform their own conversion.
        return self.jd_tt

    @property
    def earth_rotation_angle(self) -> float:
        """IAU Earth rotation angle in radians."""
        self.require_absolute("Earth rotation")
        days_from_j2000 = self.jd_utc - 2451545.0
        turns = 0.7790572732640 + 1.00273781191135448 * days_from_j2000
        return (turns % 1.0) * 2.0 * 3.141592653589793

    @property
    def greenwich_mean_sidereal_time(self) -> float:
        """Approximate IAU-1982 Greenwich mean sidereal time in radians."""
        centuries = (self.jd_utc - 2451545.0) / 36525.0
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
