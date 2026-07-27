"""Composable gravity models for RocketPy environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import resources

import numpy as np

from rocketpy.mathutils.compilation import numbify
from rocketpy.mathutils.epoch import Epoch
from rocketpy.mathutils.reference_frame import (
    WGS84,
    EarthDatum,
    ReferenceFrame,
    transform_kinematics,
)

FLOAT64_EPSILON = 2.220446049250313e-16


class Gravity(ABC):
    """Vector-valued gravitational acceleration model."""

    requires_absolute_epoch = False

    @abstractmethod
    def acceleration(
        self,
        epoch: Epoch,
        position: np.ndarray,
        *,
        frame: ReferenceFrame,
        datum: EarthDatum = WGS84,
    ) -> np.ndarray:
        """Return gravitational acceleration in the input frame, in m/s²."""

    def magnitude_at_height(self, height: float, *, datum: EarthDatum = WGS84) -> float:
        """Return equatorial gravity magnitude at a geodetic height."""
        position = np.array([datum.semi_major_axis + height, 0.0, 0.0])
        return float(
            np.linalg.norm(
                self.acceleration(
                    Epoch.relative_origin(),
                    position,
                    frame=ReferenceFrame.ITRF,
                    datum=datum,
                )
            )
        )


class ZeroGravity(Gravity):
    """Gravity model returning zero acceleration."""

    def acceleration(self, epoch, position, **kwargs) -> np.ndarray:
        del epoch, position, kwargs
        return np.zeros(3)


class VerticalGravity(Gravity):
    """Legacy scalar gravity acting along negative local Z."""

    def __init__(self, magnitude):
        self.magnitude = magnitude

    def acceleration(self, epoch, position, *, frame, **kwargs) -> np.ndarray:
        del epoch, kwargs
        if ReferenceFrame.coerce(frame) != ReferenceFrame.FLAT_EARTH:
            raise ValueError("VerticalGravity is only valid in the flat-Earth frame.")
        height = float(np.asarray(position)[2])
        value = (
            self.magnitude.get_value_opt(height)
            if hasattr(self.magnitude, "get_value_opt")
            else self.magnitude(height)
        )
        return np.array([0.0, 0.0, -float(value)])

    def magnitude_at_height(self, height, **kwargs) -> float:
        del kwargs
        return float(
            self.magnitude.get_value_opt(height)
            if hasattr(self.magnitude, "get_value_opt")
            else self.magnitude(height)
        )


class SomiglianaGravity(VerticalGravity):
    """Somigliana-derived apparent gravity for a non-rotating local frame.

    Somigliana surface gravity already contains the centrifugal correction.
    Inheriting :class:`VerticalGravity` deliberately makes use in GCRF/ITRF an
    error, preventing Earth rotation from being counted twice.
    """


class DefaultGravity(Gravity):
    """Default gravity for both local and Earth-centered Flight states.

    Flat-Earth evaluation preserves the configured legacy vertical-gravity
    function. Earth-centered evaluation uses spherical gravity, making the
    default Environment composition immediately usable for orbital Flight.
    """

    def __init__(self, vertical_magnitude, gravitational_parameter=None):
        self.local = SomiglianaGravity(vertical_magnitude)
        self.earth_centered = SphericalGravity(gravitational_parameter)

    def acceleration(self, epoch, position, *, frame, datum=WGS84):
        frame = ReferenceFrame.coerce(frame)
        model = (
            self.local if frame == ReferenceFrame.FLAT_EARTH else self.earth_centered
        )
        return model.acceleration(epoch, position, frame=frame, datum=datum)

    def magnitude_at_height(self, height, *, datum=WGS84):
        return self.local.magnitude_at_height(height, datum=datum)


class SphericalGravity(Gravity):
    """Newtonian point-mass gravity."""

    def __init__(self, gravitational_parameter: float | None = None):
        self.gravitational_parameter = gravitational_parameter

    def acceleration(self, epoch, position, *, datum=WGS84, **kwargs) -> np.ndarray:
        del epoch, kwargs
        position = np.asarray(position, dtype=float)
        radius = np.linalg.norm(position)
        if radius <= np.finfo(float).eps:
            raise ValueError("Gravity is undefined at the central-body origin.")
        mu = self.gravitational_parameter or datum.gravitational_parameter
        return -mu * position / radius**3


class ZonalGravity(SphericalGravity):
    """Earth gravity including configurable J2 and J3 zonal harmonics."""

    def __init__(
        self,
        gravitational_parameter: float | None = None,
        j2: float = 1.08262668e-3,
        j3: float = -2.5324105e-6,
    ):
        super().__init__(gravitational_parameter)
        self.j2 = j2
        self.j3 = j3

    def acceleration(
        self, epoch, position, *, frame=ReferenceFrame.GCRF, datum=WGS84, **kwargs
    ) -> np.ndarray:
        del kwargs
        frame = ReferenceFrame.coerce(frame)
        if frame not in (ReferenceFrame.GCRF, ReferenceFrame.ITRF):
            raise ValueError("ZonalGravity requires the GCRF or ITRF frame.")
        if frame == ReferenceFrame.GCRF:
            position_fixed, _, _ = datum._to_ecef_coordinates(epoch, position)
        else:
            position_fixed = np.asarray(position, dtype=float)
        x, y, z = position_fixed
        radius_squared = float(np.dot(position_fixed, position_fixed))
        radius = np.sqrt(radius_squared)
        if radius <= np.finfo(float).eps:
            raise ValueError("Gravity is undefined at the central-body origin.")
        mu = self.gravitational_parameter or datum.gravitational_parameter
        reference_radius = datum.semi_major_axis
        z_ratio = z / radius
        j2_factor = 1.5 * self.j2 * (reference_radius / radius) ** 2
        base = -mu / radius**3
        acceleration_fixed = base * np.array(
            [
                x * (1.0 + j2_factor * (1.0 - 5.0 * z_ratio**2)),
                y * (1.0 + j2_factor * (1.0 - 5.0 * z_ratio**2)),
                z * (1.0 + j2_factor * (3.0 - 5.0 * z_ratio**2)),
            ]
        )
        if self.j3:
            common = 0.5 * self.j3 * mu * reference_radius**3 / radius**7
            acceleration_fixed += np.array(
                [
                    5.0 * common * x * z * (7.0 * z_ratio**2 - 3.0),
                    5.0 * common * y * z * (7.0 * z_ratio**2 - 3.0),
                    common
                    / radius_squared
                    * (
                        3.0 * radius_squared**2
                        - 30.0 * radius_squared * z**2
                        + 35.0 * z**4
                    ),
                ]
            )
        if frame == ReferenceFrame.ITRF:
            return acceleration_fixed
        acceleration, _, _ = datum._to_eci_coordinates(epoch, acceleration_fixed)
        return acceleration


class SphericalHarmonicGravity(Gravity):
    """Fully normalized EGM2008 spherical-harmonic gravity.

    The bundled coefficient set supports degree and order through 90. Gravity
    is evaluated in ITRF and transformed back to the input frame.
    """

    requires_absolute_epoch = True

    def __init__(
        self,
        degree: int = 20,
        order: int | None = None,
        *,
        include_tides: bool = False,
        sun=None,
        moon=None,
    ):
        self.degree = int(degree)
        self.order = self.degree if order is None else int(order)
        if self.degree < 2 or self.degree > 90:
            raise ValueError("EGM2008 degree must be between 2 and 90.")
        if self.order < 0 or self.order > self.degree:
            raise ValueError("Gravity order must be between zero and degree.")
        self.include_tides = bool(include_tides)
        self.sun = sun
        self.moon = moon
        if self.include_tides and (self.sun is None or self.moon is None):
            raise ValueError(
                "Solid-Earth tides require explicit Sun and Moon ThirdBody models."
            )
        coefficient_path = resources.files("rocketpy.environment").joinpath(
            "data/egm2008.npz"
        )
        with np.load(coefficient_path) as coefficients:
            self.cosine = np.array(
                coefficients["C"][: self.degree + 1, : self.order + 1],
                dtype=float,
            )
            self.sine = np.array(
                coefficients["S"][: self.degree + 1, : self.order + 1],
                dtype=float,
            )

    def acceleration(self, epoch, position, *, frame, datum=WGS84) -> np.ndarray:
        frame = ReferenceFrame.coerce(frame)
        if frame not in (ReferenceFrame.GCRF, ReferenceFrame.ITRF):
            raise ValueError(
                "SphericalHarmonicGravity requires the GCRF or ITRF frame."
            )
        epoch.require_absolute("Spherical-harmonic gravity")
        if frame == ReferenceFrame.GCRF:
            position_itrf, _, _ = transform_kinematics(
                epoch,
                position,
                source=ReferenceFrame.GCRF,
                target=ReferenceFrame.ITRF,
                datum=datum,
            )
        else:
            position_itrf = np.asarray(position, dtype=float)
        cosine, sine = self.cosine, self.sine
        if self.include_tides:
            sun = self.sun
            moon = self.moon
            sun_itrf, _, _ = transform_kinematics(
                epoch,
                sun.position(epoch),
                source=ReferenceFrame.GCRF,
                target=ReferenceFrame.ITRF,
                datum=datum,
            )
            moon_itrf, _, _ = transform_kinematics(
                epoch,
                moon.position(epoch),
                source=ReferenceFrame.GCRF,
                target=ReferenceFrame.ITRF,
                datum=datum,
            )
            cosine, sine = _apply_solid_earth_tides(
                cosine,
                sine,
                sun_itrf,
                moon_itrf,
                datum.gravitational_parameter,
                sun.gravitational_parameter,
                moon.gravitational_parameter,
                datum.semi_major_axis,
            )
        acceleration_itrf = _harmonic_acceleration(
            position_itrf,
            cosine,
            sine,
            datum.gravitational_parameter,
            datum.semi_major_axis,
            self.degree,
            self.order,
        )
        if frame == ReferenceFrame.ITRF:
            return acceleration_itrf
        acceleration_gcrf, _, _ = transform_kinematics(
            epoch,
            acceleration_itrf,
            source=ReferenceFrame.ITRF,
            target=ReferenceFrame.GCRF,
            datum=datum,
        )
        return acceleration_gcrf


@numbify(
    signature=(
        "float64[::1](float64[::1], float64[:, ::1], float64[:, ::1], "
        "float64, float64, int64, int64)"
    ),
    nopython=True,
    cache=True,
)
def _harmonic_acceleration(position, cosine, sine, mu, radius_reference, degree, order):
    """Evaluate fully normalized harmonics using a Legendre recursion."""
    x, y, z = position
    radius = float(np.linalg.norm(position))
    if radius <= FLOAT64_EPSILON:
        raise ValueError("Gravity is undefined at the central-body origin.")
    q = radius_reference / radius
    sin_latitude = max(-1.0, min(1.0, z / radius))
    cos_latitude = max(float(np.sqrt(1.0 - sin_latitude**2)), 1e-12)
    tan_latitude = sin_latitude / cos_latitude
    longitude = np.arctan2(y, x)
    sin_longitude, cos_longitude = np.sin(longitude), np.cos(longitude)

    sectoral = np.zeros(order + 1)
    q_power = np.zeros(degree + 1)
    sectoral[0] = 1.0
    if order >= 1:
        sectoral[1] = np.sqrt(3.0) * cos_latitude
    q_power[0] = 1.0
    if degree >= 1:
        q_power[1] = q
    for index in range(2, degree + 1):
        q_power[index] = q * q_power[index - 1]
        if index <= order:
            sectoral[index] = (
                cos_latitude * np.sqrt(1.0 + 0.5 / index) * sectoral[index - 1]
            )

    longitude_component = latitude_component = radial_component = 0.0
    sin_multiple, cos_multiple = 0.0, 1.0
    for m in range(order + 1):
        p_nm = sectoral[m]
        derivative_p_nm = -m * p_nm * tan_latitude
        p_previous = p_nm
        p_previous_previous = 0.0
        c_mm = cosine[m, m] if m >= 2 else 0.0
        s_mm = sine[m, m] if m >= 2 else 0.0
        qc, qs = q_power[m] * c_mm, q_power[m] * s_mm
        sum_cos, sum_sin = qc * p_nm, qs * p_nm
        sum_lat_cos = qc * derivative_p_nm
        sum_lat_sin = qs * derivative_p_nm
        sum_rad_cos = (m + 1.0) * qc * p_nm
        sum_rad_sin = (m + 1.0) * qs * p_nm

        for n in range(m + 1, degree + 1):
            a_nm = np.sqrt(((2.0 * n - 1.0) * (2.0 * n + 1.0)) / ((n - m) * (n + m)))
            b_nm = np.sqrt(
                ((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0))
                / ((n - m) * (n + m) * (2.0 * n - 3.0))
            )
            f_nm = np.sqrt(((n**2 - m**2) * (2.0 * n + 1.0)) / (2.0 * n - 1.0))
            p_nm = a_nm * sin_latitude * p_previous - b_nm * p_previous_previous
            derivative_p_nm = (
                -n * p_nm * tan_latitude + f_nm * p_previous / cos_latitude
            )
            p_previous_previous, p_previous = p_previous, p_nm
            if n >= 2 and m < cosine.shape[1]:
                qc = q_power[n] * cosine[n, m]
                qs = q_power[n] * sine[n, m]
                sum_cos += qc * p_nm
                sum_sin += qs * p_nm
                sum_lat_cos += qc * derivative_p_nm
                sum_lat_sin += qs * derivative_p_nm
                sum_rad_cos += (n + 1.0) * qc * p_nm
                sum_rad_sin += (n + 1.0) * qs * p_nm

        longitude_component += m * (sum_cos * sin_multiple - sum_sin * cos_multiple)
        latitude_component += sum_lat_cos * cos_multiple + sum_lat_sin * sin_multiple
        radial_component += sum_rad_cos * cos_multiple + sum_rad_sin * sin_multiple
        next_cos = cos_longitude * cos_multiple - sin_multiple * sin_longitude
        next_sin = cos_longitude * sin_multiple + cos_multiple * sin_longitude
        cos_multiple, sin_multiple = next_cos, next_sin

    longitude_component *= -(mu / radius)
    latitude_component *= mu / radius
    radial_component = -(mu / radius**2) * (1.0 + radial_component)
    return np.array(
        [
            cos_latitude * cos_longitude * radial_component
            - sin_latitude * cos_longitude * latitude_component / radius
            - sin_longitude * longitude_component / (cos_latitude * radius),
            cos_latitude * sin_longitude * radial_component
            - sin_latitude * sin_longitude * latitude_component / radius
            + cos_longitude * longitude_component / (cos_latitude * radius),
            sin_latitude * radial_component
            + cos_latitude * latitude_component / radius,
        ]
    )


@numbify(
    signature=(
        "Tuple((float64[:, ::1], float64[:, ::1]))("
        "float64[:, ::1], float64[:, ::1], float64[::1], float64[::1], "
        "float64, float64, float64, float64)"
    ),
    nopython=True,
    cache=True,
)
def _apply_solid_earth_tides(
    cosine,
    sine,
    sun_itrf,
    moon_itrf,
    earth_mu,
    sun_mu,
    moon_mu,
    earth_radius,
):
    """Apply nominal degree-two/three lunisolar solid-Earth tides."""
    cosine_dynamic = cosine.copy()
    sine_dynamic = sine.copy()
    love_number_2 = 0.30190
    love_number_3 = 0.093
    for position, body_mu, include_degree_3 in (
        (sun_itrf, sun_mu, False),
        (moon_itrf, moon_mu, True),
    ):
        radius = np.linalg.norm(position)
        sin_latitude = position[2] / radius
        cos_latitude = np.hypot(position[0], position[1]) / radius
        longitude = np.arctan2(position[1], position[0])
        scale_2 = (
            love_number_2 / 5.0 * body_mu / earth_mu * (earth_radius / radius) ** 3
        )
        if cosine_dynamic.shape[0] > 2:
            polynomials_2 = (
                0.5 * np.sqrt(5.0) * (3.0 * sin_latitude**2 - 1.0),
                np.sqrt(15.0) * sin_latitude * cos_latitude,
                0.5 * np.sqrt(15.0) * cos_latitude**2,
            )
            for order, polynomial in enumerate(polynomials_2):
                if order >= cosine_dynamic.shape[1]:
                    break
                cosine_dynamic[2, order] += (
                    scale_2 * polynomial * np.cos(order * longitude)
                )
                sine_dynamic[2, order] += (
                    scale_2 * polynomial * np.sin(order * longitude)
                )
        if include_degree_3 and cosine_dynamic.shape[0] > 3:
            scale_3 = (
                love_number_3 / 7.0 * body_mu / earth_mu * (earth_radius / radius) ** 4
            )
            polynomials_3 = (
                0.5 * np.sqrt(7.0) * sin_latitude * (5.0 * sin_latitude**2 - 3.0),
                0.5 * np.sqrt(42.0) * cos_latitude * (5.0 * sin_latitude**2 - 1.0),
                0.5 * np.sqrt(105.0) * sin_latitude * cos_latitude**2,
                0.5 * np.sqrt(70.0) * cos_latitude**3,
            )
            for order, polynomial in enumerate(polynomials_3):
                if order >= cosine_dynamic.shape[1]:
                    break
                cosine_dynamic[3, order] += (
                    scale_3 * polynomial * np.cos(order * longitude)
                )
                sine_dynamic[3, order] += (
                    scale_3 * polynomial * np.sin(order * longitude)
                )
    return cosine_dynamic, sine_dynamic
