import math

import numpy as np
import pytest

from rocketpy import Function, GenericSurface, LinearGenericSurface, Rocket
from rocketpy.mathutils.vector_matrix import Vector

# TODO: review note: gotta test execution speed of changes in this branch


def _linear_surface_from_barrowman(surface):
    """Build a LinearGenericSurface that reproduces a Barrowman surface's aero.

    Reads the coefficient curves off a standard (Barrowman) aerodynamic surface
    -- its normal-force-curve slope ``clalpha`` as a function of Mach and, for fin
    sets, its roll cant and damping coefficients -- and packs them into the
    body-frame coefficient derivatives of an equivalent
    :class:`LinearGenericSurface`. The pitch- and yaw-plane slopes follow the
    Barrowman sign convention (``cN_alpha = clalpha`` and ``cY_beta = -clalpha``).

    The returned surface applies its force at its own origin (center of pressure
    ``(0, 0, 0)``); the caller is expected to add it at the surface's center of
    pressure station (``barrowman_station - clalpha_cp``) so that its force
    lands, and its static margin reads, at the same point as the Barrowman
    surface. Its ``cpz`` (the distance from the surface origin to its center of
    pressure) is exposed on the returned object as ``barrowman_cpz`` to make that
    offset easy for the caller.

    Parameters
    ----------
    surface : rocketpy.NoseCone, rocketpy.Tail or rocketpy fin set
        A standard Barrowman aerodynamic surface to copy the aero curves from.

    Returns
    -------
    rocketpy.LinearGenericSurface
        A linear generic surface with the same normal force and (for fins) roll
        behaviour as ``surface``, carrying the source surface's ``cpz`` as
        ``barrowman_cpz``.
    """
    clalpha = surface.clalpha  # normal-force-curve slope, a Function of Mach
    coefficients = {
        "cN_alpha": clalpha,
        "cY_beta": lambda mach: -clalpha.get_value_opt(mach),
    }
    # Fin sets carry roll coefficients: cant forcing (zero when uncanted) and
    # roll-rate damping. Other surfaces have no roll_parameters.
    if getattr(surface, "roll_parameters", None) is not None:
        coefficients["cl_0"] = surface.cl_0
        coefficients["cl_p"] = surface.cl_p
    linear_surface = LinearGenericSurface(
        reference_area=surface.reference_area,
        reference_length=surface.reference_length,
        coefficients=coefficients,
        center_of_pressure=(0, 0, 0),
        name=f"{surface.name}_linear",
    )
    linear_surface.barrowman_cpz = surface.cpz
    return linear_surface


def _generic_surface_from_barrowman(surface):
    """Build a :class:`GenericSurface` that reproduces a Barrowman surface's aero.

    Same idea as :func:`_linear_surface_from_barrowman`, but expressed as a plain
    :class:`GenericSurface`: instead of coefficient *slopes*, the total body-frame
    coefficients are given directly as the linear expansion of the Barrowman
    curves. The normal force is ``cN = clalpha(mach) * alpha``, the side force is
    ``cY = -clalpha(mach) * beta`` and (for fins) the roll moment is
    ``cl = cl_0(mach) + cl_p(mach) * roll_rate``. Because these are the same
    functions the linear surface builds internally, the two surfaces produce
    identical forces and moments; this fixture exercises the plain generic-surface
    code path.

    The surface applies its force at its own origin (center of pressure
    ``(0, 0, 0)``); the source surface's ``cpz`` is exposed as ``barrowman_cpz``
    so the caller can add it at the Barrowman center-of-pressure station, exactly
    as for the linear surface.

    Parameters
    ----------
    surface : rocketpy.NoseCone, rocketpy.Tail or rocketpy fin set
        A standard Barrowman aerodynamic surface to copy the aero curves from.

    Returns
    -------
    rocketpy.GenericSurface
        A generic surface with the same normal force and (for fins) roll
        behaviour as ``surface``, carrying the source surface's ``cpz`` as
        ``barrowman_cpz``.
    """
    clalpha = surface.clalpha  # normal-force-curve slope, a Function of Mach

    # Coefficient callables must accept the full 7-variable argument tuple. The
    # slope ``clalpha`` is a Function of Mach only; the fin roll coefficients
    # ``cl_0``/``cl_p`` are AeroCoefficients evaluated over the full tuple.
    def make_normal(slope):
        def cN(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate):
            return slope.get_value_opt(mach) * alpha

        return cN

    def make_side(slope):
        def cY(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate):
            return -slope.get_value_opt(mach) * beta

        return cY

    coefficients = {"cN": make_normal(clalpha), "cY": make_side(clalpha)}
    if getattr(surface, "roll_parameters", None) is not None:
        cl_0, cl_p = surface.cl_0, surface.cl_p

        def make_roll(forcing, damping):
            def cl(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate):
                args = (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                cant = forcing.get_value_opt(*args)
                return cant + damping.get_value_opt(*args) * roll_rate

            return cl

        coefficients["cl"] = make_roll(cl_0, cl_p)

    generic_surface = GenericSurface(
        reference_area=surface.reference_area,
        reference_length=surface.reference_length,
        coefficients=coefficients,
        center_of_pressure=(0, 0, 0),
        name=f"{surface.name}_generic",
    )
    generic_surface.barrowman_cpz = surface.cpz
    return generic_surface


# The three Barrowman surfaces of the Calisto rocket and the axial stations
# (in the tail_to_nose user coordinate system) at which their origins sit.
_CALISTO_SURFACE_STATIONS = (
    ("nose", 1.160),
    ("tail", -1.313),
    ("fins", -1.168),
)


def _full_body_force_and_moment(rocket, alpha, beta, mach, omega, speed=1.0):
    """Total body-frame force and moment about the dry center of mass, summed
    over every aerodynamic surface at a flow state and set of body rates.

    Reproduces the flight integrator's per-surface computation: each surface is
    fed its own local stream velocity, which includes the ``omega ^ cp``
    lever-arm term, so the sum captures the pitch/yaw damping the distributed
    surfaces produce. Evaluated at unit air density; the result scales out of any
    coefficient ratio. Used to lump the distributed surfaces into a single
    full-body coefficient set.
    """
    stream_direction = Vector([-math.tan(beta), -math.tan(alpha), -1.0])
    stream_at_cdm = stream_direction / abs(stream_direction) * speed
    body_rates = Vector(list(omega))
    density = Function(1.0)
    dynamic_viscosity = Function(1e30)  # vanishing-Reynolds limit
    speed_of_sound = speed / mach if mach > 0 else 1e30
    totals = np.zeros(6)
    for surface, _ in rocket.aerodynamic_surfaces:
        cp = rocket.surfaces_cp_to_cdm[surface]
        comp_stream = stream_at_cdm - (body_rates ^ cp)
        comp_speed = abs(comp_stream)
        forces = surface.compute_forces_and_moments(
            comp_stream,
            comp_speed,
            comp_speed / speed_of_sound,
            1.0,
            cp,
            body_rates,
            density,
            dynamic_viscosity,
            0.0,
        )
        totals += np.array(forces)
    return totals


def _full_body_derivatives(reference_rocket):
    """Lump a rocket's distributed aerodynamic surfaces into one full-body
    stability-derivative set (about the dry center of mass), as named slopes for
    a :class:`LinearGenericSurface`.

    A single surface placed at the center of mass has no lever arm of its own, so
    the pitch/yaw damping the distributed fore-and-aft surfaces produce has to be
    carried explicitly by rate derivatives. Every derivative is measured by
    finite-differencing the distributed aerodynamics
    (:func:`_full_body_force_and_moment`) and tabulated against Mach: the static
    slopes ``cN_alpha``/``cm_alpha`` (pitch) and ``cY_beta``/``cn_beta`` (yaw),
    the rate-damping slopes ``cN_q``/``cm_q`` and ``cY_r``/``cn_r``, and the fin
    roll damping ``cl_p``. Axial (drag) is left out so the rocket's own drag curve
    still applies. Handed to a linear surface, these reproduce the reference
    rocket's aerodynamics as a lumped model -- the way a measured or CFD
    stability-derivative set is used.
    """
    reference_length = 2 * reference_rocket.radius
    area = reference_rocket.area
    machs = np.arange(0.0, 3.01, 0.02)
    step = 1e-5
    dyn_area = 0.5 * area  # unit speed, unit density
    dyn_area_length = dyn_area * reference_length

    def coefficients_at(alpha, beta, red_pitch, red_yaw, red_roll, mach):
        rate_factor = 2.0 / reference_length  # reduced rate -> omega at unit speed
        omega = (red_pitch * rate_factor, red_yaw * rate_factor, red_roll * rate_factor)
        r1, r2, r3, m1, m2, m3 = _full_body_force_and_moment(
            reference_rocket, alpha, beta, mach, omega
        )
        return {
            "cN": -r2 / dyn_area,
            "cY": r1 / dyn_area,
            "cm": m1 / dyn_area_length,
            "cn": m2 / dyn_area_length,
            "cl": m3 / dyn_area_length,
        }

    def slope(field, coeff):
        values = []
        for mach in machs:
            state = {
                "alpha": 0.0,
                "beta": 0.0,
                "red_pitch": 0.0,
                "red_yaw": 0.0,
                "red_roll": 0.0,
            }
            high = dict(state, **{field: step})
            low = dict(state, **{field: -step})
            c_high = coefficients_at(mach=mach, **high)
            c_low = coefficients_at(mach=mach, **low)
            values.append((c_high[coeff] - c_low[coeff]) / (2 * step))
        return Function(
            np.column_stack([machs, values]),
            "Mach",
            f"{coeff}_slope",
            interpolation="akima",
            extrapolation="constant",
        )

    # Named slopes consumed directly by LinearGenericSurface: the rate names
    # ``cN_q``/``cm_q`` multiply the reduced pitch rate, ``cY_r``/``cn_r`` the
    # reduced yaw rate and ``cl_p`` the reduced roll rate.
    return {
        "cN_alpha": slope("alpha", "cN"),
        "cm_alpha": slope("alpha", "cm"),
        "cN_q": slope("red_pitch", "cN"),
        "cm_q": slope("red_pitch", "cm"),
        "cY_beta": slope("beta", "cY"),
        "cn_beta": slope("beta", "cn"),
        "cY_r": slope("red_yaw", "cY"),
        "cn_r": slope("red_yaw", "cn"),
        "cl_p": slope("red_roll", "cl"),
    }


@pytest.fixture
def calisto_motorless():
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same rocket that has been used in the getting started guide for years
    but without a motor.

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    return calisto


@pytest.fixture
def calisto(calisto_motorless, cesaroni_m1670):  # old name: rocket
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same rocket that has been used in the getting started guide for
    years. The Calisto rocket is the Projeto Jupiter's project launched at the
    2019 Spaceport America Cup.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    cesaroni_m1670 : rocketpy.SolidMotor
        An object of the SolidMotor class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(cesaroni_m1670, position=-1.373)
    return calisto


@pytest.fixture
def calisto_nose_to_tail(cesaroni_m1670):
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same as the calisto fixture, but with the coordinate system
    orientation set to "nose_to_tail" instead of "tail_to_nose". This allows to
    check if the coordinate system orientation is being handled correctly in
    the code.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        An object of the SolidMotor class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        The Calisto rocket with the coordinate system orientation set to
        "nose_to_tail". Rail buttons are already set, as well as the motor.
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )
    calisto.add_motor(cesaroni_m1670, position=1.373)
    calisto.set_rail_buttons(
        upper_button_position=-0.082,
        lower_button_position=0.618,
        angular_position=0,
    )
    return calisto


@pytest.fixture
def calisto_liquid_modded(calisto_motorless, liquid_motor):
    """Create a simple object of the Rocket class to be used in the tests. This
    is an example of the Calisto rocket with a liquid motor.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    liquid_motor : rocketpy.LiquidMotor

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(liquid_motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_hybrid_modded(calisto_motorless, hybrid_motor):
    """Create a simple object of the Rocket class to be used in the tests. This
    is an example of the Calisto rocket with a hybrid motor.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    hybrid_motor : rocketpy.HybridMotor

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(hybrid_motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_robust(
    calisto,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_rail_buttons,  # pylint: disable=unused-argument
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto fixture, but with
    all the aerodynamic surfaces and parachutes added. This avoids repeating the
    same code in all tests.

    Parameters
    ----------
    calisto : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    calisto_nose_cone : rocketpy.NoseCone
        The nose cone of the Calisto rocket. This is a pytest fixture too.
    calisto_tail : rocketpy.Tail
        The boat tail of the Calisto rocket. This is a pytest fixture too.
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        The trapezoidal fins of the Calisto rocket. This is a pytest fixture
    calisto_rail_buttons : rocketpy.RailButtons
        The rail buttons of the Calisto rocket. This is a pytest fixture too.
    calisto_main_chute : rocketpy.Parachute
        The main parachute of the Calisto rocket. This is a pytest fixture too.
    calisto_drogue_chute : rocketpy.Parachute
        The drogue parachute of the Calisto rocket. This is a pytest fixture

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    # we follow this format: calisto.add_surfaces(surface, position)
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.add_surfaces(calisto_trapezoidal_fins, -1.168)
    # calisto.add_surfaces(calisto_rail_buttons, -1.168)
    # TODO: if I use the line above, the calisto won't have rail buttons attribute
    #       we need to apply a check in the add_surfaces method to set the rail buttons
    calisto.set_rail_buttons(
        upper_button_position=0.082,
        lower_button_position=-0.618,
        angular_position=0,
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)
    return calisto


@pytest.fixture
def calisto_linear_generic(
    cesaroni_m1670,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Calisto built entirely from LinearGenericSurfaces instead of Barrowman ones.

    This is the same rocket as ``calisto_robust`` -- same body, motor, rail
    buttons and parachutes at the same stations -- but its nose cone, tail and
    fin set are each replaced by a body-frame ``LinearGenericSurface``. The
    coefficient curves of those linear surfaces are extracted from the matching
    standard (Barrowman) surfaces: each linear surface reuses the standard
    surface's normal-force-curve slope, center of pressure and (for the fins)
    roll damping. Because the aero data is identical, this rocket's flight
    closely tracks the standard Calisto's while exercising the linear
    generic-surface aerodynamic path -- the one whose forces and moments are
    built directly in the body frame from the coefficient derivatives, with no
    wind-to-body rotation. It is a standalone rocket (it does not reuse the
    shared ``calisto`` fixture), so a test may build both it and ``calisto_robust``
    and compare their flights.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        The Calisto motor. This is a pytest fixture too.
    calisto_nose_cone : rocketpy.NoseCone
        The standard nose cone whose aero curves are copied. This is a pytest
        fixture too.
    calisto_tail : rocketpy.Tail
        The standard boat tail whose aero curves are copied. This is a pytest
        fixture too.
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        The standard fin set whose aero curves are copied. This is a pytest
        fixture too.
    calisto_main_chute : rocketpy.Parachute
        The main parachute of the Calisto rocket. This is a pytest fixture too.
    calisto_drogue_chute : rocketpy.Parachute
        The drogue parachute of the Calisto rocket. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        The Calisto rocket whose nose cone, tail and fins are all
        LinearGenericSurfaces.
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    calisto.add_motor(cesaroni_m1670, position=-1.373)
    # Replace each Barrowman surface with an equivalent LinearGenericSurface. A
    # Barrowman surface is placed by its origin and carries its center of
    # pressure at ``cpz`` aft of that origin; a generic surface applies its force
    # at its own origin. So each linear surface is added at the Barrowman
    # surface's center-of-pressure station (station - cpz, with tail_to_nose
    # csys = +1), which lands its force -- and its static margin -- at the same
    # point as calisto_robust.
    for surface, station in (
        (calisto_nose_cone, 1.160),
        (calisto_tail, -1.313),
        (calisto_trapezoidal_fins, -1.168),
    ):
        linear_surface = _linear_surface_from_barrowman(surface)
        calisto.add_surfaces(linear_surface, station - linear_surface.barrowman_cpz)
    calisto.set_rail_buttons(
        upper_button_position=0.082,
        lower_button_position=-0.618,
        angular_position=0,
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)
    return calisto


def _bare_calisto(motor):
    """The Calisto body and motor with no aerodynamic surfaces, parachutes or
    rail buttons yet -- the shared starting point for the generic-surface
    Calistos below."""
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    calisto.add_motor(motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_generic(
    cesaroni_m1670,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Calisto built entirely from :class:`GenericSurface` objects.

    The exact counterpart of ``calisto_linear_generic``, but each surface is a
    plain :class:`GenericSurface` carrying the *total* body-frame coefficients
    (``cN = clalpha * alpha`` ...) instead of a :class:`LinearGenericSurface`
    carrying the coefficient slopes. The two express the same aerodynamics
    through different code paths, so this rocket's flight should match both
    ``calisto_robust`` and ``calisto_linear_generic``. It is standalone (it does
    not reuse the shared ``calisto`` fixture), so a test may build it alongside
    the others and compare their flights.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        The Calisto motor. This is a pytest fixture too.
    calisto_nose_cone : rocketpy.NoseCone
        The standard nose cone whose aero curves are copied. This is a pytest
        fixture too.
    calisto_tail : rocketpy.Tail
        The standard boat tail whose aero curves are copied. This is a pytest
        fixture too.
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        The standard fin set whose aero curves are copied. This is a pytest
        fixture too.
    calisto_main_chute : rocketpy.Parachute
        The main parachute of the Calisto rocket. This is a pytest fixture too.
    calisto_drogue_chute : rocketpy.Parachute
        The drogue parachute of the Calisto rocket. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        The Calisto rocket whose nose cone, tail and fins are all
        GenericSurfaces.
    """
    calisto = _bare_calisto(cesaroni_m1670)
    barrowman_surfaces = {
        "nose": calisto_nose_cone,
        "tail": calisto_tail,
        "fins": calisto_trapezoidal_fins,
    }
    for label, station in _CALISTO_SURFACE_STATIONS:
        generic_surface = _generic_surface_from_barrowman(barrowman_surfaces[label])
        calisto.add_surfaces(generic_surface, station - generic_surface.barrowman_cpz)
    calisto.set_rail_buttons(
        upper_button_position=0.082,
        lower_button_position=-0.618,
        angular_position=0,
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)
    return calisto


@pytest.fixture
def calisto_full_aerodynamics(
    cesaroni_m1670,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Calisto flown from a single full-body stability-derivative set.

    Instead of modeling each surface, this rocket carries one
    :class:`LinearGenericSurface` added through
    :meth:`Rocket.add_full_body_aerodynamics` that lumps the whole rocket into a
    set of named coefficient slopes referenced to the dry center of mass. The
    slopes are extracted from an equivalent ``calisto_generic`` rocket
    (:func:`_full_body_derivatives`): the static normal-force and
    pitch/yaw-moment slopes plus the pitch/yaw rate damping the distributed
    surfaces produce through their lever arms and the fin roll damping. The
    built-in drag curve is kept (no drag coefficient is supplied), exactly as for
    the modeled Calisto, so this rocket's flight should match ``calisto_robust``,
    ``calisto_linear_generic`` and ``calisto_generic``.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        The Calisto motor. This is a pytest fixture too.
    calisto_nose_cone : rocketpy.NoseCone
        The standard nose cone whose aero curves are lumped in. This is a pytest
        fixture too.
    calisto_tail : rocketpy.Tail
        The standard boat tail whose aero curves are lumped in. This is a pytest
        fixture too.
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        The standard fin set whose aero curves are lumped in. This is a pytest
        fixture too.
    calisto_main_chute : rocketpy.Parachute
        The main parachute of the Calisto rocket. This is a pytest fixture too.
    calisto_drogue_chute : rocketpy.Parachute
        The drogue parachute of the Calisto rocket. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        The Calisto rocket flown from a single full-body derivative set.
    """
    # Distributed reference rocket (same motor, so the same dry center of mass)
    # whose lumped derivatives feed the single full-body surface.
    reference = _bare_calisto(cesaroni_m1670)
    barrowman_surfaces = {
        "nose": calisto_nose_cone,
        "tail": calisto_tail,
        "fins": calisto_trapezoidal_fins,
    }
    for label, station in _CALISTO_SURFACE_STATIONS:
        generic_surface = _generic_surface_from_barrowman(barrowman_surfaces[label])
        reference.add_surfaces(generic_surface, station - generic_surface.barrowman_cpz)

    calisto = _bare_calisto(cesaroni_m1670)
    full_body_surface = LinearGenericSurface(
        reference_area=calisto.area,
        reference_length=2 * calisto.radius,
        coefficients=_full_body_derivatives(reference),
        name="Calisto full body aerodynamics",
    )
    calisto.add_full_body_aerodynamics(full_body_surface)
    calisto.set_rail_buttons(
        upper_button_position=0.082,
        lower_button_position=-0.618,
        angular_position=0,
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)
    return calisto


@pytest.fixture
def calisto_nose_to_tail_robust(
    calisto_nose_to_tail,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Calisto with nose to tail coordinate system orientation. This is the same
    as calisto_robust, but with the coordinate system orientation set to
    "nose_to_tail"."""
    csys = -1
    # we follow this format: calisto.add_surfaces(surface, position)
    calisto_nose_to_tail.add_surfaces(calisto_nose_cone, 1.160 * csys)
    calisto_nose_to_tail.add_surfaces(calisto_tail, -1.313 * csys)
    calisto_nose_to_tail.add_surfaces(calisto_trapezoidal_fins, -1.168 * csys)
    calisto_nose_to_tail.set_rail_buttons(
        upper_button_position=0.082 * csys,
        lower_button_position=-0.618 * csys,
        angular_position=360 - 0,
    )
    calisto_nose_to_tail.parachutes.append(calisto_main_chute)
    calisto_nose_to_tail.parachutes.append(calisto_drogue_chute)
    return calisto_nose_to_tail


@pytest.fixture
def calisto_air_brakes_clamp_on(calisto_robust, controller_function):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto_robust fixture,
    but with air brakes added, with clamping.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture.
    controller_function : function
        A function that controls the air brakes. This is a pytest fixture.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    calisto = calisto_robust
    # remove parachutes
    calisto.parachutes = []
    calisto.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        clamp=True,
    )
    return calisto


@pytest.fixture
def calisto_air_brakes_clamp_off(calisto_robust, controller_function):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto_robust fixture,
    but with air brakes added, without clamping.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture.
    controller_function : function
        A function that controls the air brakes. This is a pytest fixture.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    calisto = calisto_robust
    # remove parachutes
    calisto.parachutes = []
    calisto.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        clamp=False,
    )
    return calisto


@pytest.fixture
def calisto_with_sensors(
    calisto,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    ideal_accelerometer,
    ideal_gyroscope,
    ideal_barometer,
    ideal_gnss,
):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto fixture, but with
    a set of ideal sensors added at the center of dry mass, meaning the readings
    will be the same as the values saved on a Flight object.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.add_surfaces(calisto_trapezoidal_fins, -1.168)
    # double sensors to test using same instance twice
    calisto.add_sensor(ideal_accelerometer, -0.1180124376577797)
    calisto.add_sensor(ideal_accelerometer, -0.1180124376577797)
    calisto.add_sensor(ideal_gyroscope, -0.1180124376577797)
    calisto.add_sensor(ideal_barometer, -0.1180124376577797)
    calisto.add_sensor(ideal_gnss, -0.1180124376577797)
    return calisto


@pytest.fixture  # old name: dimensionless_rocket
def dimensionless_calisto(kg, m, dimensionless_cesaroni_m1670):
    """The dimensionless version of the Calisto rocket. This is the same rocket
    as defined in the calisto fixture, but with all the parameters converted to
    dimensionless values. This allows to check if the dimensions are being
    handled correctly in the code.

    Parameters
    ----------
    kg : numericalunits.kg
        An object of the numericalunits.kg class. This is a pytest fixture too.
    m : numericalunits.m
        An object of the numericalunits.m class. This is a pytest fixture too.
    dimensionless_cesaroni_m1670 : rocketpy.SolidMotor
        The dimensionless version of the Cesaroni M1670 motor. This is a pytest
        fixture too.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    example_rocket = Rocket(
        radius=0.0635 * m,
        mass=14.426 * kg,
        inertia=(6.321 * (kg * m**2), 6.321 * (kg * m**2), 0.034 * (kg * m**2)),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0 * m,
        coordinate_system_orientation="tail_to_nose",
    )
    example_rocket.add_motor(dimensionless_cesaroni_m1670, position=(-1.373) * m)
    return example_rocket


@pytest.fixture
def prometheus_rocket(generic_motor_cesaroni_M1520):
    """Create a simple object of the Rocket class to be used in the tests. This
    is the Prometheus rocket, a rocket documented in the Flight Examples section
    of the RocketPy documentation.

    Parameters
    ----------
    generic_motor_cesaroni_M1520 : GenericMotor
        An object of the GenericMotor class. This is a pytest fixture too.
    """

    def prometheus_cd_at_ma(mach):
        """Gives the drag coefficient of the rocket at a given mach number."""
        if mach <= 0.15:
            return 0.422
        elif mach <= 0.45:
            return 0.422 + (mach - 0.15) * (0.38 - 0.422) / (0.45 - 0.15)
        elif mach <= 0.77:
            return 0.38 + (mach - 0.45) * (0.32 - 0.38) / (0.77 - 0.45)
        elif mach <= 0.82:
            return 0.32 + (mach - 0.77) * (0.3 - 0.32) / (0.82 - 0.77)
        elif mach <= 0.88:
            return 0.3 + (mach - 0.82) * (0.3 - 0.3) / (0.88 - 0.82)
        elif mach <= 0.94:
            return 0.3 + (mach - 0.88) * (0.32 - 0.3) / (0.94 - 0.88)
        elif mach <= 0.99:
            return 0.32 + (mach - 0.94) * (0.37 - 0.32) / (0.99 - 0.94)
        elif mach <= 1.04:
            return 0.37 + (mach - 0.99) * (0.44 - 0.37) / (1.04 - 0.99)
        elif mach <= 1.24:
            return 0.44 + (mach - 1.04) * (0.43 - 0.44) / (1.24 - 1.04)
        elif mach <= 1.33:
            return 0.43 + (mach - 1.24) * (0.42 - 0.43) / (1.33 - 1.24)
        elif mach <= 1.49:
            return 0.42 + (mach - 1.33) * (0.39 - 0.42) / (1.49 - 1.33)
        else:
            return 0.39

    prometheus = Rocket(
        radius=0.06985,  # 5.5" diameter circle
        mass=13.93,
        inertia=(
            4.87,
            4.87,
            0.05,
        ),
        power_off_drag=prometheus_cd_at_ma,
        power_on_drag=lambda x: prometheus_cd_at_ma(x) * 1.02,  # 5% increase in drag
        center_of_mass_without_motor=0.9549,
        coordinate_system_orientation="tail_to_nose",
    )

    prometheus.set_rail_buttons(0.69, 0.21, 60)

    prometheus.add_motor(motor=generic_motor_cesaroni_M1520, position=0)
    prometheus.add_nose(length=0.742, kind="Von Karman", position=2.229)
    prometheus.add_trapezoidal_fins(
        n=3,
        span=0.13,
        root_chord=0.268,
        tip_chord=0.136,
        position=0.273,
        sweep_length=0.066,
    )
    prometheus.add_parachute(
        "Drogue",
        cd_s=1.6 * np.pi * 0.3048**2,  # Cd = 1.6, D_chute = 24 in
        trigger="apogee",
    )
    prometheus.add_parachute(
        "Main",
        cd_s=2.2 * np.pi * 0.9144**2,  # Cd = 2.2, D_chute = 72 in
        trigger=457.2,  # 1500 ft
    )
    return prometheus
