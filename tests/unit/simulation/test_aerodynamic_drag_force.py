"""Tests of the rocket drag helpers. They use a stand-in for the flight, so no
simulation is run."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Rocket
from rocketpy.mathutils import Vector
from rocketpy.simulation.helpers.flight_derivatives import _compute_drag_area


def _flight_stand_in(power_off_drag, power_on_drag, burn_out_time=3.0):
    """The part of a flight the drag helper reads: the rocket and its motor's
    burn out time."""
    rocket = Rocket(
        radius=0.05,
        mass=10,
        inertia=(5, 5, 0.02),
        power_off_drag=power_off_drag,
        power_on_drag=power_on_drag,
        center_of_mass_without_motor=0,
    )
    rocket.motor = SimpleNamespace(burn_out_time=burn_out_time)
    return SimpleNamespace(rocket=rocket)


def _head_on(speed):
    """Air coming straight down the rocket's axis, in the body frame."""
    return Vector([0, 0, -speed])


def test_rocket_drag_receives_reduced_rates():
    """The rocket drag must receive the same non-dimensional rates as a generic
    surface, ``rate * diameter / (2 * airspeed)``, not the rates in rad/s."""
    flight = _flight_stand_in(
        power_off_drag=lambda pitch_rate, yaw_rate, roll_rate: (
            pitch_rate + 10 * yaw_rate + 100 * roll_rate
        ),
        power_on_drag=0.0,
    )
    speed, omega = 50.0, (0.2, -0.4, 8.0)
    reduced = 2 * 0.05 / (2 * speed)
    expected_coefficient = reduced * (omega[0] + 10 * omega[1] + 100 * omega[2])

    drag_area = _compute_drag_area(
        flight, 5.0, _head_on(speed), speed, 0.15, 1.2, 0, omega
    )

    assert drag_area == pytest.approx(flight.rocket.area * expected_coefficient)


def test_rocket_drag_rates_are_zero_without_airspeed():
    """With no airspeed the reduced rates are zero, and nothing divides by it."""
    flight = _flight_stand_in(
        power_off_drag=lambda mach, roll_rate: 0.5 + roll_rate, power_on_drag=0.0
    )
    drag_area = _compute_drag_area(
        flight, 5.0, _head_on(0.0), 0.0, 0, 1.2, 0, (0, 0, 8.0)
    )
    assert drag_area == pytest.approx(0.5 * flight.rocket.area)


def test_rocket_drag_reads_the_flow_from_the_stream():
    """The drag coefficients get the angle of attack and the sideslip angle read
    from the velocity of the air in the body frame, and the Reynolds number on
    the rocket diameter."""
    flight = _flight_stand_in(
        power_off_drag=lambda alpha, beta, mach, reynolds: (
            alpha + 10 * beta + 100 * mach + reynolds / 1e6
        ),
        power_on_drag=0.0,
    )
    alpha, beta, speed = np.radians(4.0), np.radians(-7.0), 80.0
    direction = np.array([np.tan(beta), np.tan(alpha), 1.0])
    stream = Vector(list(-speed * direction / np.linalg.norm(direction)))
    density, viscosity = 1.1, 1.8e-5
    reynolds = density * speed * (2 * 0.05) / viscosity

    drag_area = _compute_drag_area(
        flight, 5.0, stream, speed, 0.25, density, viscosity, (0, 0, 0)
    )

    expected = alpha + 10 * beta + 100 * 0.25 + reynolds / 1e6
    assert drag_area == pytest.approx(flight.rocket.area * expected)


def test_rocket_drag_curve_follows_the_motor_burn():
    """The power-on curve is used until burn out, the power-off curve after."""
    flight = _flight_stand_in(power_off_drag=0.6, power_on_drag=0.4)
    before = _compute_drag_area(
        flight, 1.0, _head_on(50.0), 50.0, 0.15, 1.2, 0, (0, 0, 0)
    )
    after = _compute_drag_area(
        flight, 5.0, _head_on(50.0), 50.0, 0.15, 1.2, 0, (0, 0, 0)
    )
    assert before == pytest.approx(0.4 * flight.rocket.area)
    assert after == pytest.approx(0.6 * flight.rocket.area)


@pytest.mark.parametrize("equations", ["u_dot", "u_dot_generalized"])
@pytest.mark.parametrize("angle_deg", [0.0, 2.0, 10.0, 60.0, 90.0, 180.0])
def test_axial_drag_follows_the_air_moving_along_the_axis(angle_deg, equations):
    """Flying straight into the air the axial drag is the usual
    ``0.5 * rho * V**2 * Cd * A`` toward the tail. It fades to zero with the
    rocket sideways to the air, and pushes toward the nose (against the motion)
    when the rocket moves tail first. Checked on both 6-DOF equations of motion,
    with a stand-in for the flight, so no simulation is run."""
    # pylint: disable=import-outside-toplevel
    from rocketpy import Environment, SolidMotor
    from rocketpy.simulation.helpers import flight_derivatives

    env = Environment()
    env.set_atmospheric_model(type="standard_atmosphere")
    rocket = Rocket(
        radius=0.05,
        mass=10,
        inertia=(5, 5, 0.02),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_motor(
        SolidMotor(
            thrust_source=1000,
            burn_time=2.0,
            dry_mass=1.0,
            dry_inertia=(0.1, 0.1, 0.002),
            center_of_dry_mass_position=0.3,
            nozzle_radius=0.03,
            grain_number=1,
            grain_density=1800,
            grain_outer_radius=0.03,
            grain_initial_inner_radius=0.015,
            grain_initial_height=0.1,
            grain_separation=0.0,
            grains_center_of_mass_position=0.3,
            nozzle_position=0,
            throat_radius=0.01,
        ),
        position=-0.5,
    )
    flight = SimpleNamespace(env=env, rocket=rocket)  # all the derivative reads
    speed, angle = 80.0, np.radians(angle_deg)
    upright = [1.0, 0.0, 0.0, 0.0]  # the body axis is the inertial z axis
    velocity = [speed * np.sin(angle), 0.0, speed * np.cos(angle)]
    state = [0.0, 0.0, 1000.0, *velocity, *upright, 0.0, 0.0, 0.0]

    # After burn out, with no aerodynamic surface, the axial force is the drag
    derivative = getattr(flight_derivatives, equations)
    axial_force = derivative(flight, 10.0, state, post_processing=True)[8]

    rho = env.density.get_value_opt(1000.0)
    full = 0.5 * rho * speed**2 * 0.5 * rocket.area
    assert axial_force == pytest.approx(-full * np.cos(angle), abs=1e-9)


@pytest.mark.parametrize(
    "velocity",
    [
        (0.0, 0.0, 80.0),  # climbing straight up
        (30.0, 0.0, 0.5),  # near apogee, moving sideways
        (5.0, -3.0, -50.0),  # falling with a drift
    ],
)
def test_3dof_drag_acts_against_the_velocity(velocity):
    """A 3-DOF phase does not model the rocket's attitude (by default it stays at
    the launch direction), so the drag must act against the velocity relative to
    the air, not along the body axis. Falling, it must slow the rocket down."""
    # pylint: disable=import-outside-toplevel
    from rocketpy import Environment, PointMassMotor, PointMassRocket
    from rocketpy.simulation.helpers.flight_derivatives import u_dot_generalized_3dof

    env = Environment()
    env.set_atmospheric_model(type="standard_atmosphere")
    rocket = PointMassRocket(
        radius=0.05,
        mass=10,
        center_of_mass_without_motor=0,
        power_off_drag=0.5,
        power_on_drag=0.5,
    )
    rocket.add_motor(
        PointMassMotor(
            thrust_source=1000,
            dry_mass=1.0,
            propellant_initial_mass=1.0,
            burn_time=2.0,
        ),
        position=0,
    )
    flight = SimpleNamespace(env=env, rocket=rocket)  # all the derivative reads
    upright = [1.0, 0.0, 0.0, 0.0]
    state = [0.0, 0.0, 1000.0, *velocity, *upright, 0.0, 0.0, 0.0]

    drag = np.array(
        u_dot_generalized_3dof(flight, 10.0, state, post_processing=True)[6:9]
    )

    speed = np.linalg.norm(velocity)
    rho = env.density.get_value_opt(1000.0)
    expected = -0.5 * rho * speed * 0.5 * rocket.area * np.array(velocity)
    assert drag == pytest.approx(expected, rel=1e-9, abs=1e-12)
