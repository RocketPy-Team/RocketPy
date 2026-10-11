"""The hooks that bring thrust actuators, effectors and mass actuators into
the equations of motion, checked on Calisto flights."""

import warnings

import numpy as np
import pytest

from rocketpy import Controller, Event, Flight, GenericEffector
from rocketpy.simulation.helpers.dynamics import SIX_DOF_DYNAMICS, _PhaseDynamics
from rocketpy.simulation.helpers.flight_derivatives import _thrust_at
from tests.fixtures.actuators.actuator_fixtures import PassThrough


def _fly(rocket, environment, **options):
    rocket.parachutes = []
    options.setdefault("terminate_on_apogee", True)
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5.2,
        inclination=85,
        heading=0,
        **options,
    )


def _row_at(flight, time):
    """Return the time and canonical state of the first row at or after
    ``time``."""
    rows = np.array(flight.solution, dtype=float)
    row = rows[np.searchsorted(rows[:, 0], time)]
    return row[0], list(row[1:])


def _equivalent(t, state, fx, fy, fz, mx, my, mz):  # pylint: disable=unused-argument
    return [fx, fy, fz], [mx, my, mz]


_DERIVATIVES = {
    "standard": "u_dot_generalized",
    "solid_propulsion": "u_dot",
    "3 DOF": "u_dot_generalized_3dof",
}


def test_a_thrust_actuator_that_changes_nothing_flies_the_same(
    calisto_robust, example_plain_env
):
    plain = _fly(calisto_robust, example_plain_env)
    calisto_robust.add_actuator(PassThrough())
    passed = _fly(calisto_robust, example_plain_env)
    assert passed.apogee == pytest.approx(plain.apogee, rel=1e-9)
    assert passed.apogee_time == pytest.approx(plain.apogee_time, rel=1e-9)


@pytest.mark.parametrize("model", ["standard", "solid_propulsion", "3 DOF"])
def test_a_turned_thrust_acts_as_the_force_and_moment_it_adds(
    calisto_robust, example_plain_env, gimbal, model
):
    """Turning the thrust changes the derivative exactly as an effector
    pushing with the difference in force and moment would."""
    calisto_robust.add_actuator(gimbal)
    calisto_robust.add_actuator(
        GenericEffector(
            _equivalent,
            controls=("fx", "fy", "fz", "mx", "my", "mz"),
            name="Equivalent",
        )
    )
    options = (
        {"simulation_mode": "3 DOF"}
        if model == "3 DOF"
        else {"equations_of_motion": model}
    )
    flight = _fly(calisto_robust, example_plain_env, **options)
    dynamics = getattr(flight, _DERIVATIVES[model])
    t, x = _row_at(flight, 2.0)  # the motor burns
    u = dict(flight.control_signal)
    u.update(dict.fromkeys(u, 0.0))

    pressure = example_plain_env.pressure(x[2])
    net_thrust = flight.net_thrust(t)
    straight, straight_moment = _thrust_at(flight, t, net_thrust, pressure, u)
    turned_u = {**u, "gimbal.pitch": 0.1}
    turned, turned_moment = _thrust_at(flight, t, net_thrust, pressure, turned_u)
    force, moment = turned - straight, turned_moment - straight_moment
    pushed_u = {
        **u,
        **dict(zip(("equivalent.fx", "equivalent.fy", "equivalent.fz"), force)),
        **dict(zip(("equivalent.mx", "equivalent.my", "equivalent.mz"), moment)),
    }

    by_thrust = dynamics.derivative_at(t, x, turned_u)
    by_effector = dynamics.derivative_at(t, x, pushed_u)
    assert by_thrust == pytest.approx(by_effector, rel=1e-9, abs=1e-12)
    assert by_thrust != pytest.approx(dynamics.derivative_at(t, x, u))
    if model != "3 DOF":
        # tipped toward +x behind the center of mass, the nose turns to -x
        assert by_thrust[11] < dynamics.derivative_at(t, x, u)[11]


def test_an_effector_reaches_the_accelerations(
    calisto_robust, example_plain_env, push_effector
):
    calisto_robust.add_actuator(push_effector)
    flight = _fly(calisto_robust, example_plain_env)
    t, x = _row_at(flight, 6.0)  # coasting
    off = flight.u_dot_generalized.derivative_at(t, x, {"push.on": 0.0})
    on = flight.u_dot_generalized.derivative_at(t, x, {"push.on": 1.0})

    # 10 N along the axis: the axis, in the Earth frame, times 10 N / mass
    e0, e1, e2, e3 = x[6:10]
    axis = [2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1**2 + e2**2)]
    mass = calisto_robust.total_mass(t)
    assert [a - b for a, b in zip(on[3:6], off[3:6])] == pytest.approx(
        # the attitude matrix normalizes the quaternion, which this does not
        [10.0 * component / mass for component in axis],
        rel=1e-6,
    )
    # 0.5 N m about the axis turns the rocket about it
    assert on[12] - off[12] == pytest.approx(0.5 / calisto_robust.I_33(t), rel=1e-9)


def test_on_the_rail_only_the_axial_force_counts(
    calisto_robust, example_plain_env, push_effector, side_effector
):
    calisto_robust.add_actuator(push_effector)
    calisto_robust.add_actuator(side_effector)
    flight = _fly(calisto_robust, example_plain_env)
    t, x = _row_at(flight, 0.1)
    u = {"push.on": 0.0, "side.on": 0.0}
    rail = flight.udot_rail1
    still = rail.derivative_at(t, x, u)
    assert rail.derivative_at(t, x, {**u, "side.on": 1.0}) == still

    pushed = rail.derivative_at(t, x, {**u, "push.on": 1.0})
    along = np.linalg.norm(pushed[3:6]) - np.linalg.norm(still[3:6])
    assert along == pytest.approx(10.0 / calisto_robust.total_mass(t), rel=1e-9)


def test_a_moving_mass_changes_the_flight(
    calisto_robust, example_plain_env, moving_mass
):
    plain = _fly(calisto_robust, example_plain_env)
    calisto_robust.add_actuator(moving_mass)
    moving_mass.control_defaults["position"] = 0.3
    heavier = _fly(calisto_robust, example_plain_env)
    # one more kilogram: a lower apogee
    assert heavier.apogee < plain.apogee - 10
    assert heavier.solution.control_names == ("moving_mass.position",)


def test_the_solid_propulsion_equations_refuse_a_moving_mass(
    calisto_robust, example_plain_env, moving_mass
):
    calisto_robust.add_actuator(moving_mass)
    with pytest.raises(ValueError, match="solid_propulsion"):
        _fly(calisto_robust, example_plain_env, equations_of_motion="solid_propulsion")


def _spin_up(context):
    """Roll the rocket with the thrusters once the motor is out."""
    context.controls["push.on"] = 1.0 if context.time > 4.0 else 0.0


def test_a_controller_drives_an_effector(
    calisto_robust, example_plain_env, push_effector
):
    calisto_robust.add_actuator(push_effector)
    calisto_robust.add_controller(Controller(_spin_up, sampling_rate=10))
    flight = _fly(calisto_robust, example_plain_env)

    history = flight.control_history["push.on"]
    assert history(3.0) == 0.0 and history(6.0) == 1.0
    # 0.5 N m about the axis, from 4 s on: the roll rate keeps growing
    assert abs(flight.w3(10.0)) > abs(flight.w3(5.0)) > abs(flight.w3(3.9))


class _Canopy:
    """The brake of a steerable canopy: no physics of its own, read only by
    the glide phase below."""

    name = "Canopy"
    controls = ("brake",)
    control_defaults = {"brake": 0.0}
    control_limits = {"brake": (0.0, 1.0)}
    clamp = True


def _glide_derivative(_flight, _t, x, u, post_processing=False):
    """Fall, and turn at the rate the brake sets, in rad/s."""
    if post_processing:
        return ()
    return [x[3], x[4], x[5], 0.0, 0.0, -9.80665, u["canopy.brake"]]


GLIDE_DYNAMICS = _PhaseDynamics(
    "test_glide",
    _glide_derivative,
    ("x", "y", "z", "vx", "vy", "vz", "heading"),
    initial_state=lambda _flight, _t, canonical: [*canonical[:6], 0.0],
    reads=("canopy",),
)


def _open_canopy(context):
    context["event"].commands.set_dynamics(GLIDE_DYNAMICS)
    context["event"].commands.start_flight_phase("glide")


def _brake(context):
    context.controls["canopy.brake"] = 1.0 if context.time >= 6.0 else 0.0


def _glide_flight(rocket, environment, open_at):
    rocket.add_actuator(_Canopy())
    rocket.add_controller(Controller(_brake, sampling_rate=10))
    opening = Event(
        callback=_open_canopy,
        trigger=lambda context: context["time"] >= open_at,
        sampling_rate=10,
        trigger_only_once=True,
        changes_dynamics=True,
        name="Open canopy",
    )
    # No max_time: with a short one and no max_time_step, the solver can step
    # over the whole burn while the rocket waits on the pad
    return _fly(rocket, environment, custom_events=[opening], terminate_on_apogee=False)


def test_only_the_phase_that_names_a_part_reads_it(calisto_robust, example_plain_env):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        flight = _glide_flight(calisto_robust, example_plain_env, open_at=5.05)
    (canopy,) = calisto_robust.phase_actuators

    assert canopy not in SIX_DOF_DYNAMICS.actuators_read(calisto_robust)
    assert GLIDE_DYNAMICS.actuators_read(calisto_robust) == (canopy,)
    assert [phase.name for phase in flight.solution.phases][-1] == "glide"
    # the control inputs carry the brake for the whole flight
    brake = flight.solution["canopy.brake"]
    assert brake[0, 0] == 0.0 and brake[-1, 1] == 1.0
    # the glide phase turns at the rate the brake sets: 1 rad/s from 6 s
    with pytest.warns(UserWarning, match="not defined during"):
        heading = flight.solution["heading"]
    late = heading[heading[:, 0] >= 6.5]
    rate = (late[-1, 1] - late[0, 1]) / (late[-1, 0] - late[0, 0])
    assert rate == pytest.approx(1.0, rel=1e-6)


def test_a_part_no_phase_read_is_reported(calisto_robust, example_plain_env):
    with pytest.warns(UserWarning, match="'Canopy' were never read"):
        _glide_flight(calisto_robust, example_plain_env, open_at=100.0)
