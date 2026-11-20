"""The hooks that bring thrust and force effectors into the equations of
motion, checked on Calisto flights."""

import numpy as np
import pytest

from rocketpy import Controller, Event, Flight
from rocketpy.control import FlightConditions, ForceEffector
from rocketpy.simulation.helpers.dynamics import _PhaseDynamics
from rocketpy.simulation.helpers.flight_derivatives import _effector_thrust
from tests.fixtures.effectors.effector_fixtures import Gimbal, PassThrough


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


def _equivalent(fx, fy, fz, mx, my, mz, conditions):  # pylint: disable=unused-argument
    return [fx, fy, fz], [mx, my, mz]


_DERIVATIVES = {
    "standard": "u_dot_generalized",
    "solid_propulsion": "u_dot",
    "3 DOF": "u_dot_generalized_3dof",
}


def test_a_thrust_effector_that_changes_nothing_flies_the_same(
    calisto_robust, example_plain_env
):
    plain = _fly(calisto_robust, example_plain_env)
    calisto_robust.add_effector(PassThrough())
    passed = _fly(calisto_robust, example_plain_env)
    assert passed.apogee == pytest.approx(plain.apogee, rel=1e-9)
    assert passed.apogee_time == pytest.approx(plain.apogee_time, rel=1e-9)


@pytest.mark.parametrize("model", ["standard", "solid_propulsion", "3 DOF"])
def test_a_turned_thrust_acts_as_the_force_and_moment_it_adds(
    calisto_robust, example_plain_env, gimbal, model
):
    """Turning the thrust changes the derivative exactly as an effector
    pushing with the difference in force and moment would."""
    calisto_robust.add_effector(gimbal)
    calisto_robust.add_effector(
        ForceEffector(
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
    t, u = _row_at(flight, 2.0)  # the motor burns
    controls = dict(flight._controls)
    controls.update(dict.fromkeys(controls, 0.0))

    net_thrust = flight.net_thrust(t)
    straight, straight_moment, _ = _effector_thrust(
        flight,
        FlightConditions(flight.rocket, flight.env, t, u),
        net_thrust,
        controls,
    )
    turned_controls = {**controls, "gimbal.pitch": 0.1}
    turned, turned_moment, _ = _effector_thrust(
        flight,
        FlightConditions(flight.rocket, flight.env, t, u),
        net_thrust,
        turned_controls,
    )
    force, moment = turned - straight, turned_moment - straight_moment
    pushed_controls = {
        **controls,
        **dict(zip(("equivalent.fx", "equivalent.fy", "equivalent.fz"), force)),
        **dict(zip(("equivalent.mx", "equivalent.my", "equivalent.mz"), moment)),
    }

    by_thrust = dynamics.derivative_at(t, u, turned_controls)
    by_effector = dynamics.derivative_at(t, u, pushed_controls)
    assert by_thrust == pytest.approx(by_effector, rel=1e-9, abs=1e-12)
    assert by_thrust != pytest.approx(dynamics.derivative_at(t, u, controls))
    if model != "3 DOF":
        # tipped toward +x behind the center of mass, the nose turns to -x
        assert by_thrust[11] < dynamics.derivative_at(t, u, controls)[11]


def test_an_effector_reaches_the_accelerations(
    calisto_robust, example_plain_env, push_effector
):
    calisto_robust.add_effector(push_effector)
    flight = _fly(calisto_robust, example_plain_env)
    t, u = _row_at(flight, 6.0)  # coasting
    off = flight.u_dot_generalized.derivative_at(t, u, {"push.on": 0.0})
    on = flight.u_dot_generalized.derivative_at(t, u, {"push.on": 1.0})

    # 10 N along the axis: the axis, in the Earth frame, times 10 N / mass
    e0, e1, e2, e3 = u[6:10]
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
    calisto_robust.add_effector(push_effector)
    calisto_robust.add_effector(side_effector)
    flight = _fly(calisto_robust, example_plain_env)
    t, u = _row_at(flight, 0.1)
    controls = {"push.on": 0.0, "side.on": 0.0}
    rail = flight.udot_rail1
    still = rail.derivative_at(t, u, controls)
    assert rail.derivative_at(t, u, {**controls, "side.on": 1.0}) == still

    pushed = rail.derivative_at(t, u, {**controls, "push.on": 1.0})
    along = np.linalg.norm(pushed[3:6]) - np.linalg.norm(still[3:6])
    assert along == pytest.approx(10.0 / calisto_robust.total_mass(t), rel=1e-9)


def _spin_up(context):
    """Roll the rocket with the thrusters once the motor is out."""
    context.controls.push.on = 1.0 if context.time > 4.0 else 0.0


def test_a_controller_drives_an_effector(
    calisto_robust, example_plain_env, push_effector
):
    calisto_robust.add_effector(push_effector)
    calisto_robust.add_controller(Controller(_spin_up, sampling_rate=10))
    flight = _fly(calisto_robust, example_plain_env)

    history = flight.controls.push.on
    assert history(3.0) == 0.0 and history(6.0) == 1.0
    # 0.5 N m about the axis, from 4 s on: the roll rate keeps growing
    assert abs(flight.w3(10.0)) > abs(flight.w3(5.0)) > abs(flight.w3(3.9))


class _Canopy(ForceEffector):
    """The brake of a steerable canopy: nothing to the free-flight phases,
    read only by the glide phase below. (A canopy gets its own registration
    with the parafoil model; until then it is an effector that does nothing.)"""

    def __init__(self):
        super().__init__(
            controls=("brake",), control_limits={"brake": (0.0, 1.0)}, name="Canopy"
        )

    def force_and_moment(self, controls, conditions):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def _glide_derivative(_flight, _t, u, controls, post_processing=False):
    """Fall, and turn at the rate the brake sets, in rad/s."""
    if post_processing:
        return ()
    return [u[3], u[4], u[5], 0.0, 0.0, -9.80665, controls["canopy.brake"]]


GLIDE_DYNAMICS = _PhaseDynamics(
    "test_glide",
    _glide_derivative,
    ("x", "y", "z", "vx", "vy", "vz", "heading"),
    initial_state=lambda _flight, _t, canonical, _previous: [*canonical[:6], 0.0],
)


def _open_canopy(context):
    context["event"].commands.set_dynamics(GLIDE_DYNAMICS)
    context["event"].commands.start_flight_phase("glide")


def _brake(context):
    context.controls.canopy.brake = 1.0 if context.time >= 6.0 else 0.0


def _glide_flight(rocket, environment, open_at):
    rocket.add_effector(_Canopy())
    rocket.add_controller(Controller(_brake, sampling_rate=10))
    opening = Event(
        callback=_open_canopy,
        trigger=lambda context: context["time"] >= open_at,
        sampling_rate=10,
        trigger_only_once=True,
        name="Open canopy",
    )
    # No max_time: with a short one and no max_time_step, the solver can step
    # over the whole burn while the rocket waits on the pad
    return _fly(rocket, environment, custom_events=[opening], terminate_on_apogee=False)


def test_a_phase_of_its_own_reads_a_part(calisto_robust, example_plain_env):
    """A controllable only a custom phase reads gets its channel like any other, and
    the phase reads it by name from the control inputs."""
    flight = _glide_flight(calisto_robust, example_plain_env, open_at=5.05)

    assert [phase.name for phase in flight.solution.phases][-1] == "glide"
    # the control inputs carry the brake for the whole flight
    brake = flight.solution.controls.canopy.brake
    assert brake[0, 0] == 0.0 and brake[-1, 1] == 1.0
    # the glide phase turns at the rate the brake sets: 1 rad/s from 6 s
    with pytest.warns(UserWarning, match="not defined during"):
        heading = flight.solution["heading"]
    late = heading[heading[:, 0] >= 6.5]
    rate = (late[-1, 1] - late[0, 1]) / (late[-1, 0] - late[0, 0])
    assert rate == pytest.approx(1.0, rel=1e-6)


class _LaggingGimbal(Gimbal):
    """A gimbal whose angle follows the command with a first-order lag of
    ``time_constant`` seconds; with ``None`` its state follows nothing and it
    turns the thrust by the command at once."""

    states = ("angle",)

    def __init__(self, time_constant):
        super().__init__("Gimbal")
        self.time_constant = time_constant

    def state_derivative(self, states, commands, conditions):
        if self.time_constant is None:
            return [0.0]
        return [(commands[0] - states[0]) / self.time_constant]

    def actual_controls(self, states, commands, conditions):
        return commands if self.time_constant is None else [states[0]]


def _command_pitch_after(time, pitch):
    def controller(context):
        if context.time >= time:
            context.controls.gimbal.pitch = pitch

    return Controller(controller, sampling_rate=20)


def test_controllable_states_that_change_nothing_fly_the_same(
    calisto_robust, example_plain_env
):
    """A controllable with a state the physics never reads flies as the same
    controllable without it, on the rail and off it. Not to the last digit: the solver
    weighs its error over one more state, so its steps differ slightly."""
    calisto_robust.add_effector(Gimbal())
    calisto_robust.add_controller(_command_pitch_after(1.0, 0.05))
    plain = _fly(calisto_robust, example_plain_env)
    calisto_robust.effectors._components[0] = calisto_robust.effectors.component_tuple(
        _LaggingGimbal(None), None
    )
    stated = _fly(calisto_robust, example_plain_env)
    assert stated.apogee == pytest.approx(plain.apogee, rel=1e-5)
    assert stated.apogee_time == pytest.approx(plain.apogee_time, rel=1e-5)
    assert stated.solution.ax[-1][1] == pytest.approx(
        plain.solution.ax[-1][1], rel=1e-5
    )


def test_a_part_state_follows_its_command_through_the_phases(
    calisto_robust, example_plain_env
):
    """The gimbal's angle starts at its initial value, carries over when the
    rocket leaves the rail, and lags the command it is given at 1 s."""
    calisto_robust.add_effector(_LaggingGimbal(time_constant=0.1))
    calisto_robust.add_controller(_command_pitch_after(1.0, 0.05))
    flight = _fly(calisto_robust, example_plain_env)
    assert len(flight.solution.phases) > 1
    angle = np.array(flight.solution["gimbal_angle"], dtype=float)
    times, values = angle[:, 0], angle[:, 1]
    assert np.all(values[times < 1.0] == 0.0)
    # One time constant after the command, about 63% of the way there
    assert np.interp(1.1, times, values) == pytest.approx(
        0.05 * (1 - np.exp(-1)), rel=0.05
    )
    assert values[-1] == pytest.approx(0.05, rel=1e-3)


def test_a_continued_flight_keeps_the_controls_and_the_part_states(
    calisto_robust, example_plain_env
):
    """A flight started from another one goes on with that flight's last control
    inputs and controllables' states, so the two halves fly as the whole."""
    calisto_robust.add_effector(_LaggingGimbal(time_constant=0.1))
    calisto_robust.add_controller(_command_pitch_after(1.0, 0.05))
    whole = _fly(calisto_robust, example_plain_env)
    # Cut while the gimbal is still swinging toward its command
    first = _fly(calisto_robust, example_plain_env, max_time=1.05)
    second = _fly(calisto_robust, example_plain_env, initial_solution=first)

    assert second.t_initial == first.t_final
    assert second.solution.controls.gimbal.pitch[0, 1] == 0.05
    before = np.array(first.solution["gimbal_angle"], dtype=float)
    after = np.array(second.solution["gimbal_angle"], dtype=float)
    assert 0.0 < before[-1, 1] < 0.05
    assert after[0, 1] == before[-1, 1]
    angle = np.array(whole.solution["gimbal_angle"], dtype=float)
    for t in (1.1, 1.5, 3.0):
        assert np.interp(t, after[:, 0], after[:, 1]) == pytest.approx(
            np.interp(t, angle[:, 0], angle[:, 1]), rel=1e-3
        )
    assert second.apogee == pytest.approx(whole.apogee, rel=1e-4)
