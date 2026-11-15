"""Continuous controllers: control laws evaluated inside the equations of
motion, at every instant. They use stand-ins for the rocket and the flight,
so no simulation is run."""

import json
from types import SimpleNamespace

import pytest

from rocketpy import ContinuousController, GenericSurface
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control import FlightConditions
from rocketpy.control.controllable import ForceEffector, ThrustEffector
from rocketpy.simulation.events.views import ControlsView
from rocketpy.simulation.helpers.dynamics import (
    CANONICAL_STATE_NAMES,
    SIX_DOF_DYNAMICS,
    _PhaseDynamics,
    _with_controllables,
)

# What FlightConditions reads of a flight
_FLIGHT = SimpleNamespace(rocket=None, env=None)


def _canards(limits=(-0.2, 0.2)):
    return GenericSurface(
        0.01,
        0.1,
        {"cl": lambda mach, deflection: deflection},
        name="Canards",
        controls=("deflection",),
        control_limits={"deflection": limits},
    )


def _rocket(*controllables):
    """A rocket with these controllables, sorted as Rocket sorts them."""
    laws = tuple(c for c in controllables if isinstance(c, ContinuousController))
    parts = tuple(c for c in controllables if c not in laws)
    return SimpleNamespace(
        _parts_with_states=tuple(p for p in parts if p._acts_on_its_own()),
        _continuous_controllers=laws,
        effectors=SimpleNamespace(
            get_components=lambda: [
                p for p in parts if isinstance(p, (ThrustEffector, ForceEffector))
            ]
        ),
        controllables=controllables,
    )


def _reads(flight, t, u, controls, post_processing=False):  # pylint: disable=unused-argument
    """A phase whose derivative and post-process value are the deflection
    it read."""
    if post_processing:
        return [controls["canards.deflection"]]
    return [controls["canards.deflection"], *[0.0] * 12]


_PHASE = _PhaseDynamics("toy", _reads, CANONICAL_STATE_NAMES, post_process_vars=("d",))


def _roll_law(states, controls, conditions):  # pylint: disable=unused-argument
    controls.canards.deflection = -0.05 * conditions.state.w3


def _state(roll_rate):
    return [0.0] * 12 + [roll_rate]


def test_a_law_writes_the_controls_the_physics_reads():
    canards = _canards()
    extended = _with_controllables(
        _PHASE, _rocket(canards, ContinuousController(_roll_law))
    )
    assert extended.states == CANONICAL_STATE_NAMES
    controls = {"canards.deflection": 0.0}
    assert extended.derivative(_FLIGHT, 0.0, _state(2.0), controls)[0] == -0.1
    # In the post-processed values too, and what the law wrote is recorded
    # with them, as ``controls.<key>.<control>``
    assert extended.post_process_vars == ("d", "controls.canards.deflection")
    assert extended.derivative(
        _FLIGHT, 0.0, _state(2.0), controls, post_processing=True
    ) == [-0.1, -0.1]
    # The flight's own control inputs are left as they were
    assert controls == {"canards.deflection": 0.0}


def test_what_a_law_wrote_is_recorded_even_by_a_phase_with_no_post_values():
    bare = _PhaseDynamics("bare", _reads, CANONICAL_STATE_NAMES)
    extended = _with_controllables(
        bare, _rocket(_canards(), ContinuousController(_roll_law))
    )
    assert extended.post_process_vars == ("controls.canards.deflection",)
    assert extended.post_process_at(
        _FLIGHT, 0.0, _state(2.0), {"canards.deflection": 0.0}
    ) == [-0.1]
    # Without a law nothing is recorded: the sampled values stand
    assert _with_controllables(_PHASE, _rocket(_canards())).post_process_vars == ("d",)


def test_a_law_write_is_kept_inside_the_limits():
    clamped = _with_controllables(
        _PHASE, _rocket(_canards(), ContinuousController(_roll_law))
    )
    assert (
        clamped.derivative(_FLIGHT, 0.0, _state(10.0), {"canards.deflection": 0})[0]
        == -0.2
    )
    free = _with_controllables(
        _PHASE, _rocket(_canards(limits=None), ContinuousController(_roll_law))
    )
    assert (
        free.derivative(_FLIGHT, 0.0, _state(10.0), {"canards.deflection": 0})[0]
        == -0.5
    )


def test_a_law_cannot_invent_a_control_input():
    def wrong(states, controls, conditions):  # pylint: disable=unused-argument
        controls.canards.deflexion = 0.1

    extended = _with_controllables(
        _PHASE, _rocket(_canards(), ContinuousController(wrong))
    )
    with pytest.raises(AttributeError, match="no control named 'deflexion'"):
        extended.derivative(_FLIGHT, 0.0, _state(0.0), {"canards.deflection": 0})


def test_a_law_sees_the_control_inputs_and_not_the_states():
    def reads(states, controls, conditions):  # pylint: disable=unused-argument
        assert list(controls) == ["Canards"]
        assert "roll_pi_integral" not in controls
        assert "roll_pi" not in controls

    extended = _with_controllables(
        _PHASE,
        _rocket(_canards(), _RollPI(name="Roll PI"), ContinuousController(reads)),
    )
    extended.derivative(_FLIGHT, 0.0, [*_state(0.0), 0.0], {"canards.deflection": 0})


def test_a_law_reads_what_a_sampled_controller_wrote():
    """The law sees every channel, including another controller's output,
    and a law that leaves a channel alone leaves the sampled value in force."""

    def tracking(states, controls, conditions):  # pylint: disable=unused-argument
        controls.canards.deflection = (
            controls.guidance.rate_setpoint - conditions.state.w3
        )

    setpoint = ContinuousController.__mro__[1]("Guidance", ("rate_setpoint",))
    extended = _with_controllables(
        _PHASE,
        _rocket(_canards(limits=(-1, 1)), setpoint, ContinuousController(tracking)),
    )
    controls = {"canards.deflection": 0.0, "guidance.rate_setpoint": 0.5}
    assert extended.derivative(_FLIGHT, 0.0, _state(0.2), controls)[0] == pytest.approx(
        0.3
    )


class _RollPI(ContinuousController):
    states = ("integral",)

    def state_derivative(self, states, commands, conditions):
        return [0.5 - conditions.state.w3]

    def law(self, states, controls, conditions):
        controls.canards.deflection = 0.1 * (0.5 - conditions.state.w3) + states[0]


def _roll_pi():
    """The same PI law as ``_RollPI``, built from functions."""

    def law(states, controls, conditions):
        controls.canards.deflection = 0.1 * (0.5 - conditions.state.w3) + states[0]

    return ContinuousController(
        law,
        states=("integral",),
        state_derivative=lambda states, controls, conditions: [
            0.5 - conditions.state.w3
        ],
        name="Roll PI",
    )


@pytest.mark.parametrize("build", [_roll_pi, lambda: _RollPI(name="Roll PI")])
def test_a_law_with_states_is_integrated_with_the_flight(build):
    extended = _with_controllables(SIX_DOF_DYNAMICS, _rocket(_canards(), build()))
    assert extended.states[-1] == "roll_pi_integral"
    toy = _with_controllables(_PHASE, _rocket(_canards(), build()))
    u = [*_state(0.2), 0.04]  # the integral so far
    derivative = toy.derivative(_FLIGHT, 0.0, u, {"canards.deflection": 0.0})
    assert derivative[0] == pytest.approx(0.1 * 0.3 + 0.04)
    assert derivative[-1] == pytest.approx(0.3)


class _Servo(GenericSurface):
    """Canards that reach the commanded deflection with a first-order lag."""

    states = ("angle",)

    def state_derivative(self, states, commands, conditions):
        return [(commands[0] - states[0]) / 0.05]

    def actual_controls(self, states, commands, conditions):
        return [states[0]]


def _servo():
    return _Servo(
        0.01,
        0.1,
        {},
        name="Canards",
        controls=("deflection",),
        control_limits={"deflection": (-0.2, 0.2)},
    )


def test_a_law_runs_before_the_parts_map_their_commands():
    """A servo lag reads the command the law wrote, not the sampled value."""
    extended = _with_controllables(
        _PHASE, _rocket(_servo(), ContinuousController(_roll_law))
    )
    u = [*_state(2.0), 0.03]  # the servo's angle
    derivative = extended.derivative(_FLIGHT, 0.0, u, {"canards.deflection": 0.0})
    assert derivative[0] == 0.03  # the physics acts with the angle reached
    assert derivative[-1] == pytest.approx(
        (-0.1 - 0.03) / 0.05
    )  # toward the law's command


def test_a_controller_state_follows_the_commands_and_not_the_angle_reached():
    """The derivative of a controller's state reads the control inputs as the
    laws left them, kept inside their limits: a servo's angle does not
    replace them. The controllers' states come before the other
    controllables', in the order the two are worked out."""
    seen = []

    def law(states, controls, conditions):  # pylint: disable=unused-argument
        controls.canards.deflection = -0.5  # beyond the limit of -0.2

    def rate(states, commands, conditions):  # pylint: disable=unused-argument
        seen.append(commands.canards.deflection)
        return [1.5]

    controller = ContinuousController(
        law, states=("integral",), state_derivative=rate, name="Roll PI"
    )
    extended = _with_controllables(_PHASE, _rocket(_servo(), controller))
    assert extended.states[13:] == ("roll_pi_integral", "canards_angle")
    u = [*_state(0.0), 0.0, 0.03]  # the integral, then the servo's angle
    derivative = extended.derivative(_FLIGHT, 0.0, u, {"canards.deflection": 0.0})
    assert seen == [-0.2]
    assert derivative[0] == 0.03  # the physics acts with the angle reached
    assert derivative[13:] == [1.5, pytest.approx((-0.2 - 0.03) / 0.05)]
    # Seeded in the same order, each from its controllable
    assert extended.initial_state(None, 0.0, list(range(13)))[13:] == [0.0, 0.0]


def _still(flight, t, u, controls, post_processing=False):  # pylint: disable=unused-argument
    """A phase whose states do not change, and that reports one value."""
    return [0.0] if post_processing else [0.0] * len(u)


def _watching(phase_states):
    """Return the dynamics of a phase with ``phase_states`` flown with a law,
    and the list of what the law read of the conditions at each call."""
    seen = []

    def law(states, controls, conditions):  # pylint: disable=unused-argument
        state = conditions.state
        seen.append((state.z, state.vz, state.w3, conditions.height_agl))

    phase = _PhaseDynamics("other", _still, phase_states, post_process_vars=("d",))
    return _with_controllables(phase, _rocket(ContinuousController(law))), seen


# Stands for a flight on ground 100 m above sea level
_GROUND = SimpleNamespace(elevation=100.0)


@pytest.mark.parametrize(
    "phase_states",
    [(*CANONICAL_STATE_NAMES, "fuel"), ("fuel", *reversed(CANONICAL_STATE_NAMES))],
)
def test_a_law_reads_the_standard_states_in_a_phase_with_more_states(phase_states):
    seen = []

    def law(states, controls, conditions):  # pylint: disable=unused-argument
        state = conditions.state
        seen.append((state.z, state.vz, state.w3, state.fuel, conditions.height_agl))

    phase = _PhaseDynamics("more", _still, phase_states)
    extended = _with_controllables(phase, _rocket(ContinuousController(law)))
    values = {"z": 350.0, "vz": 3.0, "w3": 0.5, "fuel": 7.0}
    u = [values.get(name, 0.0) for name in phase_states]
    flight = SimpleNamespace(rocket=None, env=_GROUND)
    assert extended.derivative(flight, 0.0, u, {}) == [0.0] * 14
    assert seen == [(350.0, 3.0, 0.5, 7.0, 250.0)]


def test_a_law_reads_the_standard_states_in_a_phase_with_fewer_states():
    """The states the phase does not follow are the ones it began with,
    which the bound dynamics carry for the phase."""
    extended, seen = _watching(("x", "y", "z", "vx", "vy", "vz"))
    flight = SimpleNamespace(rocket=None, env=_GROUND, _controls={})
    u = [0.0, 0.0, 350.0, 0.0, 0.0, 3.0]
    bound = extended.bind(flight)
    with pytest.raises(ValueError, match="Pass start_canonical"):
        bound(0.0, u)
    began = bound.started_from([0.0] * 12 + [0.5])
    assert began(0.0, u) == [0.0] * 6
    assert began.post_process_at(0.0, u, {}) == [0.0]
    # Another phase of the same dynamics starts from its own state
    later = bound.started_from([0.0] * 12 + [0.25])
    assert later.derivative_at(0.0, u, {}) == [0.0] * 6
    assert seen == [(350.0, 3.0, 0.5, 250.0)] * 2 + [(350.0, 3.0, 0.25, 250.0)]


def test_a_phase_that_was_not_extended_is_not_given_where_it_began():
    phase = _PhaseDynamics("bare", _still, ("x", "y", "z"), post_process_vars=("d",))
    bound = phase.bind(SimpleNamespace(rocket=_rocket()))
    assert bound.started_from([0.0] * 13) is bound
    assert bound.post_process_at(0.0, [0.0] * 3, {}) == [0.0]


def test_the_equations_are_handed_the_conditions_when_the_rocket_has_effectors():
    """A derivative that names ``conditions`` gets the same object the laws
    and the controllables were given; one that does not is called as before."""
    handed = []

    def with_conditions(flight, t, u, controls, conditions=None):  # pylint: disable=unused-argument
        handed.append(conditions)
        return [0.0] * 13

    def without(flight, t, u, controls):  # pylint: disable=unused-argument
        return [0.0] * 13

    class Pusher(ForceEffector):
        def force_and_moment(self, controls, conditions):
            return [0, 0, 0], [0, 0, 0]

    rocket = _rocket(Pusher(name="Pusher"))
    flight = SimpleNamespace(rocket=None, env=_GROUND)
    extended = _with_controllables(
        _PhaseDynamics("c", with_conditions, CANONICAL_STATE_NAMES), rocket
    )
    assert extended.derivative(flight, 0.0, _state(0.5), {}) == [0.0] * 13
    assert handed[0].state.w3 == 0.5
    plain = _with_controllables(
        _PhaseDynamics("p", without, CANONICAL_STATE_NAMES), rocket
    )
    assert plain.derivative(flight, 0.0, _state(0.5), {}) == [0.0] * 13
    # A rocket with nothing to add is flown by the phase as it is
    bare = _PhaseDynamics("b", with_conditions, CANONICAL_STATE_NAMES)
    assert _with_controllables(bare, _rocket()) is bare


def test_a_continuous_controller_with_states_needs_no_law():
    """An observer only integrates its states and writes no controls."""
    observer = ContinuousController(
        states=("w3_hat",),
        state_derivative=lambda states, controls, conditions: [
            20.0 * (conditions.state.w3 - states[0])
        ],
        name="Observer",
    )
    assert observer.law_function is None
    extended = _with_controllables(_PHASE, _rocket(_canards(), observer))
    controls = {"canards.deflection": 0.1}
    derivative = extended.derivative(_FLIGHT, 0.0, [*_state(2.0), 0.5], controls)
    assert derivative[-1] == pytest.approx(20.0 * (2.0 - 0.5))
    assert controls == {"canards.deflection": 0.1}


def test_a_continuous_controller_checks_its_law():
    with pytest.raises(TypeError, match="needs a law"):
        ContinuousController()
    with pytest.raises(TypeError, match="callable"):
        ContinuousController("not a function")
    with pytest.raises(
        TypeError, match="law of 'Continuous Controller' must take 3 arguments"
    ):
        ContinuousController(lambda states, controls: None)
    with pytest.raises(TypeError, match="needs a state_derivative"):
        ContinuousController(_roll_law, states=("integral",))
    with pytest.raises(
        TypeError, match="state_derivative of 'Continuous Controller' must take"
    ):
        ContinuousController(_roll_law, states=("x",), state_derivative=lambda c: [0])
    with pytest.raises(ValueError, match="letters, digits and underscores"):
        ContinuousController(
            _roll_law, states=("bad name",), state_derivative=_roll_law
        )
    assert ContinuousController(_roll_law).controls == ()
    assert ContinuousController(_roll_law).states == ()
    assert _RollPI(name="Roll PI").states == ("integral",)


def test_two_parts_cannot_share_a_state_name(calisto):
    calisto.add_controller(_RollPI(name="Roll PI"))
    with pytest.raises(ValueError, match="state named 'roll_pi_integral'"):
        calisto.add_controller(_RollPI(name="Roll PI"))


def test_a_rocket_keeps_its_continuous_controllers(calisto):
    law = ContinuousController(_roll_law, name="Roll Law")
    calisto.add_controller(law)
    assert calisto.continuous_controllers == (law,)
    assert calisto.controllers == ()
    assert calisto.control_names == ()
    assert law not in calisto.controllables
    with pytest.raises(TypeError, match="ThrustEffector or ForceEffector"):
        calisto.add_effector(ContinuousController(_roll_law))


def test_a_continuous_controller_is_saved_and_loaded(calisto):
    calisto.add_controller(ContinuousController(_roll_law, name="Roll Law"))
    loaded = json.loads(json.dumps(calisto, cls=RocketPyEncoder), cls=RocketPyDecoder)
    (law,) = loaded.continuous_controllers
    assert isinstance(law, ContinuousController) and law.name == "Roll Law"
    controls = {"canards.deflection": 0.0}
    view = ControlsView(controls)
    law.law((), view, FlightConditions(None, None, 0.0, _state(2.0)))
    assert controls == {"canards.deflection": -0.1}
    text = json.dumps(calisto, cls=RocketPyEncoder, allow_pickle=False)
    with pytest.warns(UserWarning, match="'Roll Law' could not be loaded"):
        loaded = json.loads(text, cls=RocketPyDecoder)
    assert loaded.continuous_controllers == ()


def test_one_state_name_given_alone_is_one_state():
    controller = ContinuousController(
        states="integral", state_derivative=lambda s, c, f: [0.0]
    )
    assert controller.states == ("integral",)


def test_a_state_derivative_without_states_is_refused():
    with pytest.raises(ValueError, match="no states"):
        ContinuousController(
            lambda s, c, f: None, state_derivative=lambda s, c, f: [0.0]
        )


def test_info_prints_the_functions_and_the_states(capsys):
    def roll_law(states, controls, conditions):  # pylint: disable=unused-argument
        return None

    controller = ContinuousController(
        roll_law,
        states=("integral",),
        state_derivative=lambda s, c, f: [0.0],
        name="Roll Law",
    )
    controller.info()
    out = capsys.readouterr().out
    assert "ContinuousController: Roll Law" in out
    assert "Control law: roll_law" in out and "States: integral" in out


def test_an_effector_prints_its_controls_and_limits(capsys):
    effector = ForceEffector(
        lambda on, conditions: ([0, 0, on], [0, 0, 0]),
        controls=("on",),
        control_limits={"on": (0, 1)},
        name="Thruster",
    )
    effector.all_info()
    out = capsys.readouterr().out
    assert "ForceEffector: Thruster" in out
    assert "Control on: starts at 0.0, from 0 to 1" in out
