"""Effectors, the controllables with controls that are not aerodynamic surfaces:
adding them to a rocket, the generic effector, and the hooks that bring them
into the equations of motion."""

import json
import math
import warnings
from types import SimpleNamespace

import pytest

from rocketpy import Controller, GenericSurface
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control import (
    Controllable,
    FlightConditions,
    ForceEffector,
    ThrustEffector,
)
from rocketpy.control._helpers import initial_controls
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.simulation.helpers.dynamics import (
    SIX_DOF_DYNAMICS,
    _with_controllables,
)
from rocketpy.simulation.helpers.flight_derivatives import (
    _effector_force_and_moment,
    _effector_thrust,
)
from tests.fixtures.effectors.effector_fixtures import Gimbal


def _flight_of(rocket):
    """A stand-in flight with the control inputs of ``rocket``."""
    controls, channels, limits = initial_controls(rocket.controllables)
    flight = SimpleNamespace(
        rocket=rocket,
        _controls=controls,
        _control_names=channels,
        _control_limits=limits,
        env=None,
    )
    return flight


def _conditions(flight, t=0.0):
    """What the stand-in flight's effectors are told at ``t``."""
    return FlightConditions(flight.rocket, flight.env, t, [0.0] * 13)


def test_the_conditions_give_the_state_by_name_and_the_height_above_ground():
    env = SimpleNamespace(elevation=100.0)
    state = [0.0, 0.0, 350.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    conditions = FlightConditions(None, env, 0.0, state)
    assert conditions.state.w3 == 0.5
    assert conditions.state[12] == 0.5
    assert conditions.state.vz == conditions.state[5] == 3.0
    assert conditions.height_agl == 250.0
    # a state the phase follows besides the thirteen is read by its name
    more = FlightConditions(None, env, 0.0, state, (*"abc", "fuel"), [0, 0, 0, 7.0])
    assert more.state.fuel == 7.0
    assert more.state.w3 == 0.5


class _Both(Gimbal, ForceEffector):
    """Turns the thrust and pushes the rocket: filed under both kinds."""

    def force_and_moment(self, controls, conditions):  # pylint: disable=unused-argument
        return [0, 0, 0], [0, 0, 0]


def test_add_effector_files_the_part_by_what_it_does(calisto, gimbal, push_effector):
    calisto.add_effector(gimbal)
    calisto.add_effector(push_effector)

    assert calisto.effectors.get_components() == [gimbal, push_effector]
    assert calisto._thrust_effectors == (gimbal,)
    assert calisto._force_effectors == (push_effector,)
    assert calisto.control_names[-2:] == ("gimbal.pitch", "push.on")


def test_a_part_that_acts_in_two_ways_has_its_channels_once(calisto):
    both = _Both(name="Vectored Thruster")
    calisto.add_effector(both)
    assert calisto.effectors.get_components() == [both]
    assert calisto._thrust_effectors == (both,)
    assert calisto._force_effectors == (both,)
    assert calisto.control_names.count("vectored_thruster.pitch") == 1


def test_add_effector_refuses_what_is_not_a_part_with_controls(calisto):
    with pytest.raises(TypeError, match="ThrustEffector or ForceEffector"):
        calisto.add_effector(SimpleNamespace(name="x", controls=()))


def test_add_effector_refuses_a_surface(calisto):
    with pytest.raises(TypeError, match="add_surfaces"):
        calisto.add_effector(GenericSurface(1.0, 1.0, {}, name="Canards"))


def test_an_effector_without_its_method_cannot_be_built():
    class Forgetful(ThrustEffector):
        """A thrust effector that does not say what it does to the thrust."""

    with pytest.raises(TypeError, match="needs a thrust_and_nozzle"):
        Forgetful(name="Forgetful")
    with pytest.raises(TypeError, match="needs a force_and_moment"):
        ForceEffector(name="Forgetful")


class _Canopy(Controllable):
    def __init__(self):
        super().__init__("Canopy", ("brake",))


def test_a_part_with_no_physics_is_refused(calisto):
    """A controllable that is none of the three kinds of effector would do nothing in
    a flight, so adding it as one is an error, not a silent no-op."""
    with pytest.raises(TypeError, match="ThrustEffector or ForceEffector"):
        calisto.add_effector(_Canopy())
    assert len(calisto.effectors) == 0
    assert calisto.control_names == ()


class _Damper(ForceEffector):
    """A controllable with a state of its own and no controls."""

    states = ("deflection",)

    def __init__(self):
        super().__init__(name="Damper")

    def force_and_moment(self, controls, conditions):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def test_a_part_with_states_and_no_controls_is_still_integrated(calisto):
    damper = _Damper()
    calisto.add_effector(damper)
    assert calisto.controllables == ()
    extended = _with_controllables(SIX_DOF_DYNAMICS, calisto)
    assert extended.states[-1] == "damper_deflection"


def test_add_effector_refuses_a_name_already_used(calisto, gimbal):
    calisto.add_effector(gimbal)
    with pytest.raises(ValueError, match="both have controls"):
        calisto.add_effector(Gimbal())
    assert calisto.effectors.get_components() == [gimbal]


def test_effectors_take_no_part_in_stability(calisto, push_effector, side_effector):
    margin = calisto.static_margin(0)
    center = calisto.aerodynamic_center(0.3)
    calisto.add_effector(push_effector)
    calisto.add_effector(side_effector)
    assert calisto.static_margin(0) == margin
    assert calisto.aerodynamic_center(0.3) == center
    assert all(
        not isinstance(surface, ForceEffector)
        for surface, _ in calisto.aerodynamic_surfaces
    )


def test_effectors_are_not_in_the_surface_of_the_whole_rocket(
    calisto_robust, push_effector
):
    before = calisto_robust.to_surface()
    calisto_robust.add_effector(push_effector)
    after = calisto_robust.to_surface()
    arguments = (0.05, 0.0, 0.3, 1e6, 0.0, 0.0, 0.0)
    for old, new in zip(before, after):
        assert new.controls == ()
        assert new.cN(*arguments) == old.cN(*arguments)


def test_an_effector_built_from_a_function_is_given_the_controls_spread_out():
    effector = ForceEffector(
        lambda a, b, conditions: ([a, b, 0], [0, 0, a * b * conditions]),
        controls=("a", "b"),
        control_defaults={"b": 2.0},
        name="RCS Pair",
    )
    force, moment = effector.force_and_moment([3.0, 2.0], 0.5)
    assert list(force) == [3.0, 2.0, 0] and list(moment) == [0, 0, 3.0]
    assert effector.control_defaults == {"a": 0.0, "b": 2.0}
    assert repr(effector) == "ForceEffector(name='RCS Pair', controls=('a', 'b'))"

    gimbal = ThrustEffector(
        lambda thrust, nozzle, pitch, conditions: (
            [thrust[2] * pitch, 0, thrust[2]],
            [pitch, 0, nozzle[2]],
        ),
        controls=("pitch",),
        name="Gimbal",
    )
    assert gimbal.thrust_and_nozzle([0, 0, 10.0], [0, 0, -1.0], [0.1], None) == (
        [1.0, 0, 10.0],
        [0.1, 0, -1.0],
    )


@pytest.mark.parametrize(
    "function", [lambda conditions: None, lambda: None, "not a function"]
)
def test_an_effector_checks_its_function(function):
    with pytest.raises(TypeError, match="force_and_moment"):
        ForceEffector(function, controls=("on",))
    with pytest.raises(TypeError, match="thrust_and_nozzle"):
        ThrustEffector(function, controls=("on",))


def test_an_effector_built_from_a_function_is_saved_and_loaded(calisto, push_effector):
    calisto.add_effector(push_effector)
    loaded = json.loads(json.dumps(calisto, cls=RocketPyEncoder), cls=RocketPyDecoder)
    (effector,) = loaded.effectors.get_components()
    assert type(effector) is ForceEffector  # pylint: disable=unidiomatic-typecheck
    assert effector.name == "Push"
    assert effector.control_limits == {"on": (0, 1)}
    assert list(effector.force_and_moment([1.0], None)[0]) == [0, 0, 10.0]
    assert loaded.control_names == calisto.control_names


def test_a_part_written_by_the_user_is_saved_and_loaded(calisto, gimbal):
    """A subclass of an effector class holds its physics in code, so it is
    saved pickled and comes back as the same class, controls included."""
    calisto.add_effector(gimbal)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = json.dumps(calisto, cls=RocketPyEncoder)
        loaded = json.loads(text, cls=RocketPyDecoder)
    assert [type(c) for c in loaded.effectors.get_components()] == [Gimbal]
    assert loaded.control_names == calisto.control_names
    assert loaded.effectors[0].component.control_limits == gimbal.control_limits
    # Readable next to the pickled controllable
    saved = json.loads(text)["effectors"][0][0]  # [component, position] pairs
    assert saved["name"] == "Gimbal" and saved["controls"] == ["pitch"]


def test_a_part_saved_without_pickling_is_left_out_with_a_warning(
    calisto, gimbal, push_effector
):
    calisto.add_effector(gimbal)
    calisto.add_effector(push_effector)
    text = json.dumps(calisto, cls=RocketPyEncoder, allow_pickle=False)
    with pytest.warns(UserWarning, match="'Gimbal' could not be loaded"):
        with pytest.warns(UserWarning, match="'Push' could not be loaded"):
            loaded = json.loads(text, cls=RocketPyDecoder)
    assert len(loaded.effectors) == 0
    assert loaded.control_names == ()


def test_the_gimbal_turns_the_thrust_and_makes_a_moment(calisto, gimbal):
    calisto.add_effector(gimbal)
    flight = _flight_of(calisto)
    flight._controls["gimbal.pitch"] = 0.1

    force, moment, _ = _effector_thrust(
        flight, _conditions(flight, 1.0), 1000.0, flight._controls
    )

    assert list(force) == pytest.approx([1000 * math.sin(0.1), 0, 1000 * math.cos(0.1)])
    # the nozzle is behind the center of dry mass, so a thrust tipped toward
    # +x turns the nose toward -x: a negative moment about y
    lever = calisto.nozzle_to_cdm
    assert list(moment) == pytest.approx([0, -lever * 1000 * math.sin(0.1), 0])


def test_the_thrust_moment_includes_the_eccentricity(calisto):
    calisto.add_thrust_eccentricity(x=0.01, y=-0.02)
    calisto.add_effector(Gimbal())
    flight = _flight_of(calisto)
    _, moment, _ = _effector_thrust(
        flight, _conditions(flight, 1.0), 1000.0, flight._controls
    )
    # as the equations without effectors add it: M1 += ey T, M2 -= ex T
    assert list(moment) == pytest.approx(
        [
            calisto.thrust_eccentricity_y * 1000,
            -calisto.thrust_eccentricity_x * 1000,
            0,
        ]
    )


def test_the_nozzle_exit_is_behind_the_center_of_dry_mass(calisto):
    calisto.add_thrust_eccentricity(x=0.01, y=-0.02)
    assert list(calisto.nozzle_exit) == pytest.approx(
        [0.01, -0.02, -calisto.nozzle_to_cdm]
    )


def test_effectors_add_up(calisto, push_effector, side_effector):
    calisto.add_effector(push_effector)
    calisto.add_effector(side_effector)
    flight = _flight_of(calisto)
    flight._controls.update({"push.on": 1.0, "side.on": 0.5})
    force, moment = _effector_force_and_moment(
        flight, _conditions(flight), flight._controls
    )
    assert list(force) == [5.0, 0.0, 10.0]
    assert list(moment) == [0.0, 0.0, 0.5]


def _sideways(on, conditions):  # pylint: disable=unused-argument
    """10 N toward +x and a 0.5 N m couple about the axis, when on."""
    return [10.0 * on, 0.0, 0.0], [0.0, 0.0, 0.5 * on]


def test_a_placed_effector_makes_the_moment_of_its_force(calisto):
    cdm = calisto.center_of_dry_mass_position
    calisto.add_effector(
        ForceEffector(_sideways, controls=("on",), name="Side"), position=cdm + 0.4
    )
    assert calisto.effectors[0].position == Vector([0, 0, cdm + 0.4])
    flight = _flight_of(calisto)
    flight._controls["side.on"] = 1.0
    force, moment = _effector_force_and_moment(
        flight, _conditions(flight), flight._controls
    )
    assert list(force) == [10.0, 0.0, 0.0]
    # r x F = (0, 0, 0.4) x (10, 0, 0) = (0, 4, 0), plus the couple returned
    assert list(moment) == pytest.approx([0.0, 4.0, 0.5])


def _roll_push(on, conditions):  # pylint: disable=unused-argument
    """10 N in the positive roll direction at angular position 90, when on:
    (-cos 90, -sin 90, 0) = (0, -1, 0)."""
    return [0.0, -10.0 * on, 0.0], [0.0, 0.0, 0.0]


@pytest.mark.parametrize("fixture", ["calisto", "calisto_nose_to_tail"])
def test_an_effector_is_placed_around_the_tube_like_a_fin(request, fixture):
    rocket = request.getfixturevalue(fixture)
    cdm = rocket.center_of_dry_mass_position
    rocket.add_effector(
        ForceEffector(_roll_push, controls=("on",), name="Roll"),
        position=cdm + 0.2 * rocket._csys,
        angular_position=90,
        radius=0.1,
    )
    # In the user's coordinate system the x axis flips with the orientation
    assert list(rocket.effectors[0].position) == pytest.approx(
        [-0.1 * rocket._csys, 0.0, cdm + 0.2 * rocket._csys]
    )
    # In the body frame the point is the same: on -x, 0.2 m toward the nose
    point = rocket._effector_points[rocket.effectors[0].component]
    assert list(point) == pytest.approx([-0.1, 0.0, 0.2])
    flight = _flight_of(rocket)
    flight._controls["roll.on"] = 1.0
    _, moment = _effector_force_and_moment(
        flight, _conditions(flight), flight._controls
    )
    # A tangential force at radius R gives a roll moment R F, positive here
    assert list(moment) == pytest.approx([2.0, 0.0, 1.0])


def test_an_effector_is_placed_from_the_center_of_dry_mass(calisto):
    # The center of mass eccentricity moves the center of dry mass off the
    # axis, and with it the lever arm of an effector, as for a surface
    cdm = calisto.center_of_dry_mass_position
    effector = ForceEffector(_roll_push, controls=("on",), name="Roll")
    calisto.add_effector(effector, position=(0.1, 0.0, cdm + 0.2))
    assert list(calisto._effector_points[effector]) == pytest.approx([0.1, 0.0, 0.2])
    calisto.add_cm_eccentricity(0.01, -0.02)
    assert list(calisto._effector_points[effector]) == pytest.approx([0.09, 0.02, 0.2])


def test_an_effector_is_added_once(calisto, push_effector):
    calisto.add_effector(push_effector)
    with pytest.raises(ValueError, match="already on the rocket"):
        calisto.add_effector(push_effector, position=0.5)
    assert len(calisto.effectors) == 1


def test_an_effector_placed_at_a_point_takes_no_angular_position(calisto):
    effector = ForceEffector(_roll_push, controls=("on",), name="Roll")
    calisto.add_effector(effector, position=(0.1, 0.2, 0.3))
    assert calisto.effectors[0].position == Vector([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="not used"):
        calisto.add_effector(
            ForceEffector(_roll_push, controls=("on",), name="Other"),
            position=(0.1, 0.2, 0.3),
            radius=1,
        )


def test_an_effector_placement_is_checked(calisto, gimbal):
    with pytest.raises(TypeError, match="acts at the motor's nozzle"):
        calisto.add_effector(gimbal, position=0.5)
    with pytest.raises(ValueError, match="no position"):
        calisto.add_effector(
            ForceEffector(_roll_push, controls=("on",), name="Roll"), radius=0.1
        )
    assert len(calisto.effectors) == 0


def test_a_placed_effector_is_saved_and_loaded(calisto, side_effector):
    calisto.add_effector(side_effector, position=0.7, angular_position=45, radius=0.05)
    loaded = json.loads(json.dumps(calisto, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert list(loaded.effectors[0].position) == pytest.approx(
        list(calisto.effectors[0].position)
    )
    loaded_point = loaded._effector_points[loaded.effectors[0].component]
    point = calisto._effector_points[side_effector]
    assert list(loaded_point) == pytest.approx(list(point))


def test_controllables_are_only_the_parts_with_controls(calisto, gimbal):
    # calisto's nose cone, fins and tail have no controls
    assert calisto.controllables == ()
    canards = GenericSurface(1.0, 1.0, {}, name="Canards", controls=("deflection",))
    calisto.add_surfaces(canards, 0.5)
    calisto.add_effector(gimbal)
    assert calisto.controllables == (canards, gimbal)


class _PivotedGimbal(Gimbal):
    """A gimbal that turns the nozzle about a pivot ``length`` above its exit,
    so the exit moves with it."""

    def __init__(self, length):
        super().__init__()
        self.length = length

    def thrust_and_nozzle(self, thrust, nozzle, controls, conditions):
        turned, _ = super().thrust_and_nozzle(thrust, nozzle, controls, conditions)
        angle = controls[0]
        pivot = nozzle + Vector([0, 0, self.length])
        # the nozzle points backwards: turning the thrust toward +x swings
        # the exit toward -x
        offset = Vector(
            [-self.length * math.sin(angle), 0, -self.length * math.cos(angle)]
        )
        return turned, pivot + offset


def test_a_moved_nozzle_sets_the_moment_of_the_thrust(calisto):
    """The turned thrust acts along a line through the pivot, so the moment's
    lever is the pivot's distance to the center of dry mass, not the exit's."""
    calisto.add_effector(_PivotedGimbal(length=0.1))
    flight = _flight_of(calisto)
    flight._controls["gimbal.pitch"] = 0.1
    force, moment, nozzle = _effector_thrust(
        flight, _conditions(flight, 1.0), 1000.0, flight._controls
    )
    pivot_lever = calisto.nozzle_to_cdm - 0.1
    assert list(moment) == pytest.approx([0, -pivot_lever * force.x, 0])
    assert list(nozzle) == pytest.approx(
        [-0.1 * math.sin(0.1), 0, -pivot_lever - 0.1 * math.cos(0.1)]
    )


def test_a_nozzle_no_effector_moves_stays_at_the_rocket_exit(calisto, gimbal):
    calisto.add_effector(gimbal)
    flight = _flight_of(calisto)
    nozzle = _effector_thrust(flight, _conditions(flight), 1000.0, flight._controls)[2]
    assert list(nozzle) == list(calisto.nozzle_exit)


def test_an_effector_reads_its_own_states(calisto):
    class Thruster(ForceEffector):
        """A thruster whose force follows the gas it has left."""

        states = ("gas",)

        def force_and_moment(self, controls, conditions):
            (gas,) = conditions.states_of(self)
            return [0, 0, gas], [0, 0, 0]

    calisto.add_effector(Thruster(name="Thruster"))
    flight = _flight_of(calisto)
    # the flight follows the controllable's states by name with the rocket's
    conditions = FlightConditions(
        flight.rocket, flight.env, 0.0, [0.0] * 13, ("thruster_gas",), [0.7]
    )
    force, _ = _effector_force_and_moment(flight, conditions, flight._controls)
    assert force.z == 0.7


def test_the_rocket_prints_what_controls_it(calisto, gimbal, push_effector, capsys):
    """Its controllers, each part with controls and where each effector acts."""
    calisto.add_effector(gimbal)
    calisto.add_effector(push_effector, position=0.4)
    calisto.add_controller(Controller(lambda context: None, 10, name="Sampled"))
    calisto.prints.all()
    out = capsys.readouterr().out
    assert "Controller 'Sampled' with sampling rate 10 Hz." in out
    assert "context.controls.gimbal.<control>" in out
    assert "'Gimbal', acting on the motor's thrust" in out
    assert "'Push', at (" in out


def test_a_rocket_without_controls_prints_no_controls_block(calisto, capsys):
    calisto.prints.all()
    assert "Controls" not in capsys.readouterr().out
