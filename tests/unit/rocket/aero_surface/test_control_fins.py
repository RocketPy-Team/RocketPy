"""ControlFins: a set of individual fins a controller turns one by one. Each
test compares the set against the same individual fins added to a rocket
with a cant angle, so no simulation is run."""

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import ControlFins, Function, TrapezoidalFin
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.simulation.events.views import ControlsView

POSITION = 0.8  # where the fins are added, in calisto's coordinate system


def _fin(angular_position=0.0, cant_angle=0.0, name="Fin"):
    return TrapezoidalFin(
        angular_position=angular_position,
        root_chord=0.08,
        tip_chord=0.04,
        span=0.06,
        rocket_radius=0.0635,
        cant_angle=cant_angle,
        name=name,
    )


def _individual_fins(rocket, n, deflections, start=0.0):
    """Add ``n`` individual fins to ``rocket``, each canted by its deflection,
    where ControlFins(n=n) would put them, and return them."""
    fins = []
    for i, deflection in enumerate(deflections):
        fin = _fin(start + 360 * i / n, math.degrees(deflection), f"fin {i}")
        rocket.add_surfaces(fin, POSITION)
        fins.append(fin)
    return fins


def _random_states(count=4, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        stream = Vector(list(rng.normal(size=3) * [5, 5, 1] + [0, 0, -100]))
        omega = Vector(list(rng.normal(size=3)))
        yield stream, omega


def _forces_of_individual_fins(rocket, fins, stream, omega, reference):
    """Sum the forces of individual fins, each fed the air where it sits, the
    way the flight feeds them, with ``stream`` given at ``reference``."""
    total = np.zeros(6)
    for fin in fins:
        cp = rocket.surfaces_cp_to_cdm[fin]
        local = stream - (omega ^ (cp - reference))
        speed = abs(local)
        total += np.array(
            fin.compute_forces_and_moments(local, speed, speed / 340, 1.1, cp, omega)
        )
    return total


def test_each_fin_has_a_numbered_channel(calisto):
    canards = ControlFins(_fin(angular_position=45), n=4, name="Canards")
    calisto.add_surfaces(canards, POSITION)

    assert canards.controls == ("fin_1", "fin_2", "fin_3", "fin_4")
    assert canards.angular_positions == (45.0, 135.0, 225.0, 315.0)
    assert calisto.control_names[-4:] == (
        "canards.fin_1",
        "canards.fin_2",
        "canards.fin_3",
        "canards.fin_4",
    )


def test_the_fin_classes_themselves_have_no_controls():
    assert _fin().controls == ()


def test_the_fin_given_is_neither_changed_nor_flown():
    fin = _fin(angular_position=10, cant_angle=2)
    canards = ControlFins(fin, n=3)

    assert fin.angular_position == 10 and fin.cant_angle == 2
    assert all(each is not fin for each in canards.fins)
    # The fins of the set are uncanted: their deflection turns them
    assert all(each.cant_angle == 0 for each in canards.fins)
    assert canards.control_defaults == dict.fromkeys(canards.controls, math.radians(2))


@pytest.mark.parametrize("deflections", [(0.0,) * 4, (0.05, -0.02, 0.1, 0.0)])
def test_the_set_flies_as_the_same_individual_fins(calisto, deflections, request):
    canards = ControlFins(_fin(), n=4, name="Canards")
    calisto.add_surfaces(canards, POSITION)
    reference = calisto.surfaces_cp_to_cdm[canards]
    other = request.getfixturevalue("calisto_robust")
    fins = _individual_fins(other, 4, deflections)

    for stream, omega in _random_states():
        speed = abs(stream)
        expected = _forces_of_individual_fins(other, fins, stream, omega, reference)
        got = canards.compute_forces_and_moments(
            stream, speed, speed / 340, 1.1, reference, omega, controls=deflections
        )
        assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_the_default_deflection_is_the_cant_of_the_fin_given(calisto):
    canards = ControlFins(_fin(cant_angle=3), n=4)
    calisto.add_surfaces(canards, POSITION)
    reference = calisto.surfaces_cp_to_cdm[canards]
    stream, omega = next(_random_states())
    speed = abs(stream)

    held = canards.compute_forces_and_moments(stream, speed, 0.3, 1.1, reference, omega)
    given = canards.compute_forces_and_moments(
        stream, speed, 0.3, 1.1, reference, omega, controls=(math.radians(3),) * 4
    )
    assert held == pytest.approx(given)


def test_the_same_deflection_on_every_fin_only_rolls_the_rocket():
    canards = ControlFins(_fin(), n=4)
    stream = Vector([0, 0, -100])
    R1, R2, _, M1, M2, M3 = canards.compute_forces_and_moments(
        stream, 100, 0.3, 1.2, Vector([0, 0, 1]), (0, 0, 0), controls=(0.1,) * 4
    )
    assert [R1, R2, M1, M2] == pytest.approx([0, 0, 0, 0], abs=1e-9)
    assert M3 != 0


def test_the_static_margin_is_that_of_the_individual_fins(calisto, calisto_robust):
    canards = ControlFins(_fin(cant_angle=2), n=4)
    calisto.add_surfaces(canards, POSITION)
    _individual_fins(calisto_robust, 4, (math.radians(2),) * 4)

    for mach in (0.3, 0.9):
        assert calisto.aerodynamic_center(mach) == pytest.approx(
            calisto_robust.aerodynamic_center(mach), abs=1e-9
        )
        assert calisto.stability_margin(mach, 0) == pytest.approx(
            calisto_robust.stability_margin(mach, 0), abs=1e-9
        )


def test_the_deflection_limits_apply_to_every_fin(calisto):
    canards = ControlFins(_fin(cant_angle=20), n=2, deflection_limits=(-0.2, 0.2))
    calisto.add_surfaces(canards, POSITION)

    assert canards.control_limits == {"fin_1": (-0.2, 0.2), "fin_2": (-0.2, 0.2)}
    # The cant given is outside the limits, so the default is clamped
    assert canards.control_defaults == {"fin_1": 0.2, "fin_2": 0.2}


@pytest.mark.parametrize(
    "fin, n, error",
    [
        ("not a fin", 4, TypeError),
        (None, 4, TypeError),
        (_fin(), 2.5, TypeError),
        (_fin(), 0, ValueError),
    ],
)
def test_wrong_inputs_are_refused(fin, n, error):
    with pytest.raises(error):
        ControlFins(fin, n=n)


def test_control_fins_are_saved_and_loaded():
    canards = ControlFins(
        _fin(angular_position=30, cant_angle=1),
        n=3,
        name="Canards",
        deflection_limits=(-0.3, 0.3),
    )
    loaded = json.loads(json.dumps(canards, cls=RocketPyEncoder), cls=RocketPyDecoder)

    assert isinstance(loaded, ControlFins)
    assert loaded.name == "Canards" and loaded.n == 3
    assert loaded.angular_positions == canards.angular_positions
    assert loaded.control_limits == canards.control_limits
    assert loaded.control_defaults == pytest.approx(canards.control_defaults)
    stream = Vector([1, 2, -100])
    args = (stream, abs(stream), 0.3, 1.2, Vector([0, 0, 1]), (0.1, 0.2, 0.3))
    assert loaded.compute_forces_and_moments(*args) == pytest.approx(
        canards.compute_forces_and_moments(*args)
    )


def test_info_says_which_fin_is_which(capsys):
    ControlFins(_fin(angular_position=90), n=2, name="Canards").info()
    out = capsys.readouterr().out
    assert "fin_1" in out and "90.00 deg" in out
    assert "fin_2" in out and "270.00 deg" in out


def _placed(canards, z):
    """The set as a rocket would leave it after adding it ``z`` meters ahead
    of its center of dry mass (behind it for a negative ``z``)."""
    canards._placed_at(Vector([0, 0, z]))
    return canards


def test_deflections_mix_the_roll_pitch_and_yaw_commands():
    canards = _placed(ControlFins(_fin(), n=4), 0.4)
    assert canards.deflections(roll=0.02) == pytest.approx(
        dict.fromkeys(canards.controls, -0.02)
    )
    assert canards.deflections(yaw=0.1) == pytest.approx(
        {"fin_1": 0.1, "fin_2": 0.0, "fin_3": -0.1, "fin_4": 0.0}
    )
    assert canards.deflections(pitch=0.1) == pytest.approx(
        {"fin_1": 0.0, "fin_2": -0.1, "fin_3": 0.0, "fin_4": 0.1}
    )
    mixed = canards.deflections(roll=0.02, pitch=0.05, yaw=0.1)
    assert mixed == pytest.approx(
        {"fin_1": 0.08, "fin_2": -0.07, "fin_3": -0.12, "fin_4": 0.03}
    )
    three = _placed(ControlFins(_fin(), n=3), 0.4).deflections(yaw=0.1)
    assert list(three.values()) == pytest.approx([0.1, -0.05, -0.05])


def test_tail_fins_turn_the_other_way_for_the_same_pitch_or_yaw_command():
    canards = _placed(ControlFins(_fin(), n=4), 0.4)
    tail_fins = _placed(ControlFins(_fin(), n=4), -0.9)
    for command in ("pitch", "yaw"):
        ahead = canards.deflections(**{command: 0.1})
        behind = tail_fins.deflections(**{command: 0.1})
        assert behind == pytest.approx({fin: -angle for fin, angle in ahead.items()})
    assert tail_fins.deflections(roll=0.1) == canards.deflections(roll=0.1)


@pytest.mark.parametrize("z", [0.4, -0.9])
@pytest.mark.parametrize("command, axis", [("pitch", 0), ("yaw", 1), ("roll", 2)])
def test_a_positive_command_turns_the_rocket_toward_the_positive_rate(z, command, axis):
    """Ahead of the center of dry mass or behind it, a positive command gives
    a positive moment about its own body axis and none about the others."""
    canards = _placed(ControlFins(_fin(), n=4), z)
    deflections = canards.deflections(**{command: 0.1})
    forces = canards.compute_forces_and_moments(
        Vector([0, 0, -100]),
        100,
        0.3,
        1.2,
        Vector([0, 0, z]),
        (0, 0, 0),
        controls=[deflections[control] for control in canards.controls],
    )
    moments = forces[3:6]
    assert moments[axis] > 0
    for other in {0, 1, 2} - {axis}:
        assert moments[other] == pytest.approx(0, abs=1e-9)


def test_pitch_and_yaw_commands_need_the_set_on_a_rocket(calisto):
    canards = ControlFins(_fin(), n=4, name="Canards")
    assert canards.deflections(roll=0.1) == dict.fromkeys(canards.controls, -0.1)
    for command in ("pitch", "yaw"):
        with pytest.raises(ValueError, match="not on a rocket yet"):
            canards.deflections(**{command: 0.1})

    calisto.add_surfaces(canards, POSITION)
    assert canards.deflections(yaw=0.1)["fin_1"] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="center of dry mass"):
        _placed(ControlFins(_fin(), n=4), 0.0).deflections(pitch=0.1)


def test_the_rocket_tells_the_fins_where_they_sit(calisto):
    canards = ControlFins(_fin(), n=4, name="Canards")
    calisto.add_surfaces(canards, POSITION)
    assert canards._position == calisto.surfaces_cp_to_cdm[canards]
    assert canards._position.z > 0

    tail_fins = ControlFins(_fin(), n=4, name="Tail fins")
    calisto.add_surfaces(tail_fins, -1.0)
    assert tail_fins._position.z < 0
    assert tail_fins.deflections(yaw=0.1)["fin_1"] == pytest.approx(-0.1)


def test_effectiveness_is_the_slope_of_the_coefficients():
    canards = _placed(ControlFins(_fin(), n=4), 0.4)
    effectiveness = canards.effectiveness()
    roll, pitch, yaw = (effectiveness[key](0.3) for key in ("roll", "pitch", "yaw"))
    assert roll > 0 and pitch > 0 and yaw > 0
    assert pitch == pytest.approx(yaw)
    # The roll coefficient at a small roll command, from the coefficients
    h = 0.01
    cl = canards.cl(0, 0, 0.3, 0, 0, 0, 0, -h, -h, -h, -h)  # with reynolds
    assert roll * h == pytest.approx(cl, rel=1e-3)
    # The yaw moment at a small yaw command: the force toward +x and the
    # moment about the point the set is added at, moved to the center of dry mass
    cY = canards.cY(0, 0, 0.3, 0, 0, 0, 0, h, 0, -h, 0)
    cn = canards.cn(0, 0, 0.3, 0, 0, 0, 0, h, 0, -h, 0)
    assert yaw * h == pytest.approx(cn + cY * 0.4 / canards.reference_length, rel=1e-3)
    assert effectiveness["roll"](0.8) != pytest.approx(roll)  # changes with Mach


def test_effectiveness_of_a_set_off_a_rocket_is_the_roll_one():
    assert set(ControlFins(_fin(), n=4).effectiveness()) == {"roll"}


def test_the_hinge_position_sets_where_a_fin_moves_to_when_it_turns():
    fin = _fin()
    at_edge = ControlFins(fin, n=4, hinge_position=0.0)
    quarter = ControlFins(fin, n=4, hinge_position=0.25)
    middle = ControlFins(fin, n=4)
    assert middle.hinge_position == 0.5
    offsets = [list(each._frame(0, 0.2)[2]) for each in (at_edge, quarter, middle)]
    # Turning about the leading edge does not move it; the shift grows with the hinge
    shift_quarter = np.subtract(offsets[1], offsets[0])
    shift_middle = np.subtract(offsets[2], offsets[0])
    assert shift_middle == pytest.approx(2 * shift_quarter)
    assert list(shift_middle) == pytest.approx(
        list(fin._cant_shift(0.2, fin._rotation_fin_to_body_uncanted))
    )
    with pytest.raises(ValueError, match="hinge position"):
        ControlFins(fin, n=4, hinge_position=1.5)


def test_the_hinge_position_is_saved_and_printed(capsys):
    canards = ControlFins(_fin(), n=4, hinge_position=0.3)
    loaded = json.loads(json.dumps(canards, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert loaded.hinge_position == 0.3
    canards.info()
    assert "Hinge line: at 30% of the root chord" in capsys.readouterr().out


def test_the_hinge_moment_is_the_normal_force_times_the_arm_to_the_hinge():
    fin = _fin()
    q = 30e3  # Pa
    at_edge = ControlFins(fin, n=4, hinge_position=0.0, name="Canards")
    at_cp = ControlFins(fin, n=4, hinge_position=fin.cpz / fin.root_chord)
    behind = ControlFins(fin, n=4, hinge_position=1.0)

    # Hinge ahead of the center of pressure: the air pushes a deflected fin
    # back toward zero. On the center of pressure: no torque. Behind it: the
    # air pushes the fin further the way it is turned
    torque = at_edge.hinge_moments(q, deflections=0.1)
    assert set(torque) == set(at_edge.controls)
    assert torque["fin_1"](0.3) < 0
    assert at_edge.hinge_moments(q, deflections=-0.1)["fin_1"](0.3) > 0
    assert at_cp.hinge_moments(q, deflections=0.1)["fin_1"](0.3) == pytest.approx(0)
    assert behind.hinge_moments(q, deflections=0.1)["fin_1"](0.3) > 0

    # It is the normal force times the arm, and scales with dynamic pressure
    # and with the arm
    expected = at_edge.hinge_moments(q, deflections=0.1)["fin_1"](0.3)
    force = -expected / fin.cpz
    arm = fin.cpz - 0.25 * fin.root_chord
    quarter = ControlFins(fin, n=4, hinge_position=0.25)
    assert quarter.hinge_moments(q, deflections=0.1)["fin_1"](0.3) == pytest.approx(
        -force * arm
    )
    assert at_edge.hinge_moments(2 * q, deflections=0.1)["fin_1"](0.3) == (
        pytest.approx(2 * expected)
    )

    # Deflections per control input, the rest at their defaults, zero here
    torque = at_edge.hinge_moments(q, deflections={"fin_2": 0.1})
    assert torque["fin_1"](0.3) == pytest.approx(0)
    assert torque["fin_2"](0.3) == pytest.approx(expected)

    # Head-on air at zero deflection gives no torque. Air crossing a fin does:
    # fin 1 is on the +y axis, so sideslip (toward x) loads it and the angle
    # of attack (toward y, along its span) does not
    assert at_edge.hinge_moments(q)["fin_1"](0.3) == pytest.approx(0)
    assert at_edge.hinge_moments(q, alpha=0.05)["fin_1"](0.3) == pytest.approx(0)
    assert at_edge.hinge_moments(q, beta=0.05)["fin_1"](0.3) != pytest.approx(0)
    assert at_edge.hinge_moments(q, alpha=0.05)["fin_2"](0.3) != pytest.approx(0)


def _still_air_flight(rocket, speed, deflections, rows=5):
    """A flight stub flying ``rocket`` nose first at ``speed`` m/s through
    still sea-level air with no rotation, the fins held at ``deflections``."""
    times = np.linspace(0, 1, rows)
    constant = lambda value: Function(lambda t: value)  # noqa: E731
    canards = next(s for s, _ in rocket.aerodynamic_surfaces if s.name == "Canards")
    history = {
        "Canards": {control: constant(value) for control, value in deflections.items()}
    }
    return SimpleNamespace(
        rocket=rocket,
        time=times,
        stream_velocity_body_frame=np.tile([0.0, 0.0, -speed], (rows, 1)),
        w1=constant(0.0),
        w2=constant(0.0),
        w3=constant(0.0),
        density=constant(1.225),
        speed_of_sound=constant(340.0),
        controls=history,
        _surface_switches={},
        canards=canards,
    )


def test_the_hinge_moments_over_a_flight_match_the_ones_at_its_air(calisto, tmp_path):
    fin = _fin()
    canards = ControlFins(fin, n=4, hinge_position=0.0, name="Canards")
    calisto.add_surfaces(canards, POSITION)
    deflections = dict.fromkeys(canards.controls, 0.1)
    speed = 102.0  # m/s, Mach 0.3 at 340 m/s
    flight = _still_air_flight(calisto, speed, deflections)

    over = canards._hinge_moments_in(flight)
    at = canards.hinge_moments(0.5 * 1.225 * speed**2, deflections=deflections)
    assert set(over) == set(canards.controls)
    for control in canards.controls:
        assert over[control].x_array == pytest.approx(flight.time)
        assert over[control](0.5) == pytest.approx(at[control](0.3))
    assert over["fin_1"](0.5) < 0

    canards.plots.hinge_moments(flight, filename=str(tmp_path / "hinge.png"))
    assert (tmp_path / "hinge.png").exists()

    # A set that is not on the flight's rocket is refused
    with pytest.raises(ValueError, match="not on the rocket"):
        ControlFins(fin, n=4, name="Other")._hinge_moments_in(flight)


def test_a_controller_writes_the_deflections_at_once():
    canards = ControlFins(_fin(), n=4, name="Canards")
    controls = {f"canards.{control}": 0.0 for control in canards.controls}
    view = ControlsView(controls, [canards])
    view.canards.update(canards.deflections(roll=0.05))
    assert controls == pytest.approx(dict.fromkeys(controls, -0.05))
    with pytest.raises(KeyError):
        view.canards.update({"fin_9": 0.0})


def _mixed_fins(cants=(0.0, 0.0)):
    """Two different fins, at 30 and 200 degrees, each with its own cant."""
    small = _fin(angular_position=30, cant_angle=cants[0], name="small")
    big = TrapezoidalFin(
        angular_position=200,
        root_chord=0.12,
        tip_chord=0.03,
        span=0.09,
        rocket_radius=0.0635,
        cant_angle=cants[1],
        name="big",
    )
    return [small, big]


def test_fins_given_one_by_one_are_flown_as_given(calisto, calisto_robust):
    deflections = (0.05, -0.08)
    canards = ControlFins(_mixed_fins(), name="Canards")
    calisto.add_surfaces(canards, POSITION)
    reference = calisto.surfaces_cp_to_cdm[canards]
    fins = _mixed_fins(cants=tuple(math.degrees(d) for d in deflections))
    for fin in fins:
        calisto_robust.add_surfaces(fin, POSITION)

    assert canards.n == 2 and canards.fin is None
    assert canards.angular_positions == (30, 200)
    assert canards.controls == ("fin_1", "fin_2")
    for stream, omega in _random_states():
        speed = abs(stream)
        expected = _forces_of_individual_fins(
            calisto_robust, fins, stream, omega, reference
        )
        got = canards.compute_forces_and_moments(
            stream, speed, speed / 340, 1.1, reference, omega, controls=deflections
        )
        assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_each_fin_given_starts_at_its_own_cant():
    canards = ControlFins(_mixed_fins(cants=(1.0, -2.0)), n=2)
    assert canards.control_defaults == pytest.approx(
        {"fin_1": math.radians(1.0), "fin_2": math.radians(-2.0)}
    )
    assert all(fin.cant_angle == 0 for fin in canards.fins)


def test_fins_given_one_by_one_are_saved_and_loaded():
    canards = ControlFins(_mixed_fins(cants=(1.0, 0.0)), name="Canards")
    loaded = json.loads(json.dumps(canards, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert loaded.fin is None and loaded.n == 2
    assert loaded.angular_positions == canards.angular_positions
    assert loaded.control_defaults == pytest.approx(canards.control_defaults)
    assert [fin.root_chord for fin in loaded.fins] == [0.08, 0.12]
    stream = Vector([1, 2, -100])
    args = (stream, abs(stream), 0.3, 1.2, Vector([0, 0, 1]), (0.1, 0.2, 0.3))
    assert loaded.compute_forces_and_moments(*args) == pytest.approx(
        canards.compute_forces_and_moments(*args)
    )


@pytest.mark.parametrize(
    "fins, n, error",
    [
        ([], None, TypeError),
        ([_fin(), "not a fin"], None, TypeError),
        (_mixed_fins(), 3, ValueError),
    ],
)
def test_wrong_fin_sequences_are_refused(fins, n, error):
    with pytest.raises(error):
        ControlFins(fins, n=n)


def test_effectiveness_is_printed_and_plotted(capsys, tmp_path):
    canards = ControlFins(_fin(), n=4, name="Canards")
    canards.info()
    out = capsys.readouterr().out
    assert "Effectiveness at Mach 0.3" in out
    assert "Roll moment coefficient per rad" in out
    assert "add the set to a rocket" in out
    canards.plots.effectiveness(filename=str(tmp_path / "effectiveness.png"))
    assert (tmp_path / "effectiveness.png").exists()

    _placed(canards, 0.4).info()
    out = capsys.readouterr().out
    assert "Pitch moment coefficient per rad" in out
    assert "Yaw moment coefficient per rad" in out
    canards.plots.effectiveness(filename=str(tmp_path / "placed.png"))
    assert (tmp_path / "placed.png").exists()


def test_changed_defaults_and_limits_are_saved():
    fins = ControlFins(_fin(), n=4, deflection_limits=(-0.2, 0.2))
    fins.control_defaults["fin_1"] = 0.1
    fins.control_limits["fin_2"] = (-0.1, 0.1)
    loaded = ControlFins.from_dict(fins.to_dict())
    assert loaded.control_defaults == fins.control_defaults
    assert loaded.control_limits == fins.control_limits


def test_moving_the_hinge_drops_the_frames_kept():
    fins = ControlFins(_fin(), n=4)
    before = fins._frame(0, 0.1)[2]
    fins.hinge_position = 0.0
    assert fins._frame(0, 0.1)[2] != before


def test_a_deep_copy_of_control_fins_reads_its_own_fins(monkeypatch):
    """The coefficients of a copy are computed from the copy's fins, and match
    the original's while the two are the same."""
    canards = ControlFins(_fin(), n=4, name="Canards")
    copied = deepcopy(canards)
    state = (0.05, 0.02, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, -0.1, 0.0)
    names = ("cN", "cY", "cA", "cm", "cn", "cl")
    expected = [getattr(canards, name)(*state) for name in names]

    readers = []
    compute = ControlFins._coefficients_at

    def spy(self, *args):
        readers.append(self)
        return compute(self, *args)

    monkeypatch.setattr(ControlFins, "_coefficients_at", spy)
    assert [getattr(copied, name)(*state) for name in names] == pytest.approx(expected)
    assert readers and all(reader is copied for reader in readers)
    assert copied.cN_alpha.slice("mach")(0.5) == pytest.approx(
        canards.cN_alpha.slice("mach")(0.5)
    )


def test_moving_the_hinge_updates_the_coefficients():
    """The coefficients read after the hinge is moved are those of a set built
    with the hinge there, also where they had already been read."""
    state = (0.05, 0.02, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, -0.1, 0.0)
    names = ("cN", "cY", "cA", "cm", "cn", "cl")
    canards = ControlFins(_fin(), n=4, name="Canards", hinge_position=0.25)
    before = [getattr(canards, name)(*state) for name in names]
    version = canards._version

    canards.hinge_position = 0.9
    moved = ControlFins(_fin(), n=4, name="Canards", hinge_position=0.9)
    expected = [getattr(moved, name)(*state) for name in names]
    assert expected != pytest.approx(before)
    assert [getattr(canards, name)(*state) for name in names] == pytest.approx(expected)
    assert canards._version > version
