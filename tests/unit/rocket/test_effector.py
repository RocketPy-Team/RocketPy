"""Tests for control effectors: the Effector / GenericEffector force-moment model,
its control state, serialization, and Rocket.add_effector wiring."""

import numpy as np
import pytest

from rocketpy import GenericEffector
from rocketpy.mathutils.vector_matrix import Vector


def test_generic_effector_force_and_moment_mapping():
    """The effector returns its body-frame force and the moment about the CDM
    (its own moment plus ``force_arm x force``)."""
    effector = GenericEffector(
        force=lambda **k: (0.0, k["side"], 0.0),
        moment=lambda **k: (0.0, 0.0, k["roll"]),
        controls=("side", "roll"),
    )
    effector.set_control("side", 10.0)
    effector.set_control("roll", 4.0)

    arm = Vector([0.0, 0.0, -2.0])  # application point 2 m below the CDM
    force, moment = effector.evaluate(
        arm, 0.0, Vector([0, 0, 0]), Vector([0, 0, 0]), 0.3, None
    )
    assert tuple(force) == (0.0, 10.0, 0.0)
    # own moment (0, 0, 4) + arm x force = (0,0,-2) x (0,10,0) = (20, 0, 0)
    assert tuple(moment) == pytest.approx((20.0, 0.0, 4.0))


def test_generic_effector_defaults_to_zero():
    """A force/moment left as None contributes nothing."""
    effector = GenericEffector(controls=("x",))
    force, moment = effector.evaluate(
        Vector([0, 0, 1]), 0.0, Vector([0, 0, 0]), Vector([0, 0, 0]), 0.0, None
    )
    assert tuple(force) == (0.0, 0.0, 0.0)
    assert tuple(moment) == (0.0, 0.0, 0.0)


def test_effector_control_state_set_get_reset():
    """Control state round-trips through set/get and resets to its initial value."""
    effector = GenericEffector(controls=("torque",))
    assert effector.get_control("torque") == 0.0
    effector.set_control("torque", 3.0)
    assert effector.get_control("torque") == 3.0
    effector._reset()
    assert effector.get_control("torque") == 0.0
    with pytest.raises(KeyError):
        effector.set_control("unknown", 1.0)


def test_generic_effector_serialization_roundtrip():
    """to_dict/from_dict preserves the force/moment callables, position, controls
    and name, and evaluates identically."""
    effector = GenericEffector(
        force=lambda **k: (0.0, k["cmd"], 0.0),
        position=-1.2,
        controls=("cmd",),
        name="Side RCS",
    )
    restored = GenericEffector.from_dict(effector.to_dict())
    assert restored.name == "Side RCS"
    assert restored.position == -1.2
    assert restored.control_variables == ["cmd"]

    for eff in (effector, restored):
        eff.set_control("cmd", 5.0)
    arm = Vector([0, 0, -1.0])
    f0, m0 = effector.evaluate(arm, 0, Vector([0, 0, 0]), Vector([0, 0, 0]), 0.2, None)
    f1, m1 = restored.evaluate(arm, 0, Vector([0, 0, 0]), Vector([0, 0, 0]), 0.2, None)
    assert tuple(f1) == pytest.approx(tuple(f0))
    assert tuple(m1) == pytest.approx(tuple(m0))


def test_add_effector_registers_and_wires_controller(calisto_motorless):
    """add_effector stores the effector, computes its moment arm, keeps it out of
    the aerodynamic surfaces, and (with a controller_function) registers a
    Controller driving it."""
    rocket = calisto_motorless

    def law(**kwargs):
        kwargs["rcs"].set_control("torque", 1.0)

    effector = GenericEffector(
        moment=lambda **k: (0.0, 0.0, k["torque"]), controls=("torque",), name="RCS"
    )
    controller = rocket.add_effector(
        effector,
        position=0.3,
        controller_function=law,
        sampling_rate=50,
        controlled_objects_name="rcs",
    )

    assert effector in rocket.effectors
    assert effector in rocket.effectors_cp_to_cdm
    # arm z = (position - cdm) * csys
    expected_z = (0.3 - rocket.center_of_dry_mass_position) * rocket._csys
    assert rocket.effectors_cp_to_cdm[effector].z == pytest.approx(expected_z)
    # An effector is not an aerodynamic surface (no CoP / stability impact).
    assert effector not in [s for s, _ in rocket.aerodynamic_surfaces]
    # A controller was created and drives the effector.
    assert controller in rocket._controllers
    assert controller.controlled_objects is effector


def test_add_effector_open_loop_returns_effector(calisto_motorless):
    """Without a controller_function, add_effector registers a fixed effector and
    creates no controller."""
    rocket = calisto_motorless
    effector = GenericEffector(controls=("cmd",), name="Fixed")
    returned = rocket.add_effector(effector, position=0.0)
    assert returned is effector
    assert rocket._controllers == []


@pytest.mark.slow
def test_roll_damper_effector_reduces_roll(cesaroni_m1670, example_plain_env):
    """End-to-end: a roll-rate-damper effector measurably reduces the peak roll
    rate versus the same (canted-fin) rocket without it."""
    from rocketpy import Flight, Rocket

    env = example_plain_env
    env.set_atmospheric_model(type="standard_atmosphere")

    def build(with_damper):
        rocket = Rocket(
            radius=127 / 2000,
            mass=14.426,
            inertia=(6.321, 6.321, 0.034),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=0,
            coordinate_system_orientation="tail_to_nose",
        )
        rocket.add_motor(cesaroni_m1670, position=-1.373)
        rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)
        rocket.add_trapezoidal_fins(
            n=4, root_chord=0.12, tip_chord=0.06, span=0.11, position=-1.04956
        )
        # Canted tabs spin the rocket up (the disturbance to be damped).
        rocket.add_trapezoidal_fins(
            n=3,
            root_chord=0.05,
            tip_chord=0.03,
            span=0.04,
            position=-0.95,
            cant_angle=2.0,
            name="Roll tabs",
        )
        if with_damper:

            def damper(**kwargs):
                kwargs["rcs"].set_control("torque", -0.04 * kwargs["state"][12])

            rocket.add_effector(
                GenericEffector(
                    moment=lambda **k: (0.0, 0.0, k["torque"]),
                    controls=("torque",),
                    name="Roll RCS",
                ),
                controller_function=damper,
                sampling_rate=50,
                controlled_objects_name="rcs",
            )
        return rocket

    def peak_roll(with_damper):
        flight = Flight(
            rocket=build(with_damper),
            environment=env,
            rail_length=5.2,
            inclination=85,
            heading=0,
            terminate_on_apogee=True,
        )
        times = np.linspace(0, flight.apogee_time, 300)
        return max(abs(flight.w3.get_value_opt(t)) for t in times)

    assert peak_roll(True) < peak_roll(False)
