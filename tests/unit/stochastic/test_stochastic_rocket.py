import pytest

from rocketpy import (
    Accelerometer,
    Barometer,
    ControlFins,
    Controller,
    FreeFormFins,
    GenericSurface,
    StochasticNoseCone,
    StochasticRocket,
    TrapezoidalFin,
)
from rocketpy.rocket.rocket import Rocket


def test_str(stochastic_calisto):
    assert isinstance(str(stochastic_calisto), str)


def test_create_object(stochastic_calisto):
    """Test create object method of StochasticRocket class.

    This test checks if the create_object method of the StochasticCalisto
    class creates a StochasticCalisto object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_calisto : StochasticCalisto
        StochasticCalisto object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_calisto.create_object()
    assert isinstance(obj, Rocket)


def test_zero_dispersion_keeps_the_full_drag_and_the_stability_phase(
    calisto_robust, cesaroni_m1670
):
    """A drag that depends on more than Mach, the stability phase and a generic
    surface on the rocket survive the stochastic mirror unchanged, and the
    drag factor scales the whole coefficient."""

    rocket = calisto_robust
    table = [[a, m, 0.4 + a**2 + 0.1 * m] for a in (-0.3, 0, 0.3) for m in (0, 1, 2)]
    rocket.power_off_drag = (table, ["alpha", "mach"])
    rocket.stability_phase = "power_on"
    rocket.add_surfaces(
        GenericSurface(rocket.area, 2 * rocket.radius, {"cN": lambda alpha: alpha}), 0.5
    )
    state = (0.3, 0, 1.0, 0, 0, 0, 0)

    stochastic = StochasticRocket(rocket=rocket)
    stochastic.add_motor(cesaroni_m1670, position=(-1.255, 0))
    created = stochastic.create_object()
    assert created.power_off_drag_7d(*state) == rocket.power_off_drag_7d(*state)
    assert created.stability_phase == "power_on"
    # exactly a GenericSurface: every other surface on the rocket is a subclass
    plain = [s for s, _ in created.aerodynamic_surfaces if type(s) is GenericSurface]  # pylint: disable=unidiomatic-typecheck
    assert plain

    scaled = StochasticRocket(rocket=rocket, power_off_drag_factor=(1.2, 0))
    scaled.add_motor(cesaroni_m1670, position=(-1.255, 0))
    created = scaled.create_object()
    assert created.power_off_drag_7d(*state) == 1.2 * rocket.power_off_drag_7d(*state)


def test_surfaces_with_no_stochastic_version_are_carried_over():
    """Free-form fins, individual fins and generic surfaces have no stochastic
    version; each generated rocket keeps them, at their position."""

    rocket = Rocket(
        radius=0.0635,
        mass=14.4,
        inertia=(6.3, 6.3, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    nose = rocket.add_nose(length=0.5, kind="vonKarman", position=1.2)
    rocket.add_surfaces(
        FreeFormFins(
            n=4,
            shape_points=[(0, 0), (0.05, 0.1), (0.12, 0.1), (0.12, 0)],
            rocket_radius=0.0635,
        ),
        -1.0,
    )
    for angle in (0, 120, 240):
        rocket.add_surfaces(
            TrapezoidalFin(
                angular_position=angle,
                root_chord=0.1,
                tip_chord=0.05,
                span=0.08,
                rocket_radius=0.0635,
            ),
            -0.9,
        )
    rocket.add_surfaces(
        GenericSurface(rocket.area, 2 * rocket.radius, {"cN": 0.1}), 0.3
    )
    stochastic = StochasticRocket(rocket)
    stochastic.add_nose(StochasticNoseCone(nose), position=(1.2, 0))

    generated = stochastic.create_object()

    def surfaces(r):
        return sorted(
            (type(s).__name__, round(float(p.z), 6)) for s, p in r.aerodynamic_surfaces
        )

    assert surfaces(generated) == surfaces(rocket)
    assert generated.static_margin(0) == pytest.approx(rocket.static_margin(0))


def test_added_effectors_are_carried_over(
    stochastic_calisto, calisto_robust, push_effector, gimbal
):
    """Thrust and force effectors have no stochastic
    version; each generated rocket keeps them and their channels."""
    calisto_robust.add_effector(gimbal)
    calisto_robust.add_effector(push_effector)
    obj = stochastic_calisto.create_object()
    assert obj.effectors.get_components() == [gimbal, push_effector]
    assert obj.control_names[-2:] == ("gimbal.pitch", "push.on")


def test_air_brakes_are_dispersed_once(stochastic_calisto, calisto_air_brakes_clamp_on):
    """The base rocket's air brakes are a surface; the stochastic rocket
    replaces them with its dispersed copy rather than carrying both over."""
    (brakes,) = calisto_air_brakes_clamp_on.air_brakes
    (controller,) = calisto_air_brakes_clamp_on.controllers
    stochastic_calisto.add_air_brakes(brakes, controller)
    obj = stochastic_calisto.create_object()

    (dispersed,) = obj.air_brakes
    assert dispersed is not brakes
    assert obj._is_at_center_of_dry_mass(
        dispersed
    ) == calisto_air_brakes_clamp_on._is_at_center_of_dry_mass(brakes)
    assert obj.control_names == ("air_brakes.deployment_level",)
    assert len(obj.controllers) == 1 and obj.controllers[0] is not controller


def test_every_controller_is_carried_as_a_copy(stochastic_calisto, calisto_robust):
    """Each sampled rocket flies its own copy of every controller of the
    rocket, in the order they were added, with the surface they turn."""
    fin = TrapezoidalFin(
        angular_position=0,
        root_chord=0.08,
        tip_chord=0.04,
        span=0.06,
        rocket_radius=0.0635,
    )
    calisto_robust.add_surfaces(ControlFins(fin, n=4, name="Canards"), 0.8)
    first = Controller(lambda context: None, sampling_rate=10, name="Roll loop")
    second = Controller(lambda context: None, sampling_rate=50, name="Pitch loop")
    calisto_robust.add_controller(first)
    calisto_robust.add_controller(second)

    one = stochastic_calisto.create_object()
    other = stochastic_calisto.create_object()

    for sampled in (one, other):
        assert [c.name for c in sampled.controllers] == ["Roll loop", "Pitch loop"]
        assert [c.sampling_rate for c in sampled.controllers] == [10, 50]
        assert sampled.control_names[-4:] == calisto_robust.control_names[-4:]
    # Copies: a flight of one sample leaves the base rocket's and the other
    # sample's controllers as they were
    assert not {first, second} & {*one.controllers, *other.controllers}
    assert not set(one.controllers) & set(other.controllers)
    assert calisto_robust.controllers == (first, second)


def test_the_air_brakes_controller_is_carried_once(
    stochastic_calisto, calisto_air_brakes_clamp_on
):
    """The controller of the air brakes is one of the rocket's controllers:
    it is not added a second time for having been given to add_air_brakes."""
    (brakes,) = calisto_air_brakes_clamp_on.air_brakes
    (brakes_controller,) = calisto_air_brakes_clamp_on.controllers
    calisto_air_brakes_clamp_on.add_controller(
        Controller(lambda context: None, sampling_rate=20, name="Logger")
    )
    stochastic_calisto.add_air_brakes(brakes, brakes_controller)

    sampled = stochastic_calisto.create_object()

    assert [c.name for c in sampled.controllers] == [brakes_controller.name, "Logger"]


def test_sensors_are_carried_as_copies(stochastic_calisto, calisto_robust):
    """Each sampled rocket has the rocket's sensors, at their positions, as
    its own copies. A sensor added in two positions stays one sensor."""
    barometer = Barometer(sampling_rate=50, name="Barometer")
    accelerometer = Accelerometer(sampling_rate=100, name="Accelerometer")
    calisto_robust.add_sensor(barometer, 0.5)
    calisto_robust.add_sensor(accelerometer, (0, 0.05, 0.2))
    calisto_robust.add_sensor(accelerometer, (0, -0.05, 0.2))
    barometer.measured_data.append((0.0, 101325.0))  # as after a flight

    one = stochastic_calisto.create_object()
    other = stochastic_calisto.create_object()

    for sampled in (one, other):
        copies = sampled.sensors.get_components()
        assert [type(sensor) for sensor in copies] == [
            Barometer,
            Accelerometer,
            Accelerometer,
        ]
        assert sampled.sensors.get_positions() == calisto_robust.sensors.get_positions()
        assert copies[1] is copies[2]
        assert copies[0] is not barometer and copies[1] is not accelerometer
        assert copies[0].sampling_rate == 50 and copies[0].measured_data == []
        # Each copy knows the sampled rocket alone
        assert copies[0]._attached_rockets == {sampled: 1}
        assert copies[1]._attached_rockets == {sampled: 2}
        assert len(sampled._sensor_events) == 3
    assert one.sensors.get_components()[0] is not other.sensors.get_components()[0]
    # The rocket's own sensors do not keep the sampled rockets
    assert barometer._attached_rockets == {calisto_robust: 1}
    assert accelerometer._attached_rockets == {calisto_robust: 2}
    assert barometer.measured_data == [(0.0, 101325.0)]


def test_air_brakes_not_given_are_carried_as_they_are(
    stochastic_calisto, calisto_air_brakes_clamp_on
):
    """Without add_air_brakes the sampled rocket still has the rocket's air
    brakes, so the controller carried with it has something to set."""
    (brakes,) = calisto_air_brakes_clamp_on.air_brakes

    sampled = stochastic_calisto.create_object()

    assert sampled.air_brakes == [brakes]
    assert sampled.control_names == ("air_brakes.deployment_level",)
    assert len(sampled.controllers) == 1
    assert sampled._is_at_center_of_dry_mass(
        brakes
    ) == calisto_air_brakes_clamp_on._is_at_center_of_dry_mass(brakes)


def test_a_plain_surface_is_accepted_and_wrapped(calisto_robust):
    """add_nose, add_tail and the fins take the surface itself as well as its
    stochastic version."""
    surfaces = {type(s).__name__: s for s, _ in calisto_robust.aerodynamic_surfaces}
    stochastic = StochasticRocket(calisto_robust)

    stochastic.add_nose(surfaces["NoseCone"], position=0.001)
    stochastic.add_trapezoidal_fins(surfaces["TrapezoidalFins"], position=0.001)
    stochastic.add_tail(surfaces["Tail"], position=0.001)

    wrapped = [
        type(s).__name__ for s in stochastic.aerodynamic_surfaces.get_components()
    ]
    assert wrapped == [
        "StochasticNoseCone",
        "StochasticTrapezoidalFins",
        "StochasticTail",
    ]
