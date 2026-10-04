import pytest

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
    from rocketpy import GenericSurface
    from rocketpy.stochastic import StochasticRocket

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
    assert any(type(s) is GenericSurface for s, _ in created.aerodynamic_surfaces)

    scaled = StochasticRocket(rocket=rocket, power_off_drag_factor=(1.2, 0))
    scaled.add_motor(cesaroni_m1670, position=(-1.255, 0))
    created = scaled.create_object()
    assert created.power_off_drag_7d(*state) == 1.2 * rocket.power_off_drag_7d(*state)


def test_surfaces_with_no_stochastic_version_are_carried_over():
    """Free-form fins, individual fins and generic surfaces have no stochastic
    version; each generated rocket keeps them, at their position."""
    from rocketpy import (
        FreeFormFins,
        GenericSurface,
        StochasticNoseCone,
        StochasticRocket,
        TrapezoidalFin,
    )

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
