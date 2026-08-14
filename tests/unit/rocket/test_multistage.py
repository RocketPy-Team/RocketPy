import pytest

from rocketpy.rocket.multistage import Deployable, Stage


def test_stage_dry_mass_matches_wrapped_rocket(calisto):
    stage = Stage(name="stage_1", rocket=calisto)

    assert stage.dry_mass == calisto.dry_mass


def test_stage_burn_out_time_matches_wrapped_rocket_motor(calisto):
    stage = Stage(name="stage_1", rocket=calisto)

    assert stage.burn_out_time == calisto.motor.burn_out_time


def test_deployable_stores_constructor_arguments():
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
        radius=0.05,
    )

    assert deployable.name == "payload"
    assert deployable.mass == 4.5
    assert deployable.inertia == (0.1, 0.1, 0.001)
    assert deployable.position == 1.10
    assert deployable.radius == 0.05
    assert deployable.free_rocket is None
    assert deployable.ejection is None
    assert not deployable.surfaces


def test_add_surface_raises_when_free_rocket_already_set(calisto_nose_cone):
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
        radius=0.05,
        free_rocket=object(),
    )

    with pytest.raises(ValueError):
        deployable.add_surface(calisto_nose_cone, position=0.5)


def test_add_surface_raises_when_radius_not_set(calisto_nose_cone):
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
    )

    with pytest.raises(ValueError):
        deployable.add_surface(calisto_nose_cone, position=0.5)


def test_add_surface_appends_surface_and_position(calisto_nose_cone):
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
        radius=0.05,
    )

    deployable.add_surface(calisto_nose_cone, position=0.5)

    assert deployable.surfaces == [(calisto_nose_cone, 0.5)]
