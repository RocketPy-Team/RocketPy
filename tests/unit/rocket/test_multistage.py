import pytest

from rocketpy.rocket.multistage import Deployable, MultiStageRocket, Stage


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


def test_flight_rocket_mass_with_no_deployables_matches_the_stage(calisto):
    stage = Stage(name="stage_1", rocket=calisto)
    vehicle = MultiStageRocket(stages=[stage])

    flight_rocket = vehicle.flight_rocket(active_stages=(stage,))

    assert flight_rocket.mass == pytest.approx(calisto.mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        calisto.center_of_mass_without_motor
    )


def test_flight_rocket_composes_mass_and_center_of_mass_with_a_deployable(calisto):
    stage = Stage(name="stage_1", rocket=calisto)
    vehicle = MultiStageRocket(stages=[stage])
    deployable = vehicle.add_deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
    )

    flight_rocket = vehicle.flight_rocket(
        active_stages=(stage,), carried_deployables=(deployable,)
    )

    # Hand-computed weighted average of calisto's own mass/CoM and the
    # deployable's - independent of flight_rocket's own code path.
    expected_mass = calisto.mass + 4.5
    expected_center_of_mass = (
        calisto.mass * calisto.center_of_mass_without_motor + 4.5 * 1.10
    ) / expected_mass

    assert flight_rocket.mass == pytest.approx(expected_mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        expected_center_of_mass
    )
