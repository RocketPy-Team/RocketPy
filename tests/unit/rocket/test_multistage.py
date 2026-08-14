import pytest

from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.multistage import Deployable, MultiStageRocket, Stage
from rocketpy.rocket.rocket import Rocket


def _two_stage_vehicle():
    """Booster (bottom, firing) + sustainer (upper, inert) test rig.

    Each stage's motor is placed exactly at that stage's own
    center_of_mass_without_motor, and PointMassMotor has zero internal
    inertia and zero CoM offset from its own attachment point. That
    makes each rocket's overall center_of_mass and I_11/I_22/I_33
    time-invariant and exactly equal to the structure-only values given
    at construction - so expected values can be hand-computed directly
    from the constructor arguments below, without depending on
    flight_rocket's own code path.
    """
    booster_rocket = Rocket(
        radius=0.1,
        mass=10.0,
        inertia=(1.0, 1.0, 0.01),
        power_off_drag=0.5,
        power_on_drag=0.6,
        center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(
        PointMassMotor(
            thrust_source=100, dry_mass=1.0, propellant_initial_mass=2.0,
            burn_time=1.0,
        ),
        position=0.0,
    )
    booster = Stage(name="booster", rocket=booster_rocket)

    sustainer_rocket = Rocket(
        radius=0.08,
        mass=5.0,
        inertia=(0.5, 0.5, 0.005),
        power_off_drag=0.3,
        power_on_drag=0.4,
        center_of_mass_without_motor=2.0,
    )
    sustainer_rocket.add_motor(
        PointMassMotor(
            thrust_source=50, dry_mass=0.5, propellant_initial_mass=1.0,
            burn_time=1.0,
        ),
        position=2.0,
    )
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket)

    return booster, sustainer


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


def test_two_stage_flight_rocket_composes_bottom_stage_plus_inert_upper_stage():
    booster, sustainer = _two_stage_vehicle()
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    flight_rocket = vehicle.flight_rocket(active_stages=(booster, sustainer))

    # Hand-computed: booster contributes its structure-only mass/CoM (its
    # motor becomes flight_rocket's own motor); the inert sustainer
    # contributes its FULL mass (structure + dry motor + full propellant,
    # since its own motor clock hasn't started) at its own, time-invariant
    # center of mass. See _two_stage_vehicle for why these are exact.
    sustainer_full_mass = 5.0 + 0.5 + 1.0  # structure + motor dry + propellant
    expected_mass = 10.0 + sustainer_full_mass
    expected_center_of_mass = (10.0 * 0.0 + sustainer_full_mass * 2.0) / expected_mass

    booster_distance = expected_center_of_mass - 0.0
    sustainer_distance = expected_center_of_mass - 2.0
    expected_inertia_11 = (1.0 + 10.0 * booster_distance**2) + (
        0.5 + sustainer_full_mass * sustainer_distance**2
    )
    expected_inertia_33 = 0.01 + 0.005  # I_33 unaffected by axial offset

    assert flight_rocket.mass == pytest.approx(expected_mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        expected_center_of_mass
    )
    assert flight_rocket.I_11_without_motor == pytest.approx(expected_inertia_11)
    assert flight_rocket.I_33_without_motor == pytest.approx(expected_inertia_33)
    assert flight_rocket.motor is booster.rocket.motor


def test_flight_rocket_after_separation_uses_only_the_remaining_stage():
    booster, sustainer = _two_stage_vehicle()
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    flight_rocket = vehicle.flight_rocket(active_stages=(sustainer,))

    assert flight_rocket.mass == pytest.approx(sustainer.rocket.mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        sustainer.rocket.center_of_mass_without_motor
    )
    assert flight_rocket.motor is sustainer.rocket.motor


def test_flight_rocket_combines_surfaces_of_every_active_stage(calisto_nose_cone):
    booster, sustainer = _two_stage_vehicle()
    sustainer.rocket.add_surfaces(calisto_nose_cone, 0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    two_stage_rocket = vehicle.flight_rocket(active_stages=(booster, sustainer))
    sustainer_alone_rocket = vehicle.flight_rocket(active_stages=(sustainer,))

    assert len(two_stage_rocket.aerodynamic_surfaces) == 1
    assert len(sustainer_alone_rocket.aerodynamic_surfaces) == 1
