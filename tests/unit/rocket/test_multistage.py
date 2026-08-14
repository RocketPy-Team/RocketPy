from rocketpy.rocket.multistage import Stage


def test_stage_dry_mass_matches_wrapped_rocket(calisto):
    stage = Stage(name="stage_1", rocket=calisto)

    assert stage.dry_mass == calisto.dry_mass


def test_stage_burn_out_time_matches_wrapped_rocket_motor(calisto):
    stage = Stage(name="stage_1", rocket=calisto)

    assert stage.burn_out_time == calisto.motor.burn_out_time
