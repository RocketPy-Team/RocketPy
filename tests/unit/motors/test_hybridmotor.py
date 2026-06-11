import numpy as np
import numpy.testing as npt


def thrust_function(t):
    return 2000 - 100 * t


BURN_TIME = 10
CENTER_OF_DRY_MASS = 0
DRY_INERTIA = (4, 4, 0.1)
DRY_MASS = 8
GRAIN_DENSITY = 1700
GRAIN_NUMBER = 4
GRAIN_INITIAL_HEIGHT = 0.1
GRAIN_SEPARATION = 0
GRAIN_INITIAL_INNER_RADIUS = 0.04
GRAIN_OUTER_RADIUS = 0.1
NOZZLE_POSITION = -0.4
NOZZLE_RADIUS = 0.07
GRAINS_CENTER_OF_MASS_POSITION = -0.1
OXIDIZER_TANK_POSITION = 0.3


def test_hybrid_motor_basic_parameters(hybrid_motor):
    """Tests the HybridMotor class construction parameters.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    """
    assert hybrid_motor.burn_time == (0, BURN_TIME)
    assert hybrid_motor.dry_mass == DRY_MASS
    assert (
        hybrid_motor.dry_I_11,
        hybrid_motor.dry_I_22,
        hybrid_motor.dry_I_33,
    ) == DRY_INERTIA
    assert hybrid_motor.center_of_dry_mass_position == CENTER_OF_DRY_MASS
    assert hybrid_motor.nozzle_position == NOZZLE_POSITION
    assert hybrid_motor.nozzle_radius == NOZZLE_RADIUS
    assert hybrid_motor.solid.grain_number == GRAIN_NUMBER
    assert hybrid_motor.solid.grain_density == GRAIN_DENSITY
    assert hybrid_motor.solid.grain_initial_height == GRAIN_INITIAL_HEIGHT
    assert hybrid_motor.solid.grain_separation == GRAIN_SEPARATION
    assert hybrid_motor.solid.grain_initial_inner_radius == GRAIN_INITIAL_INNER_RADIUS
    assert hybrid_motor.solid.grain_outer_radius == GRAIN_OUTER_RADIUS
    assert (
        hybrid_motor.solid.grains_center_of_mass_position
        == GRAINS_CENTER_OF_MASS_POSITION
    )
    assert hybrid_motor.liquid.positioned_tanks[0]["position"] == 0.3


def test_hybrid_motor_thrust_parameters(hybrid_motor, oxidizer_tank):
    """Tests the HybridMotor class thrust parameters.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    spherical_oxidizer_tank : rocketpy.SphericalTank
        The SphericalTank object to be used in the tests.
    """
    t_array = np.linspace(0, 10, 50)
    expected_thrust_values = thrust_function(t_array)

    expected_total_impulse = 15000

    initial_grain_mass = (
        GRAIN_DENSITY
        * np.pi
        * (GRAIN_OUTER_RADIUS**2 - GRAIN_INITIAL_INNER_RADIUS**2)
        * GRAIN_INITIAL_HEIGHT
        * GRAIN_NUMBER
    )

    initial_oxidizer_mass = oxidizer_tank.fluid_mass(0)
    initial_mass = initial_grain_mass + initial_oxidizer_mass

    expected_exhaust_velocity = expected_total_impulse / initial_mass

    npt.assert_allclose(hybrid_motor.thrust(t_array), expected_thrust_values)
    npt.assert_allclose(hybrid_motor.total_impulse, expected_total_impulse)
    npt.assert_allclose(hybrid_motor.exhaust_velocity(0), expected_exhaust_velocity)


def test_hybrid_motor_center_of_mass(hybrid_motor, oxidizer_tank):
    """Tests the HybridMotor class center of mass.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    spherical_oxidizer_tank : rocketpy.SphericalTank
        The SphericalTank object to be used in the tests.
    """
    oxidizer_mass = oxidizer_tank.fluid_mass
    grain_mass = hybrid_motor.solid.propellant_mass

    propellant_balance = grain_mass * GRAINS_CENTER_OF_MASS_POSITION + oxidizer_mass * (
        OXIDIZER_TANK_POSITION + oxidizer_tank.center_of_mass
    )

    balance = propellant_balance + DRY_MASS * CENTER_OF_DRY_MASS
    propellant_center_of_mass = propellant_balance / (grain_mass + oxidizer_mass)
    center_of_mass = balance / (grain_mass + oxidizer_mass + DRY_MASS)

    t = np.linspace(0, BURN_TIME, 100)

    npt.assert_allclose(
        hybrid_motor.center_of_propellant_mass(t),
        propellant_center_of_mass(t),
    )

    npt.assert_allclose(
        hybrid_motor.center_of_mass(t),
        center_of_mass(t),
    )


def test_hybrid_motor_inertia(hybrid_motor, oxidizer_tank):
    """Tests the HybridMotor class inertia.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    spherical_oxidizer_tank : rocketpy.SphericalTank
        The SphericalTank object to be used in the tests.
    """
    oxidizer_mass = oxidizer_tank.fluid_mass
    oxidizer_inertia = oxidizer_tank.inertia

    grain_mass = hybrid_motor.solid.propellant_mass
    grain_inertia = hybrid_motor.solid.propellant_I_11

    propellant_mass = oxidizer_mass + grain_mass

    # Validate parallel axis theorem translation
    grain_inertia += (
        grain_mass
        * (GRAINS_CENTER_OF_MASS_POSITION - hybrid_motor.center_of_propellant_mass) ** 2
    )

    oxidizer_inertia += (
        oxidizer_mass
        * (
            OXIDIZER_TANK_POSITION
            + oxidizer_tank.center_of_mass
            - hybrid_motor.center_of_propellant_mass
        )
        ** 2
    )

    propellant_inertia = grain_inertia + oxidizer_inertia

    # Adding dry mass contributions
    inertia = (
        propellant_inertia
        + propellant_mass
        * (hybrid_motor.center_of_propellant_mass - hybrid_motor.center_of_mass) ** 2
        + DRY_INERTIA[0]
        + DRY_MASS * (-hybrid_motor.center_of_mass + CENTER_OF_DRY_MASS) ** 2
    )

    t = np.linspace(0, BURN_TIME, 100)

    actual_val = hybrid_motor.propellant_I_11(t)
    desired_val = propellant_inertia(t)

    npt.assert_allclose(actual_val, desired_val, rtol=1e-6)
    npt.assert_allclose(hybrid_motor.I_11(t), inertia(t), rtol=1e-6)
    # Assert cylindrical symmetry
    npt.assert_allclose(
        hybrid_motor.propellant_I_22(t), propellant_inertia(t), rtol=1e-6
    )
