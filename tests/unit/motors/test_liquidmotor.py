import numpy as np
import numpy.testing as npt
import scipy.integrate

BURN_TIME = (8, 20)
DRY_MASS = 10
DRY_INERTIA = (5, 5, 0.2)
CENTER_OF_DRY_MASS = 0
NOZZLE_POSITION = -1.364
NOZZLE_RADIUS = 0.069 / 2
PRESSURANT_TANK_POSITION = 2.007
FUEL_TANK_POSITION = -1.048
OXIDIZER_TANK_POSITION = 0.711


def test_liquid_motor_basic_parameters(liquid_motor):
    """Tests the LiquidMotor class construction parameters.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.burn_time == BURN_TIME
    assert liquid_motor.dry_mass == DRY_MASS
    assert (
        liquid_motor.dry_I_11,
        liquid_motor.dry_I_22,
        liquid_motor.dry_I_33,
    ) == DRY_INERTIA
    assert liquid_motor.center_of_dry_mass_position == CENTER_OF_DRY_MASS
    assert liquid_motor.nozzle_position == NOZZLE_POSITION
    assert liquid_motor.nozzle_radius == NOZZLE_RADIUS
    assert liquid_motor.positioned_tanks[0]["position"] == PRESSURANT_TANK_POSITION
    assert liquid_motor.positioned_tanks[1]["position"] == FUEL_TANK_POSITION
    assert liquid_motor.positioned_tanks[2]["position"] == OXIDIZER_TANK_POSITION


def test_liquid_motor_thrust_parameters(
    liquid_motor, pressurant_tank, fuel_tank, oxidizer_tank
):
    """Tests the LiquidMotor class thrust parameters.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_tank : rocketpy.Tank
        The expected pressurant tank.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    """
    expected_thrust = np.loadtxt(
        "data/rockets/berkeley/test124_Thrust_Curve.csv",
        delimiter=",",
    )

    time = expected_thrust[:, 0]
    expected_thrust_values = expected_thrust[:, 1]

    expected_mass_flow = (
        pressurant_tank.net_mass_flow_rate(time)
        + fuel_tank.net_mass_flow_rate(time)
        + oxidizer_tank.net_mass_flow_rate(time)
    )

    expected_total_impulse = scipy.integrate.trapezoid(
        expected_thrust_values,
        time,
    )

    npt.assert_allclose(
        liquid_motor.thrust(time),
        expected_thrust_values,
    )

    npt.assert_allclose(
        liquid_motor.mass_flow_rate(time),
        expected_mass_flow,
    )

    npt.assert_allclose(
        liquid_motor.total_impulse,
        expected_total_impulse,
    )


def test_liquid_motor_mass_volume(
    liquid_motor,
    pressurant_fluid,
    fuel_tank,
    fuel_fluid,
    fuel_pressurant,
    oxidizer_tank,
    oxidizer_fluid,
    oxidizer_pressurant,
):
    """Tests the LiquidMotor class tanks flow and method values.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_fluid : rocketpy.Fluid
        The expected pressurant fluid.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    fuel_fluid : rocketpy.Fluid
        The expected fuel fluid.
    fuel_pressurant : rocketpy.Fluid
        The expected fuel pressurant.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    oxidizer_fluid : rocketpy.Fluid
        The expected oxidizer fluid.
    oxidizer_pressurant : rocketpy.Fluid
        The expected oxidizer pressurant.
    """
    test_pressurant_tank = liquid_motor.positioned_tanks[0]["tank"]
    test_fuel_tank = liquid_motor.positioned_tanks[1]["tank"]
    test_oxidizer_tank = liquid_motor.positioned_tanks[2]["tank"]

    time = np.linspace(*BURN_TIME, 100)

    pressurant_mass_data = np.loadtxt(
        "data/rockets/berkeley/pressurantMassFiltered.csv",
        delimiter=",",
    )

    fuel_volume_data = np.loadtxt(
        "data/rockets/berkeley/test124_Propane_Volume.csv",
        delimiter=",",
    )

    oxidizer_volume_data = np.loadtxt(
        "data/rockets/berkeley/test124_Lox_Volume.csv",
        delimiter=",",
    )

    expected_pressurant_mass = np.interp(
        time,
        pressurant_mass_data[:, 0],
        pressurant_mass_data[:, 1],
    )

    expected_pressurant_volume = expected_pressurant_mass / pressurant_fluid.density

    expected_fuel_volume = (
        np.interp(
            time,
            fuel_volume_data[:, 0],
            fuel_volume_data[:, 1],
        )
        * 1e-3
    )

    expected_fuel_mass = (
        expected_fuel_volume * fuel_fluid.density
        + (-expected_fuel_volume + fuel_tank.geometry.total_volume)
        * fuel_pressurant.density
    )

    expected_oxidizer_volume = (
        np.interp(
            time,
            oxidizer_volume_data[:, 0],
            oxidizer_volume_data[:, 1],
        )
        * 1e-3
    )

    expected_oxidizer_mass = (
        expected_oxidizer_volume * oxidizer_fluid.density
        + (-expected_oxidizer_volume + oxidizer_tank.geometry.total_volume)
        * oxidizer_pressurant.density
    )

    npt.assert_allclose(
        test_pressurant_tank.fluid_mass(time),
        expected_pressurant_mass,
        rtol=1e-2,
    )

    npt.assert_allclose(
        test_fuel_tank.fluid_mass(time),
        expected_fuel_mass,
        rtol=1e-2,
    )

    npt.assert_allclose(
        test_oxidizer_tank.fluid_mass(time),
        expected_oxidizer_mass,
        rtol=1e-2,
    )

    npt.assert_allclose(
        test_pressurant_tank.gas_volume(time),
        expected_pressurant_volume,
        rtol=1e-2,
    )

    npt.assert_allclose(
        test_fuel_tank.liquid_volume(time),
        expected_fuel_volume,
        rtol=1e-2,
    )

    npt.assert_allclose(
        test_oxidizer_tank.liquid_volume(time),
        expected_oxidizer_volume,
        rtol=1e-2,
    )


def test_liquid_motor_center_of_mass(
    liquid_motor, pressurant_tank, fuel_tank, oxidizer_tank
):
    """Tests the LiquidMotor class center of mass.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_tank : rocketpy.Tank
        The expected pressurant tank.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    """
    pressurant_mass = pressurant_tank.fluid_mass
    fuel_mass = fuel_tank.fluid_mass
    oxidizer_mass = oxidizer_tank.fluid_mass

    propellant_mass = pressurant_mass + fuel_mass + oxidizer_mass

    propellant_balance = (
        pressurant_mass * (pressurant_tank.center_of_mass + PRESSURANT_TANK_POSITION)
        + fuel_mass * (fuel_tank.center_of_mass + FUEL_TANK_POSITION)
        + oxidizer_mass * (oxidizer_tank.center_of_mass + OXIDIZER_TANK_POSITION)
    )

    balance = propellant_balance + DRY_MASS * CENTER_OF_DRY_MASS

    propellant_center_of_mass = propellant_balance / propellant_mass

    center_of_mass = balance / (propellant_mass + DRY_MASS)

    time = np.linspace(*BURN_TIME, 100)

    npt.assert_allclose(
        liquid_motor.center_of_propellant_mass(time),
        propellant_center_of_mass(time),
    )

    npt.assert_allclose(
        liquid_motor.center_of_mass(time),
        center_of_mass(time),
    )


def test_liquid_motor_inertia(liquid_motor, pressurant_tank, fuel_tank, oxidizer_tank):
    """Tests the LiquidMotor class inertia.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_tank : rocketpy.Tank
        The expected pressurant tank.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    """
    pressurant_inertia = pressurant_tank.inertia
    fuel_inertia = fuel_tank.inertia
    oxidizer_inertia = oxidizer_tank.inertia

    propellant_mass = (
        pressurant_tank.fluid_mass + fuel_tank.fluid_mass + oxidizer_tank.fluid_mass
    )

    # Validate parallel axis theorem translation
    pressurant_inertia += (
        pressurant_tank.fluid_mass
        * (
            pressurant_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + PRESSURANT_TANK_POSITION
        )
        ** 2
    )

    fuel_inertia += (
        fuel_tank.fluid_mass
        * (
            fuel_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + FUEL_TANK_POSITION
        )
        ** 2
    )

    oxidizer_inertia += (
        oxidizer_tank.fluid_mass
        * (
            oxidizer_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + OXIDIZER_TANK_POSITION
        )
        ** 2
    )

    propellant_inertia = pressurant_inertia + fuel_inertia + oxidizer_inertia

    # Adding dry mass contributions
    inertia = (
        propellant_inertia
        + propellant_mass
        * (liquid_motor.center_of_propellant_mass - liquid_motor.center_of_mass) ** 2
        + DRY_INERTIA[0]
        + DRY_MASS * (-liquid_motor.center_of_mass + CENTER_OF_DRY_MASS) ** 2
    )

    time = np.linspace(*BURN_TIME, 100)

    npt.assert_allclose(
        liquid_motor.propellant_I_11(time),
        propellant_inertia(time),
    )

    npt.assert_allclose(
        liquid_motor.I_11(time),
        inertia(time),
    )

    # Assert cylindrical symmetry
    npt.assert_allclose(
        liquid_motor.propellant_I_22(time),
        propellant_inertia(time),
    )


def test_propellant_I_33_cylindrical_closed_form():
    """Tests that the propellant roll inertia of a draining cylindrical tank
    matches the closed form I_33 = 1/2 m r² for both the liquid and the gas
    columns, at initial and half-drain conditions.

    A hard-coded ``return 0`` used to make this property silently ignore the
    propellant contribution to roll dynamics (see issue #1191).
    """
    import math

    import pytest
    from rocketpy import Fluid
    from rocketpy.motors import LiquidMotor
    from rocketpy.motors.tank import MassFlowRateBasedTank
    from rocketpy.motors.tank_geometry import CylindricalTank

    r, h, rho = 0.7, 2.0, 1141.0
    geo = CylindricalTank(radius_function=r, height=h, spherical_caps=False)
    m0 = rho * math.pi * r**2 * h * 0.93  # 93% fill
    m_gas = 0.4
    burn = 120.0
    tank = MassFlowRateBasedTank(
        name="lox",
        geometry=geo,
        flux_time=(0, burn),
        liquid=Fluid(name="LOX", density=rho),
        gas=Fluid(name="He", density=5.0),
        initial_liquid_mass=m0,
        initial_gas_mass=m_gas,
        liquid_mass_flow_rate_in=0,
        gas_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=lambda t: m0 / (2 * burn),  # half drain
        gas_mass_flow_rate_out=0,
        discretize=2000,
    )
    motor = LiquidMotor(
        thrust_source=lambda t: 1e5,
        dry_mass=1000.0,
        dry_inertia=(1e5, 1e5, 2e4),
        nozzle_radius=0.2,
        center_of_dry_mass_position=1.0,
        nozzle_position=0,
        burn_time=(0, burn),
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    motor.add_tank(tank, position=1.5)

    # Cylinder: I_33 = 1/2 m r^2 exactly, for liquid and gas columns alike
    assert motor.propellant_I_33(0) == pytest.approx(
        0.5 * (m0 + m_gas) * r**2, rel=1e-3
    )
    assert motor.propellant_I_33(burn) == pytest.approx(
        0.5 * (m0 / 2 + m_gas) * r**2, rel=1e-3
    )

    # Monotonically non-increasing while draining
    time = np.linspace(0, burn, 21)
    values = motor.propellant_I_33(time)
    assert all(a >= b for a, b in zip(values[:-1], values[1:]))


def test_propellant_I_33_empty_tank_is_zero():
    """Tests that an empty tank contributes no roll inertia and that public
    density properties are exposed on Tank.
    """
    import math

    import pytest
    from rocketpy import Fluid
    from rocketpy.motors.tank import MassFlowRateBasedTank
    from rocketpy.motors.tank_geometry import CylindricalTank

    r, h = 0.5, 1.0
    geo = CylindricalTank(radius_function=r, height=h, spherical_caps=False)
    burn = 10.0
    m0 = 1000.0 * math.pi * r**2 * h  # start full, drain everything
    tank = MassFlowRateBasedTank(
        name="fuel",
        geometry=geo,
        flux_time=(0, burn),
        liquid=Fluid(name="N2O4", density=1000.0),
        gas=Fluid(name="N2", density=10.0),
        initial_liquid_mass=m0,
        initial_gas_mass=1.0,
        liquid_mass_flow_rate_in=0,
        gas_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=lambda t: m0 / burn,
        gas_mass_flow_rate_out=0,
        discretize=2000,
    )

    # Public density accessors on the Tank (constant-density fluids
    # still resolve to a constant Function of time)
    assert tank.liquid_density(0) == pytest.approx(1000.0)
    assert tank.gas_density(0) == pytest.approx(10.0)

    # At burnout the liquid column is empty; only the gas remains
    assert tank.liquid_height(burn) == pytest.approx(geo.bottom, abs=1e-4)
