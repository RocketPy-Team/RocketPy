import os
from math import isclose

import numpy as np

from rocketpy import Fluid
from rocketpy.Function import Function
from rocketpy.motors import TankGeometry
from rocketpy.motors.Fluid import Fluid
from rocketpy.motors.Tank import LevelBasedTank, MassBasedTank, MassFlowRateBasedTank


# @PBales1
def test_mass_based_motor():
    lox = Fluid(name="LOx", density=1141.7, quality=1.0)  # Placeholder quality value
    propane = Fluid(
        name="Propane", density=493, quality=1.0
    )  # Placeholder quality value
    n2 = Fluid(
        name="Nitrogen Gas", density=51.75, quality=1.0
    )  # Placeholder quality value; density value may be estimate

    top_endcap = lambda y: np.sqrt(
        0.0775**2 - (y - 0.692300000000001) ** 2
    )  # Hemisphere equation creating top endcap
    bottom_endcap = lambda y: np.sqrt(
        0.0775**2 - (0.0775 - y) ** 2
    )  # Hemisphere equation creating bottom endcap

    # Generate tank geometry {radius: height, ...}
    real_geometry = TankGeometry(
        {
            (0, 0.0559): bottom_endcap,
            (0.0559, 0.7139): lambda y: 0.0744,
            (0.7139, 0.7698): top_endcap,
        }
    )

    # Import liquid mass data
    lox_masses = "./data/berkeley/Test135LoxMass.csv"
    example_liquid_masses = "./data/berkeley/ExampleTankLiquidMassData.csv"

    # Import gas mass data
    gas_masses = "./data/berkeley/Test135GasMass.csv"
    example_gas_masses = "./data/berkeley/ExampleTankGasMassData.csv"

    # Generate tanks based on Berkeley SEB team's real tank geometries
    real_tank_lox = MassBasedTank(
        name="Real Tank",
        geometry=real_geometry,
        flux_time=(0, 10),
        liquid_mass=lox_masses,
        gas_mass=gas_masses,
        liquid=lox,
        gas=n2,
        discretize=None,
    )

    # Generate tank geometry {radius: height, ...}
    example_geometry = TankGeometry({(0, 5): 1})

    # Generate tanks based on simplified tank geometry
    example_tank_lox = MassBasedTank(
        name="Example Tank",
        geometry=example_geometry,
        flux_time=(0, 10),
        liquid_mass=example_liquid_masses,
        gas_mass=example_gas_masses,
        liquid=lox,
        gas=n2,
        discretize=None,
    )

    initial_liquid_mass = 5
    initial_gas_mass = 0.1
    liquid_mass_flow_rate_in = 0.1
    gas_mass_flow_rate_in = 0.01
    liquid_mass_flow_rate_out = 0.2
    gas_mass_flow_rate_out = 0.02

    def test(calculated, expected, t, real=False):
        """Iterate over time range and test that calculated value is close to actual value"""
        j = 0
        for i in np.arange(0, t, 0.1):
            try:
                print(calculated.get_value(i), expected(i))
                assert isclose(calculated.get_value(i), expected(i), rel_tol=5e-2)
            except IndexError:
                break

            if real:
                j += 4
            else:
                j += 1

    def test_mass():
        """Test mass function of MassBasedTank subclass of Tank"""
        example_expected = (
            lambda t: initial_liquid_mass
            + t * (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out)
            + initial_gas_mass
            + t * (gas_mass_flow_rate_in - gas_mass_flow_rate_out)
        )
        example_calculated = example_tank_lox.fluid_mass()

        lox_vals = Function(lox_masses).y_array

        real_expected = lambda t: lox_vals[t]
        real_calculated = real_tank_lox.fluid_mass()

        test(example_calculated, example_expected, 5)
        # test(real_calculated, real_expected, 15.5, real=True)

    def test_net_mfr():
        """Test net_mass_flow_rate function of MassBasedTank subclass of Tank"""
        example_expected = (
            lambda t: liquid_mass_flow_rate_in
            - liquid_mass_flow_rate_out
            + gas_mass_flow_rate_in
            - gas_mass_flow_rate_out
        )
        example_calculated = example_tank_lox.net_mass_flow_rate()

        liquid_mfrs = Function(example_liquid_masses).y_array

        gas_mfrs = Function(example_gas_masses).y_array

        real_expected = lambda t: (liquid_mfrs[t] + gas_mfrs[t]) / t
        real_calculated = real_tank_lox.net_mass_flow_rate()

        test(example_calculated, example_expected, 10)
        # test(real_calculated, real_expected, 15.5, real=True)

    def test_eval_ullage():
        """Test evaluateUllage function of MassBasedTank subclass of Tank"""
        example_expected = (
            lambda t: (
                initial_liquid_mass
                * (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out)
                * t
            )
            / lox.density
            / np.pi
        )
        example_calculated = example_tank_lox.evaluateUllageHeight()

        liquid_heights = Function(example_liquid_masses).y_array

        real_expected = lambda t: liquid_heights[t]
        real_calculated = real_tank_lox.evaluateUllageHeight()

        test(example_calculated, example_expected, 10)
        # test(real_calculated, real_expected, 15.5, real=True)

    # print("Testing MassBasedTank subclass of Tank")
    # test_mass()
    # print("Mass test passed")
    # test_net_mfr()
    # print("Net mass flow rate test passed")
    # test_eval_ullage()
    # print("Evaluate ullage test passed")


# @curtisjhu
def test_ullage_based_motor():
    lox = Fluid(name="LOx", density=1141.7, quality=1.0)
    n2 = Fluid(name="Nitrogen Gas", density=51.75, quality=1.0)

    test_dir = "./data/berkeley/"

    top_endcap = lambda y: np.sqrt(0.0775**2 - (y - 0.692300000000001) ** 2)
    bottom_endcap = lambda y: np.sqrt(0.0775**2 - (0.0775 - y) ** 2)
    tank_geometry = TankGeometry(
        {
            (0, 0.0559): bottom_endcap,
            (0.0559, 0.7139): lambda y: 0.0744,
            (0.7139, 0.7698): top_endcap,
        }
    )

    ullage_data = Function(os.path.abspath(test_dir + "loxUllage.csv")).get_source()
    levelTank = LevelBasedTank(
        name="LevelTank",
        geometry=tank_geometry,
        flux_time=(0, 10),
        gas=n2,
        liquid=lox,
        liquid_height=ullage_data,
        discretize=None,
    )

    mass_data = Function(test_dir + "loxMass.csv").get_source()
    mass_flow_rate_data = Function(test_dir + "loxMFR.csv").get_source()

    def align_time_series(small_source, large_source):
        assert isinstance(small_source, np.ndarray) and isinstance(
            large_source, np.ndarray
        ), "Must be np.ndarrays"
        if small_source.shape[0] > large_source.shape[0]:
            small_source, large_source = large_source, small_source

        result_larger_source = np.ndarray(small_source.shape)
        result_smaller_source = np.ndarray(small_source.shape)
        tolerance = 0.1
        curr_ind = 0
        for val in small_source:
            time = val[0]
            delta_time_vector = abs(time - large_source[:, 0])
            largeIndex = np.argmin(delta_time_vector)
            delta_time = abs(time - large_source[largeIndex][0])

            if delta_time < tolerance:
                result_larger_source[curr_ind] = large_source[largeIndex]
                result_smaller_source[curr_ind] = val
                curr_ind += 1
        return result_larger_source, result_smaller_source

    assert np.allclose(levelTank.liquid_height, ullage_data)

    calculated_mass = levelTank.liquid_mass.set_discrete(
        mass_data[0][0], mass_data[0][-1], len(mass_data[0])
    )
    calculated_mass, mass_data = align_time_series(
        calculated_mass.get_source(), mass_data
    )
    assert np.allclose(calculated_mass, mass_data, rtol=1, atol=2)
    # Function(calculated_mass).plot1D()
    # Function(mass_data).plot1D()

    calculated_mfr = levelTank.net_mass_flow_rate.set_discrete(
        mass_flow_rate_data[0][0],
        mass_flow_rate_data[0][-1],
        len(mass_flow_rate_data[0]),
    )
    calculated_mfr, test_mfr = align_time_series(
        calculated_mfr.get_source(), mass_flow_rate_data
    )
    # assert np.allclose(calculated_mfr, test_mfr)
    # Function(calculated_mfr).plot1D()
    # Function(test_mfr).plot1D()


# @gautamsaiy
def test_mfr_tank_basic():
    def test(t, a, tol=1e-4):
        for i in np.arange(0, 10, 1):
            print(t.get_value(i), a(i))
            assert isclose(t.get_value(i), a(i), abs_tol=tol)

    def test_nmfr():
        nmfr = (
            lambda x: liquid_mass_flow_rate_in
            + gas_mass_flow_rate_in
            - liquid_mass_flow_rate_out
            - gas_mass_flow_rate_out
        )
        test(t.net_mass_flow_rate, nmfr)

    def test_mass():
        m = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        ) + (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)
        lm = t.fluid_mass
        test(lm, m)

    def test_liquid_height():
        alv = (
            lambda x: (
                initial_liquid_mass
                + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
            )
            / lox.density
        )
        alh = lambda x: alv(x) / (np.pi)
        tlh = t.liquid_height
        test(tlh, alh)

    def test_com():
        liquid_mass = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        )  # liquid mass
        liquid_volume = lambda x: liquid_mass(x) / lox.density  # liquid volume
        liquid_height = lambda x: liquid_volume(x) / (np.pi)  # liquid height
        gas_mass = lambda x: (
            initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x
        )  # gas mass
        gas_volume = lambda x: gas_mass(x) / n2.density
        gas_height = lambda x: gas_volume(x) / np.pi + liquid_height(x)

        liquid_com = lambda x: liquid_height(x) / 2  # liquid com
        gas_com = lambda x: (gas_height(x) - liquid_height(x)) / 2 + liquid_height(
            x
        )  # gas com
        acom = lambda x: (liquid_mass(x) * liquid_com(x) + gas_mass(x) * gas_com(x)) / (
            liquid_mass(x) + gas_mass(x)
        )

        tcom = t.center_of_mass
        test(tcom, acom)

    def test_inertia():
        liquid_mass = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        )  # liquid mass
        liquid_volume = lambda x: liquid_mass(x) / lox.density  # liquid volume
        liquid_height = lambda x: liquid_volume(x) / (np.pi)  # liquid height
        gas_mass = lambda x: (
            initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x
        )  # gas mass
        gas_volume = lambda x: gas_mass(x) / n2.density
        gas_height = lambda x: gas_volume(x) / np.pi + liquid_height(x)

        liquid_com = lambda x: liquid_height(x) / 2  # liquid com
        gas_com = lambda x: (gas_height(x) - liquid_height(x)) / 2 + liquid_height(
            x
        )  # gas com
        acom = lambda x: (liquid_mass(x) * liquid_com(x) + gas_mass(x) * gas_com(x)) / (
            liquid_mass(x) + gas_mass(x)
        )

        r = 1
        ixy_gas = (
            lambda x: 1 / 4 * gas_mass(x) * r**2
            + 1 / 12 * gas_mass(x) * (gas_height(x) - liquid_height(x)) ** 2
            + gas_mass(x) * (gas_com(x) - acom(x)) ** 2
        )
        ixy_liq = (
            lambda x: 1 / 4 * liquid_mass(x) * r**2
            + 1 / 12 * liquid_mass(x) * (liquid_height(x) - t.geometry.bottom) ** 2
            + liquid_mass(x) * (liquid_com(x) - acom(x)) ** 2
        )
        ixy = lambda x: ixy_gas(x) + ixy_liq(x)
        test(t.gas_inertia, ixy_gas, tol=1e-3)
        test(t.liquid_inertia, ixy_liq, tol=1e-3)
        test(t.inertia, ixy, tol=1e-3)

    tank_radius_function = TankGeometry({(0, 5): 1})
    lox = Fluid(name="LOx", density=1141, quality=1.0)  # Placeholder quality value
    n2 = Fluid(
        name="Nitrogen Gas", density=51.75, quality=1.0
    )  # Placeholder quality value; density value may be estimate
    initial_liquid_mass = 5
    initial_gas_mass = 0.1
    liquid_mass_flow_rate_in = 0.1
    gas_mass_flow_rate_in = 0.01
    liquid_mass_flow_rate_out = 0.2
    gas_mass_flow_rate_out = 0.02

    t = MassFlowRateBasedTank(
        name="Test Tank",
        geometry=tank_radius_function,
        flux_time=(0, 10),
        initial_liquid_mass=initial_liquid_mass,
        initial_gas_mass=initial_gas_mass,
        liquid_mass_flow_rate_in=Function(0.1).set_discrete(0, 10, 1000),
        gas_mass_flow_rate_in=Function(0.01).set_discrete(0, 10, 1000),
        liquid_mass_flow_rate_out=Function(0.2).set_discrete(0, 10, 1000),
        gas_mass_flow_rate_out=Function(0.02).set_discrete(0, 10, 1000),
        liquid=lox,
        gas=n2,
        discretize=None,
    )

    test_nmfr()
    test_mass()
    test_liquid_height()
    test_com()
    test_inertia()
