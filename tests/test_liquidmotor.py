from rocketpy import Fluid
from rocketpy.motors.Tank import (
    MassBasedTank,
    UllageBasedTank,
    MassFlowRateBasedTank,
    LevelBasedTank,
)
from rocketpy.motors import TankGeometry
from rocketpy.motors.Fluid import Fluid
from rocketpy.Function import Function
from math import isclose
from scipy.optimize import fmin
import numpy as np
import pandas as pd
import os


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
        "Real Tank", real_geometry, lox_masses, gas_masses, lox, n2
    )

    # Generate tank geometry {radius: height, ...}
    example_geometry = TankGeometry({(0, 5): 1})

    # Generate tanks based on simplified tank geometry
    example_tank_lox = MassBasedTank(
        "Example Tank",
        example_geometry,
        example_liquid_masses,
        example_gas_masses,
        lox,
        n2,
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
                print(calculated.getValue(i), expected(i))
                assert isclose(calculated.getValue(i), expected(i), rel_tol=5e-2)
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
        example_calculated = example_tank_lox.mass()

        lox_vals = pd.read_csv(lox_masses, header=None)[1].values

        real_expected = lambda t: lox_vals[t]
        real_calculated = real_tank_lox.mass()

        test(example_calculated, example_expected, 5)
        # test(real_calculated, real_expected, 15.5, real=True)

    def test_net_mfr():
        """Test netMassFlowRate function of MassBasedTank subclass of Tank"""
        example_expected = (
            lambda t: liquid_mass_flow_rate_in
            - liquid_mass_flow_rate_out
            + gas_mass_flow_rate_in
            - gas_mass_flow_rate_out
        )
        example_calculated = example_tank_lox.netMassFlowRate()

        liquid_mfrs = pd.read_csv(example_liquid_masses, header=None)[1].values

        gas_mfrs = pd.read_csv(example_gas_masses, header=None)[1].values

        real_expected = lambda t: (liquid_mfrs[t] + gas_mfrs[t]) / t
        real_calculated = real_tank_lox.netMassFlowRate()

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

        liquid_heights = pd.read_csv(example_liquid_masses, header=None)[1].values

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

    ullage_data = pd.read_csv(os.path.abspath(test_dir + "loxUllage.csv")).to_numpy()
    levelTank = LevelBasedTank(
        "Ullage Tank", tank_geometry, gas=n2, liquid=lox, liquid_height=ullage_data
    )

    mass_data = pd.read_csv(test_dir + "loxMass.csv").to_numpy()
    mass_flow_rate_data = pd.read_csv(test_dir + "loxMFR.csv").to_numpy()

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

    assert np.allclose(levelTank.liquidHeight, ullage_data)

    calculated_mass = levelTank.liquidMass.setDiscrete(
        mass_data[0][0], mass_data[0][-1], len(mass_data[0])
    )
    calculated_mass, mass_data = align_time_series(
        calculated_mass.getSource(), mass_data
    )
    assert np.allclose(calculated_mass, mass_data, rtol=1, atol=2)
    # Function(calculated_mass).plot1D()
    # Function(mass_data).plot1D()

    calculated_mfr = levelTank.netMassFlowRate.setDiscrete(
        mass_flow_rate_data[0][0],
        mass_flow_rate_data[0][-1],
        len(mass_flow_rate_data[0]),
    )
    calculated_mfr, test_mfr = align_time_series(
        calculated_mfr.getSource(), mass_flow_rate_data
    )
    # assert np.allclose(calculated_mfr, test_mfr)
    # Function(calculated_mfr).plot1D()
    # Function(test_mfr).plot1D()


# @gautamsaiy
def test_mfr_tank_basic():
    def test(t, a, tol=1e-4):
        for i in np.arange(0, 10, 1):
            assert isclose(t.getValue(i), a(i), abs_tol=tol)
            # print(t.getValue(i), a(i))

    def test_nmfr():
        nmfr = (
            lambda x: liquid_mass_flow_rate_in
            + gas_mass_flow_rate_in
            - liquid_mass_flow_rate_out
            - gas_mass_flow_rate_out
        )
        test(t.netMassFlowRate, nmfr)

    def test_mass():
        m = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        ) + (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)
        lm = t.mass
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
        tlh = t.liquidHeight
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

        tcom = t.centerOfMass

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
            + 1 / 12 * liquid_mass(x) * (liquid_height(x) - t.structure.bottom) ** 2
            + liquid_mass(x) * (liquid_com(x) - acom(x)) ** 2
        )
        ixy = lambda x: ixy_gas(x) + ixy_liq(x)
        test(t.gasInertiaTensor, ixy_gas, tol=1e-3)
        test(t.liquidInertiaTensor, ixy_liq, tol=1e-3)
        test(t.inertiaTensor, ixy, tol=1e-3)

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
        "Test Tank",
        tank_radius_function,
        initial_liquid_mass,
        initial_gas_mass,
        liquid_mass_flow_rate_in,
        gas_mass_flow_rate_in,
        liquid_mass_flow_rate_out,
        gas_mass_flow_rate_out,
        lox,
        n2,
    )

    test_nmfr()
    test_mass()
    test_liquid_height()
    test_com()
    test_inertia()
