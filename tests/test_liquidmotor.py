from rocketpy import Fluid
from rocketpy.motors.LiquidMotor import Tank, LiquidMotor, MassBasedTank, UllageBasedTank, MassFlowRateBasedTank
from rocketpy.motors.Fluid import Fluid
from rocketpy.Function import Function
from math import isclose
from scipy.optimize import fmin
import numpy as np
import pandas as pd
import os


# @PBales1
def test_mass_based_motor():
    lox = Fluid(name = "LOx", density = 1141.7, quality = 1.0) #Placeholder quality value
    propane = Fluid(name = "Propane", density = 493, quality = 1.0) #Placeholder quality value
    n2 = Fluid(name = "Nitrogen Gas", density = 51.75, quality = 1.0) #Placeholder quality value; density value may be estimate
    
    top_endcap = lambda y: np.sqrt(0.0775 ** 2 - (y - 0.692300000000001) ** 2) #Hemisphere equation creating top endcap
    bottom_endcap = lambda y: np.sqrt(0.0775 ** 2 - (0.0775 - y) **2) #Hemisphere equation creating bottom endcap

    #Generate tank geometry {radius: height, ...}
    real_geometry = {(0, 0.0559): bottom_endcap, (.0559, 0.7139): lambda y: 0.0744, (0.7139, 0.7698): top_endcap} 

    #Import liquid mass data
    lox_masses = "data/berkeley/Test135LoxMass.csv"
    example_liquid_masses = "data/berkeley/ExampleTankLiquidMassData.csv"

    #Import gas mass data
    gas_masses = "data/berkeley/Test135GasMass.csv"
    example_gas_masses = "data/berkeley/ExampleTankGasMassData.csv"
    
    #Generate tanks based on Berkeley SEB team's real tank geometries
    real_tank_lox = MassBasedTank("Real Tank", real_geometry, lox_masses, gas_masses, lox, n2) 

    #Generate tank geometry {radius: height, ...}
    example_geometry = {(0, 5): 1}

    #Generate tanks based on simplified tank geometry
    example_tank_lox = MassBasedTank("Example Tank", example_geometry, example_liquid_masses, example_gas_masses, lox, n2) 

    initial_liquid_mass = 5
    initial_gas_mass = .1
    liquid_mass_flow_rate_in = .1
    gas_mass_flow_rate_in = .01
    liquid_mass_flow_rate_out = .2
    gas_mass_flow_rate_out = .02

    def test(calculated, expected, t, real=False):
        """Iterate over time range and test that calculated value is close to actual value"""
        j = 0
        for i in range(0, t, 0.1):
            try: 
                assert isclose(calculated.getValue(i), expected(j), abs_tol=10e-4)
            except IndexError:
                break
            
            if real:
                j+= 4
            else:
                j += 1

    def test_mass():
        """Test mass function of MassBasedTank subclass of Tank"""
        example_expected = lambda t: initial_liquid_mass + t * (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) \
                                + initial_gas_mass + t * (gas_mass_flow_rate_in - gas_mass_flow_rate_out)
        example_calculated = example_tank_lox.mass()

        lox_vals = []
        with open(lox_masses) as masses:
            reader = csv.reader(masses)
            for row in reader:
                lox_vals.append(row[1])

        real_expected = lambda t: lox_vals[t]
        real_calculated = real_tank_lox.mass()

        test(example_calculated, example_expected, 10)
        test(real_calculated, real_expected, 15.5, real=True)

    def test_net_mfr():
        """Test netMassFlowRate function of MassBasedTank subclass of Tank"""
        example_expected = lambda t: liquid_mass_flow_rate_in - liquid_mass_flow_rate_out \
                                + gas_mass_flow_rate_in - gas_mass_flow_rate_out
        example_calculated = example_tank_lox.netMassFlowRate()

        liquid_mfrs = []
        with open(lox_masses) as masses:
            reader = csv.reader(masses)
            initial_mass = reader[0][1]
            for row in reader:
                liquid_mfrs.append(initial_mass - row[1])

        gas_mfrs = []
        with open(gas_masses) as masses:
            reader = csv.reader(masses)
            initial_mass = reader[0][1]
            for row in reader:
                gas_mfrs.append(initial_mass - row[1])

        real_expected = lambda t: (liquid_mfrs[t] + gas_mfrs[t]) / t
        real_calculated = real_tank_lox.netMassFlowRate()

        test(example_calculated, example_expected, 10)
        test(real_calculated, real_expected, 15.5, real=True)

    def test_eval_ullage():
        """Test evaluateUllage function of MassBasedTank subclass of Tank"""
        example_expected = lambda t: (initial_liquid_mass * (liquid_mass_flow_rate_in - \
                                liquid_mass_flow_rate_out) * t) / lox.density / np.pi
        example_calculated = example_tank_lox.evaluateUllageHeight()

        liquid_heights = []
        with open(lox_masses) as masses:
            reader = csv.reader(masses)
            for row in reader:
                tank_vol = real_tank_lox.tank_vol.reverse()
                curr_height = tank_vol.getValue(row[1] / real_tank_lox.liquid.density)
                liquid_heights.append(curr_height)

        real_expected = lambda t: liquid_heights[t]
        real_calculated = real_tank_lox.evaluateUllageHeight()

        test(example_calculated, example_expected, 10)
        test(real_calculated, real_expected, 15.5, real=True)


# @curtisjhu
def test_ullage_based_motor():

    lox = Fluid(name = "LOx", density=1141.7, quality = 1.0)
    n2 = Fluid(name = "Nitrogen Gas", density=51.75, quality = 1.0)

    test_dir = '../data/e1-hotfires/test136/'

    top_endcap = lambda y: np.sqrt(0.0775 ** 2 - (y - 0.692300000000001) ** 2)
    bottom_endcap = lambda y: np.sqrt(0.0775 ** 2 - (0.0775 - y) ** 2)
    tank_geometry = {(0, 0.0559): bottom_endcap, (.0559, 0.7139): lambda y: 0.0744, (0.7139, 0.7698): top_endcap}

    ullage_data = pd.read_csv(os.path.abspath(test_dir+'loxUllage.csv')).to_numpy()
    ullageTank = UllageBasedTank("Ullage Tank", tank_geometry,
                                 gas=n2, liquid=lox, ullage=ullage_data)

    mass_data = pd.read_csv(test_dir+'loxMass.csv').to_numpy()
    mass_flow_rate_data = pd.read_csv(test_dir+'loxMFR.csv').to_numpy()

    def align_time_series(small_source, large_source):
        assert isinstance(small_source, np.ndarray) and isinstance(large_source, np.ndarray), "Must be np.ndarrays"
        if small_source.shape[0] > large_source.shape[0]:
            small_source, large_source = large_source, small_source

        result_larger_source = np.ndarray(small_source.shape)
        result_smaller_source = np.ndarray(small_source.shape)
        tolerance = .1
        curr_ind = 0
        for val in small_source:
            time = val[0]
            delta_time_vector = abs(time-large_source[:, 0])
            largeIndex = np.argmin(delta_time_vector)
            delta_time = abs(time - large_source[largeIndex][0])

            if delta_time < tolerance:
                result_larger_source[curr_ind] = large_source[largeIndex]
                result_smaller_source[curr_ind] = val
                curr_ind += 1
        return result_larger_source, result_smaller_source

    assert np.allclose(ullageTank.liquidHeight().getSource(), ullage_data)

    calculated_mass = ullageTank.liquidMass().getSource()
    calculated_mass, mass_data = align_time_series(calculated_mass, mass_data)
    assert np.allclose(calculated_mass, mass_data, rtol=1, atol=2)
    # Function(calculated_mass).plot1D()
    # Function(mass_data).plot1D()


    calculated_mfr, test_mfr = align_time_series(ullageTank.netMassFlowRate().getSource(), mass_flow_rate_data)
    # assert np.allclose(calculated_mfr, test_mfr)
    # Function(calculated_mfr).plot1D()
    # Function(test_mfr).plot1D()

# @gautamsaiy
def test_mfr_tank_basic():
    def test(t, a):
        for i in np.arange(0, 10, .2):
            assert isclose(t.getValue(i), a(i), abs_tol=1e-5)
            # print(t.getValue(i), a(i))
            # print(t(i))

    def test_nmfr():
        nmfr = lambda x: liquid_mass_flow_rate_in + gas_mass_flow_rate_in - liquid_mass_flow_rate_out - gas_mass_flow_rate_out
        test(t.netMassFlowRate(), nmfr)

    def test_mass():
        m = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) + \
            (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)
        lm = t.mass()
        test(lm, m)

    def test_liquid_height():
        alv = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) / lox.density
        alh = lambda x: alv(x) / (np.pi)
        tlh = t.liquidHeight()
        test(tlh, alh)

    def test_com():
        alv = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) / lox.density
        alh = lambda x: alv(x) / (np.pi)
        alm = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x)
        agm = lambda x: (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)

        alcom = lambda x: alh(x) / 2
        agcom = lambda x: (5 - alh(x)) / 2 + alh(x)
        acom = lambda x: (alm(x) * alcom(x) + agm(x) * agcom(x)) / (alm(x) + agm(x))

        tcom = t.centerOfMass
        test(tcom, acom)

    # def test_inertia():
    #     alv = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) / lox.density
    #     alh = lambda x: alv(x) / (np.pi)
    #     m = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) + \
    #         (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)
    #     r = 1
    #     iz = lambda x: (m(x) * r**2)/2
    #     ix = lambda x: (1/12)*m(x)*(3*r**2 + alh(x) **2)
    #     iy = lambda x: (1/12)*m(x)*(3*r**2 + alh(x) **2)
    #     test(i, 0)
    #


    tank_radius_function = {(0, 5): 1}
    lox = Fluid(name = "LOx", density = 1141, quality = 1.0) #Placeholder quality value
    n2 = Fluid(name = "Nitrogen Gas", density = 51.75, quality = 1.0) #Placeholder quality value; density value may be estimate
    initial_liquid_mass = 5
    initial_gas_mass = .1
    liquid_mass_flow_rate_in = .1
    gas_mass_flow_rate_in = .01
    liquid_mass_flow_rate_out = .2
    gas_mass_flow_rate_out = .02

    t = MassFlowRateBasedTank("Test Tank", tank_radius_function,
            initial_liquid_mass, initial_gas_mass, liquid_mass_flow_rate_in,
            gas_mass_flow_rate_in, liquid_mass_flow_rate_out, 
            gas_mass_flow_rate_out, lox, n2)

    test_nmfr()
    test_mass()
    test_liquid_height()
    test_com()
    # test_inertia()

def test_mfr_tank():
    def test(t, a):
        for i in np.arange(0, 10, .2):
            assert isclose(t.getValue(i), a(i), abs_tol=1e-5)

    def test_nmfr():
        anmfr = Function("../data/e1/nmfr.csv")
        nmfr = t.netMassFlowRate()
        test(nmfr, anmfr)

    def test_mass():
        am = Function("../data/e1/mass.csv")
        m = t.mass()
        test(m, am)
    
    def test_liquid_height():
        alh = Function("../data/e1/ullage_height.csv")
        lh = t.liquidHeight()
        test(lh, alh)

    initial_liquid_mass = Function("../data/e1/liquid_mass.csv")(0)
    initial_gas_mass = Function("../data/e1/gas_mass.csv")(0)
    liquid_mass_flow_rate_in = Function(0)
    gas_mass_flow_rate_in = Function("../data/e1/gas_mfri.csv")
    liquid_mass_flow_rate_out = Function("../data/e1/liquid_mfr.csv")
    gas_mass_flow_rate_out = Function("../data/e1/gas_mfro.csv")

    lox = Fluid(name = "LOx", density = 1141, quality = 1.0) #Placeholder quality value
    n2 = Fluid(name = "Nitrogen Gas", density = 51.75, quality = 1.0) #Placeholder quality value; density value may be estimate

    top_endcap = lambda y: np.sqrt(0.0775 ** 2 - (y - 0.692300000000001) ** 2)
    bottom_endcap = lambda y: np.sqrt(0.0775 ** 2 - (0.0775 - y) **2)
    tank_geometry = {(0, 0.0559): bottom_endcap, (.0559, 0.7139): lambda y: 0.0744, (0.7139, 0.7698): top_endcap}
    
    t = MassFlowRateBasedTank("Test Tank", tank_geometry,
            initial_liquid_mass, initial_gas_mass, liquid_mass_flow_rate_in,
            gas_mass_flow_rate_in, liquid_mass_flow_rate_out,
            gas_mass_flow_rate_out, lox, n2)

    test_nmfr()
    test_mass()
    test_liquid_height()
