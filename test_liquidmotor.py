from rocketpy.motors.LiquidMotor import Tank, LiquidMotor, MassBasedTank, UllageBasedTank, MassFlowRateBasedTank
from rocketpy.motors.Fluid import Fluid
from rocketpy.Function import Function
from math import isclose
import numpy as np
import csv


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

    #Need docs to be pushed -> "Placeholder" represents where these data analysis-created docs will be inserted


# @curtisjhu
def test_ullage_based_motor():
    lox = Fluid(name = "LOx", density = 1141, quality = 1.0) #Placeholder quality value
    propane = Fluid(name = "Propane", density = 493, quality = 1.0) #Placeholder quality value
    n2 = Fluid(name = "Nitrogen Gas", density = 51.75, quality = 1.0) #Placeholder quality value; density value may be estimate

    ullageData = []
    ullageTank = UllageBasedTank("Ullage Tank",  diameter=3, height=4, endcap="flat", gas=n2, liquid=lox, ullage=ullageData)
    
    ullageTank.centerOfMass()
    ullageTank.netMassFlowRate()
    ullageTank.mass()
    ullageTank.liquidVolume()

# @gautamsaiy
def test_mfr_tank_basic1():
    def test(t, a):
        for i in np.arange(0, 10, .2):
            print(t.getValue(i), a(i))
            # assert isclose(t.getValue(i), a(i))
    
    def test_nmfr():
        nmfr = lambda x: liquid_mass_flow_rate_in + gas_mass_flow_rate_in - liquid_mass_flow_rate_out - gas_mass_flow_rate_out
        test(t.netMassFlowRate(), nmfr)

    def test_mass():
        m = lambda x: (initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) + \
            (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)
        lm = t.mass()
        test(lm, m)

    def test_uh():
        actual_liquid_vol = lambda x: ((initial_liquid_mass + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x) / lox.density) / np.pi * list(tank_radius_function.values())[0] ** 2
        test(t.evaluateUilageHeight(), actual_liquid_vol)

    
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
    test_uh()
