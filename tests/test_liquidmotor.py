from rocketpy import Fluid
from rocketpy.motors.LiquidMotor import Tank, LiquidMotor, MassBasedTank, UllageBasedTank, MassFlowRateBasedTank
import numpy as np


# @PBales1
def test_mass_based_motor():
    lox = Fluid(name = "LOx", density = 1141.7, quality = 1.0) #Placeholder quality value
    propane = Fluid(name = "Propane", density = 493, quality = 1.0) #Placeholder quality value
    n2 = Fluid(name = "Nitrogen Gas", density = 51.75, quality = 1.0) #Placeholder quality value; density value may be estimate
    
    example_motor = MassBasedTank("Example Tank", 0.1540, 0.66, 0.7, "Placeholder", "Placeholder", lox, n2) 


# @curtisjhu
def test_ullage_based_motor():
    lox = Fluid(name = "LOx", density = 1, quality = 1.0)
    n2 = Fluid(name = "Nitrogen Gas", density = 3, quality = 1.0)

    ullageData = [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)] # constant flow rate
    tank_geometry = {(0, 6): lambda y: 1}
    ullageTank = UllageBasedTank("Ullage Tank", tank_geometry, gas=n2, liquid=lox, ullage=ullageData)


    assert np.allclose(ullageTank.centerOfMass().getSource(), np.array([3, 2.5, 2, 1.5, 1, 0.5]))
    assert np.allclose(ullageTank.mass().getSource(), np.array([18.84, 25.12, 31.4, 37.68, 43.96, 50.24]))
    assert np.allclose(ullageTank.netMassFlowRate().getSource(), np.array([-6.28, -6.28, -6.28, -6.28, -6.28, -6.28]))

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
        test(t.evaluateUllageHeight(), actual_liquid_vol)

    
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
