# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod
from cmath import tan

import numpy as np
from scipy import integrate

from rocketpy.Function import Function, PiecewiseFunction
from rocketpy.motors import Motor

# @Stano
# @PBales1
# @lperi03
# @gautamsaiy
class LiquidMotor(Motor):
    def __init__(
        self,
        thrustSource,
        burnOut,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
    ):

        super.__init__()
        self.tanks = []
        self.thrustSource = thrustSource
        self.burnOut = burnOut

    def evaluateMassFlowRate(self):
        total_mfr = 0
        for tank in self.tanks:
            total_mfr += tank.netMassFlowRate()
        return total_mfr


    def evaluateCenterOfMass(self):
        try:
            print('insert function here')
        except NameError:
            print("Chamber not added")


    def evaluateInertiaTensor(self):
        pass

    def addTank(self, tank, position):
        self.tanks.append({"tank": tank, "position": position})

    def addChamber(self, nozzleHeight, nozzleRadius, throatHeight, throatRadius):
        self.chamber = ({"nozzle": {"height": nozzleHeight, "radius": nozzleRadius}, \
            "throat": {"height": throatHeight, "radius": throatRadius}})


# @gautamsaiy
class Tank(ABC):
    def __init__(self, name, tank_geometry, gas, liquid=0):
        if isinstance(tank_geometry, dict):
            self.tank_geometry = PiecewiseFunction(tank_geometry)
        else:
            self.tank_geometry = Function(tank_geometry)
        self.tank_geometry.setInputs("y")
        self.setoutputs("radius")

        self.tank_area = self.tank_geometry ** 2 * np.pi
        self.tank_area.setInputs("y")
        self.tank_area.setOutputs("area")

        self.tank_vol = self.tank_area.integralFunction()
        self.tank_vol.setInputs("y")
        self.tank_vol.setOutputs("volume")

        self.height = sorted(self.tank_geometry.keys())[-1][1]

        self.gas = gas
        self.liquid = liquid
        self.uilageHeight = self.uilageHeight()


    @abstractmethod
    def mass(self):
        """Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        pass

    @abstractmethod
    def netMassFlowRate(self):
        """Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate exiting the tank minus the
        mass flow rate entering the tank, including liquids and gases.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        pass

    def evaluateUilageHeight(self):
        """
        Returns the height of the uilage as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Height of the uilage as a function of time.
        """
        pass
    
    def centerOfMass(self):
        """Returns the center of mass of the tank's fluids as a function of
        time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Center of mass of the tank's fluids as a function of time.
        """
        A = self.tank_geometry.integralFunction()

        insideIntegrandL = Function(lambda y: self.tank_geometry.getValue(y) * y)
        funcL = insideIntegrandL.integralFunction()
        funcL = funcL / A
        comLiquid = funcL.functionOfAFunction(self.uilageHeight())

        insideIntegrandG = Function(lambda x: (self.tank_geometry.integral(0, self.height) - self.tank_geometry.integralFunction().getValue(x)) * x)
        funcG = insideIntegrandG.integralFunction()
        funcG = funcG / A
        comGas = funcG.functionOfAFunction(self.uilageHeight())

        com = (comLiquid * self.liquidMass() + comGas * self.gasMass()) / self.mass()
        com.setInputs("Time")
        com.setOutputs("Height")
        return com


    @property
    def inertiaTensor(self):
        """Returns the inertia tensor of the tank's fluids as a function of
        time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Inertia tensor of the tank's fluids as a function of time.
        """
        lk = np.pi * self.liquid.density
        insideIntegral = (self.tank_geometry ** 2) * (self.height * self.centerOfMass()) ** 2
        integral = lk * insideIntegral.integralFunction()
        integral.setInputs("time")
        integral.setOutputs("Inertia")
        Ixx = Iyy = integral
        
        lk = lk / 2
        insideIntegral = self.tank_geometry ** 4
        integral = lk * insideIntegral.integralFunction()
        integral.setInputs("time")
        integral.setOutputs("Inertia")
        Izz = integral

        




# @MrGribel
# @gautamsaiy
class MassFlowRateBasedTank(Tank):
    def __init__(
        self,
        name,
        tank_geometry,
        initial_liquid_mass,
        initial_gas_mass,
        liquid_mass_flow_rate_in,
        gas_mass_flow_rate_in,
        liquid_mass_flow_rate_out,
        gas_mass_flow_rate_out,
        liquid,
        gas,
    ):
        super().__init__(name, tank_geometry, gas, liquid)
        self.initial_liquid_mass = Function(initial_liquid_mass, inputs="Time", outputs="Mass")
        self.initial_gas_mass = Function(initial_gas_mass, inputs="Time", outputs="Mass")
        self.liquid_mass_flow_rate_in = Function(liquid_mass_flow_rate_in, inputs="Time", outputs="Mass Flow Rate")
        self.gas_mass_flow_rate_in = Function(gas_mass_flow_rate_in, inputs="Time", outputs="Mass Flow Rate")
        self.liquid_mass_flow_rate_out = Function(liquid_mass_flow_rate_out, inputs="Time", outputs="Mass Flow Rate")
        self.gas_mass_flow_rate_out = Function(gas_mass_flow_rate_out, inputs="Time", outputs="Mass Flow Rate")

    def mass(self):
        m = self.liquidMass() + self.gasMass()
        m.setInputs("Time")
        m.setOutputs("Mass")
        return m
    
    def netMassFlowRate(self):
        mfr = (
            self.liquid_mass_flow_rate_in
            - self.liquid_mass_flow_rate_out
            + self.gas_mass_flow_rate_in
            - self.gas_mass_flow_rate_out
        )
        mfr.setInputs("Time")
        mfr.setOutputs("Mass Flow Rate")
        return mfr

    def liquidMass(self):
        lm = self.initial_liquid_mass + self.liquid_mass_flow_rate_in - self.liquid_mass_flow_rate_out
        lm.setInputs("Time")
        lm.setOutputs("Mass")
        return lm

    def gasMass(self):
        gm = self.initial_gas_mass + self.gas_mass_flow_rate_in - self.gas_mass_flow_rate_out
        gm.setInputs("Time")
        gm.setOutputs("Mass")
        return gm

    def liquidVolume(self):
        lV = self.liquidMass() / self.liquid.density
        lV.setInputs("Time")
        lV.setOutputs("Volume")
        return lV

    def gasVolume(self):
        gV = self.gasMass() / self.gas.density
        gV.setInputs("Time")
        gV.setOutputs("Volume")
        return gV

    def tankVolume(self):
        vol = (np.pi * self.tank_geometry.radius ** 2).integralFunction()
        vol.setInputs("Height")
        vol.setOutputs("Volume")
        return vol

    def evaluateUilageHeight(self):
        uH = self.tankVolume().functionOfAFunction(self.liquidVolume())
        uH.setInputs("Time")
        uH.setOutputs("Height")
        return uH


# @phmbressan
# @lperi03
# @curtisjhu
class UllageBasedTank(Tank):
    def __init__(
        self,
        name,
        tank_geometry,
        liquid,
        gas,
        ullage,
    ):
        super().__init__(name, tank_geometry, gas, liquid)
        self.ullageHeight = Function(ullage, inputs="Time", outputs="Height")

    def mass(self):
        lm = self.tank_vol.functionOfAFunction(self.ullage) * self.liquid.density
        gm = self.tank_vol.functionOfAFunction(self.ullage) * self.gas.density
        m = lm + gm
        m.setInputs("Time")
        m.setOutputs("Mass")
        return m

    def netMassFlowRate(self):
        m = self.mass()
        mfr = m.derivativeFunction()
        mfr.setInputs("Time")
        mfr.setOutputs("Mass Flow Rate")
        return mfr

    def evaluateUilageHeight(self):
        return self.ullageHeight



# @ompro07
# @PBales1
class MassBasedTank(Tank):
    def __init__(
        self,
        name,
        tank_geometry,
        liquid_mass,
        gas_mass,
        liquid,
        gas,
    ):
        super().__init__(name, tank_geometry, gas, liquid)
        self.liquid_mass = Function(liquid_mass, inputs="Time", outputs="Mass")
        self.gas_mass = Function(gas_mass, inputs="Time", outputs="Mass")

    def mass(self):
        m = self.liquid_mass + self.gas_mass
        m.setInputs("Time")
        m.setOutputs("Mass")
        return m
    
    def netMassFlowRate(self):
        m = self.mass()
        mfr = m.derivativeFunction()
        mfr.setInputs("Time")
        mfr.setOutputs("Mass Flow Rate")
        return mfr

    def evaluateUilageHeight(self):
        return super().evaluateUilageHeight()
        

