# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano and Pedro Henrique Marinho Bressan"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod
from cmath import tan

import numpy as np
from scipy import integrate

from rocketpy.Function import Function, PiecewiseFunction
from rocketpy.motors import Motor

# @Stano
class LiquidMotor(Motor):
    def __init__(
        self,
        thrustSource,
        burnOut,
        nozzleRadius,
        throatRadius,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
    ):

        super.__init__()
        self.tanks = []
        pass

    def evaluateMassFlowRate(self):
        pass

    def evaluateCenterOfMass(self):
        pass

    def evaluateInertiaTensor(self):
        pass

    def addTank(self, tank, position):
        self.tanks.append({"tank": tank, "position": position})


class Tank(ABC):
    def __init__(self, name, tank_geometry, gas, liquid=0):
        self.name = name
        if isinstance(tank_geometry, PiecewiseFunction):
            self.tank_geometry = tank_geometry
        else:
            self.tank_geometry = PiecewiseFunction(tank_geometry)
        self.tank_geometry.setInputs("y")
        self.setoutputs("radius")
        self.gas = gas
        self.liquid = liquid


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

    def uilageHeight(self):
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
    @abstractmethod
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
        pass


# @MrGribel
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

    def uilageHeight(self):
        uH = self.tankVolume().functionOfAFunction(self.liquidVolume())
        uH.setInputs("Time")
        uH.setOutputs("Height")
        return uH


# @phmbressan
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
        pass


# @ompro07
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
        pass

