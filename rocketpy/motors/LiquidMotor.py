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

    @abstractmethod
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
        pass

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
        self.initial_liquid_mass = Function(initial_liquid_mass)
        self.initial_gas_mass = Function(initial_gas_mass)
        self.liquid_mass_flow_rate_in = Function(liquid_mass_flow_rate_in)
        self.gas_mass_flow_rate_in = Function(gas_mass_flow_rate_in)
        self.liquid_mass_flow_rate_out = Function(liquid_mass_flow_rate_out)
        self.gas_mass_flow_rate_out = Function(gas_mass_flow_rate_out)

    def mass(self):
        return (
            self.initial_liquid_mass
            + self.liquid_mass_flow_rate_in
            - self.liquid_mass_flow_rate_out
            + self.initial_gas_mass
            + self.gas_mass_flow_rate_in
            - self.gas_mass_flow_rate_out
        )
    
    def netMassFlowRate(self):
        return (
            self.liquid_mass_flow_rate_in
            - self.liquid_mass_flow_rate_out
            + self.gas_mass_flow_rate_in
            - self.gas_mass_flow_rate_out
        )

    def centerOfMass(self):
        liquid_mass = (
            self.initial_liquid_mass
            + self.liquid_mass_flow_rate_in
            - self.liquid_mass_flow_rate_out
        )

        gas_mass = (
            self.initial_gas_mass
            + self.gas_mass_flow_rate_in
            - self.gas_mass_flow_rate_out
        )
        
        liquid_volume = liquid_mass / self.liquid.density
        gas_volume = gas_mass / self.gas.density

        # How to get height of liquid and gas from volume of gas



    def inertiaTensor(self):
        # How to calculate intertia tensor
        pass


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

