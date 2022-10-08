# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano and Pedro Henrique Marinho Bressan"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

import numpy as np
from scipy import integrate

from rocketpy.Function import Function
from rocketpy.motors import Motor
from rocketpy.supplement import Disk, Cylinder, Hemisphere


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
        massFlowRate = 0

        for tank in self.tanks:
            massFlowRate += tank.get("tank").netMassFlowRate

        return massFlowRate

    def evaluateCenterOfMass(self):
        totalMass = 0
        massBalance = 0

        for tankElement in self.tanks:
            tank = tankElement.get("tank")
            tankPosition = tankElement.get("position")
            totalMass += tank.mass
            massBalance += tank.mass * (tankPosition - tank.centerOfMass)

        return massBalance / totalMass

    def evaluateInertiaTensor(self):
        pass

    def addTank(self, tank, position):
        self.tanks.append({"tank": tank, "position": position})


class Tank(ABC):
    def __init__(
        self, name, diameter, height, gas, liquid=0, bottomCap="flat", upperCap="flat"
    ):
        self.name = name
        self.diameter = diameter
        self.height = height
        self.gas = gas
        self.liquid = liquid
        self.bottomCap = bottomCap
        self.upperCap = upperCap

        self.capMap = {
            "flat": Disk,
            "spherical": Hemisphere,
        }
        self.setTankGeometry()

        pass

    def setTankGeometry(self):
        self.cylinder = Cylinder(self.diameter / 2, self.height)
        self.bottomCap = self.capMap.get(self.bottomCap)(
            self.diameter / 2, fill_direction="upwards"
        )
        self.upperCap = self.capMap.get(self.upperCap)(
            self.diameter / 2, fill_direction="downwards"
        )

    def setTankFilling(self, t):
        liquidVolume = self.liquidVolume(t)

        if liquidVolume < self.bottomCap.volume:
            self.bottomCap.filled_volume = liquidVolume
        elif (
            self.bottomCap.volume
            <= liquidVolume
            <= self.bottomCap.volume + self.cylinder.volume
        ):
            self.bottomCap.filled_volume = self.bottomCap.volume
            self.cylinder.filled_volume = liquidVolume - self.bottomCap.volume
        else:
            self.bottomCap.filled_volume = self.bottomCap.volume
            self.cylinder.filled_volume = self.cylinder.volume
            self.upperCap.filled_volume = liquidVolume - (
                self.bottomCap.volume + self.cylinder.volume
            )

    @abstractmethod
    def mass(self, t):
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
    def netMassFlowRate(self, t):
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
    def liquidVolume(self, t):
        """Returns the volume of liquid inside the tank as a function
        of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Tank's liquid volume as a function of time.
        """
        pass

    def centerOfMass(self, t):
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
        self.setTankFilling(t)

        bottomCapMass = self.liquid.density * self.bottomCap.filled_volume
        cylinderMass = self.liquid.density * self.cylinder.filled_volume
        upperCapMass = self.liquid.density * self.upperCap.filled_volume

        centerOfMass = (
            self.bottomCap.filled_centroid * bottomCapMass
            + self.cylinder.filled_centroid * cylinderMass
            + self.upperCap.filled_centroid * upperCapMass
        ) / self.mass(t)

        return centerOfMass

    def inertiaTensor(self, t):
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
        ...


# @MrGribel
class MassFlowRateBasedTank(Tank):
    def __init__(
        self,
        name,
        diameter,
        height,
        endcap,
        initial_liquid_mass,
        initial_gas_mass,
        liquid_mass_flow_rate_in,
        gas_mass_flow_rate_in,
        liquid_mass_flow_rate_out,
        gas_mass_flow_rate_out,
        liquid,
        gas,
    ):
        super().__init__(name, diameter, height, endcap, gas, liquid)


# @phmbressan
class UllageBasedTank(Tank):
    def __init__(
        self,
        name,
        diameter,
        height,
        endcap,
        liquid,
        gas,
        ullage,
    ):
        super().__init__(name, diameter, height, endcap, gas, liquid)
        pass


# @ompro07
class MassBasedTank(Tank):
    def __init__(
        self,
        name,
        diameter,
        height,
        endcap,
        liquid_mass,
        gas_mass,
        liquid,
        gas,
    ):
        super().__init__(name, diameter, height, endcap, gas, liquid)
        pass
