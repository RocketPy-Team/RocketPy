# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod
from cmath import tan
from turtle import position

import numpy as np
from scipy import integrate

from rocketpy.Function import Function, PiecewiseFunction, funcify_method
from rocketpy.motors import Motor, Fluid

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
        self.thrustSource = Function(thrustSource)
        self.burnOut = burnOut

    def evaluateMassFlowRate(self):
        total_mfr = 0
        for tank in self.tanks:
            total_mfr += tank.netMassFlowRate()
        return total_mfr

    def evaluateCenterOfMass(self):
        com = Function(0)
        total_mass = Function(0)
        for tank in self.tanks:
            com += (tank["tank"].centerOfMass() + tank[position]) * tank["tank"].mass()
            total_mass += tank["tank"].mass() + Function(
                lambda t: tank["tank"].netMassFlowRate.integral(t - 0.05, t + 0.05)
            )
        com = com / total_mass
        com.setInputs("Time")
        com.setOutputs("Center of mass")
        return com

    def evaluateInertiaTensor(self):
        inertiaI = Function(0)
        inertiaZ = Function(0)
        for tank in self.tanks:
            inertia = tank["tank"].inertiaTensor()
            inertiaI += inertia[0]
            inertiaZ += inertia[1]
        inertiaI.setInputs("Time")
        inertiaI.setOutputs("Inertia tensor")
        inertiaZ.setInputs("Time")
        inertiaZ.setOutputs("Inertia tensor")
        return [inertiaI, inertiaZ]

    def addTank(self, tank, position):
        self.tanks.append({"tank": tank, "position": position})


# @gautamsaiy
class Tank(ABC):
    def __init__(self, name, tank_geometry, gas, liquid=0):
        self.structure = tank_geometry
        self.gas = gas
        self.liquid = liquid

    @abstractmethod
    @funcify_method("time (s)", "mass (kg)")
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
    @funcify_method("time (s)", "mass flow rate (kg/s)")
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
    @funcify_method("time (s)", "liquid volume (m^3)")
    def liquidVolume(self):
        pass

    @abstractmethod
    @funcify_method("time (s)", "gas volume (m³)")
    def gasVolume(self):
        pass

    @abstractmethod
    @funcify_method("time (s)", "liquid height (m)")
    def liquidHeight(self):
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

    @abstractmethod
    @funcify_method("time (s)", "gas height (m)")
    def gasHeight(self):
        pass

    @abstractmethod
    @funcify_method("time (s)", "liquid mass (kg)")
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "gas mass (kg)")
    def gasMass(self):
        """
        Returns the mass of the gas as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        pass
    
    @funcify_method("height (m)", "time density function")
    def density(self, height):
        def time_density(time):
            liquidHeight = self.liquidHeight()(time)
            gasHeight = self.gasHeight()(time)
            if 0 <= height <= liquidHeight:
                return self.liquid.density
            elif liquidHeight < height < gasHeight:
                return self.gas.density
            else:
                return 0
        return Function(lambda t: time_density(t))

    #@funcify_method("time (s)", "center of mass (m)")
    #TODO
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
        def com(t):
            # integrand of com integral
            d_mass = Function(lambda h: h * self.density(h)(t) * self.structure.area(h)).setDiscrete(25)

            return d_mass.integral(self.structure.bottom, self.gasHeight()(t)) / self.mass()(t)
        return Function(com)

    #@funcify_method("time (s)", "inertia tensor (kg*m^2)")
    #TODO
    def inertiaTensor(self):
        """
        Returns the inertia tensor of the tank's fluids as a function of
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
        def inet(t):
            den = lambda h: self.density(h,t)
            d_inertia = Function(lambda h: h**2 * den(h) * self.structure.area(h))
            return d_inertia(t)
        inertia = Function(inet)
        Ix = inertia.integral(self.structure.bottom, self.gasHeight(), numerical=True)

        # Steiner theorem to account for center of mass
        Ix -= self.mass() * self.centerOfMass()**2
        return Ix, 0


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
        self.initial_liquid_mass = initial_liquid_mass
        self.initial_gas_mass = initial_gas_mass

        self.liquid_mass_flow_rate_in = Function(
            liquid_mass_flow_rate_in, inputs="Time", outputs="Mass Flow Rate"
        )
        self.gas_mass_flow_rate_in = Function(
            gas_mass_flow_rate_in, inputs="Time", outputs="Mass Flow Rate"
        )
        self.liquid_mass_flow_rate_out = Function(
            liquid_mass_flow_rate_out, inputs="Time", outputs="Mass Flow Rate"
        )
        self.gas_mass_flow_rate_out = Function(
            gas_mass_flow_rate_out, inputs="Time", outputs="Mass Flow Rate"
        )

    def mass(self):
        return self.liquidMass() + self.gasMass()

    def liquidMass(self):
        liquid_flow = self.netLiquidFlowRate().integralFunction()
        return self.initial_liquid_mass + liquid_flow

    def gasMass(self):
        gas_flow = self.netGasFlowRate().integralFunction()
        return self.initial_gas_mass + gas_flow

    def netLiquidFlowRate(self):
        return self.liquid_mass_flow_rate_in - self.liquid_mass_flow_rate_out

    def netGasFlowRate(self):
        return self.gas_mass_flow_rate_in - self.gas_mass_flow_rate_out

    def netMassFlowRate(self):
        return self.netLiquidFlowRate() + self.netGasFlowRate()

    def liquidVolume(self):
        return self.liquidMass() / self.liquid.density

    def gasVolume(self):
        return self.gasMass() / self.gas.density

    def liquidHeight(self):
        liquid_volume = self.liquidVolume()
        inverse_volume = self.structure.inverse_volume.setDiscrete()
        return inverse_volume.compose(liquid_volume)

    def gasHeight(self):
        fluid_volume = self.gasVolume() + self.liquidVolume()
        inverse_volume = self.structure.inverse_volume.setDiscrete()
        return inverse_volume.compose(fluid_volume)
 


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
        self.ullage = Function(ullage, inputs="Time", outputs="Volume")

    def mass(self):
        return self.liquidMass() + self.gasMass()

    def netMassFlowRate(self):
        return self.mass().derivativeFunction()

    def liquidVolume(self):
        return self.structure.total_volume.item - self.ullage

    def gasVolume(self):
        return self.ullage

    def gasMass(self):
        return self.gasVolume() * self.gas.density

    def liquidMass(self):
        return self.liquidVolume() * self.liquid.density

    def liquidHeight(self):
        liquid_volume = self.liquidVolume()
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(liquid_volume)

    def gasHeight(self):
        fluid_volume = self.gasVolume() + self.liquidVolume()
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(fluid_volume)


# @phmbressan
# @lperi03
# @curtisjhu
class LevelBasedTank(Tank):
    def __init__(
        self,
        name,
        tank_geometry,
        liquid,
        gas,
        liquid_height,
    ):
        super().__init__(name, tank_geometry, gas, liquid)
        self.liquid_height = Function(liquid_height, inputs="Time", outputs="Height")

    def mass(self):
        # print(type(self.liquidMass()), type(self.gasMass()))
        return self.liquidMass() + self.gasMass()

    def netMassFlowRate(self):
        return self.mass().derivativeFunction()

    def liquidVolume(self):
        return self.structure.volume.compose(self.liquidHeight())

    def gasVolume(self):
        return self.structure.total_volume.item() - self.liquidVolume()

    def liquidHeight(self):
        return self.liquid_height

    def gasMass(self):
        return self.gasVolume() * self.gas.density

    def liquidMass(self):
        return self.liquidVolume() * self.liquid.density

    def gasHeight(self):
        fluid_volume = self.gasVolume() + self.liquidVolume()
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(fluid_volume)


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
        return self.liquidMass() + self.gasMass()

    def netMassFlowRate(self):
        return self.mass().derivativeFunction()

    def liquidMass(self):
        return self.liquid_mass

    def gasMass(self):
        return self.gas_mass

    def gasVolume(self):
        return self.gasMass() / self.gas.density

    def liquidVolume(self):
        return self.liquidMass() / self.liquid.density

    def liquidHeight(self):
        liquid_volume = self.liquidVolume()
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(liquid_volume)

    def gasHeight(self):
        fluid_volume = self.gasVolume() + self.liquidVolume()
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(fluid_volume)
