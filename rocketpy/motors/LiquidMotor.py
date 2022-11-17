# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod
from cmath import tan
from turtle import position

import numpy as np
from scipy import integrate

from rocketpy.Function import Function, PiecewiseFunction
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
            total_mass += tank["tank"].mass() + Function(lambda t: tank["tank"].netMassFlowRate.integral(t - .05, t + .05))
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
        assert isinstance(name, str)
        assert(isinstance(tank_geometry, dict))
        assert(isinstance(gas, Fluid))
        assert(isinstance(liquid, Fluid) or liquid == 0)

        self.height = sorted(tank_geometry.keys())[-1][1]
        self.tank_geometry = PiecewiseFunction(tank_geometry)
            
        self.tank_geometry.setInputs("y")
        self.tank_geometry.setOutputs("radius")

        self.tank_area = self.tank_geometry ** 2 * np.pi
        self.tank_area.setInputs("y")
        self.tank_area.setOutputs("area")

        self.tank_vol = Function(lambda x: self.tank_area.integral(0, x))
        self.tank_vol.setInputs("y")
        self.tank_vol.setOutputs("volume")

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
    def liquidMass():
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
    def gasMass():
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

    @property
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
        tank_liquid_A = Function(lambda x: self.tank_geometry.integral(0, x))
        tank_liquid_inside_integral = Function(lambda x: x * self.tank_geometry(x))
        tank_liquid_integral = Function(lambda x: tank_liquid_inside_integral.integral(0, x))
        tank_liquid_com = Function(lambda x: tank_liquid_integral(x) / (tank_liquid_A(x) + 1e-9))

        tank_gas_A = Function(lambda x: self.tank_geometry.integral(x, self.height))
        tank_gas_inside_integral = Function(lambda x: x * self.tank_geometry(x))
        tank_gas_integral = Function(lambda x: tank_gas_inside_integral.integral(x, self.height))
        tank_gas_com = Function(lambda x: tank_gas_integral(x) / (tank_gas_A(x) + 1e-9))


        tank_liquid_com_t = Function(lambda t: tank_liquid_com(self.liquidHeight()(t)))
        tank_gas_com_t = Function(lambda t: tank_gas_com(self.liquidHeight()(t)))

        lm = self.liquidMass()
        gm = self.gasMass()

        liquid_mass_c = Function(lambda t: tank_liquid_com_t(t) * lm(t))
        gas_mass_c = Function(lambda t: tank_gas_com_t(t) * gm(t))

        com = Function(lambda t: (liquid_mass_c(t) + gas_mass_c(t)) / (self.mass()(t) + 1e-9), "Time", "Center of Mass")
        return com

    @property
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
        Ix = Function(lambda h: (self.tank_geometry ** 2).integral(0, h))
        Ix = Function(lambda t: Ix(self.liquidHeight()(t)) * (1/2) * self.mass()(t), "Time", "Inertia tensor")

        return Ix
        

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
        nmfr = self.netMassFlowRate()
        m = Function(lambda t: self.initial_liquid_mass.getValue(t) 
            + self.initial_gas_mass.getValue(t) 
            + nmfr.integral(0, t))
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
        mfr.setOutputs("Net Mass Flow Rate")
        return mfr

    def liquidHeight(self):
        liquid_vol = Function(lambda t: (self.initial_liquid_mass.getValue(t)
                + self.liquid_mass_flow_rate_in.integral(0, t)
                - self.liquid_mass_flow_rate_out.integral(0, t))
                / self.liquid.density)
        uH = Function(lambda t: self.tank_vol.findOptimalInput(liquid_vol.getValue(t)))
        uH.setInputs("Time")
        uH.setOutputs("Height")
        return uH

    def liquidMass(self):
        liquid_mass = Function(lambda t: self.initial_liquid_mass.getValue(t)
                + self.liquid_mass_flow_rate_in.integral(0, t)
                - self.liquid_mass_flow_rate_out.integral(0, t))
        liquid_mass.setInputs("Time")
        liquid_mass.setOutputs("Mass")
        return liquid_mass
    
    def gasMass(self):
        gas_mass = Function(lambda t: self.initial_gas_mass.getValue(t)
                + self.gas_mass_flow_rate_in.integral(0, t)
                - self.gas_mass_flow_rate_out.integral(0, t))
        gas_mass.setInputs("Time")
        gas_mass.setOutputs("Mass")
        return gas_mass


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
        m = self.liquidMass() + self.gasMass()
        m.setInputs("Time")
        m.setOutputs("Mass")
        return m

    def netMassFlowRate(self):
        m = self.mass()
        mfr = m.derivativeFunction()
        mfr.setInputs("Time")
        mfr.setOutputs("Mass Flow Rate")
        return mfr

    def liquidHeight(self):
        return self.ullageHeight

    def liquidMass(self):
        liquid_mass = self.tank_vol.functionOfAFunction(self.ullageHeight) * self.liquid.density
        liquid_mass.setInputs("Time")
        liquid_mass.setOutputs("Liquid Mass")
        return liquid_mass

    def gasMass(self):
        gas_mass = self.tank_vol.functionOfAFunction(self.ullageHeight) * self.gas.density
        gas_mass.setInputs("Time")
        gas_mass.setOutputs("Gas Mass")
        return gas_mass




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

    def find_liquid_mass(self):
        return self.liquid_mass

    def find_gas_mass(self):
        return self.gas_mass

    def mass(self):
        m = self.find_liquid_mass() + self.find_gas_mass()
        m.setInputs("Time")
        m.setOutputs("Mass")
        return m
    
    def netMassFlowRate(self):
        m = self.mass()
        mfr = m.derivativeFunction()
        mfr.setInputs("Time")
        mfr.setOutputs("Mass Flow Rate")
        return mfr

    def evaluateUllageHeight(self):
        liquid_volume = self.liquid_mass / self.liquid.density
        tank_vol = self.tank_vol.reverse()
        ullage_height = Function(lambda t: tank_vol.getValue(liquid_volume.getValue(t)))
        ullage_height.setInputs("Time")
        ullage_height.setOutputs("Ullage Height")
        return ullage_height

        

