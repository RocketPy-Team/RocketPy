# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano and Pedro Henrique Marinho Bressan"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod
import functools

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
        self.positioned_tanks = []
        pass

    def evaluateMassFlowRate(self):
        massFlowRate = 0

        for positioned_tank in self.positioned_tanks:
            massFlowRate += positioned_tank.get("tank").netMassFlowRate

        return massFlowRate

    def evaluateCenterOfMass(self):
        totalMass = 0
        massBalance = 0

        for positioned_tank in self.positioned_tanks:
            tank = positioned_tank.get("tank")
            tankPosition = positioned_tank.get("position")
            totalMass += tank.mass
            massBalance += tank.mass * (tankPosition + tank.centerOfMass)

        return massBalance / totalMass

    def evaluateInertiaTensor(self):
        pass

    def addTank(self, tank, position):
        """
        Adds a tank to the rocket motor.

        Parameters
        ----------
        tank : Tank
            Tank object to be added to the rocket motor.
        position : float
            Position of the tank in relation to the motor nozzle, in meters.
            Should be a positive value.
        """
        self.positioned_tanks.append({"tank": tank, "position": position})


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
        liquidVolume = self.liquidVolume.getValueOpt(t)

        if 0 <= liquidVolume < self.bottomCap.volume:
            self.bottomCap.filled_volume = liquidVolume
            self.cylinder.filled_volume = 0
            self.upperCap.filled_volume = 0
        elif 0 < liquidVolume <= self.bottomCap.volume + self.cylinder.volume:
            self.bottomCap.filled_volume = self.bottomCap.volume
            self.cylinder.filled_volume = liquidVolume - self.bottomCap.volume
            self.upperCap.filled_volume = 0
        elif (
            0
            < liquidVolume
            <= self.bottomCap.volume + self.cylinder.volume + self.upperCap.volume
        ):
            self.bottomCap.filled_volume = self.bottomCap.volume
            self.cylinder.filled_volume = self.cylinder.volume
            self.upperCap.filled_volume = liquidVolume - (
                self.bottomCap.volume + self.cylinder.volume
            )
        else:
            raise ValueError(
                f"{self.name} tank liquid volume is either negative or greater than "
                "total tank volume. Check input data to make sure it is correct."
            )

    @functools.cached_property
    def volume(self):
        """Returns the total volume of the tank structure.

        Returns
        -------
        float
            Tank's total volume.
        """
        return self.bottomCap.volume + self.cylinder.volume + self.upperCap.volume

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
        Net mass flow rate is the mass flow rate entering the tank minus the
        mass flow rate exiting the tank, including liquids and gases. Positive
        is defined as a net mass flow rate entering the tank.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
            Positive is defined as a net mass flow rate entering the tank.
        """
        pass

    @abstractmethod
    def liquidVolume(self):
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

    @functools.cached_property
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
        # TODO: improve naming
        def centerOfMass(t):
            self.setTankFilling(t)

            bottomCapLiquidMass = self.liquid.density * self.bottomCap.filled_volume
            cylinderLiquidMass = self.liquid.density * self.cylinder.filled_volume
            upperCapLiquidMass = self.liquid.density * self.upperCap.filled_volume

            bottomCapGasMass = self.gas.density * self.bottomCap.empty_volume
            cylinderGasMass = self.gas.density * self.cylinder.empty_volume
            upperCapGasMass = self.gas.density * self.upperCap.empty_volume

            bottomCapMass = bottomCapLiquidMass + bottomCapGasMass
            cylinderMass = cylinderLiquidMass + cylinderGasMass
            upperCapMass = upperCapLiquidMass + upperCapGasMass

            totalMass = bottomCapMass + cylinderMass + upperCapMass

            bottomCapMassBalance = (
                bottomCapLiquidMass * self.bottomCap.filled_centroid
                + bottomCapGasMass * self.bottomCap.empty_centroid
            )
            cylinderMassBalance = cylinderLiquidMass * (
                self.cylinder.filled_centroid + self.bottomCap.height
            ) + cylinderGasMass * (self.cylinder.empty_centroid + self.bottomCap.height)
            upperCapMassBalance = upperCapLiquidMass * (
                self.upperCap.filled_centroid
                + self.bottomCap.height
                + self.cylinder.height
            ) + upperCapGasMass * (
                self.upperCap.empty_centroid
                + self.bottomCap.height
                + self.cylinder.height
            )

            centerOfMass = (
                bottomCapMassBalance + cylinderMassBalance + upperCapMassBalance
            ) / totalMass

            return centerOfMass

        centerOfMass = Function(
            centerOfMass, inputs="Time (s)", outputs="Center of Mass (m)"
        )

        return centerOfMass

    @functools.cached_property
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
        # TODO: compute inertia for non flat caps
        # TODO: compute inertia including gas mass
        def inertiaTensor(t):
            self.setTankFilling(t)

            cylinder_mass = self.cylinder.filled_volume * self.liquid.density

            # For a solid cylinder, ixx = iyy = mr²/4 + mh²/12
            inertiaI = cylinder_mass * (
                self.diameter**2 + self.cylinder.filled_height**2 / 12
            )

            # fluids considered inviscid so no shear resistance from torques in z axis
            inertiaZ = 0

            return inertiaI, inertiaZ

        inertiaTensor = Function(
            inertiaTensor, inputs="Time (s)", outputs="Inertia Tensor (kg m²)"
        )

        return inertiaTensor


class MassFlowRateBasedTank(Tank):
    def __init__(
        self,
        name,
        diameter,
        height,
        gas,
        liquid,
        initial_liquid_mass,
        initial_gas_mass,
        liquid_mass_flow_rate_in,
        gas_mass_flow_rate_in,
        liquid_mass_flow_rate_out,
        gas_mass_flow_rate_out,
        burn_out_time=300,
        bottomCap="flat",
        upperCap="flat",
    ):
        """A motor tank defined based on liquid and gas mass flow rates.

        Parameters
        ----------
        name : str
            Name of the tank.
        diameter : float
            Diameter of the tank in meters.
        height : float
            Height of the tank in meters.
        gas : Gas
            motor.Gas object.
        liquid : Liquid
            motor.Liquid object.
        initial_liquid_mass : float
            Initial mass of liquid in the tank in kg.
        initial_gas_mass : float
            Initial mass of gas in the tank in kg.
        liquid_mass_flow_rate_in : str, float, array_like or callable
            Liquid mass flow rate entering the tank as a function of time.
            All values should be positive.
            If string is given, it should be the filepath of a csv file
            containing the data. For more information, see Function.
        gas_mass_flow_rate_in : str, float, array_like or callable
            Gas mass flow rate entering the tank as a function of time.
            All values should be positive.
            If string is given, it should be the filepath of a csv file
            containing the data. For more information, see Function.
        liquid_mass_flow_rate_out : str, float, array_like or callable
            Liquid mass flow rate exiting the tank as a function of time.
            All values should be positive.
            If string is given, it should be the filepath of a csv file
            containing the data. For more information, see Function.
        gas_mass_flow_rate_out : str, float, array_like or callable
            Gas mass flow rate exiting the tank as a function of time.
            All values should be positive.
            If string is given, it should be the filepath of a csv file
            containing the data. For more information, see Function.
        burn_out_time : float, optional
            Time in seconds greater than motor burn out time to use for
            numerical integration stopping criteria. Default is 300.
        bottomCap : str
            Type of bottom cap. Options are "flat" and "spherical". Default is "flat".
        upperCap : str
            Type of upper cap. Options are "flat" and "spherical". Default is "flat".
        """
        super().__init__(name, diameter, height, gas, liquid, bottomCap, upperCap)

        self.initial_liquid_mass = initial_liquid_mass
        self.initial_gas_mass = initial_gas_mass
        self.burn_out_time = burn_out_time

        self.gas_mass_flow_rate_in = Function(
            gas_mass_flow_rate_in,
            "Time (s)",
            "Inlet Gas Propellant Mass Flow Rate (kg/s)",
            "linear",
            "zero",
        )

        self.gas_mass_flow_rate_out = Function(
            gas_mass_flow_rate_out,
            "Time (s)",
            "Outlet Gas Propellant Mass Flow Rate (kg/s)",
            "linear",
            "zero",
        )

        self.liquid_mass_flow_rate_in = Function(
            liquid_mass_flow_rate_in,
            "Time (s)",
            "Inlet Liquid Propellant Mass Flow Rate (kg/s)",
            "linear",
            "zero",
        )

        self.liquid_mass_flow_rate_out = Function(
            liquid_mass_flow_rate_out,
            "Time (s)",
            "Outlet Liquid Propellant Mass Flow Rate (kg/s)",
            "linear",
            "zero",
        )

    @functools.cached_property
    def netMassFlowRate(self):
        """Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate entering the tank minus the
        mass flow rate exiting the tank, including liquids and gases. Positive
        is defined as a net mass flow rate entering the tank.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
            Positive is defined as a net mass flow rate entering the tank.
        """
        self.liquid_net_mass_flow_rate = (
            self.liquid_mass_flow_rate_in - self.liquid_mass_flow_rate_out
        )

        self.liquid_net_mass_flow_rate.setOutputs(
            "Net Liquid Propellant Mass Flow Rate (kg/s)"
        )
        self.liquid_net_mass_flow_rate.setExtrapolation("zero")

        self.gas_net_mass_flow_rate = (
            self.gas_mass_flow_rate_in - self.gas_mass_flow_rate_out
        )

        self.gas_net_mass_flow_rate.setOutputs(
            "Net Gas Propellant Mass Flow Rate (kg/s)"
        )
        self.gas_net_mass_flow_rate.setExtrapolation("zero")

        self.net_mass_flow_rate = (
            self.liquid_net_mass_flow_rate + self.gas_net_mass_flow_rate
        )

        self.net_mass_flow_rate.setOutputs(
            "Net Propellant Mass Flow Rate Entering Tank (kg/s)"
        )
        self.net_mass_flow_rate.setExtrapolation("zero")
        self.net_mass_flow_rate.setInputs(["Time (s)"])

        return self.net_mass_flow_rate

    @functools.cached_property
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
        # Create an event function for solve_ivp

        def stopping_criteria(t, y):
            if y[0] / self.initial_liquid_mass > 0.95:
                return -1
            else:
                return self.netMassFlowRate(t)

        stopping_criteria.terminal = True

        # solve ODE's for liquid and gas masses
        sol = integrate.solve_ivp(
            lambda t, y: (
                self.liquid_net_mass_flow_rate(t),
                self.gas_net_mass_flow_rate(t),
            ),
            (0, self.burn_out_time),
            (self.initial_liquid_mass, self.initial_gas_mass),
            first_step=1e-3,
            vectorized=True,
            events=stopping_criteria,
            method="LSODA",
        )

        self.liquid_mass = Function(
            np.column_stack((sol.t, sol.y[0])),
            "Time (s)",
            "Liquid Propellant Mass In Tank (kg)",
        )

        self.gas_mass = Function(
            np.column_stack((sol.t, sol.y[1])),
            "Time (s)",
            "Gas Propellant Mass In Tank (kg)",
        )

        self.mass = self.liquid_mass + self.gas_mass
        self.mass.setOutputs("Total Propellant Mass In Tank (kg)")
        self.mass.setExtrapolation("constant")

        return self.mass

    @functools.cached_property
    def liquidVolume(self):
        """Returns the volume of liquid inside the tank as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Function
            Volume of liquid inside the tank as a function of time. Units in m^3.
        """
        self.liquid_volume = self.liquid_mass / self.liquid.density
        self.liquid_volume.setOutputs("Liquid Propellant Volume In Tank (m^3)")
        self.liquid_volume.setExtrapolation("constant")

        return self.liquid_volume


class UllageBasedTank(Tank):
    def __init__(
        self,
        name,
        diameter,
        height,
        gas,
        liquid,
        ullage,
        bottomCap="flat",
        upperCap="flat",
    ):
        """A motor tank defined based on its ullage volume, i.e., the volume
        of gas inside the tank.

        Parameters
        ----------
        name : str
            Name of the tank.
        diameter : float
            Diameter of the tank in meters.
        height : float
            Height of the tank in meters.
        gas : Gas
            motor.Gas object.
        liquid : Liquid
            motor.Liquid object.
        ullage : str, float, array_like or callable
            Ullage volume of the tank as a function of time. Units in m^3.
            If string is given, it should be the filepath of a csv file
            containing the data. For more information, see Function.
        bottomCap : str
            Type of bottom cap. Options are "flat" and "spherical".
            Default is "flat".
        upperCap : str
            Type of upper cap. Options are "flat" and "spherical".
            Default is "flat".
        """
        super().__init__(name, diameter, height, gas, liquid, bottomCap, upperCap)
        self.ullage = ullage

    @functools.cached_property
    def gasVolume(self):
        """
        Returns the volume of gas inside the tank.

        Returns
        -------
        Function
            Tank's gas volume as a function of time.
        """
        gasVolume = Function(
            self.ullage, "Time (s)", "Volume (m³)", extrapolation="constant"
        )
        gasVolume.setOutputs("Gas Propellant Volume In Tank (m³)")
        return gasVolume

    @functools.cached_property
    def liquidVolume(self):
        """Returns the volume of liquid inside the tank as a function
        of time.

        Returns
        -------
        Function
            Tank's liquid volume as a function of time.
        """
        liquidVolume = self.volume - self.gasVolume
        liquidVolume.setInputs("Time (s)")
        liquidVolume.setOutputs("Liquid Propellant Volume In Tank (m³)")
        return liquidVolume

    @functools.cached_property
    def gasMass(self):
        """
        Returns the total gas mass inside the tank.

        Returns
        -------
        Function
            Tank's gas mass as a function of time.
        """
        gasMass = self.gasVolume * self.gas.density
        gasMass.setInputs("Time (s)")
        gasMass.setOutputs("Gas Propellant Mass In Tank (kg)")
        return gasMass

    @functools.cached_property
    def liquidMass(self):
        """
        Returns the total liquid mass inside the tank.

        Returns
        -------
        Function
            Tank's liquid mass as a function of time.
        """
        liquidMass = self.liquidVolume * self.liquid.density
        liquidMass.setInputs("Time (s)")
        liquidMass.setOutputs("Liquid Propellant Mass In Tank (kg)")
        return liquidMass

    @functools.cached_property
    def mass(self):
        """Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        mass = self.gasMass + self.liquidMass
        mass.setInputs("Time (s)")
        mass.setOutputs("Total Propellant Mass In Tank (kg)")
        return mass

    @functools.cached_property
    def netMassFlowRate(self):
        """Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate exiting the tank minus the
        mass flow rate entering the tank, including liquids and gases.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        netMassFlowRate = Function(
            lambda t: self.mass.differentiate(t, dx=1e-6),
            "Time (s)",
            "Mass Flow Rate (kg/s)",
            extrapolation="zero",
        )
        netMassFlowRate.setOutputs("Net Tank Mass Flow Rate (kg/s)")
        return netMassFlowRate


class MassBasedTank(Tank):
    def __init__(
        self,
        name,
        diameter,
        height,
        bottomCap,
        upperCap,
        liquid_mass,
        gas_mass,
        liquid,
        gas,
    ):
        super().__init__(name, diameter, height, bottomCap, upperCap, gas, liquid)
        pass
