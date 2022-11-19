# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano and Pedro Henrique Marinho Bressan"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod
import functools

import numpy as np
from scipy import integrate

from rocketpy.Function import Function
from rocketpy.motors.TankGeometry import Disk, Cylinder, Hemisphere


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
        self.setGeometry()

    def setGeometry(self):
        """Sets the geometry of the tank based on the input parameters.

        Returns
        -------
        None
        """
        self.cylinder = Cylinder(self.diameter / 2, self.height)
        self.bottomCap = self.capMap.get(self.bottomCap)(
            self.diameter / 2, fill_direction="upwards"
        )
        self.upperCap = self.capMap.get(self.upperCap)(
            self.diameter / 2, fill_direction="downwards"
        )

    def evaluateTankState(self, t):
        """Gets the state of the tank at a given time: the volume of liquid and
        gases at each part of the tank (body and caps) as well as their masses.

        Parameters
        ----------
        t : float
            Time at which the state of the tank is to be calculated.

        Returns
        -------
        None
        """
        self.evaluateFilling(t)
        self.evaluateMassDistribution()
        self.evaluateCentroids()

    def evaluateFilling(self, t):
        """Calculates the distribution of volume of liquid and gases in the tank
        portions at a given time.

        Parameters
        ----------
        t : float
            Time at which the filling of the tank is to be calculated.

        Returns
        -------
        None
        """
        liquidVolume = self.liquidVolume.getValueOpt(t)

        if 0 <= liquidVolume < self.bottomCap.volume:
            self.bottomCap.filled_volume = liquidVolume
            self.cylinder.filled_volume = 0
            self.upperCap.filled_volume = 0
        elif 0 < liquidVolume <= self.bottomCap.volume + self.cylinder.volume:
            self.bottomCap.filled_volume = self.bottomCap.volume
            self.cylinder.filled_volume = liquidVolume - self.bottomCap.volume
            self.upperCap.filled_volume = 0
        elif 0 < liquidVolume <= self.volume:
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

    def evaluateMassDistribution(self):
        """Calculates the mass distribution of liquid and gases at the tank based
        on the volume distribution.

        Returns
        -------
        None
        """
        self.bottomCapLiquidMass = self.liquid.density * self.bottomCap.filled_volume
        self.cylinderLiquidMass = self.liquid.density * self.cylinder.filled_volume
        self.upperCapLiquidMass = self.liquid.density * self.upperCap.filled_volume

        self.bottomCapGasMass = self.gas.density * self.bottomCap.empty_volume
        self.cylinderGasMass = self.gas.density * self.cylinder.empty_volume
        self.upperCapGasMass = self.gas.density * self.upperCap.empty_volume

        self.bottomCapMass = self.bottomCapLiquidMass + self.bottomCapGasMass
        self.cylinderMass = self.cylinderLiquidMass + self.cylinderGasMass
        self.upperCapMass = self.upperCapLiquidMass + self.upperCapGasMass

    def evaluateCentroids(self):
        """Calculates the centroids of the liquid and gaseous portions of the tank
        based on the volume distributions.

        Returns
        -------
        None
        """
        self.bottomCapLiquidCentroid = self.bottomCap.filled_centroid
        self.bottomCapGasCentroid = self.bottomCap.empty_centroid
        self.bottomCapCentroid = self.bottomCap.centroid

        baseHeight = self.bottomCap.height
        self.cylinderLiquidCentroid = baseHeight + self.cylinder.filled_centroid
        self.cylinderGasCentroid = baseHeight + self.cylinder.empty_centroid
        self.cylinderCentroid = baseHeight + self.cylinder.centroid

        baseHeight = self.bottomCap.height + self.cylinder.height
        self.upperCapLiquidCentroid = baseHeight + self.upperCap.filled_centroid
        self.upperCapGasCentroid = baseHeight + self.upperCap.empty_centroid
        self.upperCapCentroid = baseHeight + self.upperCap.centroid

    def evaluateRelativeDistances(self, t):
        """Calculates the relative distances of the centroids of liquid and gaseous
        tank portions to its center of mass.

        Parameters
        ----------
        t : float
            Time at which the relative distances are to be calculated.

        Returns
        -------
        None
        """
        self.bottomCapRelDistLiq = self.centerOfMass(t) - self.bottomCapLiquidCentroid
        self.bottomCapRelDistGas = self.centerOfMass(t) - self.bottomCapGasCentroid
        self.cylinderRelDistLiq = self.centerOfMass(t) - self.cylinderLiquidCentroid
        self.cylinderRelDistGas = self.centerOfMass(t) - self.cylinderGasCentroid
        self.upperCapRelDistLiq = self.centerOfMass(t) - self.upperCapLiquidCentroid
        self.upperCapRelDistGas = self.centerOfMass(t) - self.upperCapGasCentroid

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

        def centerOfMass(t):
            self.evaluateTankState(t)

            bottomCapMassBalance = (
                self.bottomCapLiquidMass * self.bottomCapLiquidCentroid
                + self.bottomCapGasMass * self.bottomCapGasCentroid
            )
            cylinderMassBalance = (
                self.cylinderLiquidMass * self.cylinderLiquidCentroid
                + self.cylinderGasMass * self.cylinderGasCentroid
            )
            upperCapMassBalance = (
                self.upperCapLiquidMass * self.upperCapLiquidCentroid
                + self.upperCapGasMass * self.upperCapGasCentroid
            )

            centerOfMass = (
                bottomCapMassBalance + cylinderMassBalance + upperCapMassBalance
            ) / (self.bottomCapMass + self.cylinderMass + self.upperCapMass)

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

        def inertiaTensor(t):
            self.evaluateTankState(t)
            self.evaluateRelativeDistances(t)

            bottomCapGasInertia = self.bottomCap.empty_inertia
            cylinderGasInertia = self.cylinder.empty_inertia
            upperCapGasInertia = self.upperCap.empty_inertia

            bottomCapLiquidInertia = self.bottomCap.filled_inertia
            cylinderLiquidInertia = self.cylinder.filled_inertia
            upperCapLiquidInertia = self.upperCap.filled_inertia

            bottomCapInertia = (
                self.gas.density * bottomCapGasInertia[0]
                + self.bottomCapLiquidMass * self.bottomCapRelDistLiq**2
                + self.liquid.density * bottomCapLiquidInertia[0]
                + self.bottomCapGasMass * self.bottomCapRelDistGas**2
            )
            cylinderInertia = (
                self.gas.density * cylinderGasInertia[0]
                + self.cylinderLiquidMass * self.cylinderRelDistLiq**2
                + self.liquid.density * cylinderLiquidInertia[0]
                + self.cylinderGasMass * self.cylinderRelDistGas**2
            )
            upperCapInertia = (
                self.gas.density * upperCapGasInertia[0]
                + self.upperCapLiquidMass * self.upperCapRelDistLiq**2
                + self.liquid.density * upperCapLiquidInertia[0]
                + self.upperCapGasMass * self.upperCapRelDistGas**2
            )

            inertia_ixx = bottomCapInertia + cylinderInertia + upperCapInertia

            return inertia_ixx, 0

        inertiaTensor = Function(
            inertiaTensor,
            inputs="Time (s)",
            outputs="Inertia Tensor (kg m²)",
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
        gas,
        liquid,
        liquid_mass,
        gas_mass,
        bottomCap="flat",
        upperCap="flat",
    ):
        """
        A motor tank defined based on the masses of its liquid and gaseous
        portions.

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
        liquid_mass : str, float, array_like or callable
            Mass of the liquid portion of the tank as a function of time. Units in kg.
            If string is given, it should be the filepath of a csv file
            containing the data. For more information, see Function.
        gas_mass : str, float, array_like or callable
            Mass of the gaseous portion of the tank as a function of time. Units in kg.
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
        self.liquid_mass = liquid_mass
        self.gas_mass = gas_mass

    @functools.cached_property
    def gasMass(self):
        """
        Returns the total gas mass inside the tank.

        Returns
        -------
        Function
            Tank's gas mass as a function of time.
        """
        gasMass = Function(
            self.gas_mass, "Time (s)", "Gas Propellant Mass In Tank (kg)"
        )
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
        liquidMass = Function(
            self.liquid_mass, "Time (s)", "Liquid Propellant Mass In Tank (kg)"
        )
        return liquidMass

    @functools.cached_property
    def gasVolume(self):
        """
        Returns the volume of gas inside the tank.

        Returns
        -------
        Function
            Tank's gas volume as a function of time.
        """
        gasVolume = self.gasMass / self.gas.density
        gasVolume.setInputs("Time (s)")
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
        liquidVolume = self.liquidMass / self.liquid.density
        liquidVolume.setInputs("Time (s)")
        liquidVolume.setOutputs("Liquid Propellant Volume In Tank (m³)")
        return liquidVolume

    @functools.cached_property
    def mass(self):
        """Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        mass = self.liquidMass + self.gasMass
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
