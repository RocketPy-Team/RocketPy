# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

from rocketpy.Function import Function, funcify_method


class Tank(ABC):
    def __init__(self, name, tank_geometry, gas, liquid=0):
        self.name = name
        self.structure = tank_geometry
        self.gas = gas
        self.liquid = liquid

    @abstractmethod
    @funcify_method("time (s)", "mass (kg)")
    def mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        """
        Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate exiting the tank minus the
        mass flow rate entering the tank, including liquids and gases.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "liquid volume (m^3)")
    def liquidVolume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "gas volume (m³)")
    def gasVolume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "liquid height (m)")
    def liquidHeight(self):
        """
        Returns the liquid level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "gas height (m)")
    def gasHeight(self):
        """
        Returns the gas level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        pass

    @abstractmethod
    @funcify_method("time (s)", "liquid mass (kg)")
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time.

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

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        pass

    @funcify_method("time (s)", "center of mass of liquid (m)")
    def liquidCenterOfMass(self):
        """
        Returns the center of mass of the liquid portion of the tank
        as a function of time. This height is measured from the zero
        level of the tank geometry.

        Returns
        -------
        Function
            Center of mass of the liquid portion of the tank as a
            function of time.
        """

        def evaluate_liquid_com(t):
            mass_integrand = Function(
                lambda h: h * self.liquid.density * self.structure.area(h)
            )
            return mass_integrand.integral(
                self.structure.bottom, self.liquidHeight(t)
            ) / self.mass(t)

        return evaluate_liquid_com

    @funcify_method("time (s)", "center of mass of gas (m)")
    def gasCenterOfMass(self):
        """
        Returns the center of mass of the gas portion of the tank
        as a function of time. This height is measured from the zero
        level of the tank geometry.

        Returns
        -------
        Function
            Center of mass of the gas portion of the tank as a
            function of time.
        """

        def evaluate_gas_com(t):
            mass_integrand = Function(
                lambda h: h * self.gas.density * self.structure.area(h)
            )
            return mass_integrand.integral(
                self.liquidHeight(t), self.gasHeight(t)
            ) / self.mass(t)

        return evaluate_gas_com

    @funcify_method("time (s)", "center of mass (m)")
    def centerOfMass(self):
        """Returns the center of mass of the tank's fluids as a function of
        time. This height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Center of mass of the tank's fluids as a function of time.
        """
        return (
            self.liquidCenterOfMass * self.liquidMass
            + self.gasCenterOfMass * self.gasMass
        ) / (self.mass)

    @funcify_method("time (s)", "inertia tensor of liquid (kg*m^2)")
    def liquidInertiaTensor(self):
        """
        Returns the inertia tensor of the liquid portion of the tank
        as a function of time. The reference point is the center of
        mass of the tank.

        Returns
        -------
        Function
            Inertia tensor of the liquid portion of the tank as a
            function of time.
        """

        def evaluate_liquid_inertia(t):
            d_inertia = Function(
                lambda h: self.liquid.density
                * self.structure.area(h)
                * (h**2 + self.structure.radius(h) ** 2 / 4)
            )
            return d_inertia.integral(self.structure.bottom, self.liquidHeight(t))

        Ix = Function(evaluate_liquid_inertia)

        # Steiner theorem to account for center of mass
        Ix -= self.liquidMass * self.liquidCenterOfMass**2
        Ix += self.liquidMass * (self.liquidCenterOfMass - self.centerOfMass) ** 2
        return Ix

    @funcify_method("time (s)", "inertia tensor of liquid (kg*m^2)")
    def gasInertiaTensor(self):
        """
        Returns the inertia tensor of the gas portion of the tank
        as a function of time. The reference point is the center of
        mass of the tank.

        Returns
        -------
        Function
            Inertia tensor of the gas portion of the tank as a
            function of time.
        """

        def evaluate_gas_inertia(t):
            d_inertia = Function(
                lambda h: self.gas.density
                * self.structure.area(h)
                * (h**2 + self.structure.radius(h) ** 2 / 4)
            )
            inertiaint = d_inertia.integral(self.liquidHeight(t), self.gasHeight(t))
            return inertiaint

        Ix = Function(evaluate_gas_inertia)

        # Steiner theorem to account for center of mass
        Ix -= self.gasMass * self.gasCenterOfMass**2
        Ix += self.gasMass * (self.gasCenterOfMass - self.centerOfMass) ** 2
        return Ix

    @funcify_method("time (s)", "inertia tensor (kg*m^2)")
    def inertiaTensor(self):
        """
        Returns the inertia tensor of the tank's fluids as a function of
        time. The reference point is the center of mass of the tank.

        Returns
        -------
        Function
            Inertia tensor of the tank's fluids as a function of time.
        """
        return self.liquidInertiaTensor + self.gasInertiaTensor


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
            liquid_mass_flow_rate_in,
            inputs="Time",
            outputs="Mass Flow Rate",
            interpolation="linear",
            extrapolation="zero",
        )
        self.gas_mass_flow_rate_in = Function(
            gas_mass_flow_rate_in,
            inputs="Time",
            outputs="Mass Flow Rate",
            interpolation="linear",
            extrapolation="zero",
        )
        self.liquid_mass_flow_rate_out = Function(
            liquid_mass_flow_rate_out,
            inputs="Time",
            outputs="Mass Flow Rate",
            interpolation="linear",
            extrapolation="zero",
        )
        self.gas_mass_flow_rate_out = Function(
            gas_mass_flow_rate_out,
            inputs="Time",
            outputs="Mass Flow Rate",
            interpolation="linear",
            extrapolation="zero",
        )

    @funcify_method("time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("time (s)", "mass (kg)")
    def liquidMass(self):
        liquid_flow = self.netLiquidFlowRate.integralFunction()
        return self.initial_liquid_mass + liquid_flow

    @funcify_method("time (s)", "mass (kg)")
    def gasMass(self):
        gas_flow = self.netGasFlowRate.integralFunction()
        return self.initial_gas_mass + gas_flow

    @funcify_method("time (s)", "liquid mass flow rate (kg/s)")
    def netLiquidFlowRate(self):
        return self.liquid_mass_flow_rate_in - self.liquid_mass_flow_rate_out

    @funcify_method("time (s)", "gas mass flow rate (kg/s)")
    def netGasFlowRate(self):
        return self.gas_mass_flow_rate_in - self.gas_mass_flow_rate_out

    @funcify_method("time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.netLiquidFlowRate + self.netGasFlowRate

    @funcify_method("time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.liquidMass / self.liquid.density

    @funcify_method("time (s)", "volume (m³)")
    def gasVolume(self):
        return self.gasMass / self.gas.density

    @funcify_method("time (s)", "height (m)")
    def liquidHeight(self):
        liquid_volume = self.liquidVolume
        inverse_volume = self.structure.inverse_volume
        return inverse_volume.compose(liquid_volume)

    @funcify_method("time (s)", "height (m)")
    def gasHeight(self):
        fluid_volume = self.gasVolume + self.liquidVolume
        inverse_volume = self.structure.inverse_volume
        return inverse_volume.compose(fluid_volume)


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
        self.ullage = Function(ullage, "Time", "Volume", "linear", "constant")

    @funcify_method("time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.mass.derivativeFunction()

    @funcify_method("time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.structure.total_volume.item - self.ullage

    @funcify_method("time (s)", "volume (m³)")
    def gasVolume(self):
        return self.ullage

    @funcify_method("time (s)", "mass (kg)")
    def gasMass(self):
        return self.gasVolume * self.gas.density

    @funcify_method("time (s)", "mass (kg)")
    def liquidMass(self):
        return self.liquidVolume * self.liquid.density

    @funcify_method("time (s)", "height (m)")
    def liquidHeight(self):
        liquid_volume = self.liquidVolume
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(liquid_volume)

    @funcify_method("time (s)", "height (m)")
    def gasHeight(self):
        fluid_volume = self.gasVolume + self.liquidVolume
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(fluid_volume)


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
        self.liquid_height = Function(
            liquid_height, "Time", "Volume", "linear", "constant"
        )

    @funcify_method("time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.mass.derivativeFunction()

    @funcify_method("time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.structure.volume.compose(self.liquidHeight)

    @funcify_method("time (s)", "volume (m³)")
    def gasVolume(self):
        return self.structure.total_volume.item() - self.liquidVolume

    @funcify_method("time (s)", "height (m)")
    def liquidHeight(self):
        return self.liquid_height

    @funcify_method("time (s)", "mass (kg)")
    def gasMass(self):
        return self.gasVolume * self.gas.density

    @funcify_method("time (s)", "mass (kg)")
    def liquidMass(self):
        return self.liquidVolume * self.liquid.density

    @funcify_method("time (s)", "height (m)")
    def gasHeight(self):
        fluid_volume = self.gasVolume + self.liquidVolume
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(fluid_volume)


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
        self.liquid_mass = Function(liquid_mass, "Time", "Mass", "linear", "constant")
        self.gas_mass = Function(gas_mass, "Time", "Mass", "linear", "constant")

    @funcify_method("time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.mass.derivativeFunction()

    @funcify_method("time (s)", "mass (kg)")
    def liquidMass(self):
        return self.liquid_mass

    @funcify_method("time (s)", "mass (kg)")
    def gasMass(self):
        return self.gas_mass

    @funcify_method("time (s)", "volume (m³)")
    def gasVolume(self):
        return self.gasMass / self.gas.density

    @funcify_method("time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.liquidMass / self.liquid.density

    @funcify_method("time (s)", "height (m)")
    def liquidHeight(self):
        liquid_volume = self.liquidVolume
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(liquid_volume)

    @funcify_method("time (s)", "height (m)")
    def gasHeight(self):
        fluid_volume = self.gasVolume + self.liquidVolume
        inverse_volume = self.structure.volume.inverseFunction()
        return inverse_volume.compose(fluid_volume)
