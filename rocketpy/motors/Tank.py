# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

from rocketpy.Function import Function, funcify_method


class Tank(ABC):
    """Abstract Tank class that defines a tank object for a rocket motor, so
    that it evaluates useful properties of the tank and its fluids, such as
    mass, volume, fluid flow rate, center of mass, etc.
    """

    def __init__(self, name, geometry, flux_time, gas, liquid, discretize=100):
        """Initialize Tank class.

        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float, optional
            Tank flux time in seconds. It is the time range in which the tank
            flux is being analyzed. In general, during this time, the tank is
            being filled or emptied.
            If a float is given, the flux time is assumed to be between 0 and the
            given float, in seconds. If a tuple of float is given, the flux time
            is assumed to be between the first and second elements of the tuple.
        gas : Fluid
            Gas inside the tank as a Fluid object.
        liquid : Fluid
            Liquid inside the tank as a Fluid object.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the input
            already has a appropriate discretization, this parameter
            must be set to None. The default is 100.
        """
        self.name = name
        self.geometry = geometry
        self.flux_time = flux_time
        self.gas = gas
        self.liquid = liquid
        self.discretize = discretize

    @property
    def flux_time(self):
        """Returns the start and final times of the tank flux.

        Returns
        -------
        tuple
            Tuple containing start and final times of the tank flux.
        """
        return self._flux_time

    @flux_time.setter
    def flux_time(self, flux_time):
        """Sets the start and final times of the tank flux.

        Parameters
        ----------
        flux_time : tuple
            Tuple containing start and final times of the tank flux.
        """
        if isinstance(flux_time, (int, float)):
            self._flux_time = (0, flux_time)
        elif isinstance(flux_time, (list, tuple)):
            if len(flux_time) == 1:
                self._flux_time = (0, flux_time[0])
            elif len(flux_time) == 2:
                self._flux_time = flux_time
            else:
                raise ValueError("flux_time must be a list or tuple of length 1 or 2.")

    @property
    @abstractmethod
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

    @property
    @abstractmethod
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

    @property
    @abstractmethod
    def liquidVolume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        pass

    @property
    @abstractmethod
    def gasVolume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        pass

    @property
    @abstractmethod
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

    @property
    @abstractmethod
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

    @property
    @abstractmethod
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        pass

    @property
    @abstractmethod
    def gasMass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        pass

    @funcify_method("Time (s)", "center of mass of liquid (m)")
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
        moment = self.geometry.volume_moment(
            self.geometry.bottom, self.liquidHeight.max
        )
        liquid_moment = moment @ self.liquidHeight
        centroid = liquid_moment / self.liquidVolume

        # Check for zero liquid volume
        bound_volume = self.liquidVolume < 1e-4 * self.geometry.total_volume
        if bound_volume.any():
            # TODO: pending Function setter impl.
            centroid.yArray[bound_volume] = self.geometry.bottom
            centroid.setInterpolation()
            centroid.setExtrapolation()

        return centroid

    @funcify_method("Time (s)", "center of mass of gas (m)")
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
        moment = self.geometry.volume_moment(self.geometry.bottom, self.gasHeight.max)
        upper_moment = moment @ self.gasHeight
        lower_moment = moment @ self.liquidHeight
        centroid = (upper_moment - lower_moment) / self.gasVolume

        # Check for zero gas volume
        bound_volume = self.gasVolume < 1e-4 * self.geometry.total_volume
        if bound_volume.any():
            # TODO: pending Function setter impl.
            centroid.yArray[bound_volume] = self.liquidHeight.yArray[bound_volume]
            centroid.setInterpolation()
            centroid.setExtrapolation()

        return centroid

    @funcify_method("Time (s)", "center of mass (m)")
    def centerOfMass(self):
        """Returns the center of mass of the tank's fluids as a function of
        time. This height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Center of mass of the tank's fluids as a function of time.
        """
        centerOfMass = (
            self.liquidCenterOfMass * self.liquidMass
            + self.gasCenterOfMass * self.gasMass
        ) / (self.mass)

        # Check for zero mass
        bound_mass = self.mass < 0.001 * self.geometry.total_volume * self.gas.density
        if bound_mass.any():
            # TODO: pending Function setter impl.
            centerOfMass.yArray[bound_mass] = self.geometry.bottom
            centerOfMass.setInterpolation()
            centerOfMass.setExtrapolation()

        return centerOfMass

    @funcify_method("Time (s)", "inertia tensor of liquid (kg*m^2)")
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
        Ix_volume = self.geometry.Ix_volume(self.geometry.bottom, self.liquidHeight.max)
        Ix_volume = Ix_volume @ self.liquidHeight

        # Steiner theorem to account for center of mass
        Ix_volume -= self.liquidVolume * self.liquidCenterOfMass**2
        Ix_volume += (
            self.liquidVolume * (self.liquidCenterOfMass - self.centerOfMass) ** 2
        )

        return self.liquid.density * Ix_volume

    @funcify_method("Time (s)", "inertia tensor of gas (kg*m^2)")
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
        Ix_volume = self.geometry.Ix_volume(self.geometry.bottom, self.gasHeight.max)
        lower_inertia_volume = Ix_volume @ self.liquidHeight
        upper_inertia_volume = Ix_volume @ self.gasHeight
        inertia_volume = upper_inertia_volume - lower_inertia_volume

        # Steiner theorem to account for center of mass
        inertia_volume -= self.gasVolume * self.gasCenterOfMass**2
        inertia_volume += (
            self.gasVolume * (self.gasCenterOfMass - self.centerOfMass) ** 2
        )

        return self.gas.density * inertia_volume

    @funcify_method("Time (s)", "inertia tensor (kg*m^2)")
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
        geometry,
        flux_time,
        liquid,
        gas,
        initial_liquid_mass,
        initial_gas_mass,
        liquid_mass_flow_rate_in,
        gas_mass_flow_rate_in,
        liquid_mass_flow_rate_out,
        gas_mass_flow_rate_out,
        discretize=100,
    ):
        super().__init__(name, geometry, flux_time, gas, liquid, discretize)
        self.initial_liquid_mass = initial_liquid_mass
        self.initial_gas_mass = initial_gas_mass

        # Define flow rates
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

        # Discretize input flow if needed
        self.discretize_flow() if discretize else None

    @funcify_method("Time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "mass (kg)")
    def liquidMass(self):
        liquid_flow = self.netLiquidFlowRate.integralFunction()
        liquidMass = self.initial_liquid_mass + liquid_flow
        if (liquidMass < 0).any():
            raise ValueError(f"The tank {self.name} is underfilled.")
        return liquidMass

    @funcify_method("Time (s)", "mass (kg)")
    def gasMass(self):
        gas_flow = self.netGasFlowRate.integralFunction()
        gasMass = self.initial_gas_mass + gas_flow
        if (gasMass < 0).any():
            raise ValueError(f"The tank {self.name} is underfilled.")
        return gasMass

    @funcify_method("Time (s)", "liquid mass flow rate (kg/s)", extrapolation="zero")
    def netLiquidFlowRate(self):
        return self.liquid_mass_flow_rate_in - self.liquid_mass_flow_rate_out

    @funcify_method("Time (s)", "gas mass flow rate (kg/s)", extrapolation="zero")
    def netGasFlowRate(self):
        return self.gas_mass_flow_rate_in - self.gas_mass_flow_rate_out

    @funcify_method("Time (s)", "mass flow rate (kg/s)", extrapolation="zero")
    def netMassFlowRate(self):
        return self.netLiquidFlowRate + self.netGasFlowRate

    @funcify_method("Time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.liquidMass / self.liquid.density

    @funcify_method("Time (s)", "volume (m³)")
    def gasVolume(self):
        return self.gasMass / self.gas.density

    @funcify_method("Time (s)", "height (m)")
    def liquidHeight(self):
        return self.geometry.inverse_volume.compose(self.liquidVolume)

    @funcify_method("Time (s)", "height (m)")
    def gasHeight(self):
        fluid_volume = self.gasVolume + self.liquidVolume
        gasHeight = self.geometry.inverse_volume.compose(fluid_volume)
        if (gasHeight > self.geometry.top).any():
            raise ValueError(f"The tank {self.name} is overfilled.")
        return gasHeight

    def discretize_flow(self):
        self.liquid_mass_flow_rate_in.setDiscrete(*self.flux_time, self.discretize)
        self.gas_mass_flow_rate_in.setDiscrete(*self.flux_time, self.discretize)
        self.liquid_mass_flow_rate_out.setDiscrete(*self.flux_time, self.discretize)
        self.gas_mass_flow_rate_out.setDiscrete(*self.flux_time, self.discretize)


class UllageBasedTank(Tank):
    def __init__(
        self,
        name,
        geometry,
        flux_time,
        liquid,
        gas,
        ullage,
        discretize=100,
    ):
        super().__init__(name, geometry, flux_time, gas, liquid, discretize)

        # Define ullage
        self.ullage = Function(ullage, "Time (s)", "Volume (m³)", "linear")

        # Discretize input if needed
        self.discretize_ullage() if discretize else None

        # Check if the ullage is within bounds
        if (self.ullage > self.geometry.total_volume).any() or (self.ullage < 0).any():
            raise ValueError("The ullage volume is out of bounds.")

    @funcify_method("Time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.mass.derivativeFunction()

    @funcify_method("Time (s)", "volume (m³)")
    def liquidVolume(self):
        return -(self.ullage - self.geometry.total_volume)

    @funcify_method("Time (s)", "volume (m³)")
    def gasVolume(self):
        return self.ullage

    @funcify_method("Time (s)", "mass (kg)")
    def gasMass(self):
        return self.gasVolume * self.gas.density

    @funcify_method("Time (s)", "mass (kg)")
    def liquidMass(self):
        return self.liquidVolume * self.liquid.density

    @funcify_method("Time (s)", "height (m)")
    def liquidHeight(self):
        return self.geometry.inverse_volume.compose(self.liquidVolume)

    @funcify_method("Time (s)", "height (m)")
    def gasHeight(self):
        return self.geometry.top

    def discretize_ullage(self):
        self.ullage.setDiscrete(*self.flux_time, self.discretize)


class LevelBasedTank(Tank):
    def __init__(
        self,
        name,
        geometry,
        flux_time,
        liquid,
        gas,
        liquid_height,
        discretize=100,
    ):
        super().__init__(name, geometry, flux_time, gas, liquid, discretize)

        # Define liquid height
        self.liquid_height = Function(
            liquid_height, "Time (s)", "volume (m³)", "linear"
        )

        # Discretize input if needed
        self.discretize_liquid_height() if discretize else None

        # Check if the liquid level is within bounds
        if (self.liquid_height > self.geometry.top).any() or (
            self.liquid_height < self.geometry.bottom
        ).any():
            raise ValueError("The liquid level is out of bounds.")

    @funcify_method("Time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.mass.derivativeFunction()

    @funcify_method("Time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.geometry.volume.compose(self.liquidHeight)

    @funcify_method("Time (s)", "volume (m³)")
    def gasVolume(self):
        return self.geometry.total_volume - self.liquidVolume

    @funcify_method("Time (s)", "height (m)")
    def liquidHeight(self):
        return self.liquid_height

    @funcify_method("Time (s)", "mass (kg)")
    def gasMass(self):
        return self.gasVolume * self.gas.density

    @funcify_method("Time (s)", "mass (kg)")
    def liquidMass(self):
        return self.liquidVolume * self.liquid.density

    @funcify_method("Time (s)", "height (m)")
    def gasHeight(self):
        return self.geometry.top

    def discretize_liquid_height(self):
        self.liquid_height.setDiscrete(*self.flux_time, self.discretize)


class MassBasedTank(Tank):
    def __init__(
        self,
        name,
        geometry,
        flux_time,
        liquid,
        gas,
        liquid_mass,
        gas_mass,
        discretize=100,
    ):
        super().__init__(name, geometry, flux_time, gas, liquid, discretize)

        # Define fluid masses
        self.liquid_mass = Function(liquid_mass, "Time (s)", "mass (kg)", "linear")
        self.gas_mass = Function(gas_mass, "Time (s)", "mass (kg)", "linear")

        # Discretize input if needed
        self.discretize_masses() if discretize else None

    @funcify_method("Time (s)", "mass (kg)")
    def mass(self):
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "mass flow rate (kg/s)")
    def netMassFlowRate(self):
        return self.mass.derivativeFunction()

    @funcify_method("Time (s)", "mass (kg)")
    def liquidMass(self):
        return self.liquid_mass

    @funcify_method("Time (s)", "mass (kg)")
    def gasMass(self):
        return self.gas_mass

    @funcify_method("Time (s)", "volume (m³)")
    def gasVolume(self):
        return self.gasMass / self.gas.density

    @funcify_method("Time (s)", "volume (m³)")
    def liquidVolume(self):
        return self.liquidMass / self.liquid.density

    @funcify_method("Time (s)", "height (m)")
    def liquidHeight(self):
        return self.geometry.inverse_volume.compose(self.liquidVolume)

    @funcify_method("Time (s)", "height (m)")
    def gasHeight(self):
        fluid_volume = self.gasVolume + self.liquidVolume
        gasHeight = self.geometry.inverse_volume.compose(fluid_volume)
        if (gasHeight > self.geometry.top).any():
            raise ValueError(f"The tank {self.name} is overfilled.")
        return gasHeight

    def discretize_masses(self):
        self.liquid_mass.setDiscrete(*self.flux_time, self.discretize)
        self.gas_mass.setDiscrete(*self.flux_time, self.discretize)
