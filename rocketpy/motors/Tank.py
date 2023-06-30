# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

from rocketpy.Function import Function, funcify_method
from rocketpy.tools import tuple_handler


class Tank(ABC):
    """Abstract Tank class that defines a tank object for a rocket motor, so
    that it evaluates useful properties of the tank and its fluids, such as
    mass, volume, fluid flow rate, center of mass, etc.

    Attributes
    ----------

        Tank.name : str
            Name of the tank.
        Tank.geometry : rocketpy.motors.TankGeometry
            Geometry of the tank.
        Tank.flux_time : float, tuple of float, optional
            Tank flux time in seconds.
        Tank.liquid : rocketpy.motors.Fluid
            Liquid inside the tank as a Fluid object.
        Tank.gas : rocketpy.motors.Fluid
            Gas inside the tank as a Fluid object.
        Tank.discretize : int, optional
            Number of points to discretize fluid inputs.

    Properties
    ----------

        Tank.mass : rocketpy.Function
            Total mass of liquid and gases in kg inside the tank as a function
            of time.
        Tank.netMassFlowRate : rocketpy.Function
            Net mass flow rate of the tank in kg/s as a function of time, also
            understood as time derivative of the tank mass.
        Tank.liquidVolume : rocketpy.Function
            Volume of the liquid inside the Tank in m^3 as a function of time.
        Tank.gasVolume : rocketpy.Function
            Volume of the gas inside the Tank in m^3 as a function of time.
        Tank.liquidHeight : rocketpy.Function
            Height of the liquid inside the Tank in m as a function of time.
            The zero level reference is the same as set in Tank.geometry.
        Tank.gasHeight : rocketpy.Function
            Height of the gas inside the Tank in m as a function of time.
            The zero level reference is the same as set in Tank.geometry.
        Tank.liquidMass : rocketpy.Function
            Mass of the liquid inside the Tank in kg as a function of time.
        Tank.gasMass : rocketpy.Function
            Mass of the gas inside the Tank in kg as a function of time.
        Tank.liquidCenterOfMass : rocketpy.Function
            Center of mass of the liquid inside the Tank in m as a function of
            time. The zero level reference is the same as set in Tank.geometry.
        Tank.gasCenterOfMass : rocketpy.Function
            Center of mass of the gas inside the Tank in m as a function of
            time. The zero level reference is the same as set in Tank.geometry.
        Tank.centerOfMass : rocketpy.Function
            Center of mass of liquid and gas (i.e. propellant) inside the Tank
            in m as a function of time. The zero level reference is the same as
            set in Tank.geometry.
        Tank.liquidInertia : rocketpy.Function
            The inertia of the liquid inside the Tank in kg*m^2 as a function
            of time around a perpendicular axis to the Tank symmetry axis. The
            reference point is the Tank center of mass.
        Tank.gasInertia : rocketpy.Function
            The inertia of the gas inside the Tank in kg*m^2 as a function of
            time around a perpendicular axis to the Tank symmetry axis. The
            reference point is the Tank center of mass.
        Tank.inertia : rocketpy.Function
            The inertia of the liquid and gas (i.e. propellant) inside the Tank
            in kg*m^2 as a function of time around a perpendicular axis to the
            Tank symmetry axis. The reference point is the Tank center of mass.
    """

    def __init__(self, name, geometry, flux_time, liquid, gas, discretize=100):
        """Initialize Tank class.

        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : rocketpy.motors.TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float, optional
            Tank flux time in seconds. It is the time range in which the tank
            flux is being analyzed. In general, during this time, the tank is
            being filled or emptied.
            If a float is given, the flux time is assumed to be between 0 and the
            given float, in seconds. If a tuple of float is given, the flux time
            is assumed to be between the first and second elements of the tuple.
        gas : rocketpy.motors.Fluid
            Gas inside the tank as a Fluid object.
        liquid : rocketpy.motors.Fluid
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
        self._flux_time = tuple_handler(flux_time)

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
        Net mass flow rate is the mass flow rate entering the tank minus the
        mass flow rate exiting the tank, including liquids and gases.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        pass

    @property
    @abstractmethod
    def fluidVolume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
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

    @funcify_method("Time (s)", "Center of mass of liquid (m)")
    def liquidCenterOfMass(self):
        """
        Returns the center of mass of the liquid portion of the tank
        as a function of time. This height is measured from the zero
        level of the tank geometry.

        Returns
        -------
        rocketpy.Function
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

    @funcify_method("Time (s)", "Center of mass of gas (m)")
    def gasCenterOfMass(self):
        """
        Returns the center of mass of the gas portion of the tank
        as a function of time. This height is measured from the zero
        level of the tank geometry.

        Returns
        -------
        rocketpy.Function
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

    @funcify_method("Time (s)", "Center of mass of Fluid (m)")
    def centerOfMass(self):
        """Returns the center of mass of the tank's fluids as a function of
        time. This height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        rocketpy.Function
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

    @funcify_method("Time (s)", "Inertia tensor of liquid (kg*m²)")
    def liquidInertia(self):
        """
        Returns the inertia tensor of the liquid portion of the tank
        as a function of time. The reference point is the center of
        mass of the tank.

        Returns
        -------
        rocketpy.Function
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
    def gasInertia(self):
        """
        Returns the inertia tensor of the gas portion of the tank
        as a function of time. The reference point is the center of
        mass of the tank.

        Returns
        -------
        rocketpy.Function
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
    def inertia(self):
        """
        Returns the inertia tensor of the tank's fluids as a function of
        time. The reference point is the center of mass of the tank.

        Returns
        -------
        Function
            Inertia tensor of the tank's fluids as a function of time.
        """
        return self.liquidInertia + self.gasInertia


class MassFlowRateBasedTank(Tank):
    """Class to define a tank based on mass flow rates inputs. This class
    inherits from the Tank class. See the Tank class for more information
    on its attributes and methods.
    """

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
        """Initializes the MassFlowRateBasedTank class.

        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : rocketpy.geometry.TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float, optional
            Tank flux time in seconds. It is the time range in which the tank
            flux is being analyzed. In general, during this time, the tank is
            being filled or emptied.
            If a float is given, the flux time is assumed to be between 0 and the
            given float, in seconds. If a tuple of float is given, the flux time
            is assumed to be between the first and second elements of the tuple.
        liquid : rocketpy.motors.Fluid
            Liquid inside the tank as a Fluid object.
        gas : rocketpy.motors.Fluid
            Gas inside the tank as a Fluid object.
        initial_liquid_mass : float
            Initial liquid mass in the tank in kg.
        initial_gas_mass : float
            Initial gas mass in the tank in kg.
        liquid_mass_flow_rate_in : int, float, callable, string, array
            Liquid mass flow rate into the tank in kg/s. Always positive.
            It must be a valid rocketpy.Function source.
        gas_mass_flow_rate_in : int, float, callable, string, array
            Gas mass flow rate into the tank in kg/s. Always positive.
            It must be a valid rocketpy.Function source.
        liquid_mass_flow_rate_out : int, float, callable, string, array
            Liquid mass flow rate out of the tank in kg/s. Always positive.
            It must be a valid rocketpy.Function source.
        gas_mass_flow_rate_out : int, float, callable, string, array
            Gas mass flow rate out of the tank in kg/s. Always positive.
            It must be a valid rocketpy.Function source.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the mass flow
            rate inputs are uniformly discretized (have the same time steps)
            this parameter may be set to None. Otherwise, an uniform
            discretization will be applied based on the discretize value.
            The default is 100.
        """
        super().__init__(name, geometry, flux_time, liquid, gas, discretize)
        self.initial_liquid_mass = initial_liquid_mass
        self.initial_gas_mass = initial_gas_mass

        # Define flow rates
        self.liquid_mass_flow_rate_in = Function(
            liquid_mass_flow_rate_in,
            inputs="Time (s)",
            outputs="Mass Flow Rate (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        self.gas_mass_flow_rate_in = Function(
            gas_mass_flow_rate_in,
            inputs="Time (s)",
            outputs="Mass Flow Rate (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        self.liquid_mass_flow_rate_out = Function(
            liquid_mass_flow_rate_out,
            inputs="Time (s)",
            outputs="Mass Flow Rate (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        self.gas_mass_flow_rate_out = Function(
            gas_mass_flow_rate_out,
            inputs="Time (s)",
            outputs="Mass Flow Rate (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )

        # Discretize input flow if needed
        self.discretize_flow() if discretize else None
        return None

    @funcify_method("Time (s)", "Mass (kg)")
    def mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "Mass (kg)")
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time by integrating
        the liquid mass flow rate.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        liquid_flow = self.netLiquidFlowRate.integralFunction()
        liquidMass = self.initial_liquid_mass + liquid_flow
        if (liquidMass < 0).any():
            raise ValueError(f"The tank {self.name} is underfilled.")
        return liquidMass

    @funcify_method("Time (s)", "Mass (kg)")
    def gasMass(self):
        """
        Returns the mass of the gas as a function of time by integrating
        the gas mass flow rate.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        gas_flow = self.netGasFlowRate.integralFunction()
        gasMass = self.initial_gas_mass + gas_flow
        if (gasMass < 0).any():
            raise ValueError(f"The tank {self.name} is underfilled.")
        return gasMass

    @funcify_method("Time (s)", "liquid mass flow rate (kg/s)", extrapolation="zero")
    def netLiquidFlowRate(self):
        """
        Returns the net mass flow rate of liquid as a function of time.
        It is computed as the liquid mass flow rate entering the tank
        minus the liquid mass flow rate exiting the tank.

        Returns
        -------
        Function
            Net liquid mass flow rate of the tank as a function of time.
        """
        return self.liquid_mass_flow_rate_in - self.liquid_mass_flow_rate_out

    @funcify_method("Time (s)", "gas mass flow rate (kg/s)", extrapolation="zero")
    def netGasFlowRate(self):
        """
        Returns the net mass flow rate of gas as a function of time.
        It is computed as the gas mass flow rate entering the tank
        minus the gas mass flow rate exiting the tank.

        Returns
        -------
        Function
            Net gas mass flow rate of the tank as a function of time.
        """
        return self.gas_mass_flow_rate_in - self.gas_mass_flow_rate_out

    @funcify_method("Time (s)", "mass flow rate (kg/s)", extrapolation="zero")
    def netMassFlowRate(self):
        """
        Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate entering the tank minus the
        mass flow rate exiting the tank, including liquids and gases.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.netLiquidFlowRate + self.netGasFlowRate

    @funcify_method("Time (s)", "Volume (m³)")
    def fluidVolume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        return self.liquidVolume + self.gasVolume

    @funcify_method("Time (s)", "Volume (m³)")
    def liquidVolume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return self.liquidMass / self.liquid.density

    @funcify_method("Time (s)", "Volume (m³)")
    def gasVolume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.gasMass / self.gas.density

    @funcify_method("Time (s)", "Height (m)")
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
        return self.geometry.inverse_volume.compose(self.liquidVolume)

    @funcify_method("Time (s)", "Height (m)")
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
        fluid_volume = self.gasVolume + self.liquidVolume
        gasHeight = self.geometry.inverse_volume.compose(fluid_volume)
        if (gasHeight > self.geometry.top).any():
            raise ValueError(f"The tank {self.name} is overfilled.")
        return gasHeight

    def discretize_flow(self):
        """Discretizes the mass flow rate inputs according to the flux time and
        the discretize parameter.
        """
        self.liquid_mass_flow_rate_in.setDiscrete(*self.flux_time, self.discretize)
        self.gas_mass_flow_rate_in.setDiscrete(*self.flux_time, self.discretize)
        self.liquid_mass_flow_rate_out.setDiscrete(*self.flux_time, self.discretize)
        self.gas_mass_flow_rate_out.setDiscrete(*self.flux_time, self.discretize)


class UllageBasedTank(Tank):
    """Class to define a tank whose flow is described by ullage volume, i.e.,
    the volume of the tank that is not occupied by the liquid. It assumes that
    the ullage volume is uniformly filled by the gas. This class inherits from
    the Tank class. See the Tank class for more information on its attributes
    and methods.
    """

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
        """
        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : rocketpy.geometry.TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float, optional
            Tank flux time in seconds. It is the time range in which the tank
            flux is being analyzed. In general, during this time, the tank is
            being filled or emptied.
            If a float is given, the flux time is assumed to be between 0 and the
            given float, in seconds. If a tuple of float is given, the flux time
            is assumed to be between the first and second elements of the tuple.
        liquid : rocketpy.motors.Fluid
            Liquid inside the tank as a Fluid object.
        gas : rocketpy.motors.Fluid
            Gas inside the tank as a Fluid object.
        ullage : int, float, callable, string, array
            Ullage volume as a function of time in m^3. Also understood as the
            volume of the Tank that is not occupied by liquid. Must be a valid
            rocketpy.Function source.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the ullage input is
            already discretized this parameter may be set to None. Otherwise,
            an uniform discretization will be applied based on the discretize
            value.
            The default is 100.
        """
        super().__init__(name, geometry, flux_time, liquid, gas, discretize)

        # Define ullage
        self.ullage = Function(ullage, "Time (s)", "Volume (m³)", "linear")

        # Discretize input if needed
        self.discretize_ullage() if discretize else None

        # Check if the ullage is within bounds
        if (self.ullage > self.geometry.total_volume).any() or (self.ullage < 0).any():
            raise ValueError("The ullage volume is out of bounds.")

    @funcify_method("Time (s)", "Mass (kg)")
    def mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "Mass flow rate (kg/s)")
    def netMassFlowRate(self):
        """
        Returns the net mass flow rate of the tank as a function of time by
        taking the derivative of the mass function.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.mass.derivativeFunction()

    @funcify_method("Time (s)", "Volume (m³)")
    def fluidVolume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        return self.geometry.total_volume

    @funcify_method("Time (s)", "Volume (m³)")
    def liquidVolume(self):
        """
        Returns the volume of the liquid as a function of time. The
        volume is computed by subtracting the ullage volume from the
        total volume of the tank.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return -(self.ullage - self.geometry.total_volume)

    @funcify_method("Time (s)", "Volume (m³)")
    def gasVolume(self):
        """
        Returns the volume of the gas as a function of time. From the
        Tank assumptions the gas volume is equal to the ullage volume.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.ullage

    @funcify_method("Time (s)", "Mass (kg)")
    def gasMass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        return self.gasVolume * self.gas.density

    @funcify_method("Time (s)", "Mass (kg)")
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        return self.liquidVolume * self.liquid.density

    @funcify_method("Time (s)", "Height (m)")
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
        return self.geometry.inverse_volume.compose(self.liquidVolume)

    @funcify_method("Time (s)", "Height (m)", "linear")
    def gasHeight(self):
        """
        Returns the gas level as a function of time. This
        height is measured from the zero level of the tank
        geometry. Since the gas is assumed to be uniformly
        distributed in the ullage, the gas height is constant
        and equal to the top of the tank geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        return Function(self.geometry.top).setDiscreteBasedOnModel(self.gasVolume)

    def discretize_ullage(self):
        """Discretizes the ullage input according to the flux time and the
        discretize parameter."""
        self.ullage.setDiscrete(*self.flux_time, self.discretize)


class LevelBasedTank(Tank):
    """Class to define a tank whose flow is described by liquid level, i.e.,
    the height of the liquid inside the tank. It assumes that the volume
    above the liquid level is uniformly occupied by gas. This class inherits
    from the Tank class. See the Tank class for more information on its
    attributes and methods.
    """

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
        """
        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : rocketpy.geometry.TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float, optional
            Tank flux time in seconds. It is the time range in which the tank
            flux is being analyzed. In general, during this time, the tank is
            being filled or emptied.
            If a float is given, the flux time is assumed to be between 0 and the
            given float, in seconds. If a tuple of float is given, the flux time
            is assumed to be between the first and second elements of the tuple.
        liquid : rocketpy.motors.Fluid
            Liquid inside the tank as a Fluid object.
        gas : rocketpy.motors.Fluid
            Gas inside the tank as a Fluid object.
        liquid_height : int, float, callable, string, array
            Liquid height as a function of time in m. Must be a valid
            rocketpy.Function source. The liquid height zero level reference
            is assumed to be the same as the Tank geometry.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the liquid height
            input is already discretized this parameter may be set to None.
            Otherwise, an uniform discretization will be applied based on the
            discretize value.
            The default is 100.
        """
        super().__init__(name, geometry, flux_time, liquid, gas, discretize)

        # Define liquid height
        self.liquid_height = Function(liquid_height, "Time (s)", "height (m)", "linear")

        # Discretize input if needed
        self.discretize_liquid_height() if discretize else None

        # Check if the liquid level is within bounds
        if (self.liquid_height > self.geometry.top).any() or (
            self.liquid_height < self.geometry.bottom
        ).any():
            raise ValueError("The liquid level is out of bounds.")

    @funcify_method("Time (s)", "Mass (kg)")
    def mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "Mass flow rate (kg/s)")
    def netMassFlowRate(self):
        """
        Returns the net mass flow rate of the tank as a function of time by
        taking the derivative of the mass function.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.mass.derivativeFunction()

    @funcify_method("Time (s)", "Volume (m³)")
    def fluidVolume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        return self.geometry.total_volume

    @funcify_method("Time (s)", "Volume (m³)")
    def liquidVolume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return self.geometry.volume.compose(self.liquidHeight)

    @funcify_method("Time (s)", "Volume (m³)")
    def gasVolume(self):
        """
        Returns the volume of the gas as a function of time. The gas volume
        is assumed to uniformly occupy the volume above the liquid level.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.geometry.total_volume - self.liquidVolume

    @funcify_method("Time (s)", "Height (m)")
    def liquidHeight(self):
        """
        Returns the liquid level as a function of time. This height is
        measured from the zero level of the tank geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        return self.liquid_height

    @funcify_method("Time (s)", "Mass (kg)")
    def gasMass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        return self.gasVolume * self.gas.density

    @funcify_method("Time (s)", "Mass (kg)")
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        return self.liquidVolume * self.liquid.density

    @funcify_method("Time (s)", "Height (m)", "linear")
    def gasHeight(self):
        """
        Returns the gas level as a function of time. This
        height is measured from the zero level of the tank
        geometry. Since the gas is assumed to uniformly occupy
        the volume above the liquid level, the gas height is
        constant and equal to the top of the tank geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        return Function(self.geometry.top).setDiscreteBasedOnModel(self.liquidHeight)

    def discretize_liquid_height(self):
        """Discretizes the liquid height input according to the flux time
        and the discretize parameter.
        """
        self.liquid_height.setDiscrete(*self.flux_time, self.discretize)


class MassBasedTank(Tank):
    """Class to define a tank whose flow is described by liquid and gas masses.
    This class inherits from the Tank class. See the Tank class for more
    information on its attributes and methods.
    """

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
        """
        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : rocketpy.geometry.TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float, optional
            Tank flux time in seconds. It is the time range in which the tank
            flux is being analyzed. In general, during this time, the tank is
            being filled or emptied.
            If a float is given, the flux time is assumed to be between 0 and the
            given float, in seconds. If a tuple of float is given, the flux time
            is assumed to be between the first and second elements of the tuple.
        liquid : rocketpy.motors.Fluid
            Liquid inside the tank as a Fluid object.
        gas : rocketpy.motors.Fluid
            Gas inside the tank as a Fluid object.
        liquid_mass : int, float, callable, string, array
            Liquid mass as a function of time in kg. Must be a valid
            rocketpy.Function source.
        gas_mass : int, float, callable, string, array
            Gas mass as a function of time in kg. Must be a valid
            rocketpy.Function source.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the mass inputs
            are uniformly discretized (have the same time steps) this parameter
            may be set to None. Otherwise, an uniform discretization will be
            applied based on the discretize value.
            The default is 100.
        """
        super().__init__(name, geometry, flux_time, liquid, gas, discretize)

        # Define fluid masses
        self.liquid_mass = Function(liquid_mass, "Time (s)", "Mass (kg)", "linear")
        self.gas_mass = Function(gas_mass, "Time (s)", "Mass (kg)", "linear")

        # Discretize input if needed
        self.discretize_masses() if discretize else None

    @funcify_method("Time (s)", "Mass (kg)")
    def mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as
        a function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquidMass + self.gasMass

    @funcify_method("Time (s)", "Mass flow rate (kg/s)")
    def netMassFlowRate(self):
        """
        Returns the net mass flow rate of the tank as a function of time
        by taking the derivative of the mass function.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.mass.derivativeFunction()

    @funcify_method("Time (s)", "Mass (kg)")
    def liquidMass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        return self.liquid_mass

    @funcify_method("Time (s)", "Mass (kg)")
    def gasMass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        return self.gas_mass

    @funcify_method("Time (s)", "Volume (m³)")
    def fluidVolume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        return self.liquidVolume + self.gasVolume

    @funcify_method("Time (s)", "Volume (m³)")
    def gasVolume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.gasMass / self.gas.density

    @funcify_method("Time (s)", "Volume (m³)")
    def liquidVolume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return self.liquidMass / self.liquid.density

    @funcify_method("Time (s)", "Height (m)")
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
        return self.geometry.inverse_volume.compose(self.liquidVolume)

    @funcify_method("Time (s)", "Height (m)")
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
        fluid_volume = self.gasVolume + self.liquidVolume
        gasHeight = self.geometry.inverse_volume.compose(fluid_volume)
        if (gasHeight > self.geometry.top).any():
            raise ValueError(f"The tank {self.name} is overfilled.")
        return gasHeight

    def discretize_masses(self):
        """Discretizes the fluid mass inputs according to the flux time
        and the discretize parameter.
        """
        self.liquid_mass.setDiscrete(*self.flux_time, self.discretize)
        self.gas_mass.setDiscrete(*self.flux_time, self.discretize)
