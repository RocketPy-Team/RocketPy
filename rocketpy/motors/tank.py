from abc import ABC, abstractmethod

import numpy as np
from scipy.constants import atm, zero_Celsius

from ..mathutils.function import Function, funcify_method
from ..plots.tank_plots import _TankPlots
from ..prints.tank_prints import _TankPrints
from ..tools import deprecated, tuple_handler


class Tank(ABC):
    """Abstract Tank class that defines a tank object for a rocket motor, so
    that it evaluates useful properties of the tank and its fluids, such as
    mass, volume, fluid flow rate, center of mass, etc.

    See Also
    --------
    :ref:`tanks_usage`

    Attributes
    ----------
    Tank.name : str
        Name of the tank.
    Tank.geometry : TankGeometry
        Geometry of the tank.
    Tank.flux_time : float, tuple of float
        Tank flux time in seconds.
    Tank.liquid : Fluid
        Liquid inside the tank as a Fluid object.
    Tank.gas : Fluid
        Gas inside the tank as a Fluid object.
    Tank.discretize : int, optional
        Number of points to discretize fluid inputs.
    Tank.fluid_mass : Function
        Total mass of liquid and gases in kg inside the tank as a function
        of time.
    Tank.net_mass_flow_rate : Function
        Net mass flow rate of the tank in kg/s as a function of time, also
        understood as time derivative of the fluids mass.
    Tank.liquid_volume : Function
        Volume of the liquid inside the Tank in m^3 as a function of time.
    Tank.gas_volume : Function
        Volume of the gas inside the Tank in m^3 as a function of time.
    Tank.liquid_height : Function
        Height of the liquid inside the Tank in m as a function of time.
        The zero level reference is the same as set in Tank.geometry.
    Tank.gas_height : Function
        Height of the gas inside the Tank in m as a function of time.
        The zero level reference is the same as set in Tank.geometry.
    Tank.liquid_mass : Function
        Mass of the liquid inside the Tank in kg as a function of time.
    Tank.gas_mass : Function
        Mass of the gas inside the Tank in kg as a function of time.
    Tank.liquid_center_of_mass : Function
        Center of mass of the liquid inside the Tank in m as a function of
        time. The zero level reference is the same as set in Tank.geometry.
    Tank.gas_center_of_mass : Function
        Center of mass of the gas inside the Tank in m as a function of
        time. The zero level reference is the same as set in Tank.geometry.
    Tank.center_of_mass : Function
        Center of mass of liquid and gas (i.e. propellant) inside the Tank
        in m as a function of time. The zero level reference is the same as
        set in Tank.geometry.
    Tank.liquid_inertia : Function
        The inertia of the liquid inside the Tank in kg*m^2 as a function
        of time around a perpendicular axis to the Tank symmetry axis. The
        reference point is the Tank center of mass.
    Tank.gas_inertia : Function
        The inertia of the gas inside the Tank in kg*m^2 as a function of
        time around a perpendicular axis to the Tank symmetry axis. The
        reference point is the Tank center of mass.
    Tank.inertia : Function
        The inertia of the liquid and gas (i.e. propellant) inside the Tank
        in kg*m^2 as a function of time around a perpendicular axis to the
        Tank symmetry axis. The reference point is the Tank center of mass.
    """

    def __init__(
        self,
        name,
        geometry,
        flux_time,
        liquid,
        gas,
        discretize=100,
        temperature=None,
        pressure=None,
    ):
        """Initialize Tank class.

        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float
            Tank flux time in seconds. Time interval that the fluid flux is
            being analyzed. If a float is given, the flux time is assumed to
            be between 0 and the given float, in seconds. If a tuple of float
            is given, the flux time is assumed to be between the first and
            second elements of the tuple.
            Before the start time, the tank properties are kept at their
            initial state. After the final time, their final state is kept.
        gas : Fluid
            Gas inside the tank as a Fluid object.
        liquid : Fluid
            Liquid inside the tank as a Fluid object.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the input
            already has a appropriate uniform discretization, this parameter
            must be set to None. The default is 100.
        temperature : int, float, callable, string, array, Function
            Temperature inside the tank as a function of time in K. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the temperature in K. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of temperature.
        pressure : int, float, callable, string, array, Function
            Pressure inside the tank as a function of time in Pa. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the pressure in Pa. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of pressure.
        """
        self.name = name
        self.geometry = geometry
        self.flux_time = flux_time
        self.gas = gas
        self.liquid = liquid
        self.discretize = discretize
        self.temperature = temperature
        self.pressure = pressure

        self._liquid_density = self.liquid.get_time_variable_density(
            self.temperature, self.pressure
        )
        self._gas_density = self.gas.get_time_variable_density(
            self.temperature, self.pressure
        )

        # Initialize plots and prints object
        self.prints = _TankPrints(self)
        self.plots = _TankPlots(self)

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
    def temperature(self):
        """Returns the temperature of the tank as a function of time.

        Returns
        -------
        Function
            Temperature of the tank as a function of time.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Sets the temperature of the tank as a function of time.

        Parameters
        ----------
        temperature : int, float, callable, string, array, Function
            Temperature inside the tank as a function of time in K as
            a ``Function`` source.
        """
        if temperature is None:
            temperature = zero_Celsius

        _temperature = Function(
            temperature,
            interpolation="linear",
            extrapolation="constant",
            inputs="Time (s)",
            outputs="Temperature (K)",
        )

        if self.discretize:
            _temperature = _temperature.set_discrete(*self.flux_time, self.discretize)

        self._temperature = _temperature

    @property
    def pressure(self):
        """Returns the pressure of the tank as a function of time.

        Returns
        -------
        Function
            Pressure of the tank as a function of time.
        """
        return self._pressure

    @pressure.setter
    def pressure(self, pressure):
        """Sets the pressure of the tank as a function of time.

        Parameters
        ----------
        pressure : int, float, callable, string, array, Function
            Pressure inside the tank as a function of time in Pa as
            a ``Function`` source.
        """
        if pressure is None:
            pressure = atm

        _pressure = Function(
            pressure,
            interpolation="linear",
            extrapolation="constant",
            inputs="Time (s)",
            outputs="Pressure (Pa)",
        )

        if self.discretize:
            _pressure = _pressure.set_discrete(*self.flux_time, self.discretize)

        self._pressure = _pressure

    @property
    @abstractmethod
    def fluid_mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """

    @property
    @abstractmethod
    def net_mass_flow_rate(self):
        """
        Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate entering the tank minus the
        mass flow rate exiting the tank, including liquids and gases.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """

    @property
    @abstractmethod
    def fluid_volume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """

    @property
    @abstractmethod
    def liquid_volume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """

    @property
    @abstractmethod
    def gas_volume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """

    @property
    @abstractmethod
    def liquid_height(self):
        """
        Returns the liquid level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """

    @property
    @abstractmethod
    def gas_height(self):
        """
        Returns the gas level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """

    @property
    @abstractmethod
    def liquid_mass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """

    @property
    @abstractmethod
    def gas_mass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """

    @funcify_method("Time (s)", "Center of mass of liquid (m)")
    def liquid_center_of_mass(self):
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
            self.geometry.bottom, self.liquid_height.max
        )
        liquid_moment = moment @ self.liquid_height
        centroid = liquid_moment / self.liquid_volume

        # Check for zero liquid volume
        bound_volume = self.liquid_volume < 1e-4 * self.geometry.total_volume
        if bound_volume.any():
            # TODO: pending Function setter impl.
            centroid.y_array[bound_volume] = self.geometry.bottom
            centroid.set_interpolation()
            centroid.set_extrapolation()

        return centroid

    @funcify_method("Time (s)", "Center of mass of gas (m)")
    def gas_center_of_mass(self):
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
        moment = self.geometry.volume_moment(self.geometry.bottom, self.gas_height.max)
        upper_moment = moment @ self.gas_height
        lower_moment = moment @ self.liquid_height
        centroid = (upper_moment - lower_moment) / self.gas_volume

        # Check for zero gas volume
        bound_volume = self.gas_volume < 1e-4 * self.geometry.total_volume
        if bound_volume.any():
            # TODO: pending Function setter impl.
            centroid.y_array[bound_volume] = self.liquid_height.y_array[bound_volume]
            centroid.set_interpolation()
            centroid.set_extrapolation()

        return centroid

    @funcify_method("Time (s)", "Center of mass of Fluid (m)")
    def center_of_mass(self):
        """Returns the center of mass of the tank's fluids as a function of
        time. This height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Center of mass of the tank's fluids as a function of time.
        """
        center_of_mass = (
            self.liquid_center_of_mass * self.liquid_mass
            + self.gas_center_of_mass * self.gas_mass
        ) / (self.fluid_mass)

        # Check for zero mass
        bound_mass = (
            self.fluid_mass < 0.001 * self.geometry.total_volume * self._gas_density
        )
        if bound_mass.any():
            # TODO: pending Function setter impl.
            center_of_mass.y_array[bound_mass] = self.geometry.bottom
            center_of_mass.set_interpolation()
            center_of_mass.set_extrapolation()

        return center_of_mass

    @funcify_method("Time (s)", "Liquid Inertia (kg*m²)")
    def liquid_inertia(self):
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
        Ix_volume = self.geometry.Ix_volume(
            self.geometry.bottom, self.liquid_height.max
        )
        Ix_volume = Ix_volume @ self.liquid_height

        # Steiner theorem to account for center of mass
        Ix_volume -= self.liquid_volume * self.liquid_center_of_mass**2
        Ix_volume += (
            self.liquid_volume * (self.liquid_center_of_mass - self.center_of_mass) ** 2
        )

        return self._liquid_density * Ix_volume

    @funcify_method("Time (s)", "Gas Inertia (kg*m^2)")
    def gas_inertia(self):
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
        Ix_volume = self.geometry.Ix_volume(self.geometry.bottom, self.gas_height.max)
        lower_inertia_volume = Ix_volume @ self.liquid_height
        upper_inertia_volume = Ix_volume @ self.gas_height
        inertia_volume = upper_inertia_volume - lower_inertia_volume

        # Steiner theorem to account for center of mass
        inertia_volume -= self.gas_volume * self.gas_center_of_mass**2
        inertia_volume += (
            self.gas_volume * (self.gas_center_of_mass - self.center_of_mass) ** 2
        )

        return self._gas_density * inertia_volume

    @funcify_method("Time (s)", "Fluid Inertia (kg*m^2)")
    def inertia(self):
        """
        Returns the inertia tensor of the tank's fluids as a function of
        time. The reference point is the center of mass of the tank.

        Returns
        -------
        Function
            Inertia tensor of the tank's fluids as a function of time.
        """
        return self.liquid_inertia + self.gas_inertia

    def _check_volume_bounds(self):
        """Checks if the tank is overfilled or underfilled. Raises a ValueError
        if either the `gas_volume` or `liquid_volume` are out of tank geometry
        bounds.
        """

        def overfill_volume_exception(param_name, param):
            raise ValueError(
                f"The tank '{self.name}' is overfilled. The {param_name} is "
                + "greater than the total volume of the tank.\n\t\t"
                + "Try increasing the tank height and check out the fluid density "
                + "values.\n\t\t"
                + f"The {param_name} is {param.max:.3f} m³ at "
                + f"{param.x_array[np.argmax(param.y_array)]:.3f} s.\n\t\t"
                + f"The tank total volume is {self.geometry.total_volume:.3f} m³."
            )

        def underfill_volume_exception(param_name, param):
            raise ValueError(
                f"The tank '{self.name}' is underfilled. The {param_name} is "
                + "negative.\n\t\t"
                + "Try increasing input fluid quantities and check out the fluid "
                + "density values.\n\t\t"
                + f"The {param_name} is {param.min:.3f} m³ at "
                + f"{param.x_array[np.argmin(param.y_array)]:.3f} s.\n\t\t"
                + f"The tank total volume is {self.geometry.total_volume:.3f} m³."
            )

        for name, volume in [
            ("gas volume", self.gas_volume),
            ("liquid volume", self.liquid_volume),
        ]:
            if (volume > self.geometry.total_volume + 1e-6).any():
                overfill_volume_exception(name, volume)
            elif (volume < -1e-6).any():
                underfill_volume_exception(name, volume)

    def _check_height_bounds(self):
        """Checks if the tank is overfilled or underfilled. Raises a ValueError
        if either the `gas_height` or `liquid_height` are out of tank geometry
        bounds.
        """
        top_tolerance = self.geometry.top + 1e-4
        bottom_tolerance = self.geometry.bottom - 1e-4

        def overfill_height_exception(param_name, param):
            raise ValueError(
                f"The tank '{self.name}' is overfilled. "
                + f"The {param_name} is above the tank top.\n\t\t"
                + "Try increasing the tank height and check out the fluid density "
                + "values.\n\t\t"
                + f"The {param_name} is {param.max:.3f} m above the tank top "
                + f"at {param.x_array[np.argmax(param.y_array)]:.3f} s.\n\t\t"
                + f"The tank top is at {self.geometry.top:.3f} m."
            )

        def underfill_height_exception(param_name, param):
            raise ValueError(
                f"The tank '{self.name}' is underfilled. "
                + f"The {param_name} is below the tank bottom.\n\t\t"
                + "Try increasing input fluid quantities and check out the fluid "
                + "density values.\n\t\t"
                + f"The {param_name} is {param.min:.3f} m below the tank bottom "
                + f"at {param.x_array[np.argmin(param.y_array)]:.3f} s.\n\t\t"
                + f"The tank bottom is at {self.geometry.bottom:.3f} m."
            )

        for name, height in [
            ("gas height", self.gas_height),
            ("liquid height", self.liquid_height),
        ]:
            if (height > top_tolerance).any():
                overfill_height_exception(name, height)
            elif (height < bottom_tolerance).any():
                underfill_height_exception(name, height)

    @abstractmethod
    def _discretize_fluid_inputs(self):
        """Uniformly discretizes the parameter of inputs of fluid data ."""

    def draw(self, *, filename=None):
        """Draws the tank geometry.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        self.plots.draw(filename=filename)

    def info(self):
        """Prints out a summary of the tank properties."""
        self.prints.all()

    def all_info(self):
        """Prints out detailed information and plots of the tank
        properties.
        """
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = {
            "name": self.name,
            "geometry": self.geometry,
            "flux_time": self.flux_time,
            "liquid": self.liquid,
            "gas": self.gas,
            "discretize": self.discretize,
            "temperature": self.temperature,
            "pressure": self.pressure,
        }
        if kwargs.get("include_outputs", False):
            data.update(
                {
                    "fluid_mass": self.fluid_mass,
                    "net_mass_flow_rate": self.net_mass_flow_rate,
                    "liquid_volume": self.liquid_volume,
                    "gas_volume": self.gas_volume,
                    "liquid_height": self.liquid_height,
                    "gas_height": self.gas_height,
                    "liquid_mass": self.liquid_mass,
                    "gas_mass": self.gas_mass,
                    "liquid_center_of_mass": self.liquid_center_of_mass,
                    "gas_center_of_mass": self.gas_center_of_mass,
                    "center_of_mass": self.center_of_mass,
                    "liquid_inertia": self.liquid_inertia,
                    "gas_inertia": self.gas_inertia,
                    "inertia": self.inertia,
                }
            )
        return data


class MassFlowRateBasedTank(Tank):
    """Class to define a tank based on mass flow rates inputs. This class
    inherits from the Tank class. See the Tank class for more information
    on its attributes and methods.

    See Also
    --------
    :ref:`tanks_usage`
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
        temperature=None,
        pressure=None,
    ):
        """Initializes the MassFlowRateBasedTank class.

        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float
            Tank flux time in seconds. Time interval that the fluid flux is
            being analyzed. If a float is given, the flux time is assumed to
            be between 0 and the given float, in seconds. If a tuple of float
            is given, the flux time is assumed to be between the first and
            second elements of the tuple.
            Before the start time, the tank properties are kept at their
            initial state. After the final time, their final state is kept.
        liquid : Fluid
            Liquid inside the tank as a Fluid object.
        gas : Fluid
            Gas inside the tank as a Fluid object.
        initial_liquid_mass : float
            Initial liquid mass in the tank in kg.
        initial_gas_mass : float
            Initial gas mass in the tank in kg.
        liquid_mass_flow_rate_in : int, float, callable, string, array, Function
            Liquid mass flow rate into the tank in kg/s. Always positive.
            It must be a valid :class:`Function` source.
            If a callable is given, it must be a function of time.
            If a ``.csv`` file is given, it must have two columns, the first
            one being time in seconds and the second one being the mass flow
            rate in kg/s.
        gas_mass_flow_rate_in : int, float, callable, string, array, Function
            Gas mass flow rate into the tank in kg/s. Always positive.
            It must be a valid :class:`Function` source.
            If a callable is given, it must be a function of time.
            If a ``.csv`` file is given, it must have two columns, the first
            one being time in seconds and the second one being the mass flow
            rate in kg/s.
        liquid_mass_flow_rate_out : int, float, callable, string, array, Function
            Liquid mass flow rate out of the tank in kg/s. Always positive.
            It must be a valid :class:`Function` source.
            If a callable is given, it must be a function of time.
            If a ``.csv`` file is given, it must have two columns, the first
            one being time in seconds and the second one being the mass flow
            rate in kg/s.
        gas_mass_flow_rate_out : int, float, callable, string, array, Function
            Gas mass flow rate out of the tank in kg/s. Always positive.
            It must be a valid :class:`Function` source.
            If a callable is given, it must be a function of time.
            If a ``.csv`` file is given, it must have two columns, the first
            one being time in seconds and the second one being the mass flow
            rate in kg/s.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the mass flow
            rate inputs are uniformly discretized (have the same time steps)
            this parameter may be set to None. Otherwise, an uniform
            discretization will be applied based on the discretize value.
            The default is 100.
        temperature : int, float, callable, string, array, Function
            Temperature inside the tank as a function of time in K. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the temperature in K. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of temperature.
        pressure : int, float, callable, string, array, Function
            Pressure inside the tank as a function of time in Pa. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the pressure in Pa. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of pressure.
        """
        super().__init__(
            name, geometry, flux_time, liquid, gas, discretize, temperature, pressure
        )
        self.initial_liquid_mass = initial_liquid_mass
        self.initial_gas_mass = initial_gas_mass

        # Define flow rates
        self.liquid_mass_flow_rate_in = Function(
            liquid_mass_flow_rate_in,
            inputs="Time (s)",
            outputs="Liquid Mass Flow Rate In (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        self.gas_mass_flow_rate_in = Function(
            gas_mass_flow_rate_in,
            inputs="Time (s)",
            outputs="Gas Mass Flow Rate In (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        self.liquid_mass_flow_rate_out = Function(
            liquid_mass_flow_rate_out,
            inputs="Time (s)",
            outputs="Liquid Mass Flow Rate Out (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        self.gas_mass_flow_rate_out = Function(
            gas_mass_flow_rate_out,
            inputs="Time (s)",
            outputs="Gas Mass Flow Rate Out (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )

        self._discretize_fluid_inputs()

        # Check if the tank is overfilled or underfilled
        self._check_volume_bounds()
        self._check_height_bounds()

    @funcify_method("Time (s)", "Fluid Mass (kg)")
    def fluid_mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquid_mass + self.gas_mass

    @funcify_method("Time (s)", "Liquid Mass (kg)")
    def liquid_mass(self):
        """
        Returns the mass of the liquid as a function of time by integrating
        the liquid mass flow rate.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        liquid_flow = self.net_liquid_flow_rate.integral_function(
            datapoints=self.discretize
        )
        liquid_mass = self.initial_liquid_mass + liquid_flow
        if (liquid_mass < 0).any():
            raise ValueError(
                f"The tank {self.name} is underfilled. "
                + "The liquid mass is negative given the mass flow rates.\n\t\t"
                + "Try increasing the initial liquid mass, or reducing the mass"
                + "flow rates.\n\t\t"
                + f"The liquid mass is {np.min(liquid_mass.y_array):.3f} kg at "
                + f"{liquid_mass.x_array[np.argmin(liquid_mass.y_array)]} s."
            )
        return liquid_mass

    @funcify_method("Time (s)", "Gas Mass (kg)")
    def gas_mass(self):
        """
        Returns the mass of the gas as a function of time by integrating
        the gas mass flow rate.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        gas_flow = self.net_gas_flow_rate.integral_function(datapoints=self.discretize)
        gas_mass = self.initial_gas_mass + gas_flow
        if (gas_mass < -1e-6).any():  # -1e-6 is to avoid numerical errors
            raise ValueError(
                f"The tank {self.name} is underfilled. The gas mass is negative"
                + " given the mass flow rates.\n\t\t"
                + "Try increasing the initial gas mass, or reducing the mass"
                + " flow rates.\n\t\t"
                + f"The gas mass is {np.min(gas_mass.y_array):.3f} kg at "
                + f"{gas_mass.x_array[np.argmin(gas_mass.y_array)]} s."
            )

        return gas_mass

    @funcify_method("Time (s)", "Liquid Mass Flow Rate (kg/s)", extrapolation="zero")
    def net_liquid_flow_rate(self):
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

    @funcify_method("Time (s)", "Gas Mass Flow Rate (kg/s)", extrapolation="zero")
    def net_gas_flow_rate(self):
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

    @funcify_method("Time (s)", "Net Mass Flow Rate (kg/s)", extrapolation="zero")
    def net_mass_flow_rate(self):
        """
        Returns the net mass flow rate of the tank as a function of time.
        Net mass flow rate is the mass flow rate entering the tank minus the
        mass flow rate exiting the tank, including liquids and gases.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.net_liquid_flow_rate + self.net_gas_flow_rate

    @funcify_method("Time (s)", "Fluid Volume (m³)")
    def fluid_volume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        return self.liquid_volume + self.gas_volume

    @funcify_method("Time (s)", "Liquid Volume (m³)")
    def liquid_volume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return self.liquid_mass / self._liquid_density

    @funcify_method("Time (s)", "Gas Volume (m³)")
    def gas_volume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.gas_mass / self._gas_density

    @funcify_method("Time (s)", "Liquid Height (m)")
    def liquid_height(self):
        """
        Returns the liquid level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        liquid_height = self.geometry.inverse_volume.compose(self.liquid_volume)
        diff_bt = liquid_height - self.geometry.bottom
        diff_up = liquid_height - self.geometry.top

        if (diff_bt < 0).any():
            raise ValueError(
                f"The tank '{self.name}' is underfilled. The liquid height is "
                + "below the tank bottom.\n\t\t"
                + "Try increasing the initial liquid mass, or reducing the mass"
                + " flow rates.\n\t\t"
                + f"The liquid height is {np.min(diff_bt.y_array):.3f} m below "
                + f"the tank bottom at {diff_bt.x_array[np.argmin(diff_bt.y_array)]:.3f} s."
            )
        if (diff_up > 0).any():
            raise ValueError(
                f"The tank '{self.name}' is overfilled. The liquid height is "
                + "above the tank top.\n\t\t"
                + "Try increasing the tank height, or reducing the initial liquid"
                + " mass, or reducing the mass flow rates.\n\t\t"
                + f"The liquid height is {np.max(diff_up.y_array):.3f} m above "
                + f"the tank top at {diff_up.x_array[np.argmax(diff_up.y_array)]:.3f} s."
            )

        return liquid_height

    @funcify_method("Time (s)", "Gas Height (m)")
    def gas_height(self):
        """
        Returns the gas level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        fluid_volume = self.gas_volume + self.liquid_volume
        gas_height = self.geometry.inverse_volume.compose(fluid_volume)
        diff = gas_height - self.geometry.top
        if (diff > 0).any():
            raise ValueError(
                f"The tank '{self.name}' is overfilled. "
                + "The gas height is above the tank top.\n\t\t"
                + "Try increasing the tank height, or reducing fluids' mass,"
                + " or double check the mass flow rates.\n\t\t"
                + f"The gas height is {np.max(diff.y_array):.3f} m above "
                + f"the tank top at {diff.x_array[np.argmax(diff.y_array)]} s."
            )
        return gas_height

    def _discretize_fluid_inputs(self):
        """Uniformly discretizes the parameter of inputs of fluid data ."""
        if self.discretize:
            self.liquid_mass_flow_rate_in.set_discrete(
                *self.flux_time, self.discretize, "linear"
            )
            self.gas_mass_flow_rate_in.set_discrete(
                *self.flux_time, self.discretize, "linear"
            )
            self.liquid_mass_flow_rate_out.set_discrete(
                *self.flux_time, self.discretize, "linear"
            )
            self.gas_mass_flow_rate_out.set_discrete(
                *self.flux_time, self.discretize, "linear"
            )
        else:
            # Discretize densities for backward compatibility
            self._liquid_density.set_discrete_based_on_model(
                self.liquid_mass_flow_rate_in
            )
            self._gas_density.set_discrete_based_on_model(self.gas_mass_flow_rate_in)

    @deprecated(
        "Should not be a public member of the class.",
        "1.12.0",
        "_discretize_fluid_inputs",
    )
    def discretize_flow(self):
        """Discretizes the mass flow rate inputs according to the flux time and
        the discretize parameter.
        """
        self._discretize_fluid_inputs()

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update(
            {
                "initial_liquid_mass": self.initial_liquid_mass,
                "initial_gas_mass": self.initial_gas_mass,
                "liquid_mass_flow_rate_in": self.liquid_mass_flow_rate_in,
                "gas_mass_flow_rate_in": self.gas_mass_flow_rate_in,
                "liquid_mass_flow_rate_out": self.liquid_mass_flow_rate_out,
                "gas_mass_flow_rate_out": self.gas_mass_flow_rate_out,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            geometry=data["geometry"],
            flux_time=data["flux_time"],
            liquid=data["liquid"],
            gas=data["gas"],
            initial_liquid_mass=data["initial_liquid_mass"],
            initial_gas_mass=data["initial_gas_mass"],
            liquid_mass_flow_rate_in=data["liquid_mass_flow_rate_in"],
            gas_mass_flow_rate_in=data["gas_mass_flow_rate_in"],
            liquid_mass_flow_rate_out=data["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_out=data["gas_mass_flow_rate_out"],
            discretize=data["discretize"],
            temperature=data.get("temperature"),
            pressure=data.get("pressure"),
        )


class UllageBasedTank(Tank):
    """Class to define a tank whose flow is described by ullage volume, i.e.,
    the volume of the tank that is not occupied by the liquid. It assumes that
    the ullage volume is uniformly filled by the gas. This class inherits from
    the Tank class. See the Tank class for more information on its attributes
    and methods.

    See Also
    --------
    :ref:`tanks_usage`
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
        temperature=None,
        pressure=None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float
            Tank flux time in seconds. Time interval that the fluid flux is
            being analyzed. If a float is given, the flux time is assumed to
            be between 0 and the given float, in seconds. If a tuple of float
            is given, the flux time is assumed to be between the first and
            second elements of the tuple.
            Before the start time, the tank properties are kept at their
            initial state. After the final time, their final state is kept.
        liquid : Fluid
            Liquid inside the tank as a Fluid object.
        gas : Fluid
            Gas inside the tank as a Fluid object.
        ullage : int, float, callable, string, array, Function
            Ullage volume as a function of time in m^3. Also understood as the
            volume of the Tank that is not occupied by liquid. Must be a valid
            :class:`Function` source.
            If a callable is given, it must be a function of time in seconds.
            If a ``.csv`` file is given, the first column must be the time in
            seconds and the second column must be the ullage volume in m^3.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the ullage input is
            already discretized this parameter may be set to None. Otherwise,
            an uniform discretization will be applied based on the discretize
            value.
            The default is 100.
        temperature : int, float, callable, string, array, Function
            Temperature inside the tank as a function of time in K. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the temperature in K. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of temperature.
        pressure : int, float, callable, string, array, Function
            Pressure inside the tank as a function of time in Pa. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the pressure in Pa. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of pressure.
        """
        super().__init__(
            name, geometry, flux_time, liquid, gas, discretize, temperature, pressure
        )

        # Define ullage
        self.ullage = Function(ullage, "Time (s)", "Volume (m³)", "linear")

        # Discretize input if needed
        self._discretize_fluid_inputs()

        # Check if the tank is overfilled or underfilled
        self._check_volume_bounds()
        self._check_height_bounds()

    @funcify_method("Time (s)", "Fluid Mass (kg)")
    def fluid_mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquid_mass + self.gas_mass

    @funcify_method("Time (s)", "Net Mass Flow Rate (kg/s)")
    def net_mass_flow_rate(self):
        """
        Returns the net mass flow rate of the tank as a function of time by
        taking the derivative of the mass function.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.fluid_mass.derivative_function()

    @funcify_method("Time (s)", "Fluid Volume (m³)")
    def fluid_volume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        return Function(self.geometry.total_volume).set_discrete_based_on_model(
            self.gas_volume
        )

    @funcify_method("Time (s)", "Liquid Volume (m³)")
    def liquid_volume(self):
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

    @funcify_method("Time (s)", "Gas Volume (m³)")
    def gas_volume(self):
        """
        Returns the volume of the gas as a function of time. From the
        Tank assumptions the gas volume is equal to the ullage volume.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.ullage

    @funcify_method("Time (s)", "Gas Mass (kg)")
    def gas_mass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        return self.gas_volume * self._gas_density

    @funcify_method("Time (s)", "Liquid Mass (kg)")
    def liquid_mass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        return self.liquid_volume * self._liquid_density

    @funcify_method("Time (s)", "Liquid Height (m)")
    def liquid_height(self):
        """
        Returns the liquid level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        return self.geometry.inverse_volume.compose(self.liquid_volume)

    @funcify_method("Time (s)", "Gas Height (m)", "linear")
    def gas_height(self):
        """
        Returns the gas level as a function of time. This height is measured
        from the zero level of the tank geometry. Since the gas is assumed to
        be uniformly distributed in the ullage, the gas height is constant
        and equal to the top of the tank geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        return Function(self.geometry.top).set_discrete_based_on_model(self.gas_volume)

    def _discretize_fluid_inputs(self):
        """Uniformly discretizes the parameter of inputs of fluid data ."""
        if self.discretize:
            self.ullage.set_discrete(*self.flux_time, self.discretize, "linear")
        else:
            # Discretize densities for backward compatibility
            self._liquid_density.set_discrete_based_on_model(self.ullage)
            self._gas_density.set_discrete_based_on_model(self.ullage)

    @deprecated(
        "Should not be a public member of the class.",
        "1.12.0",
        "_discretize_fluid_inputs",
    )
    def discretize_ullage(self):
        """Discretizes the ullage input according to the flux time
        and the discretize parameter.
        """
        self._discretize_fluid_inputs()

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update({"ullage": self.ullage})
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            geometry=data["geometry"],
            flux_time=data["flux_time"],
            liquid=data["liquid"],
            gas=data["gas"],
            ullage=data["ullage"],
            discretize=data["discretize"],
            temperature=data.get("temperature"),
            pressure=data.get("pressure"),
        )


class LevelBasedTank(Tank):
    """Class to define a tank whose flow is described by liquid level, i.e.,
    the height of the liquid inside the tank. It assumes that the volume
    above the liquid level is uniformly occupied by gas. This class inherits
    from the Tank class. See the Tank class for more information on its
    attributes and methods.

    See Also
    --------
    :ref:`tanks_usage`
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
        temperature=None,
        pressure=None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float
            Tank flux time in seconds. Time interval that the fluid flux is
            being analyzed. If a float is given, the flux time is assumed to
            be between 0 and the given float, in seconds. If a tuple of float
            is given, the flux time is assumed to be between the first and
            second elements of the tuple.
            Before the start time, the tank properties are kept at their
            initial state. After the final time, their final state is kept.
        liquid : Fluid
            Liquid inside the tank as a Fluid object.
        gas : Fluid
            Gas inside the tank as a Fluid object.
        liquid_height : int, float, callable, string, array, Function
            Liquid height as a function of time in m. Must be a valid
            :class:`Function` source. The liquid height zero level
            reference is assumed to be the same as the Tank geometry.
            If a callable is given, it must be a function of time in seconds
            If a ``.csv`` file is given, the first column is assumed to be the
            time and the second column the liquid height in meters.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the liquid height
            input is already discretized this parameter may be set to None.
            Otherwise, an uniform discretization will be applied based on the
            discretize value.
            The default is 100.
        temperature : int, float, callable, string, array, Function
            Temperature inside the tank as a function of time in K. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the temperature in K. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of temperature.
        pressure : int, float, callable, string, array, Function
            Pressure inside the tank as a function of time in Pa. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the pressure in Pa. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of pressure.
        """
        super().__init__(
            name, geometry, flux_time, liquid, gas, discretize, temperature, pressure
        )

        # Define liquid level function
        self.liquid_level = Function(liquid_height, "Time (s)", "height (m)", "linear")

        self._discretize_fluid_inputs()

        # Check if the tank is overfilled or underfilled
        self._check_height_bounds()
        self._check_volume_bounds()

    @funcify_method("Time (s)", "Fluid Mass (kg)")
    def fluid_mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as a
        function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        # TODO: there's a bug in the net_mass_flow_rate if I don't discretize here
        sum_mass = self.liquid_mass + self.gas_mass
        sum_mass.set_discrete_based_on_model(self.liquid_level)
        return sum_mass

    @funcify_method("Time (s)", "Net Mass Flow Rate (kg/s)")
    def net_mass_flow_rate(self):
        """
        Returns the net mass flow rate of the tank as a function of time by
        taking the derivative of the mass function.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.fluid_mass.derivative_function()

    @funcify_method("Time (s)", "Fluid Volume (m³)")
    def fluid_volume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        volume = self.gas_volume + self.liquid_volume
        diff = volume - self.geometry.total_volume
        if (diff > 1e-6).any():
            raise ValueError(
                "The `fluid_volume`, defined as the sum of `gas_volume` and "
                + "`liquid_volume`, is not equal to the total volume of the tank."
                + "\n\t\tThe difference is more than 1e-6 m^3 at "
                + f"{diff.x_array[np.argmin(diff.y_array)]} s."
            )
        return volume

    @funcify_method("Time (s)", "Liquid Volume (m³)")
    def liquid_volume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return self.geometry.volume.compose(self.liquid_height)

    @funcify_method("Time (s)", "Gas Volume (m³)")
    def gas_volume(self):
        """
        Returns the volume of the gas as a function of time. The gas volume
        is assumed to uniformly occupy the volume above the liquid level.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        # TODO: there's a bug on the gas_center_of_mass if I don't discretize here
        func = Function(self.geometry.total_volume).set_discrete_based_on_model(
            self.liquid_volume
        )
        func -= self.liquid_volume
        return func

    @funcify_method("Time (s)", "Liquid Height (m)")
    def liquid_height(self):
        """
        Returns the liquid level as a function of time. This height is
        measured from the zero level of the tank geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        return self.liquid_level

    @funcify_method("Time (s)", "Gas Mass (kg)")
    def gas_mass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        return self.gas_volume * self._gas_density

    @funcify_method("Time (s)", "Liquid Mass (kg)")
    def liquid_mass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        return self.liquid_volume * self._liquid_density

    @funcify_method("Time (s)", "Gas Height (m)", "linear")
    def gas_height(self):
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
        return Function(self.geometry.top).set_discrete_based_on_model(
            self.liquid_level
        )

    @deprecated(
        "Should not be a public member of the class.",
        "1.12.0",
        "_discretize_fluid_inputs",
    )
    def discretize_liquid_height(self):
        """Discretizes the liquid height input according to the flux time
        and the discretize parameter.
        """
        self._discretize_fluid_inputs()

    def _discretize_fluid_inputs(self):
        """Uniformly discretizes the parameter of inputs of fluid data ."""
        if self.discretize:
            self.liquid_level.set_discrete(*self.flux_time, self.discretize, "linear")
        else:
            # Discretize densities for backward compatibility
            self._liquid_density.set_discrete_based_on_model(self.liquid_level)
            self._gas_density.set_discrete_based_on_model(self.liquid_level)

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update({"liquid_height": self.liquid_level})
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            geometry=data["geometry"],
            flux_time=data["flux_time"],
            liquid=data["liquid"],
            gas=data["gas"],
            liquid_height=data["liquid_height"],
            discretize=data["discretize"],
            temperature=data.get("temperature"),
            pressure=data.get("pressure"),
        )


class MassBasedTank(Tank):
    """Class to define a tank whose flow is described by liquid and gas masses.
    This class inherits from the Tank class. See the Tank class for more
    information on its attributes and methods.

    See Also
    --------
    :ref:`tanks_usage`
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
        temperature=None,
        pressure=None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the tank.
        geometry : TankGeometry
            Geometry of the tank.
        flux_time : float, tuple of float
            Tank flux time in seconds. Time interval that the fluid flux is
            being analyzed. If a float is given, the flux time is assumed to
            be between 0 and the given float, in seconds. If a tuple of float
            is given, the flux time is assumed to be between the first and
            second elements of the tuple.
            Before the start time, the tank properties are kept at their
            initial state. After the final time, their final state is kept.
        liquid : Fluid
            Liquid inside the tank as a Fluid object.
        gas : Fluid
            Gas inside the tank as a Fluid object.
        liquid_mass : int, float, callable, string, array, Function
            Liquid mass as a function of time in kg. Must be a valid
            :class:`Function` source.
            If a callable is given, it must be a function of time in seconds.
            If a ``.csv`` file is given, the first column must be the time in
            seconds and the second column must be the liquid mass in kg.
        gas_mass : int, float, callable, string, array, Function
            Gas mass as a function of time in kg. Must be a valid
            :class:`Function` source.
            If a callable is given, it must be a function of time in seconds.
            If a ``.csv`` file is given, the first column must be the time in
            seconds and the second column must be the gas mass in kg.
        discretize : int, optional
            Number of points to discretize fluid inputs. If the mass inputs
            are uniformly discretized (have the same time steps) this parameter
            may be set to None. Otherwise, an uniform discretization will be
            applied based on the discretize value.
            The default is 100.
        temperature : int, float, callable, string, array, Function
            Temperature inside the tank as a function of time in K. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the temperature in K. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of temperature.
        pressure : int, float, callable, string, array, Function
            Pressure inside the tank as a function of time in Pa. If a callable
            is given, it must be a function of time in seconds. An array of points
            can also be given as a list or a string with the path to a .csv file
            with two columns, the first one being the time in seconds and the
            second one being the pressure in Pa. The default is None. This
            parameter is only required if fluid (``liquid`` or ``gas`` parameters)
            densities are functions of pressure.
        """
        super().__init__(
            name, geometry, flux_time, liquid, gas, discretize, temperature, pressure
        )

        # Define fluid masses
        self.liquid_mass = Function(liquid_mass, "Time (s)", "Mass (kg)", "linear")
        self.gas_mass = Function(gas_mass, "Time (s)", "Mass (kg)", "linear")

        self._discretize_fluid_inputs()

        # Check if the tank is overfilled or underfilled
        self._check_volume_bounds()
        self._check_height_bounds()

    @funcify_method("Time (s)", "Fluid Mass (kg)")
    def fluid_mass(self):
        """
        Returns the total mass of liquid and gases inside the tank as
        a function of time.

        Returns
        -------
        Function
            Mass of the tank as a function of time. Units in kg.
        """
        return self.liquid_mass + self.gas_mass

    @funcify_method("Time (s)", "Net Mass Flow Rate (kg/s)")
    def net_mass_flow_rate(self):
        """
        Returns the net mass flow rate of the tank as a function of time
        by taking the derivative of the mass function.

        Returns
        -------
        Function
            Net mass flow rate of the tank as a function of time.
        """
        return self.fluid_mass.derivative_function()

    @funcify_method("Time (s)", "Liquid Mass (kg)")
    def liquid_mass(self):
        """
        Returns the mass of the liquid as a function of time.

        Returns
        -------
        Function
            Mass of the liquid as a function of time.
        """
        return self.liquid_mass

    @funcify_method("Time (s)", "Gas Mass (kg)")
    def gas_mass(self):
        """
        Returns the mass of the gas as a function of time.

        Returns
        -------
        Function
            Mass of the gas as a function of time.
        """
        return self.gas_mass

    @funcify_method("Time (s)", "Fluid Volume (m³)")
    def fluid_volume(self):
        """
        Returns the volume total fluid volume inside the tank as a
        function of time. This volume is the sum of the liquid and gas
        volumes.

        Returns
        -------
        Function
            Volume of the fluid as a function of time.
        """
        fluid_volume = self.liquid_volume + self.gas_volume

        # Check if within bounds
        diff = fluid_volume - self.geometry.total_volume

        if (diff > 1e-6).any():
            raise ValueError(
                f"The tank {self.name} was overfilled. The input fluid masses "
                + "produce a volume that surpasses the tank total volume by more "
                + f"than 1e-6 m^3 at {diff.x_array[np.argmax(diff.y_array)]} s."
                + "\n\t\tCheck out the input masses, fluid densities or raise the "
                + "tank height so as to increase its total volume."
            )

        return fluid_volume

    @funcify_method("Time (s)", "Gas Volume (m³)")
    def gas_volume(self):
        """
        Returns the volume of the gas as a function of time.

        Returns
        -------
        Function
            Volume of the gas as a function of time.
        """
        return self.gas_mass / self._gas_density

    @funcify_method("Time (s)", "Liquid Volume (m³)")
    def liquid_volume(self):
        """
        Returns the volume of the liquid as a function of time.

        Returns
        -------
        Function
            Volume of the liquid as a function of time.
        """
        return self.liquid_mass / self._liquid_density

    @funcify_method("Time (s)", "Liquid Height (m)")
    def liquid_height(self):
        """
        Returns the liquid level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        liquid_height = self.geometry.inverse_volume.compose(self.liquid_volume)
        diff_bt = liquid_height - self.geometry.bottom
        diff_up = liquid_height - self.geometry.top

        if (diff_bt < 0).any():
            raise ValueError(
                f"The tank {self.name} is underfilled. The liquid height is below "
                + "the tank bottom.\n\t\tTry increasing the initial liquid mass, "
                + "or reducing the mass flow rates.\n\t\t"
                + f"The liquid height is {np.min(diff_bt.y_array):.3f} m below "
                + f"the tank bottom at {diff_bt.x_array[np.argmin(diff_bt.y_array)]:.3f} s."
            )
        if (diff_up > 0).any():
            raise ValueError(
                f"The tank {self.name} is overfilled. The liquid height is above "
                + "the tank top.\n\t\tTry increasing the tank height, or reducing "
                + "the initial liquid mass, or reducing the mass flow rates.\n\t\t"
                + f"The liquid height is {np.max(diff_up.y_array):.3f} m above "
                + f"the tank top at {diff_up.x_array[np.argmax(diff_up.y_array)]:.3f} s."
            )

        return liquid_height

    @funcify_method("Time (s)", "Gas Height (m)")
    def gas_height(self):
        """
        Returns the gas level as a function of time. This
        height is measured from the zero level of the tank
        geometry.

        Returns
        -------
        Function
            Height of the ullage as a function of time.
        """
        fluid_volume = self.gas_volume + self.liquid_volume
        gas_height = self.geometry.inverse_volume.compose(fluid_volume)
        diff = gas_height - self.geometry.top
        if (diff > 0).any():
            raise ValueError(
                f"The tank {self.name} is overfilled. The gas height is above "
                + "the tank top.\n\t\tTry increasing the tank height, or "
                + "reducing fluids' mass, or double check the mass flow rates."
                + f"\n\t\tThe gas height is {np.max(diff.y_array):.3f} m "
                + f"above the tank top at {diff.x_array[np.argmax(diff.y_array)]} s."
            )
        return gas_height

    @deprecated(
        "Should not be a public member of the class.",
        "1.12.0",
        "_discretize_fluid_inputs",
    )
    def discretize_masses(self):
        """Discretizes the fluid mass inputs according to the flux time
        and the discretize parameter.
        """
        self._discretize_fluid_inputs()

    def _discretize_fluid_inputs(self):
        """Uniformly discretizes the parameter of inputs of fluid data ."""
        if self.discretize:
            self.liquid_mass.set_discrete(*self.flux_time, self.discretize, "linear")
            self.gas_mass.set_discrete(*self.flux_time, self.discretize, "linear")
        else:
            # Discretize densities for backward compatibility
            self._liquid_density.set_discrete_based_on_model(self.liquid_mass)
            self._gas_density.set_discrete_based_on_model(self.gas_mass)

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update(
            {
                "liquid_mass": self.liquid_mass,
                "gas_mass": self.gas_mass,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            geometry=data["geometry"],
            flux_time=data["flux_time"],
            liquid=data["liquid"],
            gas=data["gas"],
            liquid_mass=data["liquid_mass"],
            gas_mass=data["gas_mass"],
            discretize=data["discretize"],
            temperature=data.get("temperature"),
            pressure=data.get("pressure"),
        )
