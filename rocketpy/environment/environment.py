# pylint: disable=too-many-public-methods, too-many-instance-attributes
import bisect
import json
import re
import warnings
from collections import namedtuple
from datetime import datetime

import netCDF4
import numpy as np
import pytz

from rocketpy.environment.fetchers import (
    fetch_atmospheric_data_from_windy,
    fetch_gefs_ensemble,
    fetch_gfs_file_return_dataset,
    fetch_hiresw_file_return_dataset,
    fetch_nam_file_return_dataset,
    fetch_open_elevation,
    fetch_rap_file_return_dataset,
    fetch_wyoming_sounding,
)
from rocketpy.environment.tools import (
    calculate_wind_heading,
    calculate_wind_speed,
    convert_wind_heading_to_direction,
    find_latitude_index,
    find_longitude_index,
    find_time_index,
    geodesic_to_utm,
    get_elevation_data_from_dataset,
    get_final_date_from_time_array,
    get_initial_date_from_time_array,
    get_interval_date_from_time_array,
    get_pressure_levels_from_file,
    mask_and_clean_dataset,
)
from rocketpy.environment.weather_model_mapping import WeatherModelMapping
from rocketpy.mathutils.function import NUMERICAL_TYPES, Function, funcify_method
from rocketpy.plots.environment_plots import _EnvironmentPlots
from rocketpy.prints.environment_prints import _EnvironmentPrints
from rocketpy.tools import (
    bilinear_interpolation,
    geopotential_height_to_geometric_height,
)


class Environment:
    """Keeps all environment information stored, such as wind and temperature
    conditions, as well as gravity.

    Attributes
    ----------
    Environment.earth_radius : float
        Value of Earth's Radius as 6.3781e6 m.
    Environment.air_gas_constant : float
        Value of Air's Gas Constant as 287.05287 J/K/Kg
    Environment.gravity : Function
        Gravitational acceleration. Positive values point the
        acceleration down. See :meth:`Environment.set_gravity_model` for more
        information.
    Environment.latitude : float
        Launch site latitude.
    Environment.longitude : float
        Launch site longitude.
    Environment.datum : string
        The desired reference ellipsoid model, the following options are
        available: ``SAD69``, ``WGS84``, ``NAD83``, and ``SIRGAS2000``.
    Environment.initial_east : float
        Launch site East UTM coordinate
    Environment.initial_north :  float
        Launch site North UTM coordinate
    Environment.initial_utm_zone : int
        Launch site UTM zone number
    Environment.initial_utm_letter : string
        Launch site UTM letter, to keep the latitude band and describe the
        UTM Zone
    Environment.initial_hemisphere : string
        Launch site South/North hemisphere
    Environment.initial_ew : string
        Launch site East/West hemisphere
    Environment.elevation : float
        Launch site elevation.
    Environment.datetime_date : datetime
        Date time of launch in UTC time zone using the ``datetime`` object.
    Environment.local_date : datetime
        Date time of launch in the local time zone, defined by the
        ``Environment.timezone`` parameter.
    Environment.timezone : string
        Local time zone specification. See `pytz`_. for time zone information.

        .. _pytz: https://pytz.sourceforge.net/

    Environment.elev_lon_array : array
        Unidimensional array containing the longitude coordinates.
    Environment.elev_lat_array : array
        Unidimensional array containing the latitude coordinates.
    Environment.elev_array : array
        Two-dimensional Array containing the elevation information.
    Environment.topographic_profile_activated : bool
        True if the user already set a topographic profile. False otherwise.
    Environment.max_expected_height : float
        Maximum altitude in meters to keep weather data. The altitude must be
        Above Sea Level (ASL). Especially useful for controlling plots.
        Can be altered as desired by running ``max_expected_height = number``.
    Environment.pressure_ISA : Function
        Air pressure in Pa as a function of altitude as defined by the
        International Standard Atmosphere ISO 2533.
    Environment.temperature_ISA : Function
        Air temperature in K as a function of altitude as defined by the
        International Standard Atmosphere ISO 2533
    Environment.pressure : Function
        Air pressure in Pa as a function of altitude.
    Environment.barometric_height : Function
        Geometric height above sea level in m as a function of pressure.
    Environment.temperature : Function
        Air temperature in K as a function of altitude.
    Environment.speed_of_sound : Function
        Speed of sound in air in m/s as a function of altitude.
    Environment.density : Function
        Air density in kg/m³ as a function of altitude.
    Environment.dynamic_viscosity : Function
        Air dynamic viscosity in Pa*s as a function of altitude.
    Environment.wind_speed : Function
        Wind speed in m/s as a function of altitude.
    Environment.wind_direction : Function
        Wind direction (from which the wind blows) in degrees relative to north
        (positive clockwise) as a function of altitude.
    Environment.wind_heading : Function
        Wind heading (direction towards which the wind blows) in degrees
        relative to north (positive clockwise) as a function of altitude.
    Environment.wind_velocity_x : Function
        Wind U, or X (east) component of wind velocity in m/s as a function of
        altitude.
    Environment.wind_velocity_y : Function
        Wind V, or Y (north) component of wind velocity in m/s as a function of
        altitude.
    Environment.atmospheric_model_type : string
        Describes the atmospheric model which is being used. Can only assume the
        following values: ``standard_atmosphere``, ``custom_atmosphere``,
        ``wyoming_sounding``, ``Forecast``, ``Reanalysis``,
        ``Ensemble``.
    Environment.atmospheric_model_file : string
        Address of the file used for the atmospheric model being used. Only
        defined for ``wyoming_sounding``, ``Forecast``,
        ``Reanalysis``, ``Ensemble``
    Environment.atmospheric_model_dict : dictionary
        Dictionary used to properly interpret ``netCDF`` and ``OPeNDAP`` files.
        Only defined for ``Forecast``, ``Reanalysis``, ``Ensemble``.
    Environment.atmospheric_model_init_date : datetime
        Datetime object instance of first available date in ``netCDF``
        and ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.atmospheric_model_end_date : datetime
        Datetime object instance of last available date in ``netCDF`` and
        ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.atmospheric_model_interval : int
        Hour step between weather condition used in ``netCDF`` and
        ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.atmospheric_model_init_lat : float
        Latitude of vertex just before the launch site in ``netCDF``
        and ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.atmospheric_model_end_lat : float
        Latitude of vertex just after the launch site in ``netCDF``
        and ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.atmospheric_model_init_lon : float
        Longitude of vertex just before the launch site in ``netCDF``
        and ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.atmospheric_model_end_lon : float
        Longitude of vertex just after the launch site in ``netCDF``
        and ``OPeNDAP`` files when using ``Forecast``, ``Reanalysis`` or
        ``Ensemble``.
    Environment.lat_array : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. 2x2 matrix for each pressure level of
        latitudes corresponding to the vertices of the grid cell which
        surrounds the launch site.
    Environment.lon_array : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. 2x2 matrix for each pressure level of
        longitudes corresponding to the vertices of the grid cell which
        surrounds the launch site.
    Environment.lon_index : int
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. Index to a grid longitude which
        is just over the launch site longitude, while ``lon_index`` - 1
        points to a grid longitude which is just under the launch
        site longitude.
    Environment.lat_index : int
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. Index to a grid latitude which
        is just over the launch site latitude, while ``lat_index`` - 1
        points to a grid latitude which is just under the launch
        site latitude.
    Environment.geopotentials : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. 2x2 matrix for each pressure level of
        geopotential heights corresponding to the vertices of the grid cell
        which surrounds the launch site.
    Environment.wind_us : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. 2x2 matrix for each pressure level of
        wind U (east) component corresponding to the vertices of the grid
        cell which surrounds the launch site.
    Environment.wind_vs : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. 2x2 matrix for each pressure level of
        wind V (north) component corresponding to the vertices of the grid
        cell which surrounds the launch site.
    Environment.levels : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. List of pressure levels available in the file.
    Environment.temperatures : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. 2x2 matrix for each pressure level of
        temperatures corresponding to the vertices of the grid cell which
        surrounds the launch site.
    Environment.time_array : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. Array of dates available in the file.
    Environment.height : array
        Defined if ``netCDF`` or ``OPeNDAP`` file is used, for Forecasts,
        Reanalysis and Ensembles. List of geometric height corresponding to
        launch site location.
    Environment.level_ensemble : array
        Only defined when using Ensembles.
    Environment.height_ensemble : array
        Only defined when using Ensembles.
    Environment.temperature_ensemble : array
        Only defined when using Ensembles.
    Environment.wind_u_ensemble : array
        Only defined when using Ensembles.
    Environment.wind_v_ensemble : array
        Only defined when using Ensembles.
    Environment.wind_heading_ensemble : array
        Only defined when using Ensembles.
    Environment.wind_direction_ensemble : array
        Only defined when using Ensembles.
    Environment.wind_speed_ensemble : array
        Only defined when using Ensembles.
    Environment.num_ensemble_members : int
        Number of ensemble members. Only defined when using Ensembles.
    Environment.ensemble_member : int
        Current selected ensemble member. Only defined when using Ensembles.
    Environment.earth_rotation_vector : list[float]
        Earth's angular velocity vector in the Flight Coordinate System.

    Notes
    -----
    All the attributes listed as Function objects can be accessed as
    regular arrays, or called as a Function. See :class:`rocketpy.Function`
    for more information.
    """

    def __init__(
        self,
        gravity=None,
        date=None,
        latitude=0.0,
        longitude=0.0,
        elevation=0.0,
        datum="SIRGAS2000",
        timezone="UTC",
        max_expected_height=80000.0,
    ):
        """Initializes the Environment class, capturing essential parameters of
        the launch site, including the launch date, geographical coordinates,
        and elevation. This class is designed to calculate crucial variables
        for the Flight simulation, such as atmospheric air pressure, density,
        and gravitational acceleration.

        Note that the default atmospheric model is the International Standard
        Atmosphere as defined by ISO 2533 unless specified otherwise in
        :meth:`Environment.set_atmospheric_model`.

        Parameters
        ----------
        gravity : int, float, callable, string, array, optional
            Surface gravitational acceleration. Positive values point the
            acceleration down. If None, the Somigliana formula is used.
            See :meth:`Environment.set_gravity_model` for more information.
        date : list or tuple, optional
            List or tuple of length 4, stating (year, month, day, hour) in the
            time zone of the parameter ``timezone``.
            Alternatively, can be a ``datetime`` object specifying launch
            date and time. The dates are stored as follows:

            - :attr:`Environment.local_date`: Local time of launch in
              the time zone specified by the parameter ``timezone``.

            - :attr:`Environment.datetime_date`: UTC time of launch.

            Must be given if a Forecast, Reanalysis
            or Ensemble, will be set as an atmospheric model.
            Default is None.
            See :meth:`Environment.set_date` for more information.
        latitude : float, optional
            Latitude in degrees (ranging from -90 to 90) of rocket
            launch location. Must be given if a Forecast, Reanalysis
            or Ensemble will be used as an atmospheric model or if
            Open-Elevation will be used to compute elevation. Positive
            values correspond to the North. Default value is 0, which
            corresponds to the equator.
        longitude : float, optional
            Longitude in degrees (ranging from -180 to 180) of rocket
            launch location. Must be given if a Forecast, Reanalysis
            or Ensemble will be used as an atmospheric model or if
            Open-Elevation will be used to compute elevation. Positive
            values correspond to the East. Default value is 0, which
            corresponds to the Greenwich Meridian.
        elevation : float, optional
            Elevation of launch site measured as height above sea
            level in meters. Alternatively, can be set as
            ``Open-Elevation`` which uses the Open-Elevation API to
            find elevation data. For this option, latitude and
            longitude must also be specified. Default value is 0.
        datum : string, optional
            The desired reference ellipsoidal model, the following options are
            available: "SAD69", "WGS84", "NAD83", and "SIRGAS2000". The default
            is "SIRGAS2000".
        timezone : string, optional
            Name of the time zone. To see all time zones, import pytz and run
            ``print(pytz.all_timezones)``. Default time zone is "UTC".
        max_expected_height : float, optional
            Maximum altitude in meters to keep weather data. The altitude must
            be above sea level (ASL). Especially useful for visualization. Can
            be altered as desired by running ``max_expected_height = number``.
            Depending on the atmospheric model, this value may be automatically
            modified.

        Returns
        -------
        None
        """
        # Initialize constants and atmospheric variables
        self.__initialize_empty_variables()
        self.__initialize_constants()
        self.__initialize_elevation_and_max_height(elevation, max_expected_height)

        # Initialize plots and prints objects
        self.prints = _EnvironmentPrints(self)
        self.plots = _EnvironmentPlots(self)

        # Set the atmosphere model to the standard atmosphere
        self.set_atmospheric_model("standard_atmosphere")

        # Initialize date, latitude, longitude, and Earth geometry
        self.__initialize_date(date, timezone)
        self.set_location(latitude, longitude)
        self.__initialize_earth_geometry(datum)
        self.__initialize_utm_coordinates()
        self.__set_earth_rotation_vector()

        # Set the gravity model
        self.gravity = self.set_gravity_model(gravity)

    def __initialize_constants(self):
        """Sets some important constants and atmospheric variables."""
        self.earth_radius = 6.3781 * (10**6)
        self.air_gas_constant = 287.05287  # in J/K/kg
        self.standard_g = 9.80665
        self.__weather_model_map = WeatherModelMapping()
        self.__atm_type_file_to_function_map = {
            "forecast": {
                "GFS": fetch_gfs_file_return_dataset,
                "NAM": fetch_nam_file_return_dataset,
                "RAP": fetch_rap_file_return_dataset,
                "HIRESW": fetch_hiresw_file_return_dataset,
            },
            "ensemble": {
                "GEFS": fetch_gefs_ensemble,
            },
        }
        self.__standard_atmosphere_layers = {
            "geopotential_height": [  # in geopotential m
                -2e3,
                0,
                11e3,
                20e3,
                32e3,
                47e3,
                51e3,
                71e3,
                80e3,
            ],
            "temperature": [  # in K
                301.15,
                288.15,
                216.65,
                216.65,
                228.65,
                270.65,
                270.65,
                214.65,
                196.65,
            ],
            "beta": [-6.5e-3, -6.5e-3, 0, 1e-3, 2.8e-3, 0, -2.8e-3, -2e-3, 0],  # in K/m
            "pressure": [  # in Pa
                1.27774e5,
                1.01325e5,
                2.26320e4,
                5.47487e3,
                8.680164e2,
                1.10906e2,
                6.69384e1,
                3.95639e0,
                8.86272e-2,
            ],
        }

    def __initialize_empty_variables(self):
        self.atmospheric_model_file = str()
        self.atmospheric_model_dict = {}

    def __initialize_elevation_and_max_height(self, elevation, max_expected_height):
        """Saves the elevation and the maximum expected height."""
        self.elevation = elevation
        self.set_elevation(elevation)
        self._max_expected_height = max_expected_height

    def __initialize_date(self, date, timezone):
        """Saves the date and configure timezone."""
        if date is not None:
            self.set_date(date, timezone)
        else:
            self.date = None
            self.datetime_date = None
            self.local_date = None
            self.timezone = None

    def __initialize_earth_geometry(self, datum):
        """Initialize Earth geometry, save datum and Recalculate Earth Radius"""
        self.datum = datum
        self.ellipsoid = self.set_earth_geometry(datum)
        self.earth_radius = self.calculate_earth_radius(
            lat=self.latitude,
            semi_major_axis=self.ellipsoid.semi_major_axis,
            flattening=self.ellipsoid.flattening,
        )

    def __initialize_utm_coordinates(self):
        """Store launch site coordinates referenced to UTM projection system."""
        if -80 < self.latitude < 84:
            (
                self.initial_east,
                self.initial_north,
                self.initial_utm_zone,
                self.initial_utm_letter,
                self.initial_hemisphere,
                self.initial_ew,
            ) = geodesic_to_utm(
                lat=self.latitude,
                lon=self.longitude,
                flattening=self.ellipsoid.flattening,
                semi_major_axis=self.ellipsoid.semi_major_axis,
            )
        else:  # pragma: no cover
            warnings.warn(
                "UTM coordinates are not available for latitudes "
                "above 84 or below -80 degrees. The UTM conversions will fail."
            )
            self.initial_north = None
            self.initial_east = None
            self.initial_utm_zone = None
            self.initial_utm_letter = None
            self.initial_hemisphere = None
            self.initial_ew = None

    # Auxiliary private setters.

    def __set_pressure_function(self, source):
        self.pressure = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
            interpolation="linear",
        )

    def __set_barometric_height_function(self, source):
        self.barometric_height = Function(
            source,
            inputs="Pressure (Pa)",
            outputs="Height Above Sea Level (m)",
            interpolation="linear",
            extrapolation="natural",
        )
        if callable(self.barometric_height.source):
            # discretize to speed up flight simulation
            self.barometric_height.set_discrete(
                0,
                self.max_expected_height,
                100,
                extrapolation="constant",
                mutate_self=True,
            )

    def __set_temperature_function(self, source):
        self.temperature = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )

    def __set_wind_velocity_x_function(self, source):
        self.wind_velocity_x = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )

    def __set_wind_velocity_y_function(self, source):
        self.wind_velocity_y = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

    def __set_wind_speed_function(self, source):
        self.wind_speed = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )

    def __set_wind_direction_function(self, source):
        self.wind_direction = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )

    def __set_wind_heading_function(self, source):
        self.wind_heading = Function(
            source,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )

    def __reset_barometric_height_function(self):
        # NOTE: this assumes self.pressure and max_expected_height are already set.
        self.barometric_height = self.pressure.inverse_function()
        if callable(self.barometric_height.source):
            # discretize to speed up flight simulation
            self.barometric_height.set_discrete(
                0,
                self.max_expected_height,
                100,
                extrapolation="constant",
                mutate_self=True,
            )
        self.barometric_height.set_inputs("Pressure (Pa)")
        self.barometric_height.set_outputs("Height Above Sea Level (m)")

    def __reset_wind_speed_function(self):
        # NOTE: assume wind_velocity_x and wind_velocity_y as Function objects
        self.wind_speed = (self.wind_velocity_x**2 + self.wind_velocity_y**2) ** 0.5
        self.wind_speed.set_inputs("Height Above Sea Level (m)")
        self.wind_speed.set_outputs("Wind Speed (m/s)")
        self.wind_speed.set_title("Wind Speed Profile")

    # commented because I never finished, leave it for future implementation
    # def __reset_wind_heading_function(self):
    # NOTE: this assumes wind_u and wind_v as numpy arrays with same length.
    # TODO: should we implement arctan2 in the Function class?
    # self.wind_heading = calculate_wind_heading(
    #     self.wind_velocity_x, self.wind_velocity_y
    # )
    # self.wind_heading.set_inputs("Height Above Sea Level (m)")
    # self.wind_heading.set_outputs("Wind Heading (Deg True)")
    # self.wind_heading.set_title("Wind Heading Profile")

    def __reset_wind_direction_function(self):
        self.wind_direction = convert_wind_heading_to_direction(self.wind_heading)
        self.wind_direction.set_inputs("Height Above Sea Level (m)")
        self.wind_direction.set_outputs("Wind Direction (Deg True)")
        self.wind_direction.set_title("Wind Direction Profile")

    def __set_earth_rotation_vector(self):
        """Calculates and stores the Earth's angular velocity vector in the Flight
        Coordinate System, which is essential for evaluating inertial forces.
        """
        # Sidereal day
        T = 86164.1  # seconds

        # Earth's angular velocity magnitude
        w_earth = 2 * np.pi / T

        # Vector in the Flight Coordinate System
        lat = np.radians(self.latitude)
        w_local = [0, w_earth * np.cos(lat), w_earth * np.sin(lat)]

        # Store the attribute
        self.earth_rotation_vector = w_local

    # Validators (used to verify an attribute is being set correctly.)

    def __validate_dictionary(self, file, dictionary):
        # removed CMC until it is fixed.
        available_models = ["GFS", "NAM", "RAP", "HIRESW", "GEFS", "ERA5", "MERRA2"]
        if isinstance(dictionary, str):
            dictionary = self.__weather_model_map.get(dictionary)
        elif file in available_models:
            dictionary = self.__weather_model_map.get(file)
        if not isinstance(dictionary, dict):
            raise TypeError(
                "Please specify a dictionary or choose a valid model from the "
                f"following list: {available_models}"
            )

        return dictionary

    def __validate_datetime(self):
        if self.datetime_date is None:
            raise ValueError(
                "Please specify the launch date and time using the "
                "Environment.set_date() method."
            )

    # Define setters

    def set_date(self, date, timezone="UTC"):
        """Set date and time of launch and update weather conditions if
        date dependent atmospheric model is used.

        Parameters
        ----------
        date : list, tuple, datetime
            List or tuple of length 4, stating (year, month, day, hour) in the
            time zone of the parameter ``timezone``. See Notes for more
            information. Alternatively, can be a ``datetime`` object specifying
            launch date and time.
        timezone : string, optional
            Name of the time zone. To see all time zones, import pytz and run
            ``print(pytz.all_timezones)``. Default time zone is "UTC".

        Returns
        -------
        None

        Notes
        -----
        - If the ``date`` is given as a list or tuple, it should be in the same
          time zone as specified by the ``timezone`` parameter. This local
          time will be available in the attribute :attr:`Environment.local_date`
          while the UTC time will be available in the attribute
          :attr:`Environment.datetime_date`.

        - If the ``date`` is given as a ``datetime`` object without a time zone,
          it will be assumed to be in the same time zone as specified by the
          ``timezone`` parameter. However, if the ``datetime`` object has a time
          zone specified in its ``tzinfo`` attribute, the ``timezone``
          parameter will be ignored.

        Examples
        --------

        Let's set the launch date as an list:

        >>> date = [2000, 1, 1, 13] # January 1st, 2000 at 13:00 UTC+1
        >>> env = Environment()
        >>> env.set_date(date, timezone="Europe/Rome")
        >>> print(env.datetime_date) # Get UTC time
        2000-01-01 12:00:00+00:00
        >>> print(env.local_date)
        2000-01-01 13:00:00+01:00

        Now let's set the launch date as a ``datetime`` object:

        >>> from datetime import datetime
        >>> date = datetime(2000, 1, 1, 13, 0, 0)
        >>> env = Environment()
        >>> env.set_date(date, timezone="Europe/Rome")
        >>> print(env.datetime_date) # Get UTC time
        2000-01-01 12:00:00+00:00
        >>> print(env.local_date)
        2000-01-01 13:00:00+01:00
        """
        # Store date and configure time zone
        self.timezone = timezone
        tz = pytz.timezone(self.timezone)
        if not isinstance(date, datetime):
            local_date = datetime(*date)
        else:
            local_date = date
        if local_date.tzinfo is None:
            local_date = tz.localize(local_date)
        self.date = date
        self.local_date = local_date
        self.datetime_date = self.local_date.astimezone(pytz.UTC)

        # Update atmospheric conditions if atmosphere type is Forecast,
        # Reanalysis or Ensemble
        if hasattr(self, "atmospheric_model_type") and self.atmospheric_model_type in [
            "Forecast",
            "Reanalysis",
            "Ensemble",
        ]:
            self.set_atmospheric_model(
                type=self.atmospheric_model_type,
                file=self.atmospheric_model_file,
                dictionary=self.atmospheric_model_dict,
            )

    def set_location(self, latitude, longitude):
        """Set latitude and longitude of launch and update atmospheric
        conditions if location dependent model is being used.

        Parameters
        ----------
        latitude : float
            Latitude of launch site. May range from -90 to 90 degrees.
        longitude : float
            Longitude of launch site. Either from 0 to 360 degrees or from -180
            to 180 degrees.

        Returns
        -------
        None
        """

        if not isinstance(latitude, NUMERICAL_TYPES) and isinstance(
            longitude, NUMERICAL_TYPES
        ):  # pragma: no cover
            raise TypeError("Latitude and Longitude must be numbers!")

        # Store latitude and longitude
        self.latitude = latitude
        self.longitude = longitude

        # Update atmospheric conditions if atmosphere type is Forecast,
        # Reanalysis or Ensemble
        if hasattr(self, "atmospheric_model_type") and self.atmospheric_model_type in [
            "Forecast",
            "Reanalysis",
            "Ensemble",
        ]:
            self.set_atmospheric_model(
                type=self.atmospheric_model_type,
                file=self.atmospheric_model_file,
                dictionary=self.atmospheric_model_dict,
            )

    def set_gravity_model(self, gravity=None):
        """Defines the gravity model based on the given user input to the
        gravity parameter. The gravity model is responsible for computing the
        gravity acceleration at a given height above sea level in meters.

        Parameters
        ----------
        gravity : int, float, callable, string, list, optional
            The gravitational acceleration in m/s² to be used in the
            simulation, this value is positive when pointing downwards.
            The input type can be one of the following:

            - ``int`` or ``float``: The gravity acceleration is set as a\
              constant function with respect to height;

            - ``callable``: This callable should receive the height above\
              sea level in meters and return the gravity acceleration;

            - ``list``: The datapoints should be structured as\
              ``[(h_i,g_i), ...]`` where ``h_i`` is the height above sea\
              level in meters and ``g_i`` is the gravity acceleration in m/s²;

            - ``string``: The string should correspond to a path to a CSV file\
              containing the gravity acceleration data;

            - ``None``: The Somigliana formula is used to compute the gravity\
              acceleration.

            This parameter is used as a :class:`Function` object source, check\
            out the available input types for a more detailed explanation.

        Returns
        -------
        Function
            Function object representing the gravity model.

        Notes
        -----
        This method **does not** set the gravity acceleration, it only returns
        a :class:`Function` object representing the gravity model.

        Examples
        --------
        Let's prepare a `Environment` object with a constant gravity
        acceleration:

        >>> g_0 = 9.80665
        >>> env_cte_g = Environment(gravity=g_0)
        >>> env_cte_g.gravity([0, 100, 1000])
        [np.float64(9.80665), np.float64(9.80665), np.float64(9.80665)]

        It's also possible to variate the gravity acceleration by defining
        its function of height:

        >>> R_t = 6371000
        >>> g_func = lambda h : g_0 * (R_t / (R_t + h))**2
        >>> env_var_g = Environment(gravity=g_func)
        >>> g = env_var_g.gravity(1000)
        >>> print(f"{g:.6f}")
        9.803572
        """
        if gravity is None:
            return self.somigliana_gravity.set_discrete(
                0, self.max_expected_height, 100
            )
        else:
            return Function(gravity, "height (m)", "gravity (m/s²)").set_discrete(
                0, self.max_expected_height, 100
            )

    @property
    def max_expected_height(self):
        return self._max_expected_height

    @max_expected_height.setter
    def max_expected_height(self, value):
        if value < self.elevation:  # pragma: no cover
            raise ValueError(
                "Max expected height cannot be lower than the surface elevation"
            )
        self._max_expected_height = value
        self.plots.grid = np.linspace(self.elevation, self.max_expected_height)

    @funcify_method("height (m)", "gravity (m/s²)")
    def somigliana_gravity(self, height):
        """Computes the gravity acceleration with the Somigliana formula [1]_.
        An height correction is applied to the normal gravity that is
        accurate for heights used in aviation. The formula is based on the
        WGS84 ellipsoid, but is accurate for other reference ellipsoids.

        Parameters
        ----------
        height : float
            Height above the reference ellipsoid in meters.

        Returns
        -------
        Function
            Function object representing the gravity model.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation
        """
        a = 6378137.0  # semi_major_axis
        f = 1 / 298.257223563  # flattening_factor
        m_rot = 3.449786506841e-3  # rotation_factor
        g_e = 9.7803253359  # normal gravity at equator
        k_somgl = 1.931852652458e-3  # normal gravity formula const.
        first_ecc_sqrd = 6.694379990141e-3  # square of first eccentricity

        # Compute quantities
        sin_lat_sqrd = (np.sin(self.latitude * np.pi / 180)) ** 2

        gravity_somgl = g_e * (
            (1 + k_somgl * sin_lat_sqrd) / (np.sqrt(1 - first_ecc_sqrd * sin_lat_sqrd))
        )
        height_correction = (
            1
            - height * 2 / a * (1 + f + m_rot - 2 * f * sin_lat_sqrd)
            + 3 * height**2 / a**2
        )

        return height_correction * gravity_somgl

    def set_elevation(self, elevation="Open-Elevation"):
        """Set elevation of launch site given user input or using the
        Open-Elevation API.

        Parameters
        ----------
        elevation : float, string, optional
            Elevation of launch site measured as height above sea level in
            meters. Alternatively, can be set as ``Open-Elevation`` which uses
            the Open-Elevation API to find elevation data. For this option,
            latitude and longitude must have already been specified.

            See Also
            --------
            :meth:`rocketpy.Environment.set_location`

        Returns
        -------
        None
        """
        if elevation not in ["Open-Elevation", "SRTM"]:
            # NOTE: this is assuming the elevation is a number (i.e. float, int, etc.)
            self.elevation = elevation
        else:
            self.elevation = fetch_open_elevation(self.latitude, self.longitude)
            print(f"Elevation received: {self.elevation} m")

    def set_topographic_profile(  # pylint: disable=redefined-builtin, unused-argument
        self, type, file, dictionary="netCDF4", crs=None
    ):
        """[UNDER CONSTRUCTION] Defines the Topographic profile, importing data
        from previous downloaded files. Mainly data from the Shuttle Radar
        Topography Mission (SRTM) and NASA Digital Elevation Model will be used
        but other models and methods can be implemented in the future.
        So far, this function can only handle data from NASADEM, available at:
        https://cmr.earthdata.nasa.gov/search/concepts/C1546314436-LPDAAC_ECS.html

        Parameters
        ----------
        type : string
            Defines the topographic model to be used, usually 'NASADEM Merged
            DEM Global 1 arc second nc' can be used. To download this kind of
            data, access 'https://search.earthdata.nasa.gov/search'.
            NASADEM data products were derived from original telemetry data from
            the Shuttle Radar Topography Mission (SRTM).
        file : string
            The path/name of the topographic file. Usually .nc provided by
        dictionary : string, optional
            Dictionary which helps to read the specified file. By default
            'netCDF4' which works well with .nc files will be used.
        crs : string, optional
            Coordinate reference system, by default None, which will use the crs
            provided by the file.
        """

        if type == "NASADEM_HGT":
            if dictionary == "netCDF4":
                nasa_dem = netCDF4.Dataset(file, "r", format="NETCDF4")
                self.elev_lon_array = nasa_dem.variables["lon"][:].tolist()
                self.elev_lat_array = nasa_dem.variables["lat"][:].tolist()
                self.elev_array = nasa_dem.variables["NASADEM_HGT"][:].tolist()
                # crsArray = nasa_dem.variables['crs'][:].tolist().
                self.topographic_profile_activated = True

                print("Region covered by the Topographical file: ")
                print(
                    f"Latitude from {self.elev_lat_array[-1]:.6f}° to "
                    f"{self.elev_lat_array[0]:.6f}°"
                )
                print(
                    f"Longitude from {self.elev_lon_array[0]:.6f}° to "
                    f"{self.elev_lon_array[-1]:.6f}°"
                )

    def get_elevation_from_topographic_profile(self, lat, lon):
        """Function which receives as inputs the coordinates of a point and
        finds its elevation in the provided Topographic Profile.

        Parameters
        ----------
        lat : float
            latitude of the point.
        lon : float
            longitude of the point.

        Returns
        -------
        elevation : float | int
            Elevation provided by the topographic data, in meters.
        """
        # TODO: refactor this method.  pylint: disable=too-many-statements
        if self.topographic_profile_activated is False:  # pragma: no cover
            raise ValueError(
                "You must define a Topographic profile first, please use the "
                "Environment.set_topographic_profile() method first."
            )

        # Find latitude index
        # Check if reversed or sorted
        if self.elev_lat_array[0] < self.elev_lat_array[-1]:
            # Deal with sorted self.elev_lat_array
            lat_index = bisect.bisect(self.elev_lat_array, lat)
        else:
            # Deal with reversed self.elev_lat_array
            self.elev_lat_array.reverse()
            lat_index = len(self.elev_lat_array) - bisect.bisect_left(
                self.elev_lat_array, lat
            )
            self.elev_lat_array.reverse()
        # Take care of latitude value equal to maximum longitude in the grid
        if (
            lat_index == len(self.elev_lat_array)
            and self.elev_lat_array[lat_index - 1] == lat
        ):
            lat_index = lat_index - 1
        # Check if latitude value is inside the grid
        if lat_index == 0 or lat_index == len(self.elev_lat_array):
            raise ValueError(
                f"Latitude {lat} not inside region covered by file, which is from "
                f"{self.elev_lat_array[0]} to {self.elev_lat_array[-1]}."
            )

        # Find longitude index
        # Determine if file uses -180 to 180 or 0 to 360
        if self.elev_lon_array[0] < 0 or self.elev_lon_array[-1] < 0:
            # Convert input to -180 - 180
            lon = lon if lon < 180 else -180 + lon % 180
        else:
            # Convert input to 0 - 360
            lon = lon % 360
        # Check if reversed or sorted
        if self.elev_lon_array[0] < self.elev_lon_array[-1]:
            # Deal with sorted self.elev_lon_array
            lon_index = bisect.bisect(self.elev_lon_array, lon)
        else:
            # Deal with reversed self.elev_lon_array
            self.elev_lon_array.reverse()
            lon_index = len(self.elev_lon_array) - bisect.bisect_left(
                self.elev_lon_array, lon
            )
            self.elev_lon_array.reverse()
        # Take care of longitude value equal to maximum longitude in the grid
        if (
            lon_index == len(self.elev_lon_array)
            and self.elev_lon_array[lon_index - 1] == lon
        ):
            lon_index = lon_index - 1
        # Check if longitude value is inside the grid
        if lon_index == 0 or lon_index == len(self.elev_lon_array):
            raise ValueError(
                f"Longitude {lon} not inside region covered by file, which is from "
                f"{self.elev_lon_array[0]} to {self.elev_lon_array[-1]}."
            )

        # Get the elevation
        elevation = self.elev_array[lat_index][lon_index]

        return elevation

    def set_atmospheric_model(  # pylint: disable=too-many-statements
        self,
        type,  # pylint: disable=redefined-builtin
        file=None,
        dictionary=None,
        pressure=None,
        temperature=None,
        wind_u=0,
        wind_v=0,
    ):
        """Defines an atmospheric model for the Environment. Supported
        functionality includes using data from the `International Standard
        Atmosphere`, importing data from weather reanalysis, forecasts and
        ensemble forecasts, importing data from upper air soundings and
        inputting data as custom functions, arrays or csv files.

        Parameters
        ----------
        type : string
            One of the following options:

            - ``standard_atmosphere``: sets pressure and temperature profiles
              corresponding to the International Standard Atmosphere defined by
              ISO 2533 and ranging from -2 km to 80 km of altitude above sea
              level. Note that the wind profiles are set to zero when this type
              is chosen.

            - ``wyoming_sounding``: sets pressure, temperature, wind-u
              and wind-v profiles and surface elevation obtained from
              an upper air sounding given by the file parameter through
              an URL. This URL should point to a data webpage given by
              selecting plot type as text: list, a station and a time at
              `weather.uwyo`_.
              An example of a valid link would be:

              http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0200&TO=0200&STNM=82599

              .. _weather.uwyo: http://weather.uwyo.edu/upperair/sounding.html

            - ``windy_atmosphere``: sets pressure, temperature, wind-u and
              wind-v profiles and surface elevation obtained from the Windy API.
              See file argument to specify the model as either ``ECMWF``,
              ``GFS`` or ``ICON``.

            - ``Forecast``: sets pressure, temperature, wind-u and wind-v
              profiles and surface elevation obtained from a weather forecast
              file in ``netCDF`` format or from an ``OPeNDAP`` URL, both given
              through the file parameter. When this type is chosen, the date
              and location of the launch should already have been set through
              the date and location parameters when initializing the
              Environment. The ``netCDF`` and ``OPeNDAP`` datasets must contain
              at least geopotential height or geopotential, temperature, wind-u
              and wind-v profiles as a function of pressure levels. If surface
              geopotential or geopotential height is given, elevation is also
              set. Otherwise, elevation is not changed. Profiles are
              interpolated bi-linearly using supplied latitude and longitude.
              The date used is the nearest one to the date supplied.
              Furthermore, a dictionary must be supplied through the dictionary
              parameter in order for the dataset to be accurately read. Lastly,
              the dataset must use a rectangular grid sorted in either ascending
              or descending order of latitude and longitude.

            - ``Reanalysis``: sets pressure, temperature, wind-u and wind-v
              profiles and surface elevation obtained from a weather forecast
              file in ``netCDF`` format or from an ``OPeNDAP`` URL, both given
              through the file parameter. When this type is chosen, the date and
              location of the launch should already have been set through the
              date and location parameters when initializing the Environment.
              The ``netCDF`` and ``OPeNDAP`` datasets must contain at least
              geopotential height or geopotential, temperature, wind-u and
              wind-v profiles as a function of pressure levels. If surface
              geopotential or geopotential height is given, elevation is also
              set. Otherwise, elevation is not changed. Profiles are
              interpolated bi-linearly using supplied latitude and longitude.
              The date used is the nearest one to the date supplied.
              Furthermore, a dictionary must be supplied through the dictionary
              parameter in order for the dataset to be accurately read. Lastly,
              the dataset must use a rectangular grid sorted in either ascending
              or descending order of latitude and longitude.

            - ``Ensemble``: sets pressure, temperature, wind-u and wind-v
              profiles and surface elevation obtained from a weather forecast
              file in ``netCDF`` format or from an ``OPeNDAP`` URL, both given
              through the file parameter. When this type is chosen, the date and
              location of the launch should already have been set through the
              date and location parameters when initializing the Environment.
              The ``netCDF`` and ``OPeNDAP`` datasets must contain at least
              geopotential height or geopotential, temperature, wind-u and
              wind-v profiles as a function of pressure levels. If surface
              geopotential or geopotential height is given, elevation is also
              set. Otherwise, elevation is not changed. Profiles are
              interpolated bi-linearly using supplied latitude and longitude.
              The date used is the nearest one to the date supplied.
              Furthermore, a dictionary must be supplied through the dictionary
              parameter in order for the dataset to be accurately read. Lastly,
              the dataset must use a rectangular grid sorted in either ascending
              or descending order of latitude and longitude. By default the
              first ensemble forecast is activated.

              .. seealso::

                To activate other ensemble forecasts see
                :meth:`rocketpy.Environment.select_ensemble_member`.

            - ``custom_atmosphere``: sets pressure, temperature, wind-u and
              wind-v profiles given though the pressure, temperature, wind-u and
              wind-v parameters of this method. If pressure or temperature is
              not given, it will default to the `International Standard
              Atmosphere`. If the wind components are not given, it will default
              to 0.

        file : string, optional
            String that must be given when type is either ``wyoming_sounding``,
            ``Forecast``, ``Reanalysis``, ``Ensemble`` or ``Windy``. It
            specifies the location of the data given, either through a local
            file address or a URL. If type is ``Forecast``, this parameter can
            also be either ``GFS``, ``FV3``, ``RAP`` or ``NAM`` for latest of
            these forecasts.

            .. note::

                Time reference for the Forecasts are:

                - ``GFS``: `Global` - 0.25deg resolution - Updates every 6
                  hours, forecast for 81 points spaced by 3 hours
                - ``RAP``: `Regional USA` - 0.19deg resolution - Updates hourly,
                  forecast for 40 points spaced hourly
                - ``NAM``: `Regional CONUS Nest` - 5 km resolution - Updates
                  every 6 hours, forecast for 21 points spaced by 3 hours

            If type is ``Ensemble``, this parameter can also be ``GEFS``
            for the latest of this ensemble.

            .. note::

                Time referece for the Ensembles are:

                - GEFS: Global, bias-corrected, 0.5deg resolution, 21 forecast
                  members, Updates every 6 hours, forecast for 65 points spaced
                  by 4 hours
                - CMC (currently not available): Global, 0.5deg resolution, 21 \
                  forecast members, Updates every 12 hours, forecast for 65 \
                  points spaced by 4 hours

            If type is ``Windy``, this parameter can be either ``GFS``,
            ``ECMWF``, ``ICON`` or ``ICONEU``. Default in this case is ``ECMWF``.
        dictionary : dictionary, string, optional
            Dictionary that must be given when type is either ``Forecast``,
            ``Reanalysis`` or ``Ensemble``. It specifies the dictionary to be
            used when reading ``netCDF`` and ``OPeNDAP`` files, allowing the
            correct retrieval of data. Acceptable values include ``ECMWF``,
            ``NOAA``, ``UCAR`` and ``MERRA2`` for default dictionaries which can generally
            be used to read datasets from these institutes. Alternatively, a
            dictionary structure can also be given, specifying the short names
            used for time, latitude, longitude, pressure levels, temperature
            profile, geopotential or geopotential height profile, wind-u and
            wind-v profiles in the dataset given in the file parameter.
            Additionally, ensemble dictionaries must have the ensemble as well.
            An example is the following dictionary, used for ``NOAA``:

            .. code-block:: python

                dictionary = {
                    "time": "time",
                    "latitude": "lat",
                    "longitude": "lon",
                    "level": "lev",
                    "ensemble": "ens",
                    "temperature": "tmpprs",
                    "surface_geopotential_height": "hgtsfc",
                    "geopotential_height": "hgtprs",
                    "geopotential": None,
                    "u_wind": "ugrdprs",
                    "v_wind": "vgrdprs",
                }

        pressure : float, string, array, callable, optional
            This defines the atmospheric pressure profile.
            Should be given if the type parameter is ``custom_atmosphere``. If not,
            than the the ``Standard Atmosphere`` pressure will be used.
            If a float is given, it will define a constant pressure
            profile. The float should be in units of Pa.
            If a string is given, it should point to a `.CSV` file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the pressure in Pa.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, pressure in Pa).
            Finally, a callable or function is also accepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding pressure in Pa.
        temperature : float, string, array, callable, optional
            This defines the atmospheric temperature profile. Should be given
            if the type parameter is ``custom_atmosphere``. If not, than the the
            ``Standard Atmosphere`` temperature will be used. If a float is
            given, it will define a constant temperature profile. The float
            should be in units of K. If a string is given, it should point to a
            `.CSV` file containing at most one header line and two columns of
            data. The first column must be the geometric height above sea level
            in meters while the second column must be the temperature in K.
            If an array is given, it is expected to be a list or array of
            coordinates (height in meters, temperature in K). Finally, a
            callable or function is also accepted. The function should take one
            argument, the height above sea level in meters and return a
            corresponding temperature in K.
        wind_u : float, string, array, callable, optional
            This defines the atmospheric wind-u profile, corresponding the
            magnitude of the wind speed heading East. Should be given if the
            type parameter is ``custom_atmosphere``. If not, it will be assumed
            to be constant and equal to 0. If a float is given, it will define
            a constant wind-u profile. The float should be in units of m/s. If a
            string is given, it should point to a .CSV file containing at most
            one header line and two columns of data. The first column must be
            the geometric height above sea level in meters while the second
            column must be the wind-u in m/s. If an array is given, it is
            expected to be an array of coordinates (height in meters,
            wind-u in m/s). Finally, a callable or function is also accepted.
            The function should take one argument, the height above sea level in
            meters and return a corresponding wind-u in m/s.
        wind_v : float, string, array, callable, optional
            This defines the atmospheric wind-v profile, corresponding the
            magnitude of the wind speed heading North. Should be given if the
            type parameter is ``custom_atmosphere``. If not, it will be assumed
            to be constant and equal to 0. If a float is given, it will define a
            constant wind-v profile. The float should be in units of m/s. If a
            string is given, it should point to a .CSV file containing at most
            one header line and two columns of data. The first column must be
            the geometric height above sea level in meters while the second
            column must be the wind-v in m/s. If an array is given, it is
            expected to be an array of coordinates (height in meters, wind-v in
            m/s). Finally, a callable or function is also accepted. The function
            should take one argument, the height above sea level in meters and
            return a corresponding wind-v in m/s.

        Returns
        -------
        None
        """
        # Save atmospheric model type
        self.atmospheric_model_type = type
        type = type.lower()

        match type:
            case "standard_atmosphere":
                self.process_standard_atmosphere()
            case "wyoming_sounding":
                self.process_wyoming_sounding(file)
            case "custom_atmosphere":
                self.process_custom_atmosphere(pressure, temperature, wind_u, wind_v)
            case "windy":
                self.process_windy_atmosphere(file)
            case "forecast" | "reanalysis" | "ensemble":
                dictionary = self.__validate_dictionary(file, dictionary)
                try:
                    fetch_function = self.__atm_type_file_to_function_map[type][file]
                except KeyError:
                    fetch_function = None

                # Fetches the dataset using OpenDAP protocol or uses the file path
                dataset = fetch_function() if fetch_function is not None else file

                if type in ["forecast", "reanalysis"]:
                    self.process_forecast_reanalysis(dataset, dictionary)
                else:
                    self.process_ensemble(dataset, dictionary)
            case _:  # pragma: no cover
                raise ValueError(f"Unknown model type '{type}'.")

        if type not in ["ensemble"]:
            # Ensemble already computed these values
            self.calculate_density_profile()
            self.calculate_speed_of_sound_profile()
            self.calculate_dynamic_viscosity()

        # Save dictionary and file
        self.atmospheric_model_file = file
        self.atmospheric_model_dict = dictionary

    # Atmospheric model processing methods

    def process_standard_atmosphere(self):
        """Sets pressure and temperature profiles corresponding to the
        International Standard Atmosphere defined by ISO 2533 and
        ranging from -2 km to 80 km of altitude above sea level. Note
        that the wind profiles are set to zero.

        Returns
        -------
        None
        """
        # Save temperature, pressure and wind profiles
        self.pressure = self.pressure_ISA
        self.barometric_height = self.barometric_height_ISA
        self.temperature = self.temperature_ISA

        # Set wind profiles to zero
        self.__set_wind_direction_function(0)
        self.__set_wind_heading_function(0)
        self.__set_wind_velocity_x_function(0)
        self.__set_wind_velocity_y_function(0)
        self.__set_wind_speed_function(0)

        # 80k meters is the limit of the standard atmosphere
        self._max_expected_height = 80000

    def process_custom_atmosphere(
        self, pressure=None, temperature=None, wind_u=0, wind_v=0
    ):
        """Import pressure, temperature and wind profile given by user.

        Parameters
        ----------
        pressure : float, string, array, callable, optional
            This defines the atmospheric pressure profile.
            Should be given if the type parameter is ``custom_atmosphere``.
            If not, than the the Standard Atmosphere pressure will be used.
            If a float is given, it will define a constant pressure
            profile. The float should be in units of Pa.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the pressure in Pa.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, pressure in Pa).
            Finally, a callable or function is also accepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding pressure in Pa.
        temperature : float, string, array, callable, optional
            This defines the atmospheric temperature profile.
            Should be given if the type parameter is ``custom_atmosphere``.
            If not, than the the Standard Atmosphere temperature will be used.
            If a float is given, it will define a constant temperature
            profile. The float should be in units of K.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the temperature in K.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, temperature in K).
            Finally, a callable or function is also accepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding temperature in K.
        wind_u : float, string, array, callable, optional
            This defines the atmospheric wind-u profile, corresponding
            the the magnitude of the wind speed heading East.
            Should be given if the type parameter is ``custom_atmosphere``.
            If not, it will be assumed constant and 0.
            If a float is given, it will define a constant wind-u
            profile. The float should be in units of m/s.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the wind-u in m/s.
            If an array is given, it is expected to be an array of
            coordinates (height in meters, wind-u in m/s).
            Finally, a callable or function is also accepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding wind-u in m/s.
        wind_v : float, string, array, callable, optional
            This defines the atmospheric wind-v profile, corresponding
            the the magnitude of the wind speed heading North.
            Should be given if the type parameter is ``custom_atmosphere``.
            If not, it will be assumed constant and 0.
            If a float is given, it will define a constant wind-v
            profile. The float should be in units of m/s.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the wind-v in m/s.
            If an array is given, it is expected to be an array of
            coordinates (height in meters, wind-v in m/s).
            Finally, a callable or function is also accepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding wind-v in m/s.

        Return
        ------
        None
        """
        # Initialize an estimate of the maximum expected atmospheric model height
        max_expected_height = self.max_expected_height or 1000

        # Save pressure profile
        if pressure is None:
            # Use standard atmosphere
            self.pressure = self.pressure_ISA
            self.barometric_height = self.barometric_height_ISA
        else:
            # Use custom input
            self.__set_pressure_function(pressure)
            self.__reset_barometric_height_function()

            # Check maximum height of custom pressure input
            if not callable(self.pressure.source):
                max_expected_height = max(self.pressure[-1, 0], max_expected_height)

        # Save temperature profile
        if temperature is None:
            # Use standard atmosphere
            self.temperature = self.temperature_ISA
        else:
            self.__set_temperature_function(temperature)
            # Check maximum height of custom temperature input
            if not callable(self.temperature.source):
                max_expected_height = max(self.temperature[-1, 0], max_expected_height)

        # Save wind profile
        self.__set_wind_velocity_x_function(wind_u)
        self.__set_wind_velocity_y_function(wind_v)
        # Check maximum height of custom wind input
        if not callable(self.wind_velocity_x.source):
            max_expected_height = max(self.wind_velocity_x[-1, 0], max_expected_height)

        def wind_heading_func(h):  # TODO: create another custom reset for heading
            return calculate_wind_heading(
                self.wind_velocity_x.get_value_opt(h),
                self.wind_velocity_y.get_value_opt(h),
            )

        self.__set_wind_heading_function(wind_heading_func)

        self.__reset_wind_direction_function()
        self.__reset_wind_speed_function()

        self._max_expected_height = max_expected_height

    def process_windy_atmosphere(self, model="ECMWF"):  # pylint: disable=too-many-statements
        """Process data from Windy.com to retrieve atmospheric forecast data.

        Parameters
        ----------
        model : string, optional
            The atmospheric model to use. Default is ``ECMWF``. Options are:
            ``ECMWF`` for the `ECMWF-HRES` model, ``GFS`` for the `GFS` model,
            ``ICON`` for the `ICON-Global` model or ``ICONEU`` for the `ICON-EU`
            model.
        """

        if model.lower() not in ["ecmwf", "gfs", "icon", "iconeu"]:
            raise ValueError(
                f"Invalid model '{model}'. "
                "Valid options are 'ECMWF', 'GFS', 'ICON' or 'ICONEU'."
            )

        response = fetch_atmospheric_data_from_windy(
            self.latitude, self.longitude, model
        )

        # Determine time index from model
        time_array = np.array(response["data"]["hours"])
        time_units = "milliseconds since 1970-01-01 00:00:00"
        launch_time_in_units = netCDF4.date2num(self.datetime_date, time_units)
        # Find the index of the closest time in time_array to the launch time
        time_index = (np.abs(time_array - launch_time_in_units)).argmin()

        # Define available pressure levels
        pressure_levels = np.array(
            [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150]
        )

        # Process geopotential height array
        (
            geopotential_height_array,
            altitude_array,
            temperature_array,
            wind_u_array,
            wind_v_array,
        ) = self.__parse_windy_file(response, time_index, pressure_levels)

        # Determine wind speed, heading and direction
        wind_speed_array = calculate_wind_speed(wind_u_array, wind_v_array)
        wind_heading_array = calculate_wind_heading(wind_u_array, wind_v_array)
        wind_direction_array = convert_wind_heading_to_direction(wind_heading_array)

        # Combine all data into big array
        data_array = mask_and_clean_dataset(
            100 * pressure_levels,  # Convert hPa to Pa
            altitude_array,
            temperature_array,
            wind_u_array,
            wind_v_array,
            wind_heading_array,
            wind_direction_array,
            wind_speed_array,
        )

        # Save atmospheric data
        self.__set_pressure_function(data_array[:, (1, 0)])
        self.__set_barometric_height_function(data_array[:, (0, 1)])
        self.__set_temperature_function(data_array[:, (1, 2)])
        self.__set_wind_velocity_x_function(data_array[:, (1, 3)])
        self.__set_wind_velocity_y_function(data_array[:, (1, 4)])
        self.__set_wind_heading_function(data_array[:, (1, 5)])
        self.__set_wind_direction_function(data_array[:, (1, 6)])
        self.__set_wind_speed_function(data_array[:, (1, 7)])

        # Save maximum expected height
        self._max_expected_height = max(altitude_array[0], altitude_array[-1])

        # Get elevation data from file
        self.elevation = float(response["header"]["elevation"])

        # Compute info data
        self.atmospheric_model_init_date = get_initial_date_from_time_array(
            time_array, time_units
        )
        self.atmospheric_model_end_date = get_final_date_from_time_array(
            time_array, time_units
        )
        self.atmospheric_model_interval = get_interval_date_from_time_array(
            time_array, time_units
        )
        self.atmospheric_model_init_lat = self.latitude
        self.atmospheric_model_end_lat = self.latitude
        self.atmospheric_model_init_lon = self.longitude
        self.atmospheric_model_end_lon = self.longitude

        # Save debugging data
        self.geopotentials = geopotential_height_array
        self.wind_us = wind_u_array
        self.wind_vs = wind_v_array
        self.levels = pressure_levels
        self.temperatures = temperature_array
        self.time_array = time_array
        self.height = altitude_array

    def __parse_windy_file(self, response, time_index, pressure_levels):
        geopotential_height_array = np.array(
            [response["data"][f"gh-{pL}h"][time_index] for pL in pressure_levels]
        )
        # Convert geopotential height to geometric altitude (ASL)
        altitude_array = geopotential_height_to_geometric_height(
            geopotential_height_array, self.earth_radius
        )

        # Process temperature array (in Kelvin)
        temperature_array = np.array(
            [response["data"][f"temp-{pL}h"][time_index] for pL in pressure_levels]
        )

        # Process wind-u and wind-v array (in m/s)
        wind_u_array = np.array(
            [response["data"][f"wind_u-{pL}h"][time_index] for pL in pressure_levels]
        )
        wind_v_array = np.array(
            [response["data"][f"wind_v-{pL}h"][time_index] for pL in pressure_levels]
        )

        return (
            geopotential_height_array,
            altitude_array,
            temperature_array,
            wind_u_array,
            wind_v_array,
        )

    def process_wyoming_sounding(self, file):  # pylint: disable=too-many-statements
        """Import and process the upper air sounding data from `Wyoming
        Upper Air Soundings` database given by the url in file. Sets
        pressure, temperature, wind-u, wind-v profiles and surface elevation.

        Parameters
        ----------
        file : string
            URL of an upper air sounding data output from `Wyoming
            Upper Air Soundings` database.

            Example:

            http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0200&TO=0200&STNM=82599

        Notes
        -----
        More can be found at: http://weather.uwyo.edu/upperair/sounding.html.

        Returns
        -------
        None
        """
        # Request Wyoming Sounding from file url
        response = fetch_wyoming_sounding(file)

        # Process Wyoming Sounding by finding data table and station info
        response_split_text = re.split("(<.{0,1}PRE>)", response.text)
        data_table = response_split_text[2]
        station_info = response_split_text[6]

        # Transform data table into np array
        data_array = []
        for line in data_table.split("\n")[5:-1]:
            # Split data table into lines and remove header and footer
            columns = re.split(" +", line)  # Split line into columns
            # 12 is the number of column entries when all entries are given
            if len(columns) == 12:
                data_array.append(columns[1:])
        data_array = np.array(data_array, dtype=float)

        # Retrieve pressure from data array
        data_array[:, 0] = 100 * data_array[:, 0]  # Converts hPa to Pa
        self.__set_pressure_function(data_array[:, (1, 0)])
        self.__set_barometric_height_function(data_array[:, (0, 1)])

        # Retrieve temperature from data array
        data_array[:, 2] = data_array[:, 2] + 273.15  # Converts C to K
        self.__set_temperature_function(data_array[:, (1, 2)])

        # Retrieve wind-u and wind-v from data array
        ## Converts Knots to m/s
        data_array[:, 7] = data_array[:, 7] * 1.852 / 3.6
        ## Convert wind direction to wind heading
        data_array[:, 5] = (data_array[:, 6] + 180) % 360
        data_array[:, 3] = data_array[:, 7] * np.sin(data_array[:, 5] * np.pi / 180)
        data_array[:, 4] = data_array[:, 7] * np.cos(data_array[:, 5] * np.pi / 180)

        # Convert geopotential height to geometric height
        data_array[:, 1] = geopotential_height_to_geometric_height(
            data_array[:, 1], self.earth_radius
        )

        # Save atmospheric data
        self.__set_wind_velocity_x_function(data_array[:, (1, 3)])
        self.__set_wind_velocity_y_function(data_array[:, (1, 4)])
        self.__set_wind_heading_function(data_array[:, (1, 5)])
        self.__set_wind_direction_function(data_array[:, (1, 6)])
        self.__set_wind_speed_function(data_array[:, (1, 7)])

        # Retrieve station elevation from station info
        station_elevation_text = station_info.split("\n")[6]

        # Convert station elevation text into float value
        self.elevation = float(
            re.findall(r"[0-9]+\.[0-9]+|[0-9]+", station_elevation_text)[0]
        )

        # Save maximum expected height
        self._max_expected_height = data_array[-1, 1]

    def process_forecast_reanalysis(self, file, dictionary):  # pylint: disable=too-many-locals,too-many-statements
        """Import and process atmospheric data from weather forecasts
        and reanalysis given as ``netCDF`` or ``OPeNDAP`` files.
        Sets pressure, temperature, wind-u and wind-v
        profiles and surface elevation obtained from a weather
        file in ``netCDF`` format or from an ``OPeNDAP`` URL, both
        given through the file parameter. The date and location of the launch
        should already have been set through the date and
        location parameters when initializing the Environment.
        The ``netCDF`` and ``OPeNDAP`` datasets must contain at least
        geopotential height or geopotential, temperature,
        wind-u and wind-v profiles as a function of pressure levels.
        If surface geopotential or geopotential height is given,
        elevation is also set. Otherwise, elevation is not changed.
        Profiles are interpolated bi-linearly using supplied
        latitude and longitude. The date used is the nearest one
        to the date supplied. Furthermore, a dictionary must be
        supplied through the dictionary parameter in order for the
        dataset to be accurately read. Lastly, the dataset must use
        a rectangular grid sorted in either ascending or descending
        order of latitude and longitude.

        Parameters
        ----------
        file : string
            String containing path to local ``netCDF`` file or URL of an
            ``OPeNDAP`` file, such as NOAA's NOMAD or UCAR TRHEDDS server.
        dictionary : dictionary
            Specifies the dictionary to be used when reading ``netCDF`` and
            ``OPeNDAP`` files, allowing for the correct retrieval of data.
            The dictionary structure should specify the short names
            used for time, latitude, longitude, pressure levels,
            temperature profile, geopotential or geopotential height
            profile, wind-u and wind-v profiles in the dataset given in
            the file parameter. An example is the following dictionary,
            generally used to read ``OPeNDAP`` files from NOAA's NOMAD
            server:

            .. code-block:: python

                dictionary = {
                    "time": "time",
                    "latitude": "lat",
                    "longitude": "lon",
                    "level": "lev",
                    "temperature": "tmpprs",
                    "surface_geopotential_height": "hgtsfc",
                    "geopotential_height": "hgtprs",
                    "geopotential": None,
                    "u_wind": "ugrdprs",
                    "v_wind": "vgrdprs",
                }

        Returns
        -------
        None
        """
        # Check if date, lat and lon are known
        self.__validate_datetime()

        # Read weather file
        if isinstance(file, str):
            data = netCDF4.Dataset(file)
            if dictionary["time"] not in data.variables.keys():
                dictionary = self.__weather_model_map.get("ECMWF_v0")
        else:
            data = file

        # Get time, latitude and longitude data from file
        time_array = data.variables[dictionary["time"]]
        lon_list = data.variables[dictionary["longitude"]][:].tolist()
        lat_list = data.variables[dictionary["latitude"]][:].tolist()

        # Find time, latitude and longitude indexes
        time_index = find_time_index(self.datetime_date, time_array)
        lon, lon_index = find_longitude_index(self.longitude, lon_list)
        _, lat_index = find_latitude_index(self.latitude, lat_list)

        # Get pressure level data from file
        levels = get_pressure_levels_from_file(data, dictionary)

        # Get geopotential data from file
        try:
            geopotentials = data.variables[dictionary["geopotential_height"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except KeyError:
            try:
                geopotentials = (
                    data.variables[dictionary["geopotential"]][
                        time_index,
                        :,
                        (lat_index - 1, lat_index),
                        (lon_index - 1, lon_index),
                    ]
                    / self.standard_g
                )
            except KeyError as e:
                raise ValueError(
                    "Unable to read geopotential height"
                    " nor geopotential from file. At least"
                    " one of them is necessary. Check "
                    " file and dictionary."
                ) from e

        # Get temperature from file
        try:
            temperatures = data.variables[dictionary["temperature"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except Exception as e:
            raise ValueError(
                "Unable to read temperature from file. Check file and dictionary."
            ) from e

        # Get wind data from file
        try:
            wind_us = data.variables[dictionary["u_wind"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except KeyError as e:
            raise ValueError(
                "Unable to read wind-u component. Check file and dictionary."
            ) from e
        try:
            wind_vs = data.variables[dictionary["v_wind"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except KeyError as e:
            raise ValueError(
                "Unable to read wind-v component. Check file and dictionary."
            ) from e

        # Prepare for bilinear interpolation
        x, y = self.latitude, lon
        x1, y1 = lat_list[lat_index - 1], lon_list[lon_index - 1]
        x2, y2 = lat_list[lat_index], lon_list[lon_index]

        # Determine properties in lat, lon
        height = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            geopotentials[:, 0, 0],
            geopotentials[:, 0, 1],
            geopotentials[:, 1, 0],
            geopotentials[:, 1, 1],
        )
        temper = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            temperatures[:, 0, 0],
            temperatures[:, 0, 1],
            temperatures[:, 1, 0],
            temperatures[:, 1, 1],
        )
        wind_u = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            wind_us[:, 0, 0],
            wind_us[:, 0, 1],
            wind_us[:, 1, 0],
            wind_us[:, 1, 1],
        )
        wind_v = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            wind_vs[:, 0, 0],
            wind_vs[:, 0, 1],
            wind_vs[:, 1, 0],
            wind_vs[:, 1, 1],
        )

        # Determine wind speed, heading and direction
        wind_speed = calculate_wind_speed(wind_u, wind_v)
        wind_heading = calculate_wind_heading(wind_u, wind_v)
        wind_direction = convert_wind_heading_to_direction(wind_heading)

        # Convert geopotential height to geometric height
        height = geopotential_height_to_geometric_height(height, self.earth_radius)

        # Combine all data into big array
        data_array = mask_and_clean_dataset(
            levels,
            height,
            temper,
            wind_u,
            wind_v,
            wind_heading,
            wind_direction,
            wind_speed,
        )
        # Save atmospheric data
        self.__set_pressure_function(data_array[:, (1, 0)])
        self.__set_barometric_height_function(data_array[:, (0, 1)])
        self.__set_temperature_function(data_array[:, (1, 2)])
        self.__set_wind_velocity_x_function(data_array[:, (1, 3)])
        self.__set_wind_velocity_y_function(data_array[:, (1, 4)])
        self.__set_wind_heading_function(data_array[:, (1, 5)])
        self.__set_wind_direction_function(data_array[:, (1, 6)])
        self.__set_wind_speed_function(data_array[:, (1, 7)])

        # Save maximum expected height
        self._max_expected_height = max(height[0], height[-1])

        # Get elevation data from file
        if dictionary.get("surface_geopotential_height") is not None:
            self.elevation = get_elevation_data_from_dataset(
                dictionary, data, time_index, lat_index, lon_index, x, y, x1, x2, y1, y2
            )
        # 2. If not found, try Geopotential (m^2/s^2) and convert
        elif dictionary.get("surface_geopotential") is not None:
            temp_dict = dictionary.copy()
            temp_dict["surface_geopotential_height"] = dictionary[
                "surface_geopotential"
            ]
            surface_geopotential_value = get_elevation_data_from_dataset(
                temp_dict, data, time_index, lat_index, lon_index, x, y, x1, x2, y1, y2
            )
            self.elevation = surface_geopotential_value / self.standard_g

        # Compute info data
        self.atmospheric_model_init_date = get_initial_date_from_time_array(time_array)
        self.atmospheric_model_end_date = get_final_date_from_time_array(time_array)
        if self.atmospheric_model_init_date != self.atmospheric_model_end_date:
            self.atmospheric_model_interval = get_interval_date_from_time_array(
                time_array
            )
        else:
            self.atmospheric_model_interval = 0
        self.atmospheric_model_init_lat = lat_list[0]
        self.atmospheric_model_end_lat = lat_list[-1]
        self.atmospheric_model_init_lon = lon_list[0]
        self.atmospheric_model_end_lon = lon_list[-1]

        # Save debugging data
        self.lat_array = lat_list
        self.lon_array = lon_list
        self.lon_index = lon_index
        self.lat_index = lat_index
        self.geopotentials = geopotentials
        self.wind_us = wind_us
        self.wind_vs = wind_vs
        self.levels = levels
        self.temperatures = temperatures
        self.time_array = time_array[:].tolist()
        self.height = height

        # Close weather data
        data.close()

    def process_ensemble(self, file, dictionary):  # pylint: disable=too-many-locals,too-many-statements
        """Import and process atmospheric data from weather ensembles
        given as ``netCDF`` or ``OPeNDAP`` files. Sets pressure, temperature,
        wind-u and wind-v profiles and surface elevation obtained from a weather
        ensemble file in ``netCDF`` format or from an ``OPeNDAP`` URL, both
        given through the file parameter. The date and location of the launch
        should already have been set through the date and location parameters
        when initializing the Environment. The ``netCDF`` and ``OPeNDAP``
        datasets must contain at least geopotential height or geopotential,
        temperature, wind-u and wind-v profiles as a function of pressure
        levels. If surface geopotential or geopotential height is given,
        elevation is also set. Otherwise, elevation is not changed. Profiles are
        interpolated bi-linearly using supplied latitude and longitude. The date
        used is the nearest one to the date supplied. Furthermore, a dictionary
        must be supplied through the dictionary parameter in order for the
        dataset to be accurately read. Lastly, the dataset must use a
        rectangular grid sorted in either ascending or descending order of
        latitude and longitude. By default the first ensemble forecast is
        activated. To activate other ensemble forecasts see
        :meth:`Environment.select_ensemble_member()`.

        Parameters
        ----------
        file : string
            String containing path to local ``.nc`` file or URL of an
            ``OPeNDAP`` file, such as NOAA's NOMAD or UCAR TRHEDDS server.
        dictionary : dictionary
            Specifies the dictionary to be used when reading ``netCDF`` and
            ``OPeNDAP`` files, allowing for the correct retrieval of data.
            The dictionary structure should specify the short names
            used for time, latitude, longitude, pressure levels,
            temperature profile, geopotential or geopotential height
            profile, wind-u and wind-v profiles in the dataset given in
            the file parameter. An example is the following dictionary,
            generally used to read ``OPeNDAP`` files from NOAA's NOMAD
            server:

            .. code-block:: python

                dictionary = {
                    "time": "time",
                    "latitude": "lat",
                    "longitude": "lon",
                    "level": "lev",
                    "ensemble": "ens",
                    "surface_geopotential_height": "hgtsfc",
                    "geopotential_height": "hgtprs",
                    "geopotential": None,
                    "u_wind": "ugrdprs",
                    "v_wind": "vgrdprs",
                }

        See also
        --------
        See the :class:``rocketpy.environment.weather_model_mapping`` for some
        dictionary examples.
        """
        # Check if date, lat and lon are known
        self.__validate_datetime()

        # Read weather file
        if isinstance(file, str):
            data = netCDF4.Dataset(file)
        else:
            data = file

        # Get time, latitude and longitude data from file
        time_array = data.variables[dictionary["time"]]
        lon_list = data.variables[dictionary["longitude"]][:].tolist()
        lat_list = data.variables[dictionary["latitude"]][:].tolist()

        # Find time, latitude and longitude indexes
        time_index = find_time_index(self.datetime_date, time_array)
        lon, lon_index = find_longitude_index(self.longitude, lon_list)
        _, lat_index = find_latitude_index(self.latitude, lat_list)

        # Get ensemble data from file
        try:
            num_members = len(data.variables[dictionary["ensemble"]][:])
        except KeyError as e:
            raise ValueError(
                "Unable to read ensemble data from file. Check file and dictionary."
            ) from e

        # Get pressure level data from file
        levels = get_pressure_levels_from_file(data, dictionary)

        inverse_dictionary = {v: k for k, v in dictionary.items()}
        param_dictionary = {
            "time": time_index,
            "ensemble": range(num_members),
            "level": range(len(levels)),
            "latitude": (lat_index - 1, lat_index),
            "longitude": (lon_index - 1, lon_index),
        }

        # Get dimensions
        try:
            dimensions = data.variables[dictionary["geopotential_height"]].dimensions[:]
        except KeyError:
            dimensions = data.variables[dictionary["geopotential"]].dimensions[:]

        # Get params
        params = tuple(param_dictionary[inverse_dictionary[dim]] for dim in dimensions)

        # Get geopotential data from file
        try:
            geopotentials = data.variables[dictionary["geopotential_height"]][params]
        except KeyError:
            try:
                geopotentials = (
                    data.variables[dictionary["geopotential"]][params] / self.standard_g
                )
            except KeyError as e:
                raise ValueError(
                    "Unable to read geopotential height nor geopotential from file. "
                    "At least one of them is necessary. Check file and dictionary."
                ) from e

        # Get temperature from file
        try:
            temperatures = data.variables[dictionary["temperature"]][params]
        except KeyError as e:
            raise ValueError(
                "Unable to read temperature from file. Check file and dictionary."
            ) from e

        # Get wind data from file
        try:
            wind_us = data.variables[dictionary["u_wind"]][params]
        except KeyError as e:
            raise ValueError(
                "Unable to read wind-u component. Check file and dictionary."
            ) from e
        try:
            wind_vs = data.variables[dictionary["v_wind"]][params]
        except KeyError as e:
            raise ValueError(
                "Unable to read wind-v component. Check file and dictionary."
            ) from e

        # Prepare for bilinear interpolation
        x, y = self.latitude, lon
        x1, y1 = lat_list[lat_index - 1], lon_list[lon_index - 1]
        x2, y2 = lat_list[lat_index], lon_list[lon_index]

        # Determine properties in lat, lon
        height = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            geopotentials[:, :, 0, 0],
            geopotentials[:, :, 0, 1],
            geopotentials[:, :, 1, 0],
            geopotentials[:, :, 1, 1],
        )
        temper = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            temperatures[:, :, 0, 0],
            temperatures[:, :, 0, 1],
            temperatures[:, :, 1, 0],
            temperatures[:, :, 1, 1],
        )
        wind_u = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            wind_us[:, :, 0, 0],
            wind_us[:, :, 0, 1],
            wind_us[:, :, 1, 0],
            wind_us[:, :, 1, 1],
        )
        wind_v = bilinear_interpolation(
            x,
            y,
            x1,
            x2,
            y1,
            y2,
            wind_vs[:, :, 0, 0],
            wind_vs[:, :, 0, 1],
            wind_vs[:, :, 1, 0],
            wind_vs[:, :, 1, 1],
        )

        # Determine wind speed, heading and direction
        wind_speed = calculate_wind_speed(wind_u, wind_v)
        wind_heading = calculate_wind_heading(wind_u, wind_v)
        wind_direction = convert_wind_heading_to_direction(wind_heading)

        # Convert geopotential height to geometric height
        height = geopotential_height_to_geometric_height(height, self.earth_radius)

        # Save ensemble data
        self.level_ensemble = levels
        self.height_ensemble = height
        self.temperature_ensemble = temper
        self.wind_u_ensemble = wind_u
        self.wind_v_ensemble = wind_v
        self.wind_heading_ensemble = wind_heading
        self.wind_direction_ensemble = wind_direction
        self.wind_speed_ensemble = wind_speed
        self.num_ensemble_members = num_members

        # Activate default ensemble
        self.select_ensemble_member()

        # Get elevation data from file
        if dictionary["surface_geopotential_height"] is not None:
            self.elevation = get_elevation_data_from_dataset(
                dictionary, data, time_index, lat_index, lon_index, x, y, x1, x2, y1, y2
            )

        # Compute info data
        self.atmospheric_model_init_date = get_initial_date_from_time_array(time_array)
        self.atmospheric_model_end_date = get_final_date_from_time_array(time_array)
        self.atmospheric_model_interval = get_interval_date_from_time_array(time_array)
        self.atmospheric_model_init_lat = lat_list[0]
        self.atmospheric_model_end_lat = lat_list[-1]
        self.atmospheric_model_init_lon = lon_list[0]
        self.atmospheric_model_end_lon = lon_list[-1]

        # Save debugging data
        self.lat_array = lat_list
        self.lon_array = lon_list
        self.lon_index = lon_index
        self.lat_index = lat_index
        self.geopotentials = geopotentials
        self.wind_us = wind_us
        self.wind_vs = wind_vs
        self.levels = levels
        self.temperatures = temperatures
        self.time_array = time_array[:].tolist()
        self.height = height

        # Close weather data
        data.close()

    def select_ensemble_member(self, member=0):
        """Activates the specified ensemble member, ensuring all atmospheric
        variables read from the Environment instance correspond to the selected
        ensemble member.

        Parameters
        ----------
        member : int, optional
            The ensemble member to activate. Index starts from 0. Default is 0.

        Raises
        ------
        ValueError
            If the specified ensemble member index is out of range.

        Notes
        -----
        The first ensemble member (index 0) is activated by default when loading
        an ensemble model. This member typically represents a control member
        that is generated without perturbations. Other ensemble members are
        generated by perturbing the control member.
        """
        # Verify ensemble member
        if member >= self.num_ensemble_members:
            raise ValueError(
                f"Please choose member from 0 to {self.num_ensemble_members - 1}"
            )

        # Read ensemble member
        levels = self.level_ensemble[:]
        height = self.height_ensemble[member, :]
        temperature = self.temperature_ensemble[member, :]
        wind_u = self.wind_u_ensemble[member, :]
        wind_v = self.wind_v_ensemble[member, :]
        wind_heading = self.wind_heading_ensemble[member, :]
        wind_direction = self.wind_direction_ensemble[member, :]
        wind_speed = self.wind_speed_ensemble[member, :]

        # Combine all data into big array
        data_array = mask_and_clean_dataset(
            levels,
            height,
            temperature,
            wind_u,
            wind_v,
            wind_heading,
            wind_direction,
            wind_speed,
        )

        # Save atmospheric data
        self.__set_pressure_function(data_array[:, (1, 0)])
        self.__set_barometric_height_function(data_array[:, (0, 1)])
        self.__set_temperature_function(data_array[:, (1, 2)])
        self.__set_wind_velocity_x_function(data_array[:, (1, 3)])
        self.__set_wind_velocity_y_function(data_array[:, (1, 4)])
        self.__set_wind_heading_function(data_array[:, (1, 5)])
        self.__set_wind_direction_function(data_array[:, (1, 6)])
        self.__set_wind_speed_function(data_array[:, (1, 7)])

        # Save other attributes
        self._max_expected_height = max(height[0], height[-1])
        self.ensemble_member = member

        # Update air density, speed of sound and dynamic viscosity
        self.calculate_density_profile()
        self.calculate_speed_of_sound_profile()
        self.calculate_dynamic_viscosity()

    @funcify_method("Height Above Sea Level (m)", "Pressure (Pa)", "spline", "natural")
    def pressure_ISA(self):
        """Pressure, in Pa, as a function of height above sea level as defined
        by the `International Standard Atmosphere ISO 2533`."""
        # Retrieve lists
        pressure = self.__standard_atmosphere_layers["pressure"]
        geopotential_height = self.__standard_atmosphere_layers["geopotential_height"]
        temperature = self.__standard_atmosphere_layers["temperature"]
        beta = self.__standard_atmosphere_layers["beta"]

        # Get constants
        earth_radius = self.earth_radius
        g = self.standard_g
        R = self.air_gas_constant

        # Create function to compute pressure at a given geometric height
        def pressure_function(h):
            """Computes the pressure at a given geometric height h using the
            International Standard Atmosphere model."""
            # Convert geometric to geopotential height
            H = earth_radius * h / (earth_radius + h)

            # Check if height is within bounds, return extrapolated value if not
            if H < -2000:
                return pressure[0]
            elif H > 80000:
                return pressure[-1]

            # Find layer that contains height h
            layer = bisect.bisect(geopotential_height, H) - 1

            # Retrieve layer base geopotential height, temp, beta and pressure
            base_geopotential_height = geopotential_height[layer]
            base_temperature = temperature[layer]
            base_pressure = pressure[layer]
            B = beta[layer]

            # Compute pressure
            if B != 0:
                P = base_pressure * (
                    1 + (B / base_temperature) * (H - base_geopotential_height)
                ) ** (-g / (B * R))
            else:
                T = base_temperature + B * (H - base_geopotential_height)
                P = base_pressure * np.exp(
                    -(H - base_geopotential_height) * (g / (R * T))
                )
            return P

        # Discretize this Function to speed up the trajectory simulation
        altitudes = np.linspace(0, 80000, 100)  # TODO: should be -2k instead of 0
        pressures = [pressure_function(h) for h in altitudes]

        return np.column_stack([altitudes, pressures])

    @funcify_method("Pressure (Pa)", "Height Above Sea Level (m)")
    def barometric_height_ISA(self):
        """Returns the inverse function of the ``pressure_ISA`` function."""
        return self.pressure_ISA.inverse_function()

    @funcify_method("Height Above Sea Level (m)", "Temperature (K)", "linear")
    def temperature_ISA(self):
        """Air temperature, in K, as a function of altitude as defined by the
        `International Standard Atmosphere ISO 2533`."""
        temperature = self.__standard_atmosphere_layers["temperature"]
        geopotential_height = self.__standard_atmosphere_layers["geopotential_height"]
        altitude_asl = [
            geopotential_height_to_geometric_height(h, self.earth_radius)
            for h in geopotential_height
        ]
        return np.column_stack([altitude_asl, temperature])

    def calculate_density_profile(self):
        r"""Compute the density of the atmosphere as a function of
        height. This function is automatically called whenever a new atmospheric
        model is set.

        Notes
        -----
        1. The density is calculated as:
            .. math:: \rho = \frac{P}{RT}

        Examples
        --------
        Creating an Environment object and calculating the density
        at Sea Level:

        >>> env = Environment()
        >>> env.calculate_density_profile()
        >>> float(env.density(0))
        1.225000018124288

        Creating an Environment object and calculating the density
        at 1000m above Sea Level:

        >>> env = Environment()
        >>> env.calculate_density_profile()
        >>> float(env.density(1000))
        1.1115112430077818
        """
        # Retrieve pressure P, gas constant R and temperature T
        P = self.pressure
        R = self.air_gas_constant
        T = self.temperature

        # Compute density using P/RT
        D = P / (R * T)

        # Set new output for the calculated density
        D.set_outputs("Air Density (kg/m³)")

        # Save calculated density
        self.density = D

    def calculate_speed_of_sound_profile(self):
        r"""Compute the speed of sound in the atmosphere as a function
        of height. This function is automatically called whenever a new
        atmospheric model is set.

        Notes
        -----
        1. The speed of sound is calculated as:
            .. math:: a = \sqrt{\gamma \cdot R \cdot T}
        """
        # Retrieve gas constant R and temperature T
        R = self.air_gas_constant
        T = self.temperature
        G = 1.4

        # Compute speed of sound using sqrt(gamma*R*T)
        a = (G * R * T) ** 0.5

        # Set new output for the calculated speed of sound
        a.set_outputs("Speed of Sound (m/s)")

        # Save calculated speed of sound
        self.speed_of_sound = a

    def calculate_dynamic_viscosity(self):
        r"""Compute the dynamic viscosity of the atmosphere as a function of
        height by using the formula given in ISO 2533. This function is
        automatically called whenever a new atmospheric model is set.

        Notes
        -----
        1. The dynamic viscosity is calculated as:
            .. math::
                \mu = \frac{B \cdot T^{1.5}}{(T + S)}

            where `B` and `S` are constants, and `T` is the temperature.
        2. This equation is invalid for very high or very low temperatures.
        3. Also invalid under conditions occurring at altitudes above 90 km.
        """
        # Retrieve temperature T and set constants
        T = self.temperature
        B = 1.458e-6  # Kg/m/s/K^0.5
        S = 110.4  # K

        # Compute dynamic viscosity using u = B*T^(1.4)/(T+S) (See ISO2533)
        u = (B * T ** (1.5)) / (T + S)

        # Set new output for the calculated density
        u.set_outputs("Dynamic Viscosity (Pa s)")

        # Save calculated density
        self.dynamic_viscosity = u

    def add_wind_gust(self, wind_gust_x, wind_gust_y):
        """Adds a function to the current stored wind profile, in order to
        simulate a wind gust.

        Parameters
        ----------
        wind_gust_x : float, callable
            Callable, function of altitude, which will be added to the
            x velocity of the current stored wind profile. If float is given,
            it will be considered as a constant function in altitude.
        wind_gust_y : float, callable
            Callable, function of altitude, which will be added to the
            y velocity of the current stored wind profile. If float is given,
            it will be considered as a constant function in altitude.
        """
        # Recalculate wind_velocity_x and wind_velocity_y
        self.__set_wind_velocity_x_function(self.wind_velocity_x + wind_gust_x)
        self.__set_wind_velocity_y_function(self.wind_velocity_y + wind_gust_y)

        # Reset wind heading and velocity magnitude
        self.wind_heading = Function(
            lambda h: (180 / np.pi)
            * np.arctan2(
                self.wind_velocity_x.get_value_opt(h),
                self.wind_velocity_y.get_value_opt(h),
            )
            % 360,
            "Height (m)",
            "Wind Heading (degrees)",
            extrapolation="constant",
        )
        self.wind_speed = Function(
            lambda h: (
                self.wind_velocity_x.get_value_opt(h) ** 2
                + self.wind_velocity_y.get_value_opt(h) ** 2
            )
            ** 0.5,
            "Height (m)",
            "Wind Speed (m/s)",
            extrapolation="constant",
        )

    def info(self):
        """Prints important data and graphs available about the Environment."""
        self.prints.all()
        self.plots.info()

    def all_info(self):
        """Prints out all data and graphs available about the Environment."""
        self.prints.all()
        self.plots.all()

    # TODO: Create a better .json format and allow loading a class from it.
    def export_environment(self, filename="environment"):
        """Export important attributes of Environment class to a ``.json`` file,
        saving the information needed to recreate the same environment using
        the ``custom_atmosphere`` model.

        Parameters
        ----------
        filename : string
            The name of the file to be saved, without the extension.
        """
        pressure = self.pressure.source
        temperature = self.temperature.source
        wind_x = self.wind_velocity_x.source
        wind_y = self.wind_velocity_y.source

        export_env_dictionary = {
            "gravity": self.gravity(self.elevation),
            "date": [
                self.datetime_date.year,
                self.datetime_date.month,
                self.datetime_date.day,
                self.datetime_date.hour,
            ],
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "datum": self.datum,
            "timezone": self.timezone,
            "max_expected_height": float(self.max_expected_height),
            "atmospheric_model_type": self.atmospheric_model_type,
            "atmospheric_model_file": self.atmospheric_model_file,
            "atmospheric_model_dict": self.atmospheric_model_dict,
            "atmospheric_model_pressure_profile": pressure,
            "atmospheric_model_temperature_profile": temperature,
            "atmospheric_model_wind_velocity_x_profile": wind_x,
            "atmospheric_model_wind_velocity_y_profile": wind_y,
        }

        with open(filename + ".json", "w") as f:
            json.dump(export_env_dictionary, f, sort_keys=False, indent=4, default=str)
        print(
            f"Your Environment file was saved at '{filename}.json'. You can use "
            "it in the future by using the custom_atmosphere atmospheric model."
        )

    def set_earth_geometry(self, datum):
        """Sets the Earth geometry for the ``Environment`` class based on the
        provided datum.

        Parameters
        ----------
        datum : str
            The datum to be used for the Earth geometry. The following options
            are supported: 'SIRGAS2000', 'SAD69', 'NAD83', 'WGS84'.

        Returns
        -------
        earth_geometry : namedtuple
            The namedtuple containing the Earth geometry.
        """
        geodesy = namedtuple("earth_geometry", "semi_major_axis flattening")
        ellipsoid = {
            "SIRGAS2000": geodesy(6378137.0, 1 / 298.257223563),
            "SAD69": geodesy(6378160.0, 1 / 298.25),
            "NAD83": geodesy(6378137.0, 1 / 298.257024899),
            "WGS84": geodesy(6378137.0, 1 / 298.257223563),
        }
        try:
            return ellipsoid[datum]
        except KeyError as e:  # pragma: no cover
            available_datums = ", ".join(ellipsoid.keys())
            raise AttributeError(
                f"The reference system '{datum}' is not recognized. Please use one of "
                f"the following recognized datum: {available_datums}"
            ) from e

    # Auxiliary functions

    @staticmethod
    def calculate_earth_radius(
        lat, semi_major_axis=6378137.0, flattening=1 / 298.257223563
    ):
        """Function to calculate the Earth's radius at a specific latitude
        based on ellipsoidal reference model. The Earth radius here is
        assumed as the distance between the ellipsoid's center of gravity and a
        point on ellipsoid surface at the desired latitude.

        Parameters
        ----------
        lat : float
            latitude at which the Earth radius will be calculated
        semi_major_axis : float
            The semi-major axis of the ellipsoid used to represent the Earth,
            must be given in meters (default is 6,378,137.0 m, which corresponds
            to the WGS84 ellipsoid)
        flattening : float
            The flattening of the ellipsoid used to represent the Earth, usually
            between 1/250 and 1/150 (default is 1/298.257223563, which
            corresponds to the WGS84 ellipsoid)

        Returns
        -------
        radius : float
            Earth radius at the desired latitude, in meters

        Notes
        -----
        The ellipsoid is an approximation for the Earth model and
        will result in an estimate of the perfect distance between
        Earth's relief and its center of gravity.
        """
        semi_minor_axis = semi_major_axis * (1 - flattening)

        # Numpy trigonometric functions work with radians, so convert to radians
        lat = lat * np.pi / 180

        # Calculate the Earth Radius in meters
        e_radius = np.sqrt(
            (
                (np.cos(lat) * (semi_major_axis**2)) ** 2
                + (np.sin(lat) * (semi_minor_axis**2)) ** 2
            )
            / (
                (np.cos(lat) * semi_major_axis) ** 2
                + (np.sin(lat) * semi_minor_axis) ** 2
            )
        )

        return e_radius

    @staticmethod
    def decimal_degrees_to_arc_seconds(angle):
        """Function to convert an angle in decimal degrees to degrees, arc
        minutes and arc seconds.

        Parameters
        ----------
        angle : float
            The angle that you need convert. Must be given in decimal degrees.

        Returns
        -------
        degrees : int
            The degrees.
        arc_minutes : int
            The arc minutes. 1 arc-minute = (1/60)*degree
        arc_seconds : float
            The arc Seconds. 1 arc-second = (1/3600)*degree

        Examples
        --------
        Convert 45.5 degrees to degrees, arc minutes and arc seconds:

        >>> from rocketpy import Environment
        >>> Environment.decimal_degrees_to_arc_seconds(45.5)
        (45, 30, 0.0)
        """
        sign = -1 if angle < 0 else 1
        degrees = int(abs(angle)) * sign
        remainder = abs(angle) - abs(degrees)
        arc_minutes = int(remainder * 60)
        arc_seconds = (remainder * 60 - arc_minutes) * 60
        return degrees, arc_minutes, arc_seconds

    def to_dict(self, **kwargs):
        wind_velocity_x = self.wind_velocity_x
        wind_velocity_y = self.wind_velocity_y
        wind_heading = self.wind_heading
        wind_direction = self.wind_direction
        wind_speed = self.wind_speed
        density = self.density
        if kwargs.get("discretize", False):
            wind_velocity_x = wind_velocity_x.set_discrete(0, self.max_expected_height)
            wind_velocity_y = wind_velocity_y.set_discrete(0, self.max_expected_height)
            wind_heading = wind_heading.set_discrete(0, self.max_expected_height)
            wind_direction = wind_direction.set_discrete(0, self.max_expected_height)
            wind_speed = wind_speed.set_discrete(0, self.max_expected_height)
            density = density.set_discrete(0, self.max_expected_height)

        env_dict = {
            "gravity": self.gravity,
            "date": self.date,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "datum": self.datum,
            "timezone": self.timezone,
            "max_expected_height": self.max_expected_height,
            "atmospheric_model_type": self.atmospheric_model_type,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "wind_velocity_x": wind_velocity_x,
            "wind_velocity_y": wind_velocity_y,
            "wind_heading": wind_heading,
            "wind_direction": wind_direction,
            "wind_speed": wind_speed,
        }

        if kwargs.get("include_outputs", False):
            env_dict["density"] = density
            env_dict["barometric_height"] = self.barometric_height
            env_dict["speed_of_sound"] = self.speed_of_sound
            env_dict["dynamic_viscosity"] = self.dynamic_viscosity

        return env_dict

    @classmethod
    def from_dict(cls, data):  # pylint: disable=too-many-statements
        env = cls(
            gravity=data["gravity"],
            date=data["date"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            elevation=data["elevation"],
            datum=data["datum"],
            timezone=data["timezone"],
            max_expected_height=data["max_expected_height"],
        )
        atmospheric_model = data["atmospheric_model_type"]

        match atmospheric_model:
            case "standard_atmosphere":
                env.set_atmospheric_model("standard_atmosphere")
            case "custom_atmosphere":
                env.set_atmospheric_model(
                    type="custom_atmosphere",
                    pressure=data["pressure"],
                    temperature=data["temperature"],
                    wind_u=data["wind_velocity_x"],
                    wind_v=data["wind_velocity_y"],
                )
            case _:
                env.__set_pressure_function(data["pressure"])
                env.__set_temperature_function(data["temperature"])
                env.__set_wind_velocity_x_function(data["wind_velocity_x"])
                env.__set_wind_velocity_y_function(data["wind_velocity_y"])
                env.__set_wind_heading_function(data["wind_heading"])
                env.__set_wind_direction_function(data["wind_direction"])
                env.__set_wind_speed_function(data["wind_speed"])
                env.elevation = data["elevation"]
                env.max_expected_height = data["max_expected_height"]

        if atmospheric_model in ("windy", "forecast", "reanalysis", "ensemble"):
            env.atmospheric_model_init_date = data["atmospheric_model_init_date"]
            env.atmospheric_model_end_date = data["atmospheric_model_end_date"]
            env.atmospheric_model_interval = data["atmospheric_model_interval"]
            env.atmospheric_model_init_lat = data["atmospheric_model_init_lat"]
            env.atmospheric_model_end_lat = data["atmospheric_model_end_lat"]
            env.atmospheric_model_init_lon = data["atmospheric_model_init_lon"]
            env.atmospheric_model_end_lon = data["atmospheric_model_end_lon"]

        if atmospheric_model == "ensemble":
            env.level_ensemble = data["level_ensemble"]
            env.height_ensemble = data["height_ensemble"]
            env.temperature_ensemble = data["temperature_ensemble"]
            env.wind_u_ensemble = data["wind_u_ensemble"]
            env.wind_v_ensemble = data["wind_v_ensemble"]
            env.wind_heading_ensemble = data["wind_heading_ensemble"]
            env.wind_direction_ensemble = data["wind_direction_ensemble"]
            env.wind_speed_ensemble = data["wind_speed_ensemble"]
            env.num_ensemble_members = data["num_ensemble_members"]

        env.__reset_barometric_height_function()
        env.calculate_density_profile()
        env.calculate_speed_of_sound_profile()
        env.calculate_dynamic_viscosity()

        return env


if __name__ == "__main__":  # pragma: no cover
    import doctest

    results = doctest.testmod()
    if results.failed < 1:
        print(f"All the {results.attempted} tests passed!")
    else:
        print(f"{results.failed} out of {results.attempted} tests failed.")
