import bisect
import json
import re
import warnings
from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np
import numpy.ma as ma
import pytz
import requests

from ..mathutils.function import Function, funcify_method
from ..plots.environment_plots import _EnvironmentPlots
from ..prints.environment_prints import _EnvironmentPrints

try:
    import netCDF4
except ImportError:
    has_netCDF4 = False
    warnings.warn(
        "Unable to load netCDF4. NetCDF files and ``OPeNDAP`` will not be imported.",
        ImportWarning,
    )
else:
    has_netCDF4 = True


def requires_netCDF4(func):
    def wrapped_func(*args, **kwargs):
        if has_netCDF4:
            func(*args, **kwargs)
        else:
            raise ImportError(
                "This feature requires netCDF4 to be installed. Install it with `pip install netCDF4`"
            )

    return wrapped_func


class Environment:
    """Keeps all environment information stored, such as wind and temperature
    conditions, as well as gravity.

    Attributes
    ----------
    Environment.earth_radius : float
        Value of Earth's Radius as 6.3781e6 m.
    Environment.air_gas_constant : float
        Value of Air's Gas Constant as 287.05287 J/K/Kg
    Environment.gravity : float
        Positive value of gravitational acceleration in m/s^2.
    Environment.latitude : float
        Launch site latitude.
    Environment.longitude : float
        Launch site longitude.
    Environment.datum : string
        The desired reference ellipsoid model, the following options are
        available: "SAD69", "WGS84", "NAD83", and "SIRGAS2000". The default
        is "SIRGAS2000", then this model will be used if the user make some
        typing mistake
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
        Launch site S/N hemisphere
    Environment.initial_ew : string
        Launch site E/W hemisphere
    Environment.elevation : float
        Launch site elevation.
    Environment.date : datetime
        Date time of launch in UTC.
    Environment.local_date : datetime
        Date time of launch in the local time zone, defined by
        ``Environment.timezone``.
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
        above sea level (ASL). Especially useful for controlling plottings.
        Can be altered as desired by doing `max_expected_height = number`.
    Environment.pressure_ISA : Function
        Air pressure in Pa as a function of altitude as defined by the
        `International Standard Atmosphere ISO 2533`. Only defined after load
        ``Environment.load_international_standard_atmosphere`` has been called.
        Can be accessed as regular array, or called as a Function. See Function
        for more information.
    Environment.temperature_ISA : Function
        Air temperature in K as a function of altitude as defined by the
        `International Standard Atmosphere ISO 2533`. Only defined after load
        ``Environment.load_international_standard_atmosphere`` has been called.
        Can be accessed as regular array, or called as a Function. See Function
        for more information.
    Environment.pressure : Function
        Air pressure in Pa as a function of altitude. Can be accessed as regular
        array, or called as a Function. See Function for more information.
    Environment.temperature : Function
        Air temperature in K as a function of altitude. Can be accessed as
        regular array, or called as a Function. See Function for more
        information.
    Environment.speed_of_sound : Function
        Speed of sound in air in m/s as a function of altitude. Can be accessed
        as regular array, or called as a Function. See Function for more
        information.
    Environment.density : Function
        Air density in kg/m³ as a function of altitude. Can be accessed as
        regular array, or called as a Function. See Function for more
        information.
    Environment.dynamic_viscosity : Function
        Air dynamic viscosity in Pa*s as a function of altitude. Can be accessed
        as regular array, or called as a Function. See Function for more
        information.
    Environment.wind_speed : Function
        Wind speed in m/s as a function of altitude. Can be accessed as regular
        array, or called as a Function. See Function for more information.
    Environment.wind_direction : Function
        Wind direction (from which the wind blows) in degrees relative to north
        (positive clockwise) as a function of altitude. Can be accessed as an
        array, or called as a Function. See Function for more information.
    Environment.wind_heading : Function
        Wind heading (direction towards which the wind blows) in degrees
        relative to north (positive clockwise) as a function of altitude.
        Can be accessed as an array, or called as a Function.
        See Function for more information.
    Environment.wind_velocity_x : Function
        Wind U, or X (east) component of wind velocity in m/s as a function of
        altitude. Can be accessed as an array, or called as a Function. See
        Function for more information.
    Environment.wind_velocity_y : Function
        Wind V, or Y (north) component of wind velocity in m/s as a function of
        altitude. Can be accessed as an array, or called as a Function. See
        Function for more information.
    Environment.atmospheric_model_type : string
        Describes the atmospheric model which is being used. Can only assume the
        following values: ``standard_atmosphere``, ``custom_atmosphere``,
        ``wyoming_sounding``, ``NOAARucSounding``, ``Forecast``, ``Reanalysis``,
        ``Ensemble``.
    Environment.atmospheric_model_file : string
        Address of the file used for the atmospheric model being used. Only
        defined for ``wyoming_sounding``, ``NOAARucSounding``, ``Forecast``,
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
    """

    def __init__(
        self,
        gravity=None,
        date=None,
        latitude=0,
        longitude=0,
        elevation=0,
        datum="SIRGAS2000",
        timezone="UTC",
        max_expected_height=80000.0,
    ):
        """Initialize Environment class, saving launch rail length,
        launch date, location coordinates and elevation. Note that
        by default the standard atmosphere is loaded until another

        Parameters
        ----------
        gravity : int, float, callable, string, array, optional
            Surface gravitational acceleration. Positive values point the
            acceleration down. If None, the Somigliana formula is used to
        date : array, optional
            Array of length 4, stating (year, month, day, hour (UTC))
            of rocket launch. Must be given if a Forecast, Reanalysis
            or Ensemble, will be set as an atmospheric model.
        latitude : float, optional
            Latitude in degrees (ranging from -90 to 90) of rocket
            launch location. Must be given if a Forecast, Reanalysis
            or Ensemble will be used as an atmospheric model or if
            Open-Elevation will be used to compute elevation.
        longitude : float, optional
            Longitude in degrees (ranging from -180 to 360) of rocket
            launch location. Must be given if a Forecast, Reanalysis
            or Ensemble will be used as an atmospheric model or if
            Open-Elevation will be used to compute elevation.
        elevation : float, optional
            Elevation of launch site measured as height above sea
            level in meters. Alternatively, can be set as
            'Open-Elevation' which uses the Open-Elevation API to
            find elevation data. For this option, latitude and
            longitude must also be specified. Default value is 0.
        datum : string
            The desired reference ellipsoidal model, the following options are
            available: "SAD69", "WGS84", "NAD83", and "SIRGAS2000". The default
            is "SIRGAS2000", then this model will be used if the user make some
            typing mistake.
        timezone : string, optional
            Name of the time zone. To see all time zones, import pytz and run
            print(pytz.all_timezones). Default time zone is "UTC".
        max_expected_height : float, optional
            Maximum altitude in meters to keep weather data. The altitude must
            be above sea level (ASL). Especially useful for visualization.
            Can be altered as desired by doing `max_expected_height = number`.
            Depending on the atmospheric model, this value may be automatically
            mofified.

        Returns
        -------
        None
        """
        # Initialize constants
        self.earth_radius = 6.3781 * (10**6)
        self.air_gas_constant = 287.05287  # in J/K/Kg
        self.standard_g = 9.80665

        # Initialize launch site details
        self.elevation = elevation
        self.set_elevation(elevation)
        self._max_expected_height = max_expected_height

        # Initialize plots and prints objects
        self.prints = _EnvironmentPrints(self)
        self.plots = _EnvironmentPlots(self)

        # Initialize atmosphere
        self.set_atmospheric_model("standard_atmosphere")

        # Save date
        if date != None:
            self.set_date(date, timezone)
        else:
            self.date = None
            self.datetime_date = None
            self.local_date = None
            self.timezone = None

        # Initialize Earth geometry and save datum
        self.datum = datum
        self.ellipsoid = self.set_earth_geometry(datum)

        # Save latitude and longitude
        self.latitude = latitude
        self.longitude = longitude
        if latitude != None and longitude != None:
            self.set_location(latitude, longitude)
        else:
            self.latitude, self.longitude = None, None

        # Store launch site coordinates referenced to UTM projection system
        if self.latitude > -80 and self.latitude < 84:
            convert = self.geodesic_to_utm(
                lat=self.latitude,
                lon=self.longitude,
                flattening=self.ellipsoid.flattening,
                semi_major_axis=self.ellipsoid.semi_major_axis,
            )

            self.initial_north = convert[1]
            self.initial_east = convert[0]
            self.initial_utm_zone = convert[2]
            self.initial_utm_letter = convert[3]
            self.initial_hemisphere = convert[4]
            self.initial_ew = convert[5]

        # Set gravity model
        self.gravity = self.set_gravity_model(gravity)

        # Recalculate Earth Radius (meters)
        self.earth_radius = self.calculate_earth_radius(
            lat=self.latitude,
            semi_major_axis=self.ellipsoid.semi_major_axis,
            flattening=self.ellipsoid.flattening,
        )

        return None

    def set_date(self, date, timezone="UTC"):
        """Set date and time of launch and update weather conditions if
        date dependent atmospheric model is used.

        Parameters
        ----------
        date : Datetime
            Datetime object specifying launch date and time.
        timezone : string, optional
            Name of the time zone. To see all time zones, import pytz and run
            print(pytz.all_timezones). Default time zone is "UTC".

        Returns
        -------
        None
        """
        # Store date and configure time zone
        self.timezone = timezone
        tz = pytz.timezone(self.timezone)
        if type(date) != datetime:
            local_date = datetime(*date)
        else:
            local_date = date
        if local_date.tzinfo == None:
            local_date = tz.localize(local_date)
        self.date = date
        self.local_date = local_date
        self.datetime_date = self.local_date.astimezone(pytz.UTC)

        # Update atmospheric conditions if atmosphere type is Forecast,
        # Reanalysis or Ensemble
        try:
            if self.atmospheric_model_type in ["Forecast", "Reanalysis", "Ensemble"]:
                self.set_atmospheric_model(
                    self.atmospheric_model_file, self.atmospheric_model_dict
                )
        except AttributeError:
            pass

        return None

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
        # Store latitude and longitude
        self.latitude = latitude
        self.longitude = longitude

        # Update atmospheric conditions if atmosphere type is Forecast,
        # Reanalysis or Ensemble
        if self.atmospheric_model_type in ["Forecast", "Reanalysis", "Ensemble"]:
            self.set_atmospheric_model(
                self.atmospheric_model_file, self.atmospheric_model_dict
            )

        # Return None

    def set_gravity_model(self, gravity):
        """Sets the gravity model to be used in the simulation based on the
        given user input to the gravity parameter.

        Parameters
        ----------
        gravity : None or Function source
            If None, the Somigliana formula is used to compute the gravity
            acceleration. Otherwise, the user can provide a Function object
            representing the gravity model.

        Returns
        -------
        Function
            Function object representing the gravity model.
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
        if value < self.elevation:
            raise ValueError(
                "Max expected height cannot be lower than the surface elevation"
            )
        self._max_expected_height = value
        self.plots.grid = np.linspace(self.elevation, self.max_expected_height)

    @funcify_method("height (m)", "gravity (m/s²)")
    def somigliana_gravity(self, height):
        """Computes the gravity acceleration with the Somigliana formula.
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
            meters. Alternatively, can be set as 'Open-Elevation' which uses
            the Open-Elevation API to find elevation data. For this option,
            latitude and longitude must have already been specified.

            See Also
            --------
            Environment.set_location

        Returns
        -------
        None
        """
        if elevation != "Open-Elevation" and elevation != "SRTM":
            self.elevation = elevation
        # elif elevation == "SRTM" and self.latitude != None and self.longitude != None:
        #     # Trigger the authentication flow.
        #     #ee.Authenticate()
        #     # Initialize the library.
        #     ee.Initialize()

        #     # Calculate elevation
        #     dem  = ee.Image('USGS/SRTMGL1_003')
        #     xy   = ee.Geometry.Point([self.longitude, self.latitude])
        #     elev = dem.sample(xy, 30).first().get('elevation').getInfo()

        #     self.elevation = elev

        elif self.latitude != None and self.longitude != None:
            try:
                print("Fetching elevation from open-elevation.com...")
                request_url = "https://api.open-elevation.com/api/v1/lookup?locations={:f},{:f}".format(
                    self.latitude, self.longitude
                )
                response = requests.get(request_url)
                results = response.json()["results"]
                self.elevation = results[0]["elevation"]
                print("Elevation received:", self.elevation)
            except:
                raise RuntimeError("Unable to reach Open-Elevation API servers.")
        else:
            raise ValueError(
                "Latitude and longitude must be set to use"
                " Open-Elevation API. See Environment.set_location."
            )

    @requires_netCDF4
    def set_topographic_profile(self, type, file, dictionary="netCDF4", crs=None):
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
                rootgrp = netCDF4.Dataset(file, "r", format="NETCDF4")
                self.elev_lon_array = rootgrp.variables["lon"][:].tolist()
                self.elev_lat_array = rootgrp.variables["lat"][:].tolist()
                self.elev_array = rootgrp.variables["NASADEM_HGT"][:].tolist()
                # crsArray = rootgrp.variables['crs'][:].tolist().
                self.topographic_profile_activated = True

                print("Region covered by the Topographical file: ")
                print(
                    "Latitude from {:.6f}° to {:.6f}°".format(
                        self.elev_lat_array[-1], self.elev_lat_array[0]
                    )
                )
                print(
                    "Longitude from {:.6f}° to {:.6f}°".format(
                        self.elev_lon_array[0], self.elev_lon_array[-1]
                    )
                )

        return None

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
        elevation : float
            Elevation provided by the topographic data, in meters.
        """
        if self.topographic_profile_activated == False:
            print(
                "You must define a Topographic profile first, please use the method Environment.set_topographic_profile()"
            )
            return None

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
                "Latitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    lat, self.elev_lat_array[0], self.elev_lat_array[-1]
                )
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
                "Longitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    lon, self.elev_lon_array[0], self.elev_lon_array[-1]
                )
            )

        # Get the elevation
        elevation = self.elev_array[lat_index][lon_index]

        return elevation

    def set_atmospheric_model(
        self,
        type,
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

            - ``NOAARucSounding``: sets pressure, temperature, wind-u
              and wind-v profiles and surface elevation obtained from
              an upper air sounding given by the file parameter through
              an URL. This URL should point to a data webpage obtained
              through NOAA's Ruc Sounding servers, which can be accessed
              in `rucsoundings`_. Selecting ROABs as the
              initial data source, specifying the station through it's
              WMO-ID and opting for the ASCII (GSD format) button, the
              following example URL opens up:

              https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest

              Any ASCII GSD format page from this server can be read,
              so information from virtual soundings such as GFS and NAM
              can also be imported.

              .. _rucsoundings: https://rucsoundings.noaa.gov/

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
                ``Environment.selectEnsembleMemberMember``.

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

                Time referece for the Forecasts are:

                - ``GFS``: `Global` - 0.25deg resolution - Updates every 6
                  hours, forecast for 81 points spaced by 3 hours
                - ``FV3``: `Global` - 0.25deg resolution - Updates every 6
                  hours, forecast for 129 points spaced by 3 hours
                - ``RAP``: `Regional USA` - 0.19deg resolution - Updates hourly,
                  forecast for 40 points spaced hourly
                - ``NAM``: `Regional CONUS Nest` - 5 km resolution - Updates
                  every 6 hours, forecast for 21 points spaced by 3 hours

            If type is ``Ensemble``, this parameter can also be either ``GEFS``,
            or ``CMC`` for the latest of these ensembles.

            .. note::

                Time referece for the Ensembles are:

                - GEFS: Global, bias-corrected, 0.5deg resolution, 21 forecast
                  members, Updates every 6 hours, forecast for 65 points spaced
                  by 4 hours
                - CMC: Global, 0.5deg resolution, 21 forecast members, Updates
                  every 12 hours, forecast for 65 points spaced by 4 hours

            If type is ``Windy``, this parameter can be either ``GFS``,
            ``ECMWF``, ``ICON`` or ``ICONEU``. Default in this case is ``ECMWF``.
        dictionary : dictionary, string, optional
            Dictionary that must be given when type is either ``Forecast``,
            ``Reanalysis`` or ``Ensemble``. It specifies the dictionary to be
            used when reading ``netCDF`` and ``OPeNDAP`` files, allowing the
            correct retrieval of data. Acceptable values include ``ECMWF``,
            ``NOAA`` and ``UCAR`` for default dictionaries which can generally
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

        # Handle each case
        if type == "standard_atmosphere":
            self.process_standard_atmosphere()
        elif type == "wyoming_sounding":
            self.process_wyoming_sounding(file)
            # Save file
            self.atmospheric_model_file = file
        elif type == "NOAARucSounding":
            self.process_noaaruc_sounding(file)
            # Save file
            self.atmospheric_model_file = file
        elif type == "Forecast" or type == "Reanalysis":
            # Process default forecasts if requested
            if file == "GFS":
                # Define dictionary
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
                # Attempt to get latest forecast
                time_attempt = datetime.utcnow()
                success = False
                attempt_count = 0
                while not success and attempt_count < 10:
                    time_attempt -= timedelta(hours=6 * attempt_count)
                    file = "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{:04d}{:02d}{:02d}/gfs_0p25_{:02d}z".format(
                        time_attempt.year,
                        time_attempt.month,
                        time_attempt.day,
                        6 * (time_attempt.hour // 6),
                    )
                    try:
                        self.process_forecast_reanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attempt_count += 1
                if not success:
                    raise RuntimeError(
                        "Unable to load latest weather data for GFS through " + file
                    )
            elif file == "FV3":
                # Define dictionary
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
                # Attempt to get latest forecast
                time_attempt = datetime.utcnow()
                success = False
                attempt_count = 0
                while not success and attempt_count < 10:
                    time_attempt -= timedelta(hours=6 * attempt_count)
                    file = "https://nomads.ncep.noaa.gov/dods/gfs_0p25_parafv3/gfs{:04d}{:02d}{:02d}/gfs_0p25_parafv3_{:02d}z".format(
                        time_attempt.year,
                        time_attempt.month,
                        time_attempt.day,
                        6 * (time_attempt.hour // 6),
                    )
                    try:
                        self.process_forecast_reanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attempt_count += 1
                if not success:
                    raise RuntimeError(
                        "Unable to load latest weather data for FV3 through " + file
                    )
            elif file == "NAM":
                # Define dictionary
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
                # Attempt to get latest forecast
                time_attempt = datetime.utcnow()
                success = False
                attempt_count = 0
                while not success and attempt_count < 10:
                    time_attempt -= timedelta(hours=6 * attempt_count)
                    file = "https://nomads.ncep.noaa.gov/dods/nam/nam{:04d}{:02d}{:02d}/nam_conusnest_{:02d}z".format(
                        time_attempt.year,
                        time_attempt.month,
                        time_attempt.day,
                        6 * (time_attempt.hour // 6),
                    )
                    try:
                        self.process_forecast_reanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attempt_count += 1
                if not success:
                    raise RuntimeError(
                        "Unable to load latest weather data for NAM through " + file
                    )
            elif file == "RAP":
                # Define dictionary
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
                # Attempt to get latest forecast
                time_attempt = datetime.utcnow()
                success = False
                attempt_count = 0
                while not success and attempt_count < 10:
                    time_attempt -= timedelta(hours=1 * attempt_count)
                    file = "https://nomads.ncep.noaa.gov/dods/rap/rap{:04d}{:02d}{:02d}/rap_{:02d}z".format(
                        time_attempt.year,
                        time_attempt.month,
                        time_attempt.day,
                        time_attempt.hour,
                    )
                    try:
                        self.process_forecast_reanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attempt_count += 1
                if not success:
                    raise RuntimeError(
                        "Unable to load latest weather data for RAP through " + file
                    )
            # Process other forecasts or reanalysis
            else:
                # Check if default dictionary was requested
                if dictionary == "ECMWF":
                    dictionary = {
                        "time": "time",
                        "latitude": "latitude",
                        "longitude": "longitude",
                        "level": "level",
                        "temperature": "t",
                        "surface_geopotential_height": None,
                        "geopotential_height": None,
                        "geopotential": "z",
                        "u_wind": "u",
                        "v_wind": "v",
                    }
                elif dictionary == "NOAA":
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
                elif dictionary is None:
                    raise TypeError(
                        "Please specify a dictionary or choose a default one such as ECMWF or NOAA."
                    )
                # Process forecast or reanalysis
                self.process_forecast_reanalysis(file, dictionary)
            # Save dictionary and file
            self.atmospheric_model_file = file
            self.atmospheric_model_dict = dictionary
        elif type == "Ensemble":
            # Process default forecasts if requested
            if file == "GEFS":
                # Define dictionary
                dictionary = {
                    "time": "time",
                    "latitude": "lat",
                    "longitude": "lon",
                    "level": "lev",
                    "ensemble": "ens",
                    "temperature": "tmpprs",
                    "surface_geopotential_height": None,
                    "geopotential_height": "hgtprs",
                    "geopotential": None,
                    "u_wind": "ugrdprs",
                    "v_wind": "vgrdprs",
                }
                # Attempt to get latest forecast
                time_attempt = datetime.utcnow()
                success = False
                attempt_count = 0
                while not success and attempt_count < 10:
                    time_attempt -= timedelta(hours=6 * attempt_count)
                    file = "https://nomads.ncep.noaa.gov/dods/gens_bc/gens{:04d}{:02d}{:02d}/gep_all_{:02d}z".format(
                        time_attempt.year,
                        time_attempt.month,
                        time_attempt.day,
                        6 * (time_attempt.hour // 6),
                    )
                    try:
                        self.process_ensemble(file, dictionary)
                        success = True
                    except OSError:
                        attempt_count += 1
                if not success:
                    raise RuntimeError(
                        "Unable to load latest weather data for GEFS through " + file
                    )
            elif file == "CMC":
                # Define dictionary
                dictionary = {
                    "time": "time",
                    "latitude": "lat",
                    "longitude": "lon",
                    "level": "lev",
                    "ensemble": "ens",
                    "temperature": "tmpprs",
                    "surface_geopotential_height": None,
                    "geopotential_height": "hgtprs",
                    "geopotential": None,
                    "u_wind": "ugrdprs",
                    "v_wind": "vgrdprs",
                }
                # Attempt to get latest forecast
                time_attempt = datetime.utcnow()
                success = False
                attempt_count = 0
                while not success and attempt_count < 10:
                    time_attempt -= timedelta(hours=12 * attempt_count)
                    file = "https://nomads.ncep.noaa.gov/dods/cmcens/cmcens{:04d}{:02d}{:02d}/cmcens_all_{:02d}z".format(
                        time_attempt.year,
                        time_attempt.month,
                        time_attempt.day,
                        12 * (time_attempt.hour // 12),
                    )
                    try:
                        self.process_ensemble(file, dictionary)
                        success = True
                    except OSError:
                        attempt_count += 1
                if not success:
                    raise RuntimeError(
                        "Unable to load latest weather data for CMC through " + file
                    )
            # Process other forecasts or reanalysis
            else:
                # Check if default dictionary was requested
                if dictionary == "ECMWF":
                    dictionary = {
                        "time": "time",
                        "latitude": "latitude",
                        "longitude": "longitude",
                        "level": "level",
                        "ensemble": "number",
                        "temperature": "t",
                        "surface_geopotential_height": None,
                        "geopotential_height": None,
                        "geopotential": "z",
                        "u_wind": "u",
                        "v_wind": "v",
                    }
                elif dictionary == "NOAA":
                    dictionary = {
                        "time": "time",
                        "latitude": "lat",
                        "longitude": "lon",
                        "level": "lev",
                        "ensemble": "ens",
                        "temperature": "tmpprs",
                        "surface_geopotential_height": None,
                        "geopotential_height": "hgtprs",
                        "geopotential": None,
                        "u_wind": "ugrdprs",
                        "v_wind": "vgrdprs",
                    }
                # Process forecast or reanalysis
                self.process_ensemble(file, dictionary)
            # Save dictionary and file
            self.atmospheric_model_file = file
            self.atmospheric_model_dict = dictionary
        elif type == "custom_atmosphere":
            self.process_custom_atmosphere(pressure, temperature, wind_u, wind_v)
        elif type == "Windy":
            self.process_windy_atmosphere(file)
        else:
            raise ValueError("Unknown model type.")

        # Calculate air density
        self.calculate_density_profile()

        # Calculate speed of sound
        self.calculate_speed_of_sound_profile()

        # Update dynamic viscosity
        self.calculate_dynamic_viscosity()

        return None

    def process_standard_atmosphere(self):
        """Sets pressure and temperature profiles corresponding to the
        International Standard Atmosphere defined by ISO 2533 and
        ranging from -2 km to 80 km of altitude above sea level. Note
        that the wind profiles are set to zero.

        Returns
        -------
        None
        """
        # Load international standard atmosphere
        self.load_international_standard_atmosphere()

        # Save temperature, pressure and wind profiles
        self.pressure = self.pressure_ISA
        self.temperature = self.temperature_ISA
        self.wind_direction = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.wind_heading = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.wind_speed = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_x = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Set maximum expected height
        self.max_expected_height = 80000

        return None

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
        max_expected_height = 1000

        # Save pressure profile
        if pressure is None:
            # Use standard atmosphere
            self.pressure = self.pressure_ISA
        else:
            # Use custom input
            self.pressure = Function(
                pressure,
                inputs="Height Above Sea Level (m)",
                outputs="Pressure (Pa)",
                interpolation="linear",
            )
            # Check maximum height of custom pressure input
            if not callable(self.pressure.source):
                max_expected_height = max(self.pressure[-1, 0], max_expected_height)

        # Save temperature profile
        if temperature is None:
            # Use standard atmosphere
            self.temperature = self.temperature_ISA
        else:
            self.temperature = Function(
                temperature,
                inputs="Height Above Sea Level (m)",
                outputs="Temperature (K)",
                interpolation="linear",
            )
            # Check maximum height of custom temperature input
            if not callable(self.temperature.source):
                max_expected_height = max(self.temperature[-1, 0], max_expected_height)

        # Save wind profile
        self.wind_velocity_x = Function(
            wind_u,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            wind_v,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )
        # Check maximum height of custom wind input
        if not callable(self.wind_velocity_x.source):
            max_expected_height = max(self.wind_velocity_x[-1, 0], max_expected_height)
        if not callable(self.wind_velocity_y.source):
            max_expected_height = max(self.wind_velocity_y[-1, 0], max_expected_height)

        # Compute wind profile direction and heading
        wind_heading = (
            lambda h: np.arctan2(self.wind_velocity_x(h), self.wind_velocity_y(h))
            * (180 / np.pi)
            % 360
        )
        self.wind_heading = Function(
            wind_heading,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )

        def wind_direction(h):
            return (wind_heading(h) - 180) % 360

        self.wind_direction = Function(
            wind_direction,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )

        def wind_speed(h):
            return np.sqrt(self.wind_velocity_x(h) ** 2 + self.wind_velocity_y(h) ** 2)

        self.wind_speed = Function(
            wind_speed,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.max_expected_height = max_expected_height

        return None

    def process_windy_atmosphere(self, model="ECMWF"):
        """Process data from Windy.com to retrieve atmospheric forecast data.

        Parameters
        ----------
        model : string, optional
            The atmospheric model to use. Default is ``ECMWF``. Options are:
            ``ECMWF`` for the `ECMWF-HRES` model, ``GFS`` for the `GFS` model,
            ``ICON`` for the `ICON-Global` model or ``ICONEU`` for the `ICON-EU`
            model.
        """

        # Process the model string
        model = model.lower()
        if model[-1] == "u":  # case iconEu
            model = "".join([model[:4], model[4].upper(), model[4 + 1 :]])
        # Load data from Windy.com: json file
        url = f"https://node.windy.com/forecast/meteogram/{model}/{self.latitude}/{self.longitude}/?step=undefined"
        try:
            response = requests.get(url).json()
        except:
            if model == "iconEu":
                raise ValueError(
                    "Could not get a valid response for Icon-EU from Windy. Check if the latitude and longitude coordinates set are inside Europe.",
                )
            raise

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
        geopotential_height_array = np.array(
            [response["data"][f"gh-{pL}h"][time_index] for pL in pressure_levels]
        )
        # Convert geopotential height to geometric altitude (ASL)
        R = self.earth_radius
        altitude_array = R * geopotential_height_array / (R - geopotential_height_array)

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

        # Determine wind speed, heading and direction
        wind_speed_array = np.sqrt(wind_u_array**2 + wind_v_array**2)
        wind_heading_array = (
            np.arctan2(wind_u_array, wind_v_array) * (180 / np.pi) % 360
        )
        wind_direction_array = (wind_heading_array - 180) % 360

        # Combine all data into big array
        data_array = np.ma.column_stack(
            [
                100 * pressure_levels,  # Convert hPa to Pa
                altitude_array,
                temperature_array,
                wind_u_array,
                wind_v_array,
                wind_heading_array,
                wind_direction_array,
                wind_speed_array,
            ]
        )

        # Save atmospheric data
        self.pressure = Function(
            data_array[:, (1, 0)],
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
            interpolation="linear",
        )
        self.temperature = Function(
            data_array[:, (1, 2)],
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )
        self.wind_direction = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.wind_heading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.wind_speed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_x = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.max_expected_height = max(altitude_array[0], altitude_array[-1])

        # Get elevation data from file
        self.elevation = response["header"]["elevation"]

        # Compute info data
        self.atmospheric_model_init_date = netCDF4.num2date(
            time_array[0], units=time_units
        )
        self.atmospheric_model_end_date = netCDF4.num2date(
            time_array[-1], units=time_units
        )
        self.atmospheric_model_interval = netCDF4.num2date(
            (time_array[-1] - time_array[0]) / (len(time_array) - 1), units=time_units
        ).hour
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

    def process_wyoming_sounding(self, file):
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

            More can be found at: http://weather.uwyo.edu/upperair/sounding.html.

        Returns
        -------
        None
        """
        # Request Wyoming Sounding from file url
        response = requests.get(file)
        if response.status_code != 200:
            raise ImportError("Unable to load " + file + ".")
        if len(re.findall("Can't get .+ Observations at", response.text)):
            raise ValueError(
                re.findall("Can't get .+ Observations at .+", response.text)[0]
                + " Check station number and date."
            )
        if response.text == "Invalid OUTPUT: specified\n":
            raise ValueError(
                "Invalid OUTPUT: specified. Make sure the output is Text: List."
            )

        # Process Wyoming Sounding by finding data table and station info
        response_split_text = re.split("(<.{0,1}PRE>)", response.text)
        data_table = response_split_text[2]
        station_info = response_split_text[6]

        # Transform data table into np array
        data_array = []
        for line in data_table.split("\n")[
            5:-1
        ]:  # Split data table into lines and remove header and footer
            columns = re.split(" +", line)  # Split line into columns
            if (
                len(columns) == 12
            ):  # 12 is the number of column entries when all entries are given
                data_array.append(columns[1:])
        data_array = np.array(data_array, dtype=float)

        # Retrieve pressure from data array
        data_array[:, 0] = 100 * data_array[:, 0]  # Converts hPa to Pa
        self.pressure = Function(
            data_array[:, (1, 0)],
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
            interpolation="linear",
        )

        # Retrieve temperature from data array
        data_array[:, 2] = data_array[:, 2] + 273.15  # Converts C to K
        self.temperature = Function(
            data_array[:, (1, 2)],
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )

        # Retrieve wind-u and wind-v from data array
        data_array[:, 7] = data_array[:, 7] * 1.852 / 3.6  # Converts Knots to m/s
        data_array[:, 5] = (
            data_array[:, 6] + 180
        ) % 360  # Convert wind direction to wind heading
        data_array[:, 3] = data_array[:, 7] * np.sin(data_array[:, 5] * np.pi / 180)
        data_array[:, 4] = data_array[:, 7] * np.cos(data_array[:, 5] * np.pi / 180)

        # Convert geopotential height to geometric height
        R = self.earth_radius
        data_array[:, 1] = R * data_array[:, 1] / (R - data_array[:, 1])

        # Save atmospheric data
        self.wind_direction = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.wind_heading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.wind_speed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_x = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Retrieve station elevation from station info
        station_elevation_text = station_info.split("\n")[6]

        # Convert station elevation text into float value
        self.elevation = float(
            re.findall(r"[0-9]+\.[0-9]+|[0-9]+", station_elevation_text)[0]
        )

        # Save maximum expected height
        self.max_expected_height = data_array[-1, 1]

        return None

    def process_noaaruc_sounding(self, file):
        """Import and process the upper air sounding data from `NOAA
        Ruc Soundings` database (https://rucsoundings.noaa.gov/) given as
        ASCII GSD format pages passed by its url to the file parameter. Sets
        pressure, temperature, wind-u, wind-v profiles and surface elevation.

        Parameters
        ----------
        file : string
            URL of an upper air sounding data output from `NOAA Ruc Soundings`
            in ASCII GSD format.

            Example:

            https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest

            More can be found at: https://rucsoundings.noaa.gov/.

        Returns
        -------
        None
        """
        # Request NOAA Ruc Sounding from file url
        response = requests.get(file)
        if response.status_code != 200 or len(response.text) < 10:
            raise ImportError("Unable to load " + file + ".")

        # Split response into lines
        lines = response.text.split("\n")

        # Process GSD format (https://rucsoundings.noaa.gov/raob_format.html)

        # Extract elevation data
        for line in lines:
            # Split line into columns
            columns = re.split(" +", line)[1:]
            if len(columns) > 0:
                if columns[0] == "1" and columns[5] != "99999":
                    # Save elevation
                    self.elevation = float(columns[5])
                else:
                    # No elevation data available
                    pass

        # Extract pressure as a function of height
        pressure_array = []
        for line in lines:
            # Split line into columns
            columns = re.split(" +", line)[1:]
            if len(columns) >= 6:
                if columns[0] in ["4", "5", "6", "7", "8", "9"]:
                    # Convert columns to floats
                    columns = np.array(columns, dtype=float)
                    # Select relevant columns
                    columns = columns[[2, 1]]
                    # Check if values exist
                    if max(columns) != 99999:
                        # Save value
                        pressure_array.append(columns)
        pressure_array = np.array(pressure_array)

        # Extract temperature as a function of height
        temperature_array = []
        for line in lines:
            # Split line into columns
            columns = re.split(" +", line)[1:]
            if len(columns) >= 6:
                if columns[0] in ["4", "5", "6", "7", "8", "9"]:
                    # Convert columns to floats
                    columns = np.array(columns, dtype=float)
                    # Select relevant columns
                    columns = columns[[2, 3]]
                    # Check if values exist
                    if max(columns) != 99999:
                        # Save value
                        temperature_array.append(columns)
        temperature_array = np.array(temperature_array)

        # Extract wind speed and direction as a function of height
        wind_speed_array = []
        wind_direction_array = []
        for line in lines:
            # Split line into columns
            columns = re.split(" +", line)[1:]
            if len(columns) >= 6:
                if columns[0] in ["4", "5", "6", "7", "8", "9"]:
                    # Convert columns to floats
                    columns = np.array(columns, dtype=float)
                    # Select relevant columns
                    columns = columns[[2, 5, 6]]
                    # Check if values exist
                    if max(columns) != 99999:
                        # Save value
                        wind_direction_array.append(columns[[0, 1]])
                        wind_speed_array.append(columns[[0, 2]])
        wind_speed_array = np.array(wind_speed_array)
        wind_direction_array = np.array(wind_direction_array)

        # Converts 10*hPa to Pa and save values
        pressure_array[:, 1] = 10 * pressure_array[:, 1]
        self.pressure = Function(
            pressure_array,
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
            interpolation="linear",
        )

        # Convert 10*C to K and save values
        temperature_array[:, 1] = (
            temperature_array[:, 1] / 10 + 273.15
        )  # Converts C to K
        self.temperature = Function(
            temperature_array,
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )

        # Process wind-u and wind-v
        wind_speed_array[:, 1] = (
            wind_speed_array[:, 1] * 1.852 / 3.6
        )  # Converts Knots to m/s
        wind_heading_array = wind_direction_array[:, :] * 1
        wind_heading_array[:, 1] = (
            wind_direction_array[:, 1] + 180
        ) % 360  # Convert wind direction to wind heading
        wind_u = wind_speed_array[:, :] * 1
        wind_v = wind_speed_array[:, :] * 1
        wind_u[:, 1] = wind_speed_array[:, 1] * np.sin(
            wind_heading_array[:, 1] * np.pi / 180
        )
        wind_v[:, 1] = wind_speed_array[:, 1] * np.cos(
            wind_heading_array[:, 1] * np.pi / 180
        )

        # Save wind data
        self.wind_direction = Function(
            wind_direction_array,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.wind_heading = Function(
            wind_heading_array,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.wind_speed = Function(
            wind_speed_array,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_x = Function(
            wind_u,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            wind_v,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.max_expected_height = pressure_array[-1, 0]

    @requires_netCDF4
    def process_forecast_reanalysis(self, file, dictionary):
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
        if self.datetime_date is None:
            raise TypeError(
                "Please specify Date (array-like) when "
                "initializing this Environment. "
                "Alternatively, use the Environment.set_date"
                " method."
            )
        if self.latitude is None:
            raise TypeError(
                "Please specify Location (lat, lon). when "
                "initializing this Environment. "
                "Alternatively, use the Environment."
                "set_location method."
            )

        # Read weather file
        weather_data = netCDF4.Dataset(file)

        # Get time, latitude and longitude data from file
        time_array = weather_data.variables[dictionary["time"]]
        lon_array = weather_data.variables[dictionary["longitude"]][:].tolist()
        lat_array = weather_data.variables[dictionary["latitude"]][:].tolist()

        # Find time index
        time_index = netCDF4.date2index(
            self.datetime_date, time_array, calendar="gregorian", select="nearest"
        )
        # Convert times do dates and numbers
        input_time_num = netCDF4.date2num(
            self.datetime_date, time_array.units, calendar="gregorian"
        )
        file_time_num = time_array[time_index]
        file_time_date = netCDF4.num2date(
            time_array[time_index], time_array.units, calendar="gregorian"
        )
        # Check if time is inside range supplied by file
        if time_index == 0 and input_time_num < file_time_num:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which starts at {:}.".format(
                    file_time_date
                )
            )
        elif time_index == len(time_array) - 1 and input_time_num > file_time_num:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which ends at {:}.".format(
                    file_time_date
                )
            )
        # Check if time is exactly equal to one in the file
        if input_time_num != file_time_num:
            warnings.warn(
                "Exact chosen launch time is not available in the provided file, using {:} UTC instead.".format(
                    file_time_date
                )
            )

        # Find longitude index
        # Determine if file uses -180 to 180 or 0 to 360
        if lon_array[0] < 0 or lon_array[-1] < 0:
            # Convert input to -180 - 180
            lon = (
                self.longitude if self.longitude < 180 else -180 + self.longitude % 180
            )
        else:
            # Convert input to 0 - 360
            lon = self.longitude % 360
        # Check if reversed or sorted
        if lon_array[0] < lon_array[-1]:
            # Deal with sorted lon_array
            lon_index = bisect.bisect(lon_array, lon)
        else:
            # Deal with reversed lon_array
            lon_array.reverse()
            lon_index = len(lon_array) - bisect.bisect_left(lon_array, lon)
            lon_array.reverse()
        # Take care of longitude value equal to maximum longitude in the grid
        if lon_index == len(lon_array) and lon_array[lon_index - 1] == lon:
            lon_index = lon_index - 1
        # Check if longitude value is inside the grid
        if lon_index == 0 or lon_index == len(lon_array):
            raise ValueError(
                "Longitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    lon, lon_array[0], lon_array[-1]
                )
            )

        # Find latitude index
        # Check if reversed or sorted
        if lat_array[0] < lat_array[-1]:
            # Deal with sorted lat_array
            lat_index = bisect.bisect(lat_array, self.latitude)
        else:
            # Deal with reversed lat_array
            lat_array.reverse()
            lat_index = len(lat_array) - bisect.bisect_left(lat_array, self.latitude)
            lat_array.reverse()
        # Take care of latitude value equal to maximum longitude in the grid
        if lat_index == len(lat_array) and lat_array[lat_index - 1] == self.latitude:
            lat_index = lat_index - 1
        # Check if latitude value is inside the grid
        if lat_index == 0 or lat_index == len(lat_array):
            raise ValueError(
                "Latitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    self.latitude, lat_array[0], lat_array[-1]
                )
            )

        # Get pressure level data from file
        try:
            levels = (
                100 * weather_data.variables[dictionary["level"]][:]
            )  # Convert mbar to Pa
        except:
            raise ValueError(
                "Unable to read pressure levels from file. Check file and dictionary."
            )

        # Get geopotential data from file
        try:
            geopotentials = weather_data.variables[dictionary["geopotential_height"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except:
            try:
                geopotentials = (
                    weather_data.variables[dictionary["geopotential"]][
                        time_index,
                        :,
                        (lat_index - 1, lat_index),
                        (lon_index - 1, lon_index),
                    ]
                    / self.standard_g
                )
            except:
                raise ValueError(
                    "Unable to read geopotential height"
                    " nor geopotential from file. At least"
                    " one of them is necessary. Check "
                    " file and dictionary."
                )

        # Get temperature from file
        try:
            temperatures = weather_data.variables[dictionary["temperature"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except:
            raise ValueError(
                "Unable to read temperature from file. Check file and dictionary."
            )

        # Get wind data from file
        try:
            wind_us = weather_data.variables[dictionary["u_wind"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except:
            raise ValueError(
                "Unable to read wind-u component. Check file and dictionary."
            )
        try:
            wind_vs = weather_data.variables[dictionary["v_wind"]][
                time_index, :, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
            ]
        except:
            raise ValueError(
                "Unable to read wind-v component. Check file and dictionary."
            )

        # Prepare for bilinear interpolation
        x, y = self.latitude, lon
        x1, y1 = lat_array[lat_index - 1], lon_array[lon_index - 1]
        x2, y2 = lat_array[lat_index], lon_array[lon_index]

        # Determine geopotential in lat, lon
        f_x1_y1 = geopotentials[:, 0, 0]
        f_x1_y2 = geopotentials[:, 0, 1]
        f_x2_y1 = geopotentials[:, 1, 0]
        f_x2_y2 = geopotentials[:, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        height = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine temperature in lat, lon
        f_x1_y1 = temperatures[:, 0, 0]
        f_x1_y2 = temperatures[:, 0, 1]
        f_x2_y1 = temperatures[:, 1, 0]
        f_x2_y2 = temperatures[:, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        temperature = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind u in lat, lon
        f_x1_y1 = wind_us[:, 0, 0]
        f_x1_y2 = wind_us[:, 0, 1]
        f_x2_y1 = wind_us[:, 1, 0]
        f_x2_y2 = wind_us[:, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        wind_u = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind v in lat, lon
        f_x1_y1 = wind_vs[:, 0, 0]
        f_x1_y2 = wind_vs[:, 0, 1]
        f_x2_y1 = wind_vs[:, 1, 0]
        f_x2_y2 = wind_vs[:, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        wind_v = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind speed, heading and direction
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_heading = np.arctan2(wind_u, wind_v) * (180 / np.pi) % 360
        wind_direction = (wind_heading - 180) % 360

        # Convert geopotential height to geometric height
        R = self.earth_radius
        height = R * height / (R - height)

        # Combine all data into big array
        data_array = np.ma.column_stack(
            [
                levels,
                height,
                temperature,
                wind_u,
                wind_v,
                wind_heading,
                wind_direction,
                wind_speed,
            ]
        )

        # Remove lines with masked content
        if np.any(data_array.mask):
            data_array = np.ma.compress_rows(data_array)
            warnings.warn(
                "Some values were missing from this weather dataset, therefore, certain pressure levels were removed."
            )
        # Save atmospheric data
        self.pressure = Function(
            data_array[:, (1, 0)],
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
            interpolation="linear",
        )
        self.temperature = Function(
            data_array[:, (1, 2)],
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )
        self.wind_direction = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.wind_heading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.wind_speed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_x = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.max_expected_height = max(height[0], height[-1])

        # Get elevation data from file
        if dictionary["surface_geopotential_height"] is not None:
            try:
                elevations = weather_data.variables[
                    dictionary["surface_geopotential_height"]
                ][time_index, (lat_index - 1, lat_index), (lon_index - 1, lon_index)]
                f_x1_y1 = elevations[0, 0]
                f_x1_y2 = elevations[0, 1]
                f_x2_y1 = elevations[1, 0]
                f_x2_y2 = elevations[1, 1]
                f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + (
                    (x - x1) / (x2 - x1)
                ) * f_x2_y1
                f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + (
                    (x - x1) / (x2 - x1)
                ) * f_x2_y2
                self.elevation = ((y2 - y) / (y2 - y1)) * f_x_y1 + (
                    (y - y1) / (y2 - y1)
                ) * f_x_y2
            except:
                raise ValueError(
                    "Unable to read surface elevation data. Check file and dictionary."
                )

        # Compute info data
        self.atmospheric_model_init_date = netCDF4.num2date(
            time_array[0], time_array.units, calendar="gregorian"
        )
        self.atmospheric_model_end_date = netCDF4.num2date(
            time_array[-1], time_array.units, calendar="gregorian"
        )
        self.atmospheric_model_interval = netCDF4.num2date(
            (time_array[-1] - time_array[0]) / (len(time_array) - 1),
            time_array.units,
            calendar="gregorian",
        ).hour
        self.atmospheric_model_init_lat = lat_array[0]
        self.atmospheric_model_end_lat = lat_array[-1]
        self.atmospheric_model_init_lon = lon_array[0]
        self.atmospheric_model_end_lon = lon_array[-1]

        # Save debugging data
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.lon_index = lon_index
        self.lat_index = lat_index
        self.geopotentials = geopotentials
        self.wind_us = wind_us
        self.wind_vs = wind_vs
        self.levels = levels
        self.temperatures = temperatures
        self.time_array = time_array
        self.height = height

        # Close weather data
        weather_data.close()

        return None

    @requires_netCDF4
    def process_ensemble(self, file, dictionary):
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
        ``Environment.selectEnsembleMemberMember()``.

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
                    "ensemble": "ens",
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
        if self.datetime_date is None:
            raise TypeError(
                "Please specify Date (array-like) when "
                "initializing this Environment. "
                "Alternatively, use the Environment.set_date"
                " method."
            )
        if self.latitude is None:
            raise TypeError(
                "Please specify Location (lat, lon). when "
                "initializing this Environment. "
                "Alternatively, use the Environment."
                "set_location method."
            )

        # Read weather file
        weather_data = netCDF4.Dataset(file)

        # Get time, latitude and longitude data from file
        time_array = weather_data.variables[dictionary["time"]]
        lon_array = weather_data.variables[dictionary["longitude"]][:].tolist()
        lat_array = weather_data.variables[dictionary["latitude"]][:].tolist()

        # Find time index
        time_index = netCDF4.date2index(
            self.datetime_date, time_array, calendar="gregorian", select="nearest"
        )
        # Convert times do dates and numbers
        input_time_num = netCDF4.date2num(
            self.datetime_date, time_array.units, calendar="gregorian"
        )
        file_time_num = time_array[time_index]
        file_time_date = netCDF4.num2date(
            time_array[time_index], time_array.units, calendar="gregorian"
        )
        # Check if time is inside range supplied by file
        if time_index == 0 and input_time_num < file_time_num:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which starts at {:}.".format(
                    file_time_date
                )
            )
        elif time_index == len(time_array) - 1 and input_time_num > file_time_num:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which ends at {:}.".format(
                    file_time_date
                )
            )
        # Check if time is exactly equal to one in the file
        if input_time_num != file_time_num:
            warnings.warn(
                "Exact chosen launch time is not available in the provided file, using {:} UTC instead.".format(
                    file_time_date
                )
            )

        # Find longitude index
        # Determine if file uses -180 to 180 or 0 to 360
        if lon_array[0] < 0 or lon_array[-1] < 0:
            # Convert input to -180 - 180
            lon = (
                self.longitude if self.longitude < 180 else -180 + self.longitude % 180
            )
        else:
            # Convert input to 0 - 360
            lon = self.longitude % 360
        # Check if reversed or sorted
        if lon_array[0] < lon_array[-1]:
            # Deal with sorted lon_array
            lon_index = bisect.bisect(lon_array, lon)
        else:
            # Deal with reversed lon_array
            lon_array.reverse()
            lon_index = len(lon_array) - bisect.bisect_left(lon_array, lon)
            lon_array.reverse()
        # Take care of longitude value equal to maximum longitude in the grid
        if lon_index == len(lon_array) and lon_array[lon_index - 1] == lon:
            lon_index = lon_index - 1
        # Check if longitude value is inside the grid
        if lon_index == 0 or lon_index == len(lon_array):
            raise ValueError(
                "Longitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    lon, lon_array[0], lon_array[-1]
                )
            )

        # Find latitude index
        # Check if reversed or sorted
        if lat_array[0] < lat_array[-1]:
            # Deal with sorted lat_array
            lat_index = bisect.bisect(lat_array, self.latitude)
        else:
            # Deal with reversed lat_array
            lat_array.reverse()
            lat_index = len(lat_array) - bisect.bisect_left(lat_array, self.latitude)
            lat_array.reverse()
        # Take care of latitude value equal to maximum longitude in the grid
        if lat_index == len(lat_array) and lat_array[lat_index - 1] == self.latitude:
            lat_index = lat_index - 1
        # Check if latitude value is inside the grid
        if lat_index == 0 or lat_index == len(lat_array):
            raise ValueError(
                "Latitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    self.latitude, lat_array[0], lat_array[-1]
                )
            )

        # Get ensemble data from file
        try:
            num_members = len(weather_data.variables[dictionary["ensemble"]][:])
        except:
            raise ValueError(
                "Unable to read ensemble data from file. Check file and dictionary."
            )

        # Get pressure level data from file
        try:
            levels = (
                100 * weather_data.variables[dictionary["level"]][:]
            )  # Convert mbar to Pa
        except:
            raise ValueError(
                "Unable to read pressure levels from file. Check file and dictionary."
            )

        ##
        inverse_dictionary = {v: k for k, v in dictionary.items()}
        param_dictionary = {
            "time": time_index,
            "ensemble": range(num_members),
            "level": range(len(levels)),
            "latitude": (lat_index - 1, lat_index),
            "longitude": (lon_index - 1, lon_index),
        }
        ##

        # Get geopotential data from file
        try:
            dimensions = weather_data.variables[
                dictionary["geopotential_height"]
            ].dimensions[:]
            params = tuple(
                [param_dictionary[inverse_dictionary[dim]] for dim in dimensions]
            )
            geopotentials = weather_data.variables[dictionary["geopotential_height"]][
                params
            ]
        except:
            try:
                dimensions = weather_data.variables[
                    dictionary["geopotential"]
                ].dimensions[:]
                params = tuple(
                    [param_dictionary[inverse_dictionary[dim]] for dim in dimensions]
                )
                geopotentials = (
                    weather_data.variables[dictionary["geopotential"]][params]
                    / self.standard_g
                )
            except:
                raise ValueError(
                    "Unable to read geopotential height"
                    " nor geopotential from file. At least"
                    " one of them is necessary. Check "
                    " file and dictionary."
                )

        # Get temperature from file
        try:
            temperatures = weather_data.variables[dictionary["temperature"]][params]
        except:
            raise ValueError(
                "Unable to read temperature from file. Check file and dictionary."
            )

        # Get wind data from file
        try:
            wind_us = weather_data.variables[dictionary["u_wind"]][params]
        except:
            raise ValueError(
                "Unable to read wind-u component. Check file and dictionary."
            )
        try:
            wind_vs = weather_data.variables[dictionary["v_wind"]][params]
        except:
            raise ValueError(
                "Unable to read wind-v component. Check file and dictionary."
            )

        # Prepare for bilinear interpolation
        x, y = self.latitude, lon
        x1, y1 = lat_array[lat_index - 1], lon_array[lon_index - 1]
        x2, y2 = lat_array[lat_index], lon_array[lon_index]

        # Determine geopotential in lat, lon
        f_x1_y1 = geopotentials[:, :, 0, 0]
        f_x1_y2 = geopotentials[:, :, 0, 1]
        f_x2_y1 = geopotentials[:, :, 1, 0]
        f_x2_y2 = geopotentials[:, :, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        height = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine temperature in lat, lon
        f_x1_y1 = temperatures[:, :, 0, 0]
        f_x1_y2 = temperatures[:, :, 0, 1]
        f_x2_y1 = temperatures[:, :, 1, 0]
        f_x2_y2 = temperatures[:, :, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        temperature = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind u in lat, lon
        f_x1_y1 = wind_us[:, :, 0, 0]
        f_x1_y2 = wind_us[:, :, 0, 1]
        f_x2_y1 = wind_us[:, :, 1, 0]
        f_x2_y2 = wind_us[:, :, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        wind_u = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind v in lat, lon
        f_x1_y1 = wind_vs[:, :, 0, 0]
        f_x1_y2 = wind_vs[:, :, 0, 1]
        f_x2_y1 = wind_vs[:, :, 1, 0]
        f_x2_y2 = wind_vs[:, :, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        wind_v = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind speed, heading and direction
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_heading = np.arctan2(wind_u, wind_v) * (180 / np.pi) % 360
        wind_direction = (wind_heading - 180) % 360

        # Convert geopotential height to geometric height
        R = self.earth_radius
        height = R * height / (R - height)

        # Save ensemble data
        self.level_ensemble = levels
        self.height_ensemble = height
        self.temperature_ensemble = temperature
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
            try:
                elevations = weather_data.variables[
                    dictionary["surface_geopotential_height"]
                ][time_index, (lat_index - 1, lat_index), (lon_index - 1, lon_index)]
                f_x1_y1 = elevations[0, 0]
                f_x1_y2 = elevations[0, 1]
                f_x2_y1 = elevations[1, 0]
                f_x2_y2 = elevations[1, 1]
                f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + (
                    (x - x1) / (x2 - x1)
                ) * f_x2_y1
                f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + (
                    (x - x1) / (x2 - x1)
                ) * f_x2_y2
                self.elevation = ((y2 - y) / (y2 - y1)) * f_x_y1 + (
                    (y - y1) / (y2 - y1)
                ) * f_x_y2
            except:
                raise ValueError(
                    "Unable to read surface elevation data. Check file and dictionary."
                )

        # Compute info data
        self.atmospheric_model_init_date = netCDF4.num2date(
            time_array[0], time_array.units, calendar="gregorian"
        )
        self.atmospheric_model_end_date = netCDF4.num2date(
            time_array[-1], time_array.units, calendar="gregorian"
        )
        self.atmospheric_model_interval = netCDF4.num2date(
            (time_array[-1] - time_array[0]) / (len(time_array) - 1),
            time_array.units,
            calendar="gregorian",
        ).hour
        self.atmospheric_model_init_lat = lat_array[0]
        self.atmospheric_model_end_lat = lat_array[-1]
        self.atmospheric_model_init_lon = lon_array[0]
        self.atmospheric_model_end_lon = lon_array[-1]

        # Save debugging data
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.lon_index = lon_index
        self.lat_index = lat_index
        self.geopotentials = geopotentials
        self.wind_us = wind_us
        self.wind_vs = wind_vs
        self.levels = levels
        self.temperatures = temperatures
        self.time_array = time_array
        self.height = height

        # Close weather data
        weather_data.close()

        return None

    def select_ensemble_member(self, member=0):
        """Activates ensemble member, meaning that all atmospheric variables
        read from the Environment instance will correspond to the desired
        ensemble member.

        Parameters
        ---------
        member : int
            Ensemble member to be activated. Starts from 0.

        Returns
        -------
        None
        """
        # Verify ensemble member
        if member >= self.num_ensemble_members:
            raise ValueError(
                "Please choose member from 0 to {:d}".format(
                    self.num_ensemble_members - 1
                )
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
        data_array = np.ma.column_stack(
            [
                levels,
                height,
                temperature,
                wind_u,
                wind_v,
                wind_heading,
                wind_direction,
                wind_speed,
            ]
        )

        # Remove lines with masked content
        if np.any(data_array.mask):
            data_array = np.ma.compress_rows(data_array)
            warnings.warn(
                "Some values were missing from this weather dataset, therefore, certain pressure levels were removed."
            )

        # Save atmospheric data
        self.pressure = Function(
            data_array[:, (1, 0)],
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
            interpolation="linear",
        )
        self.temperature = Function(
            data_array[:, (1, 2)],
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )
        self.wind_direction = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.wind_heading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.wind_speed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_x = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.wind_velocity_y = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.max_expected_height = max(height[0], height[-1])

        # Save ensemble member
        self.ensemble_member = member

        # Update air density
        self.calculate_density_profile()

        # Update speed of sound
        self.calculate_speed_of_sound_profile()

        # Update dynamic viscosity
        self.calculate_dynamic_viscosity()

        return None

    def load_international_standard_atmosphere(self):
        """Defines the pressure and temperature profile functions set
        by `ISO 2533` for the International Standard atmosphere and saves
        them as ``Environment.pressure_ISA`` and ``Environment.temperature_ISA``.

        Returns
        -------
        None
        """
        # Define international standard atmosphere layers
        geopotential_height = [
            -2e3,
            0,
            11e3,
            20e3,
            32e3,
            47e3,
            51e3,
            71e3,
            80e3,
        ]  # in geopotential m
        temperature = [
            301.15,
            288.15,
            216.65,
            216.65,
            228.65,
            270.65,
            270.65,
            214.65,
            196.65,
        ]  # in K
        beta = [
            -6.5e-3,
            -6.5e-3,
            0,
            1e-3,
            2.8e-3,
            0,
            -2.8e-3,
            -2e-3,
            0,
        ]  # Temperature gradient in K/m
        pressure = [
            1.27774e5,
            1.01325e5,
            2.26320e4,
            5.47487e3,
            8.680164e2,
            1.10906e2,
            6.69384e1,
            3.95639e0,
            8.86272e-2,
        ]  # in Pa

        # Convert geopotential height to geometric height
        ER = self.earth_radius
        height = [ER * H / (ER - H) for H in geopotential_height]

        # Save international standard atmosphere temperature profile
        self.temperature_ISA = Function(
            np.column_stack([height, temperature]),
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )

        # Get gravity and R
        g = self.standard_g
        R = self.air_gas_constant

        # Create function to compute pressure at a given geometric height
        def pressure_function(h):
            # Convert geometric to geopotential height
            H = ER * h / (ER + h)

            # Check if height is within bounds, return extrapolated value if not
            if H < -2000:
                return pressure[0]
            elif H > 80000:
                return pressure[-1]

            # Find layer that contains height h
            layer = bisect.bisect(geopotential_height, H) - 1

            # Retrieve layer base geopotential height, temp, beta and pressure
            Hb = geopotential_height[layer]
            Tb = temperature[layer]
            Pb = pressure[layer]
            B = beta[layer]

            # Compute pressure
            if B != 0:
                P = Pb * (1 + (B / Tb) * (H - Hb)) ** (-g / (B * R))
            else:
                T = Tb + B * (H - Hb)
                P = Pb * np.exp(-(H - Hb) * (g / (R * T)))

            # Return answer
            return P

        # Save international standard atmosphere pressure profile
        self.pressure_ISA = Function(
            pressure_function,
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
        )

        return None

    def calculate_density_profile(self):
        """Compute the density of the atmosphere as a function of
        height by using the formula rho = P/(RT). This function is
        automatically called whenever a new atmospheric model is set.

        Returns
        -------
        None
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

        return None

    def calculate_speed_of_sound_profile(self):
        """Compute the speed of sound in the atmosphere as a function
        of height by using the formula a = sqrt(gamma*R*T). This
        function is automatically called whenever a new atmospheric
        model is set.

        Returns
        -------
        None
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

        return None

    def calculate_dynamic_viscosity(self):
        """Compute the dynamic viscosity of the atmosphere as a function of
        height by using the formula given in ISO 2533 u = B*T^(1.5)/(T+S).
        This function is automatically called whenever a new atmospheric model is set.
        Warning: This equation is invalid for very high or very low temperatures
        and under conditions occurring at altitudes above 90 km.

        Returns
        -------
        None
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

        return None

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

        Returns
        -------
        None
        """
        # Recalculate wind_velocity_x and wind_velocity_y
        self.wind_velocity_x = self.wind_velocity_x + wind_gust_x
        self.wind_velocity_y = self.wind_velocity_y + wind_gust_y

        # Reset wind_velocity_x and wind_velocity_y details
        self.wind_velocity_x.set_inputs("Height (m)")
        self.wind_velocity_x.set_outputs("Wind Velocity X (m/s)")
        self.wind_velocity_y.set_inputs("Height (m)")
        self.wind_velocity_y.set_outputs("Wind Velocity Y (m/s)")

        # Reset wind heading and velocity magnitude
        self.wind_heading = Function(
            lambda h: (180 / np.pi)
            * np.arctan2(self.wind_velocity_x(h), self.wind_velocity_y(h))
            % 360,
            "Height (m)",
            "Wind Heading (degrees)",
            extrapolation="constant",
        )
        self.wind_speed = Function(
            lambda h: (self.wind_velocity_x(h) ** 2 + self.wind_velocity_y(h) ** 2)
            ** 0.5,
            "Height (m)",
            "Wind Speed (m/s)",
            extrapolation="constant",
        )

        return None

    def info(self):
        """Prints most important data and graphs available about the
        Environment.

        Return
        ------
        None
        """

        self.prints.all()
        self.plots.info()
        return None

    def all_info(self):
        """Prints out all data and graphs available about the Environment.

        Returns
        -------
        None
        """

        self.prints.all()
        self.plots.all()

        return None

    def all_plot_info_returned(self):
        """Returns a dictionary with all plot information available about the Environment.

        Returns
        ------
        plot_info : Dict
            Dict of data relevant to plot externally
        """
        grid = np.linspace(self.elevation, self.max_expected_height)
        plot_info = dict(
            grid=[i for i in grid],
            wind_speed=[self.wind_speed(i) for i in grid],
            wind_direction=[self.wind_direction(i) for i in grid],
            speed_of_sound=[self.speed_of_sound(i) for i in grid],
            density=[self.density(i) for i in grid],
            wind_vel_x=[self.wind_velocity_x(i) for i in grid],
            wind_vel_y=[self.wind_velocity_y(i) for i in grid],
            pressure=[self.pressure(i) / 100 for i in grid],
            temperature=[self.temperature(i) for i in grid],
        )
        if self.atmospheric_model_type != "Ensemble":
            return plot_info
        current_member = self.ensemble_member
        # List for each ensemble
        plot_info["ensemble_wind_velocity_x"] = []
        for i in range(self.num_ensemble_members):
            self.select_ensemble_member(i)
            plot_info["ensemble_wind_velocity_x"].append(
                [self.wind_velocity_x(i) for i in grid]
            )
        plot_info["ensemble_wind_velocity_y"] = []
        for i in range(self.num_ensemble_members):
            self.select_ensemble_member(i)
            plot_info["ensemble_wind_velocity_y"].append(
                [self.wind_velocity_y(i) for i in grid]
            )
        plot_info["ensemble_wind_speed"] = []
        for i in range(self.num_ensemble_members):
            self.select_ensemble_member(i)
            plot_info["ensemble_wind_speed"].append([self.wind_speed(i) for i in grid])
        plot_info["ensemble_wind_direction"] = []
        for i in range(self.num_ensemble_members):
            self.select_ensemble_member(i)
            plot_info["ensemble_wind_direction"].append(
                [self.wind_direction(i) for i in grid]
            )
        plot_info["ensemble_pressure"] = []
        for i in range(self.num_ensemble_members):
            self.select_ensemble_member(i)
            plot_info["ensemble_pressure"].append([self.pressure(i) for i in grid])
        plot_info["ensemble_temperature"] = []
        for i in range(self.num_ensemble_members):
            self.select_ensemble_member(i)
            plot_info["ensemble_temperature"].append(
                [self.temperature(i) for i in grid]
            )

        # Clean up
        self.select_ensemble_member(current_member)
        return plot_info

    def all_info_returned(self):
        """Returns as dicts all data available about the Environment.

        Returns
        ------
        info : Dict
            Information relevant about the Environment class.
        """

        # Dictionary creation, if not commented follows the SI
        info = dict(
            grav=self.gravity,
            elevation=self.elevation,
            model_type=self.atmospheric_model_type,
            model_type_max_expected_height=self.max_expected_height,
            wind_speed=self.wind_speed(self.elevation),
            wind_direction=self.wind_direction(self.elevation),
            wind_heading=self.wind_heading(self.elevation),
            surface_pressure=self.pressure(self.elevation) / 100,  # in hPa
            surface_temperature=self.temperature(self.elevation),
            surface_air_density=self.density(self.elevation),
            surface_speed_of_sound=self.speed_of_sound(self.elevation),
        )
        if self.datetime_date != None:
            info["launch_date"] = self.datetime_date.strftime("%Y-%d-%m %H:%M:%S")
        if self.latitude != None and self.longitude != None:
            info["lat"] = self.latitude
            info["lon"] = self.longitude
        if info["model_type"] in ["Forecast", "Reanalysis", "Ensemble"]:
            info["init_date"] = self.atmospheric_model_init_date.strftime(
                "%Y-%d-%m %H:%M:%S"
            )
            info["endDate"] = self.atmospheric_model_end_date.strftime(
                "%Y-%d-%m %H:%M:%S"
            )
            info["interval"] = self.atmospheric_model_interval
            info["init_lat"] = self.atmospheric_model_init_lat
            info["end_lat"] = self.atmospheric_model_end_lat
            info["init_lon"] = self.atmospheric_model_init_lon
            info["end_lon"] = self.atmospheric_model_end_lon
        if info["model_type"] == "Ensemble":
            info["num_ensemble_members"] = self.num_ensemble_members
            info["selected_ensemble_member"] = self.ensemble_member
        return info

    def export_environment(self, filename="environment"):
        """Export important attributes of Environment class to a ``.json`` file,
        saving all the information needed to recreate the same environment using
        custom_atmosphere.

        Parameters
        ----------
        filename : string
            The name of the file to be saved, without the extension.

        Return
        ------
        None
        """

        try:
            atmospheric_model_file = self.atmospheric_model_file
            atmospheric_model_dict = self.atmospheric_model_dict
        except AttributeError:
            atmospheric_model_file = ""
            atmospheric_model_dict = ""

        self.export_env_dictionary = {
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
            "atmospheric_model_file": atmospheric_model_file,
            "atmospheric_model_dict": atmospheric_model_dict,
            "atmospheric_model_pressure_profile": ma.getdata(
                self.pressure.get_source()
            ).tolist(),
            "atmospheric_model_temperature_profile": ma.getdata(
                self.temperature.get_source()
            ).tolist(),
            "atmospheric_model_wind_velocity_x_profile": ma.getdata(
                self.wind_velocity_x.get_source()
            ).tolist(),
            "atmospheric_model_wind_velocity_y_profile": ma.getdata(
                self.wind_velocity_y.get_source()
            ).tolist(),
        }

        f = open(filename + ".json", "w")

        # write json object to file
        f.write(
            json.dumps(
                self.export_env_dictionary, sort_keys=False, indent=4, default=str
            )
        )

        # close file
        f.close()
        print("Your Environment file was saved, check it out: " + filename + ".json")
        print(
            "You can use it in the future by using the custom_atmosphere atmospheric model."
        )

        return None

    def set_earth_geometry(self, datum):
        """Sets the Earth geometry for the ``Environment`` class based on the
        datum provided.

        Parameters
        ----------
        datum : str
            The datum to be used for the Earth geometry.

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
        except KeyError:
            raise AttributeError(
                f"The reference system {datum} for Earth geometry " "is not recognized."
            )

    # Auxiliary functions - Geodesic Coordinates

    @staticmethod
    def geodesic_to_utm(
        lat, lon, semi_major_axis=6378137.0, flattening=1 / 298.257223563
    ):
        """Function which converts geodetic coordinates, i.e. lat/lon, to UTM
        projection coordinates. Can be used only for latitudes between -80.00°
        and 84.00°

        Parameters
        ----------
        lat : float
            The latitude coordinates of the point of analysis, must be contained
            between -80.00° and 84.00°
        lon : float
            The longitude coordinates of the point of analysis, must be
            contained between -180.00° and 180.00°
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
        x : float
            East coordinate, always positive
        y : float
            North coordinate, always positive
        utm_zone : int
            The number of the UTM zone of the point of analysis, can vary
            between 1 and 60
        utm_letter : string
            The letter of the UTM zone of the point of analysis, can vary
            between C and X, omitting the letters "I" and "O"
        hemis : string
            Returns "S" for southern hemisphere and "N" for Northern hemisphere
        EW : string
            Returns "W" for western hemisphere and "E" for eastern hemisphere
        """

        # Calculate the central meridian of UTM zone
        if lon != 0:
            signal = lon / abs(lon)
            if signal > 0:
                aux = lon - 3
                aux = aux * signal
                div = aux // 6
                lon_mc = div * 6 + 3
                EW = "E"
            else:
                aux = lon + 3
                aux = aux * signal
                div = aux // 6
                lon_mc = (div * 6 + 3) * signal
                EW = "W"
        else:
            lon_mc = 3
            EW = "W|E"

        # Evaluate the hemisphere and determine the N coordinate at the Equator
        if lat < 0:
            N0 = 10000000
            hemis = "S"
        else:
            N0 = 0
            hemis = "N"

        # Convert the input lat and lon to radians
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        lon_mc = lon_mc * np.pi / 180

        # Evaluate reference parameters
        K0 = 1 - 1 / 2500
        e2 = 2 * flattening - flattening**2
        e2lin = e2 / (1 - e2)

        # Evaluate auxiliary parameters
        A = e2 * e2
        B = A * e2
        C = np.sin(2 * lat)
        D = np.sin(4 * lat)
        E = np.sin(6 * lat)
        F = (1 - e2 / 4 - 3 * A / 64 - 5 * B / 256) * lat
        G = (3 * e2 / 8 + 3 * A / 32 + 45 * B / 1024) * C
        H = (15 * A / 256 + 45 * B / 1024) * D
        I = (35 * B / 3072) * E

        # Evaluate other reference parameters
        n = semi_major_axis / ((1 - e2 * (np.sin(lat) ** 2)) ** 0.5)
        t = np.tan(lat) ** 2
        c = e2lin * (np.cos(lat) ** 2)
        ag = (lon - lon_mc) * np.cos(lat)
        m = semi_major_axis * (F - G + H - I)

        # Evaluate new auxiliary parameters
        J = (1 - t + c) * ag * ag * ag / 6
        K = (5 - 18 * t + t * t + 72 * c - 58 * e2lin) * (ag**5) / 120
        L = (5 - t + 9 * c + 4 * c * c) * ag * ag * ag * ag / 24
        M = (61 - 58 * t + t * t + 600 * c - 330 * e2lin) * (ag**6) / 720

        # Evaluate the final coordinates
        x = 500000 + K0 * n * (ag + J + K)
        y = N0 + K0 * (m + n * np.tan(lat) * (ag * ag / 2 + L + M))

        # Convert the output lat and lon to degrees
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi
        lon_mc = lon_mc * 180 / np.pi

        # Calculate the UTM zone number
        utm_zone = int((lon_mc + 183) / 6)

        # Calculate the UTM zone letter
        letters = "CDEFGHJKLMNPQRSTUVWXX"
        utm_letter = letters[int(80 + lat) >> 3]

        return x, y, utm_zone, utm_letter, hemis, EW

    @staticmethod
    def utm_to_geodesic(
        x, y, utm_zone, hemis, semi_major_axis=6378137.0, flattening=1 / 298.257223563
    ):
        """Function to convert UTM coordinates to geodesic coordinates
        (i.e. latitude and longitude). The latitude should be between -80°
        and 84°

        Parameters
        ----------
        x : float
            East UTM coordinate in meters
        y : float
            North UTM coordinate in meters
        utm_zone : int
            The number of the UTM zone of the point of analysis, can vary
            between 1 and 60
        hemis : string
            Equals to "S" for southern hemisphere and "N" for Northern
            hemisphere
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
        lat : float
            latitude of the analyzed point
        lon : float
            latitude of the analyzed point
        """

        if hemis == "N":
            y = y + 10000000

        # Calculate the Central Meridian from the UTM zone number
        central_meridian = utm_zone * 6 - 183  # degrees

        # Calculate reference values
        K0 = 1 - 1 / 2500
        e2 = 2 * flattening - flattening**2
        e2lin = e2 / (1 - e2)
        e1 = (1 - (1 - e2) ** 0.5) / (1 + (1 - e2) ** 0.5)

        # Calculate auxiliary values
        A = e2 * e2
        B = A * e2
        C = e1 * e1
        D = e1 * C
        E = e1 * D

        m = (y - 10000000) / K0
        mi = m / (semi_major_axis * (1 - e2 / 4 - 3 * A / 64 - 5 * B / 256))

        # Calculate other auxiliary values
        F = (3 * e1 / 2 - 27 * D / 32) * np.sin(2 * mi)
        G = (21 * C / 16 - 55 * E / 32) * np.sin(4 * mi)
        H = (151 * D / 96) * np.sin(6 * mi)

        lat1 = mi + F + G + H
        c1 = e2lin * (np.cos(lat1) ** 2)
        t1 = np.tan(lat1) ** 2
        n1 = semi_major_axis / ((1 - e2 * (np.sin(lat1) ** 2)) ** 0.5)
        quoc = (1 - e2 * np.sin(lat1) * np.sin(lat1)) ** 3
        r1 = semi_major_axis * (1 - e2) / (quoc**0.5)
        d = (x - 500000) / (n1 * K0)

        # Calculate other auxiliary values
        I = (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * e2lin) * d * d * d * d / 24
        J = (
            (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * e2lin - 3 * c1 * c1)
            * (d**6)
            / 720
        )
        K = d - (1 + 2 * t1 + c1) * d * d * d / 6
        L = (
            (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * e2lin + 24 * t1 * t1)
            * (d**5)
            / 120
        )

        # Finally calculate the coordinates in lat/lot
        lat = lat1 - (n1 * np.tan(lat1) / r1) * (d * d / 2 - I + J)
        lon = central_meridian * np.pi / 180 + (K + L) / np.cos(lat1)

        # Convert final lat/lon to Degrees
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi

        return lat, lon

    @staticmethod
    def calculate_earth_radius(
        lat, semi_major_axis=6378137.0, flattening=1 / 298.257223563
    ):
        """Simple function to calculate the Earth Radius at a specific latitude
        based on ellipsoidal reference model (datum). The earth radius here is
        assumed as the distance between the ellipsoid's center of gravity and a
        point on ellipsoid surface at the desired
        Pay attention: The ellipsoid is an approximation for the earth model and
        will obviously output an estimate of the perfect distance between
        earth's relief and its center of gravity.

        Parameters
        ----------
        lat : float
            latitude in which the Earth radius will be calculated
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
            Earth Radius at the desired latitude in meters
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
        """Function to convert an angle in decimal degrees to deg/min/sec.
         Converts (°) to (° ' ")

        Parameters
        ----------
        angle : float
            The angle that you need convert to deg/min/sec. Must be given in
            decimal degrees.

        Returns
        -------
        degrees : float
            The degrees.
        arc_minutes : float
            The arc minutes. 1 arc-minute = (1/60)*degree
        arc_seconds : float
            The arc Seconds. 1 arc-second = (1/3600)*degree
        """
        sign = -1 if angle < 0 else 1
        degrees = int(abs(angle)) * sign
        remainder = abs(angle) - abs(degrees)
        arc_minutes = int(remainder * 60)
        arc_seconds = (remainder * 60 - arc_minutes) * 60
        return degrees, arc_minutes, arc_seconds
