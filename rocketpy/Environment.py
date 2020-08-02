# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"

import re
import math
import bisect
import warnings
import time
from datetime import datetime, timedelta
from inspect import signature, getsourcelines
from collections import namedtuple

import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import requests

try:
    import netCDF4
except ImportError:
    warnings.warn("Unable to load netCDF4. NetCDF files and OPeNDAP will not be imported.", ImportWarning)

from .Function import Function

class Environment:
    """Keeps all environment information stored, such as wind and temperature
    conditions, as well as gravity and rail length.
    
    Attributes
    ----------
    
        Constants
        Environment.earthRadius : float
            Value of Earth's Radius = 6.3781e6 m.
        Environment.airGasConstant : float
            Value of Air's Gast Constant = 287.05287 J/K/Kg

        Gravity and Launch Rail Length:
        Environment.rl : float
            Launch rail length in meters.
        Environment.g : float
            Positive value of gravitational acceleration in m/s^2.

        Coordinates and Date:
        Environment.lat : float
            Launch site latitude.
        Environment.lon : float
            Launch site longitude.
        Environment.elevation : float
            Launch site elevation.
        Environment.date : datetime
            Date time of launch.    
        Environment.maxExpectedHeight : float
            Maximum altitude in meters to keep weather data.
            Used especially for plotting range.
            Can be altered as desired.

        Atmosphere Static Conditions:
        Environment.pressureISA : Function
            Air pressure in Pa as a function of altitude as defined
            by the International Standard Atmosphere ISO 2533.
            Only defined after load Environment.loadInternationalStandardAtmosphere
            has been called.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.temperatureISA : Function
            Air temperature in K as a function of altitude  as defined
            by the International Standard Atmosphere ISO 2533.
            Only defined after load Environment.loadInternationalStandardAtmosphere
            has been called.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.pressure : Function
            Air pressure in Pa as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.temperature : Function
            Air temperature in K as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.speedOfSound : Function
            Speed of sound in air in m/s as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.density : Function
            Air density in kg/m³ as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.dynamicViscosity : Function
            Air dynamic viscosity in Pa s as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.

        Atmosphere Wind Conditions: 
        Environment.windSpeed : Function
            Wind speed in m/s as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.windDirection : Function
            Wind direction (from which the wind blows)
            in degrees relative to north (positive clockwise)
            as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.windHeading : Function
            Wind heading (direction towards which the wind blows)
            in degrees relative to north (positive clockwise)
            as a function of altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.windVelocityX : Function
            Wind U, or X (east) component of wind velocity in m/s as a function of
            altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.
        Environment.windVelocityY : Function
            Wind V, or Y (east) component of wind velocity in m/s as a function of
            altitude.
            Can be accessed as regular array, or called
            as a function. See Function for more information.

        Atmospheric Model Details
        Environment.atmosphericModelType : string
            Describes the atmospheric model which is being used.
            Can take the only the following values: 'StandardAtmosphere',
            'CustomAtmosphere', 'WyomingSounding', 'NOAARucSounding',
            'Forecast', 'Reanalysis', 'Ensemble'.
        Environment.atmosphericModelFile : string
            Adress of the file used for the atmospheric model being used.
            Only defined for 'WyomingSounding', 'NOAARucSounding',
            'Forecast', 'Reanalysis', 'Ensemble'
        Environment.atmosphericModelDict : dictionary
            Dictionary used to properly interpret netCDF and OPeNDAP
            files. Only defined for 'Forecast', 'Reanalysis', 'Ensemble'.
        Environment.atmosphericModelInitDate : datetime
            Datetime object instance of first availabe date in netCDF
            and OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.
        Environment.atmosphericModelEndDate : datetime
            Datetime object instance of last availabe date in netCDF
            and OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.
        Environment.atmosphericModelInterval : int
            Hour step between weather condition used in netCDF and
            OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.
        Environment.atmosphericModelInitLat : float
            Latitude of vertex just before the launch site in netCDF
            and OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.
        Environment.atmosphericModelEndLat : float
            Latitude of vertex just after the launch site in netCDF
            and OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.
        Environment.atmosphericModelInitLon : float
            Longitude of vertex just before the launch site in netCDF
            and OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.
        Environment.atmosphericModelEndLon : float
            Longitude of vertex just after the launch site in netCDF
            and OPeNDAP files when using 'Forecast', 'Reanalysis' or
            'Ensemble'.

        Atmospheric Model Storage
        Environment.latArray : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. 2x2 matrix for each pressure level of latitudes
            corresponding to the vertices of the grid cell which surrounds
            the launch site.
        Environment.lonArray : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. 2x2 matrix for each pressure level of longitudes
            corresponding to the vertices of the grid cell which surrounds
            the launch site.
        Environment.lonIndex : int
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. Index to a grid longitude which
            is just over the launch site longitude, while lonIndex - 1
            points to a grid longitude which is just under the launch
            site longitude.
        Environment.latIndex : int
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. Index to a grid latitude which
            is just over the launch site latitude, while lonIndex - 1
            points to a grid latitude which is just under the launch
            site latitude.
        Environment.geopotentials : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. 2x2 matrix for each pressure level of geopotential heights
            corresponding to the vertices of the grid cell which surrounds
            the launch site.
        Environment.windUs : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. 2x2 matrix for each pressure level of wind U (east) component
            corresponding to the vertices of the grid cell which surrounds
            the launch site.
        Environment.windVs : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. 2x2 matrix for each pressure level of wind V (north) component
            corresponding to the vertices of the grid cell which surrounds
            the launch site.
        Environment.levels : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. List of pressure levels available
            in the file.
        Environment.temperatures : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. 2x2 matrix for each pressure level of temperatures
            corresponding to the vertices of the grid cell which surrounds
            the launch site.
        Environment.timeArray : array
            Defined if netCDF or OPeNDAP file is used, for Forecasts,
            Reanalysis and Ensembles. Array of dates available in the
            file.
        Environment.height : array
           Defined if netCDF or OPeNDAP file is used, for Forecasts,
           Reanalysis and Ensembles. List of geometric height
           corresponding to launch site location.
        
        Atmospheric Model Ensemble Specific Data
        Environment.levelEnsemble : array
            Only defined when using Ensembles.
        Environment.heightEnsemble : array
            Only defined when using Ensembles.
        Environment.temperatureEnsemble : array
            Only defined when using Ensembles.
        Environment.windUEnsemble : array
            Only defined when using Ensembles.
        Environment.windVEnsemble : array
            Only defined when using Ensembles.
        Environment.windHeadingEnsemble : arrray
            Only defined when using Ensembles.
        Environment.windDirectionEnsemble : array
            Only defined when using Ensembles.
        Environment.windSpeedEnsemble : array
            Only defined when using Ensembles.
        Environment.numEnsembleMembers : int
            Number of ensemble members. Only defined when using Ensembles.
        Environment.ensembleMember : int
            Current selected ensemble member. Only defined when using Ensembles.
        """

    def __init__(
        self,
        railLength,
        gravity=9.80665,
        date=None,
        latitude=None,
        longitude=None,
        elevation=0,
    ):
        """Initialize Environment class, saving launch rail length,
        launch date, location coordinates and elevation. Note that
        by default the standard atmosphere is loaded until another
        atmospheric model is used. See Environment.setAtmosphericModel
        for details.

        Parameters
        ----------
        railLength : scalar
            Length in which the rocket will be attached to the rail, only
            moving along a fixed direction, that is, the line parallel to the
            rail.
        gravity : scalar, optional
            Surface gravitational acceleration. Positive values point the
            acceleration down. Default value is 9.80665.
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

        Returns
        -------
        None
        """
        # Save launch rail ength
        self.rL = railLength

        # Save gravity value
        self.g = gravity

        # Initialize constants
        self.earthRadius = 6.3781e6
        self.airGasConstant = 287.05287  # in J/K/Kg

        # Initialize atmosphere
        self.setAtmosphericModel("StandardAtmosphere")

        # Save date
        if date != None:
            self.setDate(date)
        else:
            self.date = None

        # Save latitude and longitude
        if latitude != None and longitude != None:
            self.setLocation(latitude, longitude)
        else:
            self.lat, self.lon = None, None

        # Save elevation
        self.setElevation(elevation)

        return None

    def setDate(self, date):
        """Set date and time of launch and update weather conditions if
        date dependent atmospheric model is used.
        
        Parameters
        ----------
        date : Date
            Date object specifying launch date and time.
        
        Return
        ------
        None
        """
        # Store date
        self.date = datetime(*date)

        # Update atmospheric conditions if atmosphere type is Forecast,
        # Reanalysis or Ensemble
        if self.atmosphericModelType in ["Forecast", "Reanalysis", "Ensemble"]:
            self.setAtmosphericModel(
                self.atmosphericModelFile, self.atmosphericModelDict
            )

        return None

    def setLocation(self, latitude, longitude):
        """Set latitude and longitude of launch and update atmospheric
        conditions if location dependent model is being used.
        
        Parameters
        ----------
        latitude : float
            Latitude of launch site. May range from -90 to 90
            degrees.
        longitude : float
            Longitude of launch site. Either from 0 to 360 degrees
            or from -180 to 180 degrees.
        
        Return
        ------
        None
        """
        # Store latitude and longitude
        self.lat = latitude
        self.lon = longitude

        # Update atmospheric conditions if atmosphere type is Forecast,
        # Reanalysis or Ensemble
        if self.atmosphericModelType in ["Forecast", "Reanalysis", "Ensemble"]:
            self.setAtmosphericModel(
                self.atmosphericModelFile, self.atmosphericModelDict
            )

        # Return None

    def setElevation(self, elevation="Open-Elevation"):
        """Set elevation of launch site given user input or using the
        Open-Elevation API.
        
        Parameters
        ----------
        elevation : float, string, optional
            Elevation of launch site measured as height above sea
            level in meters.
            Alternatively, can be set as 'Open-Elevation' which uses
            the Open-Elevation API to find elevation data. For this
            option, latitude and longitude must have already been
            specified. See Environment.setLocation for more details.
        
        Return
        ------
        None
        """
        if elevation != "Open-Elevation":
            self.elevation = elevation
        elif self.lat != None and self.lon != None:
            try:
                print("Fetching elevation from open-elevation.com...")
                requestURL = "https://api.open-elevation.com/api/v1/lookup?locations={:f},{:f}".format(
                    self.lat, self.lon
                )
                response = requests.get(requestURL)
                results = response.json()["results"]
                self.elevation = results[0]["elevation"]
                print("Elevation received: ", self.elevation)
            except:
                raise RuntimeError("Unabel to reach Open-Elevation API servers.")
        else:
            raise ValueError(
                "Latitude and longitude must be set to use"
                " Open-Elevation API. See Environment.setLocation."
            )

    def setAtmosphericModel(
        self,
        type,
        file=None,
        dictionary=None,
        pressure=None,
        temperature=None,
        wind_u=0,
        wind_v=0,
    ):
        """ Defines an atmospheric model for the Environment.
        Supported functionality includes using data from the
        International Standard Atmosphere, importing data from
        weather reanalysis, forecasts and ensemble forecasts,
        importing data from upper air soundings and inputing
        data as costum functions, arrays or csv files.

        Parameters
        ----------
        type : string 
            One of the following options:
            - 'StandardAtmosphere': sets pressure and temperature
            profiles corresponding to the International Standard
            Atmosphere defined by ISO 2533 and ranging from -2 km
            to 80 km of altitude above sea level. Note that the wind
            profiles are set to zero when this type is chosen.

            - 'WyomingSounding': sets pressure, temperature, wind-u
            and wind-v profiles and surface elevation obtained from
            an upper air sounding given by the file parameter through
            an URL. This URL should point to a data webpage given by
            selecting plot type as text: list, a station and a time at
            http://weather.uwyo.edu/upperair/sounding.html.
            An example of a valid link would be:
            http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0200&TO=0200&STNM=82599
            
            - 'NOAARucSounding': sets pressure, temperature, wind-u
            and wind-v profiles and surface elevation obtained from
            an upper air sounding given by the file parameter through
            an URL. This URL should point to a data webpage obtained
            through NOAA's Ruc Sounding servers, which can be accessed
            in https://rucsoundings.noaa.gov/. Selecting ROABs as the
            initial data source, specifying the station through it's
            WMO-ID and opting for the ASCII (GSD format) button, the
            following example URL opens up: https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest
            Any ASCII GSD format page from this server can be read,
            so information from virtual soundings such as GFS and NAM
            can also be imported.

            - 'Forecast': sets pressure, temperature, wind-u and wind-v
            profiles and surface elevation obtained from a weather
            forecast file in netCDF format or from an OPeNDAP URL, both
            given through the file parameter. When this type
            is chosen, the date and location of the launch
            should already have been set through the date and 
            location parameters when initializing the Environment.
            The netCDF and OPeNDAP datasets must contain at least 
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
            
            - 'Reanalysis': sets pressure, temperature, wind-u and wind-v
            profiles and surface elevation obtained from a weather
            forecast file in netCDF format or from an OPeNDAP URL, both
            given through the file parameter. When this type
            is chosen, the date and location of the launch
            should already have been set through the date and 
            location parameters when initializing the Environment.
            The netCDF and OPeNDAP datasets must contain at least 
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
            
            - 'Ensemble': sets pressure, temperature, wind-u and wind-v
            profiles and surface elevation obtained from a weather
            forecast file in netCDF format or from an OPeNDAP URL, both
            given through the file parameter. When this type
            is chosen, the date and location of the launch
            should already have been set through the date and 
            location parameters when initializing the Environment.
            The netCDF and OPeNDAP datasets must contain at least 
            geopotential height or geopotential, temperature, 
            wind-u and wind-v profiles as a function of pressure
            levels. If surface geopotential or geopotential height
            is given, elevation is also set. Otherwise, elevation is not
            changed. Profiles are interpolated bi-linearly using supplied
            latitude and longitude. The date used is the nearest one
            to the date supplied. Furthermore, a dictionary must be
            supplied through the dictionary parameter in order for the
            dataset to be accurately read. Lastly, the dataset must use
            a rectangular grid sorted in either ascending or descending
            order of latitude and longitude. By default the first ensemble
            forecast is activated. To activate other ensemble forecasts
            see Environment.selectEnsembleMemberMember().
            
            - 'CustomAtmosphere': sets pressure, temperature, wind-u
            and wind-v profiles given though the pressure, temperature,
            wind-u and wind-v parameters of this method. If pressure
            or temperature is not given, it will default to the
            International Standard Atmosphere. If the wind components
            are not given, it will default to 0.
        file : string, optional
            String that must be given when type is either
            'WyomingSounding', 'Forecast', 'Reanalysis' or 'Ensemble'.
            It specifies the location of the data given, either through
            a local file adress or a URL.
            If type is 'Forecast', this parameter can also be either
            'GFS', 'FV3', 'RAP' or 'NAM' for latest of these forecasts.
            References: GFS: Global - 0.25deg resolution - Updates every 6 hours, forecast for 81 points spaced by 3 hours
                        FV3: Global - 0.25deg resolution - Updates every 6 hours, forecast for 129 points spaced by 3 hours
                        RAP: Regional USA - 0.19deg resolution - Updates hourly, forecast for 40 points spaced hourly
                        NAM: Regional CONUS Nest - 5 km resolution - Updates every 6 hours, forecast for 21 points spaced by 3 hours
            If type is 'Ensemble', this parameter can also be either
            'GEFS', or 'CMC' for the latest of these ensembles.
            Refrences: GEFS: Global, bias-corrected, 0.5deg resolution, 21 forecast members, Updates every 6 hours, forecast for 65 points spaced by 4 hours
                       CMC: Global, 0.5deg resolution, 21 forecast members, Updates every 12 hours, forecast for 65 points spaced by 4 hours
        dictionary : dictionary, string, optional
            Dictionary that must be given when type is either
            'Forecast', 'Reanalysis' or 'Ensemble'.
            It specifies the dictionary to be used when reading netCDF
            and OPeNDAP files, allowing the correct retrieval of data.
            Acceptable values include 'ECMWF', 'NOAA' and 'UCAR' for
            default dicitonaries which can generally be used to read
            datasets from these institutes.
            Alternatively, a dictionary structure can also be given,
            specifying the short names used for time, latitude, longitude,
            pressure levels, temperature profile, geopotential or
            geopotential height profile, wind-u and wind-v profiles in
            the dataset given in the file parameter. Additionally,
            ensemble dictionaries must have the ensemble as well.
            An example is the following dicitonary, used for 'NOAA':
                                  {'time': 'time',
                               'latitude': 'lat',
                              'longitude': 'lon', 
                                  'level': 'lev',
                               'ensemble': 'ens',
                            'temperature': 'tmpprs',
            'surface_geopotential_height': 'hgtsfc',
                    'geopotential_height': 'hgtprs',
                           'geopotential': None,
                                 'u_wind': 'ugrdprs',
                                 'v_wind': 'vgrdprs'}
        pressure : float, string, array, callable, optional
            This defines the atmospheric pressure profile.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            than the the Standard Atmosphere pressure will be used.
            If a float is given, it will define a constant pressure
            profile. The float should be in units of Pa.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the pressure in Pa.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, pressure in Pa).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding pressure in Pa.
        temperature : float, string, array, callable, optional
            This defines the atmospheric temperature profile.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            than the the Standard Atmosphere temperature will be used.
            If a float is given, it will define a constant temperature
            profile. The float should be in units of K.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the temperature in K.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, temperature in K).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding temperature in K.
        wind_u : float, string, array, callable, optional
            This defines the atmospheric wind-u profile, corresponding
            the the magnitude of the wind speed heading East.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            it will be assumed constant and 0.
            If a float is given, it will define a constant wind-u
            profile. The float should be in units of m/s.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the wind-u in m/s.
            If an array is given, it is expected to be an array of
            coordinates (height in meters, wind-u in m/s).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding wind-u in m/s.
        wind_v : float, string, array, callable, optional
            This defines the atmospheric wind-v profile, corresponding
            the the magnitude of the wind speed heading North.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            it will be assumed constant and 0.
            If a float is given, it will define a constant wind-v
            profile. The float should be in units of m/s.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the wind-v in m/s.
            If an array is given, it is expected to be an array of
            coordinates (height in meters, wind-v in m/s).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding wind-v in m/s.
        
        Return
        ------
        None
        """
        # Save atmospheric model type
        self.atmosphericModelType = type

        # Handle each case
        if type == "StandardAtmosphere":
            self.processStandardAtmosphere()
        elif type == "WyomingSounding":
            self.processWyomingSounding(file)
            # Save file
            self.atmosphericModelFile = file
        elif type == "NOAARucSounding":
            self.processNOAARUCSounding(file)
            # Save file
            self.atmosphericModelFile = file
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
                timeAttempt = datetime.utcnow()
                success = False
                attemptCount = 0
                while not success and attemptCount < 10:
                    timeAttempt -= timedelta(hours=6 * attemptCount)
                    file = "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{:04d}{:02d}{:02d}/gfs_0p25_{:02d}z".format(
                        timeAttempt.year,
                        timeAttempt.month,
                        timeAttempt.day,
                        6 * (timeAttempt.hour // 6),
                    )
                    try:
                        self.processForecastReanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attemptCount += 1
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
                timeAttempt = datetime.utcnow()
                success = False
                attemptCount = 0
                while not success and attemptCount < 10:
                    timeAttempt -= timedelta(hours=6 * attemptCount)
                    file = "https://nomads.ncep.noaa.gov/dods/gfs_0p25_parafv3/gfs{:04d}{:02d}{:02d}/gfs_0p25_parafv3_{:02d}z".format(
                        timeAttempt.year,
                        timeAttempt.month,
                        timeAttempt.day,
                        6 * (timeAttempt.hour // 6),
                    )
                    try:
                        self.processForecastReanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attemptCount += 1
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
                timeAttempt = datetime.utcnow()
                success = False
                attemptCount = 0
                while not success and attemptCount < 10:
                    timeAttempt -= timedelta(hours=6 * attemptCount)
                    file = "https://nomads.ncep.noaa.gov/dods/nam/nam{:04d}{:02d}{:02d}/nam_conusnest_{:02d}z".format(
                        timeAttempt.year,
                        timeAttempt.month,
                        timeAttempt.day,
                        6 * (timeAttempt.hour // 6),
                    )
                    try:
                        self.processForecastReanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attemptCount += 1
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
                timeAttempt = datetime.utcnow()
                success = False
                attemptCount = 0
                while not success and attemptCount < 10:
                    timeAttempt -= timedelta(hours=1 * attemptCount)
                    file = "https://nomads.ncep.noaa.gov/dods/rap/rap{:04d}{:02d}{:02d}/rap_{:02d}z".format(
                        timeAttempt.year,
                        timeAttempt.month,
                        timeAttempt.day,
                        timeAttempt.hour,
                    )
                    try:
                        self.processForecastReanalysis(file, dictionary)
                        success = True
                    except OSError:
                        attemptCount += 1
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
                self.processForecastReanalysis(file, dictionary)
            # Save dicitonary and file
            self.atmosphericModelFile = file
            self.atmosphericModelDict = dictionary
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
                timeAttempt = datetime.utcnow()
                success = False
                attemptCount = 0
                while not success and attemptCount < 10:
                    timeAttempt -= timedelta(hours=6 * attemptCount)
                    file = "https://nomads.ncep.noaa.gov/dods/gens_bc/gens{:04d}{:02d}{:02d}/gep_all_{:02d}z".format(
                        timeAttempt.year,
                        timeAttempt.month,
                        timeAttempt.day,
                        6 * (timeAttempt.hour // 6),
                    )
                    try:
                        self.processEnsemble(file, dictionary)
                        success = True
                    except OSError:
                        attemptCount += 1
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
                timeAttempt = datetime.utcnow()
                success = False
                attemptCount = 0
                while not success and attemptCount < 10:
                    timeAttempt -= timedelta(hours=12 * attemptCount)
                    file = "https://nomads.ncep.noaa.gov/dods/cmcens/cmcens{:04d}{:02d}{:02d}/cmcens_all_{:02d}z".format(
                        timeAttempt.year,
                        timeAttempt.month,
                        timeAttempt.day,
                        12 * (timeAttempt.hour // 12),
                    )
                    try:
                        self.processEnsemble(file, dictionary)
                        success = True
                    except OSError:
                        attemptCount += 1
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
                self.processEnsemble(file, dictionary)
            # Save dicitonary and file
            self.atmosphericModelFile = file
            self.atmosphericModelDict = dictionary
        elif type == "CostumAtmosphere":
            self.processCostumAtmosphere(pressure, temperature, wind_u, wind_v)
        else:
            raise ValueError("Unknown model type.")

        # Calculate air density
        self.calculateDensityProfile()

        # Calculate speed of sound
        self.calculateSpeedOfSoundProfile()

        # Update dynamic viscosity
        self.calculateDynamicViscosity()

        return None

    def processStandardAtmosphere(self):
        """ Sets pressure and temperature profiles corresponding to the
        International Standard Atmosphere defined by ISO 2533 and
        ranging from -2 km to 80 km of altitude above sea level. Note
        that the wind profiles are set to zero.

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        # Load international standard atmosphere
        self.loadInternationalStandardAtmosphere()

        # Save temperature, pressure and wind profiles
        self.pressure = self.pressureISA
        self.temperature = self.temperatureISA
        self.windDirection = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.windHeading = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.windSpeed = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.windVelocityX = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.windVelocityY = Function(
            0,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Set maximum expected height
        self.maxExpectedHeight = 80000

        return None

    def processCostumAtmosphere(
        self, pressure=None, temperature=None, wind_u=0, wind_v=0
    ):
        """ Import pressure, temperature and wind profile given by user.

        Parameters
        ----------
        pressure : float, string, array, callable, optional
            This defines the atmospheric pressure profile.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            than the the Standard Atmosphere pressure will be used.
            If a float is given, it will define a constant pressure
            profile. The float should be in units of Pa.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the pressure in Pa.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, pressure in Pa).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding pressure in Pa.
        temperature : float, string, array, callable, optional
            This defines the atmospheric temperature profile.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            than the the Standard Atmosphere temperature will be used.
            If a float is given, it will define a constant temperature
            profile. The float should be in units of K.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the temperature in K.
            If an array is given, it is expected to be a list or array
            of coordinates (height in meters, temperature in K).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding temperature in K.
        wind_u : float, string, array, callable, optional
            This defines the atmospheric wind-u profile, corresponding
            the the magnitude of the wind speed heading East.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            it will be assumed constant and 0.
            If a float is given, it will define a constant wind-u
            profile. The float should be in units of m/s.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the wind-u in m/s.
            If an array is given, it is expected to be an array of
            coordinates (height in meters, wind-u in m/s).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding wind-u in m/s.
        wind_v : float, string, array, callable, optional
            This defines the atmospheric wind-v profile, corresponding
            the the magnitude of the wind speed heading North.
            Should be given if the type parameter is 'CustomAtmosphere'. If not,
            it will be assumed constant and 0.
            If a float is given, it will define a constant wind-v
            profile. The float should be in units of m/s.
            If a string is given, it should point to a .CSV file
            containing at most one header line and two columns of data.
            The first column must be the geometric height above sea level in
            meters while the second column must be the wind-v in m/s.
            If an array is given, it is expected to be an array of
            coordinates (height in meters, wind-v in m/s).
            Finally, a callable or function is also acepted. The
            function should take one argument, the height above sea
            level in meters and return a corresponding wind-v in m/s.
        
        Return
        ------
        None
        """
        # Intialize a estimage of maximum expected atmospheric model height
        maxExpectedHeight = 1000

        # Save pressure profile
        if pressure is None:
            # Use standard atmosphere
            self.pressure = self.pressureISA
        else:
            # Use costum input
            self.pressure = Function(
                pressure,
                inputs="Height Above Sea Level (m)",
                outputs="Pressure (Pa)",
                interpolation="linear",
            )
            # Check maximum height of costum pressure input
            if not callable(self.pressure.source):
                maxExpectedHeight = max(self.pressure[-1, 0], maxExpectedHeight)

        # Save temperature profile
        if temperature is None:
            # Use standard atmosphere
            self.temperature = self.temperatureISA
        else:
            self.temperature = Function(
                temperature,
                inputs="Height Above Sea Level (m)",
                outputs="Temperature (K)",
                interpolation="linear",
            )
            # Check maximum height of costum temperature input
            if not callable(self.temperature.source):
                maxExpectedHeight = max(self.temperature[-1, 0], maxExpectedHeight)

        # Save wind profile
        self.windVelocityX = Function(
            wind_u,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.windVelocityY = Function(
            wind_v,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )
        # Check maximum height of costum wind input
        if not callable(self.windVelocityX.source):
            maxExpectedHeight = max(self.windVelocityX[-1, 0], maxExpectedHeight)
        if not callable(self.windVelocityY.source):
            maxExpectedHeight = max(self.windVelocityY[-1, 0], maxExpectedHeight)

        # Compute wind profile direction and heading
        windHeading = (
            lambda h: np.arctan2(self.windVelocityX(h), self.windVelocityY(h))
            * (180 / np.pi)
            % 360
        )
        self.windHeading = Function(
            windHeading,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )

        windDirection = lambda h: (windHeading(h) - 180) % 360
        self.windDirection = Function(
            windDirection,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )

        windSpeed = lambda h: np.sqrt(
            self.windVelocityX(h) ** 2 + self.windVelocityY(h) ** 2
        )
        self.windSpeed = Function(
            windSpeed,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.maxExpectedHeight = maxExpectedHeight

        return None

    def processWyomingSounding(self, file):
        """ Import and process the upper air sounding data from Wyoming
        Upper Air Soundings database given by the url in file. Sets
        pressure, temperature, wind-u, wind-v profiles and surface elevation.

        Parameters
        ----------
        file : string
            URL of an upper air sounding data output from Wyoming
            Upper Air Soundigs database.
            Example:
            http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0200&TO=0200&STNM=82599
            More can be found at:
            http://weather.uwyo.edu/upperair/sounding.html.
            
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

        # Process Wyoming Souding by finding data table and station info
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
        R = self.earthRadius
        data_array[:, 1] = R * data_array[:, 1] / (R - data_array[:, 1])

        # Save atmospheric data
        self.windDirection = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.windHeading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.windSpeed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.windVelocityX = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.windVelocityY = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Retrieve station elevation from station info
        station_elevation_text = station_info.split("\n")[6]

        # Convert station elevation text into float value
        self.elevation = float(
            re.findall("[0-9]+\.[0-9]+|[0-9]+", station_elevation_text)[0]
        )

        # Save maximum expected height
        self.maxExpectedHeight = data_array[-1, 1]

        return None

    def processNOAARUCSounding(self, file):
        """ Import and process the upper air sounding data from NOAA
        Ruc Soundings database (https://rucsoundings.noaa.gov/) given as
        ASCII GSD format pages passed by its url to the file parameter. Sets
        pressure, temperature, wind-u, wind-v profiles and surface elevation.

        Parameters
        ----------
        file : string
            URL of an upper air sounding data output from NOAA Ruc Soundings
            in ASCII GSD format.
            Example:
            https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest
            More can be found at:
            https://rucsoundings.noaa.gov/.
            
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
        windSpeed_array = []
        windDirection_array = []
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
                        windDirection_array.append(columns[[0, 1]])
                        windSpeed_array.append(columns[[0, 2]])
        windSpeed_array = np.array(windSpeed_array)
        windDirection_array = np.array(windDirection_array)

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
        windSpeed_array[:, 1] = (
            windSpeed_array[:, 1] * 1.852 / 3.6
        )  # Converts Knots to m/s
        windHeading_array = windDirection_array[:, :] * 1
        windHeading_array[:, 1] = (
            windDirection_array[:, 1] + 180
        ) % 360  # Convert wind direction to wind heading
        windU = windSpeed_array[:, :] * 1
        windV = windSpeed_array[:, :] * 1
        windU[:, 1] = windSpeed_array[:, 1] * np.sin(
            windHeading_array[:, 1] * np.pi / 180
        )
        windV[:, 1] = windSpeed_array[:, 1] * np.cos(
            windHeading_array[:, 1] * np.pi / 180
        )

        # Save wind data
        self.windDirection = Function(
            windDirection_array,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.windHeading = Function(
            windHeading_array,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.windSpeed = Function(
            windSpeed_array,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.windVelocityX = Function(
            windU,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.windVelocityY = Function(
            windV,
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.maxExpectedHeight = pressure_array[-1, 0]

    def processForecastReanalysis(self, file, dictionary):
        """ Import and process atmospheric data from weather forecasts
        and reanalysis given as netCDF or OPeNDAP files.
        Sets pressure, temperature, wind-u and wind-v
        profiles and surface elevation obtained from a weather
        file in netCDF format or from an OPeNDAP URL, both
        given through the file parameter. The date and location of the launch
        should already have been set through the date and 
        location parameters when initializing the Environment.
        The netCDF and OPeNDAP datasets must contain at least 
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
            String containing path to local netCDF file or URL of an
            OPeNDAP file, such as NOAA's NOMAD or UCAR TRHEDDS server.
        dicitonary : dictionary
            Specifies the dictionary to be used when reading netCDF and
            OPeNDAP files, allowing for the correct retrieval of data.
            The dictionary structure should specify the short names
            used for time, latitude, longitude, pressure levels,
            temperature profile, geopotential or geopotential height
            profile, wind-u and wind-v profiles in the dataset given in
            the file parameter. An example is the following dicitonary,
            generally used to read OPeNDAP files from NOAA's NOMAD
            server:               {'time': 'time',
                               'latitude': 'lat',
                              'longitude': 'lon', 
                                  'level': 'lev',
                            'temperature': 'tmpprs',
            'surface_geopotential_height': 'hgtsfc',
                    'geopotential_height': 'hgtprs',
                           'geopotential': None,
                                 'u_wind': 'ugrdprs',
                                 'v_wind': 'vgrdprs'}
        
        Returns
        -------
        None
        """
        # Check if date, lat and lon are known
        if self.date is None:
            raise TypeError(
                "Please specify Date (array-like) when "
                "initializing this Environment. "
                "Alternatively, use the Environment.setDate"
                " method."
            )
        if self.lat is None:
            raise TypeError(
                "Please specify Location (lat, lon). when "
                "initializing this Environment. "
                "Alternatively, use the Environment.setLoc"
                "ation method."
            )

        # Read weather file
        weatherData = netCDF4.Dataset(file)

        # Get time, latitude and longitude data from file
        timeArray = weatherData.variables[dictionary["time"]]
        lonArray = weatherData.variables[dictionary["longitude"]][:].tolist()
        latArray = weatherData.variables[dictionary["latitude"]][:].tolist()

        # Find time index
        timeIndex = netCDF4.date2index(self.date, timeArray, select="nearest")
        # Convert times do dates and numbers
        inputTimeNum = netCDF4.date2num(self.date, timeArray.units)
        fileTimeNum = timeArray[timeIndex]
        fileTimeDate = netCDF4.num2date(timeArray[timeIndex], timeArray.units)
        # Check if time is inside range supplied by file
        if timeIndex == 0 and inputTimeNum < fileTimeNum:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which starts at {:}.".format(
                    fileTimeDate
                )
            )
        elif timeIndex == len(timeArray) - 1 and inputTimeNum > fileTimeNum:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which ends at {:}.".format(
                    fileTimeDate
                )
            )
        # Check if time is exactly equal to one in the file
        if inputTimeNum != fileTimeNum:
            warnings.warn(
                "Exact chosen launch time is not available in the provided file, using {:} UTC instead.".format(
                    fileTimeDate
                )
            )

        # Find longitude index
        # Determine if file uses -180 to 180 or 0 to 360
        if lonArray[0] < 0 or lonArray[-1] < 0:
            # Convert input to -180 - 180
            lon = self.lon if self.lon < 180 else -180 + self.lon % 180
        else:
            # Convert input to 0 - 360
            lon = self.lon % 360
        # Check if reversed or sorted
        if lonArray[0] < lonArray[-1]:
            # Deal with sorted lonArray
            lonIndex = bisect.bisect(lonArray, lon)
        else:
            # Deal with reversed lonArray
            lonArray.reverse()
            lonIndex = len(lonArray) - bisect.bisect_left(lonArray, lon)
            lonArray.reverse()
        # Take care of longitude value equal to maximum longitude in the grid
        if lonIndex == len(lonArray) and lonArray[lonIndex - 1] == lon:
            lonIndex = lonIndex - 1
        # Check if longitude value is inside the grid
        if lonIndex == 0 or lonIndex == len(lonArray):
            raise ValueError(
                "Longitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    lon, lonArray[0], lonArray[-1]
                )
            )

        # Find latitude index
        # Check if reversed or sorted
        if latArray[0] < latArray[-1]:
            # Deal with sorted latArray
            latIndex = bisect.bisect(latArray, self.lat)
        else:
            # Deal with reversed latArray
            latArray.reverse()
            latIndex = len(latArray) - bisect.bisect_left(latArray, self.lat)
            latArray.reverse()
        # Take care of latitude value equal to maximum longitude in the grid
        if latIndex == len(latArray) and latArray[latIndex - 1] == self.lat:
            latIndex = latIndex - 1
        # Check if latitude value is inside the grid
        if latIndex == 0 or latIndex == len(latArray):
            raise ValueError(
                "Latitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    self.lat, latArray[0], latArray[-1]
                )
            )

        # Get pressure level data from file
        try:
            levels = (
                100 * weatherData.variables[dictionary["level"]][:]
            )  # Convert mbar to Pa
        except:
            raise ValueError(
                "Unable to read pressure levels from file. Check file and dictionary."
            )

        # Get geopotential data from file
        try:
            geopotentials = weatherData.variables[dictionary["geopotential_height"]][
                timeIndex, :, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)
            ]
        except:
            try:
                geopotentials = (
                    weatherData.variables[dictionary["geopotential"]][
                        timeIndex, :, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)
                    ]
                    / self.g
                )
            except:
                raise ValueError(
                    "Unable to read geopontential height"
                    " nor geopotential from file. At least"
                    " one of them is necessary. Check "
                    " file and dictionary."
                )

        # Get temperature from file
        try:
            temperatures = weatherData.variables[dictionary["temperature"]][
                timeIndex, :, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)
            ]
        except:
            raise ValueError(
                "Unable to read temperature from file. Check file and dictionary."
            )

        # Get wind data from file
        try:
            windUs = weatherData.variables[dictionary["u_wind"]][
                timeIndex, :, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)
            ]
        except:
            raise ValueError(
                "Unable to read wind-u component. Check file and dictionary."
            )
        try:
            windVs = weatherData.variables[dictionary["v_wind"]][
                timeIndex, :, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)
            ]
        except:
            raise ValueError(
                "Unable to read wind-v component. Check file and dictionary."
            )

        # Prepare for bilinear interpolation
        x, y = self.lat, lon
        x1, y1 = latArray[latIndex - 1], lonArray[lonIndex - 1]
        x2, y2 = latArray[latIndex], lonArray[lonIndex]

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
        f_x1_y1 = windUs[:, 0, 0]
        f_x1_y2 = windUs[:, 0, 1]
        f_x2_y1 = windUs[:, 1, 0]
        f_x2_y2 = windUs[:, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        windU = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind v in lat, lon
        f_x1_y1 = windVs[:, 0, 0]
        f_x1_y2 = windVs[:, 0, 1]
        f_x2_y1 = windVs[:, 1, 0]
        f_x2_y2 = windVs[:, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        windV = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind speed, heading and direction
        windSpeed = np.sqrt(windU ** 2 + windV ** 2)
        windHeading = np.arctan2(windU, windV) * (180 / np.pi) % 360
        windDirection = (windHeading - 180) % 360

        # Convert geopotential height to geometric height
        R = self.earthRadius
        height = R * height / (R - height)

        # Combine all data into big array
        data_array = np.ma.column_stack(
            [
                levels,
                height,
                temperature,
                windU,
                windV,
                windHeading,
                windDirection,
                windSpeed,
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
        self.windDirection = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.windHeading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.windSpeed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.windVelocityX = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.windVelocityY = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.maxExpectedHeight = max(height[0], height[-1])

        # Get elevation data from file
        if dictionary["surface_geopotential_height"] is not None:
            try:
                elevations = weatherData.variables[
                    dictionary["surface_geopotential_height"]
                ][timeIndex, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)]
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
        self.atmosphericModelInitDate = netCDF4.num2date(timeArray[0], timeArray.units)
        self.atmosphericModelEndDate = netCDF4.num2date(timeArray[-1], timeArray.units)
        self.atmosphericModelInterval = netCDF4.num2date(
            (timeArray[-1] - timeArray[0]) / (len(timeArray) - 1), timeArray.units
        ).hour
        self.atmosphericModelInitLat = latArray[0]
        self.atmosphericModelEndLat = latArray[-1]
        self.atmosphericModelInitLon = lonArray[0]
        self.atmosphericModelEndLon = lonArray[-1]

        # Save debugging data
        self.latArray = latArray
        self.lonArray = lonArray
        self.lonIndex = lonIndex
        self.latIndex = latIndex
        self.geopotentials = geopotentials
        self.windUs = windUs
        self.windVs = windVs
        self.levels = levels
        self.temperatures = temperatures
        self.timeArray = timeArray
        self.height = height

        # Close weather data
        weatherData.close()

        return None

    def processEnsemble(self, file, dictionary):
        """ Import and process atmospheric data from weather ensembles
        given as netCDF or OPeNDAP files.
        Sets pressure, temperature, wind-u and wind-v
        profiles and surface elevation obtained from a weather
        ensemble file in netCDF format or from an OPeNDAP URL, both
        given through the file parameter. The date and location of the launch
        should already have been set through the date and 
        location parameters when initializing the Environment.
        The netCDF and OPeNDAP datasets must contain at least 
        geopotential height or geopotential, temperature, 
        wind-u and wind-v profiles as a function of pressure
        levels. If surface geopotential or geopotential height
        is given, elevation is also set. Otherwise, elevation is not
        changed. Profiles are interpolated bi-linearly using supplied
        latitude and longitude. The date used is the nearest one
        to the date supplied. Furthermore, a dictionary must be
        supplied through the dictionary parameter in order for the
        dataset to be accurately read. Lastly, the dataset must use
        a rectangular grid sorted in either ascending or descending
        order of latitude and longitude. By default the first ensemble
        forecast is activated. To activate other ensemble forecasts
        see Environment.selectEnsembleMemberMember().

        Parameters
        ----------
        file : string
            String containing path to local netCDF file or URL of an
            OPeNDAP file, such as NOAA's NOMAD or UCAR TRHEDDS server.
        dicitonary : dictionary
            Specifies the dictionary to be used when reading netCDF and
            OPeNDAP files, allowing for the correct retrieval of data.
            The dictionary structure should specify the short names
            used for time, latitude, longitude, pressure levels,
            temperature profile, geopotential or geopotential height
            profile, wind-u and wind-v profiles in the dataset given in
            the file parameter. An example is the following dicitonary,
            generally used to read OPeNDAP files from NOAA's NOMAD
            server:               {'time': 'time',
                               'latitude': 'lat',
                              'longitude': 'lon', 
                                  'level': 'lev',
                               'ensemble': 'ens',
            'surface_geopotential_height': 'hgtsfc',
                    'geopotential_height': 'hgtprs',
                           'geopotential': None,
                                 'u_wind': 'ugrdprs',
                                 'v_wind': 'vgrdprs'}
        
        Returns
        -------
        None
        """
        # Check if date, lat and lon are known
        if self.date is None:
            raise TypeError(
                "Please specify Date (array-like) when "
                "initializing this Environment. "
                "Alternatively, use the Environment.setDate"
                " method."
            )
        if self.lat is None:
            raise TypeError(
                "Please specify Location (lat, lon). when "
                "initializing this Environment. "
                "Alternatively, use the Environment.setLoc"
                "ation method."
            )

        # Read weather file
        weatherData = netCDF4.Dataset(file)

        # Get time, latitude and longitude data from file
        timeArray = weatherData.variables[dictionary["time"]]
        lonArray = weatherData.variables[dictionary["longitude"]][:].tolist()
        latArray = weatherData.variables[dictionary["latitude"]][:].tolist()

        # Find time index
        timeIndex = netCDF4.date2index(self.date, timeArray, select="nearest")
        # Convert times do dates and numbers
        inputTimeNum = netCDF4.date2num(self.date, timeArray.units)
        fileTimeNum = timeArray[timeIndex]
        fileTimeDate = netCDF4.num2date(timeArray[timeIndex], timeArray.units)
        # Check if time is inside range supplied by file
        if timeIndex == 0 and inputTimeNum < fileTimeNum:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which starts at {:}.".format(
                    fileTimeDate
                )
            )
        elif timeIndex == len(timeArray) - 1 and inputTimeNum > fileTimeNum:
            raise ValueError(
                "Chosen launch time is not available in the provided file, which ends at {:}.".format(
                    fileTimeDate
                )
            )
        # Check if time is exactly equal to one in the file
        if inputTimeNum != fileTimeNum:
            warnings.warn(
                "Exact chosen launch time is not available in the provided file, using {:} UTC instead.".format(
                    fileTimeDate
                )
            )

        # Find longitude index
        # Determine if file uses -180 to 180 or 0 to 360
        if lonArray[0] < 0 or lonArray[-1] < 0:
            # Convert input to -180 - 180
            lon = self.lon if self.lon < 180 else -180 + self.lon % 180
        else:
            # Convert input to 0 - 360
            lon = self.lon % 360
        # Check if reversed or sorted
        if lonArray[0] < lonArray[-1]:
            # Deal with sorted lonArray
            lonIndex = bisect.bisect(lonArray, lon)
        else:
            # Deal with reversed lonArray
            lonArray.reverse()
            lonIndex = len(lonArray) - bisect.bisect_left(lonArray, lon)
            lonArray.reverse()
        # Take care of longitude value equal to maximum longitude in the grid
        if lonIndex == len(lonArray) and lonArray[lonIndex - 1] == lon:
            lonIndex = lonIndex - 1
        # Check if longitude value is inside the grid
        if lonIndex == 0 or lonIndex == len(lonArray):
            raise ValueError(
                "Longitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    lon, lonArray[0], lonArray[-1]
                )
            )

        # Find latitude index
        # Check if reversed or sorted
        if latArray[0] < latArray[-1]:
            # Deal with sorted latArray
            latIndex = bisect.bisect(latArray, self.lat)
        else:
            # Deal with reversed latArray
            latArray.reverse()
            latIndex = len(latArray) - bisect.bisect_left(latArray, self.lat)
            latArray.reverse()
        # Take care of latitude value equal to maximum longitude in the grid
        if latIndex == len(latArray) and latArray[latIndex - 1] == self.lat:
            latIndex = latIndex - 1
        # Check if latitude value is inside the grid
        if latIndex == 0 or latIndex == len(latArray):
            raise ValueError(
                "Latitude {:f} not inside region covered by file, which is from {:f} to {:f}.".format(
                    self.lat, latArray[0], latArray[-1]
                )
            )

        # Get ensemble data from file
        try:
            numMembers = len(weatherData.variables[dictionary["ensemble"]][:])
        except:
            raise ValueError(
                "Unable to read ensemble data from file. Check file and dictionary."
            )

        # Get pressure level data from file
        try:
            levels = (
                100 * weatherData.variables[dictionary["level"]][:]
            )  # Convert mbar to Pa
        except:
            raise ValueError(
                "Unable to read pressure levels from file. Check file and dictionary."
            )

        ##
        inverseDictionary = {v: k for k, v in dictionary.items()}
        paramDictionary = {
            "time": timeIndex,
            "ensemble": range(numMembers),
            "level": range(len(levels)),
            "latitude": (latIndex - 1, latIndex),
            "longitude": (lonIndex - 1, lonIndex),
        }
        ##

        # Get geopotential data from file
        try:
            dimensions = weatherData.variables[
                dictionary["geopotential_height"]
            ].dimensions[:]
            params = tuple(
                [paramDictionary[inverseDictionary[dim]] for dim in dimensions]
            )
            geopotentials = weatherData.variables[dictionary["geopotential_height"]][
                params
            ]
        except:
            try:
                dimensions = weatherData.variables[
                    dictionary["geopotential"]
                ].dimensions[:]
                params = tuple(
                    [paramDictionary[inverseDictionary[dim]] for dim in dimensions]
                )
                geopotentials = (
                    weatherData.variables[dictionary["geopotential"]][params] / self.g
                )
            except:
                raise ValueError(
                    "Unable to read geopontential height"
                    " nor geopotential from file. At least"
                    " one of them is necessary. Check "
                    " file and dictionary."
                )

        # Get temperature from file
        try:
            temperatures = weatherData.variables[dictionary["temperature"]][params]
        except:
            raise ValueError(
                "Unable to read temperature from file. Check file and dictionary."
            )

        # Get wind data from file
        try:
            windUs = weatherData.variables[dictionary["u_wind"]][params]
        except:
            raise ValueError(
                "Unable to read wind-u component. Check file and dictionary."
            )
        try:
            windVs = weatherData.variables[dictionary["v_wind"]][params]
        except:
            raise ValueError(
                "Unable to read wind-v component. Check file and dictionary."
            )

        # Prepare for bilinear interpolation
        x, y = self.lat, lon
        x1, y1 = latArray[latIndex - 1], lonArray[lonIndex - 1]
        x2, y2 = latArray[latIndex], lonArray[lonIndex]

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
        f_x1_y1 = windUs[:, :, 0, 0]
        f_x1_y2 = windUs[:, :, 0, 1]
        f_x2_y1 = windUs[:, :, 1, 0]
        f_x2_y2 = windUs[:, :, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        windU = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind v in lat, lon
        f_x1_y1 = windVs[:, :, 0, 0]
        f_x1_y2 = windVs[:, :, 0, 1]
        f_x2_y1 = windVs[:, :, 1, 0]
        f_x2_y2 = windVs[:, :, 1, 1]
        f_x_y1 = ((x2 - x) / (x2 - x1)) * f_x1_y1 + ((x - x1) / (x2 - x1)) * f_x2_y1
        f_x_y2 = ((x2 - x) / (x2 - x1)) * f_x1_y2 + ((x - x1) / (x2 - x1)) * f_x2_y2
        windV = ((y2 - y) / (y2 - y1)) * f_x_y1 + ((y - y1) / (y2 - y1)) * f_x_y2

        # Determine wind speed, heading and direction
        windSpeed = np.sqrt(windU ** 2 + windV ** 2)
        windHeading = np.arctan2(windU, windV) * (180 / np.pi) % 360
        windDirection = (windHeading - 180) % 360

        # Convert geopotential height to geometric height
        R = self.earthRadius
        height = R * height / (R - height)

        # Save enesemble data
        self.levelEnsemble = levels
        self.heightEnsemble = height
        self.temperatureEnsemble = temperature
        self.windUEnsemble = windU
        self.windVEnsemble = windV
        self.windHeadingEnsemble = windHeading
        self.windDirectionEnsemble = windDirection
        self.windSpeedEnsemble = windSpeed
        self.numEnsembleMembers = numMembers

        # Activate default ensemble
        self.selectEnsembleMember()

        # Get elevation data from file
        if dictionary["surface_geopotential_height"] is not None:
            try:
                elevations = weatherData.variables[
                    dictionary["surface_geopotential_height"]
                ][timeIndex, (latIndex - 1, latIndex), (lonIndex - 1, lonIndex)]
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
        self.atmosphericModelInitDate = netCDF4.num2date(timeArray[0], timeArray.units)
        self.atmosphericModelEndDate = netCDF4.num2date(timeArray[-1], timeArray.units)
        self.atmosphericModelInterval = netCDF4.num2date(
            (timeArray[-1] - timeArray[0]) / (len(timeArray) - 1), timeArray.units
        ).hour
        self.atmosphericModelInitLat = latArray[0]
        self.atmosphericModelEndLat = latArray[-1]
        self.atmosphericModelInitLon = lonArray[0]
        self.atmosphericModelEndLon = lonArray[-1]

        # Save debugging data
        self.latArray = latArray
        self.lonArray = lonArray
        self.lonIndex = lonIndex
        self.latIndex = latIndex
        self.geopotentials = geopotentials
        self.windUs = windUs
        self.windVs = windVs
        self.levels = levels
        self.temperatures = temperatures
        self.timeArray = timeArray
        self.height = height

        # Close weather data
        weatherData.close()

        return None

    def selectEnsembleMember(self, member=0):
        """ Activates ensemble member, meaning that all atmospheric
        variables read from the Environment instance will correspond
        to the desired ensemble member.

        Parameters
        ---------
        member : int
            Ensemble member to be activated. Starts from 0.

        Returns
        -------
        None
        """
        # Verify ensemble member
        if member >= self.numEnsembleMembers:
            raise ValueError(
                "Please choose member from 0 to {:d}".format(
                    self.numEnsembleMembers - 1
                )
            )

        # Read ensemble member
        levels = self.levelEnsemble[:]
        height = self.heightEnsemble[member, :]
        temperature = self.temperatureEnsemble[member, :]
        windU = self.windUEnsemble[member, :]
        windV = self.windVEnsemble[member, :]
        windHeading = self.windHeadingEnsemble[member, :]
        windDirection = self.windDirectionEnsemble[member, :]
        windSpeed = self.windSpeedEnsemble[member, :]

        # Combine all data into big array
        data_array = np.ma.column_stack(
            [
                levels,
                height,
                temperature,
                windU,
                windV,
                windHeading,
                windDirection,
                windSpeed,
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
        self.windDirection = Function(
            data_array[:, (1, 6)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Direction (Deg True)",
            interpolation="linear",
        )
        self.windHeading = Function(
            data_array[:, (1, 5)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Heading (Deg True)",
            interpolation="linear",
        )
        self.windSpeed = Function(
            data_array[:, (1, 7)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Speed (m/s)",
            interpolation="linear",
        )
        self.windVelocityX = Function(
            data_array[:, (1, 3)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity X (m/s)",
            interpolation="linear",
        )
        self.windVelocityY = Function(
            data_array[:, (1, 4)],
            inputs="Height Above Sea Level (m)",
            outputs="Wind Velocity Y (m/s)",
            interpolation="linear",
        )

        # Save maximum expected height
        self.maxExpectedHeight = max(height[0], height[-1])

        # Save ensemble member
        self.ensembleMember = member

        # Update air density
        self.calculateDensityProfile()

        # Update speed of sound
        self.calculateSpeedOfSoundProfile()

        # Update dynamic viscosity
        self.calculateDynamicViscosity()

        return None

    def loadInternationalStandardAtmosphere(self):
        """ Defines the pressure and temperature profile functions set
        by ISO 2533 for the International Standard atmosphere and saves
        them as self.pressureISA and self.temperatureISA.

        Parameters
        ---------
        None

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
        ER = self.earthRadius
        height = [ER * H / (ER - H) for H in geopotential_height]
        height = geopotential_height

        # Save international standard atmosphere temperature profile
        self.temperatureISA = Function(
            np.column_stack([height, temperature]),
            inputs="Height Above Sea Level (m)",
            outputs="Temperature (K)",
            interpolation="linear",
        )

        # Get gravity and R
        g = self.g
        R = self.airGasConstant

        # Create function to compute pressure profile
        def pressure_function(h):
            # Convert geometric to geopotential height
            H = ER * h / (ER + h)
            H = h

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

            # Compute presure
            if B != 0:
                P = Pb * (1 + (B / Tb) * (H - Hb)) ** (-g / (B * R))
            else:
                T = Tb + B * (H - Hb)
                P = Pb * np.exp(-(H - Hb) * (g / (R * T)))

            # Return answer
            return P

        # Save international standard atmosphere pressure profile
        self.pressureISA = Function(
            pressure_function,
            inputs="Height Above Sea Level (m)",
            outputs="Pressure (Pa)",
        )

        return None

    def calculateDensityProfile(self):
        """ Compute the density of the atmosphere as a function of
        heigth by using the formula rho = P/(RT). This function is
        automatically called whenever a new atmospheric model is set.

        Parameters
        ----------
        None

        Returns
        -------
        None       
        """
        # Retrieve pressure P, gas constant R and temperature T
        P = self.pressure
        R = self.airGasConstant
        T = self.temperature

        # Compute density using P/RT
        D = P / (R * T)

        # Set new output for the calculated density
        D.setOutputs("Air Density (kg/m³)")

        # Save calculated density
        self.density = D

        return None

    def calculateSpeedOfSoundProfile(self):
        """ Compute the speed of sound in the atmosphere as a function
        of heigth by using the formula a = sqrt(gamma*R*T). This
        function is automatically called whenever a new atmospheric
        model is set.

        Parameters
        ----------
        None

        Returns
        -------
        None       
        """
        # Retrieve gas constant R and temperature T
        R = self.airGasConstant
        T = self.temperature
        G = 1.4

        # Compute speed of sound using sqrt(gamma*R*T)
        a = (1.4 * R * T) ** 0.5

        # Set new output for the calculated speed of sound
        a.setOutputs("Speed of Sound (m/s)")

        # Save calculated speed of sound
        self.speedOfSound = a

        return None

    def calculateDynamicViscosity(self):
        """ Compute the dynamic viscosity of the atmosphere as a function of
        heigth by using the formula given in ISO 2533 u = B*T^(1.5)/(T+S).
        This function is automatically called whenever a new atmospheric model is set.

        Parameters
        ----------
        None

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
        u.setOutputs("Dynamic Viscosity (Pa s)")

        # Save calculated density
        self.dynamicViscosity = u

        return None

    def addWindGust(self, windGustX, windGustY):
        """ Adds a function to the current stored wind profile, in order to
        simulate a wind gust.

        Parameters
        ----------
        windGustX : float, callable
            Callable, function of altitude, which will be added to the
            x velocity of the current stored wind profile. If float is given,
            it will be considered as a constant function in altitude.
        windGustY : float, callable
            Callable, function of altitude, which will be added to the
            y velocity of the current stored wind profile. If float is given,
            it will be considered as a constant function in altitude.

        Returns
        -------
        None
        """
        # Recalculate windVelocityX and windVelocityY
        self.windVelocityX = self.windVelocityX + windGustX
        self.windVelocityY = self.windVelocityY + windGustY

        # Reset windVelocityX and windVelocityY details
        self.windVelocityX.setInputs("Height (m)")
        self.windVelocityX.setOutputs("Wind Velocity X (m/s)")
        self.windVelocityY.setInputs("Height (m)")
        self.windVelocityY.setOutputs("Wind Velocity Y (m/s)")

        # Reset wind heading and velocity magnitude
        self.windHeading = Function(
            lambda h: (180 / np.pi)
            * np.arctan2(self.windVelocityX(h), self.windVelocityY(h))
            % 360,
            "Height (m)",
            "Wind Heading (degrees)",
            extrapolation="constant",
        )
        self.windSpeed = Function(
            lambda h: (self.windVelocityX(h) ** 2 + self.windVelocityY(h) ** 2) ** 0.5,
            "Height (m)",
            "Wind Speed (m/s)",
            extrapolation="constant",
        )

        return None

    def info(self):
        """Prints most important data and graphs available about the
        Environment.

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Print launch site details
        print("Launch Site Details")
        print("\nLaunch Rail Length: ", self.rL, " m")
        if self.date != None:
            print("Launch Date: ", self.date, " UTC")
        if self.lat != None and self.lon != None:
            print("Launch Site Latitude: {:.5f}°".format(self.lat))
            print("Launch Site Longitude: {:.5f}°".format(self.lon))
        print("Launch Site Surface Elevation: {:.1f} m".format(self.elevation))

        # Print atmospheric model details
        print("\n\nAtmospheric Model Details")
        modelType = self.atmosphericModelType
        print("\nAtmospheric Model Type: ", modelType)
        print(
            modelType
            + " Maximum Height: {:.3f} km".format(self.maxExpectedHeight / 1000)
        )
        if modelType in ["Forecast", "Reanalysis", "Ensemble"]:
            # Determine time period
            initDate = self.atmosphericModelInitDate
            endDate = self.atmosphericModelEndDate
            interval = self.atmosphericModelInterval
            print(modelType + " Time Period: From ", initDate, " to ", endDate, " UTC")
            print(modelType + " Hour Interval: ", interval, " hrs")
            # Determine latitude and longitude range
            initLat = self.atmosphericModelInitLat
            endLat = self.atmosphericModelEndLat
            initLon = self.atmosphericModelInitLon
            endLon = self.atmosphericModelEndLon
            print(modelType + " Latitude Range: From ", initLat, "° To ", endLat, "°")
            print(modelType + " Longitude Range: From ", initLon, "° To ", endLon, "°")
        if modelType == "Ensemble":
            print("Number of Ensemble Members: ", self.numEnsembleMembers)
            print("Selected Ensemble Member: ", self.ensembleMember, " (Starts from 0)")

        # Print atmospheric conditions
        print("\n\nSurface Atmospheric Conditions")
        print("\nSurface Wind Speed: {:.2f} m/s".format(self.windSpeed(self.elevation)))
        print(
            "Surface Wind Direction: {:.2f}°".format(self.windDirection(self.elevation))
        )
        print("Surface Wind Heading: {:.2f}°".format(self.windHeading(self.elevation)))
        print(
            "Surface Pressure: {:.2f} hPa".format(self.pressure(self.elevation) / 100)
        )
        print("Surface Temperature: {:.2f} K".format(self.temperature(self.elevation)))
        print("Surface Air Density: {:.3f} kg/m³".format(self.density(self.elevation)))
        print(
            "Surface Speed of Sound: {:.2f} m/s".format(
                self.speedOfSound(self.elevation)
            )
        )

        # Plot graphs
        print("\n\nAtmospheric Model Plots")
        # Create height grid
        grid = np.linspace(self.elevation, self.maxExpectedHeight)

        # Create figure
        plt.figure(figsize=(9, 4.5))

        # Create wind speed and wind direction subplot
        ax1 = plt.subplot(121)
        ax1.plot(
            [self.windSpeed(i) for i in grid], grid, "#ff7f0e", label="Speed of Sound"
        )
        ax1.set_xlabel("Wind Speed (m/s)", color="#ff7f0e")
        ax1.tick_params("x", colors="#ff7f0e")
        ax1up = ax1.twiny()
        ax1up.plot(
            [self.windDirection(i) for i in grid],
            grid,
            color="#1f77b4",
            label="Density",
        )
        ax1up.set_xlabel("Wind Direction (°)", color="#1f77b4")
        ax1up.tick_params("x", colors="#1f77b4")
        ax1up.set_xlim(0, 360)
        ax1.set_ylabel("Height Above Sea Level (m)")
        ax1.grid(True)

        # Create density and speed of sound subplot
        ax2 = plt.subplot(122)
        ax2.plot(
            [self.speedOfSound(i) for i in grid],
            grid,
            "#ff7f0e",
            label="Speed of Sound",
        )
        ax2.set_xlabel("Speed of Sound (m/s)", color="#ff7f0e")
        ax2.tick_params("x", colors="#ff7f0e")
        ax2up = ax2.twiny()
        ax2up.plot(
            [self.density(i) for i in grid], grid, color="#1f77b4", label="Density"
        )
        ax2up.set_xlabel("Density (kg/m³)", color="#1f77b4")
        ax2up.tick_params("x", colors="#1f77b4")
        ax2.set_ylabel("Height Above Sea Level (m)")
        ax2.grid(True)

        plt.subplots_adjust(wspace=0.5)
        plt.show()

    def allInfo(self):
        """Prints out all data and graphs available about the Environment.

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Print gravity details
        print("Gravity Details")
        print("\nAcceleration of Gravity: " + str(self.g) + " m/s²")

        # Print launch site details
        print("\n\nLaunch Site Details")
        print("\nLaunch Rail Length: ", self.rL, " m")
        if self.date != None:
            print("Launch Date: ", self.date, " UTC")
        if self.lat != None and self.lon != None:
            print("Launch Site Latitude: {:.5f}°".format(self.lat))
            print("Launch Site Longitude: {:.5f}°".format(self.lon))
        print("Launch Site Surface Elevation: {:.1f} m".format(self.elevation))

        # Print atmospheric model details
        print("\n\nAtmospheric Model Details")
        modelType = self.atmosphericModelType
        print("\nAtmospheric Model Type: ", modelType)
        print(
            modelType
            + " Maximum Height: {:.3f} km".format(self.maxExpectedHeight / 1000)
        )
        if modelType in ["Forecast", "Reanalysis", "Ensemble"]:
            # Determine time period
            initDate = self.atmosphericModelInitDate
            endDate = self.atmosphericModelEndDate
            interval = self.atmosphericModelInterval
            print(modelType + " Time Period: From ", initDate, " to ", endDate, " UTC")
            print(modelType + " Hour Interval: ", interval, " hrs")
            # Determine latitude and longitude range
            initLat = self.atmosphericModelInitLat
            endLat = self.atmosphericModelEndLat
            initLon = self.atmosphericModelInitLon
            endLon = self.atmosphericModelEndLon
            print(modelType + " Latitude Range: From ", initLat, "° To ", endLat, "°")
            print(modelType + " Longitude Range: From ", initLon, "° To ", endLon, "°")
        if modelType == "Ensemble":
            print("Number of Ensemble Members: ", self.numEnsembleMembers)
            print("Selected Ensemble Member: ", self.ensembleMember, " (Starts from 0)")

        # Print atmospheric conditions
        print("\n\nSurface Atmospheric Conditions")
        print("\nSurface Wind Speed: {:.2f} m/s".format(self.windSpeed(self.elevation)))
        print(
            "Surface Wind Direction: {:.2f}°".format(self.windDirection(self.elevation))
        )
        print("Surface Wind Heading: {:.2f}°".format(self.windHeading(self.elevation)))
        print(
            "Surface Pressure: {:.2f} hPa".format(self.pressure(self.elevation) / 100)
        )
        print("Surface Temperature: {:.2f} K".format(self.temperature(self.elevation)))
        print("Surface Air Density: {:.3f} kg/m³".format(self.density(self.elevation)))
        print(
            "Surface Speed of Sound: {:.2f} m/s".format(
                self.speedOfSound(self.elevation)
            )
        )

        # Plot graphs
        print("\n\nAtmospheric Model Plots")
        # Create height grid
        grid = np.linspace(self.elevation, self.maxExpectedHeight)

        # Create figure
        plt.figure(figsize=(9, 9))

        # Create wind speed and wind direction subplot
        ax1 = plt.subplot(221)
        ax1.plot(
            [self.windSpeed(i) for i in grid], grid, "#ff7f0e", label="Speed of Sound"
        )
        ax1.set_xlabel("Wind Speed (m/s)", color="#ff7f0e")
        ax1.tick_params("x", colors="#ff7f0e")
        ax1up = ax1.twiny()
        ax1up.plot(
            [self.windDirection(i) for i in grid],
            grid,
            color="#1f77b4",
            label="Density",
        )
        ax1up.set_xlabel("Wind Direction (°)", color="#1f77b4")
        ax1up.tick_params("x", colors="#1f77b4")
        ax1up.set_xlim(0, 360)
        ax1.set_ylabel("Height Above Sea Level (m)")
        ax1.grid(True)

        # Create density and speed of sound subplot
        ax2 = plt.subplot(222)
        ax2.plot(
            [self.speedOfSound(i) for i in grid],
            grid,
            "#ff7f0e",
            label="Speed of Sound",
        )
        ax2.set_xlabel("Speed of Sound (m/s)", color="#ff7f0e")
        ax2.tick_params("x", colors="#ff7f0e")
        ax2up = ax2.twiny()
        ax2up.plot(
            [self.density(i) for i in grid], grid, color="#1f77b4", label="Density"
        )
        ax2up.set_xlabel("Density (kg/m³)", color="#1f77b4")
        ax2up.tick_params("x", colors="#1f77b4")
        ax2.set_ylabel("Height Above Sea Level (m)")
        ax2.grid(True)

        # Create wind u and wind v subplot
        ax3 = plt.subplot(223)
        ax3.plot([self.windVelocityX(i) for i in grid], grid, label="Wind U")
        ax3.plot([self.windVelocityY(i) for i in grid], grid, label="Wind V")
        ax3.legend(loc="best").set_draggable(True)
        ax3.set_ylabel("Height Above Sea Level (m)")
        ax3.set_xlabel("Wind Speed (m/s)")
        ax3.grid(True)

        # Create pressure and temperature subplot
        ax4 = plt.subplot(224)
        ax4.plot(
            [self.pressure(i) / 100 for i in grid], grid, "#ff7f0e", label="Pressure"
        )
        ax4.set_xlabel("Pressure (hPa)", color="#ff7f0e")
        ax4.tick_params("x", colors="#ff7f0e")
        ax4up = ax4.twiny()
        ax4up.plot(
            [self.temperature(i) for i in grid],
            grid,
            color="#1f77b4",
            label="Temperature",
        )
        ax4up.set_xlabel("Temperature (K)", color="#1f77b4")
        ax4up.tick_params("x", colors="#1f77b4")
        ax4.set_ylabel("Height Above Sea Level (m)")
        ax4.grid(True)

        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()

        # Plot ensemble member comparison
        if self.atmosphericModelType != "Ensemble":
            return None

        print("\n\nEnsemble Members Comparison")
        currentMember = self.ensembleMember

        # Create figure
        plt.figure(figsize=(9, 13.5))

        # Create wind u subplot
        ax5 = plt.subplot(321)
        for i in range(self.numEnsembleMembers):
            self.selectEnsembleMember(i)
            ax5.plot([self.windVelocityX(i) for i in grid], grid, label=i)
        # ax5.legend(loc='best').set_draggable(True)
        ax5.set_ylabel("Height Above Sea Level (m)")
        ax5.set_xlabel("Wind Speed (m/s)")
        ax5.set_title("Wind U - Ensemble Members")
        ax5.grid(True)

        # Create wind v subplot
        ax6 = plt.subplot(322)
        for i in range(self.numEnsembleMembers):
            self.selectEnsembleMember(i)
            ax6.plot([self.windVelocityY(i) for i in grid], grid, label=i)
        # ax6.legend(loc='best').set_draggable(True)
        ax6.set_ylabel("Height Above Sea Level (m)")
        ax6.set_xlabel("Wind Speed (m/s)")
        ax6.set_title("Wind V - Ensemble Members")
        ax6.grid(True)

        # Create wind speed subplot
        ax7 = plt.subplot(323)
        for i in range(self.numEnsembleMembers):
            self.selectEnsembleMember(i)
            ax7.plot([self.windSpeed(i) for i in grid], grid, label=i)
        # ax7.legend(loc='best').set_draggable(True)
        ax7.set_ylabel("Height Above Sea Level (m)")
        ax7.set_xlabel("Wind Speed (m/s)")
        ax7.set_title("Wind Speed Magnitude - Ensemble Members")
        ax7.grid(True)

        # Create wind direction subplot
        ax8 = plt.subplot(324)
        for i in range(self.numEnsembleMembers):
            self.selectEnsembleMember(i)
            ax8.plot([self.windDirection(i) for i in grid], grid, label=i)
        # ax8.legend(loc='best').set_draggable(True)
        ax8.set_ylabel("Height Above Sea Level (m)")
        ax8.set_xlabel("Degrees True (°)")
        ax8.set_title("Wind Direction - Ensemble Members")
        ax8.grid(True)

        # Create pressure subplot
        ax9 = plt.subplot(325)
        for i in range(self.numEnsembleMembers):
            self.selectEnsembleMember(i)
            ax9.plot([self.pressure(i) for i in grid], grid, label=i)
        # ax9.legend(loc='best').set_draggable(True)
        ax9.set_ylabel("Height Above Sea Level (m)")
        ax9.set_xlabel("Pressure (P)")
        ax9.set_title("Pressure - Ensemble Members")
        ax9.grid(True)

        # Create temperature subplot
        ax10 = plt.subplot(326)
        for i in range(self.numEnsembleMembers):
            self.selectEnsembleMember(i)
            ax10.plot([self.temperature(i) for i in grid], grid, label=i)
        # ax10.legend(loc='best').set_draggable(True)
        ax10.set_ylabel("Height Above Sea Level (m)")
        ax10.set_xlabel("Temperature (K)")
        ax10.set_title("Temperature - Ensemble Members")
        ax10.grid(True)

        # Display plot
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()

        # Clean up
        self.selectEnsembleMember(currentMember)

        return None

