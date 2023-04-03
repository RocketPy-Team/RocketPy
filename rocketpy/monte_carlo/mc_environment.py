from typing import Any, List, Tuple, Union

from pydantic import Field, StrictFloat, StrictInt, StrictStr, root_validator, validator

from ..Environment import Environment
from .DispersionModel import DispersionModel


# TODO: name suggestions: `DispersionEnvironment`, `DispEnvironment`, EnvironmentDispersion`, EnvironmentDisp` or `EnvironmentData`


class McEnvironment(DispersionModel):
    """Holds information about an Environment to be used in the Dispersion class.

    Parameters
    ----------
    environment : Environment
        Environment to extract data from. See help(Environment) for more information.
    railLength : int, float, tuple, optional
        Length in which the rocket will be attached to the rail, only moving along a fixed
        direction, that is, the line parallel to the rail. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLength used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribution function to be used in the dispersion. If no string is
        passed, a normal distribution is used.
    date : list, optional
        Array of length 4, stating (year, month, day, hour (UTC)) of rocket launch.
        Must be given if a Forecast, Reanalysis or Ensemble, will be set as an
        atmospheric model. Should be given as a list of these arrays of length 4.
        Each item of the list will be randomly chosen in each iteration of the dispersion.
    elevation : int, float, tuple, optional
        Elevation of launch site measured as height above sea level in meters.
        Alternatively, can be set as 'Open-Elevation' which uses the Open-Elevation API
        to find elevation data. For this option, latitude and longitude must also be
        specified. Default value is 0. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLength used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribution function to be used in the dispersion. If no string is
        passed, a normal distribution is used.
    gravity : int, float, tuple, optional
        Surface gravitational acceleration. Positive values point the acceleration down.
        Default value is 9.80665. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLength used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribution function to be used in the dispersion. If no string is
        passed, a normal distribution is used.
    latitude : int, float, tuple, optional
        Latitude in degrees (ranging from -90 to 90) of rocket launch location. Must be
        given if a Forecast, Reanalysis or Ensemble will be used as an atmospheric model
        or if Open-Elevation will be used to compute elevation. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLength used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribution function to be used in the dispersion. If no string is
        passed, a normal distribution is used.
    longitude : int, float, tuple, optional
        Longitude in degrees (ranging from 180 to 360) of rocket launch location.
        Must be given if a Forecast, Reanalysis or Ensemble will be used as an atmospheric
        model or if Open-Elevation will be used to compute elevation. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLength used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribution function to be used in the dispersion. If no string is
        passed, a normal distribution is used.
    datum : list, optional
        The desired reference ellipsoidal model, the following options are available:
        "SAD69", "WGS84", "NAD83", and "SIRGAS2000". The default is "SIRGAS2000", then
        this model will be used if the user make some typing mistake. Should be given as
        a list of strings. Each item of the list will be randomly chosen in each iteration
        of the dispersion.
    timeZone : list, optional
        Name of the time zone. To see all time zones, import pytz and run. Should be given
        as a list of strings. Each item of the list will be randomly chosen in each
        iteration of the dispersion.

    TODO: it seems you are copying documentation from the Environment class. This is
    not a good practice. You should only document the parameters that are specific to
    this class. The rest should be documented in the Environment class. If you want to
    document the parameters that are related to other classes, you can refer them here.
    For example, you can type 'see rocketpy.Environment for more information'.
    Examples: https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.stats.basic_stats
    """

    environment: Environment = Field(..., exclude=True)
    railLength: Any = 0
    date: List[Union[Tuple[int, int, int, int], None]] = []
    elevation: Any = 0
    gravity: Any = 0
    latitude: Any = 0
    longitude: Any = 0
    ensembleMember: List[StrictInt] = []
    windXFactor: Any = (1, 0)
    windYFactor: Any = (1, 0)
    datum: List[Union[StrictStr, None]] = []
    timeZone: List[Union[StrictStr, None]] = []

    @validator("ensembleMember")
    def val_ensemble(cls, v, values):
        """Validator for ensembleMember argument. Checks if environment has the correct
        atmospheric model type and if the list does not overflow the ensemble members.
        """
        if v:
            assert values["environment"].atmosphericModelType in [
                "Ensemble",
                "Reanalysis",
            ], f"\tEnvironment with {values['environment'].atmosphericModelType} does not have emsemble members"
            assert (
                max(v) < values["environment"].numEnsembleMembers and min(v) >= 0
            ), f"\tPlease choose ensembleMember from 0 to {values['environment'].numEnsembleMembers - 1}"
        return v
