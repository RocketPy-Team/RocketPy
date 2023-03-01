from typing import Any, List, Tuple, Union

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    validator,
    root_validator,
)

from .Environment import Environment


class EnvironmentData(BaseModel):
    """Class that holds that about an environment to be used in the Dispersion class.

    Parameters
    ----------
    environment : Environment
        Environment to extract data from. See help(Environment) for more information.
    railLength : int, float, tuple, optional
        Length in which the rocket will be attached to the rail, only moving along a fixed
        direction, that is, the line parallel to the rail. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLenght used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribuition function to be used in the dispersion. If no string is
        passed, a normal distribuition is used.
    date : list, optional
        Array of length 4, stating (year, month, day, hour (UTC)) of rocket launch.
        Must be given if a Forecast, Reanalysis or Ensemble, will be set as an
        atmospheric model. Should be given as a list of these arrays of length 4.
        Each item of the list will be ramdonly chosen in each iteration of the dispersion.
    elevation : int, float, tuple, optional
        Elevation of launch site measured as height above sea level in meters.
        Alternatively, can be set as 'Open-Elevation' which uses the Open-Elevation API
        to find elevation data. For this option, latitude and longitude must also be
        specified. Default value is 0. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLenght used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribuition function to be used in the dispersion. If no string is
        passed, a normal distribuition is used.
    gravity : int, float, tuple, optional
        Surface gravitational acceleration. Positive values point the acceleration down.
        Default value is 9.80665. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLenght used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribuition function to be used in the dispersion. If no string is
        passed, a normal distribuition is used.
    latitude : int, float, tuple, optional
        Latitude in degrees (ranging from -90 to 90) of rocket launch location. Must be
        given if a Forecast, Reanalysis or Ensemble will be used as an atmospheric model
        or if Open-Elevation will be used to compute elevation. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLenght used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribuition function to be used in the dispersion. If no string is
        passed, a normal distribuition is used.
    longitude : int, float, tuple, optional
        Longitude in degrees (ranging from -180 to 360) of rocket launch location.
        Must be given if a Forecast, Reanalysis or Ensemble will be used as an atmospheric
        model or if Open-Elevation will be used to compute elevation. Should be given as an int or
        float, or a tuple. If int or float, the value passed will be used to construct a
        tuple where the first item is the values of the railLenght used in the environment
        object, and the second item is the value passed. This value represents the
        standard deviation to be used in the Dispersion. If a tuple is passed, the first
        item must be the standard deviation and the second must be a string containing the
        name of the distribuition function to be used in the dispersion. If no string is
        passed, a normal distribuition is used.
    datum : list, optional
        The desired reference ellipsoidal model, the following options are available:
        "SAD69", "WGS84", "NAD83", and "SIRGAS2000". The default is "SIRGAS2000", then
        this model will be used if the user make some typing mistake. Should be given as
        a list of strings. Each item of the list will be ramdonly chosen in each iteration
        of the dispersion.
    timeZone : list, optional
        Name of the time zone. To see all time zones, import pytz and run. Should be given
        as a list of strings. Each item of the list will be ramdonly chosen in each
        iteration of the dispersion.
    """

    environment: Environment = Field(..., repr=False)
    railLength: Any = 0
    date: List[Union[Tuple[int, int, int, int], None]] = []
    elevation: Any = 0
    gravity: Any = 0
    latitude: Any = 0
    longitude: Any = 0
    datum: List[Union[StrictStr, None]] = []
    timeZone: List[Union[StrictStr, None]] = []

    class Config:
        arbitrary_types_allowed = True

    @root_validator(skip_on_failure=True)
    def set_attr(cls, values):
        """Validates inputs that can be either tuples, lists, ints or floats and
        saves them in the format (nom_val,std) or (nom_val,std,'dist_func').
        Lists are saved as they are inputted.
        Inputs can be given as floats or ints, refering to the standard deviation.
        In this case, the nominal value of that attribute will come from the rocket
        object passed. If the distribuition function whants to be specified, then a
        tuple with the standard deviation as the first item, and the string containing
        the name a numpy.random distribuition function can be passed.
        If a tuple with a nominal value and a standard deviation is passed, then it
        will take priority over the rocket object attriubute's value. A third item
        can also be added to the tuple specifying the distribuition function"""
        # TODO: add a way to check if the strings refering to the distribuition func
        # are valid names for numpy.random functions

        validate_fields = [
            "railLength",
            "elevation",
            "gravity",
            "latitude",
            "longitude",
            "date",
            "datum",
            "timeZone",
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"\nField '{field}': \n\tFirst item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"\nField '{field}': \n\tSecond item of tuple must be either an int, float or string \n  If the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation \n  If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribuition function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (
                            getattr(values["environment"], field),
                            v[0],
                            v[1],
                        )
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"\nField '{field}': \n\tSecond item of tuple must be either an int or float \n  The second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"\nField '{field}': \n\tThird item of tuple must be a string \n  The string should contain the name of a valid numpy.random distribuition function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values["environment"], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"\nField '{field}': \n\tItems in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values["environment"], field), v)
            else:
                raise ValueError(
                    f"\nField '{field}': \n\tMust be either a tuple, list, int or float"
                )
        return values
