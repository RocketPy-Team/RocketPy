from typing import Any, Callable, List, Tuple, Union

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PrivateAttr,
    StrictFloat,
    StrictInt,
    StrictStr,
    root_validator,
    validator,
)

from .Parachute import Parachute


class ParachuteData(BaseModel):
    CdS: Any
    trigger: List[Callable]
    samplingRate: Any
    lag: Any
    name: List[StrictStr]
    noise: List[
        Tuple[
            Union[StrictInt, StrictFloat],
            Union[StrictInt, StrictFloat],
            Union[StrictInt, StrictFloat],
        ]
    ] = [(0, 0, 0)]

    @root_validator(skip_on_failure=True)
    def val_basic(cls, values):
        """Validates inputs that can be either tuples or lists.
        Tuples can have either 2 or 3 items. First two must be either float or int,
        representing the nominal value and standard deviation. Third item must be
        a string containing the name of a numpy.random distribuition function"""

        validate_fields = ["CdS", "samplingRate", "lag"]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first two items are valid
                assert isinstance(v[0], (int, float)) and isinstance(
                    v[1], (int, float)
                ), f"\nField '{field}': \n\tFirst two items of tuple must be either an int or float \n\tFirst item refers to nominal value, and the second to the standard deviation"
                # extra check for third item if passed
                if len(v) == 3:
                    assert isinstance(
                        v[2], str
                    ), f"\nField '{field}': \n\tThird item of tuple must be either a string \n\tThe third item must be a string containing the name of a numpy.random distribuition function"
                # all good, sets inputs
                values[field] = v
            elif isinstance(v, list):
                # guarantee all values are valid (ints or floats)
                assert all(
                    isinstance(item, (int, float)) for item in v
                ), f"\nField '{field}': \n\tItems in list must be either ints or floats"
                # all good, sets inputs
                values[field] = v
            else:
                raise ValueError(
                    f"\nField '{field}': \n\tMust be either a tuple or list"
                )
        return values


class ParachuteDataByParachute(BaseModel):
    parachute: Parachute = Field(..., repr=False)
    CdS: Any = 0
    trigger: List[Union[Callable, None]] = []
    samplingRate: Any = 0
    lag: Any = 0
    name: List[Union[StrictStr, None]] = []
    noise: List[
        Union[
            Tuple[
                Union[StrictInt, StrictFloat],
                Union[StrictInt, StrictFloat],
                Union[StrictInt, StrictFloat],
            ],
            None,
        ]
    ] = []

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

        validate_fields = ["CdS", "trigger", "samplingRate", "lag", "name", "noise"]
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
                            getattr(values["parachute"], field),
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
                    values[field] = [getattr(values["parachute"], field)]
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
                values[field] = (getattr(values["parachute"], field), v)
            else:
                raise ValueError(
                    f"\nField '{field}': \n\tMust be either a tuple, list, int or float"
                )
        return values
