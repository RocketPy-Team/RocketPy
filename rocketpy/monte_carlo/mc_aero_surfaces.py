# TODO: add heading and description of the file here, as usual in RocketPy

from typing import Any, List, Tuple, Union

from pydantic import (
    BaseModel,
    Extra,
    Field,
    FilePath,
    StrictInt,
    StrictStr,
    root_validator,
)

from ..AeroSurfaces import EllipticalFins, NoseCone, Tail, TrapezoidalFins


class McNoseCone(BaseModel):
    """TODO: add docstring here, maybe with some examples

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    nosecone: NoseCone = Field(..., repr=False)
    length: Any = 0
    kind: List[Union[StrictStr, None]] = []
    baseRadius: Any = 0  # TODO: is this really necessary?
    rocketRadius: Any = 0  # TODO: is this really necessary?
    name: List[StrictStr] = []
    # TODO: question: how can we document the above code lines? Why are they here?

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
        # TODO: please comment WHY this step is necessary

    @root_validator(skip_on_failure=True)
    def set_attr(cls, values):
        """Validates inputs that can be either tuples, lists, ints or floats and
        saves them in the format (mean, std) or (mean, std,'dist_func').
        Lists are saved as they are inputted.
        Inputs can be given as floats or ints, referring to the standard deviation.
        In this case, the nominal value of that attribute will come from the rocket
        object passed. If the distribution function needs to be specified, then a
        tuple with the standard deviation as the first item, and the string containing
        the name a numpy.random distribution function can be passed.
        If a tuple with a nominal value and a standard deviation is passed, then it
        will take priority over the rocket object attribute's value. A third item
        can also be added to the tuple specifying the distribution function

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        validate_fields = [
            "length",
            "kind",
            "baseRadius",
            "rocketRadius",
            "name",
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"Field '{field}': \tFirst item of tuple must be either an int or float"
                # if len == 1, it can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"Field '{field}': Second item of tuple must be either an int, float or string. If the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation. If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (getattr(values["nosecone"], field), v[0], v[1])
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"Field '{field}': second item of tuple must be either an int or float. The second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"Field '{field}': \tThird item of tuple must be a string. The string should contain the name of a valid numpy.random distribution function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values["nosecone"], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"Field '{field}': items in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values["nosecone"], field), v)
            else:
                raise ValueError(
                    f"The field '{field}' must be a tuple, list, int or float"
                )
        return values


class McTrapezoidalFins(BaseModel):
    """TODO: add docstring here, maybe with some examples

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    trapezoidalFins: TrapezoidalFins = Field(..., repr=False)
    n: List[StrictInt] = []
    rootChord: Any = 0
    tipChord: Any = 0
    span: Any = 0
    rocketRadius: Any = 0
    cantAngle: Any = 0
    sweepLength: Any = 0
    sweepAngle: Any = 0
    airfoil: List[Union[Tuple[FilePath, StrictStr], None]] = []
    name: List[StrictStr] = []

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    @root_validator(skip_on_failure=True)
    def set_attr(cls, values):
        """Validates inputs that can be either tuples, lists, ints or floats and
        saves them in the format (nom_val,std) or (nom_val,std,'dist_func').
        Lists are saved as they are inputted.
        Inputs can be given as floats or ints, referring to the standard deviation.
        In this case, the nominal value of that attribute will come from the rocket
        object passed. If the distribution function needs to be specified, then a
        tuple with the standard deviation as the first item, and the string containing
        the name a numpy.random distribution function can be passed.
        If a tuple with a nominal value and a standard deviation is passed, then it
        will take priority over the rocket object attribute's value. A third item
        can also be added to the tuple specifying the distribution function"""
        # TODO: add a way to check if the strings referring to the distribution func
        # are valid names for numpy.random functions

        validate_fields = [
            "n",
            "rootChord",
            "tipChord",
            "span",
            "rocketRadius",
            "cantAngle",
            "sweepLength",
            "sweepAngle",
            "airfoil",
            "name",
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"Field '{field}': \tFirst item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"Field '{field}': \tSecond item of tuple must be either an int, float or string   If the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation   If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (
                            getattr(values["trapezoidalFins"], field),
                            v[0],
                            v[1],
                        )
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"Field '{field}': \tSecond item of tuple must be either an int or float   The second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"Field '{field}': \tThird item of tuple must be a string   The string should contain the name of a valid numpy.random distribution function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values["trapezoidalFins"], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"Field '{field}': \tItems in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values["trapezoidalFins"], field), v)
            else:
                raise ValueError(
                    f"Field '{field}': \tMust be either a tuple, list, int or float"
                )
        return values


class McEllipticalFins(BaseModel):
    """TODO: add docstring here, maybe with some examples

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    ellipticalFins: EllipticalFins = Field(..., repr=False)
    n: Any = 0
    rootChord: Any = 0
    span: Any = 0
    rocketRadius: Any = 0
    cantAngle: Any = 0
    airfoil: List[Union[Tuple[FilePath, StrictStr], None]] = []
    name: List[StrictStr] = []

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    @root_validator(skip_on_failure=True)
    def set_attr(cls, values):
        """Validates inputs that can be either tuples, lists, ints or floats and
        saves them in the format (nom_val,std) or (nom_val,std,'dist_func').
        Lists are saved as they are inputted.
        Inputs can be given as floats or ints, referring to the standard deviation.
        In this case, the nominal value of that attribute will come from the rocket
        object passed. If the distribution function needs to be specified, then a
        tuple with the standard deviation as the first item, and the string containing
        the name a numpy.random distribution function can be passed.
        If a tuple with a nominal value and a standard deviation is passed, then it
        will take priority over the rocket object attribute's value. A third item
        can also be added to the tuple specifying the distribution function"""
        # TODO: add a way to check if the strings referring to the distribution func
        # are valid names for numpy.random functions

        validate_fields = [
            "n",
            "rootChord",
            "span",
            "rocketRadius",
            "cantAngle",
            "airfoil",
            "name",
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"Field '{field}': \tFirst item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"Field '{field}': \tSecond item of tuple must be either an int, float or string   If the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation   If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (
                            getattr(values["ellipticalFins"], field),
                            v[0],
                            v[1],
                        )
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"Field '{field}': \tSecond item of tuple must be either an int or float   The second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"Field '{field}': \tThird item of tuple must be a string   The string should contain the name of a valid numpy.random distribution function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values["ellipticalFins"], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"Field '{field}': \tItems in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values["ellipticalFins"], field), v)
            else:
                raise ValueError(
                    f"Field '{field}': \tMust be either a tuple, list, int or float"
                )
        return values


class McTail(BaseModel):
    """TODO: add docstring

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    tail: Tail = Field(..., repr=False)
    topRadius: Any = 0
    bottomRadius: Any = 0
    length: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    @root_validator(skip_on_failure=True)
    def set_attr(cls, values):
        """Validates inputs that can be either tuples, lists, ints or floats and
        saves them in the format (nom_val,std) or (nom_val,std,'dist_func').
        Lists are saved as they are inputted.
        Inputs can be given as floats or ints, referring to the standard deviation.
        In this case, the nominal value of that attribute will come from the rocket
        object passed. If the distribution function needs to be specified, then a
        tuple with the standard deviation as the first item, and the string containing
        the name a numpy.random distribution function can be passed.
        If a tuple with a nominal value and a standard deviation is passed, then it
        will take priority over the rocket object attribute's value. A third item
        can also be added to the tuple specifying the distribution function"""
        # TODO: add a way to check if the strings referring to the distribution func
        # are valid names for numpy.random functions

        validate_fields = [
            "topRadius",
            "bottomRadius",
            "length",
            "rocketRadius",
            "name",
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"Field '{field}': \tFirst item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"Field '{field}': \tSecond item of tuple must be either an int, float or string   If the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation   If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (getattr(values["tail"], field), v[0], v[1])
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"Field '{field}': \tSecond item of tuple must be either an int or float   The second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"Field '{field}': \tThird item of tuple must be a string   The string should contain the name of a valid numpy.random distribution function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values["tail"], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"Field '{field}': \tItems in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values["tail"], field), v)
            else:
                raise ValueError(
                    f"Field '{field}': \tMust be either a tuple, list, int or float"
                )
        return values
