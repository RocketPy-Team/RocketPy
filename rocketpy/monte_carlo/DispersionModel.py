from pydantic import BaseModel, Extra, root_validator, validator


class DispersionModel(BaseModel):
    """_summary_

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

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    # TODO: find a way to validate if distribuition_function string is the name
    # of a valid np.random function
    # Currently the error is only raised in get_distribuition in tools.py
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

        # gets name of the object to be used in getattr()
        obj_name = list(cls.__fields__.keys())[0]

        # defines a list of fields that must not be validated
        # special exception for fiedls that should not be validated
        # or that have their own validator
        exception_list = [
            "initialSolution",
            "terminateOnApogee",
            "ensembleMember",
            "windXFactor",
            "windYFactor",
            "powerOffDragFactor",
            "powerOnDragFactor",
        ]

        # create list with name of the fields that need validation
        # which are all fields except the one refering to the object
        validate_fields = [
            field_name
            for field_name, field_type in cls.__fields__.items()
            if not field_type.required and field_name not in exception_list
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"\nField '{field}' \n\tFirst item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"\nField '{field}' \n\tSecond item of tuple must be either an int, float or string \n\tIf the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation \n\tIf the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (
                            getattr(values[obj_name], field),
                            v[0],
                            v[1],
                        )
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"\nField '{field}' \n\tSecond item of tuple must be either an int or float \n\tThe second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"\nField '{field}' \n\tThird item of tuple must be a string \n\tThe string should contain the name of a valid numpy.random distribution function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values[obj_name], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"\nField '{field}' \n\tItems in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values[obj_name], field), v)
            else:
                raise ValueError(
                    f"\nField '{field}' \n\tMust be either a tuple, list, int or float"
                )
        return values

    @validator(
        "windXFactor",
        "windYFactor",
        "powerOffDragFactor",
        "powerOnDragFactor",
        check_fields=False,
    )
    def val_factors(cls, v):
        """Validator for factor arguments. Checks if input is in a valid format.
        Factors can only be tuples of two or three items, or lists."""

        # checks if tuple
        if isinstance(v, tuple):
            # checks if first and second items are valid
            assert isinstance(v[0], (int, float)) and isinstance(
                v[1], (int, float)
            ), f"\tFirst and second items of Factors tuple must be either an int or float"
            # if len is three, then (nom_val, std, 'dist_func')
            if len(v) == 3:
                assert isinstance(
                    v[2], str
                ), f"\tThird item of tuple must be a string \n\tThe string should contain the name of a valid numpy.random distribution function"
                return v
            return v
        elif isinstance(v, list):
            # guarantee all values are valid (ints or floats)
            assert all(
                isinstance(item, (int, float)) for item in v
            ), f"\tItems in list must be either ints or floats"
            # all good, sets inputs
            return v
        else:
            raise ValueError(f"\tMust be either a tuple or list")
