__author__ = "Mateus Stano Junqueira, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


from random import choice

from pydantic import BaseModel, Extra, root_validator, validator

from ..tools import get_distribution


class DispersionModel(BaseModel):
    """Base class for all dispersion models. This class is used to validate
    the input parameters of the dispersion models, based on the pydantic library.
    """

    class Config:
        """Configures pydantic to allow arbitrary types and extra fields."""

        # Allows fields to be checked if they are of RocketPy classes types
        arbitrary_types_allowed = True
        # Allows the dataclass to contain additional fields that are not
        # defined in the class. Specially useful in McRocket
        extra = Extra.allow

    def __str__(self):
        s = ""
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                # Format the tuple as a string with the mean and standard deviation.
                value_str = f"{value[0]:.5f} ± {value[1]:.5f} (numpy.random.{value[2].__name__})"
            else:
                # Otherwise, just use the default string representation of the value.
                value_str = str(value)
            s += f"{key}: {value_str}\n"
        return s.strip()

    def __repr__(self):
        return self.__str__()

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
        # special exception for fields that should not be validated
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
        # which are all fields except the one referring to the object
        # and except the ones in exception_list
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
                ), f"Field '{field}': First item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"Field '{field}': second item of tuple must be an int, float or string. tIf the first value refers to the nominal value of {field}, then the item's second value should be the desired standard deviation. If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, 'dist_func')
                    else:
                        # check if 'dist_func' is a valid name
                        get_distribution(v[1])
                        # saves values
                        values[field] = (
                            getattr(values[obj_name], field),
                            v[0],
                            v[1],
                        )
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"Field '{field}': Second item of tuple must be either an int or float, representing the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"Field '{field}': Third item of tuple must be a string containing the name of a valid numpy.random distribution function"
                    # check if 'dist_func' is a valid name
                    get_distribution(v[2])
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
                    ), f"Field '{field}' should be either int or float"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values[obj_name], field), v)
            else:
                raise ValueError(f"Field '{field}' must be a tuple, list, int or float")
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
        Factors can only be tuples of two or three items, or lists. Currently,
        the supported factors are: windXFactor, windYFactor, powerOffDragFactor,
        powerOnDragFactor.

        Raises
        ------
        ValueError
            If input is not a tuple, list, int or float
        """

        # checks if tuple
        if isinstance(v, tuple):
            # checks if first and second items are valid
            assert isinstance(v[0], (int, float)) and isinstance(
                v[1], (int, float)
            ), f"First and second items of Factors tuple must be either an int or float"
            # if len is three, then (nom_val, std, 'dist_func')
            if len(v) == 3:
                assert isinstance(
                    v[2], str
                ), f"Third item of tuple must be a string containing the name of a valid numpy.random distribution function"
                get_distribution(v[2])
                return v
            return v
        elif isinstance(v, list):
            # guarantee all values are valid (ints or floats)
            assert all(
                isinstance(item, (int, float)) for item in v
            ), f"Items in list must be either ints or floats"
            # all good, sets inputs
            return v
        else:
            raise ValueError(f"Must be either a tuple or list")
