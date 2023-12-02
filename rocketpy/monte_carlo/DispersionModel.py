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
    def validate_generic_field(cls, values):
        """Validates generic fields, which are those that can be either tuples,
        lists, ints or floats, and saves them as atrributes of the object.
        The saved values are either tuples or lists, depending on the input type.
        If the input is a tuple, int or float, then it is saved in the format
        (nominal value, standard deviation, distribution function).If the input
        is a list, then it is saved as a list of values.

        The following validation rules are applied according to the input type:

        - Tuples are validated as follows:
            - Must have length 2 or 3
            - First item must be either an int or float
            - If length is two, then the type of the second item must be either
              an int, float or str.
                - If the second item is an int or float, then it is assumed that
                  the first item is the nominal value and the second item is the
                  standard deviation.
                - If the second item is a string, then it is assumed that the
                  first item is the standard deviation, and the second item is
                  the distribution function. In this case, the nominal value
                  will be taken from the object passed.
            - If length is three, then it is assumed that the first item is the
              nominal value, the second item is the standard deviation and the
              third item is the distribution function.
        - Lists are validated as follows:
            - If the list is empty, then the value will be taken from the object
              passed and saved as a list with one item.
            - If the list is not empty, then all items must be either ints or
              floats.
        - Ints or floats are validated as follows:
            - The value is assumed to be the standard deviation, the nominal
              value will be taken from the object passed and the distribution
              function will be set to "normal".

        Parameters
        ----------
        values : dict
            Dictionary with the object's arguments and their values.

        Returns
        -------
        values : dict
            Dictionary with the object's arguments and their values.
        """

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
            current_value = values[field]
            # checks if tuple
            if isinstance(current_value, tuple):
                # checks if tuple has acceptable length
                assert len(current_value) in [
                    2,
                    3,
                ], f"Field '{field}': tuple must have length 2 or 3"
                # checks if first item is valid
                assert isinstance(current_value[0], (int, float)), (
                    f"Field '{field}': First item of tuple must be either an "
                    "int or float"
                )
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(current_value) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(current_value[1], (int, float, str)), (
                        f"Field '{field}': second item of tuple must be an "
                        "int, float or string. If the first value refers to "
                        "the nominal value of {field}, then the tuple's second "
                        "value should be the desired standard deviation. If "
                        "the first value is the standard deviation, then the "
                        "tuple's second value should be a string containing a "
                        "name of a numpy.random distribution function"
                    )
                    # if second item is not str, then (nom_val, std, "normal")
                    if not isinstance(current_value[1], str):
                        values[field] = (
                            current_value[0],
                            current_value[1],
                            get_distribution("normal"),
                        )
                    # if second item is str, then (nom_val, std, 'dist_func')
                    else:
                        # check if 'dist_func' is a valid name
                        dist_func = get_distribution(current_value[1])
                        # saves values
                        values[field] = (
                            getattr(values[obj_name], field),
                            current_value[0],
                            dist_func,
                        )
                # if len is three, then (nom_val, std, 'dist_func')
                if len(current_value) == 3:
                    assert isinstance(current_value[1], (int, float)), (
                        f"Field '{field}': Second item of tuple must be either "
                        "an int or float, representing the standard deviation "
                        "to be used in the simulation."
                    )
                    assert isinstance(current_value[2], str), (
                        f"Field '{field}': Third item of tuple must be a "
                        "string containing the name of a valid numpy.random "
                        "distribution function"
                    )
                    # check if 'dist_func' is a valid name
                    dist_func = get_distribution(current_value[2])
                    values[field] = (current_value[0], current_value[1], dist_func)
            elif isinstance(current_value, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(current_value) == 0:
                    values[field] = [getattr(values[obj_name], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in current_value
                    ), (
                        f"Field '{field}' should be a list with items of type "
                        "int or float"
                    )
                    # all good, sets inputs
                    values[field] = current_value
            elif isinstance(current_value, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std, "normal")
                values[field] = (
                    getattr(values[obj_name], field),
                    current_value,
                    get_distribution("normal"),
                )
            else:
                raise ValueError(
                    f"Field '{field}' must be a tuple, list, int " "or float"
                )
        return values

    @validator(
        "windXFactor",
        "windYFactor",
        "powerOffDragFactor",
        "powerOnDragFactor",
        check_fields=False,
        always=True,
    )
    def validate_factors(cls, v):
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
            # checks if tuple has acceptable length
            assert len(v) in [2, 3], f"Factors tuple must have length 2 or 3"
            # checks if first and second items are valid
            assert isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)), (
                f"First and second items of Factors tuple must be either an "
                "int or float"
            )
            # len is two, then (nom_val, std, "normal")
            if len(v) == 2:
                return (v[0], v[1], get_distribution("normal"))
            # if len is three, then (nom_val, std, 'dist_func')
            if len(v) == 3:
                assert isinstance(v[2], str), (
                    f"Third item of tuple must be a string containing the name "
                    "of a valid numpy.random distribution function"
                )
                # check if 'dist_func' is a valid name
                dist_func = get_distribution(v[2])
                return (v[0], v[1], dist_func)
        elif isinstance(v, list):
            # guarantee all values are valid (ints or floats)
            assert all(
                isinstance(item, (int, float)) for item in v
            ), f"Items in list must be either ints or floats"
            # all good, sets inputs
            return v
        else:
            raise ValueError(f"Must be either a tuple or list")

    def dict_generator(self):
        """Generates a dictionary with the randomized values of the object's
        arguments and saves it in self.last_rnd_dict. Dictionary is keys are
        the object's arguments and values are the randomized values.

        Parameters
        ----------
        None

        Yields
        ------
        gen_dict : dict
            Dictionary with randomized values of the object's arguments.
        """
        gen_dict = {}
        for arg, value in self.dict().items():
            if isinstance(value, tuple):
                gen_dict[arg] = value[-1](value[0], value[1])
            elif isinstance(value, list):
                gen_dict[arg] = choice(value) if value else value
        self.last_rnd_dict = gen_dict
        yield gen_dict
