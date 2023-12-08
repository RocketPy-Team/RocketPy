from random import choice

from ..tools import get_distribution


class DispersionModel:
    """Base class for all dispersion models. This class is used to validate
    the input parameters of the dispersion models, based on the pydantic library.
    """

    # List of arguments that are validated in child classes
    exception_list = [
        "initial_solution",
        "terminate_on_apogee",
        "ensemble_member",
    ]

    def __init__(self, object, **kwargs):
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
            - Else, the list is saved as is.
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
        self.object = object

        for input_name, input_value in kwargs.items():
            if input_name not in self.exception_list and input_value is not None:
                if "factor" in input_name:
                    attr_value = self._validate_factors(input_name, input_value)
                elif input_name not in self.exception_list:
                    if isinstance(input_value, tuple):
                        attr_value = self._validate_tuple(input_name, input_value)
                    elif isinstance(input_value, list):
                        attr_value = self._validate_list(input_name, input_value)
                    elif isinstance(input_value, (int, float)):
                        attr_value = self._validate_scalar(input_name, input_value)
                    else:
                        raise ValueError(
                            f"'{input_name}' must be a tuple, list, int, or float"
                        )
                setattr(self, input_name, attr_value)

    def __str__(self):
        s = ""
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # Skip attributes starting with underscore
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

    def _validate_tuple(self, input_name, input_value):
        assert len(input_value) in [
            2,
            3,
        ], f"'{input_name}': tuple must have length 2 or 3"
        assert isinstance(
            input_value[0], (int, float)
        ), f"'{input_name}': First item of tuple must be either an int or float"

        if len(input_value) == 2:
            return self._validate_tuple_length_two(input_name, input_value)
        elif len(input_value) == 3:
            return self._validate_tuple_length_three(input_name, input_value)

    def _validate_tuple_length_two(self, input_name, input_value):
        assert isinstance(input_value[1], (int, float, str)), (
            f"'{input_name}': second item of tuple must be an " "int, float, or string."
        )

        if isinstance(input_value[1], str):
            # if second item is a string, then it is assumed that the first item
            # is the standard deviation, and the second item is the distribution
            # function. In this case, the nominal value will be taken from the
            # object passed.
            dist_func = get_distribution(input_value[1])
            return (getattr(self.object, input_name), input_value[0], dist_func)
        else:
            # if second item is an int or float, then it is assumed that the
            # first item is the nominal value and the second item is the
            # standard deviation. The distribution function will be set to
            # "normal".
            return (input_value[0], input_value[1], get_distribution("normal"))

    def _validate_tuple_length_three(self, input_name, input_value):
        assert isinstance(input_value[1], (int, float)), (
            f"'{input_name}': Second item of a tuple with length 3 must be "
            "an int or float."
        )
        assert isinstance(input_value[2], str), (
            f"'{input_name}': Third item of tuple must be a "
            "string containing the name of a valid numpy.random "
            "distribution function."
        )
        dist_func = get_distribution(input_value[2])
        return (input_value[0], input_value[1], dist_func)

    def _validate_list(self, input_name, input_value):
        if not input_value:
            # if list is empty, then the value will be taken from the object
            # passed and saved as a list with one item.
            return [getattr(self.object, input_name)]
        else:
            # else, the list is saved as is.
            return input_value

    def _validate_scalar(self, input_name, input_value):
        return (
            getattr(self.object, input_name),
            input_value,
            get_distribution("normal"),
        )

    def _validate_factors(self, input_name, input_value):
        """Validator for factor arguments. Checks if input is in a valid format.
        Factors can only be tuples of two or three items, or lists. Currently,
        the supported factors are: windXFactor, windYFactor, powerOffDragFactor,
        powerOnDragFactor.

        Raises
        ------
        ValueError
            If input is not a tuple, list, int, or float
        """
        # Save original value of attribute that factor is applied to as an
        # private attribute
        attribute_name = input_name.replace("_factor", "")
        setattr(self, f"_{attribute_name}", getattr(self.object, attribute_name))

        if isinstance(input_value, tuple):
            return self._validate_tuple_factor(input_name, input_value)
        elif isinstance(input_value, list):
            return self._validate_list_factor(input_name, input_value)
        else:
            raise AssertionError(f"`{input_name}`: must be either a tuple or list")

    def _validate_tuple_factor(self, input_name, factor_tuple):
        assert len(factor_tuple) in [
            2,
            3,
        ], f"'{input_name}`: Factors tuple must have length 2 or 3"
        assert all(
            isinstance(item, (int, float)) for item in factor_tuple[:2]
        ), f"'{input_name}`: First and second items of Factors tuple must be either an int or float"

        if len(factor_tuple) == 2:
            return (factor_tuple[0], factor_tuple[1], get_distribution("normal"))
        elif len(factor_tuple) == 3:
            assert isinstance(factor_tuple[2], str), (
                f"'{input_name}`: Third item of tuple must be a string containing the name "
                "of a valid numpy.random distribution function"
            )
            dist_func = get_distribution(factor_tuple[2])
            return (factor_tuple[0], factor_tuple[1], dist_func)

    def _validate_list_factor(self, input_name, factor_list):
        assert all(
            isinstance(item, (int, float)) for item in factor_list
        ), f"'{input_name}`: Items in list must be either ints or floats"
        return factor_list

    def dict_generator(self):
        """Generates a dictionary with the randomized values of the object's
        arguments and saves it in self.last_rnd_dict. Dictionary is keys are
        the object's arguments and values are the randomized values.

        Parameters
        ----------
        None

        Yields
        ------
        generated_dict : dict
            Dictionary with randomized values of the object's arguments.
        """
        generated_dict = {}
        for arg, value in self.__dict__.items():
            if isinstance(value, tuple):
                generated_dict[arg] = value[-1](value[0], value[1])
            elif isinstance(value, list):
                generated_dict[arg] = choice(value) if value else value
        self.last_rnd_dict = generated_dict
        yield generated_dict
