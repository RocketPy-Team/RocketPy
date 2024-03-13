"""Defines the StochasticModel class, which will be used as a base class for all
other Stochastic classes."""

from random import choice

import numpy as np

from rocketpy.mathutils.function import Function

from ..tools import get_distribution


class StochasticModel:
    """Base class for all Stochastic classes. This class is used to validate
    the input arguments of the child classes. The input arguments are validated
    and saved as attributes of the class in the correct format. The attributes
    are then used to generate a dictionary with the randomly generated input
    arguments. The dictionary is saved as an attribute of the class.
    """

    # List of arguments that are validated in child classes
    exception_list = [
        "initial_solution",
        "terminate_on_apogee",
        "date",
        "ensemble_member",
    ]

    def __init__(self, object, **kwargs):
        """Initialize the StochasticModel class with validated input arguments.

        Parameters
        ----------
        object : object
            The main object of the class.
        **kwargs : dict
            Dictionary with input arguments for the class. Arguments should be
            provided as keyword arguments, where the key is the argument name,
            and the value is the argument value. Valid argument types include
            tuples, lists, ints, floats, or None. The arguments will then be
            validated and saved as attributes of the class in the correct
            format. See each validation method for more information. None values
            are allowed and will be replaced by the value of the attribute in
            the main object. When saved as an attribute, the value will be saved
            as a list with one item. If in the child class constructor an
            argument of the original class is not allowed, then it has to be
            passed as None in the super().__init__ call.

        Raises
        ------
        AssertionError
            If the input arguments do not conform to the specified formats.
        """
        self.object = object
        self.last_rnd_dict = {}

        for input_name, input_value in kwargs.items():
            if input_name not in self.exception_list:
                if input_value is not None:
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
                            raise AssertionError(
                                f"'{input_name}' must be a tuple, list, int, or float"
                            )
                else:
                    # if input_value is None, then the value will be taken from
                    # the main object and saved as a one item list.
                    attr_value = [getattr(self.object, input_name)]
                setattr(self, input_name, attr_value)

    def __str__(self):
        s = ""
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # Skip attributes starting with underscore
            if isinstance(value, tuple):
                # Format the tuple as a string with the mean and standard deviation.
                value_str = (
                    f"{value[0]:.5f} ± {value[1]:.5f} "
                    f"(numpy.random.{value[2].__name__})"
                )
            else:
                # Otherwise, just use the default string representation of the value.
                value_str = str(value)
            s += f"{key}: {value_str}\n"
        return s.strip()

    def __repr__(self):
        return f"{self.__class__.__name__}(object={self.object}, **kwargs)"

    def _validate_tuple(self, input_name, input_value, getattr=getattr):
        """Validator for tuple arguments. Checks if input is in a valid format.
        Tuples are validated as follows:
            - Must have length 2 or 3;
            - First item must be either an int or float;
            - If length is two, then the type of the second item must be either
              an int, float or str:
                - If the second item is an int or float, then it is assumed that
                  the first item is the nominal value and the second item is the
                  standard deviation;
                - If the second item is a string, then it is assumed that the
                  first item is the standard deviation, and the second item is
                  the distribution function string. In this case, the nominal
                  value will be taken from the main object;
            - If length is three, then it is assumed that the first item is the
              nominal value, the second item is the standard deviation and the
              third item is the distribution function string.

        Tuples are always saved as a tuple with length 3, where the first item
        is the nominal value, the second item is the standard deviation and the
        third item is the numpy distribution function.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple
            Value of the input argument.

        Returns
        -------
        tuple
            Tuple with length 3, where the first item is the nominal value, the
            second item is the standard deviation and the third item is the
            numpy distribution function.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        assert len(input_value) in [
            2,
            3,
        ], f"'{input_name}': tuple must have length 2 or 3"
        assert isinstance(
            input_value[0], (int, float)
        ), f"'{input_name}': First item of tuple must be either an int or float"

        if len(input_value) == 2:
            return self._validate_tuple_length_two(input_name, input_value, getattr)
        if len(input_value) == 3:
            return self._validate_tuple_length_three(input_name, input_value, getattr)

    def _validate_tuple_length_two(self, input_name, input_value, getattr=getattr):
        """Validator for tuples with length 2. Checks if input is in a valid
        format. If length is two, then the type of the second item must be
        either an int, float or str:

        - If the second item is an int or float, then it is assumed that the
          first item is the nominal value and the second item is the standard
          deviation;
        - If the second item is a string, then it is assumed that the first
          item is the standard deviation, and the second item is the
          distribution function string. In this case, the nominal value will
          be taken from the main object;

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple
            Value of the input argument.

        Returns
        -------
        tuple
            Tuple with length 3, where the first item is the nominal value, the
            second item is the standard deviation and the third item is the
            numpy distribution function.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        assert isinstance(
            input_value[1], (int, float, str)
        ), f"'{input_name}': second item of tuple must be an int, float, or string."

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

    def _validate_tuple_length_three(self, input_name, input_value, getattr=getattr):
        """Validator for tuples with length 3. Checks if input is in a valid
        format. If length is three, then it is assumed that the first item is
        the nominal value, the second item is the standard deviation and the
        third item is the distribution function string.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple
            Value of the input argument.

        Returns
        -------
        tuple
            Tuple with length 3, where the first item is the nominal value, the
            second item is the standard deviation and the third item is the
            numpy distribution function.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
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

    def _validate_list(self, input_name, input_value, getattr=getattr):
        """Validator for list arguments. Checks if input is in a valid format.
        Lists are validated as follows:
            - If the list is empty, then the value will be taken from the object
              passed and returned as a list with one item.
            - Else, the list is returned as is.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : list
            Value of the input argument.

        Returns
        -------
        list
            List with the input value.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        if not input_value:
            # if list is empty, then the value will be taken from the object
            # passed and saved as a list with one item.
            return [getattr(self.object, input_name)]
        else:
            # else, the list is saved as is.
            return input_value

    def _validate_scalar(self, input_name, input_value, getattr=getattr):
        """Validator for scalar arguments. Checks if input is in a valid format.
        Scalars are validated as follows:
            - The value is assumed to be the standard deviation, the nominal
              value will be taken from the object passed and the distribution
              function will be set to "normal".

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : float
            Value of the input argument.

        Returns
        -------
        tuple
            Tuple with length 3, where the first item is the nominal value, the
            second item is the standard deviation and the third item is the
            numpy distribution function.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        return (
            getattr(self.object, input_name),
            input_value,
            get_distribution("normal"),
        )

    def _validate_factors(self, input_name, input_value):
        """Validator for factor arguments. Checks if input is in a valid format.
        Factors can only be tuples of two or three items, or lists. Currently,
        the supported factors are: wind_velocity_x_factor,
        wind_velocity_y_factor, power_off_drag_factor, power_on_drag_factor.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple or list
            Value of the input argument.

        Returns
        -------
        tuple or list
            Tuple or list in the correct format.

        Raises
        ------
        AssertionError
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
        """Validator for tuple factors. Checks if input is in a valid format.
        Tuple factors can only have length 2 or 3. If length is two, then the
        type of the second item must be either an int, float or str. If length
        is three, then it is assumed that the first item is the nominal value,
        the second item is the standard deviation and the third item is the
        distribution function string.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        factor_tuple : tuple
            Value of the input argument.

        Returns
        -------
        tuple
            Tuple in the correct format.

        Raises
        ------
        AssertionError
            If input is not in a valid format.
        """
        assert len(factor_tuple) in [
            2,
            3,
        ], f"'{input_name}`: Factors tuple must have length 2 or 3"
        assert all(isinstance(item, (int, float)) for item in factor_tuple[:2]), (
            f"'{input_name}`: First and second items of Factors tuple must be "
            "either an int or float"
        )

        if len(factor_tuple) == 2:
            return (factor_tuple[0], factor_tuple[1], get_distribution("normal"))
        elif len(factor_tuple) == 3:
            assert isinstance(factor_tuple[2], str), (
                f"'{input_name}`: Third item of tuple must be a string containing "
                "the name of a valid numpy.random distribution function"
            )
            dist_func = get_distribution(factor_tuple[2])
            return (factor_tuple[0], factor_tuple[1], dist_func)

    def _validate_list_factor(self, input_name, factor_list):
        """Validator for list factors. Checks if input is in a valid format.
        List factors can only be lists of ints or floats.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        factor_list : list
            Value of the input argument.

        Returns
        -------
        list
            List in the correct format.

        Raises
        ------
        AssertionError
            If input is not in a valid format.
        """
        assert all(
            isinstance(item, (int, float)) for item in factor_list
        ), f"'{input_name}`: Items in list must be either ints or floats"
        return factor_list

    def _validate_1d_array_like(self, input_name, input_value):
        """Validator for 1D array like arguments. Checks if input is in a valid
        format. 1D array like arguments can only be lists of strings, lists of
        Functions, or lists of lists with shape (n,2).

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : list
            Value of the input argument.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If input is not in a valid format.
        """
        if input_value is not None:
            error_msg = (
                f"`{input_name}` must be a list of path strings, "
                + "lists with shape (n,2), or Functions."
            )

            # Inputs must always be a list
            if not isinstance(input_value, list):
                raise AssertionError(error_msg)

            for member in input_value:
                # if item is a list, then it must have shape (n,2)
                if isinstance(member, list):
                    if len(np.shape(member)) != 2 and np.shape(member)[1] != 2:
                        raise AssertionError(error_msg)

                # If item is not a string or Function, then raise error
                elif not isinstance(member, (str, Function)):
                    raise AssertionError(error_msg)

    def _validate_positive_int_list(self, input_name, input_value):
        """Validates the input argument: if it is not None, it must be a list
        of positive integers.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : list
            Value of the input argument.

        Raises
        ------
        AssertionError
            If input is not in a valid format.
        """
        if input_value is not None:
            assert isinstance(input_value, list) and all(
                isinstance(member, int) and member >= 0 for member in input_value
            ), f"`{input_name}` must be a list of positive integers"

    def _validate_airfoil(self, airfoil):
        """Validates the input argument: if it is not None, it must be a list
        of tuples with two items, where the first can be a 1D array like or
        a string, and the second item must be a string.

        Parameters
        ----------
        airfoil : list
            List of tuples with two items.

        Raises
        ------
        AssertionError
            If input is not in a valid format.
        """
        if airfoil is not None:
            assert isinstance(airfoil, list) and all(
                isinstance(member, tuple) for member in airfoil
            ), "`airfoil` must be a list of tuples"
            for member in airfoil:
                assert len(member) == 2, "`airfoil` tuples must have length 2"
                assert isinstance(
                    member[1], str
                ), "`airfoil` tuples must have a string as second item"
                # if item is a list, then it must have shape (n,2)
                if isinstance(member[0], list):
                    if len(np.shape(member[0])) != 2 and np.shape(member[0])[1] != 2:
                        raise AssertionError("`airfoil` tuples must have shape (n,2)")

                # If item is not a string or Function, then raise error
                elif not isinstance(member[0], (str, Function)):
                    raise AssertionError(
                        "`airfoil` tuples must have a string as first item"
                    )

    def dict_generator(self):
        """Generator that yields a dictionary with the randomly generated input
        arguments. The dictionary is saved as an attribute of the class.
        The dictionary is generated by looping through all attributes of the
        class and generating a random value for each attribute. The random
        values are generated according to the format of each attribute. Tuples
        are generated using the distribution function specified in the tuple.
        Lists are generated using the random.choice function.

        Parameters
        ----------
        None

        Yields
        -------
        dict
            Dictionary with the randomly generated input arguments.
        """
        generated_dict = {}
        for arg, value in self.__dict__.items():
            if isinstance(value, tuple):
                generated_dict[arg] = value[-1](value[0], value[1])
            elif isinstance(value, list):
                generated_dict[arg] = choice(value) if value else value
        self.last_rnd_dict = generated_dict
        yield generated_dict
