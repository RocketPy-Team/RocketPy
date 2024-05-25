"""
Defines the `StochasticModel` class, which is used as a base class for all other
Stochastic classes.
"""

from random import choice

import numpy as np

from rocketpy.mathutils.function import Function

from ..tools import get_distribution

# TODO: Stop using assert in production code. Use exceptions instead.
# TODO: Each validation method should have a test case.
# TODO: elaborate a short, concise version of the __str__ method
# NOTE: I really think type hints would be quite useful to this class.


class StochasticModel:
    """
    Base class for all Stochastic classes. This class validates input arguments,
    saves them as attributes, and generates a dictionary with randomly generated
    input arguments.

    See also
    --------
    :ref:`Working with Stochastic Models <stochastic_usage>`

    Notes
    -----
    Please notice that the methods starting with an underscore are not meant to
    be called directly by the user. These methods may receive breaking changes
    without notice, so use them at your own risk.
    """

    # Arguments that are validated only in child classes
    exception_list = [
        "initial_solution",
        "terminate_on_apogee",
        "date",
        "ensemble_member",
    ]

    def __init__(self, object, **kwargs):
        """
        Initialize the StochasticModel class with validated input arguments.

        Parameters
        ----------
        object : object
            The main object of the class.
        **kwargs : dict
            Dictionary of input arguments for the class. Valid argument types
            include tuples, lists, ints, floats, or None. Arguments will be
            validated and saved as class attributes in a specific format, which
            is described in the
            ":ref:`Working with Stochastic Models <stochastic_usage>`" page.

        Raises
        ------
        AssertionError
            If the input arguments do not conform to the specified formats.
        """
        # TODO: don't use "object" as a variable name, it's a built-in function.
        #       We can simply change to "obj". Pylint W0622

        self.object = object
        self.last_rnd_dict = {}

        # TODO: This code block is too complex. Refactor it.
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
                    attr_value = [getattr(self.object, input_name)]
                setattr(self, input_name, attr_value)

    def __str__(self):
        """
        Returns a string representation of the StochasticModel object.

        Returns
        -------
        str
            String containing the class attributes and their values.
        """
        s = ""
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, tuple):
                value_str = (
                    f"{value[0]:.5f} ± {value[1]:.5f} "
                    f"(numpy.random.{value[2].__name__})"
                )
            else:
                value_str = str(value)
            s += f"{key}: {value_str}\n"
        return s.strip()

    def _validate_tuple(self, input_name, input_value, getattr=getattr):
        """
        Validate tuple arguments.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple
            Value of the input argument. This is the tuple to be validated.
        getattr : function
            Function used to get the attribute value from the object.

        Returns
        -------
        tuple
            Validated tuple in the format (nominal value, standard deviation, \
                distribution function).

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
        ), f"'{input_name}': First item of tuple must be an int or float"

        if len(input_value) == 2:
            return self._validate_tuple_length_two(input_name, input_value, getattr)
        if len(input_value) == 3:
            return self._validate_tuple_length_three(input_name, input_value, getattr)

    def _validate_tuple_length_two(self, input_name, input_value, getattr=getattr):
        """
        Validate tuples with length 2.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple
            Value of the input argument.
        getattr : function
            Function to get the attribute value from the object.

        Returns
        -------
        tuple
            Validated tuple in the format (nominal value, standard deviation, \
                distribution function).

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        # TODO: The `getattr` parameter is not used. Remove it.
        #       If it is important, don't use this name! it's a built-in function.
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
        """
        Validate tuples with length 3.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple
            Value of the input argument.
        getattr : function
            Function to get the attribute value from the object.

        Returns
        -------
        tuple
            Validated tuple in the format (nominal value, standard deviation, \
                distribution function).

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        assert isinstance(input_value[1], (int, float)), (
            f"'{input_name}': Second item of a tuple with length 3 must be an "
            "int or float."
        )
        assert isinstance(input_value[2], str), (
            f"'{input_name}': Third item of tuple must be a string containing the "
            "name of a valid numpy.random distribution function."
        )
        dist_func = get_distribution(input_value[2])
        return (input_value[0], input_value[1], dist_func)

    def _validate_list(self, input_name, input_value, getattr=getattr):
        """
        Validate list arguments.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : list
            Value of the input argument.
        getattr : function
            Function to get the attribute value from the object.

        Returns
        -------
        list
            Validated list.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        if not input_value:
            return [getattr(self.object, input_name)]
        else:
            return input_value

    def _validate_scalar(self, input_name, input_value, getattr=getattr):
        """
        Validate scalar arguments. If the input is a scalar, the nominal value
        will be taken from the object passed, and the standard deviation will be
        the scalar value. The distribution function will be set to "normal".

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : float
            Value of the input argument.
        getattr : function
            Function to get the attribute value from the object.

        Returns
        -------
        tuple
            Validated tuple in the format (nominal value, standard deviation, \
                distribution function).
        """
        return (
            getattr(self.object, input_name),
            input_value,
            get_distribution("normal"),
        )

    def _validate_factors(self, input_name, input_value):
        """
        Validate factor arguments.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : tuple or list
            Value of the input argument.

        Returns
        -------
        tuple or list
            Validated tuple or list.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        attribute_name = input_name.replace("_factor", "")
        setattr(self, f"_{attribute_name}", getattr(self.object, attribute_name))

        if isinstance(input_value, tuple):
            return self._validate_tuple_factor(input_name, input_value)
        elif isinstance(input_value, list):
            return self._validate_list_factor(input_name, input_value)
        else:
            raise AssertionError(f"`{input_name}`: must be either a tuple or list")

    def _validate_tuple_factor(self, input_name, factor_tuple):
        """
        Validate tuple factors.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        factor_tuple : tuple
            Value of the input argument.

        Returns
        -------
        tuple
            Validated tuple.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
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
        """
        Validate list factors.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        factor_list : list
            Value of the input argument.

        Returns
        -------
        list
            Validated list.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        assert all(
            isinstance(item, (int, float)) for item in factor_list
        ), f"'{input_name}`: Items in list must be either ints or floats"
        return factor_list

    def _validate_1d_array_like(self, input_name, input_value):
        """
        Validate 1D array-like arguments.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : list
            Value of the input argument.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        if input_value is not None:
            error_msg = (
                f"`{input_name}` must be a list of path strings, lists "
                "with shape (n,2), or Functions."
            )

            if not isinstance(input_value, list):
                raise AssertionError(error_msg)

            for member in input_value:
                if isinstance(member, list):
                    if len(np.shape(member)) != 2 or np.shape(member)[1] != 2:
                        raise AssertionError(error_msg)
                elif not isinstance(member, (str, Function)):
                    raise AssertionError(error_msg)

    def _validate_positive_int_list(self, input_name, input_value):
        """
        Validate lists of positive integers.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        input_value : list
            Value of the input argument.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        if input_value is not None:
            assert isinstance(input_value, list) and all(
                isinstance(member, int) and member >= 0 for member in input_value
            ), f"`{input_name}` must be a list of positive integers"

    def _validate_airfoil(self, airfoil):
        """
        Validate airfoil input.

        Parameters
        ----------
        airfoil : list
            List of tuples with two items.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        # TODO: The _validate_airfoil should be defined in a child class. No?
        if airfoil is not None:
            assert isinstance(airfoil, list) and all(
                isinstance(member, tuple) for member in airfoil
            ), "`airfoil` must be a list of tuples"
            for member in airfoil:
                assert len(member) == 2, "`airfoil` tuples must have length 2"
                assert isinstance(
                    member[1], str
                ), "`airfoil` tuples must have a string as the second item"
                if isinstance(member[0], list):
                    if len(np.shape(member[0])) != 2 and np.shape(member[0])[1] != 2:
                        raise AssertionError("`airfoil` tuples must have shape (n,2)")
                elif not isinstance(member[0], (str, Function)):
                    # TODO: Is the exception message correct? Please verify it.
                    raise AssertionError(
                        "`airfoil` tuples must have a string as the first item"
                    )

    def dict_generator(self):
        """
        Generate a dictionary with randomly generated input arguments.
        The last generated dictionary is saved as a class attribute called
        `last_rnd_dict`.

        Yields
        ------
        dict
            Dictionary with the randomly generated input arguments.

        Notes
        -----
        1. The dictionary is generated by iterating over the class attributes and:
            a. If the attribute is a tuple, the value is generated using the\
                distribution function specified in the tuple.
            b. If the attribute is a list, the value is randomly chosen from the list.
        """
        generated_dict = {}
        for arg, value in self.__dict__.items():
            if isinstance(value, tuple):
                generated_dict[arg] = value[-1](value[0], value[1])
            elif isinstance(value, list):
                generated_dict[arg] = choice(value) if value else value
        self.last_rnd_dict = generated_dict
        yield generated_dict
