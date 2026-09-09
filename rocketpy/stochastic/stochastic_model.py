"""
Defines the `StochasticModel` class, which is used as a base class for all other
Stochastic classes.
"""

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.stochastic.custom_sampler import CustomSampler

from ..tools import _seed_sequence_to_int, get_distribution


def _names_as_spawn_key(input_names):
    """Encode names into spawn-key words that no other set of names produces.

    A hash would be shorter, but a collision puts two samplers back on one
    stream, which is the bug this keying exists to prevent. The length prefix
    before each name is what makes it injective.
    """
    payload = b""
    for name in input_names:
        encoded = name.encode("utf-8")
        payload += len(encoded).to_bytes(4, "little") + encoded
    payload += b"\0" * (-len(payload) % 4)
    return tuple(
        int.from_bytes(payload[at : at + 4], "little")
        for at in range(0, len(payload), 4)
    )


def _format_number(value):
    """Format a nominal value or a standard deviation for the attribute report.

    An array-valued input, such as a fin outline, has no single number to show,
    and a fixed-width format raises a ``TypeError`` on it, so its shape stands
    in for the coordinates.
    """
    if np.ndim(value) == 0:
        return f"{value:.5f}"
    return f"array of shape {np.shape(value)}"


def _seed_as_entropy(seed):
    """A seed as something ``SeedSequence`` will take as entropy.

    A parallel run is handed a ``SeedSequence``, which it will not take. Any
    other seed goes through untouched, so the stream an int reaches stays where
    it was.
    """
    if not isinstance(seed, np.random.SeedSequence):
        return seed
    return _seed_sequence_to_int(seed)


def _sampler_seed(seed, input_names):
    """Derive a seed for one sampler, or for one group that shares a generator.

    Keyed by the names rather than by position, so declaring another parameter
    does not move the stream of the ones already there. A group is keyed by all
    of its members, so its stream does not depend on which of them happens to
    be reset last.
    """
    if isinstance(input_names, str):
        input_names = (input_names,)
    # Sorted here rather than trusting the caller, so a future call site cannot
    # give one group two different seeds by listing its members another way.
    root = np.random.SeedSequence(
        entropy=_seed_as_entropy(seed),
        spawn_key=_names_as_spawn_key(tuple(sorted(input_names))),
    )
    return _seed_sequence_to_int(root)


# TODO: Stop using assert in production code. Use exceptions instead.
# TODO: Each validation method should have a test case.


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

    # Arguments whose nominal value is an array of numbers rather than a single
    # number, such as the outline of a free-form fin. Declared by name so that
    # validation, sampling and the attribute report all agree on which ones they
    # are, instead of each deciding for itself.
    array_valued_inputs = ()

    def __init__(self, obj, seed=None, **kwargs):
        """
        Initialize the StochasticModel class with validated input arguments.

        Parameters
        ----------
        obj : object
            The main object of the class.
        seed : int, optional
            Seed for the random number generator. The default is None so that
            a new ``numpy.random.Generator`` object is created.
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

        self.obj = obj
        self.last_rnd_dict = {}
        self.__stochastic_dict = kwargs
        self._set_stochastic(seed)

    def _declare_stochastic_input(self, input_name, input_value):
        """Declare an input that an ``add_*`` method installs after ``__init__``.

        ``dict_generator`` walks the inputs a model declared rather than every
        attribute on it (#1109), and that list is built in ``__init__``. Anything
        added afterwards is set on the instance and never drawn from unless it
        says so here.

        The value is the argument as given, not the validated form, because
        ``_set_stochastic`` validates it again on every reseed and binds the
        distribution to the generator that is live then.
        """
        if input_value is None:
            return
        self.__stochastic_dict[input_name] = input_value

    def _set_stochastic(self, seed=None):
        """Set the stochastic attributes from the input dictionary.
        This method is useful to reset or reseed the attributes of the instance.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.
        """
        self.__random_number_generator = np.random.default_rng(seed)
        # A stream of its own, derived from the same seed, for picking between
        # the candidate values of a list input. Kept apart from the one above so
        # that declaring a list input does not shift the numbers every other
        # input draws, which would move each existing fixed-seed baseline.
        self.__choice_generator = np.random.default_rng(
            _sampler_seed(seed, ("__list_choice__",))
        )
        self.last_rnd_dict = {}

        self._reset_custom_samplers(seed)

        # TODO: This code block is too complex. Refactor it.
        # TODO: Resetting a instance should not require re-validation.
        for input_name, input_value in self.__stochastic_dict.items():
            if input_name not in self.exception_list:
                attr_value = None
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
                        elif isinstance(input_value, CustomSampler):
                            attr_value = self._validate_custom_sampler(
                                input_name, input_value
                            )
                        else:
                            raise AssertionError(
                                f"'{input_name}' must be a tuple, list, int, or float"
                                "or a custom sampler"
                            )
                else:
                    attr_value = [getattr(self.obj, input_name)]
                setattr(self, input_name, attr_value)

    def __repr__(self):
        return f"'{self.__class__.__name__}() object'"

    def _choose(self, values):
        """Pick one of the candidate values of a list input.

        ``random.choice`` was used here, which draws from the interpreter-wide
        stream that ``_set_stochastic`` does not reseed: the same seed did not
        reproduce the same choices, and Monte Carlo workers forked from one
        process inherited a single stream and walked it together instead of
        sampling independently.

        Parameters
        ----------
        values : list
            Candidate values of the input.

        Returns
        -------
        object
            One of the candidates, or ``values`` itself when there are none.
        """
        if len(values) == 0:
            return values
        return values[self.__choice_generator.integers(len(values))]

    def _nominal_value(self, input_name, value):
        """Return the nominal value of an input as the distribution needs it.

        The distributions are called as ``dist_func(nominal, std_dev)``, so an
        array-valued input has to arrive as an array for the deviation to
        broadcast over its entries. A list of ``(x, y)`` tuples, which is how a
        fin outline is written, would not.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        value : object
            Nominal value of the input argument.

        Returns
        -------
        object
            The value, as an array of floats for the array-valued inputs.
        """
        if input_name in self.array_valued_inputs:
            return np.asarray(value, dtype=float)
        return value

    def _validate_tuple(self, input_name, input_value, getattr=getattr):  # pylint: disable=redefined-builtin
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
        if len(input_value) not in [
            2,
            3,
        ]:
            raise AssertionError(f"'{input_name}': tuple must have length 2 or 3")
        if not isinstance(input_value[0], (int, float)):
            if input_name not in self.array_valued_inputs:
                raise AssertionError(
                    f"'{input_name}': First item of tuple must be an int or float"
                )
            # An array-valued input carries its whole nominal value here, so the
            # single number the others require is not what to expect. The child
            # class that declared it has already checked the value itself.
            input_value = (self._nominal_value(input_name, input_value[0]),) + tuple(
                input_value[1:]
            )

        if len(input_value) == 2:
            return self._validate_tuple_length_two(input_name, input_value, getattr)
        if len(input_value) == 3:
            return self._validate_tuple_length_three(input_name, input_value, getattr)

    def _validate_tuple_length_two(self, input_name, input_value, getattr=getattr):  # pylint: disable=redefined-builtin
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
        if not isinstance(input_value[1], (int, float, str)):
            raise AssertionError(
                f"'{input_name}': second item of tuple must be an int, float, or string."
            )

        if isinstance(input_value[1], str):
            # if second item is a string, then it is assumed that the first item
            # is the standard deviation, and the second item is the distribution
            # function. In this case, the nominal value will be taken from the
            # object passed.
            dist_func = get_distribution(input_value[1], self.__random_number_generator)
            return (
                self._nominal_value(input_name, getattr(self.obj, input_name)),
                input_value[0],
                dist_func,
            )
        else:
            # if second item is an int or float, then it is assumed that the
            # first item is the nominal value and the second item is the
            # standard deviation. The distribution function will be set to
            # "normal".
            return (
                input_value[0],
                input_value[1],
                get_distribution("normal", self.__random_number_generator),
            )

    def _validate_tuple_length_three(self, input_name, input_value, getattr=getattr):  # pylint: disable=redefined-builtin,unused-argument
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
        if not isinstance(input_value[1], (int, float)):
            raise AssertionError(
                f"'{input_name}': Second item of a tuple with length 3 must be an "
                "int or float."
            )
        if not isinstance(input_value[2], str):
            raise AssertionError(
                f"'{input_name}': Third item of tuple must be a string containing the "
                "name of a valid numpy.random distribution function."
            )
        dist_func = get_distribution(input_value[2], self.__random_number_generator)
        return (input_value[0], input_value[1], dist_func)

    def _validate_list(self, input_name, input_value, getattr=getattr):  # pylint: disable=redefined-builtin
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
            return [getattr(self.obj, input_name)]
        else:
            return input_value

    def _validate_scalar(self, input_name, input_value, getattr=getattr):  # pylint: disable=redefined-builtin
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
            self._nominal_value(input_name, getattr(self.obj, input_name)),
            input_value,
            get_distribution("normal", self.__random_number_generator),
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
        setattr(self, f"_{attribute_name}", getattr(self.obj, attribute_name))

        if isinstance(input_value, tuple):
            return self._validate_tuple_factor(input_name, input_value)
        elif isinstance(input_value, list):
            return self._validate_list_factor(input_name, input_value)
        elif isinstance(input_value, CustomSampler):
            return self._validate_custom_sampler(input_name, input_value)
        else:
            raise AssertionError(
                f"`{input_name}`: must be either a tuple or listor a custom sampler"
            )

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
        if len(factor_tuple) not in [
            2,
            3,
        ]:
            raise AssertionError(
                f"'{input_name}`: Factors tuple must have length 2 or 3"
            )
        if not all(isinstance(item, (int, float)) for item in factor_tuple[:2]):
            raise AssertionError(
                f"'{input_name}`: First and second items of Factors tuple must be "
                "either an int or float"
            )

        if len(factor_tuple) == 2:
            return (
                factor_tuple[0],
                factor_tuple[1],
                get_distribution("normal", self.__random_number_generator),
            )
        elif len(factor_tuple) == 3:
            if not isinstance(factor_tuple[2], str):
                raise AssertionError(
                    f"'{input_name}`: Third item of tuple must be a string containing "
                    "the name of a valid numpy.random distribution function"
                )
            dist_func = get_distribution(
                factor_tuple[2], self.__random_number_generator
            )
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
        if not all(isinstance(item, (int, float)) for item in factor_list):
            raise AssertionError(
                f"'{input_name}`: Items in list must be either ints or floats"
            )
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
            if not (
                isinstance(input_value, list)
                and all(
                    isinstance(member, int) and member >= 0 for member in input_value
                )
            ):
                raise AssertionError(
                    f"`{input_name}` must be a list of positive integers"
                )

    def _reset_custom_samplers(self, seed):
        """Seed each sampler, and each shared generator once.

        Kept out of the validation loop in ``_set_stochastic``, whose order
        sets ``__dict__`` and with it the order every other input is drawn in.
        """
        groups = {}
        for input_name in sorted(self.__stochastic_dict):
            sampler = self.__stochastic_dict[input_name]
            if isinstance(sampler, CustomSampler):
                # Kept in the value too: `id` is unique only among live
                # objects, so the group has to outlive the dict.
                group = sampler.seed_group
                shared = groups.setdefault(id(group), ([], sampler, group))
                shared[0].append(input_name)

        for names, sampler, group in groups.values():
            # The group holds the shared state, so reset it directly; a member
            # may reset differently, or keep state of its own.
            resetter = group if hasattr(group, "reset_seed") else sampler
            try:
                resetter.reset_seed(_sampler_seed(seed, names))
            except Exception as error:
                # Broad: the seed is 128 bits, which legacy RandomState refuses
                # with a ValueError that does not name the sampler.
                raise RuntimeError(
                    f"An error occurred in the 'reset_seed' method of the "
                    f"CustomSampler for {', '.join(names)}"
                ) from error

    def _validate_custom_sampler(self, input_name, sampler):
        """
        Validate a custom sampler.

        Seeding happens in ``_reset_custom_samplers``, not here.

        Parameters
        ----------
        input_name : str
            Name of the input argument.
        sampler : CustomSampler object
            Custom sampler provided by the user

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        # Raised, not asserted: `python -O` strips asserts. AssertionError is
        # kept so callers that catch it still do. Same as #1103.
        if not isinstance(sampler, CustomSampler):
            raise AssertionError(
                f"`{input_name}` must be a CustomSampler, not {type(sampler).__name__}"
            )
        return sampler

    def _validate_airfoil(self, airfoil):
        """
        Validate airfoil input.

        Parameters
        ----------
        airfoil : list[tuple]
            List of tuples with two items.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        # TODO: The _validate_airfoil should be defined in a child class.
        if airfoil is not None:
            if not (
                isinstance(airfoil, list)
                and all(isinstance(member, tuple) for member in airfoil)
            ):
                raise AssertionError("`airfoil` must be a list of tuples")
            for member in airfoil:
                if not len(member) == 2:
                    raise AssertionError("`airfoil` tuples must have length 2")
                if not isinstance(member[1], str):
                    raise AssertionError(
                        "`airfoil` tuples must have a string as the second item"
                    )
                if isinstance(member[0], list):
                    if len(np.shape(member[0])) != 2 and np.shape(member[0])[1] != 2:
                        raise AssertionError("`airfoil` tuples must have shape (n,2)")
                elif not isinstance(member[0], str) and not callable(member[0]):
                    raise AssertionError(
                        "`airfoil` tuples must have a string or Function as "
                        "the first item"
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
        1. The dictionary is generated by iterating over the *declared*
           stochastic inputs (constructor kwargs), not every attribute on
           ``self``. This avoids treating opaque tuples such as
           ``initial_solution`` as ``(nominal, spread, sampler)`` triples.

           a. If the attribute is a tuple, the value is generated using the
              distribution function specified in the tuple.
           b. If the attribute is a list, the value is randomly chosen from
              the list.
        """
        generated_dict = {}
        for arg in self.__stochastic_dict:
            if not hasattr(self, arg):
                continue
            value = getattr(self, arg)
            if isinstance(value, tuple):
                dist_sampler = value[-1]
                generated_dict[arg] = dist_sampler(value[0], value[1])
            elif isinstance(value, list):
                generated_dict[arg] = self._choose(value)
            elif isinstance(value, CustomSampler):
                try:
                    generated_dict[arg] = value.sample(n_samples=1)[0]
                except RuntimeError as e:
                    raise RuntimeError(
                        f"An error occurred in the 'sample' method of {arg} CustomSampler"
                    ) from e
        self.last_rnd_dict = generated_dict
        yield generated_dict

    # pylint: disable=too-many-statements
    def visualize_attributes(self):
        """
        This method prints a report of the attributes stored in the Stochastic
        Model object. The report includes the variable name, the nominal value,
        the standard deviation, and the distribution function used to generate
        the random attributes.

        Returns
        -------
        str
            The formatted report. It is also printed for interactive use.
        """

        def format_attribute(attr, value):
            if isinstance(value, list):
                return (
                    f"\t{attr.ljust(max_str_length)} {value[0]}"
                    if len(value) == 1
                    else f"\t{attr} {value}"
                )
            elif isinstance(value, tuple):
                nominal_value, std_dev, dist_func = value
                if callable(dist_func) and dist_func.__name__ == "uniform":
                    lower_bound = nominal_value
                    upper_bound = std_dev
                    return (
                        f"\t{attr.ljust(max_str_length)} "
                        f"{_format_number(lower_bound)}, "
                        f"{_format_number(upper_bound)} ({dist_func.__name__})"
                    )
                else:
                    return (
                        f"\t{attr.ljust(max_str_length)} "
                        f"{_format_number(nominal_value)} ± "
                        f"{_format_number(std_dev)} ({dist_func.__name__})"
                    )
            elif isinstance(value, CustomSampler):
                sampler_name = type(value).__name__
                return (
                    f"\t{attr.ljust(max_str_length)} "
                    f"\t{sampler_name.ljust(max_str_length)} "
                )
            return None

        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        max_str_length = max(len(var) for var in attributes) + 2

        report = [
            f"Reporting the attributes of the `{self.__class__.__name__}` object:"
        ]

        # Sorting alphabetically makes the report more readable
        items = attributes.items()
        items = sorted(items, key=lambda x: x[0])

        to_exclude = ["object", "last_rnd_dict", "exception_list", "parachutes"]
        items = [item for item in items if item[0] not in to_exclude]

        constant_attributes = [
            attr for attr, val in items if isinstance(val, list) and len(val) == 1
        ]
        tuple_attributes = [attr for attr, val in items if isinstance(val, tuple)]
        list_attributes = [
            attr for attr, val in items if isinstance(val, list) and len(val) > 1
        ]
        custom_attributes = [
            attr for attr, val in items if isinstance(val, CustomSampler)
        ]

        if constant_attributes:
            report.append("\nConstant Attributes:")
            report.extend(
                format_attribute(attr, attributes[attr]) for attr in constant_attributes
            )

        if tuple_attributes:
            report.append("\nStochastic Attributes:")
            report.extend(
                format_attribute(attr, attributes[attr]) for attr in tuple_attributes
            )

        if list_attributes:
            report.append("\nStochastic Attributes with choice of values:")
            report.extend(
                format_attribute(attr, attributes[attr]) for attr in list_attributes
            )
        if custom_attributes:
            report.append("\nStochastic Attributes with Custom user samplers:")
            report.extend(
                format_attribute(attr, attributes[attr]) for attr in custom_attributes
            )

        # This is an explicit, user-invoked display method, so it prints
        # unconditionally (like ``info``/``all_info`` elsewhere) rather than
        # logging at INFO level, which is silenced by default. The report is
        # also returned so it can be used programmatically.
        report_str = "\n".join(filter(None, report))
        print(report_str)
        return report_str
