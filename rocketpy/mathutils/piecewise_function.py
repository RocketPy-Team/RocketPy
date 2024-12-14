import numpy as np

from rocketpy.mathutils.function import Function


class PiecewiseFunction(Function):
    """Class for creating piecewise functions. These kind of functions are
    defined by a dictionary of functions, where the keys are tuples that
    represent the domain of the function. The domains must be disjoint.
    """

    def __new__(
        cls,
        source,
        inputs=None,
        outputs=None,
        interpolation="spline",
        extrapolation=None,
        datapoints=100,
    ):
        """
        Creates a piecewise function from a dictionary of functions. The keys of
        the dictionary must be tuples that represent the domain of the function.
        The domains must be disjoint. The piecewise function will be evaluated
        at datapoints points to create Function object.

        Parameters
        ----------
        source: dictionary
            A dictionary of Function objects, where the keys are the domains.
        inputs : list of strings
            A list of strings that represent the inputs of the function.
        outputs: list of strings
            A list of strings that represent the outputs of the function.
        interpolation: str
            The type of interpolation to use. The default value is 'spline'.
        extrapolation: str
            The type of extrapolation to use. The default value is None.
        datapoints: int
            The number of points in which the piecewise function will be
            evaluated to create a base function. The default value is 100.
        """
        if inputs is None:
            inputs = ["Scalar"]
        if outputs is None:
            outputs = ["Scalar"]
        # Check if source is a dictionary
        if not isinstance(source, dict):
            raise TypeError("source must be a dictionary")
        # Check if all keys are tuples
        for key in source.keys():
            if not isinstance(key, tuple):
                raise TypeError("keys of source must be tuples")
        # Check if all domains are disjoint
        for key1 in source.keys():
            for key2 in source.keys():
                if key1 != key2:
                    if key1[0] < key2[1] and key1[1] > key2[0]:
                        raise ValueError("domains must be disjoint")

        # Crate Function
        def calc_output(func, inputs):
            """Receives a list of inputs value and a function, populates another
            list with the results corresponding to the same results.

            Parameters
            ----------
            func : Function
                The Function object to be
            inputs : list, tuple, np.array
                The array of points to applied the func to.

            Examples
            --------
            >>> inputs = [0, 1, 2, 3, 4, 5]
            >>> def func(x):
            ...     return x*10
            >>> calc_output(func, inputs)
            [0, 10, 20, 30, 40, 50]

            Notes
            -----
            In the future, consider using the built-in map function from python.
            """
            output = np.zeros(len(inputs))
            for j, value in enumerate(inputs):
                output[j] = func.get_value_opt(value)
            return output

        input_data = []
        output_data = []
        for key in sorted(source.keys()):
            i = np.linspace(key[0], key[1], datapoints)
            i = i[~np.isin(i, input_data)]
            input_data = np.concatenate((input_data, i))

            f = Function(source[key])
            output_data = np.concatenate((output_data, calc_output(f, i)))

        return Function(
            np.concatenate(([input_data], [output_data])).T,
            inputs=inputs,
            outputs=outputs,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )
