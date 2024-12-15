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
        cls.__validate__source(source)
        if inputs is None:
            inputs = ["Scalar"]
        if outputs is None:
            outputs = ["Scalar"]

        input_data = np.array([])
        output_data = np.array([])
        for lower, upper in sorted(source.keys()):
            grid = np.linspace(lower, upper, datapoints)

            # since intervals are disjoint and sorted, we only need to check
            # if the first point is already included
            if input_data.size != 0:
                if lower == input_data[-1]:
                    grid = np.delete(grid, 0)
            input_data = np.concatenate((input_data, grid))

            f = Function(source[(lower, upper)])
            output_data = np.concatenate((output_data, f.get_value(grid)))

        return Function(
            np.concatenate(([input_data], [output_data])).T,
            inputs=inputs,
            outputs=outputs,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

    @staticmethod
    def __validate__source(source):
        """Validates that source is dictionary with non-overlapping
        intervals

        Parameters
        ----------
        source : dict
            A dictionary of Function objects, where the keys are the domains.
        """
        # Check if source is a dictionary
        if not isinstance(source, dict):
            raise TypeError("source must be a dictionary")
        # Check if all keys are tuples
        for key in source.keys():
            if not isinstance(key, tuple):
                raise TypeError("keys of source must be tuples")
        # Check if all domains are disjoint
        for lower1, upper1 in source.keys():
            for lower2, upper2 in source.keys():
                if (lower1, upper1) != (lower2, upper2):
                    if lower1 < upper2 and upper1 > lower2:
                        raise ValueError("domains must be disjoint")
