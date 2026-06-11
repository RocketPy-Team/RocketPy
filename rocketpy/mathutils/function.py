# pylint: disable=too-many-lines
"""The mathutils/function.py is a rocketpy module totally dedicated to function
operations, including interpolation, extrapolation, integration, differentiation
and more. This is a core class of our package, and should be maintained
carefully as it may impact all the rest of the project.
"""

import logging
import operator
import warnings
from collections.abc import Iterable
from copy import deepcopy
from enum import Enum
from inspect import signature
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

from rocketpy.mathutils._calc import build_interpolation_evaluator
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

from rocketpy.plots.plot_helpers import show_or_save_plot
from rocketpy.tools import deprecated, from_hex_decode, to_hex_encode
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

NUMERICAL_TYPES = (float, int, complex, np.integer, np.floating)
_LIST_VECTORIZE_THRESHOLD = 16


def _safe_truediv(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
        return np.nan_to_num(result)


_OPERATOR_SYMBOLS = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    _safe_truediv: "/",
    operator.pow: "**",
    operator.mod: "%",
}


class SourceType(Enum):
    """Enumeration of the source types for the Function class.
    The source can be a callable, an array, or a scalar constant.
    """

    CALLABLE = 0
    ARRAY = 1
    SCALAR = 2


class Function:  # pylint: disable=too-many-public-methods
    """Class converts a python function or a data sequence into an object
    which can be handled more naturally, enabling easy interpolation,
    extrapolation, plotting and algebra.
    """

    # Arithmetic priority
    __array_ufunc__ = None

    def __init__(
        self,
        source,
        inputs=None,
        outputs=None,
        interpolation=None,
        extrapolation=None,
        title=None,
    ):
        """Convert source into a Function, to be used more naturally.
        Set inputs, outputs, domain dimension, interpolation and extrapolation
        method, and process the source.

        Parameters
        ----------
        source : callable, scalar, ndarray, string, or Function
            The data source to be used for the function:

            - ``Callable``: Called for evaluation with input values. Must have \
                the desired inputs as arguments and return a single output \
                value. Input order is important. Example: Python functions.
            - ``int`` or ``float``: Treated as a constant value function.
            - ``np.ndarray``: Used for interpolation. Format as [(x0, y0, z0), \
            (x1, y1, z1), ..., (xn, yn, zn)], where 'x' and 'y' are inputs, \
            and 'z' is the output.
            - ``str``: Path to a CSV file. The file is read and converted into an \
            ndarray. The file can optionally contain a single header line.
            - ``Function``: Copies the source of the provided Function object, \
            creating a new Function with adjusted inputs and outputs.

        inputs : string, sequence of strings, optional
            The name of the inputs of the function. Will be used for
            representation and graphing (axis names). 'Scalar' is default.
            If source is function, int or float and has multiple inputs,
            this parameter must be given for correct operation.
        outputs : string, sequence of strings, optional
            The name of the outputs of the function. Will be used for
            representation and graphing (axis names). Scalar is default.
        interpolation : string, optional
            Interpolation method to be used if source type is ndarray.
            For 1-D functions, linear, polynomial, akima and spline are
            supported. For N-D functions, linear, shepard, rbf and
            regular_grid are supported.
            Default for 1-D functions is spline and for N-D functions is
            shepard.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function at the nearest edge of
            the domain, and 'zero', which returns zero for all points outside
            of source range.
            Multidimensional 'natural' extrapolation for 'linear' interpolation
            use a 'rbf' algorithm for smoother results.
            Default for 1-D functions is constant and for N-D functions
            is natural.
        title : string, optional
            Title to be displayed in the plots' figures. If none, the title will
            be constructed using the inputs and outputs arguments in the form
            of  "{inputs} x {outputs}".

        Returns
        -------
        None

        Notes
        -----
        (I) CSV files may include an optional single header line. If this
        header line is present and contains names for each data column, those
        names will be used to label the inputs and outputs unless specified
        otherwise by the `inputs` and `outputs` arguments.
        If the header is specified for only a few columns, it is ignored.

        Commas in a header will be interpreted as a delimiter, which may cause
        undesired input or output labeling. To avoid this, specify each input
        and output name using the `inputs` and `outputs` arguments.

        (II) Fields in CSV files may be enclosed in double quotes. If fields
        are not quoted, double quotes should not appear inside them.
        """
        self.source = source
        self.__inputs__ = inputs
        self.__outputs__ = outputs
        self.__interpolation__ = interpolation
        self.__extrapolation__ = extrapolation
        self.title = title
        self.__img_dim__ = 1  # always 1, here for backwards compatibility
        self.__cropped_domain__ = None

        # args must be passed from self.
        self.set_source(self.source)
        self.set_inputs(self.__inputs__)
        self.set_outputs(self.__outputs__)
        self.set_title(self.title)

    @classmethod
    def from_regular_grid_csv(
        cls, csv_source, variable_names, coeff_name, extrapolation
    ):
        """Create a regular-grid Function from CSV samples when possible.

        Parameters
        ----------
        csv_source : str
            Path to the CSV file.
        variable_names : list[str]
            Ordered independent variable names present in the CSV.
        coeff_name : str
            Name of the output coefficient.
        extrapolation : str
            Extrapolation method passed to the Function constructor.

        Returns
        -------
        Function or None
            A ``Function`` configured with ``regular_grid`` interpolation when
            the CSV forms a strict Cartesian grid, otherwise ``None``.
        """
        try:
            data = np.loadtxt(csv_source, delimiter=",", skiprows=1, dtype=np.float64)
        except (OSError, ValueError):
            return None

        data = np.atleast_2d(data)
        expected_columns = len(variable_names) + 1
        if data.shape[1] != expected_columns:
            return None

        coordinates = data[:, :-1]
        values = data[:, -1]

        if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
            return None

        axes = [np.unique(coordinates[:, i]) for i in range(len(variable_names))]
        expected_size = int(np.prod([axis.size for axis in axes]))
        if expected_size != coordinates.shape[0]:
            return None

        sorting_keys = [
            coordinates[:, i] for i in range(len(variable_names) - 1, -1, -1)
        ]
        sorted_indices = np.lexsort(tuple(sorting_keys))
        sorted_coordinates = coordinates[sorted_indices]
        sorted_values = values[sorted_indices]

        expected_coordinates = np.column_stack(
            [axis_values.ravel() for axis_values in np.meshgrid(*axes, indexing="ij")]
        )
        if not np.allclose(
            sorted_coordinates, expected_coordinates, rtol=0, atol=1e-12
        ):
            return None

        grid_data = sorted_values.reshape(tuple(axis.size for axis in axes))
        return cls(
            (axes, grid_data),
            inputs=variable_names,
            outputs=[coeff_name],
            interpolation="regular_grid",
            extrapolation=extrapolation,
        )

    # Define all set methods
    def set_inputs(self, inputs):
        """Set the name and number of the incoming arguments of the Function.

        Parameters
        ----------
        inputs : string, sequence of strings
            The name of the parameters (inputs) of the Function.

        Returns
        -------
        self : Function
        """
        self.__inputs__ = self.__validate_inputs(inputs)
        return self

    def set_outputs(self, outputs):
        """Set the name and number of the output of the Function.

        Parameters
        ----------
        outputs : string, sequence of strings
            The name of the output of the function. Example: Distance (m).

        Returns
        -------
        self : Function
        """
        self.__outputs__ = self.__validate_outputs(outputs)
        return self

    def set_source(self, source):  # pylint: disable=too-many-statements
        """Sets the data source for the function, defining how the function
        produces output from a given input.

        Parameters
        ----------
        source : callable, scalar, ndarray, string, or Function
            The data source to be used for the function:

            - ``Callable``: Called for evaluation with input values. Must have \
                the desired inputs as arguments and return a single output \
                value. Input order is important. Example: Python functions.
            - ``int`` or ``float``: Treated as a constant value function.
            - ``np.ndarray``: Used for interpolation. Format as [(x0, y0, z0), \
            (x1, y1, z1), ..., (xn, yn, zn)], where 'x' and 'y' are inputs, \
            and 'z' is the output.
            - ``str``: Path to a CSV file. The file is read and converted into an \
            ndarray. The file can optionally contain a single header line.
            - ``Function``: Copies the source of the provided Function object, \
            creating a new Function with adjusted inputs and outputs.

        Notes
        -----
        (I) **CSV files may include an optional single header line**: \
            If this header line is present and contains names for each data \
            column, those names will be used to label the inputs and outputs \
            unless specified otherwise. If the header is specified for only a \
            few columns, it is ignored.

        (II) **Commas in a header will be interpreted as a delimiter**: \
            this may cause undesired input or output labeling. To avoid this, \
            specify each input and output name using the `inputs` and `outputs` \
            arguments.

        (III) **Fields in CSV files may be enclosed in double quotes**: \
            If fields are not quoted, double quotes should not appear inside them.

        Returns
        -------
        self : Function
            Returns the Function instance with the new source set.
        """
        source = self.__validate_source(source)

        # Handle scalar (constant) source
        if isinstance(source, NUMERICAL_TYPES):
            self._source_type = SourceType.SCALAR
            self._scalar_value = source
            self.__dom_dim__ = 1
            self.__interpolation__ = None
            self.__extrapolation__ = None

        # Handle callable source
        elif callable(source):
            self._source_type = SourceType.CALLABLE
            self.__interpolation__ = None
            self.__extrapolation__ = None

            # Set arguments name and domain dimensions
            if hasattr(source, "__dom_dim__"):
                self.__dom_dim__ = source.__dom_dim__
            else:
                parameters = signature(source).parameters
                self.__dom_dim__ = len(parameters)
                if self.__inputs__ is None:
                    self.__inputs__ = list(parameters)

        # Handle ndarray source
        else:
            self._source_type = SourceType.ARRAY
            # Evaluate dimension
            self.__dom_dim__ = source.shape[1] - 1
            self._domain = source[:, :-1]
            self._image = source[:, -1]

            # set x and y. If Function is 2D, also set z
            if self.__dom_dim__ == 1:
                source = source[source[:, 0].argsort()]
                self.x_array = source[:, 0]
                self.x_initial, self.x_final = self.x_array[0], self.x_array[-1]
                self.y_array = source[:, 1]
                self.y_initial, self.y_final = self.y_array[0], self.y_array[-1]
            elif self.__dom_dim__ > 1:
                self.x_array = source[:, 0]
                self.x_initial, self.x_final = self.x_array[0], self.x_array[-1]
                self.y_array = source[:, 1]
                self.y_initial, self.y_final = self.y_array[0], self.y_array[-1]
                self.z_array = source[:, 2]
                self.z_initial, self.z_final = self.z_array[0], self.z_array[-1]

        self.source = source
        self.set_interpolation(self.__interpolation__)
        self.set_extrapolation(self.__extrapolation__)
        self.set_get_value_opt()
        return self

    @property
    def min(self):
        """Get the minimum value of the Function y_array.
        Raises an error if the Function is lambda based.

        Returns
        -------
        minimum : float
        """
        return self.y_array.min()

    @property
    def max(self):
        """Get the maximum value of the Function y_array.
        Raises an error if the Function is lambda based.

        Returns
        -------
        maximum : float
        """
        return self.y_array.max()

    def set_interpolation(self, method="spline"):
        """Set interpolation method and process data is method requires.

        Parameters
        ----------
        method : string, optional
            Interpolation method to be used if source type is ndarray.
            For 1-D functions, linear, polynomial, akima and spline is
            supported. For N-D functions, linear, shepard, rbf and
            regular_grid are supported.
            Default for 1-D functions is spline and for N-D functions is
            shepard.

        Returns
        -------
        self : Function
        """
        if self._source_type is SourceType.ARRAY:
            self.__interpolation__ = self.__validate_interpolation(method)
            self._build_interp_extrap()
        return self

    def set_extrapolation(self, method="constant"):
        """Set extrapolation behavior of data set.

        Parameters
        ----------
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function at the nearest edge of
            the domain, and 'zero', which returns zero for all points outside
            of source range.
            Multidimensional 'natural' extrapolation for 'linear' interpolation
            use a 'rbf' algorithm for smoother results.
            Default for 1-D functions is constant and for N-D functions
            is natural.

        Returns
        -------
        self : Function
            The Function object.
        """
        if self._source_type is SourceType.ARRAY:
            self.__extrapolation__ = self.__validate_extrapolation(method)
            self._build_interp_extrap()
        return self

    def __process_grid_source(self, source):
        """Validate and process a ``(axes, grid_data)`` tuple into a flat
        scatter :class:`numpy.ndarray` ready for :meth:`set_source`.

        As a side-effect, stores ``self._grid_axes`` and ``self._grid_data``
        so that :meth:`__set_interpolation_func` (case 6) and
        :meth:`__set_extrapolation_func` can build the
        :class:`~scipy.interpolate.RegularGridInterpolator`.

        Parameters
        ----------
        source : tuple
            A 2-element tuple ``(axes, grid_data)`` where *axes* is a list of
            1-D arrays sorted in ascending order (one per input dimension) and
            *grid_data* is a matching N-dimensional :class:`numpy.ndarray` of
            values.

        Returns
        -------
        flat_source : numpy.ndarray
            Array of shape ``(n_points, n_dims + 1)`` with all grid points
            unrolled in row-major (C) order.

        Raises
        ------
        ValueError
            If *source* is not a 2-element tuple, if the number of axes
            mismatches the grid dimensionality, or if an axis length mismatches
            the corresponding grid dimension.
        """
        if not (isinstance(source, Iterable) and len(source) == 2):
            raise ValueError(
                "For 'regular_grid' interpolation, source must be a "
                "(axes, grid_data) tuple where axes is a list of 1-D arrays "
                "and grid_data is a matching N-dimensional ndarray."
            )

        raw_axes, raw_data = source
        if not isinstance(raw_axes, Iterable):
            raise ValueError(
                "The first element of the source tuple must be a list or tuple "
                "of 1-D arrays representing the grid axes."
            )

        axes = [np.asarray(ax) for ax in raw_axes]
        grid_data = np.asarray(raw_data, dtype=np.float64)

        if len(axes) != grid_data.ndim:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match grid_data dimensions "
                f"({grid_data.ndim})."
            )
        for i, ax in enumerate(axes):
            if len(ax) != grid_data.shape[i]:
                raise ValueError(
                    f"Axis {i} has {len(ax)} points but grid dimension {i} has "
                    f"{grid_data.shape[i]} points."
                )
            if not np.all(np.diff(ax) > 0):
                warnings.warn(
                    f"Axis {i} is not strictly sorted in ascending order. "
                    "RegularGridInterpolator requires sorted axes.",
                    UserWarning,
                )

        self._grid_axes = axes
        self._grid_data = grid_data

        mesh = np.meshgrid(*axes, indexing="ij")
        domain_points = np.column_stack([m.ravel() for m in mesh])
        return np.column_stack([domain_points, grid_data.ravel()])

    def __set_interpolation_func(self):  # pylint: disable=too-many-statements
        """Defines interpolation function used by the Function. Each
        interpolation method has its own function`.
        The function is stored in the attribute _interpolation_func."""
        interpolation = INTERPOLATION_TYPES[self.__interpolation__]
        match interpolation:
            case 0:  # linear
                if self.__dom_dim__ == 1:

                    def linear_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                        x_interval = bisect_left(x_data, x)
                        x_left = x_data[x_interval - 1]
                        y_left = y_data[x_interval - 1]
                        dx = float(x_data[x_interval] - x_left)
                        dy = float(y_data[x_interval] - y_left)
                        return (x - x_left) * (dy / dx) + y_left

                else:
                    tri = Delaunay(self._domain)
                    interpolator = LinearNDInterpolator(tri, self._image)
                    self._nd_triangulation = tri

                    def linear_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                        return interpolator(x)

                self._interpolation_func = linear_interpolation

            case 1:  # polynomial

                def polynomial_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                    return np.sum(coeffs * x ** np.arange(len(coeffs)))

                self._interpolation_func = polynomial_interpolation

            case 2:  # akima

                def akima_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                    x_interval = bisect_left(x_data, x)
                    x_interval = x_interval if x_interval != 0 else 1
                    a = coeffs[4 * x_interval - 4 : 4 * x_interval]
                    return a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]

                self._interpolation_func = akima_interpolation

            case 3:  # spline

                def spline_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                    x_interval = bisect_left(x_data, x)
                    x_interval = max(x_interval, 1)
                    a = coeffs[:, x_interval - 1]
                    x = x - x_data[x_interval - 1]
                    return a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]

                self._interpolation_func = spline_interpolation

            case 4:  # shepard
                # pylint: disable=unused-argument
                def shepard_interpolation(x, x_min, x_max, x_data, y_data, _):
                    arg_qty, arg_dim = x.shape
                    result = np.empty(arg_qty)
                    x = x.reshape((arg_qty, 1, arg_dim))
                    sub_matrix = x_data - x
                    distances_squared = np.sum(sub_matrix**2, axis=2)

                    # Remove zero distances from further calculations
                    zero_distances = np.where(distances_squared == 0)
                    valid_indexes = np.ones(arg_qty, dtype=bool)
                    valid_indexes[zero_distances[0]] = False

                    weights = distances_squared[valid_indexes] ** (-1.5)
                    numerator_sum = np.sum(y_data * weights, axis=1)
                    denominator_sum = np.sum(weights, axis=1)
                    result[valid_indexes] = numerator_sum / denominator_sum
                    result[~valid_indexes] = y_data[zero_distances[1]]

                    return result

                self._interpolation_func = shepard_interpolation

            case 5:  # RBF
                interpolator = RBFInterpolator(self._domain, self._image, neighbors=100)

                def rbf_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                    return interpolator(x)

                self._interpolation_func = rbf_interpolation

            case 6:  # regular_grid (RegularGridInterpolator)
                if not hasattr(self, "_grid_axes") or not hasattr(self, "_grid_data"):
                    raise AttributeError(
                        "The 'regular_grid' interpolation requires '_grid_axes' and "
                        "'_grid_data' to be set on the Function instance before calling "
                        "set_interpolation('regular_grid')."
                    )
                grid_interpolator = RegularGridInterpolator(
                    self._grid_axes,
                    self._grid_data,
                    method="linear",
                    bounds_error=True,
                )
                # Store so extrapolation funcs can reuse it
                self._grid_interpolator = grid_interpolator

                def grid_interpolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                    return grid_interpolator(x)

                self._interpolation_func = grid_interpolation

            case _:
                raise ValueError(
                    f"Interpolation {interpolation} method not recognized."
                )

    def __set_extrapolation_func(self):  # pylint: disable=too-many-statements
        """Defines extrapolation function used by the Function. Each
        extrapolation method has its own function. The function is stored in
        the attribute _extrapolation_func."""
        interpolation = INTERPOLATION_TYPES[self.__interpolation__]
        extrapolation = EXTRAPOLATION_TYPES[self.__extrapolation__]

        match extrapolation:
            case 0:  # zero

                def zero_extrapolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                    return 0

                self._extrapolation_func = zero_extrapolation
            case 1:  # natural
                match interpolation:
                    case 0:  # linear
                        if self.__dom_dim__ == 1:

                            def natural_extrapolation(
                                x, x_min, x_max, x_data, y_data, coeffs
                            ):  # pylint: disable=unused-argument
                                x_interval = 1 if x < x_min else -1
                                x_left = x_data[x_interval - 1]
                                y_left = y_data[x_interval - 1]
                                dx = float(x_data[x_interval] - x_left)
                                dy = float(y_data[x_interval] - y_left)
                                return (x - x_left) * (dy / dx) + y_left

                        else:
                            interpolator = RBFInterpolator(
                                self._domain, self._image, neighbors=100
                            )

                            def natural_extrapolation(
                                x, x_min, x_max, x_data, y_data, coeffs
                            ):  # pylint: disable=unused-argument
                                return interpolator(x)

                    case 1:  # polynomial

                        def natural_extrapolation(  # pylint: disable=function-redefined
                            x, x_min, x_max, x_data, y_data, coeffs
                        ):  # pylint: disable=unused-argument
                            return np.sum(coeffs * x ** np.arange(len(coeffs)))

                    case 2:  # akima

                        def natural_extrapolation(  # pylint: disable=function-redefined
                            x, x_min, x_max, x_data, y_data, coeffs
                        ):  # pylint: disable=unused-argument
                            a = coeffs[:4] if x < x_min else coeffs[-4:]
                            return a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]

                    case 3:  # spline

                        def natural_extrapolation(  # pylint: disable=function-redefined
                            x, x_min, x_max, x_data, y_data, coeffs
                        ):  # pylint: disable=unused-argument
                            if x < x_min:
                                a = coeffs[:, 0]
                                x_offset = x - x_data[0]
                            else:
                                a = coeffs[:, -1]
                                x_offset = x - x_data[-2]
                            return (
                                a[3] * x_offset**3
                                + a[2] * x_offset**2
                                + a[1] * x_offset
                                + a[0]
                            )

                    case 4:  # shepard
                        # pylint: disable=unused-argument,function-redefined
                        def natural_extrapolation(x, x_min, x_max, x_data, y_data, _):
                            arg_qty, arg_dim = x.shape
                            result = np.empty(arg_qty)
                            x = x.reshape((arg_qty, 1, arg_dim))
                            sub_matrix = x_data - x
                            distances_squared = np.sum(sub_matrix**2, axis=2)

                            # Remove zero distances from further calculations
                            zero_distances = np.where(distances_squared == 0)
                            valid_indexes = np.ones(arg_qty, dtype=bool)
                            valid_indexes[zero_distances[0]] = False

                            weights = distances_squared[valid_indexes] ** (-1.5)
                            numerator_sum = np.sum(y_data * weights, axis=1)
                            denominator_sum = np.sum(weights, axis=1)
                            result[valid_indexes] = numerator_sum / denominator_sum
                            result[~valid_indexes] = y_data[zero_distances[1]]

                            return result

                    case 5:  # RBF
                        interpolator = RBFInterpolator(
                            self._domain, self._image, neighbors=100
                        )

                        def natural_extrapolation(  # pylint: disable=function-redefined
                            x, x_min, x_max, x_data, y_data, coeffs
                        ):  # pylint: disable=unused-argument
                            return interpolator(x)

                    case 6:  # regular_grid
                        grid_extrapolator = RegularGridInterpolator(
                            self._grid_axes,
                            self._grid_data,
                            method="linear",
                            bounds_error=False,
                            fill_value=None,  # linear extrapolation beyond edges
                        )

                        def natural_extrapolation(  # pylint: disable=function-redefined
                            x, x_min, x_max, x_data, y_data, coeffs
                        ):  # pylint: disable=unused-argument
                            return grid_extrapolator(x)

                    case _:
                        raise ValueError(
                            f"Natural extrapolation not defined for {interpolation}."
                        )

                self._extrapolation_func = natural_extrapolation
            case 2:  # constant
                if self.__dom_dim__ == 1:

                    def constant_extrapolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                        return y_data[0] if x < x_min else y_data[-1]

                elif self.__interpolation__ == "regular_grid":
                    grid_axes = self._grid_axes
                    grid_interpolator_const = self._grid_interpolator

                    def constant_extrapolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                        # Clamp each coordinate to its axis bounds, then interpolate
                        x_clamped = np.copy(x)
                        for i, axis in enumerate(grid_axes):
                            x_clamped[:, i] = np.clip(
                                x_clamped[:, i], axis[0], axis[-1]
                            )
                        return grid_interpolator_const(x_clamped)

                else:
                    extrapolator = NearestNDInterpolator(self._domain, self._image)

                    def constant_extrapolation(x, x_min, x_max, x_data, y_data, coeffs):  # pylint: disable=unused-argument
                        return extrapolator(x)

                self._extrapolation_func = constant_extrapolation
            case _:
                raise ValueError(
                    f"Extrapolation {extrapolation} method not recognized."
                )

    def set_get_value_opt(self):
        """Defines a method that evaluates interpolations.

        Returns
        -------
        self : Function
        """
        if self._source_type is SourceType.CALLABLE:
            self.get_value_opt = self.source
        elif self._source_type is SourceType.SCALAR:
            self.get_value_opt = self.__get_value_opt_scalar
        elif self._source_type is SourceType.ARRAY:
            # Direct call to reduce call stack overhead
            self.get_value_opt = self._evaluator.expose()
        return self

    def __get_value_opt_scalar(self, x):  # pylint: disable=unused-argument
        """Evaluate a scalar (constant) Function. Always returns the stored
        constant value regardless of the input.

        Parameters
        ----------
        x : scalar or ndarray
            Value(s) where the Function is to be evaluated (ignored).

        Returns
        -------
        y : float or ndarray
            The constant scalar value, or an array filled with it.
        """
        if hasattr(x, "__iter__"):
            return np.full_like(x, self._scalar_value, dtype=float)
        return self._scalar_value

    def __resolve_bounds(self, lower, upper, samples):
        """Normalize lower, upper and samples to lists of length ``dom_dim``.

        For callable sources the default domain is ``[0, 10]`` per dimension.
        Cropped-domain constraints are applied first; explicitly supplied
        values always take precedence.

        Parameters
        ----------
        lower : scalar, list, or None
            Lower bound(s). Scalars are broadcast to all dimensions.
        upper : scalar, list, or None
            Upper bound(s). Scalars are broadcast to all dimensions.
        samples : int, list, or None
            Sample count(s). Scalars are broadcast to all dimensions.

        Returns
        -------
        tuple
            ``(lowers, uppers, samples_list)``, each a plain list of length
            ``dom_dim``.
        """
        n = self.__dom_dim__
        default_lo = [0.0] * n
        default_hi = [10.0] * n

        # Tighten defaults with any recorded cropped-domain constraints
        if self.__cropped_domain__ is not None:
            for i, lim in enumerate(self.__cropped_domain__):
                if i < n and lim is not None:
                    lo, hi = lim
                    if lo is not None:
                        default_lo[i] = max(default_lo[i], lo)
                    if hi is not None:
                        default_hi[i] = min(default_hi[i], hi)

        def _to_list(param, default):
            if param is None:
                return list(default)
            if isinstance(param, NUMERICAL_TYPES):
                return [float(param)] * n
            return [float(p) for p in param]

        if isinstance(samples, NUMERICAL_TYPES):
            samples_list = [int(samples)] * n
        elif samples is None:
            samples_list = [50] * n
        else:
            samples_list = [int(s) for s in samples]

        return _to_list(lower, default_lo), _to_list(upper, default_hi), samples_list

    def __build_nd_grid(self, lowers, uppers, samples_list):
        """Build an N-D evaluation grid and return flattened input columns.

        For 1-D returns a single-element list ``[xs]``. For higher dimensions
        an open meshgrid is created and each axis array is ravelled.

        Parameters
        ----------
        lowers : list of float
            Lower bound per dimension.
        uppers : list of float
            Upper bound per dimension.
        samples_list : list of int
            Number of sample points per dimension.

        Returns
        -------
        list of ndarray
            One flat array per input dimension; all arrays share the same
            length (``samples_list[0]`` for 1-D, or the product of all
            sample counts for N-D).
        """
        axes = [
            np.linspace(lowers[i], uppers[i], samples_list[i])
            for i in range(self.__dom_dim__)
        ]
        if self.__dom_dim__ == 1:
            return axes
        mesh = np.meshgrid(*axes)
        return [m.ravel() for m in mesh]

    def set_discrete(
        self,
        lower=None,
        upper=None,
        samples=200,
        interpolation="spline",
        extrapolation="constant",
        one_by_one=True,  # pylint: disable=unused-argument
        mutate_self=True,
    ):
        """This method discretizes a 1-D or 2-D Function by evaluating it at
        certain points (sampling range) and storing the results in a list,
        which is converted into a Function and then returned. By default, the
        original Function object is replaced by the new one, which can be
        changed by the attribute `mutate_self`.

        This method is specially useful to change a dataset sampling or to
        convert a Function defined by a callable into a list based Function.

        Parameters
        ----------
        lower : scalar, optional
            Value where sampling range will start. Default is None.
        upper : scalar, optional
            Value where sampling range will end. Default is None.
        samples : int, optional
            Number of samples to be taken from inside range. Default is 200.
        interpolation : string
            Interpolation method to be used if source type is ndarray.
            For 1-D functions, linear, polynomial, akima and spline are
            supported. For N-D functions, linear, shepard, rbf and
            regular_grid are supported.
            Default for 1-D functions is spline and for N-D functions is
            shepard.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function at the nearest edge of
            the domain, and 'zero', which returns zero for all points outside
            of source range. Default for 1-D functions is constant and for
            N-D functions is natural.
        one_by_one : boolean, optional
            If True, evaluate Function in each sample point separately. If
            False, evaluates Function in vectorized form. Default is True.
        mutate_self : boolean, optional
            If True, the original Function object source will be replaced by
            the new one. If False, the original Function object source will
            remain unchanged, and the new one is simply returned.
            Default is True.

        Returns
        -------
        self : Function

        Notes
        -----
        1. This method performs by default in place replacement of the original
        Function object source. This can be changed by the attribute `mutate_self`.

        2. For N-D functions (dim > 1) the interpolation is forced to
        ``shepard`` and extrapolation to ``natural``, regardless of the
        arguments passed.
        """
        func = deepcopy(self) if not mutate_self else self

        lowers, uppers, samples = self.__resolve_bounds(lower, upper, samples)
        columns = self.__build_nd_grid(lowers, uppers, samples)

        if self.__dom_dim__ == 1:
            func.__interpolation__ = interpolation
            func.__extrapolation__ = extrapolation
        else:
            func.__interpolation__ = "shepard"
            func.__extrapolation__ = "natural"

        zs = np.array(func.get_value(*columns))
        return func.set_source(np.column_stack(columns + [zs]))

    def set_discrete_based_on_model(
        self, model_function, one_by_one=True, keep_self=True, mutate_self=True
    ):  # pylint: disable=unused-argument
        """This method transforms the domain of an N-D Function instance into a
        list of discrete points based on the domain of a model Function
        instance. It does so by retrieving the domain, domain name,
        interpolation method and extrapolation method of the model Function
        instance. It then evaluates the original Function instance in all
        points of the retrieved domain to generate the list of discrete points
        that will be used for interpolation when this Function is called.

        By default, the original Function object is replaced by the new one,
        which can be changed by the attribute `mutate_self`.

        Parameters
        ----------
        model_function : Function
            Function object that will be used to define the sampling points,
            interpolation method and extrapolation method.
            Must be a Function whose source attribute is a list (i.e. a list
            based Function instance). Must have the same domain dimension as the
            Function to be discretized.
        one_by_one : boolean, optional
            If True, evaluate Function in each sample point separately. If
            False, evaluates Function in vectorized form. Default is True.
        keep_self : boolean, optional
            If True, the original Function interpolation and extrapolation
            methods will be kept. If False, those are substituted by the ones
            from the model Function. Default is True.
        mutate_self : boolean, optional
            If True, the original Function object source will be replaced by
            the new one. If False, the original Function object source will
            remain unchanged, and the new one is simply returned.

        Returns
        -------
        self : Function

        See also
        --------
        Function.set_discrete

        Examples
        --------
        This method is particularly useful when algebraic operations are carried
        out using Function instances defined by different discretized domains
        (same range, but different mesh size). Once an algebraic operation is
        done, it will not directly be applied between the list of discrete
        points of the two Function instances. Instead, the result will be a
        Function instance defined by a callable that calls both Function
        instances and performs the operation. This makes the evaluation of the
        resulting Function inefficient, due to extra function calling overhead
        and multiple interpolations being carried out.

        >>> from rocketpy import Function
        >>> f = Function([(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)])
        >>> g = Function([(0, 0), (2, 2), (4, 4)])
        >>> h = f * g
        >>> type(h.source)
        <class 'function'>

        Therefore, it is good practice to make sure both Function instances are
        defined by the same domain, i.e. by the same list of mesh points. This
        way, the algebraic operation will be carried out directly between the
        lists of discrete points, generating a new Function instance defined by
        this result. When it is evaluated, there are no extra function calling
        overheads neither multiple interpolations.

        >>> g.set_discrete_based_on_model(f)
        'Function from R1 to R1 : (Scalar) → (Scalar)'
        >>> h = f * g
        >>> h.source
        array([[ 0.,  0.],
               [ 1.,  1.],
               [ 2.,  8.],
               [ 3., 27.],
               [ 4., 64.]])

        Notes
        -----
        1. This method performs by default in place replacement of the original
        Function object source. This can be changed by the attribute `mutate_self`.

        2. This method is similar to set_discrete, but it uses the domain of a
        model Function to define the domain of the new Function instance.

        3. This method supports functions of any domain dimension.
        """
        if model_function._source_type is not SourceType.ARRAY:
            raise TypeError("model_function must be a list based Function.")
        if model_function.__dom_dim__ != self.__dom_dim__:
            raise ValueError("model_function must have the same domain dimension.")

        func = deepcopy(self) if not mutate_self else self

        if not keep_self:
            func.__interpolation__ = model_function.__interpolation__
            func.__extrapolation__ = model_function.__extrapolation__

        n = func.__dom_dim__
        columns = [model_function.source[:, i] for i in range(n)]

        zs = np.array(func.get_value(*columns))
        return func.set_source(np.column_stack(columns + [zs]))

    def reset(
        self,
        inputs=None,
        outputs=None,
        interpolation=None,
        extrapolation=None,
        title=None,
    ):
        """This method allows the user to reset the inputs, outputs,
        interpolation and extrapolation settings of a Function object, all at
        once, without having to call each of the corresponding methods.

        Parameters
        ----------
        inputs : string, sequence of strings, optional
            List of input variable names. If None, the original inputs are kept.
            See Function.set_inputs for more information.
        outputs : string, sequence of strings, optional
            List of output variable names. If None, the original outputs are
            kept. See Function.set_outputs for more information.
        interpolation : string, optional
            Interpolation method to be used if source type is ndarray.
            See Function.set_interpolation for more information.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            See Function.set_extrapolation for more information.

        Examples
        --------
        A simple use case is to reset the inputs and outputs of a Function
        object that has been defined by algebraic manipulation of other Function
        objects.

        >>> from rocketpy import Function
        >>> v = Function(lambda t: (9.8*t**2)/2, inputs='t', outputs='v')
        >>> mass = 10 # Mass
        >>> kinetic_energy = mass * v**2 / 2
        >>> v.get_inputs(), v.get_outputs()
        (['t'], ['v'])
        >>> kinetic_energy
        'Function from R1 to R1 : (t) → (Scalar)'
        >>> kinetic_energy.reset(inputs='t', outputs='Kinetic Energy');
        'Function from R1 to R1 : (t) → (Kinetic Energy)'

        Returns
        -------
        self : Function
        """
        if inputs is not None:
            self.set_inputs(inputs)
        if outputs is not None:
            self.set_outputs(outputs)
        if interpolation is not None and interpolation != self.__interpolation__:
            self.set_interpolation(interpolation)
        if extrapolation is not None and extrapolation != self.__extrapolation__:
            self.set_extrapolation(extrapolation)

        self.set_title(title)

        return self

    def __crop_input(self, func, x_lim):
        """Restrict input domain of func to the intervals in x_lim.

        Records the bounds in ``func.__cropped_domain__`` for any source type
        so that plotting and discretisation helpers can honour the restriction.
        For array sources the rows that fall outside the specified ranges are
        also removed in-place via a vectorised boolean mask.

        Parameters
        ----------
        func : Function
            The Function instance to be modified in-place.
        x_lim : list[tuple | None]
            Per-dimension ``(lower, upper)`` pairs.  ``None`` entries skip
            that dimension.
        """
        n = func.__dom_dim__

        # Build unified cropped_domain list: [(lo, hi) | None, ...]
        cropped = [None] * n
        for i, lim in enumerate(x_lim):
            if lim is not None and lim[0] < lim[1]:
                cropped[i] = lim
        func.__cropped_domain__ = cropped

        # Mask array data with a single vectorised pass
        if isinstance(func.source, np.ndarray):
            mask = np.ones(len(func.source), dtype=bool)
            for i, lim in enumerate(x_lim):
                if lim is not None:
                    mask &= (func.source[:, i] >= lim[0]) & (
                        func.source[:, i] <= lim[1]
                    )
            func.source = func.source[mask]

    def crop(self, x_lim):
        """Restrict the **input** domain of the Function to specified ranges.

        This method limits the input values of the Function to the intervals
        defined in `x_lim`, effectively trimming the data so that only values
        within the specified ranges are retained. For multi-dimensional
        functions, each dimension can be cropped independently by providing a
        tuple with lower and upper bounds for each input variable. If a
        dimension is set to `None`, it will not be cropped.

        Parameters
        ----------
        x_lim : list[tuple]
            Range of values with lower and upper limits for input values to be
            cropped within.

        Returns
        -------
        Function
            A new Function instance with the cropped domain.

        See also
        --------
        Function.clip

        Examples
        --------
        >>> from rocketpy import Function
        >>> import numpy as np

        Create two 2D functions:
        >>> f1 = Function(
        ...     lambda x1, x2: np.sin(x1)*np.cos(x2),
        ...     inputs=['x1', 'x2'],
        ...     outputs='y'
        ... )
        >>> f2 = Function(
        ...     lambda x1, x2: np.cos(x1)*np.sin(x2),
        ...     inputs=['x1', 'x2'],
        ...     outputs='y'
        ... )

        Crop their domains:
        >>> f1_cropped = f1.crop([(-1, 1), (-2, 2)])
        >>> f2_cropped = f2.crop([None, (-2, 2)])

        Compare the cropped functions using Function.compare_plots:
        >>> # Function.compare_plots([
        >>> #     (f1_cropped, 'sin(x1)*cos(x2), cropped'),
        >>> #     (f2_cropped, 'cos(x1)*sin(x2), cropped')
        >>> # ])
        """
        if not isinstance(x_lim, list):
            raise TypeError("x_lim must be a list of tuples.")

        if len(x_lim) > self.__dom_dim__:
            raise ValueError(
                "x_lim must not exceed the length of the domain dimension."
            )

        cropped_func = deepcopy(self)
        self.__crop_input(cropped_func, x_lim)
        cropped_func.set_source(cropped_func.source)
        return cropped_func

    def __clip_output(self, func, y_lim: list[tuple]):
        """Restrict the output of func to the ranges specified in y_lim.

        Dispatches on the source type of func:

        - ``ndarray``: removes rows whose output column falls outside the
          range via a vectorised boolean mask.
        - Scalar: raises ``ArithmeticError`` when the constant value is
          outside every output range.
        - Callable: wraps the callable so that returned values are clamped
          to the specified ranges without removing any inputs.

        Parameters
        ----------
        func : Function
            The Function instance to be modified in-place.
        y_lim : list[tuple]
            Per-output ``(lower, upper)`` pairs.
        """
        if isinstance(func.source, np.ndarray):
            mask = np.ones(len(func.source), dtype=bool)
            for i, (lo, hi) in enumerate(y_lim):
                col = func.__dom_dim__ + i
                mask &= (func.source[:, col] >= lo) & (func.source[:, col] <= hi)
            func.source = func.source[mask]
        elif func._source_type is SourceType.SCALAR:
            # Clamp the scalar value to the output range
            lo, hi = y_lim[0]
            clamped = max(lo, min(hi, func._scalar_value))
            func.set_source(clamped)
        elif callable(func.source):
            original = func.source

            def _clipped(*args):
                result = original(*args)
                if isinstance(result, (tuple, list)):
                    clipped = [
                        max(lo, min(hi, result[i])) for i, (lo, hi) in enumerate(y_lim)
                    ]
                    return tuple(clipped) if len(clipped) > 1 else clipped[0]
                # Scalar result
                lo, hi = y_lim[0]
                return max(lo, min(hi, result))

            func.source = _clipped

    def clip(self, y_lim):
        """Restrict the **output** values of the Function to specified ranges.

        This method limits the output values of the Function to the intervals
        defined in `y_lim`, effectively removing all input-output pairs where
        the output values fall outside the specified ranges. This operation
        filters the data based on output constraints rather than input domain
        restrictions.

        Parameters
        ----------
        y_lim : list[tuple]
            Range of values with lower and upper limits for output values to be
            clipped within.

        Returns
        -------
        Function
            A new Function instance with the clipped output values.

        See also
        --------
        Function.crop

        Examples
        --------
        >>> from rocketpy import Function
        >>> f = Function(lambda x: x**2, inputs='x', outputs='y')
        >>> f_clip = f.clip([(-5.0, 5.0)])
        >>> f.get_value(-3.0), f.get_value(0.0), f.get_value(3.0)
        (9.0, 0.0, 9.0)
        >>> f_clip.get_value(-3.0), f_clip.get_value(0.0), f_clip.get_value(3.0)
        (5.0, 0.0, 5.0)
        """
        if not isinstance(y_lim, list):
            raise TypeError("y_lim must be a list of tuples.")

        if len(y_lim) != len(self.__outputs__):
            raise ValueError(
                "y_lim must have the same length as the output dimensions."
            )

        clipped_func = deepcopy(self)

        self.__clip_output(clipped_func, y_lim)

        return clipped_func.set_source(clipped_func.source)

    # Define all get methods
    def get_inputs(self):
        "Return tuple of inputs of the function."
        return self.__inputs__

    def get_outputs(self):
        "Return tuple of outputs of the function."
        return self.__outputs__

    def get_source(self):
        "Return source list or function of the Function."
        return self.source

    def get_source_type(self):
        """Return the Function source type.

        Returns
        -------
        SourceType
            Enum describing whether the source is callable, array, or scalar.
        """
        return self._source_type

    def is_scalar_source(self):
        """Return True if the Function is a constant scalar source."""
        return self._source_type is SourceType.SCALAR

    def is_array_source(self):
        """Return True if the Function source is array-based."""
        return self._source_type is SourceType.ARRAY

    def is_callable_source(self):
        """Return True if the Function source is callable-based."""
        return self._source_type is SourceType.CALLABLE

    def get_scalar_value(self):
        """Return the scalar value for a constant Function.

        Raises
        ------
        ValueError
            If the Function is not scalar-based.
        """
        if self._source_type is not SourceType.SCALAR:
            raise ValueError("Function is not scalar-based")
        return self._scalar_value

    def get_image_dim(self):
        "Return int describing dimension of the image space of the function."
        return self.__img_dim__

    def get_domain_dim(self):
        "Return int describing dimension of the domain space of the function."
        return self.__dom_dim__

    def get_interpolation_method(self):
        "Return string describing interpolation method used."
        return self.__interpolation__

    def get_extrapolation_method(self):
        "Return string describing extrapolation method used."
        return self.__extrapolation__

    def get_value(self, *args):
        """This method returns the value of the Function at the specified
        point. See Function.get_value_opt for a faster, but limited,
        implementation.

        Parameters
        ----------
        args : scalar, list
            Value where the Function is to be evaluated. If the Function is
            1-D, only one argument is expected, which may be an int, a float
            or a list of ints or floats, in which case the Function will be
            evaluated at all points in the list and a list of floats will be
            returned. If the function is N-D, N arguments must be given, each
            one being an scalar or list.

        Returns
        -------
        ans : scalar, list
            Value of the Function at the specified point(s).

        Examples
        --------
        >>> from rocketpy import Function

        Testing with callable source (1 dimension):

        >>> f = Function(lambda x: x**2)
        >>> f.get_value(2)
        4
        >>> f.get_value(2.5)
        6.25
        >>> f.get_value([1, 2, 3])
        [1, 4, 9]
        >>> f.get_value([1, 2.5, 4.0])
        [1, 6.25, 16.0]

        Testing with callable source (2 dimensions):

        >>> f2 = Function(lambda x, y: x**2 + y**2)
        >>> f2.get_value(1, 2)
        5
        >>> f2.get_value([1, 2, 3], [1, 2, 3])
        [2, 8, 18]
        >>> f2.get_value([5], [5])
        [50]

        Testing with ndarray source (1 dimension):

        >>> f3 = Function(
        ...    [(0, 0), (1, 1), (1.5, 2.25), (2, 4), (2.5, 6.25), (3, 9), (4, 16)]
        ... )
        >>> f3.get_value(2)
        np.float64(4.0)
        >>> f3.get_value(2.5)
        np.float64(6.25)
        >>> f3.get_value([1, 2, 3])
        [np.float64(1.0), np.float64(4.0), np.float64(9.0)]
        >>> f3.get_value([1, 2.5, 4.0])
        [np.float64(1.0), np.float64(6.25), np.float64(16.0)]

        Testing with ndarray source (2 dimensions):

        >>> f4 = Function(
        ...    [(0, 0, 0), (1, 1, 1), (1, 2, 2), (2, 4, 8), (3, 9, 27)]
        ... )
        >>> f4.get_value(1, 1)
        1.0
        >>> f4.get_value(2, 4)
        8.0
        >>> abs(f4.get_value(1, 1.5) - 1.5) < 1e-2  # the interpolation is not perfect
        True
        >>> f4.get_value(3, 9)
        27.0
        """
        if len(args) != self.__dom_dim__:
            raise ValueError(
                f"This Function takes {self.__dom_dim__} arguments, {len(args)} given."
            )

        source_type = self._source_type
        arg = args[0]
        _is_iterable = hasattr(arg, "__iter__")

        if source_type is SourceType.ARRAY:
            if self.__dom_dim__ == 1:
                if not _is_iterable:
                    return self.get_value_opt(arg, _is_iterable=False)
                else:
                    if len(arg) < _LIST_VECTORIZE_THRESHOLD:
                        return np.array(
                            [self.get_value_opt(x) for x in arg], dtype=float
                        )
                    return self.get_value_opt(
                        np.asarray(arg, dtype=float), _is_iterable=True
                    )
            else:
                return self.get_value_opt(*args, _is_iterable=_is_iterable)

        if source_type is SourceType.SCALAR:
            if not _is_iterable:
                return self._scalar_value
            else:
                return np.full_like(arg, self._scalar_value, dtype=float)

        if source_type is SourceType.CALLABLE:
            if self.__dom_dim__ == 1:
                if not _is_iterable:
                    return self.source(arg)
                else:
                    return np.array([self.source(x) for x in arg], dtype=float)
            else:
                if not _is_iterable:
                    return self.source(*args)
                else:
                    broadcasted = np.broadcast_arrays(*args)
                    return np.array(
                        [self.source(*a) for a in zip(*broadcasted)], dtype=float
                    )

    def __getitem__(self, args):
        """Returns item of the Function source. If the source is not an array,
        an error will result.

        Parameters
        ----------
        args : int, float
            Index of the item to be retrieved.

        Returns
        -------
        self.source[args] : float, array
            Item specified from Function.source.
        """
        return self.source[args]

    def __len__(self):
        """Returns length of the Function source. If the source is not an
        array, an error will result.

        Returns
        -------
        len(self.source) : int
            Length of Function.source.
        """
        return len(self.source)

    def __bool__(self):
        """Returns true if self exists. This is to avoid getting into __len__
        method in boolean statements.

        Returns
        -------
        bool : bool
            Always True.
        """
        return True

    # Define all conversion methods
    def to_frequency_domain(self, lower, upper, sampling_frequency, remove_dc=True):
        """Performs the conversion of the Function to the Frequency Domain and
        returns the result. This is done by taking the Fourier transform of the
        Function. The resulting frequency domain is symmetric, i.e., the
        negative frequencies are included as well.

        Parameters
        ----------
        lower : float
            Lower bound of the time range.
        upper : float
            Upper bound of the time range.
        sampling_frequency : float
            Sampling frequency at which to perform the Fourier transform.
        remove_dc : bool, optional
            If True, the DC component is removed from the Fourier transform.

        Returns
        -------
        Function
            The Function in the frequency domain.

        Examples
        --------
        >>> from rocketpy import Function
        >>> import numpy as np
        >>> main_frequency = 10 # Hz
        >>> time = np.linspace(0, 10, 1000)
        >>> signal = np.sin(2 * np.pi * main_frequency * time)
        >>> time_domain = Function(np.array([time, signal]).T)
        >>> frequency_domain = time_domain.to_frequency_domain(
        ...     lower=0, upper=10, sampling_frequency=100
        ... )
        >>> peak_frequencies_index = np.where(frequency_domain[:, 1] > 0.001)
        >>> peak_frequencies = frequency_domain[peak_frequencies_index, 0]
        >>> print(peak_frequencies)
        [[-10.  10.]]
        """
        # Get the time domain data
        sampling_time_step = 1.0 / sampling_frequency
        sampling_range = np.arange(lower, upper, sampling_time_step)
        number_of_samples = len(sampling_range)
        sampled_points = self(sampling_range)
        if remove_dc:
            sampled_points -= np.mean(sampled_points)
        fourier_amplitude = np.abs(np.fft.fft(sampled_points) / (number_of_samples / 2))
        fourier_frequencies = np.fft.fftfreq(number_of_samples, sampling_time_step)
        return Function(
            source=np.array([fourier_frequencies, fourier_amplitude]).T,
            inputs="Frequency (Hz)",
            outputs="Amplitude",
            interpolation="linear",
            extrapolation="zero",
        )

    def short_time_fft(
        self,
        lower,
        upper,
        sampling_frequency,
        window_size,
        step_size,
        remove_dc=True,
        only_positive=True,
    ):
        r"""
        Performs the Short-Time Fourier Transform (STFT) of the Function and
        returns the result. The STFT is computed by applying the Fourier
        transform to overlapping windows of the Function.

        Parameters
        ----------
        lower : float
            Lower bound of the time range.
        upper : float
            Upper bound of the time range.
        sampling_frequency : float
            Sampling frequency at which to perform the Fourier transform.
        window_size : float
            Size of the window for the STFT, in seconds.
        step_size : float
            Step size for the window, in seconds.
        remove_dc : bool, optional
            If True, the DC component is removed from each window before
            computing the Fourier transform.
        only_positive: bool, optional
            If True, only the positive frequencies are returned.

        Returns
        -------
        list[Function]
            A list of Functions, each representing the STFT of a window.

        Examples
        --------

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from rocketpy import Function

        Generate a signal with varying frequency:

        >>> T_x, N = 1 / 20 , 1000  # 20 Hz sampling rate for 50 s signal
        >>> t_x = np.arange(N) * T_x  # time indexes for signal
        >>> f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5 # varying frequency
        >>> signal = np.sin(2 * np.pi * np.cumsum(f_i) * T_x)  # the signal

        Create the Function object and perform the STFT:

        >>> time_domain = Function(np.array([t_x, signal]).T)
        >>> stft_result = time_domain.short_time_fft(
        ...     lower=0,
        ...     upper=50,
        ...     sampling_frequency=95,
        ...     window_size=2,
        ...     step_size=0.5,
        ... )

        Plot the spectrogram:

        >>> Sx = np.abs([window[:, 1] for window in stft_result])
        >>> t_lo, t_hi = t_x[0], t_x[-1]
        >>> fig1, ax1 = plt.subplots(figsize=(10, 6))
        >>> im1 = ax1.imshow(
        ...     Sx.T,
        ...     origin='lower',
        ...     aspect='auto',
        ...     extent=[t_lo, t_hi, 0, 50],
        ...     cmap='viridis'
        ... )
        >>> _ = ax1.set_title(rf"STFT (2$\,s$ Gaussian window, $\sigma_t=0.4\,$s)")
        >>> _ = ax1.set(
        ...     xlabel=f"Time $t$ in seconds",
        ...     ylabel=f"Freq. $f$ in Hz)",
        ...     xlim=(t_lo, t_hi)
        ... )
        >>> _ = ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
        >>> _ = fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
        >>> # Shade areas where window slices stick out to the side
        >>> for t0_, t1_ in [(t_lo, 1), (49, t_hi)]:
        ...     _ = ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
        >>> # Mark signal borders with vertical line
        >>> for t_ in [t_lo, t_hi]:
        ...     _ = ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
        >>> # Add legend and finalize plot
        >>> _ = ax1.legend()
        >>> fig1.tight_layout()
        >>> # plt.show() # uncomment to show the plot

        References
        ----------
        Example adapted from the SciPy documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html
        """
        # Get the time domain data
        sampling_time_step = 1.0 / sampling_frequency
        sampling_range = np.arange(lower, upper, sampling_time_step)
        sampled_points = self(sampling_range)
        samples_per_window = int(window_size * sampling_frequency)
        samples_skipped_per_step = int(step_size * sampling_frequency)
        stft_results = []

        max_start = len(sampled_points) - samples_per_window + 1

        for start in range(0, max_start, samples_skipped_per_step):
            windowed_samples = sampled_points[start : start + samples_per_window]
            if remove_dc:
                windowed_samples -= np.mean(windowed_samples)
            fourier_amplitude = np.abs(
                np.fft.fft(windowed_samples) / (samples_per_window / 2)
            )
            fourier_frequencies = np.fft.fftfreq(samples_per_window, sampling_time_step)

            # Filter to keep only positive frequencies if specified
            if only_positive:
                positive_indices = fourier_frequencies > 0
                fourier_frequencies = fourier_frequencies[positive_indices]
                fourier_amplitude = fourier_amplitude[positive_indices]

            stft_results.append(
                Function(
                    source=np.array([fourier_frequencies, fourier_amplitude]).T,
                    inputs="Frequency (Hz)",
                    outputs="Amplitude",
                    interpolation="linear",
                    extrapolation="zero",
                )
            )

        return stft_results

    def low_pass_filter(self, alpha, file_path=None):
        """Implements a low pass filter with a moving average filter. This does
        not mutate the original Function object, but returns a new one with the
        filtered source. The filtered source is also saved to a CSV file if a
        file path is given.

        Parameters
        ----------
        alpha : float
            Attenuation coefficient, 0 <= alpha <= 1
            For a given dataset, the larger alpha is, the more closely the
            filtered function returned will match the function the smaller
            alpha is, the smoother the filtered function returned will be
            (but with a phase shift)
        file_path : string, optional
            File path or file name of the CSV to save. Don't save any CSV if
            if no argument is passed. Initiated to None.

        Returns
        -------
        Function
            The function with the incoming source filtered
        """
        filtered_signal = np.zeros_like(self.source)
        filtered_signal[0] = self.source[0]

        for i in range(1, len(self.source)):
            # for each point of our dataset, we apply a exponential smoothing
            filtered_signal[i] = (
                alpha * self.source[i] + (1 - alpha) * filtered_signal[i - 1]
            )

        if isinstance(file_path, str):
            self.savetxt(file_path)

        return Function(
            source=filtered_signal,
            inputs=self.__inputs__,
            outputs=self.__outputs__,
            interpolation=self.__interpolation__,
            extrapolation=self.__extrapolation__,
            title=self.title,
        )

    def remove_outliers_iqr(self, threshold=1.5):
        """Remove outliers from the Function source using the interquartile
        range method. The Function should have an array-like source.

        Parameters
        ----------
        threshold : float, optional
            Threshold for the interquartile range method. Default is 1.5.

        Returns
        -------
        Function
            The Function with the outliers removed.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Outlier#Tukey's_fences
        """

        if self._source_type is not SourceType.ARRAY:
            raise TypeError(
                "Cannot remove outliers if the source is not array-based."
                + " The Function.source should be array-like."
            )

        x = self.x_array
        y = self.y_array
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        y_iqr = y_q3 - y_q1
        y_lower = y_q1 - threshold * y_iqr
        y_upper = y_q3 + threshold * y_iqr

        y_filtered = y[(y >= y_lower) & (y <= y_upper)]
        x_filtered = x[(y >= y_lower) & (y <= y_upper)]

        return Function(
            source=np.column_stack((x_filtered, y_filtered)),
            inputs=self.__inputs__,
            outputs=self.__outputs__,
            interpolation=self.__interpolation__,
            extrapolation=self.__extrapolation__,
            title=self.title,
        )

    # Define all presentation methods
    def __call__(self, *args, filename=None):
        """Plot the Function if no argument is given. If an
        argument is given, return the value of the function at the desired
        point.

        Parameters
        ----------
        args : scalar, list, optional
            Value where the Function is to be evaluated. If the Function is
            1-D, only one argument is expected, which may be an int, a float
            or a list of ints or floats, in which case the Function will be
            evaluated at all points in the list and a list of floats will be
            returned. If the function is N-D, N arguments must be given, each
            one being an scalar or list.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        ans : None, scalar, list
        """
        if not args:
            return self.plot(filename=filename)

        return self.get_value(*args)

    def __str__(self):
        "Return a string representation of the Function"
        return str(
            "Function from R"
            + str(self.__dom_dim__)
            + " to R"
            + str(self.__img_dim__)
            + " : ("
            + ", ".join(self.__inputs__)
            + ") → ("
            + ", ".join(self.__outputs__)
            + ")"
        )

    def __repr__(self):
        "Return a string representation of the Function"
        return repr(
            "Function from R"
            + str(self.__dom_dim__)
            + " to R"
            + str(self.__img_dim__)
            + " : ("
            + ", ".join(self.__inputs__)
            + ") → ("
            + ", ".join(self.__outputs__)
            + ")"
        )

    def set_title(self, title):
        """Used to define the title of the Function object.

        Parameters
        ----------
        title : str
            Title to be assigned to the Function.
        """
        if title:
            self.title = title
        else:
            if self.__dom_dim__ == 1:
                self.title = (
                    self.__outputs__[0].title() + " x " + self.__inputs__[0].title()
                )
            elif self.__dom_dim__ == 2:
                self.title = (
                    self.__outputs__[0].title()
                    + " x "
                    + self.__inputs__[0].title()
                    + " x "
                    + self.__inputs__[1].title()
                )

    def plot(self, *args, **kwargs):
        """Call Function.plot_1d if Function is 1-Dimensional or call
        Function.plot_2d if Function is 2-Dimensional and forward arguments
        and key-word arguments."""
        if isinstance(self, list):
            # Extract filename from kwargs
            filename = kwargs.get("filename", None)

            # Compare multiple plots
            Function.compare_plots(self, filename)
        else:
            if self.__dom_dim__ == 1:
                self.plot_1d(*args, **kwargs)
            elif self.__dom_dim__ == 2:
                self.plot_2d(*args, **kwargs)
            else:
                logger.error("Only functions with 1D or 2D domains can be plotted.")

    @deprecated(
        reason="The `Function.plot1D` method is set to be deprecated and fully "
        "removed in rocketpy v2.0.0",
        alternative="Function.plot_1d",
    )
    def plot1D(self, *args, **kwargs):  # pragma: no cover
        """Deprecated method, use Function.plot_1d instead."""
        return self.plot_1d(*args, **kwargs)

    def plot_1d(  # pylint: disable=too-many-statements
        self,
        lower=None,
        upper=None,
        samples=1000,
        force_data=False,
        force_points=False,
        return_object=False,
        equal_axis=False,
        *,
        filename=None,
    ):
        """Plot 1-Dimensional Function, from a lower limit to an upper limit,
        by sampling the Function several times in the interval. The title of
        the graph is given by the name of the axes, which are taken from
        the Function`s input and output names.

        Parameters
        ----------
        lower : scalar, optional
            The lower limit of the interval in which the function is to be
            plotted. The default value for function type Functions is 0. By
            contrast, if the Function is given by a dataset, the default
            value is the start of the dataset.
        upper : scalar, optional
            The upper limit of the interval in which the function is to be
            plotted. The default value for function type Functions is 10. By
            contrast, if the Function is given by a dataset, the default
            value is the end of the dataset.
        samples : int, optional
            The number of samples in which the function will be evaluated for
            plotting it, which draws lines between each evaluated point.
            The default value is 1000.
        force_data : Boolean, optional
            If Function is given by an interpolated dataset, setting force_data
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        force_points : Boolean, optional
            Setting force_points to True will plot all points, as a scatter, in
            which the Function was evaluated in the dataset. Default value is
            False.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Define a mesh and y values at mesh nodes for plotting
        fig = plt.figure()
        ax = fig.axes
        if self._source_type is not SourceType.ARRAY:
            # Determine boundaries
            domain = [0, 10]
            # pylint: disable=unsubscriptable-object
            if (
                self.__cropped_domain__ is not None
                and self.__cropped_domain__[0] is not None
            ):
                lo, hi = self.__cropped_domain__[0]
                # pylint: enable=unsubscriptable-object
                if lo is not None and lo > domain[0]:
                    domain[0] = lo
                if hi is not None and hi < domain[1]:
                    domain[1] = hi
            lower = domain[0] if lower is None else lower
            upper = domain[1] if upper is None else upper
        else:
            # Determine boundaries
            x_data = self.x_array
            x_min, x_max = self.x_initial, self.x_final
            lower = x_min if lower is None else lower
            upper = x_max if upper is None else upper
            # Plot data points if force_data = True
            too_low = x_min >= lower
            too_high = x_max <= upper
            lo_ind = 0 if too_low else np.where(x_data >= lower)[0][0]
            up_ind = len(x_data) - 1 if too_high else np.where(x_data <= upper)[0][0]
            points = self.source[lo_ind : (up_ind + 1), :].T.tolist()
            if force_data:
                plt.scatter(points[0], points[1], marker="o")
        # Calculate function at mesh nodes
        x = np.linspace(lower, upper, samples)
        y = self.get_value(x.tolist())
        # Plots function
        if force_points:
            plt.scatter(x, y, marker="o")
        if equal_axis:
            plt.axis("equal")
        plt.plot(x, y)
        # Turn on grid and set title and axis
        plt.grid(True)
        plt.title(self.title)
        plt.xlabel(self.__inputs__[0].title())
        plt.ylabel(self.__outputs__[0].title())
        show_or_save_plot(filename)
        if return_object:
            return fig, ax

    @deprecated(
        reason="The `Function.plot2D` method is set to be deprecated and fully "
        "removed in rocketpy v2.0.0",
        alternative="Function.plot_2d",
    )
    def plot2D(self, *args, **kwargs):  # pragma: no cover
        """Deprecated method, use Function.plot_2d instead."""
        return self.plot_2d(*args, **kwargs)

    def plot_2d(  # pylint: disable=too-many-statements
        self,
        lower=None,
        upper=None,
        samples=None,
        force_data=True,
        disp_type="surface",
        alpha=0.6,
        cmap="viridis",
        *,
        filename=None,
    ):
        """Plot 2-Dimensional Function, from a lower limit to an upper limit,
        by sampling the Function several times in the interval. The title of
        the graph is given by the name of the axis, which are taken from
        the Function`s inputs and output names.

        Parameters
        ----------
        lower : scalar, array of int or float, optional
            The lower limits of the interval in which the function is to be
            plotted, which can be an int or float, which is repeated for both
            axis, or an array specifying the limit for each axis. The default
            value for function type Functions is 0. By contrast, if the
            Function is given by a dataset, the default value is the start of
            the dataset for each axis.
        upper : scalar, array of int or float, optional
            The upper limits of the interval in which the function is to be
            plotted, which can be an int or float, which is repeated for both
            axis, or an array specifying the limit for each axis. The default
            value for function type Functions is 0. By contrast, if the
            Function is given by a dataset, the default value is the end of
            the dataset for each axis.
        samples : int, array of int, optional
            The number of samples in which the function will be evaluated for
            plotting it, which draws lines between each evaluated point.
            The default value is 30 for each axis.
        force_data : Boolean, optional
            If Function is given by an interpolated dataset, setting force_data
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        disp_type : string, optional
            Display type of plotted graph, which can be surface, wireframe,
            contour, or contourf. Default value is surface.
        alpha : float, optional
            Transparency of plotted graph, which can be a value between 0 and
            1. Default value is 0.6.
        cmap : string, optional
            Colormap of plotted graph, which can be any of the color maps
            available in matplotlib. Default value is viridis.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        if samples is None:
            samples = [30, 30]
        # Prepare plot
        figure = plt.figure()
        axes = figure.add_subplot(111, projection="3d")
        # Define a mesh and f values at mesh nodes for plotting
        if self._source_type is not SourceType.ARRAY:
            # Determine boundaries
            domain = [[0, 10], [0, 10]]
            if self.__cropped_domain__ is not None:
                for i in range(0, 2):
                    if self.__cropped_domain__[i] is not None:
                        if self.__cropped_domain__[i][0] > domain[i][0]:
                            domain[i][0] = self.__cropped_domain__[i][0]
                        if self.__cropped_domain__[i][1] < domain[i][1]:
                            domain[i][1] = self.__cropped_domain__[i][1]
            lower = [domain[0][0], domain[1][0]] if lower is None else lower
            lower = 2 * [lower] if isinstance(lower, NUMERICAL_TYPES) else lower
            upper = [domain[0][1], domain[1][1]] if upper is None else upper
            upper = 2 * [upper] if isinstance(upper, NUMERICAL_TYPES) else upper
        else:
            # Determine boundaries
            x_data = self.x_array
            y_data = self.y_array
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            lower = [x_min, y_min] if lower is None else lower
            lower = 2 * [lower] if isinstance(lower, NUMERICAL_TYPES) else lower
            upper = [x_max, y_max] if upper is None else upper
            upper = 2 * [upper] if isinstance(upper, NUMERICAL_TYPES) else upper
            # Plot data points if force_data = True
            if force_data:
                axes.scatter(x_data, y_data, self.source[:, -1])
        # Create nodes to evaluate function
        x = np.linspace(lower[0], upper[0], samples[0])
        y = np.linspace(lower[1], upper[1], samples[1])
        mesh_x, mesh_y = np.meshgrid(x, y)

        # Evaluate function at all mesh nodes and convert it to matrix
        z = np.array(self.get_value(mesh_x.flatten(), mesh_y.flatten())).reshape(
            mesh_x.shape
        )
        z_min, z_max = z.min(), z.max()
        color_map = plt.colormaps[cmap]

        # Plot function
        if disp_type == "surface":
            surf = axes.plot_surface(
                mesh_x,
                mesh_y,
                z,
                rstride=1,
                cstride=1,
                cmap=color_map,
                linewidth=0,
                alpha=alpha,
                vmin=z_min,
                vmax=z_max,
            )
            figure.colorbar(surf)
        match disp_type:
            case "wireframe":
                axes.plot_wireframe(mesh_x, mesh_y, z, rstride=1, cstride=1)
            case "contour":
                figure.clf()
                contour_set = plt.contour(mesh_x, mesh_y, z)
                plt.clabel(contour_set, inline=1, fontsize=10)
            case "contourf":
                figure.clf()
                contour_set = plt.contour(mesh_x, mesh_y, z)
                plt.contourf(mesh_x, mesh_y, z)
                plt.clabel(contour_set, inline=1, fontsize=10)
        plt.title(self.title)
        axes.set_xlabel(self.__inputs__[0].title())
        axes.set_ylabel(self.__inputs__[1].title())
        axes.set_zlabel(self.__outputs__[0].title())
        show_or_save_plot(filename)

    @staticmethod
    def compare_plots(  # pylint: disable=too-many-statements
        plot_list,
        lower=None,
        upper=None,
        samples=1000,
        title="",
        xlabel="",
        ylabel="",
        force_data=False,
        force_points=False,
        return_object=False,
        show=True,
        *,
        filename=None,
    ):
        """Plots N 1-Dimensional Functions in the same plot, from a lower
        limit to an upper limit, by sampling the Functions several times in
        the interval.

        Parameters
        ----------
        plot_list : list[Tuple[Function,str]]
            List of Functions or list of tuples in the format (Function,
            label), where label is a string which will be displayed in the
            legend.
        lower : float, optional
            This represents the lower limit of the interval for plotting the
            Functions. If the Functions are defined by a dataset, the smallest
            value from the dataset is used. If no value is provided (None), and
            the Functions are of Function type, 0 is used as the default.
        upper : float, optional
            This represents the upper limit of the interval for plotting the
            Functions. If the Functions are defined by a dataset, the largest
            value from the dataset is used. If no value is provided (None), and
            the Functions are of Function type, 10 is used as the default.
        samples : int, optional
            The number of samples in which the functions will be evaluated for
            plotting it, which draws lines between each evaluated point.
            The default value is 1000.
        title : str, optional
            Title of the plot. Default value is an empty string.
        xlabel : str, optional
            X-axis label. Default value is an empty string.
        ylabel : str, optional
            Y-axis label. Default value is an empty string.
        force_data : bool, optional
            If Function is given by an interpolated dataset, setting force_data
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        force_points : bool, optional
            Setting force_points to True will plot all points, as a scatter, in
            which the Function was evaluated to plot it. Default value is
            False.
        return_object : bool, optional
            If True, returns the figure and axis objects. Default value is
            False.
        show : bool, optional
            If True, shows the plot. Default value is True.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        no_range_specified = lower is None and upper is None
        # Convert to list of tuples if list of Function was given
        plots = []
        for plot in plot_list:
            if isinstance(plot, (tuple, list)):
                plots.append(plot)
            else:
                plots.append((plot, ""))

        # Create plot figure
        fig, ax = plt.subplots()

        # Define a mesh and y values at mesh nodes for plotting
        if lower is None:
            lower = 0
            for plot in plots:
                if plot[0]._source_type is SourceType.ARRAY:
                    # Determine boundaries
                    x_min = plot[0].source[0, 0]
                    lower = x_min if x_min < lower else lower
        if upper is None:
            upper = 10
            for plot in plots:
                if plot[0]._source_type is SourceType.ARRAY:
                    # Determine boundaries
                    x_max = plot[0].source[-1, 0]
                    upper = x_max if x_max > upper else upper
        x = np.linspace(lower, upper, samples)

        # Iterate to plot all plots
        for plot in plots:
            # Deal with discrete data sets when no range is given
            if no_range_specified and plot[0]._source_type is SourceType.ARRAY:
                ax.plot(plot[0][:, 0], plot[0][:, 1], label=plot[1])
                if force_points:
                    ax.scatter(plot[0][:, 0], plot[0][:, 1], marker="o")
            else:
                # Calculate function at mesh nodes
                y = plot[0].get_value(x.tolist())
                # Plots function
                ax.plot(x, y, label=plot[1])
                if force_points:
                    ax.scatter(x, y, marker="o")

        # Plot data points if specified
        if force_data:
            for plot in plots:
                if plot[0]._source_type is SourceType.ARRAY:
                    x_data = plot[0].source[:, 0]
                    x_min, x_max = x_data[0], x_data[-1]
                    too_low = x_min >= lower
                    too_high = x_max <= upper
                    lo_ind = 0 if too_low else np.where(x_data >= lower)[0][0]
                    up_ind = (
                        len(x_data) - 1 if too_high else np.where(x_data <= upper)[0][0]
                    )
                    points = plot[0].source[lo_ind : (up_ind + 1), :].T.tolist()
                    ax.scatter(points[0], points[1], marker="o")

        # Setup legend
        if any(plot[1] for plot in plots):
            ax.legend(loc="best", shadow=True)

        # Turn on grid and set title and axis
        plt.grid(True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            show_or_save_plot(filename)

        if return_object:
            return fig, ax

    def __neg__(self):
        """Negates the Function object. The result has the same effect as
        multiplying the Function by -1.

        Returns
        -------
        Function
            The negated Function object.
        """
        if self._source_type is SourceType.SCALAR:
            return Function(
                -self._scalar_value,
                self.__inputs__,
                self.__outputs__,
            )
        elif self._source_type is SourceType.ARRAY:
            neg_source = self.source.copy()
            neg_source[:, -1] = -neg_source[:, -1]
            return Function(
                neg_source,
                self.__inputs__,
                self.__outputs__,
                self.__interpolation__,
                self.__extrapolation__,
            )
        else:
            if self.__dom_dim__ == 1:
                return Function(
                    lambda x: -self.source(x),
                    self.__inputs__,
                    self.__outputs__,
                )
            else:
                param_names = [f"x{i}" for i in range(self.__dom_dim__)]
                param_str = ", ".join(param_names)
                func_str = f"lambda {param_str}: -func({param_str})"
                return Function(
                    # pylint: disable=eval-used
                    eval(func_str, {"func": self.source}),
                    self.__inputs__,
                    self.__outputs__,
                )

    def __ge__(self, other):
        """Greater than or equal to comparison operator. It can be used to
        compare a Function object with a scalar or another Function object.
        This has the same effect as comparing numpy arrays.

        Note that it only works for Functions if at least one of them is
        defined by a set of points so that the bounds of the domain can be
        set.
        If both are defined by a set of points, they must have the same
        discretization.

        Parameters
        ----------
        other : scalar or Function

        Returns
        -------
        numpy.ndarray of bool
            The result of the comparison one by one.
        """
        other_is_function = isinstance(other, Function)

        if self._source_type is SourceType.ARRAY:
            if other_is_function:
                try:
                    return self.y_array >= other.y_array
                except AttributeError:
                    # Other is lambda based Function
                    return self.y_array >= other(self.x_array)
                except ValueError as exc:
                    raise ValueError(
                        "Comparison not supported between instances of the "
                        "Function class with different domain discretization."
                    ) from exc
            else:
                # Other is not a Function
                try:
                    return self.y_array >= other
                except TypeError as exc:
                    raise TypeError(
                        "Comparison not supported between instances of "
                        f"'Function' and '{type(other)}'."
                    ) from exc
        else:
            # self is lambda based Function
            if other_is_function:
                try:
                    return self(other.x_array) >= other.y_array
                except AttributeError as exc:
                    raise TypeError(
                        "Comparison not supported between two instances of "
                        "the Function class with callable sources."
                    ) from exc

    def __le__(self, other):
        """Less than or equal to comparison operator. It can be used to
        compare a Function object with a scalar or another Function object.
        This has the same effect as comparing numpy arrays.

        Note that it only works for Functions if at least one of them is
        defined by a set of points so that the bounds of the domain can be
        set.
        If both are defined by a set of points, they must have the same
        discretization.

        Parameters
        ----------
        other : scalar or Function

        Returns
        -------
        numpy.ndarray of bool
            The result of the comparison one by one.
        """
        other_is_function = isinstance(other, Function)

        if self._source_type is SourceType.ARRAY:
            if other_is_function:
                try:
                    return self.y_array <= other.y_array
                except AttributeError:
                    # Other is lambda based Function
                    return self.y_array <= other(self.x_array)
                except ValueError as exc:
                    raise ValueError(
                        "Operands should have the same discretization."
                    ) from exc
            else:
                # Other is not a Function
                try:
                    return self.y_array <= other
                except TypeError as exc:
                    raise TypeError(
                        "Comparison not supported between instances of "
                        f"'Function' and '{type(other)}'."
                    ) from exc
        else:
            # self is lambda based Function
            if other_is_function:
                try:
                    return self(other.x_array) <= other.y_array
                except AttributeError as exc:
                    raise TypeError(
                        "Comparison not supported between two instances of "
                        "the Function class with callable sources."
                    ) from exc

    def __gt__(self, other):
        """Greater than comparison operator. It can be used to compare a
        Function object with a scalar or another Function object. This has
        the same effect as comparing numpy arrays.

        Note that it only works for Functions if at least one of them is
        defined by a set of points so that the bounds of the domain can be
        set.
        If both are defined by a set of points, they must have the same
        discretization.

        Parameters
        ----------
        other : scalar or Function

        Returns
        -------
        numpy.ndarray of bool
            The result of the comparison one by one.
        """
        return ~self.__le__(other)

    def __lt__(self, other):
        """Less than comparison operator. It can be used to compare a
        Function object with a scalar or another Function object. This has
        the same effect as comparing numpy arrays.

        Note that it only works for Functions if at least one of them is
        defined by a set of points so that the bounds of the domain can be
        set.
        If both are defined by a set of points, they must have the same
        discretization.

        Parameters
        ----------
        other : scalar or Function

        Returns
        -------
        numpy.ndarray of bool
            The result of the comparison one by one.
        """
        return ~self.__ge__(other)

    # Define all possible algebraic operations
    def __arithmetic_operation(self, other, op):
        # pylint: disable=too-many-statements
        """Generic handler for arithmetic operations between a Function and
        another operand.

        Parameters
        ----------
        other : Function, int, float, callable
            The operand to combine with self.
        op : callable
            The binary operator to apply (e.g. operator.add).

        Returns
        -------
        Function
        """
        other_is_func = isinstance(other, Function)
        other_is_array = (
            other._source_type is SourceType.ARRAY if other_is_func else False
        )

        # If other is a scalar Function, extract its constant value so the
        # rest of the logic treats it like a plain number.
        other_is_scalar = (
            other._source_type is SourceType.SCALAR if other_is_func else False
        )
        if other_is_scalar:
            other = other._scalar_value
            other_is_func = False
            other_is_array = False

        inputs = self.__inputs__[:]
        interp = self.__interpolation__
        extrap = self.__extrapolation__
        dom_dim = self.__dom_dim__
        op_symbol = _OPERATOR_SYMBOLS.get(op, op.__name__)

        # For division, use _safe_truediv (with nan_to_num) only on
        # pre-computed arrays; callable lambdas use plain operator.truediv
        # to avoid overhead from nan_to_num on every evaluation.
        lambda_op = operator.truediv if op is _safe_truediv else op

        # Scalar self fast path: self is a constant value
        if self._source_type is SourceType.SCALAR:
            sv = self._scalar_value
            # SCALAR op number → SCALAR
            if isinstance(other, NUMERICAL_TYPES) or self.__is_single_element_array(
                other
            ):
                return Function(
                    lambda_op(sv, float(other)),
                    inputs,
                )
            # SCALAR op ARRAY Function → ARRAY
            if other_is_array:
                return Function(
                    np.column_stack((other._domain, op(sv, other._image))),
                    other.__inputs__[:],
                    f"({self.__outputs__[0]}{op_symbol}{other.__outputs__[0]})",
                    other.__interpolation__,
                    other.__extrapolation__,
                )
            # SCALAR op callable → CALLABLE
            if callable(other):
                if other_is_func:
                    other_dim = other.__dom_dim__
                    other_callable = (
                        other.get_value_opt if other_is_array else other.source
                    )
                    if other_dim > dom_dim:
                        inputs = other.__inputs__[:]
                else:
                    other_dim = len(signature(other).parameters)
                    other_callable = other
                # Reverse: op(sv, other_callable(...)) — the scalar is the
                # "other" and the callable is the "func" from make_arith_lambda's
                # perspective, with reverse=True.
                return Function(
                    self.__make_arith_lambda(
                        lambda_op, other_callable, sv, other_dim, reverse=True
                    ),
                    inputs,
                )

        # Fast path: both array-based with identical domains.
        # Domain equality implies equal dom_dim (same number of columns).
        if (
            self._source_type is SourceType.ARRAY
            and other_is_array
            and np.array_equal(self._domain, other._domain)
        ):
            return Function(
                np.column_stack((self._domain, op(self._image, other._image))),
                inputs,
                f"({self.__outputs__[0]}{op_symbol}{other.__outputs__[0]})",
                interp,
                extrap,
            )

        # Scalar path
        if isinstance(other, NUMERICAL_TYPES) or self.__is_single_element_array(other):
            if self._source_type is SourceType.ARRAY:
                return Function(
                    np.column_stack((self._domain, op(self._image, other))),
                    inputs,
                    f"({self.__outputs__[0]}{op_symbol}{other})",
                    interp,
                    extrap,
                )
            else:
                self_callable = (
                    self.source
                    if self._source_type is SourceType.CALLABLE
                    else self.get_value_opt
                )
                return Function(
                    self.__make_arith_lambda(lambda_op, self_callable, other, dom_dim),
                    inputs,
                )

        # Callable path
        if callable(other):
            if other_is_func:
                other_dim = other.__dom_dim__
                other_callable = other.get_value_opt if other_is_array else other.source
                if other_dim > dom_dim:
                    inputs = other.__inputs__[:]
            else:
                other_dim = len(signature(other).parameters)
                other_callable = other

            if dom_dim != 1 and other_dim != 1 and dom_dim != other_dim:
                raise TypeError(
                    f"The number of parameters in the function to be operated on "
                    f"({other_dim}) does not match the number of parameters of the "
                    f"Function ({dom_dim})."
                )

            self_callable = (
                self.source
                if self._source_type is SourceType.CALLABLE
                else self.get_value_opt
            )
            return Function(
                self.__make_arith_lambda(
                    lambda_op, self_callable, other_callable, dom_dim, other_dim
                ),
                inputs,
            )

        raise TypeError(
            f"Unsupported type for arithmetic operation '{op_symbol}': {type(other)}"
        )

    def __reverse_arithmetic_operation(self, other, op):
        """Handles reverse arithmetic operations where other is guaranteed
        to not be a Function instance.

        Parameters
        ----------
        other : int, float, callable
            The left-hand operand.
        op : callable
            The binary operator to apply as op(other, self).
        """
        inputs = self.__inputs__[:]
        interp = self.__interpolation__
        extrap = self.__extrapolation__
        dom_dim = self.__dom_dim__
        op_symbol = _OPERATOR_SYMBOLS.get(op, op.__name__)
        lambda_op = operator.truediv if op is _safe_truediv else op

        # Scalar self fast path: self is a constant value
        if self._source_type is SourceType.SCALAR:
            sv = self._scalar_value
            # number op SCALAR → SCALAR
            if isinstance(other, NUMERICAL_TYPES) or self.__is_single_element_array(
                other
            ):
                return Function(lambda_op(float(other), sv), inputs)
            # callable op SCALAR → CALLABLE
            if callable(other):
                other_dim = len(signature(other).parameters)
                return Function(
                    self.__make_arith_lambda(
                        lambda_op,
                        other,
                        sv,
                        other_dim,
                        0,
                        reverse=False,
                    ),
                    inputs,
                )

        # Scalar path
        if isinstance(other, NUMERICAL_TYPES) or self.__is_single_element_array(other):
            if self._source_type is SourceType.ARRAY:
                return Function(
                    np.column_stack((self._domain, op(other, self._image))),
                    inputs,
                    f"({other}{op_symbol}{self.__outputs__[0]})",
                    interp,
                    extrap,
                )
            else:
                self_callable = (
                    self.source
                    if self._source_type is SourceType.CALLABLE
                    else self.get_value_opt
                )
                return Function(
                    self.__make_arith_lambda(
                        lambda_op, self_callable, other, dom_dim, reverse=True
                    ),
                    inputs,
                )

        # Callable path — other is a plain callable, never a Function
        if callable(other):
            other_dim = len(signature(other).parameters)

            if dom_dim != 1 and other_dim != 1 and dom_dim != other_dim:
                raise TypeError(
                    f"The number of parameters in the function to be operated on "
                    f"({other_dim}) does not match the number of parameters of the "
                    f"Function ({dom_dim})."
                )
            self_callable = (
                self.source
                if self._source_type is SourceType.CALLABLE
                else self.get_value_opt
            )
            return Function(
                self.__make_arith_lambda(
                    lambda_op,
                    self_callable,
                    other,
                    dom_dim,
                    other_dim,
                    reverse=True,
                ),
                inputs,
            )

        raise TypeError(
            "Unsupported type for reverse arithmetic "
            f"operation '{op_symbol}': {type(other)}"
        )

    def __add__(self, other):
        """Sums a Function object and 'other', returns a new Function
        object which gives the result of the sum.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be added to. If other and self are Function
            objects which are based on a list of points, have the exact same
            domain (are defined in the same grid points) and have the same
            dimension, then a special implementation is used.
            This implementation is faster, however behavior between grid
            points is only interpolated, not calculated as it would be;
            the resultant Function has the same interpolation as self.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)+other(x).
        """
        return self.__arithmetic_operation(other, operator.add)

    def __radd__(self, other):
        """Sums 'other' and a Function object and returns a new Function
        object which gives the result of the sum.

        Parameters
        ----------
        other : int, float, callable
            What self will be added to.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)+self(x).
        """
        return self.__reverse_arithmetic_operation(other, operator.add)

    def __sub__(self, other):
        """Subtracts from a Function object and returns a new Function object
        which gives the result of the subtraction.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be subtracted by. If other and self are Function
            objects which are based on a list of points, have the exact same
            domain (are defined in the same grid points) and have the same
            dimension, then a special implementation is used.
            This implementation is faster, however behavior between grid
            points is only interpolated, not calculated as it would be;
            the resultant Function has the same interpolation as self.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)-other(x).
        """
        return self.__arithmetic_operation(other, operator.sub)

    def __rsub__(self, other):
        """Subtracts a Function object from 'other' and returns a new Function
        object which gives the result of the subtraction. Only implemented for
        1D domains.

        Parameters
        ----------
        other : int, float, callable
            What self will subtract from.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)-self(x).
        """
        return self.__reverse_arithmetic_operation(other, operator.sub)

    def __mul__(self, other):
        """Multiplies a Function object and returns a new Function object
        which gives the result of the multiplication.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be multiplied by. If other and self are Function
            objects which are based on a list of points, have the exact same
            domain (are defined in the same grid points) and have the same
            dimension, then a special implementation is used.
            This implementation is faster, however behavior between grid
            points is only interpolated, not calculated as it would be;
            the resultant Function has the same interpolation as self.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)*other(x).
        """
        return self.__arithmetic_operation(other, operator.mul)

    def __rmul__(self, other):
        """Multiplies 'other' by a Function object and returns a new Function
        object which gives the result of the multiplication.

        Parameters
        ----------
        other : int, float, callable
            What self will be multiplied by.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)*self(x).
        """
        return self.__reverse_arithmetic_operation(other, operator.mul)

    def __truediv__(self, other):
        """Divides a Function object and returns a new Function object
        which gives the result of the division.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be divided by. If other and self are Function
            objects which are based on a list of points, have the exact same
            domain (are defined in the same grid points) and have the same
            dimension, then a special implementation is used.
            This implementation is faster, however behavior between grid
            points is only interpolated, not calculated as it would be;
            the resultant Function has the same interpolation as self.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)/other(x).
        """
        return self.__arithmetic_operation(other, _safe_truediv)

    def __rtruediv__(self, other):
        """Divides 'other' by a Function object and returns a new Function
        object which gives the result of the division.

        Parameters
        ----------
        other : int, float, callable
            What self will divide.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)/self(x).
        """
        return self.__reverse_arithmetic_operation(other, _safe_truediv)

    def __pow__(self, other):
        """Raises a Function object to the power of 'other' and
        returns a new Function object which gives the result.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be raised to. If other and self are Function
            objects which are based on a list of points, have the exact same
            domain (are defined in the same grid points) and have the same
            dimension, then a special implementation is used.
            This implementation is faster, however behavior between grid
            points is only interpolated, not calculated as it would be;
            the resultant Function has the same interpolation as self.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)**other(x).
        """
        return self.__arithmetic_operation(other, operator.pow)

    def __rpow__(self, other):
        """Raises 'other' to the power of a Function object and returns
        a new Function object which gives the result.

        Parameters
        ----------
        other : int, float, callable
            The object that will be exponentiated by the function.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)**self(x).
        """
        return self.__reverse_arithmetic_operation(other, operator.pow)

    def __mod__(self, other):
        """Operator % as an alias for modulo operation."""
        return self.__arithmetic_operation(other, operator.mod)

    def __matmul__(self, other):
        """Operator @ as an alias for composition. Therefore, this
        method is a shorthand for Function.compose(other).

        Parameters
        ----------
        other : Function
            Function object to be composed with self.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(other(x)).

        See Also
        --------
        Function.compose
        """
        return self.compose(other)

    def integral(self, a, b, numerical=False):
        """Evaluate a definite integral of a 1-D Function in the interval
        from a to b.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        numerical : bool
            If True, forces the definite integral to be evaluated numerically.
            The current numerical method used is scipy.integrate.quad.
            If False, try to calculate using the precomputed analytical
            antiderivative when available (all 1-D array-based methods).
            Falls back to numerical integration otherwise.

        Returns
        -------
        ans : float
            Evaluated integral.
        """
        # Guarantee a < b
        integration_sign = np.sign(b - a)
        if integration_sign == -1:
            a, b = b, a
        elif integration_sign == 0:
            return 0.0  # b == a

        if (
            not numerical
            and self._source_type is SourceType.ARRAY
            and self.__dom_dim__ == 1
        ):
            ans = self._evaluator.definite_integral(a, b)
            return float(integration_sign * ans)

        ans, _ = integrate.quad(self, a, b, epsabs=1e-4, epsrel=1e-3, limit=1000)

        return float(integration_sign * ans)

    def differentiate(self, x, dx=1e-6, order=1):
        """Differentiate a Function object at a given point.

        Parameters
        ----------
        x : float
            Point at which to differentiate.
        dx : float
            Step size to use for numerical differentiation (fallback).
        order : int
            Order of differentiation.

        Returns
        -------
        ans : float
            Evaluated derivative.
        """
        if self._source_type is SourceType.ARRAY and self.__dom_dim__ == 1:
            try:
                if order == 1:
                    return float(self._evaluator.derivative(x))
                else:
                    return float(self._evaluator.second_derivative(x))
            except (NotImplementedError, AttributeError):
                pass

        match order:
            case 1:
                return (self.get_value_opt(x + dx) - self.get_value_opt(x - dx)) / (
                    2 * dx
                )
            case 2:
                return (
                    self.get_value_opt(x + dx)
                    - 2 * self.get_value_opt(x)
                    + self.get_value_opt(x - dx)
                ) / dx**2

    def differentiate_complex_step(self, x, dx=1e-200, order=1):
        """Differentiate a Function object at a given point using the complex
        step method. This method can be faster than ``Function.differentiate``
        since it requires only one evaluation of the function. However, the
        evaluated function must accept complex numbers as input.

        Parameters
        ----------
        x : float
            Point at which to differentiate.
        dx : float, optional
            Step size to use for numerical differentiation, by default 1e-200.
        order : int, optional
            Order of differentiation, by default 1. Right now, only first order
            derivative is supported.

        Returns
        -------
        float
            The real part of the derivative of the function at the given point.

        References
        ----------
        [1] https://mdolab.engin.umich.edu/wiki/guide-complex-step-derivative-approximation
        """
        if order == 1:
            return float(self.get_value_opt(x + dx * 1j).imag / dx)
        else:  # pragma: no cover
            raise NotImplementedError(
                "Only 1st order derivatives are supported yet. Set order=1."
            )

    def identity_function(self):
        """Returns a Function object that correspond to the identity mapping,
        i.e. f(x) = x.
        If the Function object is defined on an array, the identity Function
        follows the same discretization, and has linear interpolation and
        extrapolation.
        If the Function is defined by a lambda, the identity Function is the
        identity map 'lambda x: x'.

        Returns
        -------
        result : Function
            A Function object that corresponds to the identity mapping.
        """
        if self._source_type is SourceType.ARRAY:
            return Function(
                np.column_stack((self.x_array, self.x_array)),
                inputs=self.__inputs__,
                outputs=f"identity of {self.__outputs__}",
                interpolation="linear",
                extrapolation="natural",
            )
        else:
            return Function(
                lambda x: x,
                inputs=self.__inputs__,
                outputs=f"identity of {self.__outputs__}",
            )

    def derivative_function(self, order=1):
        """Returns a Function object which gives the derivative of the Function object.

        Returns
        -------
        result : Function
            A Function object which gives the derivative of self.
        """
        inputs = self.__inputs__[:]

        if order == 1:
            outputs = f"d({self.__outputs__[0]})/d({inputs[0]})"
        elif order == 2:
            outputs = f"d^2({self.__outputs__[0]})/d({inputs[0]})^2"
        else:
            raise NotImplementedError(
                "Only first and second derivatives are supported."
            )

        if self._source_type is SourceType.SCALAR:
            return Function(0.0, inputs, outputs)

        if self._source_type is SourceType.ARRAY:
            if self.__dom_dim__ == 1 and hasattr(self._evaluator, "derivative"):
                xs = self.x_array
                ys = (
                    self._evaluator.derivative(xs)
                    if order == 1
                    else self._evaluator.second_derivative(xs)
                )
                source = np.column_stack((xs, ys))
            else:
                dy = np.gradient(self.y_array, self.x_array)
                ys = dy if order == 1 else np.gradient(dy, self.x_array)
                source = np.column_stack((self.x_array, ys))
        else:

            def source_function(x):
                return self.differentiate(x, order=order)

            source = source_function

        return Function(
            source, inputs, outputs, self.__interpolation__, self.__extrapolation__
        )

    def integral_function(self, lower=None, upper=None, datapoints=100):
        """Returns a Function object representing the integral of the Function
        object.

        Parameters
        ----------
        lower : scalar, optional
            The lower limit of the interval in which the function is to be
            evaluated at. If the Function is given by a dataset, the default
            value is the start of the dataset.
        upper : scalar, optional
            The upper limit of the interval in which the function is to be
            evaluated at. If the Function is given by a dataset, the default
            value is the end of the dataset.
        datapoints : int, optional
            The number of points in which the integral will be evaluated for
            plotting it, which draws lines between each evaluated point.
            The default value is 100.

        Returns
        -------
        result : Function
            The integral of the Function object.
        """
        if self._source_type is SourceType.SCALAR:
            c = self._scalar_value
            lower = 0 if lower is None else lower
            return Function(
                lambda x: c * (x - lower),
                inputs=self.__inputs__,
                outputs=[o + " Integral" for o in self.__outputs__],
            )

        if self._source_type is SourceType.ARRAY:
            lower = self.source[0, 0] if lower is None else lower
            upper = self.source[-1, 0] if upper is None else upper

            x_data = np.linspace(lower, upper, datapoints)

            if self.__dom_dim__ == 1:
                y_data = self._evaluator.definite_integral(lower, x_data)
            else:
                raise NotImplementedError(
                    "Integral function is only implemented for 1-D array-based Functions."
                )

            return Function(
                np.column_stack((x_data, y_data)),
                inputs=self.__inputs__,
                outputs=[o + " Integral" for o in self.__outputs__],
            )
        else:
            lower = 0 if lower is None else lower
            return Function(
                lambda x: self.integral(lower, x),
                inputs=self.__inputs__,
                outputs=[o + " Integral" for o in self.__outputs__],
            )

    def isbijective(self):
        """Checks whether the Function is bijective. Only applicable to
        Functions whose source is a list of points, raises an error otherwise.

        Returns
        -------
        result : bool
            True if the Function is bijective, False otherwise.
        """
        if self._source_type is SourceType.ARRAY:
            x_data_distinct = set(self.x_array)
            y_data_distinct = set(self.y_array)
            distinct_map = set(zip(x_data_distinct, y_data_distinct))
            return len(distinct_map) == len(x_data_distinct) == len(y_data_distinct)
        else:
            raise TypeError(
                "`isbijective()` method only supports Functions whose "
                "source is an array."
            )

    def is_strictly_bijective(self):
        """Checks whether the Function is "strictly" bijective.
        Only applicable to Functions whose source is a list of points,
        raises an error otherwise.

        Notes
        -----
        By "strictly" bijective, this implementation considers the
        list-of-points-defined Function bijective between each consecutive pair
        of points. Therefore, the Function may be flagged as not bijective even
        if the mapping between the set of points which define the Function is
        bijective.

        Returns
        -------
        result : bool
            True if the Function is "strictly" bijective, False otherwise.

        Examples
        --------
        >>> f = Function([[0, 0], [1, 1], [2, 4]])
        >>> f.isbijective() == True
        True
        >>> f.is_strictly_bijective() == True
        np.True_

        >>> f = Function([[-1, 1], [0, 0], [1, 1], [2, 4]])
        >>> f.isbijective()
        False
        >>> f.is_strictly_bijective()
        np.False_

        A Function which is not "strictly" bijective, but is bijective, can be
        constructed as x^2 defined at -1, 0 and 2.

        >>> f = Function([[-1, 1], [0, 0], [2, 4]])
        >>> f.isbijective()
        True
        >>> f.is_strictly_bijective()
        np.False_
        """
        if self._source_type is SourceType.ARRAY:
            # Assuming domain is sorted, range must also be
            y_data = self.y_array
            # Both ascending and descending order means Function is bijective
            y_data_diff = np.diff(y_data)
            return np.all(y_data_diff >= 0) or np.all(y_data_diff <= 0)
        else:
            raise TypeError(
                "`is_strictly_bijective()` method only supports Functions "
                "whose source is an array."
            )

    def inverse_function(self, approx_func=None, tol=1e-4):
        """
        Returns the inverse of the Function. The inverse function of F is a
        function that undoes the operation of F. The inverse of F exists if
        and only if F is bijective. Makes the domain the range and the range
        the domain.

        If the Function is given by a list of points, the method
        `is_strictly_bijective()` is called and an error is raised if the
        Function is not bijective.
        If the Function is given by a function, its bijection is not
        checked and may lead to inaccuracies outside of its bijective region.

        Parameters
        ----------
        approx_func : callable, optional
            A function that approximates the inverse of the Function. This
            function is used to find the starting guesses for the inverse
            root finding algorithm. This is better used when the inverse
            in complex but has a simple approximation or when the root
            finding algorithm performs poorly due to default start point.
            The default is None in which case the starting point is zero.

        tol : float, optional
            The tolerance for the inverse root finding algorithm. The default
            is 1e-4.

        Returns
        -------
        result : Function
            A Function whose domain and range have been inverted.
        """
        if self._source_type is SourceType.ARRAY:
            if self.is_strictly_bijective():
                # Swap the columns
                source = np.flip(self.source, axis=1)
            else:
                raise ValueError(
                    "Function is not bijective, so it does not have an inverse."
                )
        else:
            if approx_func is not None:

                def source_function(x):
                    return self.find_input(x, start=approx_func(x), tol=tol)

            else:

                def source_function(x):
                    return self.find_input(x, start=0, tol=tol)

            source = source_function
        return Function(
            source,
            inputs=self.__outputs__,
            outputs=self.__inputs__,
            interpolation=self.__interpolation__,
            extrapolation=self.__extrapolation__,
        )

    def find_input(self, val, start, tol=1e-4):
        """
        Finds the optimal input for a given output.

        Parameters
        ----------
        val : int, float
            The value of the output.
        start : int, float
            Initial guess of the output.
        tol : int, float
            Tolerance for termination.

        Returns
        -------
        result : ndarray
            The value of the input which gives the output closest to val.
        """
        return optimize.root(
            lambda x: self.get_value(x)[0] - val,
            start,
            tol=tol,
        ).x[0]

    def average(self, lower, upper):
        """
        Returns the average of the function.

        Parameters
        ----------
        lower : float
            Lower point of the region that the average will be calculated at.
        upper : float
            Upper point of the region that the average will be calculated at.

        Returns
        -------
        result : float
            The average of the function.
        """
        return self.integral(lower, upper) / (upper - lower)

    def average_function(self, lower=None):
        """
        Returns a Function object representing the average of the Function
        object.

        Parameters
        ----------
        lower : float
            Lower limit of the new domain. Only required if the Function's
            source is a callable instead of a list of points.

        Returns
        -------
        result : Function
            The average of the Function object.
        """
        if self._source_type is SourceType.ARRAY:
            lower = self.source[0, 0] if lower is None else lower
            upper = self.source[-1, 0]

            x_data = np.linspace(lower, upper, 100)
            y_data = np.zeros_like(x_data)

            y_data[0] = self.get_value_opt(lower)

            if self.__dom_dim__ == 1:
                integrals = self._evaluator.definite_integral(lower, x_data[1:])
                y_data[1:] = integrals / (x_data[1:] - lower)
            else:
                y_data[1:] = np.array([self.average(lower, x) for x in x_data[1:]])

            return Function(
                np.column_stack((x_data, y_data)),
                inputs=self.__inputs__,
                outputs=[o + " Average" for o in self.__outputs__],
            )
        else:
            lower = 0 if lower is None else lower
            return Function(
                lambda x: self.average(lower, x),
                inputs=self.__inputs__,
                outputs=[o + " Average" for o in self.__outputs__],
            )

    def compose(self, func, extrapolate=False):
        """
        Returns a Function object which is the result of inputting a function
        into a function (i.e. f(g(x))). The domain will become the domain of
        the input function and the range will become the range of the original
        function.

        Parameters
        ----------
        func : Function
            The function to be inputted into the function.

        extrapolate : bool, optional
            Whether or not to extrapolate the function if the input function's
            range is outside of the original function's domain. The default is
            False.

        Returns
        -------
        result : Function
            The result of inputting the function into the function.
        """
        # Check if the input is a function
        if not isinstance(func, Function):  # pragma: no cover
            raise TypeError("Input must be a Function object.")

        if (
            self._source_type is SourceType.ARRAY
            and func._source_type is SourceType.ARRAY
        ):
            # Perform bounds check for composition
            if not extrapolate:  # pragma: no cover
                if func.min < self.x_initial or func.max > self.x_final:
                    raise ValueError(
                        f"Input Function image {func.min, func.max} must be within "
                        f"the domain of the Function {self.x_initial, self.x_final}."
                    )

            return Function(
                np.concatenate(([func.x_array], [self(func.y_array)])).T,
                inputs=func.__inputs__,
                outputs=self.__outputs__,
                interpolation=self.__interpolation__,
                extrapolation=self.__extrapolation__,
            )
        else:
            return Function(
                lambda x: self(func(x)),
                inputs=func.__inputs__,
                outputs=self.__outputs__,
                interpolation=self.__interpolation__,
                extrapolation=self.__extrapolation__,
            )

    def savetxt(
        self,
        filename,
        lower=None,
        upper=None,
        samples=None,
        fmt="%.6f",
        delimiter=",",
        newline="\n",
        encoding=None,
    ):
        r"""Save a Function object to a text file. The first line is the header
        with inputs and outputs. The following lines are the data. The text file
        can have any extension, but it is recommended to use .csv or .txt.

        Parameters
        ----------
        filename : str
            The name of the file to be saved, with the extension.
        lower : float or int, optional
            The lower bound of the range for which data is to be generated.
            This is required if the source is a callable function.
        upper : float or int, optional
            The upper bound of the range for which data is to be generated.
            This is required if the source is a callable function.
        samples : int, optional
            The number of sample points to generate within the specified range.
            This is required if the source is a callable function.
        fmt : str, optional
            The format string for each line of the file, by default "%.6f".
        delimiter : str, optional
            The string used to separate values, by default ",".
        newline : str, optional
            The string used to separate lines in the file, by default "\n".
        encoding : str, optional
            The encoding to be used for the file, by default None (which means
            using the system default encoding).

        Raises
        ------
        ValueError
            Raised if `lower`, `upper`, and `samples` are not provided when
            the source is a callable function. These parameters are necessary
            to generate the data points for saving.
        """
        # create the header
        header_line = delimiter.join(self.__inputs__ + self.__outputs__)

        # create the datapoints
        if self._source_type is not SourceType.ARRAY:
            if lower is None or upper is None or samples is None:  # pragma: no cover
                raise ValueError(
                    "If the source is a callable, lower, upper and samples"
                    + " must be provided."
                )
            # Generate the data points using the callable
            data_points = self.set_discrete(
                lower, upper, samples, mutate_self=False
            ).source
        else:
            # If the source is already an array, use it as is
            data_points = self.source

            if lower and upper and samples:
                data_points = self.set_discrete(
                    lower, upper, samples, mutate_self=False
                ).source

        # export to a file
        with open(filename, "w", encoding=encoding) as file:
            file.write(header_line + newline)
            np.savetxt(file, data_points, fmt=fmt, delimiter=delimiter, newline=newline)

    @staticmethod
    def __is_single_element_array(var):
        return isinstance(var, np.ndarray) and var.size == 1

    # Input validators
    def __validate_source(self, source):  # pylint: disable=too-many-statements
        """Used to validate the source parameter for creating a Function object.

        Parameters
        ----------
        source : np.ndarray, callable, str, Path, Function, list
            The source data of the Function object. This can be a numpy array,
            a callable function, a string or Path object to a csv or txt file,
            a Function object, or a list of numbers.

        Returns
        -------
        np.ndarray, callable
            The validated source parameter.

        Raises
        ------
        ValueError
            If the source is not a valid type or if the source is not a 2D array
            or a callable function.
        """
        if isinstance(source, Function):
            return source.get_source()

        if isinstance(source, (str, Path)):
            # Read csv or txt files and create a numpy array
            try:
                source = np.loadtxt(source, delimiter=",", dtype=np.float64)
            except ValueError:
                with open(source, "r") as file:
                    header, *data = file.read().splitlines()

                header = [label.strip("'").strip('"') for label in header.split(",")]
                source = np.loadtxt(data, delimiter=",", dtype=np.float64)

                if len(source[0]) == len(header):
                    if self.__inputs__ is None:
                        self.__inputs__ = header[:-1]
                    if self.__outputs__ is None:
                        self.__outputs__ = [header[-1]]
            except Exception as e:  # pragma: no cover
                raise ValueError(
                    "Could not read the csv or txt file to create Function source."
                ) from e

        if isinstance(source, Iterable):
            # Triggers an error if source is not a list of numbers
            if self.__interpolation__ == "regular_grid":
                return self.__process_grid_source(source)

            source = np.array(source, dtype=np.float64)

            # Checks if 2D array
            if len(source.shape) != 2:
                raise ValueError(
                    "Source must be a 2D array in the form [[x1, x2 ..., xn, y], ...]."
                )

            source_len, source_dim = source.shape
            if not source_len == 1:  # do not check for one point Functions
                if source_len < source_dim:
                    raise ValueError(
                        "Too few data points to define a domain. The number of rows "
                        "must be greater than or equal to the number of columns."
                    )

            return source

        if isinstance(source, NUMERICAL_TYPES):
            # Return scalar directly — set_source will handle it as SCALAR
            return float(source)

        # If source is a callable function
        return source

    def __validate_inputs(self, inputs):
        """Used to validate the inputs parameter for creating a Function object.
        It sets a default value if it is not provided.

        Parameters
        ----------
        inputs : list of str, None
            The name(s) of the input variable(s). If None, defaults to "Scalar".

        Returns
        -------
        list
            The validated inputs parameter.
        """
        if self.__dom_dim__ == 1:
            if inputs is None:
                return ["Scalar"]
            if isinstance(inputs, str):
                return [inputs]
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 1:
                    return inputs
            # pragma: no cover
            raise ValueError(
                "Inputs must be a string or a list of strings with "
                "the length of the domain dimension."
            )
        if self.__dom_dim__ > 1:
            if inputs is None:
                return [f"Input {i + 1}" for i in range(self.__dom_dim__)]
            if isinstance(inputs, list):
                if len(inputs) == self.__dom_dim__ and all(
                    isinstance(i, str) for i in inputs
                ):
                    return inputs
            # pragma: no cover
            raise ValueError(
                "Inputs must be a list of strings with "
                "the length of the domain dimension."
            )

    def __validate_outputs(self, outputs):
        """Used to validate the outputs parameter for creating a Function object.
        It sets a default value if it is not provided.

        Parameters
        ----------
        outputs : str, list of str, None
            The name of the output variables. If None, defaults to "Scalar".

        Returns
        -------
        list
            The validated outputs parameter.
        """
        if outputs is None:
            return ["Scalar"]
        if isinstance(outputs, str):
            return [outputs]
        if isinstance(outputs, (list, tuple)):
            if len(outputs) > 1:
                raise ValueError(
                    "Output must either be a string or a list of strings with "
                    + f"one item. It currently has dimension ({len(outputs)})."
                )
            return outputs

    def __validate_interpolation(self, interpolation):
        if self.__dom_dim__ == 1:
            # possible interpolation values: linear, polynomial, akima and spline
            if interpolation is None:
                interpolation = "spline"
            elif interpolation.lower() not in [
                "spline",
                "linear",
                "polynomial",
                "akima",
                "pchip",
            ]:
                warnings.warn(
                    "Interpolation method set to 'spline' because the "
                    f"{interpolation} method is not supported."
                )
                interpolation = "spline"
        ## multiple dimensions
        elif self.__dom_dim__ > 1:
            if interpolation is None:
                interpolation = "shepard"
            if interpolation.lower() not in [
                "shepard",
                "linear",
                "rbf",
                "regular_grid",
            ]:
                warnings.warn(
                    (
                        "Interpolation method set to 'shepard'. The methods "
                        "'linear', 'shepard', 'rbf' and 'regular_grid' are supported for "
                        "multiple dimensions."
                    ),
                )
                interpolation = "shepard"
        return interpolation

    def __validate_extrapolation(self, extrapolation):
        if self.__dom_dim__ == 1:
            if extrapolation is None:
                extrapolation = "constant"
            elif extrapolation.lower() not in ["constant", "natural", "zero"]:
                warnings.warn(
                    "Extrapolation method set to 'constant' because the "
                    f"{extrapolation} method is not supported."
                )
                extrapolation = "constant"

        ## multiple dimensions
        elif self.__dom_dim__ > 1:
            if extrapolation is None:
                extrapolation = "natural"
            if extrapolation.lower() not in ["constant", "natural", "zero"]:
                warnings.warn(
                    "Extrapolation method set to 'natural' because the "
                    f"{extrapolation} method is not supported."
                )
                extrapolation = "natural"
        return extrapolation

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        """Serializes the Function instance to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the Function's attributes.
        """
        source = self.source

        if callable(source):
            if kwargs.get("allow_pickle", True):
                source = to_hex_encode(source)
            else:
                source = source.__name__

        return {
            "source": source,
            "title": self.title,
            "inputs": self.__inputs__,
            "outputs": self.__outputs__,
            "interpolation": self.__interpolation__,
            "extrapolation": self.__extrapolation__,
        }

    @classmethod
    def from_grid(
        cls,
        grid_data,
        axes,
        inputs=None,
        outputs=None,
        interpolation="regular_grid",
        extrapolation="constant",
        flatten_for_compatibility=True,
        **kwargs,
    ):  # pylint: disable=too-many-statements #TODO: Refactor this method into smaller methods
        """Creates a Function from N-dimensional grid data.

        This method is designed for structured grid data, such as CFD simulation
        results where values are computed on a regular grid. It uses
        scipy.interpolate.RegularGridInterpolator for efficient interpolation.

        Parameters
        ----------
        grid_data : ndarray
            N-dimensional array containing the function values on the grid.
            For example, for a 3D function Cd(M, Re, α), this would be a 3D array
            where grid_data[i, j, k] = Cd(M[i], Re[j], α[k]).
        axes : list of ndarray
            List of 1D arrays defining the grid points along each axis.
            Each array should be sorted in ascending order.
            For example: [M_axis, Re_axis, alpha_axis].
        inputs : list of str, optional
            Names of the input variables. If None, generic names will be used.
            For example: ['Mach', 'Reynolds', 'Alpha'].
        outputs : str, optional
            Name of the output variable. For example: 'Cd'.
        interpolation : str, optional
            Interpolation method. Default is 'regular_grid'.
            Currently only 'regular_grid' is supported for grid data.
        extrapolation : str, optional
            Extrapolation behavior. Default is ``'constant'`` which clamps to
            edge values. Supported options are::

                'constant'
                    Use nearest edge value for out-of-bounds points (clamp).
                'zero'
                    Return zero for out-of-bounds points.
                'natural'
                    Use the interpolator's natural behavior: when the
                    underlying ``RegularGridInterpolator`` is created with
                    ``fill_value=None`` and ``method='linear'``, this results
                    in linear extrapolation based on the edge gradients.

            If an unsupported extrapolation value is supplied a ``ValueError``
            is raised.
        flatten_for_compatibility : bool, optional
            If True (default), creates flattened ``_domain``, ``_image``, and
            ``source`` arrays for backward compatibility with existing Function
            methods and serialization. For large N-dimensional grids (e.g.,
            100x100x100 points), this requires O(n^d) additional memory where n
            is the typical axis length and d is the number of dimensions.
            Set to False to skip this flattening and reduce memory usage if
            compatibility with legacy code paths is not required.
        **kwargs : dict, optional
            Additional arguments passed to the Function constructor.

        Returns
        -------
        Function
            A Function object using RegularGridInterpolator for evaluation.

        Notes
        -----
        - Grid data must be on a regular (structured) grid.
        - For unstructured data, use the regular Function constructor with
          scattered points.
        - Extrapolation with 'constant' mode uses the nearest edge values,
          which is appropriate for aerodynamic coefficients where extrapolation
          beyond the data range should be avoided.

        Examples
        --------
        >>> import numpy as np
        >>> # Create 3D drag coefficient data
        >>> mach = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        >>> reynolds = np.array([1e5, 5e5, 1e6])
        >>> alpha = np.array([0.0, 2.0, 4.0, 6.0])
        >>> # Create a simple drag coefficient function
        >>> M, Re, A = np.meshgrid(mach, reynolds, alpha, indexing='ij')
        >>> cd_data = 0.3 + 0.1 * M + 1e-7 * Re + 0.01 * A
        >>> # Create Function object
        >>> cd_func = Function.from_grid(
        ...     cd_data,
        ...     [mach, reynolds, alpha],
        ...     inputs=['Mach', 'Reynolds', 'Alpha'],
        ...     outputs='Cd'
        ... )
        >>> # Evaluate at a point
        >>> cd_func(1.2, 3e5, 3.0)
        0.48000000000000004

        """
        # Validate inputs
        if not isinstance(grid_data, np.ndarray):
            grid_data = np.array(grid_data)

        if not isinstance(axes, (list, tuple)):
            raise ValueError("axes must be a list or tuple of 1D arrays")

        # Ensure all axes are numpy arrays
        axes = [
            np.array(axis) if not isinstance(axis, np.ndarray) else axis
            for axis in axes
        ]

        # Check dimensions match
        if len(axes) != grid_data.ndim:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match grid_data dimensions "
                f"({grid_data.ndim})"
            )

        # Check each axis matches corresponding grid dimension and is sorted
        for i, axis in enumerate(axes):
            if len(axis) != grid_data.shape[i]:
                raise ValueError(
                    f"Axis {i} has {len(axis)} points but grid dimension {i} "
                    f"has {grid_data.shape[i]} points"
                )
            # Check if axis is sorted in ascending order
            if not np.all(np.diff(axis) > 0):
                warnings.warn(
                    f"Axis {i} is not strictly sorted in ascending order. "
                    "RegularGridInterpolator requires sorted axes. "
                    "This may cause unexpected interpolation results.",
                    UserWarning,
                )

        # Set default inputs if not provided
        if inputs is None:
            inputs = [f"x{i}" for i in range(len(axes))]
        elif len(inputs) != len(axes):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) must match number of axes ({len(axes)})"
            )

        func = cls.__new__(cls)

        allowed_extrap = ("constant", "zero", "natural")
        if extrapolation not in allowed_extrap:
            raise ValueError(
                "Unsupported extrapolation for grid interpolation. "
                f"Supported values: {allowed_extrap}"
            )

        func._grid_axes = axes
        func._grid_data = grid_data

        func._evaluator = build_interpolation_evaluator(
            method="regular_grid",
            extrap_method=extrapolation,
            dom_dim=len(axes),
            grid_axes=axes,
            grid_data=grid_data,
        )

        if flatten_for_compatibility:
            mesh = np.meshgrid(*axes, indexing="ij")
            domain_points = np.column_stack([m.ravel() for m in mesh])
            func._domain = domain_points
            func._image = grid_data.ravel()
            func.source = np.column_stack([domain_points, func._image])
        else:
            func._domain = None
            func._image = None
            func.source = None

        func.__inputs__ = inputs
        func.__outputs__ = outputs if outputs is not None else "f"
        func.__interpolation__ = interpolation
        func.__extrapolation__ = extrapolation
        func.title = kwargs.get("title", None)
        func.__img_dim__ = 1
        func.__cropped_domain__ = None
        func._source_type = SourceType.ARRAY
        func.__dom_dim__ = len(axes)

        func.x_array = axes[0]
        func.x_initial, func.x_final = float(axes[0][0]), float(axes[0][-1])

        if flatten_for_compatibility:
            func.y_array = func._image
            func.y_initial = float(func._image.min())
            func.y_final = float(func._image.max())
        else:
            func.y_array = None
            func.y_initial = float(grid_data.min())
            func.y_final = float(grid_data.max())

        if len(axes) > 2:
            func.z_array = axes[2]
            func.z_initial, func.z_final = axes[2][0], axes[2][-1]

        func.set_inputs(inputs)
        func.set_outputs(outputs)
        func.set_title(func.title)

        func.set_get_value_opt()

        return func

    @classmethod
    def from_dict(cls, func_dict):
        """Creates a Function instance from a dictionary.

        Parameters
        ----------
        func_dict
            The JSON like Function dictionary.
        """
        source = func_dict["source"]
        if func_dict["interpolation"] is None and func_dict["extrapolation"] is None:
            if isinstance(source, str):
                source = from_hex_decode(source)

        return cls(
            source=source,
            interpolation=func_dict["interpolation"],
            extrapolation=func_dict["extrapolation"],
            inputs=func_dict["inputs"],
            outputs=func_dict["outputs"],
            title=func_dict["title"],
        )

    @staticmethod
    def __make_arith_lambda(
        operator, func, other, func_dim, other_dim=0, reverse=False
    ):
        # pylint: disable=function-redefined
        """Creates a callable for arithmetic operations between
        multidimensional functions, without using eval.

        Parameters
        ----------
        operator : callable
            The binary mathematical operation to perform.
        func : callable
            The first operand (always self's get_value_opt).
        other : callable or scalar
            The second operand.
        func_dim : int
            Number of positional input parameters of func.
        other_dim : int, optional
            Number of positional input parameters of other. 0 means scalar.
        reverse : bool, optional
            If True, reverses operand order: operator(other(...), func(...)).
        """
        # Optimized 1D×1D case: avoid *args overhead
        if func_dim == 1:
            if other_dim == 1:
                if reverse:
                    return lambda x: operator(other(x), func(x))
                return lambda x: operator(func(x), other(x))
            elif other_dim == 0:
                if reverse:
                    return lambda x: operator(other, func(x))
                return lambda x: operator(func(x), other)

        # Scalar case
        if other_dim == 0:
            if reverse:

                def result(*args):
                    return operator(other, func(*args[:func_dim]))
            else:

                def result(*args):
                    return operator(func(*args[:func_dim]), other)

            result.__dom_dim__ = func_dim

            return result

        # General multidimensional callable case
        if reverse:

            def result(*args):
                return operator(other(*args[:other_dim]), func(*args[:func_dim]))
        else:

            def result(*args):
                return operator(func(*args[:func_dim]), other(*args[:other_dim]))

        result.__dom_dim__ = max(func_dim, other_dim)

        return result


def funcify_method(*args, **kwargs):  # pylint: disable=too-many-statements
    """Decorator factory to wrap methods as Function objects and save them as
    cached properties.

    Parameters
    ----------
    *args : list
        Positional arguments to be passed to rocketpy.Function.
    **kwargs : dict
        Keyword arguments to be passed to rocketpy.Function.

    Returns
    -------
    decorator : function
        Decorator function to wrap callables as Function objects.

    Examples
    --------
    There are 3 types of methods that this decorator supports:

    1. Method which returns a valid rocketpy.Function source argument.

    >>> from rocketpy.mathutils import funcify_method
    >>> class Example():
    ...     @funcify_method(inputs=['x'], outputs=['y'])
    ...     def f(self):
    ...         return lambda x: x**2
    >>> example = Example()
    >>> example.f
    'Function from R1 to R1 : (x) → (y)'

    Normal algebra can be performed afterwards:

    >>> g = 2*example.f + 3
    >>> g(2)
    11

    2. Method which returns a rocketpy.Function instance. An interesting use is
    to reset input and output names after algebraic operations.

    >>> class Example():
    ...     @funcify_method(inputs=['x'], outputs=['x**3'])
    ...     def cube(self):
    ...         f = Function(lambda x: x**2)
    ...         g = Function(lambda x: x**5)
    ...         return g / f
    >>> example = Example()
    >>> example.cube
    'Function from R1 to R1 : (x) → (x**3)'

    3. Method which is itself a valid rocketpy.Function source argument.

    >>> class Example():
    ...     @funcify_method('x', 'f(x)')
    ...     def f(self, x):
    ...         return x**2
    >>> example = Example()
    >>> example.f
    'Function from R1 to R1 : (x) → (f(x))'

    In order to reset the cache, just delete the attribute from the instance:

    >>> del example.f

    Once it is requested again, it will be re-created as a new Function object:

    >>> example.f
    'Function from R1 to R1 : (x) → (f(x))'
    """
    func = None
    if len(args) == 1 and callable(args[0]):
        func = args[0]
        args = []

    class funcify_method_decorator:
        """Decorator class to transform a cached property that is being defined
        inside a class to a Function object. This improves readability of the
        code since it will not require the user to directly invoke the Function
        class.
        """

        # pylint: disable=C0103,R0903
        def __init__(self, func):
            self.func = func
            self.attrname = None
            self.__doc__ = func.__doc__

        def __set_name__(self, owner, name):
            self.attrname = name

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            cache = instance.__dict__
            try:
                # If cache is ready, return it
                val = cache[self.attrname]
            except KeyError:
                # If cache is not ready, create it
                try:
                    # Handle methods which return Function instances
                    val = self.func(instance).reset(*args, **kwargs)
                except AttributeError:
                    # Handle methods which return a valid source
                    source = self.func(instance)
                    val = Function(source, *args, **kwargs)
                except TypeError:
                    # Handle methods which are the source themselves
                    def source_function(*_):
                        return self.func(instance, *_)

                    source = source_function
                    val = Function(source, *args, **kwargs)
                # pylint: disable=W0201
                val.__doc__ = self.__doc__
                val.__cached__ = True
                cache[self.attrname] = val
            return val

    if func:
        return funcify_method_decorator(func)
    else:
        return funcify_method_decorator


def reset_funcified_methods(instance):
    """Resets all the funcified methods of the instance. It does so by
    deleting the current Functions, which will make the interpreter redefine
    them when they are called. This is useful when the instance has changed
    and the methods need to be recalculated.

    Parameters
    ----------
    instance : object
        The instance of the class whose funcified methods will be recalculated.
        The class must have a mutable __dict__ attribute.

    Return
    ------
    None
    """
    for key in list(instance.__dict__):
        if hasattr(instance.__dict__[key], "__cached__"):
            instance.__dict__.pop(key)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    results = doctest.testmod()
    if results.failed < 1:
        logger.info("All the %d tests passed!", results.attempted)
    else:
        logger.error("%d out of %d tests failed.", results.failed, results.attempted)
