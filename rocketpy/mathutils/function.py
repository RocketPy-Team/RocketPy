""" The mathutils/function.py is a rocketpy module totally dedicated to function
operations, including interpolation, extrapolation, integration, differentiation
and more. This is a core class of our package, and should be maintained
carefully as it may impact all the rest of the project.
"""
import warnings
from collections.abc import Iterable
from inspect import signature
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, linalg, optimize

try:
    from functools import cached_property
except ImportError:
    from ..tools import cached_property


class Function:
    """Class converts a python function or a data sequence into an object
    which can be handled more naturally, enabling easy interpolation,
    extrapolation, plotting and algebra.
    """

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

            - Callable: Called for evaluation with input values. Must have the
              desired inputs as arguments and return a single output value.
              Input order is important. Example: Python functions, classes, and
              methods.

            - int or float: Treated as a constant value function.

            - ndarray: Used for interpolation. Format as [(x0, y0, z0),
            (x1, y1, z1), ..., (xn, yn, zn)], where 'x' and 'y' are inputs,
            and 'z' is the output.

            - string: Path to a CSV file. The file is read and converted into an
            ndarray. The file can optionally contain a single header line.

            - Function: Copies the source of the provided Function object,
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
            supported. For N-D functions, only shepard is supported.
            Default for 1-D functions is spline.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function at the edge of the interval,
            and 'zero', which returns zero for all points outside of source
            range. Default for 1-D functions is constant.
        title : string, optional
            Title to be displayed in the plots' figures. If none, the title will
            be constructed using the inputs and outputs arguments in the form
            of  "{inputs} x {outputs}".

        Returns
        -------
        None

        Notes
        -----
        (I) CSV files can optionally contain a single header line. If present,
        the header is ignored during processing.
        (II) Fields in CSV files may be enclosed in double quotes. If fields are
        not quoted, double quotes should not appear inside them.
        """
        # Set input and output
        if inputs is None:
            inputs = ["Scalar"]
        if outputs is None:
            outputs = ["Scalar"]

        inputs, outputs, interpolation, extrapolation = self._check_user_input(
            source, inputs, outputs, interpolation, extrapolation
        )

        # initialize variables to avoid errors when being called by other methods
        self.get_value_opt = None
        self.__polynomial_coefficients__ = None
        self.__akima_coefficients__ = None
        self.__spline_coefficients__ = None

        # store variables
        self.set_inputs(inputs)
        self.set_outputs(outputs)
        self.__interpolation__ = interpolation
        self.__extrapolation__ = extrapolation
        self.set_source(source)
        self.set_title(title)

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
        self.__inputs__ = [inputs] if isinstance(inputs, str) else list(inputs)
        self.__dom_dim__ = len(self.__inputs__)
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
        self.__outputs__ = [outputs] if isinstance(outputs, str) else list(outputs)
        self.__img_dim__ = len(self.__outputs__)
        return self

    def set_source(self, source):
        """Sets the data source for the function, defining how the function
        produces output from a given input.

        Parameters
        ----------
        source : callable, scalar, ndarray, string, or Function
            The data source to be used for the function:

            - Callable: Called for evaluation with input values. Must have the
              desired inputs as arguments and return a single output value.
              Input order is important. Example: Python functions, classes, and
              methods.

            - int or float: Treated as a constant value function.

            - ndarray: Used for interpolation. Format as [(x0, y0, z0),
            (x1, y1, z1), ..., (xn, yn, zn)], where 'x' and 'y' are inputs,
            and 'z' is the output.

            - string: Path to a CSV file. The file is read and converted into an
            ndarray. The file can optionally contain a single header line.

            - Function: Copies the source of the provided Function object,
            creating a new Function with adjusted inputs and outputs.

        Notes
        -----
        (I) CSV files can optionally contain a single header line. If present,
        the header is ignored during processing.
        (II) Fields in CSV files may be enclosed in double quotes. If fields are
        not quoted, double quotes should not appear inside them.

        Returns
        -------
        self : Function
            Returns the Function instance.
        """
        _ = self._check_user_input(
            source,
            self.__inputs__,
            self.__outputs__,
            self.__interpolation__,
            self.__extrapolation__,
        )
        # If the source is a Function
        if isinstance(source, Function):
            source = source.get_source()
        # Import CSV if source is a string or Path and convert values to ndarray
        if isinstance(source, (str, Path)):
            with open(source, "r") as file:
                try:
                    source = np.loadtxt(file, delimiter=",", dtype=float)
                except ValueError:
                    # If an error occurs, headers are present
                    source = np.loadtxt(source, delimiter=",", dtype=float, skiprows=1)
                except Exception as e:
                    raise ValueError(
                        "The source file is not a valid csv or txt file."
                    ) from e

        # Convert to ndarray if source is a list
        if isinstance(source, (list, tuple)):
            source = np.array(source, dtype=np.float64)
        # Convert number source into vectorized lambda function
        if isinstance(source, (int, float)):
            temp = 1 * source

            def source_function(_):
                return temp

            source = source_function

        # Handle callable source or number source
        if callable(source):
            # Set source
            self.source = source
            # Set get_value_opt
            self.get_value_opt = source
            # Set arguments name and domain dimensions
            parameters = signature(source).parameters
            self.__dom_dim__ = len(parameters)
            if self.__inputs__ == ["Scalar"]:
                self.__inputs__ = list(parameters)
            # Set interpolation and extrapolation
            self.__interpolation__ = None
            self.__extrapolation__ = None
        # Handle ndarray source
        else:
            # Check to see if dimensions match incoming data set
            new_total_dim = len(source[0, :])
            old_total_dim = self.__dom_dim__ + self.__img_dim__

            # If they don't, update default values or throw error
            if new_total_dim != old_total_dim:
                # Update dimensions and inputs
                self.__dom_dim__ = new_total_dim - 1
                self.__inputs__ = self.__dom_dim__ * self.__inputs__

            # Do things if domDim is 1
            if self.__dom_dim__ == 1:
                source = source[source[:, 0].argsort()]

                self.x_array = source[:, 0]
                self.x_initial, self.x_final = self.x_array[0], self.x_array[-1]

                self.y_array = source[:, 1]
                self.y_initial, self.y_final = self.y_array[0], self.y_array[-1]

                # Finally set data source as source
                self.source = source
            # Do things if function is multivariate
            else:
                self.x_array = source[:, 0]
                self.x_initial, self.x_final = self.x_array[0], self.x_array[-1]

                self.y_array = source[:, 1]
                self.y_initial, self.y_final = self.y_array[0], self.y_array[-1]

                self.z_array = source[:, 2]
                self.z_initial, self.z_final = self.z_array[0], self.z_array[-1]

                # Finally set data source as source
                self.source = source
            # Update extrapolation method
            if self.__extrapolation__ is None:
                self.set_extrapolation()
            # Set default interpolation for point source if it hasn't
            if self.__interpolation__ is None:
                self.set_interpolation()
            else:
                # Updates interpolation coefficients
                self.set_interpolation(self.__interpolation__)
        return self

    @cached_property
    def min(self):
        """Get the minimum value of the Function y_array.
        Raises an error if the Function is lambda based.

        Returns
        -------
        minimum : float
        """
        return self.y_array.min()

    @cached_property
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
            supported. For N-D functions, only shepard is supported.
            Default is 'spline'.

        Returns
        -------
        self : Function
        """
        # Set interpolation method
        self.__interpolation__ = method
        # Spline, akima and polynomial need data processing
        # Shepard, and linear do not
        if method == "spline":
            self.__interpolate_spline__()
        elif method == "polynomial":
            self.__interpolate_polynomial__()
        elif method == "akima":
            self.__interpolate_akima__()

        # Set get_value_opt
        self.set_get_value_opt()

        return self

    def set_extrapolation(self, method="constant"):
        """Set extrapolation behavior of data set.

        Parameters
        ----------
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function at the edge of the interval,
            and 'zero', which returns zero for all points outside of source
            range. Default is 'constant'.

        Returns
        -------
        self : Function
            The Function object.
        """
        self.__extrapolation__ = method
        return self

    def set_get_value_opt(self):
        """Crates a method that evaluates interpolations rather quickly
        when compared to other options available, such as just calling
        the object instance or calling ``Function.get_value`` directly. See
        ``Function.get_value_opt`` for documentation.

        Returns
        -------
        self : Function
        """
        # Retrieve general info
        x_data = self.x_array
        y_data = self.y_array
        x_min, x_max = self.x_initial, self.x_final
        if self.__extrapolation__ == "zero":
            extrapolation = 0  # Extrapolation is zero
        elif self.__extrapolation__ == "natural":
            extrapolation = 1  # Extrapolation is natural
        elif self.__extrapolation__ == "constant":
            extrapolation = 2  # Extrapolation is constant
        else:
            raise ValueError(f"Invalid extrapolation type {self.__extrapolation__}")

        # Crete method to interpolate this info for each interpolation type
        if self.__interpolation__ == "spline":
            coeffs = self.__spline_coefficients__

            def get_value_opt(x):
                x_interval = np.searchsorted(x_data, x)
                # Interval found... interpolate... or extrapolate
                if x_min <= x <= x_max:
                    # Interpolate
                    x_interval = x_interval if x_interval != 0 else 1
                    a = coeffs[:, x_interval - 1]
                    x = x - x_data[x_interval - 1]
                    y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        a = coeffs[:, 0] if x < x_min else coeffs[:, -1]
                        x = x - x_data[0] if x < x_min else x - x_data[-2]
                        y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                    else:  # Extrapolation is set to constant
                        y = y_data[0] if x < x_min else y_data[-1]
                return y

        elif self.__interpolation__ == "linear":

            def get_value_opt(x):
                x_interval = np.searchsorted(x_data, x)
                # Interval found... interpolate... or extrapolate
                if x_min <= x <= x_max:
                    # Interpolate
                    dx = float(x_data[x_interval] - x_data[x_interval - 1])
                    dy = float(y_data[x_interval] - y_data[x_interval - 1])
                    y = (x - x_data[x_interval - 1]) * (dy / dx) + y_data[
                        x_interval - 1
                    ]
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        x_interval = 1 if x < x_min else -1
                        dx = float(x_data[x_interval] - x_data[x_interval - 1])
                        dy = float(y_data[x_interval] - y_data[x_interval - 1])
                        y = (x - x_data[x_interval - 1]) * (dy / dx) + y_data[
                            x_interval - 1
                        ]
                    else:  # Extrapolation is set to constant
                        y = y_data[0] if x < x_min else y_data[-1]
                return y

        elif self.__interpolation__ == "akima":
            coeffs = np.array(self.__akima_coefficients__)

            def get_value_opt(x):
                x_interval = np.searchsorted(x_data, x)
                # Interval found... interpolate... or extrapolate
                if x_min <= x <= x_max:
                    # Interpolate
                    x_interval = x_interval if x_interval != 0 else 1
                    a = coeffs[4 * x_interval - 4 : 4 * x_interval]
                    y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        a = coeffs[:4] if x < x_min else coeffs[-4:]
                        y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                    else:  # Extrapolation is set to constant
                        y = y_data[0] if x < x_min else y_data[-1]
                return y

        elif self.__interpolation__ == "polynomial":
            coeffs = self.__polynomial_coefficients__

            def get_value_opt(x):
                # Interpolate... or extrapolate
                if x_min <= x <= x_max:
                    # Interpolate
                    y = 0
                    for i, coef in enumerate(coeffs):
                        y += coef * (x**i)
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        y = 0
                        for i, coef in enumerate(coeffs):
                            y += coef * (x**i)
                    else:  # Extrapolation is set to constant
                        y = y_data[0] if x < x_min else y_data[-1]
                return y

        elif self.__interpolation__ == "shepard":
            x_data = self.source[:, 0:-1]  # Support for N-Dimensions
            y_data = self.source[:, -1]
            len_y_data = len(y_data)  # A little speed up

            # change the function's name to avoid mypy's error
            def get_value_opt_multiple(*args):
                x = np.array([[float(x) for x in list(args)]])
                numerator_sum = 0
                denominator_sum = 0
                for i in range(len_y_data):
                    sub = x_data[i] - x
                    distance = np.linalg.norm(sub)
                    if distance == 0:
                        numerator_sum = y_data[i]
                        denominator_sum = 1
                        break
                    weight = distance ** (-3)
                    numerator_sum = numerator_sum + y_data[i] * weight
                    denominator_sum = denominator_sum + weight
                return numerator_sum / denominator_sum

            get_value_opt = get_value_opt_multiple

        self.get_value_opt = get_value_opt
        return self

    def set_discrete(
        self,
        lower=0,
        upper=10,
        samples=200,
        interpolation="spline",
        extrapolation="constant",
        one_by_one=True,
    ):
        """This method transforms function defined Functions into list
        defined Functions. It evaluates the function at certain points
        (sampling range) and stores the results in a list, which is converted
        into a Function and then returned. The original Function object is
        replaced by the new one.

        Parameters
        ----------
        lower : scalar, optional
            Value where sampling range will start. Default is 0.
        upper : scalar, optional
            Value where sampling range will end. Default is 10.
        samples : int, optional
            Number of samples to be taken from inside range. Default is 200.
        interpolation : string
            Interpolation method to be used if source type is ndarray.
            For 1-D functions, linear, polynomial, akima and spline is
            supported. For N-D functions, only shepard is supported.
            Default is 'spline'.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function at the edge of the interval,
            and 'zero', which returns zero for all points outside of source
            range. Default is 'constant'.
        one_by_one : boolean, optional
            If True, evaluate Function in each sample point separately. If
            False, evaluates Function in vectorized form. Default is True.

        Returns
        -------
        self : Function
        """
        if self.__dom_dim__ == 1:
            xs = np.linspace(lower, upper, samples)
            ys = self.get_value(xs.tolist()) if one_by_one else self.get_value(xs)
            self.set_source(np.concatenate(([xs], [ys])).transpose())
            self.set_interpolation(interpolation)
            self.set_extrapolation(extrapolation)
        elif self.__dom_dim__ == 2:
            lower = 2 * [lower] if isinstance(lower, (int, float)) else lower
            upper = 2 * [upper] if isinstance(upper, (int, float)) else upper
            sam = 2 * [samples] if isinstance(samples, (int, float)) else samples
            # Create nodes to evaluate function
            xs = np.linspace(lower[0], upper[0], sam[0])
            ys = np.linspace(lower[1], upper[1], sam[1])
            xs, ys = np.meshgrid(xs, ys)
            xs, ys = xs.flatten(), ys.flatten()
            mesh = [[xs[i], ys[i]] for i in range(len(xs))]
            # Evaluate function at all mesh nodes and convert it to matrix
            zs = np.array(self.get_value(mesh))
            self.set_source(np.concatenate(([xs], [ys], [zs])).transpose())
            self.__interpolation__ = "shepard"
            self.__extrapolation__ = "natural"
        return self

    def set_discrete_based_on_model(
        self, model_function, one_by_one=True, keep_self=True
    ):
        """This method transforms the domain of Function instance into a list of
        discrete points based on the domain of a model Function instance. It
        does so by retrieving the domain, domain name, interpolation method and
        extrapolation method of the model Function instance. It then evaluates
        the original Function instance in all points of the retrieved domain to
        generate the list of discrete points that will be used for interpolation
        when this Function is called.

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

        keepSelf : boolean, optional
            If True, the original Function interpolation and extrapolation
            methods will be kept. If False, those are substituted by the ones
            from the model Function. Default is True.

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
        1. This method performs in place replacement of the original Function
        object source.

        2. This method is similar to set_discrete, but it uses the domain of a
        model Function to define the domain of the new Function instance.
        """
        if not isinstance(model_function.source, np.ndarray):
            raise TypeError("model_function must be a list based Function.")
        if model_function.__dom_dim__ != self.__dom_dim__:
            raise ValueError("model_function must have the same domain dimension.")

        if self.__dom_dim__ == 1:
            xs = model_function.source[:, 0]
            ys = self.get_value(xs.tolist()) if one_by_one else self.get_value(xs)
            self.set_source(np.concatenate(([xs], [ys])).transpose())
        elif self.__dom_dim__ == 2:
            # Create nodes to evaluate function
            xs = model_function.source[:, 0]
            ys = model_function.source[:, 1]
            xs, ys = np.meshgrid(xs, ys)
            xs, ys = xs.flatten(), ys.flatten()
            mesh = [[xs[i], ys[i]] for i in range(len(xs))]
            # Evaluate function at all mesh nodes and convert it to matrix
            zs = np.array(self.get_value(mesh))
            self.set_source(np.concatenate(([xs], [ys], [zs])).transpose())

        interp = (
            self.__interpolation__ if keep_self else model_function.__interpolation__
        )
        extrap = (
            self.__extrapolation__ if keep_self else model_function.__extrapolation__
        )

        self.set_interpolation(interp)
        self.set_extrapolation(extrap)

        return self

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
        'Function from R1 to R1 : (x) → (Scalar)'
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
        4.0
        >>> f3.get_value(2.5)
        6.25
        >>> f3.get_value([1, 2, 3])
        [1.0, 4.0, 9.0]
        >>> f3.get_value([1, 2.5, 4.0])
        [1.0, 6.25, 16.0]

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

        # Return value for Function of function type
        if callable(self.source):
            # if the function is 1-D:
            if self.__dom_dim__ == 1:
                # if the args is a simple number (int or float)
                if isinstance(args[0], (int, float)):
                    return self.source(args[0])
                # if the arguments are iterable, we map and return a list
                if isinstance(args[0], Iterable):
                    return list(map(self.source, args[0]))

            # if the function is n-D:
            else:
                # if each arg is a simple number (int or float)
                if all(isinstance(arg, (int, float)) for arg in args):
                    return self.source(*args)
                # if each arg is iterable, we map and return a list
                if all(isinstance(arg, Iterable) for arg in args):
                    return [self.source(*arg) for arg in zip(*args)]

        # Returns value for shepard interpolation
        elif self.__interpolation__ == "shepard":
            if all(isinstance(arg, Iterable) for arg in args):
                x = list(np.column_stack(args))
            else:
                x = [[float(x) for x in list(args)]]
            ans = x
            x_data = self.source[:, 0:-1]
            y_data = self.source[:, -1]
            for i, _ in enumerate(x):
                numerator_sum = 0
                denominator_sum = 0
                for o, _ in enumerate(y_data):
                    sub = x_data[o] - x[i]
                    distance = (sub.dot(sub)) ** (0.5)
                    if distance == 0:
                        numerator_sum = y_data[o]
                        denominator_sum = 1
                        break
                    weight = distance ** (-3)
                    numerator_sum = numerator_sum + y_data[o] * weight
                    denominator_sum = denominator_sum + weight
                ans[i] = numerator_sum / denominator_sum
            return ans if len(ans) > 1 else ans[0]
        # Returns value for polynomial interpolation function type
        elif self.__interpolation__ == "polynomial":
            if isinstance(args[0], (int, float)):
                args = [list(args)]
            x = np.array(args[0])
            x_data = self.x_array
            y_data = self.y_array
            x_min, x_max = self.x_initial, self.x_final
            coeffs = self.__polynomial_coefficients__
            matrix = np.zeros((len(args[0]), coeffs.shape[0]))
            for i in range(coeffs.shape[0]):
                matrix[:, i] = x**i
            ans = matrix.dot(coeffs).tolist()
            for i, _ in enumerate(x):
                if not x_min <= x[i] <= x_max:
                    if self.__extrapolation__ == "constant":
                        ans[i] = y_data[0] if x[i] < x_min else y_data[-1]
                    elif self.__extrapolation__ == "zero":
                        ans[i] = 0
            return ans if len(ans) > 1 else ans[0]
        # Returns value for spline, akima or linear interpolation function type
        elif self.__interpolation__ in ["spline", "akima", "linear"]:
            if isinstance(args[0], (int, float, complex, np.integer)):
                args = [list(args)]
            x = list(args[0])
            x_data = self.x_array
            y_data = self.y_array
            x_intervals = np.searchsorted(x_data, x)
            x_min, x_max = self.x_initial, self.x_final
            if self.__interpolation__ == "spline":
                coeffs = self.__spline_coefficients__
                for i, _ in enumerate(x):
                    if x[i] == x_min or x[i] == x_max:
                        x[i] = y_data[x_intervals[i]]
                    elif x_min < x[i] < x_max or (self.__extrapolation__ == "natural"):
                        if not x_min < x[i] < x_max:
                            a = coeffs[:, 0] if x[i] < x_min else coeffs[:, -1]
                            x[i] = (
                                x[i] - x_data[0] if x[i] < x_min else x[i] - x_data[-2]
                            )
                        else:
                            a = coeffs[:, x_intervals[i] - 1]
                            x[i] = x[i] - x_data[x_intervals[i] - 1]
                        x[i] = a[3] * x[i] ** 3 + a[2] * x[i] ** 2 + a[1] * x[i] + a[0]
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == "zero":
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = y_data[0] if x[i] < x_min else y_data[-1]
            elif self.__interpolation__ == "linear":
                for i, _ in enumerate(x):
                    # Interval found... interpolate... or extrapolate
                    inter = x_intervals[i]
                    if x_min <= x[i] <= x_max:
                        # Interpolate
                        dx = float(x_data[inter] - x_data[inter - 1])
                        dy = float(y_data[inter] - y_data[inter - 1])
                        x[i] = (x[i] - x_data[inter - 1]) * (dy / dx) + y_data[
                            inter - 1
                        ]
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == "zero":  # Extrapolation == zero
                            x[i] = 0
                        elif (
                            self.__extrapolation__ == "natural"
                        ):  # Extrapolation == natural
                            inter = 1 if x[i] < x_min else -1
                            dx = float(x_data[inter] - x_data[inter - 1])
                            dy = float(y_data[inter] - y_data[inter - 1])
                            x[i] = (x[i] - x_data[inter - 1]) * (dy / dx) + y_data[
                                inter - 1
                            ]
                        else:  # Extrapolation is set to constant
                            x[i] = y_data[0] if x[i] < x_min else y_data[-1]
            else:
                coeffs = self.__akima_coefficients__
                for i, _ in enumerate(x):
                    if x[i] == x_min or x[i] == x_max:
                        x[i] = y_data[x_intervals[i]]
                    elif x_min < x[i] < x_max or (self.__extrapolation__ == "natural"):
                        if not x_min < x[i] < x_max:
                            a = coeffs[:4] if x[i] < x_min else coeffs[-4:]
                        else:
                            a = coeffs[4 * x_intervals[i] - 4 : 4 * x_intervals[i]]
                        x[i] = a[3] * x[i] ** 3 + a[2] * x[i] ** 2 + a[1] * x[i] + a[0]
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == "zero":
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = y_data[0] if x[i] < x_min else y_data[-1]
            if isinstance(args[0], np.ndarray):
                return np.array(x)
            else:
                return x if len(x) > 1 else x[0]

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

    # Define all presentation methods
    def __call__(self, *args):
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

        Returns
        -------
        ans : None, scalar, list
        """
        if len(args) == 0:
            return self.plot()
        else:
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
            # Compare multiple plots
            Function.compare_plots(self)
        else:
            if self.__dom_dim__ == 1:
                self.plot_1d(*args, **kwargs)
            elif self.__dom_dim__ == 2:
                self.plot_2d(*args, **kwargs)
            else:
                print("Error: Only functions with 1D or 2D domains are plottable!")

    def plot1D(self, *args, **kwargs):
        """Deprecated method, use Function.plot_1d instead."""
        warnings.warn(
            "The `Function.plot1D` method is set to be deprecated and fully "
            + "removed in rocketpy v2.0.0, use `Function.plot_1d` instead. "
            + "This method is calling `Function.plot_1d`.",
            DeprecationWarning,
        )
        return self.plot_1d(*args, **kwargs)

    def plot_1d(
        self,
        lower=None,
        upper=None,
        samples=1000,
        force_data=False,
        force_points=False,
        return_object=False,
        equal_axis=False,
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

        Returns
        -------
        None
        """
        # Define a mesh and y values at mesh nodes for plotting
        fig = plt.figure()
        ax = fig.axes
        if callable(self.source):
            # Determine boundaries
            lower = 0 if lower is None else lower
            upper = 10 if upper is None else upper
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
        plt.show()
        if return_object:
            return fig, ax

    def plot2D(self, *args, **kwargs):
        """Deprecated method, use Function.plot_2d instead."""
        warnings.warn(
            "The `Function.plot2D` method is set to be deprecated and fully "
            + "removed in rocketpy v2.0.0, use `Function.plot_2d` instead. "
            + "This method is calling `Function.plot_2d`.",
            DeprecationWarning,
        )
        return self.plot_2d(*args, **kwargs)

    def plot_2d(
        self,
        lower=None,
        upper=None,
        samples=None,
        force_data=True,
        disp_type="surface",
        alpha=0.6,
        cmap="viridis",
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
        if callable(self.source):
            # Determine boundaries
            lower = [0, 0] if lower is None else lower
            lower = 2 * [lower] if isinstance(lower, (int, float)) else lower
            upper = [10, 10] if upper is None else upper
            upper = 2 * [upper] if isinstance(upper, (int, float)) else upper
        else:
            # Determine boundaries
            x_data = self.x_array
            y_data = self.y_array
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            lower = [x_min, y_min] if lower is None else lower
            lower = 2 * [lower] if isinstance(lower, (int, float)) else lower
            upper = [x_max, y_max] if upper is None else upper
            upper = 2 * [upper] if isinstance(upper, (int, float)) else upper
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
        norm = plt.Normalize(z_min, z_max)

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
        elif disp_type == "wireframe":
            axes.plot_wireframe(mesh_x, mesh_y, z, rstride=1, cstride=1)
        elif disp_type == "contour":
            figure.clf()
            contour_set = plt.contour(mesh_x, mesh_y, z)
            plt.clabel(contour_set, inline=1, fontsize=10)
        elif disp_type == "contourf":
            figure.clf()
            contour_set = plt.contour(mesh_x, mesh_y, z)
            plt.contourf(mesh_x, mesh_y, z)
            plt.clabel(contour_set, inline=1, fontsize=10)
        plt.title(self.title)
        axes.set_xlabel(self.__inputs__[0].title())
        axes.set_ylabel(self.__inputs__[1].title())
        axes.set_zlabel(self.__outputs__[0].title())
        plt.show()

    @staticmethod
    def compare_plots(
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
    ):
        """Plots N 1-Dimensional Functions in the same plot, from a lower
        limit to an upper limit, by sampling the Functions several times in
        the interval.

        Parameters
        ----------
        plot_list : list
            List of Functions or list of tuples in the format (Function,
            label), where label is a string which will be displayed in the
            legend.
        lower : scalar, optional
            The lower limit of the interval in which the Functions are to be
            plotted. The default value for function type Functions is 0. By
            contrast, if the Functions given are defined by a dataset, the
            default value is the lowest value of the datasets.
        upper : scalar, optional
            The upper limit of the interval in which the Functions are to be
            plotted. The default value for function type Functions is 10. By
            contrast, if the Functions given are defined by a dataset, the
            default value is the highest value of the datasets.
        samples : int, optional
            The number of samples in which the functions will be evaluated for
            plotting it, which draws lines between each evaluated point.
            The default value is 1000.
        title : string, optional
            Title of the plot. Default value is an empty string.
        xlabel : string, optional
            X-axis label. Default value is an empty string.
        ylabel : string, optional
            Y-axis label. Default value is an empty string.
        force_data : Boolean, optional
            If Function is given by an interpolated dataset, setting force_data
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        force_points : Boolean, optional
            Setting force_points to True will plot all points, as a scatter, in
            which the Function was evaluated to plot it. Default value is
            False.

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
                if not callable(plot[0].source):
                    # Determine boundaries
                    x_min = plot[0].source[0, 0]
                    lower = x_min if x_min < lower else lower
        if upper is None:
            upper = 10
            for plot in plots:
                if not callable(plot[0].source):
                    # Determine boundaries
                    x_max = plot[0].source[-1, 0]
                    upper = x_max if x_max > upper else upper
        x = np.linspace(lower, upper, samples)

        # Iterate to plot all plots
        for plot in plots:
            # Deal with discrete data sets when no range is given
            if no_range_specified and not callable(plot[0].source):
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
                if not callable(plot[0].source):
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
        ax.legend(loc="best", shadow=True)

        # Turn on grid and set title and axis
        plt.grid(True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show()

        if return_object:
            return fig, ax

    # Define all interpolation methods
    def __interpolate_polynomial__(self):
        """Calculate polynomial coefficients that fit the data exactly."""
        # Find the degree of the polynomial interpolation
        degree = self.source.shape[0] - 1
        # Get x and y values for all supplied points.
        x = self.x_array
        y = self.y_array
        # Check if interpolation requires large numbers
        if np.amax(x) ** degree > 1e308:
            warnings.warn(
                "Polynomial interpolation of too many points can't be done."
                " Once the degree is too high, numbers get too large."
                " The process becomes inefficient. Using spline instead."
            )
            return self.set_interpolation("spline")
        # Create coefficient matrix1
        sys_coeffs = np.zeros((degree + 1, degree + 1))
        for i in range(degree + 1):
            sys_coeffs[:, i] = x**i
        # Solve the system and store the resultant coefficients
        self.__polynomial_coefficients__ = np.linalg.solve(sys_coeffs, y)

    def __interpolate_spline__(self):
        """Calculate natural spline coefficients that fit the data exactly."""
        # Get x and y values for all supplied points
        x, y = self.x_array, self.y_array
        m_dim = len(x)
        h = np.diff(x)
        # Initialize the matrix
        banded_matrix = np.zeros((3, m_dim))
        banded_matrix[1, 0] = banded_matrix[1, m_dim - 1] = 1
        # Construct the Ab banded matrix and B vector
        vector_b = [0]
        banded_matrix[2, :-2] = h[:-1]
        banded_matrix[1, 1:-1] = 2 * (h[:-1] + h[1:])
        banded_matrix[0, 2:] = h[1:]
        vector_b.extend(3 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1]))
        vector_b.append(0)
        # Solve the system for c coefficients
        c = linalg.solve_banded(
            (1, 1), banded_matrix, vector_b, overwrite_ab=True, overwrite_b=True
        )
        # Calculate other coefficients
        b = (y[1:] - y[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3 * h)
        # Store coefficients
        self.__spline_coefficients__ = np.vstack([y[:-1], b, c[:-1], d])

    def __interpolate_akima__(self):
        """Calculate akima spline coefficients that fit the data exactly"""
        # Get x and y values for all supplied points
        x, y = self.x_array, self.y_array
        # Estimate derivatives at each point
        d = [0] * len(x)
        d[0] = (y[1] - y[0]) / (x[1] - x[0])
        d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        for i in range(1, len(x) - 1):
            w1, w2 = (x[i] - x[i - 1]), (x[i + 1] - x[i])
            d1, d2 = ((y[i] - y[i - 1]) / w1), ((y[i + 1] - y[i]) / w2)
            d[i] = (w1 * d2 + w2 * d1) / (w1 + w2)
        # Calculate coefficients for each interval with system already solved
        coeffs = [0] * 4 * (len(x) - 1)
        for i in range(len(x) - 1):
            xl, xr = x[i], x[i + 1]
            yl, yr = y[i], y[i + 1]
            dl, dr = d[i], d[i + 1]
            matrix = np.array(
                [
                    [1, xl, xl**2, xl**3],
                    [1, xr, xr**2, xr**3],
                    [0, 1, 2 * xl, 3 * xl**2],
                    [0, 1, 2 * xr, 3 * xr**2],
                ]
            )
            result = np.array([yl, yr, dl, dr]).T
            coeffs[4 * i : 4 * i + 4] = np.linalg.solve(matrix, result)
        self.__akima_coefficients__ = coeffs

    def __neg__(self):
        """Negates the Function object. The result has the same effect as
        multiplying the Function by -1.

        Returns
        -------
        Function
            The negated Function object.
        """
        if isinstance(self.source, np.ndarray):
            neg_source = np.column_stack((self.x_array, -self.y_array))
            return Function(
                neg_source,
                self.__inputs__,
                self.__outputs__,
                self.__interpolation__,
                self.__extrapolation__,
            )
        else:
            return Function(
                lambda x: -self.source(x),
                self.__inputs__,
                self.__outputs__,
                self.__interpolation__,
                self.__extrapolation__,
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

        if isinstance(self.source, np.ndarray):
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
        return None

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

        if isinstance(self.source, np.ndarray):
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
        return None

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
    def __add__(self, other):
        """Sums a Function object and 'other', returns a new Function
        object which gives the result of the sum. Only implemented for
        1D domains.

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
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have the same domain discretization
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__dom_dim__ == other.__dom_dim__
                and np.array_equal(self.x_array, other.x_array)
            ):
                # Operate on grid values
                ys = self.y_array + other.y_array
                xs = self.x_array
                source = np.concatenate(([xs], [ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + " + " + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                extrapolation = self.__extrapolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation, extrapolation)
            else:
                return Function(lambda x: (self.get_value(x) + other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    ys = self.y_array + other
                    xs = self.x_array
                    source = np.concatenate(([xs], [ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + " + " + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    extrapolation = self.__extrapolation__
                    # Create new Function object
                    return Function(
                        source, inputs, outputs, interpolation, extrapolation
                    )
                else:
                    return Function(lambda x: (self.get_value(x) + other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.get_value(x) + other(x)))

    def __radd__(self, other):
        """Sums 'other' and a Function object and returns a new Function
        object which gives the result of the sum. Only implemented for
        1D domains.

        Parameters
        ----------
        other : int, float, callable
            What self will be added to.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)/+self(x).
        """
        return self + other

    def __sub__(self, other):
        """Subtracts from a Function object and returns a new Function object
        which gives the result of the subtraction. Only implemented for 1D
        domains.

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
        try:
            return self + (-other)
        except TypeError:
            return Function(lambda x: (self.get_value(x) - other(x)))

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
        return other + (-self)

    def __mul__(self, other):
        """Multiplies a Function object and returns a new Function object
        which gives the result of the multiplication. Only implemented for 1D
        domains.

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
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have the same domain discretization
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__dom_dim__ == other.__dom_dim__
                and np.array_equal(self.x_array, other.x_array)
            ):
                # Operate on grid values
                ys = self.y_array * other.y_array
                xs = self.x_array
                source = np.concatenate(([xs], [ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + "*" + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                extrapolation = self.__extrapolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation, extrapolation)
            else:
                return Function(lambda x: (self.get_value(x) * other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    ys = self.y_array * other
                    xs = self.x_array
                    source = np.concatenate(([xs], [ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + "*" + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    extrapolation = self.__extrapolation__
                    # Create new Function object
                    return Function(
                        source, inputs, outputs, interpolation, extrapolation
                    )
                else:
                    return Function(lambda x: (self.get_value(x) * other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.get_value(x) * other(x)))

    def __rmul__(self, other):
        """Multiplies 'other' by a Function object and returns a new Function
        object which gives the result of the multiplication. Only implemented for
        1D domains.

        Parameters
        ----------
        other : int, float, callable
            What self will be multiplied by.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)*self(x).
        """
        return self * other

    def __truediv__(self, other):
        """Divides a Function object and returns a new Function object
        which gives the result of the division. Only implemented for 1D
        domains.

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
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have the same domain discretization
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__dom_dim__ == other.__dom_dim__
                and np.array_equal(self.x_array, other.x_array)
            ):
                # operate on grid values
                with np.errstate(divide="ignore", invalid="ignore"):
                    ys = self.source[:, 1] / other.source[:, 1]
                    ys = np.nan_to_num(ys)
                xs = self.source[:, 0]
                source = np.concatenate(([xs], [ys])).transpose()
                # retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + "/" + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                extrapolation = self.__extrapolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation, extrapolation)
            else:
                return Function(lambda x: (self.get_value_opt(x) / other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    ys = self.y_array / other
                    xs = self.x_array
                    source = np.concatenate(([xs], [ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + "/" + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    extrapolation = self.__extrapolation__
                    # Create new Function object
                    return Function(
                        source, inputs, outputs, interpolation, extrapolation
                    )
                else:
                    return Function(lambda x: (self.get_value_opt(x) / other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.get_value_opt(x) / other(x)))

    def __rtruediv__(self, other):
        """Divides 'other' by a Function object and returns a new Function
        object which gives the result of the division. Only implemented for
        1D domains.

        Parameters
        ----------
        other : int, float, callable
            What self will divide.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)/self(x).
        """
        # Check if Function object source is array and other is float
        if isinstance(other, (float, int, complex)):
            if isinstance(self.source, np.ndarray):
                # Operate on grid values
                ys = other / self.y_array
                xs = self.x_array
                source = np.concatenate(([xs], [ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + "/" + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                extrapolation = self.__extrapolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation, extrapolation)
            else:
                return Function(lambda x: (other / self.get_value_opt(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) / self.get_value_opt(x)))

    def __pow__(self, other):
        """Raises a Function object to the power of 'other' and
        returns a new Function object which gives the result. Only
        implemented for 1D domains.

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
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have the same domain discretization
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__dom_dim__ == other.__dom_dim__
                and np.any(self.x_array - other.x_array) is False
                and np.array_equal(self.x_array, other.x_array)
            ):
                # Operate on grid values
                ys = self.y_array**other.y_array
                xs = self.x_array
                source = np.concatenate(([xs], [ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + "**" + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                extrapolation = self.__extrapolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation, extrapolation)
            else:
                return Function(lambda x: (self.get_value_opt(x) ** other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    ys = self.y_array**other
                    xs = self.x_array
                    source = np.concatenate(([xs], [ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + "**" + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    extrapolation = self.__extrapolation__
                    # Create new Function object
                    return Function(
                        source, inputs, outputs, interpolation, extrapolation
                    )
                else:
                    return Function(lambda x: (self.get_value(x) ** other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.get_value(x) ** other(x)))

    def __rpow__(self, other):
        """Raises 'other' to the power of a Function object and returns
        a new Function object which gives the result. Only implemented
        for 1D domains.

        Parameters
        ----------
        other : int, float, callable
            The object that will be exponentiated by the function.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)**self(x).
        """
        # Check if Function object source is array and other is float
        if isinstance(other, (float, int, complex)):
            if isinstance(self.source, np.ndarray):
                # Operate on grid values
                ys = other**self.y_array
                xs = self.x_array
                source = np.concatenate(([xs], [ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + "**" + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                extrapolation = self.__extrapolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation, extrapolation)
            else:
                return Function(lambda x: (other ** self.get_value(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) ** self.get_value(x)))

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
            If False, try to calculate using interpolation information.
            Currently, only available for spline and linear interpolation. If
            unavailable, calculate numerically anyways.

        Returns
        -------
        ans : float
            Evaluated integral.
        """
        # Guarantee a < b
        integration_sign = np.sign(b - a)
        if integration_sign == -1:
            a, b = b, a
        # Different implementations depending on interpolation
        if self.__interpolation__ == "spline" and numerical is False:
            x_data = self.x_array
            y_data = self.y_array
            coeffs = self.__spline_coefficients__
            ans = 0
            # Check to see if interval starts before point data
            if a < x_data[0]:
                if self.__extrapolation__ == "constant":
                    ans += y_data[0] * (min(x_data[0], b) - a)
                elif self.__extrapolation__ == "natural":
                    c = coeffs[:, 0]
                    sub_b = a - x_data[0]
                    sub_a = min(b, x_data[0]) - x_data[0]
                    ans += (
                        (c[3] * sub_a**4) / 4
                        + (c[2] * sub_a**3 / 3)
                        + (c[1] * sub_a**2 / 2)
                        + c[0] * sub_a
                    )
                    ans -= (
                        (c[3] * sub_b**4) / 4
                        + (c[2] * sub_b**3 / 3)
                        + (c[1] * sub_b**2 / 2)
                        + c[0] * sub_b
                    )
                else:
                    # self.__extrapolation__ = 'zero'
                    pass

            # Integrate in subintervals between xs of given data up to b
            i = max(np.searchsorted(x_data, a, side="left") - 1, 0)
            while i < len(x_data) - 1 and x_data[i] < b:
                if x_data[i] <= a <= x_data[i + 1] and x_data[i] <= b <= x_data[i + 1]:
                    sub_a = a - x_data[i]
                    sub_b = b - x_data[i]
                elif x_data[i] <= a <= x_data[i + 1]:
                    sub_a = a - x_data[i]
                    sub_b = x_data[i + 1] - x_data[i]
                elif b <= x_data[i + 1]:
                    sub_a = 0
                    sub_b = b - x_data[i]
                else:
                    sub_a = 0
                    sub_b = x_data[i + 1] - x_data[i]
                c = coeffs[:, i]
                ans += (
                    (c[3] * sub_b**4) / 4
                    + (c[2] * sub_b**3 / 3)
                    + (c[1] * sub_b**2 / 2)
                    + c[0] * sub_b
                )
                ans -= (
                    (c[3] * sub_a**4) / 4
                    + (c[2] * sub_a**3 / 3)
                    + (c[1] * sub_a**2 / 2)
                    + c[0] * sub_a
                )
                i += 1
            # Check to see if interval ends after point data
            if b > x_data[-1]:
                if self.__extrapolation__ == "constant":
                    ans += y_data[-1] * (b - max(x_data[-1], a))
                elif self.__extrapolation__ == "natural":
                    c = coeffs[:, -1]
                    sub_a = max(x_data[-1], a) - x_data[-2]
                    sub_b = b - x_data[-2]
                    ans -= (
                        (c[3] * sub_a**4) / 4
                        + (c[2] * sub_a**3 / 3)
                        + (c[1] * sub_a**2 / 2)
                        + c[0] * sub_a
                    )
                    ans += (
                        (c[3] * sub_b**4) / 4
                        + (c[2] * sub_b**3 / 3)
                        + (c[1] * sub_b**2 / 2)
                        + c[0] * sub_b
                    )
                else:
                    # self.__extrapolation__ = 'zero'
                    pass
        elif self.__interpolation__ == "linear" and numerical is False:
            # Integrate from a to b using np.trapz
            x_data = self.x_array
            y_data = self.y_array
            # Get data in interval
            x_integration_data = x_data[(x_data >= a) & (x_data <= b)]
            y_integration_data = y_data[(x_data >= a) & (x_data <= b)]
            # Add integration limits to data
            if self.__extrapolation__ == "zero":
                if a >= x_data[0]:
                    x_integration_data = np.concatenate(([a], x_integration_data))
                    y_integration_data = np.concatenate(([self(a)], y_integration_data))
                if b <= x_data[-1]:
                    x_integration_data = np.concatenate((x_integration_data, [b]))
                    y_integration_data = np.concatenate((y_integration_data, [self(b)]))
            else:
                x_integration_data = np.concatenate(([a], x_integration_data))
                y_integration_data = np.concatenate(([self(a)], y_integration_data))
                x_integration_data = np.concatenate((x_integration_data, [b]))
                y_integration_data = np.concatenate((y_integration_data, [self(b)]))
            # Integrate using np.trapz
            ans = np.trapz(y_integration_data, x_integration_data)
        else:
            # Integrate numerically
            ans, _ = integrate.quad(self, a, b, epsabs=1e-4, epsrel=1e-3, limit=1000)
        return integration_sign * ans

    def differentiate(self, x, dx=1e-6, order=1):
        """Differentiate a Function object at a given point.

        Parameters
        ----------
        x : float
            Point at which to differentiate.
        dx : float
            Step size to use for numerical differentiation.
        order : int
            Order of differentiation.

        Returns
        -------
        ans : float
            Evaluated derivative.
        """
        if order == 1:
            return (self.get_value(x + dx) - self.get_value(x - dx)) / (2 * dx)
        elif order == 2:
            return (
                self.get_value(x + dx) - 2 * self.get_value(x) + self.get_value(x - dx)
            ) / dx**2

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

        # Check if Function object source is array
        if isinstance(self.source, np.ndarray):
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

    def derivative_function(self):
        """Returns a Function object which gives the derivative of the Function object.

        Returns
        -------
        result : Function
            A Function object which gives the derivative of self.
        """
        # Check if Function object source is array
        if isinstance(self.source, np.ndarray):
            # Operate on grid values
            ys = np.diff(self.y_array) / np.diff(self.x_array)
            xs = self.source[:-1, 0] + np.diff(self.x_array) / 2
            source = np.column_stack((xs, ys))
            # Retrieve inputs, outputs and interpolation
            inputs = self.__inputs__[:]
            outputs = f"d({self.__outputs__[0]})/d({inputs[0]})"
        else:

            def source_function(x):
                return self.differentiate(x)

            source = source_function
            inputs = self.__inputs__[:]
            outputs = f"d({self.__outputs__[0]})/d({inputs[0]})"

        # Create new Function object
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
        if isinstance(self.source, np.ndarray):
            lower = self.source[0, 0] if lower is None else lower
            upper = self.source[-1, 0] if upper is None else upper
            x_data = np.linspace(lower, upper, datapoints)
            y_data = np.zeros(datapoints)
            for i in range(datapoints):
                y_data[i] = self.integral(lower, x_data[i])
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
        if isinstance(self.source, np.ndarray):
            x_data_distinct = set(self.x_array)
            y_data_distinct = set(self.y_array)
            distinct_map = set(zip(x_data_distinct, y_data_distinct))
            return len(distinct_map) == len(x_data_distinct) == len(y_data_distinct)
        else:
            raise TypeError(
                "Only Functions whose source is a list of points can be "
                "checked for bijectivity."
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
        >>> f.isbijective()
        True
        >>> f.is_strictly_bijective()
        True

        >>> f = Function([[-1, 1], [0, 0], [1, 1], [2, 4]])
        >>> f.isbijective()
        False
        >>> f.is_strictly_bijective()
        False

        A Function which is not "strictly" bijective, but is bijective, can be
        constructed as x^2 defined at -1, 0 and 2.

        >>> f = Function([[-1, 1], [0, 0], [2, 4]])
        >>> f.isbijective()
        True
        >>> f.is_strictly_bijective()
        False
        """
        if isinstance(self.source, np.ndarray):
            # Assuming domain is sorted, range must also be
            y_data = self.y_array
            # Both ascending and descending order means Function is bijective
            y_data_diff = np.diff(y_data)
            return np.all(y_data_diff >= 0) or np.all(y_data_diff <= 0)
        else:
            raise TypeError(
                "Only Functions whose source is a list of points can be "
                "checked for bijectivity."
            )

    def inverse_function(self, approx_func=None, tol=1e-4):
        """
        Returns the inverse of the Function. The inverse function of F is a
        function that undoes the operation of F. The inverse of F exists if
        and only if F is bijective. Makes the domain the range and the range
        the domain.

        If the Function is given by a list of points, its bijectivity is
        checked and an error is raised if it is not bijective.
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
        if isinstance(self.source, np.ndarray):
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
        if isinstance(self.source, np.ndarray):
            if lower is None:
                lower = self.source[0, 0]
            upper = self.source[-1, 0]
            x_data = np.linspace(lower, upper, 100)
            y_data = np.zeros(100)
            y_data[0] = self.source[:, 1][0]
            for i in range(1, 100):
                y_data[i] = self.average(lower, x_data[i])
            return Function(
                np.concatenate(([x_data], [y_data])).transpose(),
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
        if not isinstance(func, Function):
            raise TypeError("Input must be a Function object.")

        if isinstance(self.source, np.ndarray) and isinstance(func.source, np.ndarray):
            # Perform bounds check for composition
            if not extrapolate:
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

    @staticmethod
    def _check_user_input(
        source,
        inputs=None,
        outputs=None,
        interpolation=None,
        extrapolation=None,
    ):
        """
        Validates and processes the user input parameters for creating or
        modifying a Function object. This function ensures the inputs, outputs,
        interpolation, and extrapolation parameters are compatible with the
        given source. It converts the source to a numpy array if necessary, sets
        default values and raises warnings or errors for incompatible or
        ill-defined parameters.

        Parameters
        ----------
        source : list, np.ndarray, or callable
            The source data or Function object. If a list or ndarray, it should
            contain numeric data. If a Function, its inputs and outputs are
            checked against the provided inputs and outputs.
        inputs : list of str or None
            The names of the input variables. If None, defaults are generated
            based on the dimensionality of the source.
        outputs : str or list of str
            The name(s) of the output variable(s). If a list is provided, it
            must have a single element.
        interpolation : str or None
            The method of interpolation to be used. For multidimensional sources
            it defaults to 'shepard' if not provided.
        extrapolation : str or None
            The method of extrapolation to be used. For multidimensional sources
            it defaults to 'natural' if not provided.

        Returns
        -------
        tuple
            A tuple containing the processed inputs, outputs, interpolation, and
            extrapolation parameters.

        Raises
        ------
        ValueError
            If the dimensionality of the source does not match the combined
            dimensions of inputs and outputs. If the outputs list has more than
            one element.
        TypeError
            If the source is not a list, np.ndarray, Function object, str or
            Path.
        Warning
            If inputs or outputs do not match for a Function source, or if
            defaults are used for inputs, interpolation,and extrapolation for a
            multidimensional source.

        Examples
        --------
        >>> from rocketpy import Function
        >>> source = np.array([(1, 1), (2, 4), (3, 9)])
        >>> inputs = "x"
        >>> outputs = ["y"]
        >>> interpolation = 'linear'
        >>> extrapolation = 'zero'
        >>> inputs, outputs, interpolation, extrapolation = Function._check_user_input(
        ...     source, inputs, outputs, interpolation, extrapolation
        ... )
        >>> inputs
        ['x']
        >>> outputs
        ['y']
        >>> interpolation
        'linear'
        >>> extrapolation
        'zero'
        """
        # check output type and dimensions
        if isinstance(outputs, str):
            outputs = [outputs]
        if isinstance(inputs, str):
            inputs = [inputs]

        elif len(outputs) > 1:
            raise ValueError(
                "Output must either be a string or have dimension 1, "
                + f"it currently has dimension ({len(outputs)})."
            )

        # check source for data type
        # if list or ndarray, check for dimensions, interpolation and extrapolation
        if isinstance(source, (list, np.ndarray, str, Path)):
            # Deal with csv or txt
            if isinstance(source, (str, Path)):
                # Convert to numpy array
                try:
                    source = np.loadtxt(source, delimiter=",", dtype=float)
                except ValueError:
                    # Skip header
                    source = np.loadtxt(source, delimiter=",", dtype=float, skiprows=1)
                except Exception as e:
                    raise ValueError(
                        "The source file is not a valid csv or txt file."
                    ) from e

            else:
                # this will also trigger an error if the source is not a list of
                # numbers or if the array is not homogeneous
                source = np.array(source, dtype=np.float64)

            # check dimensions
            source_dim = source.shape[1]

            # check interpolation and extrapolation

            ## single dimension
            if source_dim == 2:
                # possible interpolation values: llinear, polynomial, akima and spline
                if interpolation is None:
                    interpolation = "spline"
                elif interpolation.lower() not in [
                    "spline",
                    "linear",
                    "polynomial",
                    "akima",
                ]:
                    warnings.warn(
                        "Interpolation method for single dimensional functions was "
                        + f"set to 'spline', the {interpolation} method is not supported."
                    )
                    interpolation = "spline"

                # possible extrapolation values: constant, natural, zero
                if extrapolation is None:
                    extrapolation = "constant"
                elif extrapolation.lower() not in ["constant", "natural", "zero"]:
                    warnings.warn(
                        "Extrapolation method for single dimensional functions was "
                        + f"set to 'constant', the {extrapolation} method is not supported."
                    )
                    extrapolation = "constant"

            ## multiple dimensions
            elif source_dim > 2:
                # check for inputs and outputs
                if inputs == ["Scalar"]:
                    inputs = [f"Input {i+1}" for i in range(source_dim - 1)]

                if interpolation not in [None, "shepard"]:
                    warnings.warn(
                        (
                            "Interpolation method for multidimensional functions was"
                            "set to 'shepard', other methods are not supported yet."
                        ),
                    )
                interpolation = "shepard"

                if extrapolation not in [None, "natural"]:
                    warnings.warn(
                        "Extrapolation method for multidimensional functions was set"
                        "to 'natural', other methods are not supported yet."
                    )
                extrapolation = "natural"

            # check input dimensions
            in_out_dim = len(inputs) + len(outputs)
            if source_dim != in_out_dim:
                raise ValueError(
                    "Source dimension ({source_dim}) does not match input "
                    + f"and output dimension ({in_out_dim})."
                )
        return inputs, outputs, interpolation, extrapolation


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
                output[j] = func.get_value(value)
            return output

        input_data = []
        output_data = []
        for key in sorted(source.keys()):
            i = np.linspace(key[0], key[1], datapoints)
            i = i[~np.in1d(i, input_data)]
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


def funcify_method(*args, **kwargs):
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

    In order to reset the cache, just delete de attribute from the instance:

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


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()
    if results.failed < 1:
        print(f"All the {results.attempted} tests passed!")
    else:
        print(f"{results.failed} out of {results.attempted} tests failed.")
