# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Lucas Kierulff Balabram"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from inspect import signature

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, linalg


class Function:
    """Class converts a python function or a data sequence into an object
    which can be handled more naturally, enabling easy interpolation,
    extrapolation, plotting and algebra.
    """

    def __init__(
        self,
        source,
        inputs=["Scalar"],
        outputs=["Scalar"],
        interpolation=None,
        extrapolation=None,
        title=None,
    ):
        """Convert source into a Function, to be used more naturally.
        Set inputs, outputs, domain dimension, interpolation and extrapolation
        method, and process the source.

        Parameters
        ----------
        source : function, scalar, ndarray, string
            The actual function. If type is function, it will be called for
            evaluation. If type is int or float, it will be treated as a
            constant function. If ndarray, its points will be used for
            interpolation. An ndarray should be as [(x0, y0, z0), (x1, y1, z1),
            (x2, y2, z2), ...] where x0 and y0 are inputs and z0 is output. If
            string, imports file named by the string and treats it as csv.
            The file is converted into ndarray and should not have headers.
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
            Title to be displayed in the plots' figures. If none, the title will be constructed using the inputs and outputs arguments in the form "{inputs} x {outputs}".

        Returns
        -------
        None
        """
        # Set input and output
        self.setInputs(inputs)
        self.setOutputs(outputs)
        # Save interpolation method
        self.__interpolation__ = interpolation
        self.__extrapolation__ = extrapolation
        # Initialize last_interval
        self.last_interval = 0
        # Set source
        self.setSource(source)
        #  Set function title
        if title:
            self.setTitle(title)
        else:
            if self.__domDim__ == 1:
                self.setTitle(
                    self.__outputs__[0].title() + " x " + self.__inputs__[0].title()
                )
            elif self.__domDim__ == 2:
                self.setTitle(
                    self.__outputs__[0].title()
                    + " x "
                    + self.__inputs__[0].title()
                    + " x "
                    + self.__inputs__[1].title()
                )
        # Return
        return None

    # Define all set methods
    def setInputs(self, inputs):
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
        self.__domDim__ = len(self.__inputs__)
        return self

    def setOutputs(self, outputs):
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
        self.__imgDim__ = len(self.__outputs__)
        return self

    def setSource(self, source):
        """Set the source which defines the output of the function giving a
        certain input.

        Parameters
        ----------
        source : function, scalar, ndarray, string
            The actual function. If type is function, it will be called for
            evaluation. If type is int or float, it will be treated as a
            constant function. If ndarray, its points will be used for
            interpolation. An ndarray should be as [(x0, y0, z0), (x1, y1, z1),
            (x2, y2, z2), ...] where x0 and y0 are inputs and z0 is output. If
            string, imports file named by the string and treats it as csv.
            The file is converted into ndarray and should not have headers.

        Returns
        -------
        self : Function
        """
        # Import CSV if source is a string and convert values to ndarray
        if isinstance(source, str):
            # Read file and check for headers
            f = open(source, "r")
            firstLine = f.readline()
            # If headers are found...
            if firstLine[0] in ['"', "'"]:
                # Headers available
                firstLine = firstLine.replace('"', " ").replace("'", " ")
                firstLine = firstLine.split(" , ")
                self.setInputs(firstLine[0])
                self.setOutputs(firstLine[1:])
                source = np.loadtxt(source, delimiter=",", skiprows=1, dtype=float)
            # if headers are not found
            else:
                source = np.loadtxt(source, delimiter=",", dtype=float)
        # Convert to ndarray if source is a list
        if isinstance(source, (list, tuple)):
            source = np.array(source, dtype=np.float64)
        # Convert number source into vectorized lambda function
        if isinstance(source, (int, float)):
            temp = 1 * source

            def source(x):
                return temp

        # Handle callable source or number source
        if callable(source):
            # Set source
            self.source = source
            # Set geValueOpt2
            self.getValueOpt = source
            # Set arguments name and domain dimensions
            parameters = signature(source).parameters
            self.__domDim__ = len(parameters)
            if self.__inputs__ == ["Scalar"]:
                self.__inputs__ = list(parameters)
            # Set interpolation and extrapolation
            self.__interpolation__ = None
            self.__extrapolation__ = None
        # Handle ndarray source
        else:
            # Check to see if dimensions match incoming data set
            newTotalDim = len(source[0, :])
            oldTotalDim = self.__domDim__ + self.__imgDim__
            dV = self.__inputs__ == ["Scalar"] and self.__outputs__ == ["Scalar"]
            # If they don't, update default values or throw error
            if newTotalDim != oldTotalDim:
                if dV:
                    # Update dimensions and inputs
                    self.__domDim__ = newTotalDim - 1
                    self.__inputs__ = self.__domDim__ * self.__inputs__
                else:
                    # User has made a mistake inputting inputs and outputs
                    print("Error in input and output dimensions!")
                    return None
            # Do things if domDim is 1
            if self.__domDim__ == 1:
                source = source[source[:, 0].argsort()]
                # Finally set data source as source
                self.source = source
                # Set default interpolation for point source if it hasn't
                if self.__interpolation__ is None:
                    self.setInterpolation()
                else:
                    # Updates interpolation coefficients
                    self.setInterpolation(self.__interpolation__)
            # Do things if function is multivariate
            else:
                # Finally set data source as source
                self.source = source
                if self.__interpolation__ is None:
                    self.setInterpolation("shepard")
            # Update extrapolation method
            if self.__extrapolation__ is None:
                self.setExtrapolation()
        # Return self
        return self

    def setInterpolation(self, method="spline"):
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
            self.__interpolateSpline__()
        elif method == "polynomial":
            self.__interpolatePolynomial__()
        elif method == "akima":
            self.__interpolateAkima__()

        # Set geValueOpt
        self.setGetValueOpt()

        # Returns self
        return self

    def setExtrapolation(self, method="constant"):
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
        """
        # Set extrapolation method
        self.__extrapolation__ = method
        # Return self
        return self

    def setGetValueOpt(self):
        """Crates a method that evaluates interpolations rather quickly
        when compared to other options available, such as just calling
        the object instance or calling self.getValue directly. See
        Function.getValueOpt for documentation.

        Returns
        -------
        self : Function
        """
        # Retrieve general info
        xData = self.source[:, 0]
        yData = self.source[:, 1]
        xmin, xmax = xData[0], xData[-1]
        if self.__extrapolation__ == "zero":
            extrapolation = 0  # Extrapolation is zero
        elif self.__extrapolation__ == "natural":
            extrapolation = 1  # Extrapolation is natural
        else:
            extrapolation = 2  # Extrapolation is constant

        # Crete method to interpolate this info for each interpolation type
        if self.__interpolation__ == "spline":
            coeffs = self.__splineCoefficients__

            def getValueOpt(x):
                xInterval = np.searchsorted(xData, x)
                # Interval found... interpolate... or extrapolate
                if xmin <= x <= xmax:
                    # Interpolate
                    xInterval = xInterval if xInterval != 0 else 1
                    a = coeffs[:, xInterval - 1]
                    x = x - xData[xInterval - 1]
                    y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        a = coeffs[:, 0] if x < xmin else coeffs[:, -1]
                        x = x - xData[0] if x < xmin else x - xData[-2]
                        y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                    else:  # Extrapolation is set to constant
                        y = yData[0] if x < xmin else yData[-1]
                return y

            self.getValueOpt = getValueOpt

        elif self.__interpolation__ == "linear":

            def getValueOpt(x):
                xInterval = np.searchsorted(xData, x)
                # Interval found... interpolate... or extrapolate
                if xmin <= x <= xmax:
                    # Interpolate
                    dx = float(xData[xInterval] - xData[xInterval - 1])
                    dy = float(yData[xInterval] - yData[xInterval - 1])
                    y = (x - xData[xInterval - 1]) * (dy / dx) + yData[xInterval - 1]
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        xInterval = 1 if x < xmin else -1
                        dx = float(xData[xInterval] - xData[xInterval - 1])
                        dy = float(yData[xInterval] - yData[xInterval - 1])
                        y = (x - xData[xInterval - 1]) * (dy / dx) + yData[
                            xInterval - 1
                        ]
                    else:  # Extrapolation is set to constant
                        y = yData[0] if x < xmin else yData[-1]
                return y

            self.getValueOpt = getValueOpt

        elif self.__interpolation__ == "akima":
            coeffs = np.array(self.__akimaCoefficients__)

            def getValueOpt(x):
                xInterval = np.searchsorted(xData, x)
                # Interval found... interpolate... or extrapolate
                if xmin <= x <= xmax:
                    # Interpolate
                    xInterval = xInterval if xInterval != 0 else 1
                    a = coeffs[4 * xInterval - 4 : 4 * xInterval]
                    y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        a = coeffs[:4] if x < xmin else coeffs[-4:]
                        y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                    else:  # Extrapolation is set to constant
                        y = yData[0] if x < xmin else yData[-1]
                return y

            self.getValueOpt = getValueOpt

        elif self.__interpolation__ == "polynomial":
            coeffs = self.__polynomialCoefficients__

            def getValueOpt(x):
                # Interpolate... or extrapolate
                if xmin <= x <= xmax:
                    # Interpolate
                    y = 0
                    for i in range(len(coeffs)):
                        y += coeffs[i] * (x**i)
                else:
                    # Extrapolate
                    if extrapolation == 0:  # Extrapolation == zero
                        y = 0
                    elif extrapolation == 1:  # Extrapolation == natural
                        y = 0
                        for i in range(len(coeffs)):
                            y += coeffs[i] * (x**i)
                    else:  # Extrapolation is set to constant
                        y = yData[0] if x < xmin else yData[-1]
                return y

            self.getValueOpt = getValueOpt

        elif self.__interpolation__ == "shepard":
            xData = self.source[:, 0:-1]  # Support for N-Dimensions
            len_yData = len(yData)  # A little speed up

            def getValueOpt(*args):
                x = np.array([[float(x) for x in list(args)]])
                numeratorSum = 0
                denominatorSum = 0
                for i in range(len_yData):
                    sub = xData[i] - x
                    distance = np.linalg.norm(sub)
                    if distance == 0:
                        numeratorSum = yData[i]
                        denominatorSum = 1
                        break
                    else:
                        weight = distance ** (-3)
                        numeratorSum = numeratorSum + yData[i] * weight
                        denominatorSum = denominatorSum + weight
                return numeratorSum / denominatorSum

            self.getValueOpt = getValueOpt

        # Returns self
        return self

    def setDiscrete(
        self,
        lower=0,
        upper=10,
        samples=200,
        interpolation="spline",
        extrapolation="constant",
        oneByOne=True,
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
        oneByOne : boolean, optional
            If True, evaluate Function in each sample point separately. If
            False, evaluates Function in vectorized form. Default is True.

        Returns
        -------
        self : Function
        """
        if self.__domDim__ == 1:
            Xs = np.linspace(lower, upper, samples)
            Ys = self.getValue(Xs.tolist()) if oneByOne else self.getValue(Xs)
            self.source = np.concatenate(([Xs], [Ys])).transpose()
            self.setInterpolation(interpolation)
            self.setExtrapolation(extrapolation)
        elif self.__domDim__ == 2:
            lower = 2 * [lower] if isinstance(lower, (int, float)) else lower
            upper = 2 * [upper] if isinstance(upper, (int, float)) else upper
            sam = 2 * [samples] if isinstance(samples, (int, float)) else samples
            # Create nodes to evaluate function
            Xs = np.linspace(lower[0], upper[0], sam[0])
            Ys = np.linspace(lower[1], upper[1], sam[1])
            Xs, Ys = np.meshgrid(Xs, Ys)
            Xs, Ys = Xs.flatten(), Ys.flatten()
            mesh = [[Xs[i], Ys[i]] for i in range(len(Xs))]
            # Evaluate function at all mesh nodes and convert it to matrix
            Zs = np.array(self.getValue(mesh))
            self.source = np.concatenate(([Xs], [Ys], [Zs])).transpose()
            self.__interpolation__ = "shepard"
        return self

    def setDiscreteBasedOnModel(self, modelFunction, oneByOne=True):
        """This method transforms the domain of Function instance into a list of
        discrete points based on the domain of a model Function instance. It does so by
        retrieving the domain, domain name, interpolation method and extrapolation
        method of the model Function instance. It then evaluates the original Function
        instance in all points of the retrieved domain to generate the list of discrete
        points that will be used for interpolation when this Function is called.

        Parameters
        ----------
        modelFunction : Function
            Function object that will be used to define the sampling points,
            interpolation method and extrapolation method.
            Must be a Function whose source attribute is a list (i.e. a list based
            Function instance).
            Must have the same domain dimension as the Function to be discretized.

        oneByOne : boolean, optional
            If True, evaluate Function in each sample point separately. If
            False, evaluates Function in vectorized form. Default is True.

        Returns
        -------
        self : Function

        See also
        --------
        Function.setDiscrete

        Examples
        --------
        This method is particularly useful when algebraic operations are carried out
        using Function instances defined by different discretized domains (same range,
        but different mesh size). Once an algebraic operation is done, it will not
        directly be applied between the list of discrete points of the two Function
        instances. Instead, the result will be a Function instance defined by a callable
        that calls both Function instances and performs the operation. This makes the
        evaluation of the resulting Function inefficient, due to extra function calling
        overhead and multiple interpolations being carried out.

        >>> from rocketpy import Function
        >>> f = Function([(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)])
        >>> g = Function([(0, 0), (2, 2), (4, 4)])
        >>> h = f * g
        >>> type(h.source)
        <class 'function'>

        Therefore, it is good practice to make sure both Function instances are defined
        by the same domain, i.e. by the same list of mesh points. This way, the
        algebraic operation will be carried out directly between the lists of discrete
        points, generating a new Function instance defined by this result. When it is
        evaluated, there are no extra function calling overheads neither multiple
        interpolations.

        >>> g.setDiscreteBasedOnModel(f)
        Function from R1 to R1 : (Scalar) → (Scalar)
        >>> h = f * g
        >>> h.source
        array([[ 0.,  0.],
               [ 1.,  1.],
               [ 2.,  8.],
               [ 3., 27.],
               [ 4., 64.]])

        Notes
        -----
        1. This method performs in place replacement of the original Function object
        source.

        2. This method is similar to setDiscrete, but it uses the domain of a model
        Function to define the domain of the new Function instance.
        """
        if not isinstance(modelFunction.source, np.ndarray):
            raise TypeError("modelFunction must be a list based Function.")
        if modelFunction.__domDim__ != self.__domDim__:
            raise ValueError("modelFunction must have the same domain dimension.")

        if self.__domDim__ == 1:
            Xs = modelFunction.source[:, 0]
            Ys = self.getValue(Xs.tolist()) if oneByOne else self.getValue(Xs)
            self.source = np.concatenate(([Xs], [Ys])).transpose()
        elif self.__domDim__ == 2:
            # Create nodes to evaluate function
            Xs = modelFunction.source[:, 0]
            Ys = modelFunction.source[:, 1]
            Xs, Ys = np.meshgrid(Xs, Ys)
            Xs, Ys = Xs.flatten(), Ys.flatten()
            mesh = [[Xs[i], Ys[i]] for i in range(len(Xs))]
            # Evaluate function at all mesh nodes and convert it to matrix
            Zs = np.array(self.getValue(mesh))
            self.source = np.concatenate(([Xs], [Ys], [Zs])).transpose()

        self.setInterpolation(modelFunction.__interpolation__)
        self.setExtrapolation(modelFunction.__extrapolation__)
        return self

    def reset(
        self,
        inputs=None,
        outputs=None,
        interpolation=None,
        extrapolation=None,
    ):
        """This method allows the user to reset the inputs, outputs, interpolation
        and extrapolation settings of a Function object, all at once, without
        having to call each of the corresponding methods.

        Parameters
        ----------
        inputs : string, sequence of strings, optional
            List of input variable names. If None, the original inputs are kept.
            See Function.setInputs for more information.
        outputs : string, sequence of strings, optional
            List of output variable names. If None, the original outputs are kept.
            See Function.setOutputs for more information.
        interpolation : string, optional
            Interpolation method to be used if source type is ndarray.
            See Function.setInterpolation for more information.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            See Function.setExtrapolation for more information.

        Examples
        --------
        A simple use case is to reset the inputs and outputs of a Function object
        that has been defined by algebraic manipulation of other Function objects.

        >>> from rocketpy import Function
        >>> v = Function(lambda t: (9.8*t**2)/2, inputs='t', outputs='v')
        >>> mass = 10 # Mass
        >>> kinetic_energy = mass * v**2 / 2
        >>> v.getInputs(), v.getOutputs()
        (['t'], ['v'])
        >>> kinetic_energy
        Function from R1 to R1 : (x) → (Scalar)
        >>> kinetic_energy.reset(inputs='t', outputs='Kinetic Energy');
        Function from R1 to R1 : (t) → (Kinetic Energy)

        Returns
        -------
        self : Function
        """
        if inputs is not None:
            self.setInputs(inputs)
        if outputs is not None:
            self.setOutputs(outputs)
        if interpolation is not None and interpolation != self.__interpolation__:
            self.setInterpolation(interpolation)
        if extrapolation is not None and extrapolation != self.__extrapolation__:
            self.setExtrapolation(extrapolation)

        return self

    # Define all get methods
    def getInputs(self):
        "Return tuple of inputs of the function."
        return self.__inputs__

    def getOutputs(self):
        "Return tuple of outputs of the function."
        return self.__outputs__

    def getSource(self):
        "Return source list or function of the Function."
        return self.source

    def getImageDim(self):
        "Return int describing dimension of the image space of the function."
        return self.__imgDim__

    def getDomainDim(self):
        "Return int describing dimension of the domain space of the function."
        return self.__domDim__

    def getInterpolationMethod(self):
        "Return string describing interpolation method used."
        return self.__interpolation__

    def getExtrapolationMethod(self):
        "Return string describing extrapolation method used."
        return self.__extrapolation__

    def getValue(self, *args):
        """This method returns the value of the Function at the specified
        point. See Function.getValueOpt for a faster, but limited,
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
        """
        # Return value for Function of function type
        if callable(self.source):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                if isinstance(args[0][0], (tuple, list)):
                    return [self.source(*arg) for arg in args[0]]
                else:
                    return [self.source(arg) for arg in args[0]]
            elif len(args) == 1 and isinstance(args[0], np.ndarray):
                return self.source(args[0])
            else:
                return self.source(*args)
        # Returns value for shepard interpolation
        elif self.__interpolation__ == "shepard":
            if isinstance(args[0], (list, tuple)):
                x = list(args[0])
            else:
                x = [[float(x) for x in list(args)]]
            ans = x
            xData = self.source[:, 0:-1]
            yData = self.source[:, -1]
            for i in range(len(x)):
                numeratorSum = 0
                denominatorSum = 0
                for o in range(len(yData)):
                    sub = xData[o] - x[i]
                    distance = (sub.dot(sub)) ** (0.5)
                    # print(xData[o], x[i], distance)
                    if distance == 0:
                        numeratorSum = yData[o]
                        denominatorSum = 1
                        break
                    else:
                        weight = distance ** (-3)
                        numeratorSum = numeratorSum + yData[o] * weight
                        denominatorSum = denominatorSum + weight
                ans[i] = numeratorSum / denominatorSum
            return ans if len(ans) > 1 else ans[0]
        # Returns value for polynomial interpolation function type
        elif self.__interpolation__ == "polynomial":
            if isinstance(args[0], (int, float)):
                args = [list(args)]
            x = np.array(args[0])
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            xmin, xmax = xData[0], xData[-1]
            coeffs = self.__polynomialCoefficients__
            A = np.zeros((len(args[0]), coeffs.shape[0]))
            for i in range(coeffs.shape[0]):
                A[:, i] = x**i
            ans = A.dot(coeffs).tolist()
            for i in range(len(x)):
                if not (xmin <= x[i] <= xmax):
                    if self.__extrapolation__ == "constant":
                        ans[i] = yData[0] if x[i] < xmin else yData[-1]
                    elif self.__extrapolation__ == "zero":
                        ans[i] = 0
            return ans if len(ans) > 1 else ans[0]
        # Returns value for spline, akima or linear interpolation function type
        elif self.__interpolation__ in ["spline", "akima", "linear"]:
            if isinstance(args[0], (int, float, complex)):
                args = [list(args)]
            x = [arg for arg in args[0]]
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            xIntervals = np.searchsorted(xData, x)
            xmin, xmax = xData[0], xData[-1]
            if self.__interpolation__ == "spline":
                coeffs = self.__splineCoefficients__
                for i in range(len(x)):
                    if x[i] == xmin or x[i] == xmax:
                        x[i] = yData[xIntervals[i]]
                    elif xmin < x[i] < xmax or (self.__extrapolation__ == "natural"):
                        if not xmin < x[i] < xmax:
                            a = coeffs[:, 0] if x[i] < xmin else coeffs[:, -1]
                            x[i] = x[i] - xData[0] if x[i] < xmin else x[i] - xData[-2]
                        else:
                            a = coeffs[:, xIntervals[i] - 1]
                            x[i] = x[i] - xData[xIntervals[i] - 1]
                        x[i] = a[3] * x[i] ** 3 + a[2] * x[i] ** 2 + a[1] * x[i] + a[0]
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == "zero":
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = yData[0] if x[i] < xmin else yData[-1]
            elif self.__interpolation__ == "linear":
                for i in range(len(x)):
                    # Interval found... interpolate... or extrapolate
                    inter = xIntervals[i]
                    if xmin <= x[i] <= xmax:
                        # Interpolate
                        dx = float(xData[inter] - xData[inter - 1])
                        dy = float(yData[inter] - yData[inter - 1])
                        x[i] = (x[i] - xData[inter - 1]) * (dy / dx) + yData[inter - 1]
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == "zero":  # Extrapolation == zero
                            x[i] = 0
                        elif (
                            self.__extrapolation__ == "natural"
                        ):  # Extrapolation == natural
                            inter = 1 if x[i] < xmin else -1
                            dx = float(xData[inter] - xData[inter - 1])
                            dy = float(yData[inter] - yData[inter - 1])
                            x[i] = (x[i] - xData[inter - 1]) * (dy / dx) + yData[
                                inter - 1
                            ]
                        else:  # Extrapolation is set to constant
                            x[i] = yData[0] if x[i] < xmin else yData[-1]
            else:
                coeffs = self.__akimaCoefficients__
                for i in range(len(x)):
                    if x[i] == xmin or x[i] == xmax:
                        x[i] = yData[xIntervals[i]]
                    elif xmin < x[i] < xmax or (self.__extrapolation__ == "natural"):
                        if not (xmin < x[i] < xmax):
                            a = coeffs[:4] if x[i] < xmin else coeffs[-4:]
                        else:
                            a = coeffs[4 * xIntervals[i] - 4 : 4 * xIntervals[i]]
                        x[i] = a[3] * x[i] ** 3 + a[2] * x[i] ** 2 + a[1] * x[i] + a[0]
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == "zero":
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = yData[0] if x[i] < xmin else yData[-1]
            if isinstance(args[0], np.ndarray):
                return np.array(x)
            else:
                return x if len(x) > 1 else x[0]

    def getValueOpt_deprecated(self, *args):
        """THE CODE BELOW IS HERE FOR DOCUMENTATION PURPOSES ONLY. IT WAS
        REPLACED FOR ALL INSTANCES BY THE FUNCTION.SETGETVALUEOPT METHOD.

        This method returns the value of the Function at the specified
        point in a limited but optimized manner. See Function.getValue for an
        implementation which allows more kinds of inputs.
        This method optimizes the Function.getValue method by only
        implementing function evaluations of single inputs, i.e., it is not
        vectorized. Furthermore, it actually implements a different method
        for each interpolation type, eliminating some if statements.
        Currently supports callables and spline, linear, akima, polynomial and
        shepard interpolated Function objects.

        Parameters
        ----------
        args : scalar
            Value where the Function is to be evaluated. If the Function is
            1-D, only one argument is expected, which may be an int or a float
            If the function is N-D, N arguments must be given, each one being
            an int or a float.

        Returns
        -------
        x : scalar
        """
        # Callables
        if callable(self.source):
            return self.source(*args)

        # Interpolated Function
        # Retrieve general info
        xData = self.source[:, 0]
        yData = self.source[:, 1]
        xmin, xmax = xData[0], xData[-1]
        if self.__extrapolation__ == "zero":
            extrapolation = 0  # Extrapolation is zero
        elif self.__extrapolation__ == "natural":
            extrapolation = 1  # Extrapolation is natural
        else:
            extrapolation = 2  # Extrapolation is constant

        # Interpolate this info for each interpolation type
        # Spline
        if self.__interpolation__ == "spline":
            x = args[0]
            coeffs = self.__splineCoefficients__
            xInterval = np.searchsorted(xData, x)
            # Interval found... interpolate... or extrapolate
            if xmin <= x <= xmax:
                # Interpolate
                xInterval = xInterval if xInterval != 0 else 1
                a = coeffs[:, xInterval - 1]
                x = x - xData[xInterval - 1]
                y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
            else:
                # Extrapolate
                if extrapolation == 0:  # Extrapolation == zero
                    y = 0
                elif extrapolation == 1:  # Extrapolation == natural
                    a = coeffs[:, 0] if x < xmin else coeffs[:, -1]
                    x = x - xData[0] if x < xmin else x - xData[-2]
                    y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:  # Extrapolation is set to constant
                    y = yData[0] if x < xmin else yData[-1]
            return y
        # Linear
        elif self.__interpolation__ == "linear":
            x = args[0]
            xInterval = np.searchsorted(xData, x)
            # Interval found... interpolate... or extrapolate
            if xmin <= x <= xmax:
                # Interpolate
                dx = float(xData[xInterval] - xData[xInterval - 1])
                dy = float(yData[xInterval] - yData[xInterval - 1])
                y = (x - xData[xInterval - 1]) * (dy / dx) + yData[xInterval - 1]
            else:
                # Extrapolate
                if extrapolation == 0:  # Extrapolation == zero
                    y = 0
                elif extrapolation == 1:  # Extrapolation == natural
                    xInterval = 1 if x < xmin else -1
                    dx = float(xData[xInterval] - xData[xInterval - 1])
                    dy = float(yData[xInterval] - yData[xInterval - 1])
                    y = (x - xData[xInterval - 1]) * (dy / dx) + yData[xInterval - 1]
                else:  # Extrapolation is set to constant
                    y = yData[0] if x < xmin else yData[-1]
            return y
        # Akima
        elif self.__interpolation__ == "akima":
            x = args[0]
            coeffs = np.array(self.__akimaCoefficients__)
            xInterval = np.searchsorted(xData, x)
            # Interval found... interpolate... or extrapolate
            if xmin <= x <= xmax:
                # Interpolate
                xInterval = xInterval if xInterval != 0 else 1
                a = coeffs[4 * xInterval - 4 : 4 * xInterval]
                y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
            else:
                # Extrapolate
                if extrapolation == 0:  # Extrapolation == zero
                    y = 0
                elif extrapolation == 1:  # Extrapolation == natural
                    a = coeffs[:4] if x < xmin else coeffs[-4:]
                    y = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:  # Extrapolation is set to constant
                    y = yData[0] if x < xmin else yData[-1]
            return y
        # Polynomial
        elif self.__interpolation__ == "polynomial":
            x = args[0]
            coeffs = self.__polynomialCoefficients__
            # Interpolate... or extrapolate
            if xmin <= x <= xmax:
                # Interpolate
                y = 0
                for i in range(len(coeffs)):
                    y += coeffs[i] * (x**i)
            else:
                # Extrapolate
                if extrapolation == 0:  # Extrapolation == zero
                    y = 0
                elif extrapolation == 1:  # Extrapolation == natural
                    y = 0
                    for i in range(len(coeffs)):
                        y += coeffs[i] * (x**i)
                else:  # Extrapolation is set to constant
                    y = yData[0] if x < xmin else yData[-1]
            return y
        # Shepard
        elif self.__interpolation__ == "shepard":
            xData = self.source[:, 0:-1]  # Support for N-Dimensions
            len_yData = len(yData)  # A little speed up
            x = np.array([[float(x) for x in list(args)]])
            numeratorSum = 0
            denominatorSum = 0
            for i in range(len_yData):
                sub = xData[i] - x
                distance = np.linalg.norm(sub)
                if distance == 0:
                    numeratorSum = yData[i]
                    denominatorSum = 1
                    break
                else:
                    weight = distance ** (-3)
                    numeratorSum = numeratorSum + yData[i] * weight
                    denominatorSum = denominatorSum + weight
            return numeratorSum / denominatorSum

    def getValueOpt2(self, *args):
        """DEPRECATED!! - See Function.getValueOpt for new version.
        This method returns the value of the Function at the specified
        point in a limited but optimized manner. See Function.getValue for an
        implementation which allows more kinds of inputs.
        This method optimizes the Function.getValue method by only
        implementing function evaluations of single inputs, i.e., it is not
        vectorized. Furthermore, it actually implements a different method
        for each interpolation type, eliminating some if statements.
        Finally, it uses Numba to compile the methods, which further optimizes
        the implementation.
        The code below is here for documentation purposes only. It is
        overwritten for all instances by the Function.setGetValuteOpt2 method.

        Parameters
        ----------
        args : scalar
            Value where the Function is to be evaluated. If the Function is
            1-D, only one argument is expected, which may be an int or a float
            If the function is N-D, N arguments must be given, each one being
            an int or a float.

        Returns
        -------
        x : scalar
        """
        # Returns value for function function type
        if callable(self.source):
            return self.source(*args)
        # Returns value for spline, akima or linear interpolation function type
        elif self.__interpolation__ in ["spline", "akima", "linear"]:
            x = args[0]
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            # Hunt in intervals near the last interval which was used.
            xInterval = self.last_interval
            if xData[xInterval - 1] <= x <= xData[xInterval]:
                pass
            else:
                xInterval = np.searchsorted(xData, x)
                self.last_interval = xInterval if xInterval < len(xData) else 0
            # Interval found... keep going
            xmin, xmax = xData[0], xData[-1]
            if self.__interpolation__ == "spline":
                coeffs = self.__splineCoefficients__
                if x == xmin or x == xmax:
                    x = yData[xInterval]
                elif xmin < x < xmax or (self.__extrapolation__ == "natural"):
                    if not xmin < x < xmax:
                        a = coeffs[:, 0] if x < xmin else coeffs[:, -1]
                        x = x - xData[0] if x < xmin else x - xData[-2]
                    else:
                        a = coeffs[:, xInterval - 1]
                        x = x - xData[xInterval - 1]
                    x = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:
                    # Extrapolate
                    if self.__extrapolation__ == "zero":
                        x = 0
                    else:  # Extrapolation is set to constant
                        x = yData[0] if x < xmin else yData[-1]
            elif self.__interpolation__ == "linear":
                if x == xmin or x == xmax:
                    x = yData[xInterval]
                elif xmin < x < xmax or (self.__extrapolation__ == "natural"):
                    dx = float(xData[xInterval] - xData[xInterval - 1])
                    dy = float(yData[xInterval] - yData[xInterval - 1])
                    x = (x - xData[xInterval - 1]) * (dy / dx) + yData[xInterval - 1]
                elif self.__extrapolation__ == "natural":
                    y0 = yData[0] if x < xmin else yData[-1]
                    xInterval = 1 if x < xmin else -1
                    dx = float(xData[xInterval] - xData[xInterval - 1])
                    dy = float(yData[xInterval] - yData[xInterval - 1])
                    x = (x - xData[xInterval - 1]) * (dy / dx) + y0
                else:
                    # Extrapolate
                    if self.__extrapolation__ == "zero":
                        x = 0
                    else:  # Extrapolation is set to constant
                        x = yData[0] if x < xmin else yData[-1]
            else:
                if self.__interpolation__ == "akima":
                    coeffs = self.__akimaCoefficients__
                if x == xmin or x == xmax:
                    x = yData[xInterval]
                elif xmin < x < xmax:
                    a = coeffs[4 * xInterval - 4 : 4 * xInterval]
                    x = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                elif self.__extrapolation__ == "natural":
                    a = coeffs[:4] if x < xmin else coeffs[-4:]
                    x = a[3] * x**3 + a[2] * x**2 + a[1] * x + a[0]
                else:
                    # Extrapolate
                    if self.__extrapolation__ == "zero":
                        x = 0
                    else:  # Extrapolation is set to constant
                        x = yData[0] if x < xmin else yData[-1]
            return x

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

    # Define all conversion methods
    def toFrequencyDomain(self, lower, upper, samplingFrequency, removeDC=True):
        """Performs the conversion of the Function to the Frequency Domain and returns
        the result. This is done by taking the Fourier transform of the Function.
        The resulting frequency domain is symmetric, i.e., the negative frequencies are
        included as well.

        Parameters
        ----------
        lower : float
            Lower bound of the time range.
        upper : float
            Upper bound of the time range.
        samplingFrequency : float
            Sampling frequency at which to perform the Fourier transform.
        removeDC : bool, optional
            If True, the DC component is removed from the Fourier transform.

        Returns
        -------
        Function
            The Function in the frequency domain.

        Examples
        --------
        >>> from rocketpy import Function
        >>> import numpy as np
        >>> mainFrequency = 10 # Hz
        >>> time = np.linspace(0, 10, 1000)
        >>> signal = np.sin(2 * np.pi * mainFrequency * time)
        >>> timeDomain = Function(np.array([time, signal]).T)
        >>> frequencyDomain = timeDomain.toFrequencyDomain(lower=0, upper=10, samplingFrequency=100)
        >>> peakFrequenciesIndex = np.where(frequencyDomain[:, 1] > 0.001)
        >>> peakFrequencies = frequencyDomain[peakFrequenciesIndex, 0]
        >>> print(peakFrequencies)
        [[-10.  10.]]
        """
        # Get the time domain data
        samplingTimeStep = 1.0 / samplingFrequency
        samplingRange = np.arange(lower, upper, samplingTimeStep)
        numberOfSamples = len(samplingRange)
        sampledPoints = self(samplingRange)
        if removeDC:
            sampledPoints -= np.mean(sampledPoints)
        FourierAmplitude = np.abs(np.fft.fft(sampledPoints) / (numberOfSamples / 2))
        FourierFrequencies = np.fft.fftfreq(numberOfSamples, samplingTimeStep)
        return Function(
            source=np.array([FourierFrequencies, FourierAmplitude]).T,
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
            return self.getValue(*args)

    def __str__(self):
        "Return a string representation of the Function"
        return (
            "Function from R"
            + str(self.__domDim__)
            + " to R"
            + str(self.__imgDim__)
            + " : ("
            + ", ".join(self.__inputs__)
            + ") → ("
            + ", ".join(self.__outputs__)
            + ")"
        )

    def __repr__(self):
        "Return a string representation of the Function"
        return (
            "Function from R"
            + str(self.__domDim__)
            + " to R"
            + str(self.__imgDim__)
            + " : ("
            + ", ".join(self.__inputs__)
            + ") → ("
            + ", ".join(self.__outputs__)
            + ")"
        )

    def setTitle(self, title):
        self.title = title

    def plot(self, *args, **kwargs):
        """Call Function.plot1D if Function is 1-Dimensional or call
        Function.plot2D if Function is 2-Dimensional and forward arguments
        and key-word arguments."""
        if isinstance(self, list):
            # Compare multiple plots
            Function.comparePlots(self)
        else:
            if self.__domDim__ == 1:
                self.plot1D(*args, **kwargs)
            elif self.__domDim__ == 2:
                self.plot2D(*args, **kwargs)
            else:
                print("Error: Only functions with 1D or 2D domains are plottable!")

    def plot1D(
        self,
        lower=None,
        upper=None,
        samples=1000,
        forceData=False,
        forcePoints=False,
        returnObject=False,
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
        forceData : Boolean, optional
            If Function is given by an interpolated dataset, setting forceData
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        forcePoints : Boolean, optional
            Setting forcePoints to True will plot all points, as a scatter, in
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
            xData = self.source[:, 0]
            xmin, xmax = xData[0], xData[-1]
            lower = xmin if lower is None else lower
            upper = xmax if upper is None else upper
            # Plot data points if forceData = True
            tooLow = True if xmin >= lower else False
            tooHigh = True if xmax <= upper else False
            loInd = 0 if tooLow else np.where(xData >= lower)[0][0]
            upInd = len(xData) - 1 if tooHigh else np.where(xData <= upper)[0][0]
            points = self.source[loInd : (upInd + 1), :].T.tolist()
            if forceData:
                plt.scatter(points[0], points[1], marker="o")
        # Calculate function at mesh nodes
        x = np.linspace(lower, upper, samples)
        y = self.getValue(x.tolist())
        # Plots function
        if forcePoints:
            plt.scatter(x, y, marker="o")
        plt.plot(x, y)
        # Turn on grid and set title and axis
        plt.grid(True)
        plt.title(self.title)
        plt.xlabel(self.__inputs__[0].title())
        plt.ylabel(self.__outputs__[0].title())
        plt.show()
        if returnObject:
            return fig, ax

    def plot2D(
        self,
        lower=None,
        upper=None,
        samples=[30, 30],
        forceData=True,
        dispType="surface",
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
        forceData : Boolean, optional
            If Function is given by an interpolated dataset, setting forceData
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        dispType : string, optional
            Display type of plotted graph, which can be surface, wireframe,
            contour, or contourf. Default value is surface.

        Returns
        -------
        None
        """
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
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            xMin, xMax = xData.min(), xData.max()
            yMin, yMax = yData.min(), yData.max()
            lower = [xMin, yMin] if lower is None else lower
            lower = 2 * [lower] if isinstance(lower, (int, float)) else lower
            upper = [xMax, yMax] if upper is None else upper
            upper = 2 * [upper] if isinstance(upper, (int, float)) else upper
            # Plot data points if forceData = True
            if forceData:
                axes.scatter(xData, yData, self.source[:, -1])
        # Create nodes to evaluate function
        x = np.linspace(lower[0], upper[0], samples[0])
        y = np.linspace(lower[1], upper[1], samples[1])
        meshX, meshY = np.meshgrid(x, y)
        meshXFlat, meshYFlat = meshX.flatten(), meshY.flatten()
        mesh = [[meshXFlat[i], meshYFlat[i]] for i in range(len(meshXFlat))]
        # Evaluate function at all mesh nodes and convert it to matrix
        z = np.array(self.getValue(mesh)).reshape(meshX.shape)
        # Plot function
        if dispType == "surface":
            surf = axes.plot_surface(
                meshX,
                meshY,
                z,
                rstride=1,
                cstride=1,
                # cmap=cm.coolwarm,
                linewidth=0,
                alpha=0.6,
            )
            figure.colorbar(surf)
        elif dispType == "wireframe":
            axes.plot_wireframe(meshX, meshY, z, rstride=1, cstride=1)
        elif dispType == "contour":
            figure.clf()
            CS = plt.contour(meshX, meshY, z)
            plt.clabel(CS, inline=1, fontsize=10)
        elif dispType == "contourf":
            figure.clf()
            CS = plt.contour(meshX, meshY, z)
            plt.contourf(meshX, meshY, z)
            plt.clabel(CS, inline=1, fontsize=10)
        # axes.contourf(meshX, meshY, z, zdir='x', offset=xMin, cmap=cm.coolwarm)
        # axes.contourf(meshX, meshY, z, zdir='y', offset=yMax, cmap=cm.coolwarm)
        plt.title(self.title)
        axes.set_xlabel(self.__inputs__[0].title())
        axes.set_ylabel(self.__inputs__[1].title())
        axes.set_zlabel(self.__outputs__[0].title())
        plt.show()

    @staticmethod
    def comparePlots(
        plot_list,
        lower=None,
        upper=None,
        samples=1000,
        title="",
        xlabel="",
        ylabel="",
        forceData=False,
        forcePoints=False,
        returnObject=False,
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
        forceData : Boolean, optional
            If Function is given by an interpolated dataset, setting forceData
            to True will plot all points, as a scatter, in the dataset.
            Default value is False.
        forcePoints : Boolean, optional
            Setting forcePoints to True will plot all points, as a scatter, in
            which the Function was evaluated to plot it. Default value is
            False.

        Returns
        -------
        None
        """
        noRangeSpecified = True if lower is None and upper is None else False
        # Convert to list of tuples if list of Function was given
        plots = []
        for plot in plot_list:
            if isinstance(plot, (tuple, list)):
                plots.append(plot)
            else:
                plots.append((plot, ""))

        # plots = []
        # if isinstance(plot_list[0], (tuple, list)) == False:
        #     for plot in plot_list:
        #         plots.append((plot, " "))
        # else:
        #     plots = plot_list

        # Create plot figure
        fig, ax = plt.subplots()

        # Define a mesh and y values at mesh nodes for plotting
        if lower is None:
            lower = 0
            for plot in plots:
                if not callable(plot[0].source):
                    # Determine boundaries
                    xmin = plot[0].source[0, 0]
                    lower = xmin if xmin < lower else lower
        if upper is None:
            upper = 10
            for plot in plots:
                if not callable(plot[0].source):
                    # Determine boundaries
                    xmax = plot[0].source[-1, 0]
                    upper = xmax if xmax > upper else upper
        x = np.linspace(lower, upper, samples)

        # Iterate to plot all plots
        for plot in plots:
            # Deal with discrete data sets when no range is given
            if noRangeSpecified and not callable(plot[0].source):
                ax.plot(plot[0][:, 0], plot[0][:, 1], label=plot[1])
                if forcePoints:
                    ax.scatter(plot[0][:, 0], plot[0][:, 1], marker="o")
            else:
                # Calculate function at mesh nodes
                y = plot[0].getValue(x.tolist())
                # Plots function
                ax.plot(x, y, label=plot[1])
                if forcePoints:
                    ax.scatter(x, y, marker="o")

        # Plot data points if specified
        if forceData:
            for plot in plots:
                if not callable(plot[0].source):
                    xData = plot[0].source[:, 0]
                    xmin, xmax = xData[0], xData[-1]
                    tooLow = True if xmin >= lower else False
                    tooHigh = True if xmax <= upper else False
                    loInd = 0 if tooLow else np.where(xData >= lower)[0][0]
                    upInd = (
                        len(xData) - 1 if tooHigh else np.where(xData <= upper)[0][0]
                    )
                    points = plot[0].source[loInd : (upInd + 1), :].T.tolist()
                    ax.scatter(points[0], points[1], marker="o")

        # Setup legend
        ax.legend(loc="best", shadow=True)

        # Turn on grid and set title and axis
        plt.grid(True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Show plot
        plt.show()

        if returnObject:
            return fig, ax

    # Define all interpolation methods
    def __interpolatePolynomial__(self):
        """Calculate polynomail coefficients that fit the data exactly."""
        # Find the degree of the polynomial interpolation
        degree = self.source.shape[0] - 1
        # Get x and y values for all supplied points.
        x = self.source[:, 0]
        y = self.source[:, 1]
        # Check if interpolation requires large numbers
        if np.amax(x) ** degree > 1e308:
            print(
                "Polynomial interpolation of too many points can't be done."
                " Once the degree is too high, numbers get too large."
                " The process becomes inefficient. Using spline instead."
            )
            return self.setInterpolation("spline")
        # Create coefficient matrix1
        A = np.zeros((degree + 1, degree + 1))
        for i in range(degree + 1):
            A[:, i] = x**i
        # Solve the system and store the resultant coefficients
        self.__polynomialCoefficients__ = np.linalg.solve(A, y)

    def __interpolateSpline__(self):
        """Calculate natural spline coefficients that fit the data exactly."""
        # Get x and y values for all supplied points
        x = self.source[:, 0]
        y = self.source[:, 1]
        mdim = len(x)
        h = [x[i + 1] - x[i] for i in range(0, mdim - 1)]
        # Initialize the matrix
        Ab = np.zeros((3, mdim))
        # Construct the Ab banded matrix and B vector
        Ab[1, 0] = 1  # A[0, 0] = 1
        B = [0]
        for i in range(1, mdim - 1):
            Ab[2, i - 1] = h[i - 1]  # A[i, i - 1] = h[i - 1]
            Ab[1, i] = 2 * (h[i] + h[i - 1])  # A[i, i] = 2*(h[i] + h[i - 1])
            Ab[0, i + 1] = h[i]  # A[i, i + 1] = h[i]
            B.append(3 * ((y[i + 1] - y[i]) / (h[i]) - (y[i] - y[i - 1]) / (h[i - 1])))
        Ab[1, mdim - 1] = 1  # A[-1, -1] = 1
        B.append(0)
        # Solve the system for c coefficients
        c = linalg.solve_banded((1, 1), Ab, B, True, True)
        # Calculate other coefficients
        b = [
            ((y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3)
            for i in range(0, mdim - 1)
        ]
        d = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(0, mdim - 1)]
        # Store coefficients
        self.__splineCoefficients__ = np.array([y[0:-1], b, c[0:-1], d])

    def __interpolateAkima__(self):
        """Calculate akima spline coefficients that fit the data exactly"""
        # Get x and y values for all supplied points
        x = self.source[:, 0]
        y = self.source[:, 1]
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
            A = np.array(
                [
                    [1, xl, xl**2, xl**3],
                    [1, xr, xr**2, xr**3],
                    [0, 1, 2 * xl, 3 * xl**2],
                    [0, 1, 2 * xr, 3 * xr**2],
                ]
            )
            Y = np.array([yl, yr, dl, dr]).T
            coeffs[4 * i : 4 * i + 4] = np.linalg.solve(A, Y)
            """For some reason this doesn't always work!
            coeffs[4*i] = (dr*xl**2*xr*(-xl + xr) + dl*xl*xr**2*(-xl + xr) +
                           3*xl*xr**2*yl - xr**3*yl + xl**3*yr -
                          3*xl**2*xr*yr)/(xl-xr)**3
            coeffs[4*i+1] = (dr*xl*(xl**2 + xl*xr - 2*xr**2) -
                             xr*(dl*(-2*xl**2 + xl*xr + xr**2) +
                             6*xl*(yl - yr)))/(xl-xr)**3
            coeffs[4*i+2] = (-dl*(xl**2 + xl*xr - 2*xr**2) +
                             dr*(-2*xl**2 + xl*xr + xr**2) +
                             3*(xl + xr)*(yl - yr))/(xl-xr)**3
            coeffs[4*i+3] = (dl*(xl - xr) + dr*(xl - xr) -
                             2*yl + 2*yr)/(xl-xr)**3"""
        self.__akimaCoefficients__ = coeffs

    # Define all possible algebraic operations
    def __truediv__(self, other):
        """Divides a Function object and returns a new Function object
        which gives the result of the division. Only implemented for 1D
        domains.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be divided by. If other and self are Function
            objects which are based on interpolation, have the exact same
            domain (are defined in the same grid points), have the same
            interpolation method and have the same input name, then a
            special implementation is used. This implementation is faster,
            however behavior between grid points is only interpolated,
            not calculated as it would be.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)/other(x).
        """
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have same interpolation and domain
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__interpolation__ == other.__interpolation__
                and self.__inputs__ == other.__inputs__
                and np.array_equal(self.source[:, 0], other.source[:, 0])
            ):
                # Operate on grid values
                with np.errstate(divide="ignore"):
                    Ys = self.source[:, 1] / other.source[:, 1]
                    Ys = np.nan_to_num(Ys)
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + "/" + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (self.getValueOpt2(x) / other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    Ys = self.source[:, 1] / other
                    Xs = self.source[:, 0]
                    source = np.concatenate(([Xs], [Ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + "/" + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    # Create new Function object
                    return Function(source, inputs, outputs, interpolation)
                else:
                    return Function(lambda x: (self.getValueOpt2(x) / other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.getValueOpt2(x) / other(x)))

    def __rtruediv__(self, other):
        """Devides 'other' by a Function object and returns a new Function
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
                Ys = other / self.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + "/" + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (other / self.getValueOpt2(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) / self.getValueOpt2(x)))

    def __pow__(self, other):
        """Raises a Function object to the power of 'other' and
        returns a new Function object which gives the result. Only
        implemented for 1D domains.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be raised to. If other and self are Function
            objects which are based on interpolation, have the exact same
            domain (are defined in the same grid points), have the same
            interpolation method and have the same input name, then a
            special implementation is used. This implementation is faster,
            however behavior between grid points is only interpolated,
            not calculated as it would be.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)**other(x).
        """
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have same interpolation and domain
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__interpolation__ == other.__interpolation__
                and self.__inputs__ == other.__inputs__
                and np.array_equal(self.source[:, 0], other.source[:, 0])
            ):
                # Operate on grid values
                Ys = self.source[:, 1] ** other.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + "**" + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (self.getValueOpt2(x) ** other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    Ys = self.source[:, 1] ** other
                    Xs = self.source[:, 0]
                    source = np.concatenate(([Xs], [Ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + "**" + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    # Create new Function object
                    return Function(source, inputs, outputs, interpolation)
                else:
                    return Function(lambda x: (self.getValue(x) ** other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.getValue(x) ** other(x)))

    def __rpow__(self, other):
        """Raises 'other' to the power of a Function object and returns
        a new Function object which gives the result. Only implemented
        for 1D domains.

        Parameters
        ----------
        other : int, float, callable
            What self will exponentiate.

        Returns
        -------
        result : Function
            A Function object which gives the result of other(x)**self(x).
        """
        # Check if Function object source is array and other is float
        if isinstance(other, (float, int, complex)):
            if isinstance(self.source, np.ndarray):
                # Operate on grid values
                Ys = other ** self.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + "**" + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (other ** self.getValue(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) ** self.getValue(x)))

    def __mul__(self, other):
        """Multiplies a Function object and returns a new Function object
        which gives the result of the multiplication. Only implemented for 1D
        domains.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be multiplied by. If other and self are Function
            objects which are based on interpolation, have the exact same
            domain (are defined in the same grid points), have the same
            interpolation method and have the same input name, then a
            special implementation is used. This implementation is faster,
            however behavior between grid points is only interpolated,
            not calculated as it would be.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)*other(x).
        """
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have same interpolation and domain
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__interpolation__ == other.__interpolation__
                and self.__inputs__ == other.__inputs__
                and np.array_equal(self.source[:, 0], other.source[:, 0])
            ):
                # Operate on grid values
                Ys = self.source[:, 1] * other.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + "*" + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (self.getValue(x) * other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    Ys = self.source[:, 1] * other
                    Xs = self.source[:, 0]
                    source = np.concatenate(([Xs], [Ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + "*" + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    # Create new Function object
                    return Function(source, inputs, outputs, interpolation)
                else:
                    return Function(lambda x: (self.getValue(x) * other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.getValue(x) * other(x)))

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
        # Check if Function object source is array and other is float
        if isinstance(other, (float, int, complex)):
            if isinstance(self.source, np.ndarray):
                # Operate on grid values
                Ys = other * self.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + "*" + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (other * self.getValue(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) * self.getValue(x)))

    def __add__(self, other):
        """Sums a Function object and 'other', returns a new Function
        object which gives the result of the sum. Only implemented for
        1D domains.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be added to. If other and self are Function
            objects which are based on interpolation, have the exact same
            domain (are defined in the same grid points), have the same
            interpolation method and have the same input name, then a
            special implementation is used. This implementation is faster,
            however behavior between grid points is only interpolated,
            not calculated as it would be.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)+other(x).
        """
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have same interpolation and domain
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__interpolation__ == other.__interpolation__
                and self.__inputs__ == other.__inputs__
                and np.array_equal(self.source[:, 0], other.source[:, 0])
            ):
                # Operate on grid values
                Ys = self.source[:, 1] + other.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + " + " + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (self.getValue(x) + other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    Ys = self.source[:, 1] + other
                    Xs = self.source[:, 0]
                    source = np.concatenate(([Xs], [Ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + " + " + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    # Create new Function object
                    return Function(source, inputs, outputs, interpolation)
                else:
                    return Function(lambda x: (self.getValue(x) + other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.getValue(x) + other(x)))

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
        # Check if Function object source is array and other is float
        if isinstance(other, (float, int, complex)):
            if isinstance(self.source, np.ndarray):
                # Operate on grid values
                Ys = other + self.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + " + " + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (other + self.getValue(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) + self.getValue(x)))

    def __sub__(self, other):
        """Subtracts from a Function object and returns a new Function object
        which gives the result of the subtraction. Only implemented for 1D
        domains.

        Parameters
        ----------
        other : Function, int, float, callable
            What self will be subtracted by. If other and self are Function
            objects which are based on interpolation, have the exact same
            domain (are defined in the same grid points), have the same
            interpolation method and have the same input name, then a
            special implementation is used. This implementation is faster,
            however behavior between grid points is only interpolated,
            not calculated as it would be.

        Returns
        -------
        result : Function
            A Function object which gives the result of self(x)-other(x).
        """
        # If other is Function try...
        try:
            # Check if Function objects source is array or callable
            # Check if Function objects have same interpolation and domain
            if (
                isinstance(other.source, np.ndarray)
                and isinstance(self.source, np.ndarray)
                and self.__interpolation__ == other.__interpolation__
                and self.__inputs__ == other.__inputs__
                and np.array_equal(self.source[:, 0], other.source[:, 0])
            ):
                # Operate on grid values
                Ys = self.source[:, 1] - other.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = self.__outputs__[0] + " - " + other.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (self.getValue(x) * other(x)))
        # If other is Float except...
        except AttributeError:
            if isinstance(other, (float, int, complex)):
                # Check if Function object source is array or callable
                if isinstance(self.source, np.ndarray):
                    # Operate on grid values
                    Ys = self.source[:, 1] - other
                    Xs = self.source[:, 0]
                    source = np.concatenate(([Xs], [Ys])).transpose()
                    # Retrieve inputs, outputs and interpolation
                    inputs = self.__inputs__[:]
                    outputs = self.__outputs__[0] + " - " + str(other)
                    outputs = "(" + outputs + ")"
                    interpolation = self.__interpolation__
                    # Create new Function object
                    return Function(source, inputs, outputs, interpolation)
                else:
                    return Function(lambda x: (self.getValue(x) - other))
            # Or if it is just a callable
            elif callable(other):
                return Function(lambda x: (self.getValue(x) - other(x)))

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
        # Check if Function object source is array and other is float
        if isinstance(other, (float, int, complex)):
            if isinstance(self.source, np.ndarray):
                # Operate on grid values
                Ys = other - self.source[:, 1]
                Xs = self.source[:, 0]
                source = np.concatenate(([Xs], [Ys])).transpose()
                # Retrieve inputs, outputs and interpolation
                inputs = self.__inputs__[:]
                outputs = str(other) + " - " + self.__outputs__[0]
                outputs = "(" + outputs + ")"
                interpolation = self.__interpolation__
                # Create new Function object
                return Function(source, inputs, outputs, interpolation)
            else:
                return Function(lambda x: (other - self.getValue(x)))
        # Or if it is just a callable
        elif callable(other):
            return Function(lambda x: (other(x) - self.getValue(x)))

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
        integrationSign = np.sign(b - a)
        if integrationSign == -1:
            a, b = b, a
        # Different implementations depending on interpolation
        if self.__interpolation__ == "spline" and numerical is False:
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            coeffs = self.__splineCoefficients__
            ans = 0
            # Check to see if interval starts before point data
            if a < xData[0]:
                if self.__extrapolation__ == "constant":
                    ans += yData[0] * (xData[0] - a)
                elif self.__extrapolation__ == "natural":
                    c = coeffs[:, 0]
                    subB = a - xData[0]  # subA = 0
                    ans -= (
                        (c[3] * subB**4) / 4
                        + (c[2] * subB**3 / 3)
                        + (c[1] * subB**2 / 2)
                        + c[0] * subB
                    )
                else:
                    # self.__extrapolation__ = 'zero'
                    pass
            # Integrate in subintervals between Xs of given data up to b
            i = 0
            while i < len(xData) - 1 and xData[i] < b:
                if b < xData[i + 1]:
                    subB = b - xData[i]  # subA = 0
                else:
                    subB = xData[i + 1] - xData[i]  # subA = 0
                c = coeffs[:, i]
                subB = xData[i + 1] - xData[i]  # subA = 0
                ans += (
                    (c[3] * subB**4) / 4
                    + (c[2] * subB**3 / 3)
                    + (c[1] * subB**2 / 2)
                    + c[0] * subB
                )
                i += 1
            # Check to see if interval ends after point data
            if b > xData[-1]:
                if self.__extrapolation__ == "constant":
                    ans += yData[-1] * (b - xData[-1])
                elif self.__extrapolation__ == "natural":
                    c = coeffs[:, -1]
                    subA = xData[-1] - xData[-2]
                    subB = b - xData[-2]
                    ans -= (
                        (c[3] * subA**4) / 4
                        + (c[2] * subA**3 / 3)
                        + (c[1] * subA**2 / 2)
                        + c[0] * subA
                    )
                    ans += (
                        (c[3] * subB**4) / 4
                        + (c[2] * subB**3 / 3)
                        + (c[1] * subB**2 / 2)
                        + c[0] * subB
                    )
                else:
                    # self.__extrapolation__ = 'zero'
                    pass
        elif self.__interpolation__ == "linear" and numerical is False:
            # Integrate from a to b using np.trapz
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            # Get data in interval
            xIntegrationData = xData[(xData >= a) & (xData <= b)]
            yIntegrationData = yData[(xData >= a) & (xData <= b)]
            # Add integration limits to data
            if self.__extrapolation__ == "zero":
                if a >= xData[0]:
                    xIntegrationData = np.concatenate(([a], xIntegrationData))
                    yIntegrationData = np.concatenate(([self(a)], yIntegrationData))
                if b <= xData[-1]:
                    xIntegrationData = np.concatenate((xIntegrationData, [b]))
                    yIntegrationData = np.concatenate((yIntegrationData, [self(b)]))
            else:
                xIntegrationData = np.concatenate(([a], xIntegrationData))
                yIntegrationData = np.concatenate(([self(a)], yIntegrationData))
                xIntegrationData = np.concatenate((xIntegrationData, [b]))
                yIntegrationData = np.concatenate((yIntegrationData, [self(b)]))
            # Integrate using np.trapz
            ans = np.trapz(yIntegrationData, xIntegrationData)
        else:
            # Integrate numerically
            ans, _ = integrate.quad(self, a, b, epsabs=0.1, limit=10000)
        return integrationSign * ans

    # Not implemented
    def differentiate(self, x, dx=1e-6):
        return (self.getValue(x + dx) - self.getValue(x - dx)) / (2 * dx)
        # h = (10)**-300
        # z = x + h*1j
        # return self(z).imag/h


def funcify_method(*args, **kwargs):
    """Decorator factory to wrap methods as Function objects and save them as cached
    properties.

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

    >>> from rocketpy.Function import funcify_method
    >>> class Example():
    ...     @funcify_method(inputs=['x'], outputs=['y'])
    ...     def f(self):
    ...         return lambda x: x**2
    >>> example = Example()
    >>> example.f
    Function from R1 to R1 : (x) → (y)

    Normal algebra can be performed afterwards:

    >>> g = 2*example.f + 3
    >>> g(2)
    11

    2. Method which returns a rocketpy.Function instance. An interesting use is to reset
    input and output names after algebraic operations.

    >>> class Example():
    ...     @funcify_method(inputs=['x'], outputs=['x**3'])
    ...     def cube(self):
    ...         f = Function(lambda x: x**2)
    ...         g = Function(lambda x: x**5)
    ...         return g / f
    >>> example = Example()
    >>> example.cube
    Function from R1 to R1 : (x) → (x**3)

    3. Method which is itself a valid rocketpy.Function source argument.

    >>> class Example():
    ...     @funcify_method('x', 'f(x)')
    ...     def f(self, x):
    ...         return x**2
    >>> example = Example()
    >>> example.f
    Function from R1 to R1 : (x) → (f(x))

    In order to reset the cache, just delete de attribute from the instance:

    >>> del example.f

    Once it is requested again, it will be re-created as a new Function object:

    >>> example.f
    Function from R1 to R1 : (x) → (f(x))
    """
    func = None
    if len(args) == 1 and callable(args[0]):
        func = args[0]
        args = []

    class funcify_method_decorator:
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
                    source = lambda *_: self.func(instance, *_)
                    val = Function(source, *args, **kwargs)
                except Exception:
                    raise Exception(
                        "Could not create Function object from method "
                        f"{self.func.__name__}."
                    )

                val.__doc__ = self.__doc__
                cache[self.attrname] = val
            return val

    if func:
        return funcify_method_decorator(func)
    else:
        return funcify_method_decorator


if __name__ == "__main__":
    import doctest

    doctest.testmod()
