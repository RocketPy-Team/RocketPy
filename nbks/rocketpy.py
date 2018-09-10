# -*- coding: utf-8 -*-
"""
Who knows what this even is...

@authors: Giovani Ceotto, Matheus Marques Araujo, Rodrigo Schmitt
"""
import math
import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from datetime import datetime
from inspect import signature
import bisect
try:
    import netCDF4
except ImportError:
    print('Unable to load netCDF4. NetCDF files will not be imported.')


class Function:
    """Class converts a python function or a data sequence into an object
    which can be handled more naturally, enabling easy interpolation,
    extrapolation, ploting and algebra.
    """
    def __init__(self,
                 source,
                 inputs=['Scalar'],
                 outputs=['Scalar'],
                 interpolation=None,
                 extrapolation=None):
        """ Convert source into a Function, to be used more naturally.
        Set inputs, outputs, domain dimension, interpolation and extrapolation
        method, and process the source.

        Parameters
        ----------
        source : function, scalar, ndarray, string
            The actual function. If type is function, it will be called for
            evaluation. If type is int or float, it will be treated as a
            constant function. If ndarray, its poitns will be used for
            interpolation. A ndarray should be as [(x0, y0, z0), (x1, y1, z1),
            (x2, y2, z2), ...] where x0 and y0 are inputs and z0 is output. If
            string, imports file named by the string and treats it as csv.
            The file is converted into ndarray and should not have headers.
        inputs : string, sequence of strigns, optional
            The name of the inputs of the function. Will be used for
            representation and graphing (axis names). 'Scalar' is default.
            If souce is function, int or float and has multiple inputs,
            this parameters must be giving for correct operation.
        outputs : string, sequence of strigns, optional
            The name of the outputs of the function. Will be used for
            representation and graphing (axis names). Scalar is default.
        interpolation : string, optional
            Interpolation method to be used if source type is ndarray.
            For 1-D functions, linear, polynomail, akima and spline are
            supported. For N-D functions, only shepard is suporrted.
            Default for 1-D functions is spline.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function of edge of the interval,
            and 'zero', wich returns zero for all points outside of source
            range. Default for 1-D functions is constant.

        Returns
        -------
        None
        """
        # Set input and output
        self.setInputs(inputs)
        self.setOutputs(outputs)
        # Set interpolation method
        self.__interpolation__ = interpolation
        self.__extrapolation__ = extrapolation
        # Set source
        self.setSource(source)
        self.last_interval = 0
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
        """Set the name and number of the ouput of the Function.

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
            constant function. If ndarray, its poitns will be used for
            interpolation. A ndarray should be as [(x0, y0, z0), (x1, y1, z1),
            (x2, y2, z2), ...] where x0 and y0 are inputs and z0 is output. If
            string, imports file named by the string and treats it as csv.
            The file is converted into ndarray and should not have headers.

        Returns
        -------
        self : Function
        """
        # Import CSV if source is a string and convert values to ndarray
        if isinstance(source, str):
            source = np.loadtxt(source, delimiter=',', dtype=np.float64)
        # Convert to ndarray if source is a list
        if isinstance(source, (list, tuple)):
            source = np.array(source, dtype=np.float64)
        # Convert number source into vectorized lambda function
        if isinstance(source, (int, float)):
            temp = 1*source
            source = lambda x: 0*x + temp
        # Handle callable source or number source
        if callable(source):
            # Set source
            self.source = source
            # Set arguments name and domain dimensions
            parameters = signature(source).parameters
            self.__domDim__ = len(parameters)
            if self.__inputs__ == ['Time (s)']:
                self.__inputs__ = list(parameters)
            # Set interpolation and extrapolation
            self.__interpolation__ = None
            self.__extrapolation__ = None
        # Handle ndarray source
        else:
            # Check to see if dimensions match incoming data set
            newTotalDim = len(source[0, :])
            oldTotalDim = self.__domDim__ + self.__imgDim__
            dV = (self.__inputs__ == ['Scalar'] and
                  self.__outputs__ == ['Scalar'])
            # If they dont, update default values or throw error
            if newTotalDim != oldTotalDim:
                if dV:
                    # Update dimensions and inputs
                    self.__domDim__ = newTotalDim - 1
                    self.__inputs__ = self.__domDim__*self.__inputs__
                else:
                    # User has made a mistake inputting inputs and outputs
                    print('Error in input and output dimensions!')
                    return None
            # Do things if domDim is 1
            if self.__domDim__ == 1:
                source = source[source[:, 0].argsort()]
                # Finally set data source as source
                self.source = source
                # Set default interpolation for point source if it hasnt
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
                    self.setInterpolation('shepard')
            # Update extrapolation method
            if self.__extrapolation__ is None:
                self.setExtrapolation()
        # Return self
        return self

    def setInterpolation(self, method='spline'):
        """Set interpolation method and process data is method requires.

        Parameters
        ----------
        method : string, optional
            Interpolation method to be used if source type is ndarray.
            For 1-D functions, linear, polynomail, akima and spline is
            supported. For N-D functions, only shepard is suporrted.
            Default is 'spline'.

        Returns
        -------
        self : Function
        """
        # Set interpolation method
        self.__interpolation__ = method
        # Spline, akima and polynomial need data processing
        # Shepard, and linear do not
        if method == 'spline':
            self.__interpolateSpline__()
        elif method == 'polynomial':
            self.__interpolatePolynomial__()
        elif method == 'akima':
            self.__interpolateAkima__()
        else:
            pass
        # Returns self
        return self

    def setExtrapolation(self, method='constant'):
        """Set extrapolation behaviour of data set.

        Parameters
        ----------
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function of edge of the interval,
            and 'zero', wich returns zero for all points outside of source
            range. Default is 'zero'.

        Returns
        -------
        self : Function
        """
        # Set extrapolation method
        self.__extrapolation__ = method
        # Return self
        return self

    def setDiscrete(self, lower=0, upper=10, samples=200,
                    interpolation='spline', extrapolation='constant',
                    oneByOne=True):
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
            For 1-D functions, linear, polynomail, akima and spline is
            supported. For N-D functions, only shepard is suporrted.
            Default is 'spline'.
        extrapolation : string, optional
            Extrapolation method to be used if source type is ndarray.
            Options are 'natural', which keeps interpolation, 'constant',
            which returns the value of the function of edge of the interval,
            and 'zero', wich returns zero for all points outside of source
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
            lower = 2*[lower] if isinstance(lower, (int, float)) else lower
            upper = 2*[upper] if isinstance(upper, (int, float)) else upper
            sam = 2*[samples] if isinstance(samples, (int, float)) else samples
            # Create nodes to evaluate function
            Xs = np.linspace(lower[0], upper[0], sam[0])
            Ys = np.linspace(lower[1], upper[1], sam[1])
            Xs, Ys = np.meshgrid(Xs, Ys)
            Xs, Ys = Xs.flatten(), Ys.flatten()
            mesh = [[Xs[i], Ys[i]] for i in range(len(Xs))]
            # Evaluate function at all mesh nodes and convert it to matrix
            Zs = np.array(self.getValue(mesh))
            self.source = np.concatenate(([Xs], [Ys], [Zs])).transpose()
            self.__interpolation__ = 'shepard'
        return self

    # Define all get methods
    def getInputs(self):
        'Return tuple of inputs of the function.'
        return self.__inputs__

    def getOutputs(self):
        'Return tuple of outputs of the function.'
        return self.__outputs__

    def getSource(self):
        'Return source list or function of the function.'
        return self.source

    def getImageDim(self):
        'Return int describing dimension of the image space of the function.'
        return self.__imgDim__

    def getDomainDim(self):
        'Return int describing dimension of the domain space of the function.'
        return self.__domDim__

    def getInterpolationMethod(self):
        'Return string describing interpolation method used.'
        return self.__interpolation__

    def getExtrapolationMethod(self):
        'Return string describing extrapolation method used.'
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
        elif self.__interpolation__ == 'shepard':
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
                    distance = (sub.dot(sub))**(0.5)
                    # print(xData[o], x[i], distance)
                    if distance == 0:
                        numeratorSum = yData[o]
                        denominatorSum = 1
                        break
                    else:
                        weigth = distance**(-3)
                        numeratorSum = numeratorSum + yData[o]*weigth
                        denominatorSum = denominatorSum + weigth
                ans[i] = numeratorSum/denominatorSum
            return ans if len(ans) > 1 else ans[0]
        # Returns value for polynomial interpolation function type
        elif self.__interpolation__ == 'polynomial':
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
                    if self.__extrapolation__ == 'constant':
                        ans[i] = yData[0] if x[i] < xmin else yData[-1]
                    elif self.__extrapolation__ == 'zero':
                        ans[i] = 0
            return ans if len(ans) > 1 else ans[0]
        # Returns value for spline, akima or linear interpolation function type
        elif self.__interpolation__ in ['spline', 'akima', 'linear']:
            if isinstance(args[0], (int, float, complex)):
                args = [list(args)]
            x = [arg for arg in args[0]]
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            xIntervals = np.searchsorted(xData, x)
            xmin, xmax = xData[0], xData[-1]
            if self.__interpolation__ == 'spline':
                coeffs = self.__splineCoefficients__
                for i in range(len(x)):
                    if x[i] == xmin or x[i] == xmax:
                        x[i] = yData[xIntervals[i]]
                    elif xmin < x[i] < xmax or (self.__extrapolation__ ==
                                                'natural'):
                        if not xmin < x[i] < xmax:
                            a = coeffs[:, 0] if x[i] < xmin else coeffs[:, -1]
                            x[i] = x[i] - xData[0] if x[i] < xmin else x[i] - xData[-2]
                        else:
                            a = coeffs[:, xIntervals[i]-1]
                            x[i] = x[i] - xData[xIntervals[i]-1]
                        x[i] = (a[3]*x[i]**3 + a[2]*x[i]**2 + a[1]*x[i] + a[0])
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == 'zero':
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = yData[0] if x[i] < xmin else yData[-1]
            elif self.__interpolation__ == 'linear':
                for i in range(len(x)):
                    inter = xIntervals[i]
                    if x[i] == xmin or x[i] == xmax:
                        x[i] = yData[inter]
                    elif xmin < x[i] < xmax or (self.__extrapolation__ ==
                                                'natural'):
                        if not(xmin < x[i] < xmax):
                            y0 = yData[0] if x[i] < xmin else yData[-1]
                            inter = 1 if x[i] < xmin else -1
                        else:
                            y0 = yData[inter-1]
                        dx = float(xData[inter]-xData[inter-1])
                        dy = float(yData[inter]-yData[inter-1])
                        x[i] = ((x[i]-xData[inter-1])*(dy/dx) + y0)
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == 'zero':
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = yData[0] if x[i] < xmin else yData[-1]
            else:
                coeffs = self.__akimaCoefficients__
                for i in range(len(x)):
                    if x[i] == xmin or x[i] == xmax:
                        x[i] = yData[xIntervals[i]]
                    elif xmin < x[i] < xmax or (self.__extrapolation__ ==
                                                'natural'):
                        if not (xmin < x[i] < xmax):
                            a = coeffs[:4] if x[i] < xmin else coeffs[-4:]
                        else:
                            a = coeffs[4*xIntervals[i]-4:4*xIntervals[i]]
                        x[i] = (a[3]*x[i]**3 + a[2]*x[i]**2 +
                                a[1]*x[i] + a[0])
                    else:
                        # Extrapolate
                        if self.__extrapolation__ == 'zero':
                            x[i] = 0
                        else:  # Extrapolation is set to constant
                            x[i] = yData[0] if x[i] < xmin else yData[-1]
            if isinstance(args[0], np.ndarray):
                return np.array(x)
            else:
                return x if len(x) > 1 else x[0]

    def getValueOpt(self, *args):
        """This method returns the value of the Function at the specified
        point in a limited but optimized manner. See Function.getValue for an
        implementation which allows more kinds of inputs.
        It is optimized for Functions which interpolates lists of data and are
        called repeatedly in such a way that the next call will want to know
        the value of the function at a point near the last point used. 

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
        elif self.__interpolation__ in ['spline', 'akima', 'linear']:
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
            if self.__interpolation__ == 'spline':
                coeffs = self.__splineCoefficients__
                if x == xmin or x == xmax:
                    x = yData[xInterval]
                elif xmin < x < xmax or (self.__extrapolation__ == 'natural'):
                    if not xmin < x < xmax:
                        a = coeffs[:, 0] if x < xmin else coeffs[:, -1]
                        x = x - xData[0] if x < xmin else x - xData[-2]
                    else:
                        a = coeffs[:, xInterval-1]
                        x = x - xData[xInterval-1]
                    x = (a[3]*x**3 + a[2]*x**2 + a[1]*x + a[0])
                else:
                    # Extrapolate
                    if self.__extrapolation__ == 'zero':
                        x = 0
                    else:  # Extrapolation is set to constant
                        x = yData[0] if x < xmin else yData[-1]
            elif self.__interpolation__ == 'linear':
                if x == xmin or x == xmax:
                    x = yData[xInterval]
                elif xmin < x < xmax or (self.__extrapolation__ == 'natural'):
                    dx = float(xData[xInterval]-xData[xInterval-1])
                    dy = float(yData[xInterval]-yData[xInterval-1])
                    x = ((x-xData[xInterval-1])*(dy/dx) + yData[xInterval - 1])
                elif self.__extrapolation__ == 'natural':
                    y0 = yData[0] if x < xmin else yData[-1]
                    xInterval = 1 if x < xmin else -1
                    dx = float(xData[xInterval]-xData[xInterval-1])
                    dy = float(yData[xInterval]-yData[xInterval-1])
                    x = ((x-xData[xInterval-1])*(dy/dx) + y0)
                else:
                    # Extrapolate
                    if self.__extrapolation__ == 'zero':
                        x = 0
                    else:  # Extrapolation is set to constant
                        x = yData[0] if x < xmin else yData[-1]
            else:
                if self.__interpolation__ == 'akima':
                    coeffs = self.__akimaCoefficients__
                if x == xmin or x == xmax:
                    x = yData[xInterval]
                elif xmin < x < xmax:
                    a = coeffs[4*xInterval - 4:4*xInterval]
                    x = (a[3]*x**3 + a[2]*x**2 + a[1]*x + a[0])
                elif self.__extrapolation__ == 'natural':
                    a = coeffs[:4] if x < xmin else coeffs[-4:]
                    x = (a[3]*x**3 + a[2]*x**2 + a[1]*x + a[0])
                else:
                    # Extrapolate
                    if self.__extrapolation__ == 'zero':
                        x = 0
                    else:  # Extrapolation is set to constant
                        x = yData[0] if x < xmin else yData[-1]
            return x

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
        return ('Function from R' + str(self.__domDim__) + ' to R' +
                str(self.__imgDim__) +
                ' : (' +
                ', '.join(self.__inputs__) + ') → (' +
                ', '.join(self.__outputs__)+')')

    def __repr__(self):
        "Return a string representation of the Function"
        return ('Function from R' + str(self.__domDim__) + ' to R' +
                str(self.__imgDim__) +
                ' : (' +
                ', '.join(self.__inputs__) + ') → (' +
                ', '.join(self.__outputs__)+')')

    def plot(self, *args, **kwargs):
        """Call Function.plot1D if Function is 1-Dimensional or call
        Function.plot2D if Function is 2-Dimensional and forward arguments
        and key-word arguments."""
        if self.__domDim__ == 1:
            self.plot1D(*args, **kwargs)
        elif self.__domDim__ == 2:
            self.plot2D(*args, **kwargs)
        else:
            print('Error: Only functions with 1D or 2D domains are plottable!')

    def plot1D(self, lower=None, upper=None, samples=1000, forceData=False,
               forcePoints=False):
        """ Plot 1-Dimensional Function, from a lower limit to an upper limit,
        by sampling the Function several times in the interval. The title of
        the graph is given by the name of the axis, which are taken from
        the Function`s input and ouput names.

        Parameters
        ----------
        lower : scalar, optional
            The lower limit of the interval in which the function is to be
            ploted. The default value for function type Functions is 0. By
            contrast, if the Function is given by a dataset, the default
            value is the start of the dataset.
        upper : scalar, optional
            The upper limit of the interval in which the function is to be
            ploted. The default value for function type Functions is 10. By
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
        figure = plt.figure()
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
            points = self.source[loInd:(upInd+1), :].T.tolist()
            if forceData:
                plt.scatter(points[0], points[1], marker='o')
        # Calculate function at mesh nodes
        x = np.linspace(lower, upper, samples)
        y = self.getValue(x.tolist())
        # Plots function
        if forcePoints:
            plt.scatter(x, y, marker='o')
        plt.plot(x, y)
        # Turn on grid and set title and axis
        plt.grid(True)
        plt.title(self.__outputs__[0].title() + ' x ' +
                  self.__inputs__[0].title())
        plt.xlabel(self.__inputs__[0].title())
        plt.ylabel(self.__outputs__[0].title())
        plt.show()

    def plot2D(self, lower=None, upper=None, samples=[30, 30], forceData=True,
               dispType='surface'):
        """ Plot 2-Dimensional Function, from a lower limit to an upper limit,
        by sampling the Function several times in the interval. The title of
        the graph is given by the name of the axis, which are taken from
        the Function`s inputs and ouput names.

        Parameters
        ----------
        lower : scalar, array of int or float, optional
            The lower limits of the interval in which the function is to be
            ploted, which can be an int or float, which is repeated for both
            axis, or an array specifying the limit for each axis. The default
            value for function type Functions is 0. By contrast, if the
            Function is given by a dataset, the default value is the start of
            the dataset for each axis.
        upper : scalar, array of int or float, optional
            The upper limits of the interval in which the function is to be
            ploted, which can be an int or float, which is repeated for both
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
        axes = figure.gca(projection='3d')
        # Define a mesh and f values at mesh nodes for plotting
        if callable(self.source):
            # Determine boundaries
            lower = [0, 0] if lower is None else lower
            lower = 2*[lower] if isinstance(lower, (int, float)) else lower
            upper = [10, 10] if upper is None else upper
            upper = 2*[upper] if isinstance(upper, (int, float)) else upper
        else:
            # Determine boundaries
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            xMin, xMax = xData.min(), xData.max()
            yMin, yMax = yData.min(), yData.max()
            lower = [xMin, yMin] if lower is None else lower
            lower = 2*[lower] if isinstance(lower, (int, float)) else lower
            upper = [xMax, yMax] if upper is None else upper
            upper = 2*[upper] if isinstance(upper, (int, float)) else upper
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
        if dispType == 'surface':
            surf = axes.plot_surface(meshX, meshY, z, rstride=1, cstride=1,
                                     cmap=cm.coolwarm, linewidth=0, alpha=0.6)
            figure.colorbar(surf)
        elif dispType == 'wireframe':
            axes.plot_wireframe(meshX, meshY, z, rstride=1, cstride=1)
        elif dispType == 'contour':
            figure.clf()
            CS = plt.contour(meshX, meshY, z)
            plt.clabel(CS, inline=1, fontsize=10)
        elif dispType == 'contourf':
            figure.clf()
            CS = plt.contour(meshX, meshY, z)
            plt.contourf(meshX, meshY, z)
            plt.clabel(CS, inline=1, fontsize=10)
        # axes.contourf(meshX, meshY, z, zdir='x', offset=xMin, cmap=cm.coolwarm)
        # axes.contourf(meshX, meshY, z, zdir='y', offset=yMax, cmap=cm.coolwarm)
        plt.title(self.__outputs__[0].title() + ' x ' +
                  self.__inputs__[0].title() + ' x ' +
                  self.__inputs__[1].title())
        axes.set_xlabel(self.__inputs__[0].title())
        axes.set_ylabel(self.__inputs__[1].title())
        axes.set_zlabel(self.__outputs__[0].title())
        plt.show()

    # Define all interpolation methods
    def __interpolatePolynomial__(self):
        """"Calculate polynomail coefficients that fit the data exactly."""
        # Find the degree of the polynomial interpolation
        degree = self.source.shape[0] - 1
        # Get x and y values for all supplied points.
        x = self.source[:, 0]
        y = self.source[:, 1]
        # Check if interpolation requires large numbers
        if np.amax(x)**degree > 1e308:
            print("Polynomial interpolation of too many points can't be done."
                  " Once the degree is too high, numbers get too large."
                  " The process becomes ineficient. Using spline instead.")
            return self.setInterpolation('spline')
        # Create coefficient matrix1
        A = np.zeros((degree + 1, degree + 1))
        for i in range(degree + 1):
            A[:, i] = x**i
        # Solve the system and store the resultant coefficients
        self.__polynomialCoefficients__ = np.linalg.solve(A, y)

    def __interpolateSpline__(self):
        """Calculate splines coefficients that fit the data exactly."""
        # Get x and y values for all supplied points
        x = self.source[:, 0]
        y = self.source[:, 1]
        mdim = len(x)
        h = [x[i+1] - x[i] for i in range(0, mdim - 1)]
        # Initialize de matrix
        Ab = np.zeros((3, mdim))
        # Construct the Ab banded matrix and B vector
        Ab[1, 0] = 1  # A[0, 0] = 1
        B = [0]
        for i in range(1, mdim-1):
            Ab[2, i - 1] = h[i - 1]  # A[i, i - 1] = h[i - 1]
            Ab[1, i] = 2*(h[i] + h[i - 1])  # A[i, i] = 2*(h[i] + h[i - 1])
            Ab[0, i + 1] = h[i]  # A[i, i + 1] = h[i]
            B.append(3*((y[i + 1] - y[i])/(h[i]) - (y[i] - y[i - 1])/(h[i-1])))
        Ab[1, mdim - 1] = 1  # A[-1, -1] = 1
        B.append(0)
        # Solve the system for c coefficients
        c = linalg.solve_banded((1, 1), Ab, B, True, True)
        # Calculate other coefficients
        b = [((y[i+1] - y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3)
             for i in range(0, mdim-1)]
        d = [(c[i+1] - c[i])/(3*h[i]) for i in range(0, mdim - 1)]
        # Store coefficients
        self.__splineCoefficients__ = np.array([y[0:-1], b, c[0:-1], d])

    def __interpolateAkima__(self):
        """Calculate akima spline coefficients that fit the data exactly"""
        # Get x and y values for all supplied points
        x = self.source[:, 0]
        y = self.source[:, 1]
        # Estimate derivatives at each point
        d = [0]*len(x)
        d[0] = (y[1] - y[0])/(x[1] - x[0])
        d[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
        for i in range(1, len(x) - 1):
            w1, w2 = (x[i] - x[i-1]), (x[i+1] - x[i])
            d1, d2 = ((y[i] - y[i-1])/w1), ((y[i+1] - y[i])/w2)
            d[i] = (w1*d2+w2*d1)/(w1+w2)
        # Calculate coefficients for each interval with system already solved
        coeffs = [0]*4*(len(x)-1)
        for i in range(len(x)-1):
            xl, xr = x[i], x[i+1]
            yl, yr = y[i], y[i+1]
            dl, dr = d[i], d[i+1]
            A = np.array([[1, xl, xl**2, xl**3],
                          [1, xr, xr**2, xr**3],
                          [0, 1, 2*xl, 3*xl**2],
                          [0, 1, 2*xr, 3*xr**2]])
            Y = np.array([yl, yr, dl, dr]).T
            coeffs[4*i:4*i+4] = np.linalg.solve(A, Y)
            '''For some reason this doesnt always work!
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
                             2*yl + 2*yr)/(xl-xr)**3'''
        self.__akimaCoefficients__ = coeffs

    # Define all possible algebraic operations
    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (self.getValue(x)/float(other)))
        elif callable(other):
            return Function(lambda x: (self.getValue(x)/other(x)))

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (float(other)/self.getValue(x)))
        elif callable(other):
            return Function(lambda x: (other(x)/self.getValue(x)))

    def __pow__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (self.getValue(x)**float(other)))
        elif callable(other):
            return Function(lambda x: (self.getValue(x)**other(x)))

    def __rpow__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (float(other)**self.getValue(x)))
        elif callable(other):
            return Function(lambda x: (other(x)**self.getValue(x)))

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (self.getValue(x)*float(other)))
        elif callable(other):
            return Function(lambda x: (self.getValue(x)*other(x)))

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (float(other)*self.getValue(x)))
        elif callable(other):
            return Function(lambda x: (other(x)*self.getValue(x)))

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (self.getValue(x)+float(other)))
        elif callable(other):
            return Function(lambda x: (self.getValue(x)+other(x)))

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (float(other)+self.getValue(x)))
        elif callable(other):
            return Function(lambda x: (other(x)+self.getValue(x)))

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (self.getValue(x)-float(other)))
        elif callable(other):
            return Function(lambda x: (self.getValue(x)-other(x)))

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return Function(lambda x: (float(other)-self.getValue(x)))
        elif callable(other):
            return Function(lambda x: (other(x)-self.getValue(x)))

    def integral(self, a, b, numerical=False):
        """ Evaluate a definite integral of a 1-D Function in the interval
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
            unavailabe, calculate numerically anyways.
        
        Returns
        -------
        ans : float
            Evaluated integral.
        """
        if self.__interpolation__ == 'spline' and numerical is False:
            # Integrate using spline coefficients
            xData = self.source[:, 0]
            yData = self.source[:, 1]
            coeffs = self.__splineCoefficients__
            ans = 0
            # Check to see if interval starts before point data
            if a < xData[0]:
                if self.__extrapolation__ == 'constant':
                    ans += yData[0] * (xData[0] - a)
                elif self.__extrapolation__ == 'natural':
                    c = coeffs[:, 0]
                    subB = a - xData[0] # subA = 0
                    ans -= ((c[3]*subB**4)/4 +
                            (c[2]*subB**3/3) +
                            (c[1]*subB**2/2) +
                            c[0]*subB)
                else:
                    # self.__extrapolation__ = 'zero'
                    pass
            # Integrate in subintervals between Xs of given data up to b
            i = 0
            while i < len(xData) - 1 and xData[i] < b:
                if b < xData[i+1]:
                    subB = b - xData[i] # subA = 0
                else:
                    subB = xData[i+1] - xData[i] # subA = 0
                c = coeffs[:, i]
                subB = xData[i+1] - xData[i] # subA = 0
                ans += ((c[3]*subB**4)/4 +
                        (c[2]*subB**3/3) +
                        (c[1]*subB**2/2) +
                        c[0]*subB)
                i += 1
            # Check to see if interval ends after point data
            if b > xData[-1]:
                if self.__extrapolation__ == 'constant':
                    ans += yData[-1] * (b - xData[-1])
                elif self.__extrapolation__ == 'natural':
                    c = coeffs[:, -1]
                    subA = xData[-1] - xData[-2]
                    subB = b - xData[-2]
                    ans -= ((c[3]*subA**4)/4 +
                            (c[2]*subA**3/3) +
                            (c[1]*subA**2/2) +
                            c[0]*subA)
                    ans += ((c[3]*subB**4)/4 +
                            (c[2]*subB**3/3) +
                            (c[1]*subB**2/2) +
                            c[0]*subB)
                else:
                    # self.__extrapolation__ = 'zero'
                    pass
        elif self.__interpolation__ == 'linear' and numerical is False:
            return np.trapz(self.source[:,1], x=self.source[:,0])
        else:
            # Integrate numerically
            ans, error = integrate.quad(self, a, b, epsabs=0.1, limit=10000)
        return ans

    # Not implemented
    def differentiate(self, x, dx=1e-6):
        return (self.getValue(x+dx) - self.getValue(x-dx))/(2*dx)
        # h = (10)**-300
        # z = x + h*1j
        # return self(z).imag/h


class Environment:
    '''Keeps all environment information stored, such as wind and temperature
    conditions, as well as gravity and rail length.'''
    def __init__(self,
                 railLength,
                 gravity=9.8,
                 windData=(0, 0),
                 location=None,
                 date=None):
        """Initialize Environment class, process rail, gravity and
        atmospheric data and store the results.

        Parameters
        ----------
        railLength : scalar
            Length in which the rocket will be attached to the rail, only
            moving along a fixed direction, that is, the line parallel to the
            rail.
        gravity : scalar, optional
            Surface gravitational accelertion. Positive values point the
            acceleration down. Default value is 9.8.
        windData : array of scalar or Function, string, optional
            Wind and atmospheric data input. If array of two scalars or
            Functions is given, the first value is interpreted as constant
            wind speed, while the second value is intepreted as an angle, in
            degrees, specifying direction relative to north (0 deg). If matrix
            (Nx3) is given, the first column must be height data, the second
            wind speed data, and the third must be wind direction data. If
            string is given, it must point to a CSV or a netCDF file, which
            will be  imported. In the case of the CSV file, it must have no
            headers, the first column must contain height values, the second
            column must contain wind speed data, in m/s, and the third column
            must contain wind direction data, in deg. In the case of the
            netCDF file, it should contain wind velocity data and geopotential
            data, both for several pressure levels. Default value is (0, 0),
            corresponding to no wind.
        location : array of float, optional
            Array of length 2, stating (latitude, longitude) of rocket launch
            location. Must be given if wind data source is a netCDF file,
            other wise it is optional.
        date : array, optional
            Array of length 4, stating (year, month, day, hour (UTC)) of
            rocket launch. Must be given if wind data source is a netCDF file,
            other wise it is optional.

        Returns
        -------
        None
        """
        # Define Rail Length
        self.rL = railLength

        # Define gravity
        self.g = gravity

        # Define air density as a function of height.
        # TO-DO: improve density calculation
        # TO-DO: improve speed of sound calculation
        self.density = Function([(0, 1.15), (3100, 0.869)], interpolation='linear')

        # Define Spacetime Location
        if location is None:
            self.lat, self.lon = None, None
        else:
            self.lat, self.lon = location
        if date is None:
            self.date = None
        else:
            self.date = datetime(*date)

        # Import wind data
        # Process string input
        if isinstance(windData, str):
            # Store CSV input
            if windData[-4:] == '.csv':
                self.windDataSource = 'CSV'
                windData = np.loadtxt(windData, delimiter=',')
                h, ws, wd = windData[:, 0], windData[:, 1], windData[:, 2]
                wx, wy = ws * np.sin(wd*np.pi/180), ws * np.cos(wd*np.pi/180)
                self.windSpeed = Function(np.column_stack((h, ws)), 'Height (m)', 'Wind Speed (m/s)')
                self.windDirection = Function(np.column_stack((h, wd)), 'Height (m)', 'Wind Heading (deg)')
                self.windVelocityX = Function(np.column_stack((h, wx)), 'Height (m)', 'Wind Velocity X (m/s)')
                self.windVelocityY = Function(np.column_stack((h, wy)), 'Height (m)', 'Wind Velocity y (m/s)')
            # Store netCDF input
            elif windData[-3:] == '.nc':
                self.windDataSource = 'netCDF'
                windData = netCDF4.Dataset(windData)
                self.processNetCDFFile(windData)
            # Throw error if string input not recognized
            else:
                raise TypeError('Only .csv and .nc file formats '
                                'are supported.')
        # Process array input
        elif isinstance(windData, (tuple, list, np.ndarray)):
            # Process pair array input
            if len(windData) == 2:
                self.windDataSource = 'pair'
                ws, wd = windData[0], windData[1]
                if callable(ws) and callable(wd):
                    wx = Function(lambda h: ws(h) * np.sin(wd(h)*np.pi/180))
                    wy = Function(lambda h: ws(h) * np.cos(wd(h)*np.pi/180))
                elif callable(ws):
                    wx = Function(lambda h: ws(h) * np.sin(wd*np.pi/180))
                    wy = Function(lambda h: ws(h) * np.cos(wd*np.pi/180))
                elif callable(wd):
                    wx = Function(lambda h: ws * np.sin(wd(h)*np.pi/180))
                    wy = Function(lambda h: ws * np.cos(wd(h)*np.pi/180))
                else:
                    wx = ws * np.sin(wd*np.pi/180)
                    wy = ws * np.cos(wd*np.pi/180)
                self.windSpeed = Function(ws, 'Height (m)', 'Wind Speed (m/s)')
                self.windDirection = Function(wd, 'Height (m)', 'Wind Heading (deg)')
                self.windVelocityX = Function(wx, 'Height (m)', 'Wind Velocity X (m/s)')
                self.windVelocityY = Function(wy, 'Height (m)', 'Wind Velocity Y (m/s)')
            # Process dataset matrix input
            elif np.array(windData).shape[1] == 3:
                self.windDataSource = 'matrix'
                windData = np.array(windData)
                h, ws, wd = windData[:, 0], windData[:, 1], windData[:, 2]
                wx, wy = ws * np.sin(wd*np.pi/180), ws * np.cos(wd*np.pi/180)
                self.windSpeed = Function(np.column_stack((h, ws)), 'Height (m)', 'Wind Speed (m/s)')
                self.windDirection = Function(np.column_stack((h, wd)), 'Height (m)', 'Wind Heading (deg)')
                self.windVelocityX = Function(np.column_stack((h, wx)), 'Height (m)', 'Wind Velocity X (m/s)')
                self.windVelocityY = Function(np.column_stack((h, wy)), 'Height (m)', 'Wind Velocity y (m/s)')
            # Throw error if array input not recognized
            else:
                raise TypeError('Only arrays of length 2  and matrices (Nx3) '
                                'are accepeted.')
        return None

    def info(self):
        'Prints out details about the environment and plots'
        # Print gravity details
        print('Gravity Details')
        print('Acceleration of Gravity: ' + str(self.g) + ' m/s2')

        # Print rail details
        print('\nRail Details')
        print('Rail Length: ' + str(self.rL) + ' m')

        # Print spacetime details
        print('\nSpacetime Details')
        print('Date: ', self.date)
        print('Latitude: ', self.lat)
        print('Longitude: ', self.lon)

        # Show plots
        print('\nWind Plots')
        self.windVelocityX()
        self.windVelocityY()
        self.windDirection()
        self.windSpeed()

    def processNetCDFFile(self, windData):
        '''Process netCDF File and store attmospheric data to be used.
        
        Parameters
        ----------
        windData : netCDF4 Dataset
            Dataset containing atmospheric data
        
        Return
        ------
        None
        '''
        # Check if date, lat and lon is know
        if self.date is None:
            raise TypeError('Please specify Date (array-like).')
        if self.lat is None:
            raise TypeError('Please specify Location (lat, lon).')

        # Get data from file
        times = windData.variables['time']
        lons = windData.variables['longitude']
        lats = windData.variables['latitude']
        levels = windData.variables['level']
        geopotentials = windData.variables['z']
        windUs = windData.variables['u']
        windVs = windData.variables['v']

        # Find time index
        timeIndex = netCDF4.date2index(self.date, times, select='nearest')

        # Find longitude index
        lon = self.lon%360
        lonIndex = None
        for i in range(1, len(lons)):
            # Determine if longitude is between lons[i - 1] and lons[i]
            if (lons[i - 1] - lon)*(lon - lons[i]) >= 0:
                lonIndex = i
        if lonIndex is None:
            raise ValueError('Longitude not inside region covered by file.')

        # Find latitude index
        latIndex = None
        for i in range(1, len(lats)):
            # Determine if longitude is between lons[i - 1] and lons[i]
            if (lats[i - 1] - self.lat)*(self.lat - lats[i]) >= 0:
                latIndex = i
        if latIndex is None:
            raise ValueError('Latitude not inside region covered by file.')

        # Determine wind u and v components and height
        windU = []
        for i in range(len(levels)):
            # a = windUs[timeIndex, i, latIndex, lonIndex]
            # b = windUs[timeIndex, i, latIndex - 1, lonIndex]
            # d = windUs[timeIndex, i, latIndex, lonIndex - 1]
            # e = windUs[timeIndex, i, latIndex - 1, lonIndex - 1]
            # c = (((a - b)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + a)
            # f = (((d - e)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + d)
            # r = (((c - f)/(lons[lonIndex] - lons[lonIndex - 1]))*(lon - lons[lonIndex]) + c)
            # windU.append(r)
            windU.append(windUs[timeIndex, i, latIndex, lonIndex])
        windV = []
        for i in range(len(levels)):
            # a = windVs[timeIndex, i, latIndex, lonIndex]
            # b = windVs[timeIndex, i, latIndex - 1, lonIndex]
            # d = windVs[timeIndex, i, latIndex, lonIndex - 1]
            # e = windVs[timeIndex, i, latIndex - 1, lonIndex - 1]
            # c = (((a - b)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + a)
            # f = (((d - e)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + d)
            # r = (((c - f)/(lons[lonIndex] - lons[lonIndex - 1]))*(lon - lons[lonIndex]) + c)
            # windV.append(r)
            windV.append(windVs[timeIndex, i, latIndex, lonIndex])
        height = []
        for i in range(len(levels)):
            # a = geopotentials[timeIndex, i, latIndex, lonIndex]
            # b = geopotentials[timeIndex, i, latIndex - 1, lonIndex]
            # d = geopotentials[timeIndex, i, latIndex, lonIndex - 1]
            # e = geopotentials[timeIndex, i, latIndex - 1, lonIndex - 1]
            # c = (((a - b)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + a)
            # f = (((d - e)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + d)
            # r = (((c - f)/(lons[lonIndex] - lons[lonIndex - 1]))*(lon - lons[lonIndex]) + c)
            # height.append(r/self.g)
            height.append(geopotentials[timeIndex, i, latIndex, lonIndex]/self.g)

        # Convert wind data into functions
        self.windVelocityX = Function(np.array([height, windU]).T,
                                      'Height (m)', 'Wind Velocity X (m/s)', extrapolation='constant')
        self.windVelocityY = Function(np.array([height, windV]).T,
                                      'Height (m)', 'Wind Velocity y (m/s)', extrapolation='constant')
        
        # Store data
        self.lats = lats
        self.lons = lons
        self.lonIndex = lonIndex
        self.latIndex = latIndex
        self.geopotentials = geopotentials
        self.windUs = windUs
        self.windVs = windVs
        self.levels = levels
        self.times = times
        self.height = height
        return None

    def reprocessNetCDFFile(self):
        '''Reprocess netCDF or Grib File'''
        # Restore data
        lats = self.lats
        lons = self.lons
        lonIndex = self.lonIndex
        latIndex = self.latIndex
        geopotentials = self.geopotentials
        windUs = self.windUs
        windVs = self.windVs
        levels = self.levels
        times = self.times

        # Find time index
        timeIndex = netCDF4.date2index(self.date, times, select='nearest')

        # Determine wind u and v components and height
        windU = []
        for i in range(len(levels)):
            # a = windUs[timeIndex, i, latIndex, lonIndex]
            # b = windUs[timeIndex, i, latIndex - 1, lonIndex]
            # d = windUs[timeIndex, i, latIndex, lonIndex - 1]
            # e = windUs[timeIndex, i, latIndex - 1, lonIndex - 1]
            # c = (((a - b)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + a)
            # f = (((d - e)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + d)
            # r = (((c - f)/(lons[lonIndex] - lons[lonIndex - 1]))*(self.lon - lons[lonIndex]) + c)
            # windU.append(r)
            windU.append(windUs[timeIndex, i, latIndex, lonIndex])
        windV = []
        for i in range(len(levels)):
            # a = windVs[timeIndex, i, latIndex, lonIndex]
            # b = windVs[timeIndex, i, latIndex - 1, lonIndex]
            # d = windVs[timeIndex, i, latIndex, lonIndex - 1]
            # e = windVs[timeIndex, i, latIndex - 1, lonIndex - 1]
            # c = (((a - b)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + a)
            # f = (((d - e)/(lats[latIndex] - lats[latIndex - 1]))*(self.lat - lats[latIndex]) + d)
            # r = (((c - f)/(lons[lonIndex] - lons[lonIndex - 1]))*(self.lon - lons[lonIndex]) + c)
            windV.append(windVs[timeIndex, i, latIndex, lonIndex])

        # Convert wind data into functions
        self.windVelocityX = Function(np.array([self.height, windU]).T,
                                      'Height (m)', 'Wind Velocity X (m/s)', extrapolation='constant')
        self.windVelocityY = Function(np.array([self.height, windV]).T,
                                      'Height (m)', 'Wind Velocity y (m/s)', extrapolation='constant')
        return self

    def addWindGust(self, windGust):
        pass

    def setDate(self, date):
        self.date = datetime(*date)
        if self.windDataSource == 'netCDF':
            self.reprocessNetCDFFile()
        return self


class Motor:
    """Class to specify characteriscts and useful operations for a motor."""
    def __init__(self,
                 thrustSource=1000,
                 burnOut=5.78,
                 reshapeThrustCurve=False,
                 interpolationMethod='linear',
                 nozzleRadius=0.0335,
                 throatRadius=0.0114,
                 grainNumber=5,
                 grainSeparation=0.010,
                 grainDensity=1700,
                 grainOuterRadius=0.047,
                 grainInitialInnerRadius=0.016,
                 grainInitialHeight=0.156):
        # Thrust parameters in SI units
        self.interpolate = interpolationMethod
        self.burnOutTime = burnOut
        # Check if thrustSource is csv, eng, function or other
        if isinstance(thrustSource, str):
            # Determine if csv or eng
            if thrustSource[-3:] == 'eng':
                # Import content
                comments, desc, points = self.importEng(thrustSource)
                # Process description and points
                diameter = float(desc[1])/1000
                height = float(desc[2])/1000
                mass = float(desc[4])
                nozzleRadius = diameter/4
                throatRadius = diameter/8
                grainNumber = 1
                grainVolume = height*np.pi*((diameter/2)**2 -(diameter/4)**2)
                grainDensity = mass/grainVolume
                grainOuterRadius = diameter/2
                grainInitialInnerRadius = diameter/4
                grainInitialHeight = height
                thrustSource = points
                self.burnOutTime = points[-1][0]

        # Create thrust function
        self.thrust = Function(thrustSource, 'Time (s)', 'Thrust (N)',
                               self.interpolate, 'zero')
        if callable(thrustSource) or isinstance(thrustSource, (int, float)):
            self.thrust.setDiscrete(0, burnOut, 50, self.interpolate, 'zero')

        # Reshape curve and calculate impulse
        if reshapeThrustCurve:
            self.reshapeThrustCurve(*reshapeThrustCurve)
        else:
            self.evaluateTotalImpulse()

        # Additional thrust information - maximum and avarege
        self.maxThrust = np.amax(self.thrust.source[:, 1])
        maxThrustIndex = np.argmax(self.thrust.source[:, 1])
        self.maxThrustTime = self.thrust.source[maxThrustIndex, 0]
        self.averageThrust = self.totalImpulse/self.burnOutTime

        # Grain and Nozzle parameters in SI units
        self.nozzleRadius = nozzleRadius
        self.throatRadius = throatRadius
        self.grainNumber = grainNumber
        self.grainSeparation = grainSeparation
        self.grainDensity = grainDensity
        self.grainOuterRadius = grainOuterRadius
        self.grainInitialInnerRadius = grainInitialInnerRadius
        self.grainInitialHeight = grainInitialHeight

        # Grain calculated parameters in SI units
        self.grainInitialVolume = (self.grainInitialHeight * np.pi *
                                   (self.grainOuterRadius**2 -
                                    self.grainInitialInnerRadius**2))
        self.grainInitalMass = self.grainDensity*self.grainInitialVolume
        self.propellantInitialMass = self.grainNumber*self.grainInitalMass

        # Important quantities that shall be computed
        self.exhaustVelocity = None
        self.massDot = None
        self.mass = None
        self.grainInnerRadius = None
        self.grainHeight = None
        self.burnArea = None
        self.Kn = None
        self.burnRate = None
        self.inertiaI = None
        self.inertiaIDot = None
        self.inertiaZ = None
        self.inertiaDot = None

        # Calculate important quantities
        self.evaluateExhaustVelocity()
        self.evaluateMassDot()
        self.evaluateMass()
        self.evaluateGeometry()
        self.evaluateInertia()

    def refresh(self):
        'Re-evaluate quantities'
        # Thrust parameters in SI units
        self.maxThrust = np.amax(self.thrust.source[:, 1])
        maxThrustIndex = np.argmax(self.thrust.source[:, 1])
        self.maxThrustTime = self.thrust.source[maxThrustIndex, 0]
        self.averageThrust = self.evaluateTotalImpulse()/self.burnOutTime

        # Grain calculated parameters in SI units
        self.grainInitialVolume = (self.grainInitialHeight * np.pi *
                                   (self.grainOuterRadius**2 -
                                    self.grainInitialInnerRadius**2))
        self.grainInitalMass = self.grainDensity*self.grainInitialVolume
        self.propellantInitialMass = self.grainNumber*self.grainInitalMass

        # Calculate important quantities
        self.evaluateExhaustVelocity()
        self.evaluateMassDot()
        self.evaluateMass()
        self.evaluateGeometry()
        self.evaluateInertia()

    def reshapeThrustCurve(self, burnTime, totalImpulse,
                           oldTotalImpulse=None, startAtZero=True):
        '''Transforms self.thrust into a curve with a different burn time and
        different total impulse'''
        timeArray = self.thrust.source[:, 0]
        thrustArray = self.thrust.source[:, 1]
        # Move start to time = 0
        if startAtZero and timeArray[0] != 0:
            timeArray = timeArray - timeArray[0]
        # Reshape time - set burn time to burnTime
        self.thrust.source[:, 0] = (burnTime/timeArray[-1])*timeArray
        self.burnOutTime = burnTime
        self.thrust.setInterpolation(self.interpolate)
        # Reshape thrust - set total impulse
        if oldTotalImpulse is None:
            oldTotalImpulse = self.evaluateTotalImpulse()
        self.thrust.source[:, 1] = (totalImpulse/oldTotalImpulse)*thrustArray
        self.thrust.setInterpolation(self.interpolate)
        self.totalImpulse = totalImpulse
        return self

    def info(self):
        'Prints out details about nozzle, grain, motor and plots'
        # Print nozzle details
        print('Nozzle Details')
        print('Nozzle Radius: ' + str(self.nozzleRadius) + ' m')
        print('Nozzle Throat Radius: ' + str(self.throatRadius) + ' m')

        # Print grain details
        print('\nGrain Details')
        print('Number of Grains: ' + str(self.grainNumber))
        print('Grain Spacing: ' + str(self.grainSeparation) + ' m')
        print('Grain Density: ' + str(self.grainDensity) + ' kg/m3')
        print('Grain Outer Radius: ' + str(self.grainOuterRadius) + ' m')
        print('Grain Inner Radius: ' + str(self.grainInitialInnerRadius) +
              ' m')
        print('Grain Height: ' + str(self.grainInitialHeight) + ' m')
        print('Grain Volume: ' + "{:.3f}".format(self.grainInitialVolume) +
              ' m3')
        print('Grain Mass: ' + "{:.3f}".format(self.grainInitalMass) + ' kg')

        # Print motor details
        print('\nMotor Details')
        print('Total Burning Time: ' + str(self.burnOutTime) + ' s')
        print('Total Propellant Mass: ' +
              "{:.3f}".format(self.propellantInitialMass) + ' kg')
        print('Propellant Exhaust Velocity: ' +
              "{:.3f}".format(self.exhaustVelocity) + ' m/s')
        print('Average Thrust: ' + "{:.3f}".format(self.averageThrust) + ' N')
        print('Maximum Thrust: ' + str(self.maxThrust) + ' N at ' +
              str(self.maxThrustTime) + ' s after ignition.')
        print('Total Impulse: ' + "{:.3f}".format(self.totalImpulse) + ' Ns')

        # Show plots
        print('\nPlots')
        self.thrust()
        self.mass()
        self.massDot()
        self.grainInnerRadius()
        self.grainHeight()
        self.burnRate()
        self.burnArea()
        self.Kn()
        self.inertiaI()
        self.inertiaIDot()
        self.inertiaZ()
        self.inertiaZDot()

    def evaluateTotalImpulse(self):
        'Calculates, saves and returns motor total impulse in SI units.'
        self.totalImpulse = self.thrust.integral(0, self.burnOutTime)
        return self.totalImpulse

    def evaluateExhaustVelocity(self):
        'Calculates, saves and returns exaust velocity in SI units.'
        if self.totalImpulse is None:
            self.evaluateTotalImpulse()
        self.exhaustVelocity = self.totalImpulse/self.propellantInitialMass
        return self.exhaustVelocity

    def evaluateMassDot(self):
        '''Calculates, saves and returns mass derivative in time as a Function
        of time.'''
        if self.exhaustVelocity is None:
            self.evaluateExhaustVelocity()
        self.massDot = (-1)*self.thrust/self.exhaustVelocity
        self.massDot.setDiscrete(0, self.burnOutTime, 500)
        self.massDot.setExtrapolation('zero')
        self.massDot.setInputs('Tims (s)')
        self.massDot.setOutputs('Mass Dot (kg/s)')
        return self.massDot

    def evaluateMass(self):
        'Calculates, saves and returns total grain mass as a Function of time.'
        if self.massDot is None:
            self.evaluateMassDot()
        # Define initial conditions for integration
        y0 = self.propellantInitialMass
        # Define time mesh
        t = np.linspace(0, self.burnOutTime, 200)
        # Solve the system of differential equations
        sol = integrate.odeint(lambda y, t: self.massDot.getValueOpt(t), y0, t)
        # Write down function for propellant mass
        self.mass = Function(np.concatenate(([t], [sol[:, 0]])).
                             transpose().tolist(), 'Time (s)',
                             'Propellant Total Mass (kg)', 'spline',
                             'constant')
        return self.mass

    def evaluateGeometry(self):
        '''Calculates, saves and returns motor grain geometry parameters as a
           Function of time.'''
        # Define initial conditions for integration
        y0 = [self.grainInitialInnerRadius, self.grainInitialHeight]
        # Define time mesh
        t = np.linspace(0, self.burnOutTime, 200)
        # Solve the system of differential equations
        sol = integrate.odeint(self.__burnRate, y0, t)
        # Write down functions for innerRadius and height
        self.grainInnerRadius = Function(np.concatenate(([t], [sol[:, 0]])).
                                         transpose().tolist(), 'Time (s)',
                                         'Grain Inner Radius (m)',
                                         'spline', 'constant')
        self.grainHeight = Function(np.concatenate(([t], [sol[:, 1]])).
                                    transpose().tolist(),
                                    'Time (s)', 'Grain Height (m)',
                                    'spline', 'constant')

        # Create functions describing burn rate, Kn and burn area
        # Burn Area
        self.burnArea = 2*np.pi*(self.grainOuterRadius**2 -
                                 self.grainInnerRadius**2 +
                                 self.grainInnerRadius *
                                 self.grainHeight)*self.grainNumber
        self.burnArea.setDiscrete(0, self.burnOutTime, 200)
        self.burnArea.setInputs('Time (s)')
        self.burnArea.setOutputs('Burn Area (m2)')

        # Kn
        throatArea = np.pi*(self.throatRadius)**2
        KnSource = (np.concatenate(([self.grainInnerRadius.source[:, 1]],
                                    [self.burnArea.source[:, 1]/throatArea]
                                    )).transpose()).tolist()
        self.Kn = Function(KnSource, 'Grain Inner Radius (m)',
                           'Kn (m2/m2)', 'linear', 'constant')
        # Burn Rate
        self.burnRate = (-1)*self.massDot/(self.burnArea*self.grainDensity)
        self.burnRate.setDiscrete(0, self.burnOutTime, 500)
        self.burnRate.setInputs('Time (s)')
        self.burnRate.setOutputs('Burn Rate (m/s)')

        return

    def evaluateInertia(self):
        '''Calculates, saves and returns motor inertia parameters as a
           Function of time.'''
        # Inertia I
        # Calculate inertia for each grain
        grainMass = self.mass/self.grainNumber
        grainMassDot = self.massDot/self.grainNumber
        grainNumber = self.grainNumber
        grainInertiaI = grainMass*((1/4)*(self.grainOuterRadius**2 +
                                          self.grainInnerRadius**2) +
                                   (1/12)*self.grainHeight**2)

        # Calculate each grain's distance d to propellant center of mass
        initialValue = (grainNumber - 1)/2
        d = np.linspace(-initialValue, initialValue, self.grainNumber)
        d = d*(self.grainInitialHeight + self.grainSeparation)

        # Calculate inertia for all grains
        self.inertiaI = grainNumber*(grainInertiaI) + grainMass*np.sum(d**2)
        self.inertiaI.setDiscrete(0, self.burnOutTime, 200)
        self.inertiaI.setInputs('Time (s)')
        self.inertiaI.setOutputs('Propellant Inertia I (kg*m2)')

        # Inertia I Dot
        # Calculate each grain's inertia I dot
        grainInertiaIDot = (grainMassDot*((1/4)*(self.grainOuterRadius**2 +
                                                 self.grainInnerRadius**2) +
                                          (1/12)*self.grainHeight**2) +
                            grainMass*((1/2)*self.grainInnerRadius -
                                       (1/3)*self.grainHeight)*self.burnRate)

        # Calculate inertia I dot for all grains
        self.inertiaIDot = (grainNumber*(grainInertiaIDot) +
                            grainMassDot*np.sum(d**2))
        self.inertiaIDot.setDiscrete(0, self.burnOutTime, 500)
        self.inertiaIDot.setInputs('Time (s)')
        self.inertiaIDot.setOutputs('Propellant Inertia I Dot (kg*m2/s)')

        # Inertia Z
        self.inertiaZ = (1/2.0)*self.mass*(self.grainOuterRadius**2 +
                                           self.grainInnerRadius**2)
        self.inertiaZ.setDiscrete(0, self.burnOutTime, 200)
        self.inertiaZ.setInputs('Time (s)')
        self.inertiaZ.setOutputs('Propellant Inertia Z (kg*m2)')

        # Inertia Z Dot
        self.inertiaZDot = ((1/2.0)*(self.massDot*self.grainOuterRadius**2) +
                            (1/2.0)*(self.massDot*self.grainInnerRadius**2) +
                            self.mass*self.grainInnerRadius*self.burnRate)
        self.inertiaZDot.setDiscrete(0, self.burnOutTime, 500)
        self.inertiaZDot.setInputs('Time (s)')
        self.inertiaZDot.setOutputs('Propellant Inertia Z Dot (kg*m2/s)')

        return

    def __burnRate(self, y, t):
        grainMassDot = self.massDot(t)/self.grainNumber
        density = self.grainDensity
        rO = self.grainOuterRadius
        rI, h = y
        rIDot = -0.5*grainMassDot/(density*np.pi*(rO**2-rI**2+rI*h))
        hDot = 1.0*grainMassDot/(density*np.pi*(rO**2-rI**2+rI*h))
        return [rIDot, hDot]

    def importEng(self, fileName):
        """ Read content from .eng file named fileName and process it,
        in order to return the coments, description and data points.

        Parameters
        ----------
        fileName : string
            Name of the .eng file. E.g. 'test.eng'

        Returns
        -------
        comments : list
            All comments in the .eng file, separeted by line in a list. Each
            line is an entry of the list.
        description: list
            Description of the motor. All attributes are returned separeted in
            a list. E.g. "F32 24 124 5-10-15 .0377 .0695 RV\n" is return as
            ['F32', '24', '124', '5-10-15', '.0377', '.0695', 'RV\n']
        dataPoints: list
            List of all data points in file. Each data point is an entry in
            the returned list and written as a list of two entries.
        """
        import re
        comments = []
        description = []
        dataPoints = [[0, 0]]
        with open(fileName) as file:
            for line in file:
                if line[0] == ';':
                    comments.append(line)
                else:
                    if description == []:
                        description = line.split(' ')
                    else:
                        time, thrust = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        dataPoints.append([float(time), float(thrust)])
        return comments, description, dataPoints

    def exportEng(self, fileName, motorName='Mandioca'):
        'Exports motor to M file'
        file = open(fileName, 'w')
        file.write(motorName + ' {:3.1f} {:3.1f} 0 {:2.3} {:2.3} PJ \n'.format(2000*self.grainOuterRadius, 1000*self.grainNumber*(self.grainInitialHeight+self.grainSeparation), self.propellantInitialMass, self.propellantInitialMass))
        for item in self.thrust.source[:-1, :]:
            time = item[0]
            thrust = item[1]
            file.write('{:.4f} {:.3f}\n'.format(time, thrust))
        file.write('{:.4f} {:.3f}\n'.format(self.thrust.source[-1, 0], 0))
        file.close()


class Rocket:
    """Keeps all rocket information and has a method to initiate flight.
    """
    def __init__(self,
                 motor,
                 mass=26.43,
                 inertiaI=14.68,
                 inertiaZ=0.09,
                 radius=0.075,
                 distanceRocketNoseTip=2.061724,
                 distanceRocketFins=1.002776,
                 distanceRocketTail=1.210075,
                 distanceRocketNozzle=1.279915,
                 distanceRocketPropellant=0.729776,
                 offCenter=0.000,
                 noseLength=0.675,
                 finNumber=4,
                 finSpan=0.16,
                 finRootChord=0.185,
                 finTipChord=0.050,
                 tailLength=0.065,
                 tailRadius=0.0425,
                 powerOffDrag='powerOffDragCurve.csv',
                 powerOnDrag='powerOnDragCurve.csv',
                 drogueArea=1,
                 drogueCd=0.9,
                 drogueLag=1,
                 mainArea=7,
                 mainCd=1.43,
                 mainAlt=200):
        # Parachute data
        self.drogueArea = drogueArea
        self.drogueCd = drogueCd
        self.drogueLag = drogueLag
        self.mainArea = mainArea
        self.mainCd = mainCd
        self.mainAlt = mainAlt

        # Define motor to be used
        self.motor = motor

        # Define rocket inertia attributes in SI units
        self.mass = mass
        self.inertiaI = inertiaI
        self.inertiaZ = inertiaZ
        self.centerOfMass = (distanceRocketPropellant*motor.mass /
                             (mass + motor.mass))

        # Define rocket geometrical parameters in SI units
        self.radius = radius
        self.area = np.pi*self.radius**2
        # Center of mass distance to points of interest
        self.distanceRocketNoseTip = distanceRocketNoseTip
        self.distanceRocketFins = distanceRocketFins
        self.distanceRocketTail = distanceRocketTail
        self.distanceRocketNozzle = distanceRocketNozzle
        self.distanceRocketPropellant = distanceRocketPropellant
        self.offCenter = offCenter
        # Nose, fins and tail dimensions
        self.noseLength = noseLength
        self.finNumber = finNumber
        self.finSpan = finSpan
        self.finRootChord = finRootChord
        self.finTipChord = finTipChord
        self.finMidChordLength = np.sqrt((finRootChord/2 - finTipChord/2)**2 +
                                         finSpan**2)
        self.finArea = finSpan*(finRootChord + finTipChord)/2
        self.tailLength = tailLength
        self.tailRadius = tailRadius

        # Calculate centers of pressure distance to rocket center of gravity
        Yr = finRootChord + finTipChord
        Cr = finRootChord
        Ct = finTipChord
        Lt = tailLength
        r = radius/tailRadius
        self.noseCenterOfPressure = (distanceRocketNoseTip - 0.5*noseLength)
        self.finCenterOfPressure = -(distanceRocketFins +
                                     ((Cr - Ct)/3)*((Cr + 2*Ct)/(Cr + Ct)) +
                                     (1/6)*(Cr + Ct - Cr*Ct/(Cr + Ct)))
        self.tailCenterOfPressure = -(distanceRocketTail +
                                      (Lt/3)*(1 + (1-r)/(1 - r**2)))

        # Caulculate lift coefficient derivative
        self.noseLiftCoeffDer = 2.0
        n = finNumber
        s = finSpan
        d = 2*radius
        Lf = self.finMidChordLength
        self.finLiftCoeffDer = (4*n*(s/d)**2)/(1 + np.sqrt(1 + (2*Lf/Yr)**2))
        self.finLiftCoeffDer *= (1 + radius/(s + radius))
        self.tailLiftCoeffDer = -2*(1 - r**(-2))
        self.totalLiftCoeffDer = (self.noseLiftCoeffDer +
                                  self.finLiftCoeffDer +
                                  self.tailLiftCoeffDer)

        # Calculate Static Margin
        self.cpPosition = ((self.finLiftCoeffDer*self.finCenterOfPressure +
                            self.noseLiftCoeffDer*self.noseCenterOfPressure +
                            self.tailLiftCoeffDer*self.tailCenterOfPressure) /
                           (self.totalLiftCoeffDer))
        self.staticMargin = (abs(self.cpPosition) - self.centerOfMass(0))/d

        # Important varying quantities to be calculated
        self.reducedMass = None
        self.totalMass = None

        # Calculate important quantities
        if self.motor is None:
            print('''Please associate this rocket with a motor and call
                     rocket.refresh()!''')
        else:
            self.evaluateReducedMass()
            self.evaluateTotalMass()

        # Define aerodynamic coefficients
        self.powerOffDrag = Function(powerOffDrag, 'Mach Number',
                                     'Drag Coefficient with Power Off',
                                     'spline', 'constant')
        self.powerOnDrag = Function(powerOnDrag, 'Mach Number',
                                    'Drag Coefficient with Power On',
                                    'spline', 'constant')

    def info(self):
        'Prints out details about the rocket and plots'
        # Print inertia details
        print('Inertia Details')
        print('Rocket Mass: ' + str(self.mass) + ' kg (No Propellant)')
        print('Inertia I: ' + str(self.inertiaI) + ' kg*m2')
        print('Inertia Z: ' + str(self.inertiaZ) + ' kg*m2')

        # Print rocket geometrical parameters
        print('\nGeometrical Parameters')
        print('Rocket Radius: ' + str(self.radius) + ' m')
        print('Rocket Frontal Area: ' + "{:.6f}".format(self.area) + ' m2')
        print('\nRocket Distances')
        print('Rocket Center of Mass - Nose Tip Distance: ' +
              str(self.distanceRocketNoseTip) + ' m')
        print('Rocket Center of Mass - Fins Start Distance: ' +
              str(self.distanceRocketFins) + ' m')
        print('Rocket Center of Mass - Tail Start Distance: ' +
              str(self.distanceRocketTail) + ' m')
        print('Rocket Center of Mass - Nozzle Exit Distance: ' +
              str(self.distanceRocketNozzle) + ' m')
        print('Rocket Center of Mass - Propellant Center of Mass Distance: ' +
              str(self.distanceRocketPropellant) + ' m')
        print('\nAerodynamic Coponents Parameters')
        print('Nose Length: ' + str(self.noseLength) + ' m')
        print('Number of Fins: ' + str(self.finNumber))
        print('Fin Span: ' + str(self.finSpan) + ' m')
        print('Fin Root Chord: ' + str(self.finRootChord) + ' m')
        print('Fin Tip Chord: ' + str(self.finTipChord) + ' m')
        print('Fin Mid Chord: ' + "{:.5f}".format(self.finMidChordLength)+' m')
        print('Fin Area: ' + str(self.finArea) + ' m2')
        print('Tail Length: ' + str(self.tailLength) + ' m')
        print('Tail Radius: ' + str(self.tailRadius) + ' m')

        # Print rocket aerodynamics quantities
        print('\nAerodynamics Lift Coefficient Derivatives')
        print('Nose Cone Lift Coefficient Derivative: ' +
              "{:.3f}".format(self.noseLiftCoeffDer))
        print('Fins Lift Coefficient Derivative: ' +
              "{:.3f}".format(self.finLiftCoeffDer))
        print('Tail Lift Coefficient Derivative: ' +
              "{:.3f}".format(self.tailLiftCoeffDer))
        print('Total Lift Coefficient Derivative: ' +
              "{:.3f}".format(self.totalLiftCoeffDer))
        print('\nAerodynamics Center of Pressure')
        print('Nose Center of Pressure: ' +
              "{:.3f}".format(self.noseCenterOfPressure) + ' m')
        print('Fins Center of Pressure: ' +
              "{:.3f}".format(self.finCenterOfPressure) + ' m')
        print('Tail Center of Pressure: ' +
              "{:.3f}".format(self.tailCenterOfPressure) + ' m')
        print('Static Center of Pressure Position: ' +
              "{:.3f}".format(self.cpPosition) + ' m')
        print('Static Margin: ' +
              "{:.3f}".format(self.staticMargin) + ' c')

        # Show plots
        print('\nMass Plots')
        self.totalMass()
        self.reducedMass()
        print('\nAerodynamics Plots')
        self.powerOnDrag()
        self.powerOffDrag()

    def refresh(self):
        'Recalculates important quantities'
        if self.motor is None:
            print('Please associate this rocket with a motor!')
        else:
            self.evaluateReducedMass()
            self.evaluateTotalMass()

    def evaluateReducedMass(self):
        'Calculates and returns rocket\'s reduced mass as a Function of time.'
        if self.motor is None:
            print('Please associate this rocket with a motor!')
            return False
        motorMass = self.motor.mass
        mass = self.mass
        self.reducedMass = motorMass*mass/(motorMass+mass)
        self.reducedMass.setDiscrete(0, self.motor.burnOutTime, 200)
        self.reducedMass.setInputs('Time (s)')
        self.reducedMass.setOutputs('Reduced Mass (kg)')
        return self.reducedMass

    def evaluateTotalMass(self):
        'Calculates and returns rocket\'s total mass as a Function of time.'
        if self.motor is None:
            print('Please associate this rocket with a motor!')
            return False
        self.totalMass = self.mass + self.motor.mass
        self.totalMass.setDiscrete(0, self.motor.burnOutTime, 200)
        self.totalMass.setInputs('Time (s)')
        self.totalMass.setOutputs('Total Mass (Rocket + Propellant) (kg)')
        return self.totalMass

    def setMotor(self, motor):
        'Set motor and refresh important quantities'
        if isinstance(motor, Motor):
            self.motor = motor
            self.refresh()
        else:
            print('Please specify a valid motor!')

    def addFins(self, numberOfFins=4, cl=2*np.pi, cpr=1, cpz=1,
                gammas=[0, 0, 0, 0], angularPositions=None):
        "Hey! I will document this function later"
        self.aerodynamicSurfaces = []
        pi = np.pi
        # Calculate angular postions if not given
        if angularPositions is None:
            angularPositions = np.array(range(numberOfFins))*2*pi/numberOfFins
        else:
            angularPositions = np.array(angularPositions)*pi/180
        # Convert gammas to degree
        if isinstance(gammas, (int, float)):
            gammas = [(pi/180)*gammas for i in range(numberOfFins)]
        else:
            gammas = [(pi/180)*gamma for gamma in gammas]
        for i in range(numberOfFins):
            # Get angular position and inclination for current fin
            angularPosition = angularPositions[i]
            gamma = gammas[i]
            # Calculate position vector
            cpx = cpr*np.cos(angularPosition)
            cpy = cpr*np.sin(angularPosition)
            positionVector = np.array([cpx, cpy, cpz])
            # Calculate chord vector
            auxVector = np.array([cpy, -cpx, 0])/(cpr)
            chordVector = (np.cos(gamma)*np.array([0, 0, 1]) -
                           np.sin(gamma)*auxVector)
            self.aerodynamicSurfaces.append([positionVector, chordVector])
        return None


class Flight:
    '''Keeps all flight information and has a method to simulate flight.
    '''
    def __init__(self, rocket, environment,
                 inclination=80, heading=140, flightPhases=-1, timeStep=0.1,
                 initialState=None):
        # Save rocket and environment
        self.env = environment
        self.rocket = rocket
        # Define initial conditions
        xInit, yInit, zInit = 0, 0, 0
        vxInit, vyInit, vzInit = 0, 0, 0
        w1Init, w2Init, w3Init = 0, 0, 0
        # Define launch heading and angle - initial attitude
        launchAngle = (90 - inclination)*(np.pi/180)
        rotAngle = (90 - heading)*(np.pi/180)
        rotAxis = np.array([-np.sin(rotAngle), np.cos(rotAngle), 0])
        e0Init = np.cos(launchAngle/2)
        e1Init = np.sin(launchAngle/2) * rotAxis[0]
        e2Init = np.sin(launchAngle/2) * rotAxis[1]
        e3Init = 0
        self.uInitial = [xInit, yInit, zInit, vxInit, vyInit, vzInit,
                         e0Init, e1Init, e2Init, e3Init,
                         w1Init, w2Init, w3Init]
        if initialState is not None:
            self.uInitial = initialState
        # Save initial conditions
        self.step = timeStep
        self.maxTime = 600
        self.tInitial = 0.00
        
        # Initialize solution
        self.solution = []
        self.solution.append([self.tInitial, *self.uInitial])
        self.outOfRailTime = 0
        self.outOfRailState = 0
        self.outOfRailVelocity = 0
        self.apogeeState = 0
        self.apogeeTime = 0
        self.apogeeX = 0
        self.apogeeY = 0
        self.apogee = 0
        self.drogueOpeningState = 0
        self.drogueOpeningTime = 0
        self.drogueOpeningVelocity = 0
        self.mainOpeningState = 0
        self.mainOpeningTime = 0
        self.drogueX = 0
        self.drogueY = 0
        self.drogueZ = 0
        self.xImpact = 0
        self.yImpact = 0
        self.impactVelocity = 0
        # Ignition to out of rail flight phase
        self.solver = integrate.ode(self.__uDotRailOpt)
        self.solver.set_integrator('vode', method='adams')
        self.solver.set_initial_value(self.uInitial, self.tInitial)
        while self.solver.successful() and (self.solver.y[0]**2 +
                                            self.solver.y[1]**2 +
                                            self.solver.y[2]**2 <=
                                            self.env.rL**2):
            self.solver.integrate(self.solver.t + self.step)
            self.solution.append([self.solver.t, *self.solver.y])
        self.outOfRailTime = self.solver.t
        self.outOfRailState = self.solver.y
        self.outOfRailVelocity = (self.solver.y[3]**2 +
                                  self.solver.y[4]**2 +
                                  self.solver.y[5]**2)**(0.5)
        if flightPhases == 0:
            self.tFinal = self.solver.t
            return None
        # Out of rail to apogee flight phase
        self.solver = integrate.ode(self.__uDotOpt)
        self.solver.set_integrator('vode', method='adams')
        self.solver.set_initial_value(self.outOfRailState, self.outOfRailTime)
        while self.solver.successful() and (self.solver.y[5] >= 0 and
                                            self.solver.t < self.maxTime):
            self.solver.integrate(self.solver.t + self.step)
            self.solution.append([self.solver.t, *self.solver.y])
        self.apogeeState = self.solver.y
        self.apogeeTime = self.solver.t
        self.apogeeX = self.solver.y[0]
        self.apogeeY = self.solver.y[1]
        self.apogee = self.solver.y[2]
        if flightPhases == 1:
            self.tFinal = self.solver.t
            return None
        # Determine if rocket has drogue parachute
        if self.rocket.drogueCd is False:
            # Solve while rocket is descending without drogue parachute
            while (self.solver.successful() and
                   self.solver.y[2] >= self.rocket.mainAlt and
                   self.solver.t < self.maxTime):
                self.solver.integrate(self.solver.t + self.step)
                self.solution.append([self.solver.t, *self.solver.y])
            if flightPhases == 2:
                self.tFinal = self.solver.t
                return None
        else:
            # Solve during drogue delay
            while (self.solver.successful() and
                   self.solver.t - self.apogeeTime < self.rocket.drogueLag and
                   self.solver.t < self.maxTime):
                self.solver.integrate(self.solver.t + self.step)
                self.solution.append([self.solver.t, *self.solver.y])
            self.drogueOpeningState = self.solver.y
            self.drogueOpeningTime = self.solver.t
            self.drogueOpeningVelocity = (self.solver.y[3]**2 +
                                          self.solver.y[4]**2 +
                                          self.solver.y[5]**2)**0.5
            if flightPhases == 1.5:
                self.tFinal = self.solver.t
                return None
            # Solve while rocket is descending with drogue parachute
            self.solver = integrate.ode(self.__uDotDrogue)
            self.solver.set_integrator('vode', method='adams')
            self.solver.set_initial_value(self.drogueOpeningState, self.drogueOpeningTime)
            while (self.solver.successful() and
                   self.solver.y[2] >= self.rocket.mainAlt and
                   self.solver.t < self.maxTime):
                self.solver.integrate(self.solver.t + self.step)
                self.solution.append([self.solver.t, *self.solver.y])
            if flightPhases == 2:
                self.tFinal = self.solver.t
                return None
        # Determine if rocket has main parachute
        if self.rocket.mainCd is False:
            # Solve while rocket is descending without main parachute
            while (self.solver.successful() and
                   self.solver.y[2] >= 0 and
                   self.solver.t < self.maxTime):
                self.solver.integrate(self.solver.t + self.step)
                self.solution.append([self.solver.t, *self.solver.y])
        else:
            # Solve while rocket is descending with main parachute
            self.mainOpeningState = self.solver.y
            self.mainOpeningTime = self.solver.t
            self.drogueX = self.solver.y[0]
            self.drogueY = self.solver.y[1]
            self.drogueZ = self.solver.y[2]
            self.solver = integrate.ode(self.__uDotMain)
            self.solver.set_integrator('vode', method='adams')
            self.solver.set_initial_value(self.mainOpeningState, self.mainOpeningTime)
            while (self.solver.successful() and
                   self.solver.y[2] >= 0 and
                   self.solver.t < self.maxTime):
                self.solver.integrate(self.solver.t + self.step)
                self.solution.append([self.solver.t, *self.solver.y])
        # Impact
        self.xImpact = self.solver.y[0]
        self.yImpact = self.solver.y[1]
        self.impactVelocity = (self.solver.y[3]**2 + self.solver.y[4]**2 + self.solver.y[5]**2)**0.5
        self.tFinal = self.solver.t

    def postProcess(self):
        'Post-processing of integration  results'
        # Transform solution into Functions
        sol = np.array(self.solution)
        self.x = Function(sol[:, [0, 1]], 'Time (s)', 'X (m)', 'spline', extrapolation="natural")
        self.y = Function(sol[:, [0, 2]], 'Time (s)', 'Y (m)', 'spline', extrapolation="natural")
        self.z = Function(sol[:, [0, 3]], 'Time (s)', 'Z (m)', 'spline', extrapolation="natural")
        self.vx = Function(sol[:, [0, 4]], 'Time (s)', 'Vx (m/s)', 'spline', extrapolation="natural")
        self.vy = Function(sol[:, [0, 5]], 'Time (s)', 'Vy (m/s)', 'spline', extrapolation="natural")
        self.vz = Function(sol[:, [0, 6]], 'Time (s)', 'Vz (m/s)', 'spline', extrapolation="natural")
        self.e0 = Function(sol[:, [0, 7]], 'Time (s)', 'e0', 'spline', extrapolation="natural")
        self.e1 = Function(sol[:, [0, 8]], 'Time (s)', 'e1', 'spline', extrapolation="natural")
        self.e2 = Function(sol[:, [0, 9]], 'Time (s)', 'e2', 'spline', extrapolation="natural")
        self.e3 = Function(sol[:, [0, 10]], 'Time (s)', 'e3', 'spline', extrapolation="natural")
        self.w1 = Function(sol[:, [0, 11]], 'Time (s)', 'ω1 (rad/s)', 'spline', extrapolation="natural")
        self.w2 = Function(sol[:, [0, 12]], 'Time (s)', 'ω2 (rad/s)', 'spline', extrapolation="natural")
        self.w3 = Function(sol[:, [0, 13]], 'Time (s)', 'ω3 (rad/s)', 'spline', extrapolation="natural")
        # Calculate aerodynamic forces and accelerations
        self.cpPosition1, self.cpPosition2, self.cpPosition3 = [], [], []
        self.staticMargin = []
        self.noseStreamSpeed, self.noseAttackAngle, self.noseLift = [], [], []
        self.finStreamSpeed, self.finAttackAngle, self.finLift = [], [], []
        self.tailStreamSpeed, self.tailAttackAngle, self.tailLift = [], [], []
        self.attackAngle, self.freestreamSpeed = [], []
        self.R1, self.R2, self.R3 = [], [], []
        self.M1, self.M2, self.M3 = [], [], []
        self.ax, self.ay, self.az = [], [], []
        self.alp1, self.alp2, self.alp3 = [], [], []
        self.streamVelX, self.streamVelY, self.streamVelZ = [], [], []
        for step in self.solution:
            if step[0] <= self.outOfRailTime:
                self.__uDotRailOpt(step[0], step[1:], verbose=True)
            elif (step[0] <= self.apogeeTime or
                  step[0] <= self.drogueOpeningTime or
                  (step[0] <= self.mainOpeningTime and
                   self.rocket.drogueCd == False) or
                  (step[0] <= self.tFinal and self.rocket.drogueCd == False and
                   self.rocket.mainCd == False)):
                self.__uDotOpt(step[0], step[1:], verbose=True)
            elif  step[0] <= self.mainOpeningTime:
                self.__uDotDrogue(step[0], step[1:], verbose=True)
            else:
                self.__uDotMain(step[0], step[1:], verbose=True)
        self.noseStreamSpeed = Function(self.noseStreamSpeed, 'Time (s)',
                                        'Nose Stream Speed (m/s)', 'spline')
        self.noseAttackAngle = Function(self.noseAttackAngle, 'Time (s)',
                                        'Nose Attack Angle', 'spline')
        self.noseLift = Function(self.noseLift, 'Time (s)',
                                 'Nose Lift Force (N)', 'spline')
        self.finStreamSpeed = Function(self.finStreamSpeed, 'Time (s)',
                                       'Fin Stream Speed (m/s)', 'spline')
        self.finAttackAngle = Function(self.finAttackAngle, 'Time (s)',
                                       'Fin Attack Angle', 'spline')
        self.finLift = Function(self.finLift, 'Time (s)',
                                'Fin Lift Force (N)', 'spline')
        self.tailStreamSpeed = Function(self.tailStreamSpeed, 'Time (s)',
                                        'Tail Stream Speed (m/s)', 'linear')
        self.tailAttackAngle = Function(self.tailAttackAngle, 'Time (s)',
                                        'Tail Attack Angle', 'linear')
        self.tailLift = Function(self.tailLift, 'Time (s)',
                                 'Tail Lift Force (N)', 'linear')
        self.cpPosition1 = Function(self.cpPosition1, 'Time (s)',
                                    'CP Position in X (m)', 'linear')
        self.cpPosition2 = Function(self.cpPosition2, 'Time (s)',
                                    'CP Position in Y (m)', 'linear')
        self.cpPosition3 = Function(self.cpPosition3, 'Time (s)',
                                    'CP Position in Z (m)', 'linear')
        self.staticMargin = Function(self.staticMargin, 'Time (s)',
                                     'Static Margin (c)', 'linear')
        self.attackAngle = Function(self.attackAngle, 'Time (s)',
                                    'Angle of Attack', 'linear')
        self.freestreamSpeed = Function(self.freestreamSpeed, 'Time (s)',
                                        'Freestream Speed (m/s)', 'linear')
        self.streamVelX = Function(self.streamVelX, 'Time (s)',
                                   'Freestream VelX (m/s)', 'linear')
        self.streamVelY = Function(self.streamVelY, 'Time (s)',
                                   'Freestream VelY (m/s)', 'linear')
        self.streamVelZ = Function(self.streamVelZ, 'Time (s)',
                                   'Freestream VelZ (m/s)', 'linear')
        self.R1 = Function(self.R1, 'Time (s)', 'R1 (N)', 'linear')
        self.R2 = Function(self.R2, 'Time (s)', 'R2 (N)', 'linear')
        self.R3 = Function(self.R3, 'Time (s)', 'R3 (N)', 'linear')
        self.M1 = Function(self.M1, 'Time (s)', 'M1 (Nm)', 'linear')
        self.M2 = Function(self.M2, 'Time (s)', 'M2 (Nm)', 'linear')
        self.M3 = Function(self.M3, 'Time (s)', 'M3 (Nm)', 'linear')
        self.ax = Function(self.ax, 'Time (s)', 'Ax (m/s2)', 'linear')
        self.ay = Function(self.ay, 'Time (s)', 'Ay (m/s2)', 'linear')
        self.az = Function(self.az, 'Time (s)', 'Az (m/s2)', 'linear')
        self.alp1 = Function(self.alp1, 'Time (s)', 'α1 (rad/s2)', 'linear')
        self.alp2 = Function(self.alp2, 'Time (s)', 'α2 (rad/s2)', 'linear')
        self.alp3 = Function(self.alp3, 'Time (s)', 'α3 (rad/s2)', 'linear')
        # Process velocity and acceleration magnitude
        self.v = (self.vx**2 + self.vy**2 + self.vz**2)**0.5
        self.v.setDiscrete(0, self.tFinal, 1000)
        self.v.setInputs('Time (s)')
        self.v.setOutputs('Velocity Magnitude (m/s)')
        self.a = (self.ax**2 + self.ay**2 + self.az**2)**0.5
        self.a.setDiscrete(0, self.tFinal, 1000)
        self.a.setInputs('Time (s)')
        self.a.setOutputs('Acceleration Magnitude (m/s)')
        # Find out of rail and apogee velocity
        self.outOfRailVelocity = (self.vx(self.outOfRailTime)**2 +
                                  self.vy(self.outOfRailTime)**2 +
                                  self.vz(self.outOfRailTime)**2)**(0.5)
        self.apogeeVelocity = (self.vx(self.apogeeTime)**2 +
                               self.vy(self.apogeeTime)**2)**(0.5)
        # Find out maximum velocity and acceleration
        self.maxVel = np.amax(self.v.source[:, 1])
        self.maxAcc = np.amax(self.a.source[:, 1])
        # Calculate Energies
        # Retrieve variables
        # Geometry
        b = self.rocket.distanceRocketPropellant
        # Mass
        totalMass = self.rocket.totalMass
        mu = self.rocket.reducedMass
        # Inertias
        Rz = self.rocket.inertiaZ
        Ri = self.rocket.inertiaI
        Tz = self.rocket.motor.inertiaZ
        Ti = self.rocket.motor.inertiaI
        I1, I2, I3 = (Ri + Ti + mu*b**2), (Ri + Ti + mu*b**2), (Rz + Tz)
        # Velocities
        vx, vy, vz = self.vx, self.vy, self.vz
        w1, w2, w3 = self.w1, self.w2, self.w3
        # Calculate Energy Quantities
        # Kinetic Energy
        self.rotationalEnergy = 0.5*(I1*w1**2 + I2*w2**2 + I3*w3**2)
        self.rotationalEnergy.setInputs('Time (s)')
        self.rotationalEnergy.setOutputs('Rotational Kinetic Energy (J)')
        self.rotationalEnergy.setDiscrete(self.tInitial, self.tFinal, 1000)
        self.translationalEnergy = 0.5*totalMass*(vx**2 + vy**2 + vz**2)
        self.translationalEnergy.setInputs('Time (s)')
        self.translationalEnergy.setOutputs('Translational Kinetic Energy (J)')
        self.translationalEnergy.setDiscrete(self.tInitial, self.tFinal, 1000)
        self.kineticEnergy = self.rotationalEnergy + self.translationalEnergy
        self.kineticEnergy.setInputs('Time (s)')
        self.kineticEnergy.setOutputs('Kinetic Energy (J)')
        self.kineticEnergy.setDiscrete(self.tInitial, self.tFinal, 1000)
        # Potential Energy
        self.potentialEnergy = self.rocket.totalMass*self.env.g*self.z
        self.potentialEnergy.setInputs('Time (s)')
        self.potentialEnergy.setOutputs('Potential Energy (J)')
        self.potentialEnergy.setDiscrete(self.tInitial, self.tFinal, 1000)
        # Total Mechanical Energy
        self.totalEnergy = self.kineticEnergy + self.potentialEnergy
        self.totalEnergy.setInputs('Time (s)')
        self.totalEnergy.setOutputs('Total Mechanical Energy (J)')
        self.totalEnergy.setDiscrete(self.tInitial, self.tFinal, 1000)

    def info(self):
        'Prints out details about the flight'
        # Post-process results
        self.postProcess()
        # Print environment details
        print('Environment Details')
        print('Gravitational Acceleration: ' + str(self.env.g) + ' m/s2')
        print('Rail Length: ' + str(self.env.rL) + ' m')

        # Print initial conditions
        print('\nInitial Conditions')
        print('Position - x: ' + str(self.x(0)) + ' m')
        print('Position - y: ' + str(self.y(0)) + ' m')
        print('Position - z: ' + str(self.z(0)) + ' m')
        print('Velocity - Vx: ' + str(self.vx(0)) + ' m/s')
        print('Velocity - Vy: ' + str(self.vy(0)) + ' m/s')
        print('Velocity - Vz: ' + str(self.vz(0)) + ' m/s')
        print('Orientation - e0: ' + str(self.e0(0)))
        print('Orientation - e1: ' + str(self.e1(0)))
        print('Orientation - e2: ' + str(self.e2(0)))
        print('Orientation - e3: ' + str(self.e3(0)))
        print('Angular Velocity - ω1: ' + str(self.w1(0)) + ' rad/s')
        print('Angular Velocity - ω2: ' + str(self.w2(0)) + ' rad/s')
        print('Angular Velocity - ω3: ' + str(self.w3(0)) + ' rad/s')

        # Print 0ff rail conditions
        print('\nOff Rail Conditions')
        print('Rail Departure Time: ' +
              "{:.3f}".format(self.outOfRailTime) + ' s')
        print('Rail Departure Velocity: ' +
              "{:.3f}".format(self.outOfRailVelocity) + ' m/s')
        # Print Apogee conditions
        print('\nApogee')
        print('Height: ' + "{:.3f}".format(self.apogee) + ' m')
        print('Velocity: ' + "{:.3f}".format(self.apogeeVelocity) + ' m/s')
        print('Time: ' + "{:.3f}".format(self.apogeeTime) + ' s')
        print('Freestream Speed: ' +
              "{:.3f}".format(self.freestreamSpeed(self.apogeeTime)) + ' m/s')
        # Print Impact conditions
        print('\nImpact')
        print('X Impact: ' + "{:.3f}".format(self.xImpact) + ' m')
        print('Y Impact: ' + "{:.3f}".format(self.yImpact) + ' m')
        print('Time of Impact: ' + "{:.3f}".format(self.tFinal) + ' s')
        print('Velocity at Impact: ' + "{:.3f}".format(self.impactVelocity) +
              ' m/s')
        # Print maximum velocity and maximum acceleration
        print('\nMaximum Velocity and Acceleration')
        print('Velocity: ' + "{:.3f}".format(self.maxVel) + ' m/s')
        print('Acceleration: ' + "{:.3f}".format(self.maxAcc) + ' m/s2')

        print('\nTrajectory Plots')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.plot(self.x.source[:, 1], self.y.source[:, 1], self.z.source[:, 1])
        plt.show()
        self.x()
        self.y()
        self.z()
        print('\nVelocity Plots')
        self.vx()
        self.vy()
        self.vz()
        print('\nAcceleration Plots')
        self.ax()
        self.ay()
        self.az()
        print('\nAttitude Plots')
        self.e0()
        self.e1()
        self.e2()
        self.e3()
        print('\nAngular Velocity Plots')
        self.w1()
        self.w2()
        self.w3()
        print('\nAngular Acceleration Plots')
        self.alp1()
        self.alp2()
        self.alp3()
        print('\nEnergy Plots')
        self.rotationalEnergy()
        self.translationalEnergy()
        self.kineticEnergy()
        self.potentialEnergy()
        self.totalEnergy()
        print('\nAerodynamic Forces')
        self.attackAngle()
        self.freestreamSpeed()
        self.R1()
        self.R2()
        self.R3()
        self.M1()
        self.M2()
        self.M3()

    def animate(self, start=0, stop=None, fps=12, speed=4,
                elev=None, azim=None):
        'Plays an animation the flight.'
        # Set up stopping time
        stop = self.tFinal if stop is None else stop
        # Speed = 4 makes it almost real time - matplotlib is way to slow
        # Set up graph
        fig = plt.figure(figsize=(12, 9))
        axes = fig.gca(projection='3d')
        # Initialize time
        timeRange = np.linspace(start, stop, fps * (stop - start))
        # Intialize first frame
        axes.set_title('Trajectory and Velocity Animation')
        axes.set_xlabel('X (m)')
        axes.set_ylabel('Y (m)')
        axes.set_zlabel('Z (m)')
        axes.view_init(elev, azim)
        R = axes.quiver(0, 0, 0, 0, 0, 0, color='r', label='Rocket')
        V = axes.quiver(0, 0, 0, 0, 0, 0, color='g', label='Velocity')
        W = axes.quiver(0, 0, 0, 0, 0, 0, color='b', label='Wind')
        S = axes.quiver(0, 0, 0, 0, 0, 0, color='black', label='Freestream')
        axes.legend()
        # Animate
        for t in timeRange:
            R.remove()
            V.remove()
            W.remove()
            S.remove()
            # Calculate rocket position
            Rx, Ry, Rz = self.x(t), self.y(t), self.z(t)
            Ru = 1*(2*(self.e1(t)*self.e3(t) + self.e0(t)*self.e2(t)))
            Rv = 1*(2*(self.e2(t)*self.e3(t) - self.e0(t)*self.e1(t)))
            Rw = 1*(1 - 2*(self.e1(t)**2 + self.e2(t)**2))
            # Caclulate rocket Mach number
            Vx = self.vx(t)/340.40
            Vy = self.vy(t)/340.40
            Vz = self.vz(t)/340.40
            # Caculate wind Mach Number
            z = self.z(t)
            Wx = self.env.windVelocityX(z)/20
            Wy = self.env.windVelocityY(z)/20
            # Calculate freestream Mach Number
            Sx = self.streamVelX(t)/340.40
            Sy = self.streamVelY(t)/340.40
            Sz = self.streamVelZ(t)/340.40
            # Plot Quivers
            R = axes.quiver(Rx, Ry, Rz, Ru, Rv, Rw, color='r')
            V = axes.quiver(Rx, Ry, Rz, -Vx, -Vy, -Vz, color='g')
            W = axes.quiver(Rx - Vx, Ry - Vy, Rz - Vz, Wx, Wy, 0, color='b')
            S = axes.quiver(Rx, Ry, Rz, Sx, Sy, Sz, color='black')
            # Adjust axis
            axes.set_xlim(Rx - 1, Rx + 1)
            axes.set_ylim(Ry - 1, Ry + 1)
            axes.set_zlim(Rz - 1, Rz + 1)
            # plt.pause(1/(fps*speed))
            try:
                plt.pause(1/(fps*speed))
            except:
                time.sleep(1/(fps*speed))

    def __uDotRail(self, t, u, verbose=False):
        '''Derivative of u in respect to time to be integrated by odesolver
        while on rail.'''
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Makes calculation more accurate, or does it?
        if x**2 + y**2 + z**2 > self.env.rL**2:
            return self.__uDot(t, u)

        # Retrieve important quantities
        # Mass
        M = self.rocket.totalMass(t)

        # Get freestrean velocity
        windVelocity = np.array([self.env.windVelocityX(z),
                                 self.env.windVelocityY(z), 0])
        rocketVelocity = np.array([vx, vy, vz])
        freestreamVelocity = windVelocity - rocketVelocity
        freestreamSpeed = np.linalg.norm(freestreamVelocity)
        freestreamMach = freestreamSpeed/340.40
        dragCoeff = self.rocket.powerOnDrag(freestreamMach)

        # Calculate Forces
        Thrust = self.rocket.motor.thrust(t)
        R3 = -0.5*self.env.density(z)*(freestreamSpeed**2)*self.rocket.area*dragCoeff

        # Calculate Linear acceleration
        a3 = (R3 + Thrust - (e0**2 - e1**2 - e2**2 + e3**2)*self.env.g)/M
        if a3 > 0:
            K = [[1 - 2*(e2**2 + e3**2), 2*(e1*e2 - e0*e3), 2*(e1*e3 + e0*e2)],
                 [2*(e1*e2 + e0*e3), 1 - 2*(e1**2 + e3**2), 2*(e2*e3 - e0*e1)],
                 [2*(e1*e3 - e0*e2), 2*(e2*e3 + e0*e1), 1 - 2*(e1**2 + e2**2)]]
            ax, ay, az = np.dot(K, [0, 0, a3])
        else:
            ax, ay, az = 0, 0, 0
        if verbose:
            self.attackAngle.append([t, 0])
            self.freestreamSpeed.append([t, freestreamSpeed])
            self.streamVelX.append([t, freestreamVelocity[0]])
            self.streamVelY.append([t, freestreamVelocity[1]])
            self.streamVelZ.append([t, freestreamVelocity[2]])
            self.R1.append([t, 0])
            self.R2.append([t, 0])
            self.R3.append([t, R3])
            self.M1.append([t, 0])
            self.M2.append([t, 0])
            self.M3.append([t, 0])
            self.ax.append([t, ax])
            self.ay.append([t, ay])
            self.az.append([t, az])
        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def __uDot(self, t, u, verbose=False, damping=True):
        'Derivative of u in respect to time to be integrated by odesolver.'
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Retrieve important quantities
        # Inertias
        Rz = self.rocket.inertiaZ
        Ri = self.rocket.inertiaI
        Tz = self.rocket.motor.inertiaZ(t)
        Ti = self.rocket.motor.inertiaI(t)
        TzDot = self.rocket.motor.inertiaZDot(t)
        TiDot = self.rocket.motor.inertiaIDot(t)
        # Mass
        MtDot = self.rocket.motor.massDot(t)
        Mt = self.rocket.motor.mass(t)
        Mr = self.rocket.mass
        M = Mt + Mr
        mu = (Mt * Mr)/(Mt + Mr)
        # Geometry
        b = self.rocket.distanceRocketPropellant
        c = self.rocket.distanceRocketNozzle
        rN = self.rocket.motor.nozzleRadius
        # Transformation matrix: (123) -> (XYZ)
        K = [[1 - 2*(e2**2 + e3**2), 2*(e1*e2 - e0*e3), 2*(e1*e3 + e0*e2)],
             [2*(e1*e2 + e0*e3), 1 - 2*(e1**2 + e3**2), 2*(e2*e3 - e0*e1)],
             [2*(e1*e3 - e0*e2), 2*(e2*e3 + e0*e1), 1 - 2*(e1**2 + e2**2)]]
        Kt = np.transpose(K)

        # Calculate Forces and Moments
        Thrust = self.rocket.motor.thrust(t)
        # Get freestrean velocity
        windVelocity = np.array([self.env.windVelocityX(z),
                                 self.env.windVelocityY(z), 0])
        rocketVelocity = np.array([vx, vy, vz])
        freestreamVelocity = windVelocity - rocketVelocity
        freestreamSpeed = np.linalg.norm(freestreamVelocity)
        freestreamMach = freestreamSpeed/340.40

        # Determine aerodynamics forces
        # Determine Drag Force
        if freestreamSpeed > 0:
            if t > self.rocket.motor.burnOutTime:
                dragCoeff = self.rocket.powerOffDrag(freestreamMach)
            else:
                dragCoeff = self.rocket.powerOnDrag(freestreamMach)
            R3 = -0.5*self.env.density(z)*(freestreamSpeed**2)*self.rocket.area*dragCoeff
        else:
            R3 = 0
        # Determine lift force
        if damping:
            # Calculate rocket attackAngle for graphing porpuses
            # Get stream direction in 123 base - from xyz base
            streamDirection = freestreamVelocity/freestreamSpeed
            streamDirection = np.dot(Kt, streamDirection)
            attackAngle = (0 if -1 * streamDirection[2] >= 1 else
                           np.arccos(-streamDirection[2]))
            # Correct drag direction
            if attackAngle == np.pi:
                R3 *= -1
            if math.isnan(attackAngle):
                print('Error: NaN at t: ' + str(t))
                attackAngle = 0
            # Get rocket velocity in body frame
            rocketVelocityB = np.dot(Kt, rocketVelocity)
            # Get rocket components velocity in body frame
            tempVar = np.array([omega2, -omega1, 0])
            noseVelocityB = (rocketVelocityB +
                             self.rocket.noseCenterOfPressure*tempVar)
            finVelocityB = (rocketVelocityB +
                            self.rocket.finCenterOfPressure*tempVar)
            tailVelocityB = (rocketVelocityB +
                             self.rocket.tailCenterOfPressure*tempVar)
            # Get wind velocity for every rocket component
            noseZ = z + self.rocket.noseCenterOfPressure
            noseWindVelocity = np.array([self.env.windVelocityX(noseZ),
                                         self.env.windVelocityY(noseZ), 0])
            finZ = z + self.rocket.finCenterOfPressure
            finWindVelocity = np.array([self.env.windVelocityX(finZ),
                                        self.env.windVelocityY(finZ), 0])
            tailZ = z + self.rocket.tailCenterOfPressure
            tailWindVelocity = np.array([self.env.windVelocityX(tailZ),
                                         self.env.windVelocityY(tailZ), 0])
            # Get rocket components freestream velocity in body frame
            noseWindVelocityB = np.dot(Kt, noseWindVelocity)
            noseStreamB = noseWindVelocityB - noseVelocityB
            noseStreamSpeed = np.linalg.norm(noseStreamB)
            finWindVelocityB = np.dot(Kt, finWindVelocity)
            finStreamB = finWindVelocityB - finVelocityB
            finStreamSpeed = np.linalg.norm(finStreamB)
            tailWindVelocityB = np.dot(Kt, tailWindVelocity)
            tailStreamB = tailWindVelocityB - tailVelocityB
            tailStreamSpeed = np.linalg.norm(tailStreamB)
            # Get rocket components angle of attack and lift force
            noseAttackAngle = 0
            noseLift = np.array([0, 0, 0])
            if noseStreamSpeed != 0:
                noseStreamB = noseStreamB/noseStreamSpeed
                if -1 * noseStreamB[2] < 1:
                    noseAttackAngle = np.arccos(-noseStreamB[2])
                    noseLift = np.array([noseStreamB[0], noseStreamB[1], 0])
                    noseLift = (noseLift if noseAttackAngle % np.pi == 0 else
                                noseLift/np.linalg.norm(noseLift))
                    noseLift = (0.5*self.env.density(z)*(noseStreamSpeed**2)*self.rocket.area *
                                noseAttackAngle*self.rocket.noseLiftCoeffDer *
                                noseLift)
            finAttackAngle = 0
            finLift = np.array([0, 0, 0])
            if finStreamSpeed != 0:
                finStreamB = finStreamB/finStreamSpeed
                if -1 * finStreamB[2] < 1:
                    finAttackAngle = np.arccos(-finStreamB[2])
                    finLift = np.array([finStreamB[0], finStreamB[1], 0])
                    finLift = (finLift if finAttackAngle % np.pi == 0 else
                               finLift/np.linalg.norm(finLift))
                    finLift = (0.5*self.env.density(z)*(finStreamSpeed**2)*self.rocket.area *
                               finAttackAngle*self.rocket.finLiftCoeffDer *
                               finLift)
            tailAttackAngle = 0
            tailLift = np.array([0, 0, 0])
            if tailStreamSpeed != 0:
                tailStreamB = tailStreamB/tailStreamSpeed
                if -1 * tailStreamB[2] < 1:
                    tailAttackAngle = np.arccos(-tailStreamB[2])
                    tailLift = np.array([tailStreamB[0], tailStreamB[1], 0])
                    tailLift = (tailLift if tailAttackAngle % np.pi == 0 else
                                tailLift/np.linalg.norm(tailLift))
                    tailLift = (0.5*self.env.density(z)*(tailStreamSpeed**2)*self.rocket.area *
                                tailAttackAngle*self.rocket.tailLiftCoeffDer *
                                tailLift)
            # Total lift force
            totalLift = noseLift + finLift + tailLift
            R1, R2 = totalLift[0], totalLift[1]
            # Determine Moments
            B = b*Mt/M
            noseR = np.array([0, 0,
                             (self.rocket.noseCenterOfPressure + B)])
            finR = np.array([0, 0,
                            (self.rocket.finCenterOfPressure + B)])
            tailR = np.array([0, 0,
                             (self.rocket.tailCenterOfPressure + B)])
            Moment = (np.cross(noseR, noseLift) + np.cross(finR, finLift) +
                      np.cross(tailR, tailLift))
            M1, M2, M3 = Moment[0], Moment[1], Moment[2]
        elif freestreamSpeed > 0:
            # Get stream direction in 123 base - from xyz base
            streamDirection = freestreamVelocity/freestreamSpeed
            streamDirection = np.dot(Kt, streamDirection)
            attackAngle = (0 if -1 * streamDirection[2] >= 1 else
                           np.arccos(-streamDirection[2]))
            if attackAngle == np.pi:
                R3 *= -1
                R1, R2, M1, M2, M3 = 0, 0, 0, 0, 0
            elif math.isnan(attackAngle):
                print('Error: NaN at t: ' + str(t))
                attackAngle = 0
                R1, R2, M1, M2, M3 = 0, 0, 0, 0, 0
            else:
                liftVector = np.array([streamDirection[0],
                                       streamDirection[1], 0])
                liftVersor = liftVector/np.linalg.norm(liftVector)
                # Lift force for each aerodynamic part
                A = self.rocket.area
                tempVar = 0.5*self.env.density(z)*(freestreamSpeed**2)*A*attackAngle*liftVersor
                noseLift = tempVar*self.rocket.noseLiftCoeffDer
                finLift = tempVar*self.rocket.finLiftCoeffDer
                tailLift = tempVar*self.rocket.tailLiftCoeffDer
                # Total lift force
                totalLift = noseLift + finLift + tailLift
                R1, R2 = totalLift[0], totalLift[1]
                # Determine Moments
                B = b*Mt/M
                noseR = np.array([0, 0,
                                 (self.rocket.noseCenterOfPressure + B)])
                finR = np.array([0, 0,
                                (self.rocket.finCenterOfPressure + B)])
                tailR = np.array([0, 0,
                                 (self.rocket.tailCenterOfPressure + B)])
                Moment = (np.cross(noseR, noseLift) + np.cross(finR, finLift) +
                          np.cross(tailR, tailLift))
                M1, M2, M3 = Moment[0], Moment[1], Moment[2]
        else:
            attackAngle = 0
            R1, R2 = 0, 0
            M1, M2, M3 = 0, 0, 0
        # Not a number error check and alert
        if math.isnan(R1) or math.isnan(R2) or math.isnan(R3):
            print('Time: ' + str(t))
            print(noseStreamB)
            print(noseAttackAngle)
            print(liftVector)
            print(liftVersor)
            print('Major error: NaN')
            print('R1: ' + str(R1) + ' R2: ' + str(R2) + ' R3: ' + str(R3))
        if math.isnan(M1) or math.isnan(M2) or math.isnan(M3):
            print('Time: ' + str(t))
            print('Major error: NaN')
            print('M1: ' + str(M1) + ' M2: ' + str(M2) + ' M3: ' + str(M3))
        # Calculate derivatives
        # Angular acceleration
        alpha1 = (M1 - (omega2*omega3*(Rz + Tz - Ri - Ti - mu*b**2) +
                  omega1*((TiDot + MtDot*(b*mu/Mt)**2) -
                          MtDot*((rN/2)**2 + (c - b*mu/Mr)**2))))/(Ri + Ti +
                                                                   mu*b**2)
        alpha2 = (M2 - (omega1*omega3*(Ri + Ti + mu*b**2 - Rz - Tz) +
                  omega2*((TiDot + MtDot*(b*mu/Mt)**2) -
                          MtDot*((rN/2)**2 + (c - b*mu/Mr)**2))))/(Ri + Ti +
                                                                   mu*b**2)
        alpha3 = (M3 - omega3*(TzDot - MtDot*(rN**2)/2))/(Rz + Tz)

        # Euler parameters derivative
        e0Dot = 0.5*(-omega1*e1 - omega2*e2 - omega3*e3)
        e1Dot = 0.5*(omega1*e0 + omega3*e2 - omega2*e3)
        e2Dot = 0.5*(omega2*e0 - omega3*e1 + omega1*e3)
        e3Dot = 0.5*(omega3*e0 + omega2*e1 - omega1*e2)

        # Linear acceleration
        L = [(R1 - b*Mt*(omega2**2 + omega3**2) - 2*c*MtDot*omega2)/M,
             (R2 + b*Mt*(alpha3 + omega1*omega2) + 2*c*MtDot*omega1)/M,
             (R3 - b*Mt*(alpha2 - omega1*omega3) + Thrust)/M]
        ax, ay, az = np.dot(K, L)
        az -= self.env.g  # Include gravity

        # Create uDot
        uDot = [vx, vy, vz, ax, ay, az, e0Dot, e1Dot, e2Dot, e3Dot,
                alpha1, alpha2, alpha3]

        if verbose:
            self.attackAngle.append([t, attackAngle*180/np.pi])
            self.freestreamSpeed.append([t, freestreamSpeed])
            self.streamVelX.append([t, freestreamVelocity[0]])
            self.streamVelY.append([t, freestreamVelocity[1]])
            self.streamVelZ.append([t, freestreamVelocity[2]])
            self.R1.append([t, R1])
            self.R2.append([t, R2])
            self.R3.append([t, R3])
            self.M1.append([t, M1])
            self.M2.append([t, M2])
            self.M3.append([t, M3])
            self.ax.append([t, ax])
            self.ay.append([t, ay])
            self.az.append([t, az])
            self.alp1.append([t, alpha1])
            self.alp2.append([t, alpha2])
            self.alp3.append([t, alpha3])

        # Return uDot
        return uDot

    def __uDotRailOpt(self, t, u, verbose=False):
        '''Derivative of u in respect to time to be integrated by odesolver
        while on rail.'''
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Retrieve important quantities
        # Mass
        M = self.rocket.totalMass.getValueOpt(t)

        # Get freestrean speed
        freestreamSpeed = ((self.env.windVelocityX.getValueOpt(z) - vx)**2 +
                           (self.env.windVelocityY.getValueOpt(z) - vy)**2 +
                           (vz)**2)**0.5
        freestreamMach = freestreamSpeed/340.40
        dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)

        # Calculate Forces
        Thrust = self.rocket.motor.thrust.getValueOpt(t)
        rho = self.env.density.getValueOpt(z)
        R3 = -0.5*rho*(freestreamSpeed**2)*self.rocket.area*(dragCoeff)

        # Calculate Linear acceleration
        a3 = (R3 + Thrust)/M - (e0**2 - e1**2 - e2**2 + e3**2)*self.env.g
        if a3 > 0:
            ax = 2*(e1*e3 + e0*e2) * a3
            ay = 2*(e2*e3 - e0*e1) * a3
            az = (1 - 2*(e1**2 + e2**2)) * a3
        else:
            ax, ay, az = 0, 0, 0

        if verbose:
            windVelocity = np.array([self.env.windVelocityX(z),
                                     self.env.windVelocityY(z), 0])
            rocketVelocity = np.array([vx, vy, vz])
            freestreamVelocity = windVelocity - rocketVelocity
            staticMargin = (abs(self.rocket.cpPosition) - self.rocket.centerOfMass(t))/(2*self.rocket.radius)
            self.attackAngle.append([t, 0])
            self.staticMargin.append([t, staticMargin])
            self.freestreamSpeed.append([t, freestreamSpeed])
            self.streamVelX.append([t, freestreamVelocity[0]])
            self.streamVelY.append([t, freestreamVelocity[1]])
            self.streamVelZ.append([t, freestreamVelocity[2]])
            self.R1.append([t, 0])
            self.R2.append([t, 0])
            self.R3.append([t, R3])
            self.M1.append([t, 0])
            self.M2.append([t, 0])
            self.M3.append([t, 0])
            self.ax.append([t, ax])
            self.ay.append([t, ay])
            self.az.append([t, az])
        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def __uDotOpt(self, t, u, verbose=False, slow=False):
        'Derivative of u in respect to time to be integrated by odesolver.'
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Determine current behaviour
        if t < self.rocket.motor.burnOutTime or slow:
            # Motor burning
            # Retrieve important motor quantities
            # Inertias
            Tz = self.rocket.motor.inertiaZ.getValueOpt(t)
            Ti = self.rocket.motor.inertiaI.getValueOpt(t)
            TzDot = self.rocket.motor.inertiaZDot.getValueOpt(t)
            TiDot = self.rocket.motor.inertiaIDot.getValueOpt(t)
            # Mass
            MtDot = self.rocket.motor.massDot.getValueOpt(t)
            Mt = self.rocket.motor.mass.getValueOpt(t)
            # Thrust
            Thrust = self.rocket.motor.thrust.getValueOpt(t)
            # Off center moment
            MoffCenter = self.rocket.offCenter*Thrust
        else:
            # Motor stopped
            # Retrieve important motor quantities
            # Inertias
            Tz, Ti, TzDot, TiDot = 0, 0, 0, 0
            # Mass
            MtDot, Mt = 0, 0
            # Thrust
            Thrust = 0
            # Off center moment
            MoffCenter = 0

        # Retrieve important quantities
        # Inertias
        Rz = self.rocket.inertiaZ
        Ri = self.rocket.inertiaI
        # Mass
        Mr = self.rocket.mass
        M = Mt + Mr
        mu = (Mt * Mr)/(Mt + Mr)
        # Geometry
        b = self.rocket.distanceRocketPropellant
        c = self.rocket.distanceRocketNozzle
        a = b*Mt/M
        rN = self.rocket.motor.nozzleRadius
        # Prepare transformation matrix
        a11 = 1 - 2*(e2**2 + e3**2)
        a12 = 2*(e1*e2 - e0*e3)
        a13 = 2*(e1*e3 + e0*e2)
        a21 = 2*(e1*e2 + e0*e3)
        a22 = 1 - 2*(e1**2 + e3**2)
        a23 = 2*(e2*e3 - e0*e1)
        a31 = 2*(e1*e3 - e0*e2)
        a32 = 2*(e2*e3 + e0*e1)
        a33 = 1 - 2*(e1**2 + e2**2)
        # Transformation matrix: (123) -> (XYZ)
        K = [[a11, a12, a13],
             [a21, a22, a23],
             [a31, a32, a33]]
        # Transformation matrix: (XYZ) -> (123) or K transpose
        Kt = [[a11, a21, a31],
              [a12, a22, a32],
              [a13, a23, a33]]

        # Calculate Forces and Moments
        # Get freestrean speed
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = ((windVelocityX - vx)**2 +
                           (windVelocityY - vy)**2 +
                           (vz)**2)**0.5
        freestreamMach = freestreamSpeed/340.40

        # Determine aerodynamics forces
        # Determine Drag Force
        if t > self.rocket.motor.burnOutTime:
            dragCoeff = self.rocket.powerOffDrag.getValueOpt(freestreamMach)
        else:
            dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)
        rho = self.env.density.getValueOpt(z)
        R3 = -0.5*rho*(freestreamSpeed**2)*self.rocket.area*(dragCoeff)
        # Determine lift force
        # Get rocket velocity in body frame
        vxB = a11*vx + a21*vy + a31*vz
        vyB = a12*vx + a22*vy + a32*vz
        vzB = a13*vx + a23*vy + a33*vz
        # Calculate lift for each component of the rocket
        # Nose lift
        # Nose absolute velocity in body frame
        noseCp = self.rocket.noseCenterOfPressure
        noseVxB = vxB + noseCp * omega2
        noseVyB = vyB - noseCp * omega1
        noseVzB = vzB
        # Wind velocity at nose
        noseZ = z + noseCp
        noseWindVx = self.env.windVelocityX.getValueOpt(noseZ)
        noseWindVy = self.env.windVelocityY.getValueOpt(noseZ)
        # Nose freestream velocity in body frame
        noseWindVxB = a11*noseWindVx + a21*noseWindVy
        noseWindVyB = a12*noseWindVx + a22*noseWindVy
        noseWindVzB = a13*noseWindVx + a23*noseWindVy
        noseStreamVxB = noseWindVxB - noseVxB
        noseStreamVyB = noseWindVyB - noseVyB
        noseStreamVzB = noseWindVzB - noseVzB
        noseStreamSpeed = (noseStreamVxB**2 +
                           noseStreamVyB**2 +
                           noseStreamVzB**2)**0.5
        # Nose attack angle and lift force
        noseAttackAngle = 0
        noseLift, noseLiftXB, noseLiftYB = 0, 0, 0
        if noseStreamVxB**2 + noseStreamVyB**2 != 0:
            # Normalize nose stream velocity in body frame
            noseStreamVzBn = noseStreamVzB/noseStreamSpeed
            if -1 * noseStreamVzBn < 1:
                noseAttackAngle = np.arccos(-noseStreamVzBn)
                # Nose lift force magnitude
                noseLift = (0.5*self.env.density(z)*(noseStreamSpeed**2)*self.rocket.area *
                            self.rocket.noseLiftCoeffDer*noseAttackAngle)
                # Nose lift force components
                liftDirNormalization = (noseStreamVxB**2+noseStreamVyB**2)**0.5
                noseLiftXB = noseLift*(noseStreamVxB/liftDirNormalization)
                noseLiftYB = noseLift*(noseStreamVyB/liftDirNormalization)
        # Fin lift
        # Fin absolute velocity in body frame
        finCp = self.rocket.finCenterOfPressure
        finVxB = vxB + finCp * omega2
        finVyB = vyB - finCp * omega1
        finVzB = vzB
        # Wind velocity at fin
        finZ = z + finCp
        finWindVx = self.env.windVelocityX.getValueOpt(finZ)
        finWindVy = self.env.windVelocityY.getValueOpt(finZ)
        # Fin freestream velocity in body frame
        finWindVxB = a11*finWindVx + a21*finWindVy
        finWindVyB = a12*finWindVx + a22*finWindVy
        finWindVzB = a13*finWindVx + a23*finWindVy
        finStreamVxB = finWindVxB - finVxB
        finStreamVyB = finWindVyB - finVyB
        finStreamVzB = finWindVzB - finVzB
        finStreamSpeed = (finStreamVxB**2 +
                          finStreamVyB**2 +
                          finStreamVzB**2)**0.5
        # Fin attack angle and lift force
        finAttackAngle = 0
        finLift, finLiftXB, finLiftYB = 0, 0, 0
        if finStreamVxB**2 + finStreamVyB**2 != 0:
            # Normalize fin stream velocity in body frame
            finStreamVzBn = finStreamVzB/finStreamSpeed
            if -1 * finStreamVzBn < 1:
                finAttackAngle = np.arccos(-finStreamVzBn)
                # Fin lift force magnitude
                finLift = (0.5*self.env.density(z)*(finStreamSpeed**2)*self.rocket.area *
                           self.rocket.finLiftCoeffDer*finAttackAngle)
                # Fin lift force components
                liftDirNormalization = (finStreamVxB**2 + finStreamVyB**2)**0.5
                finLiftXB = finLift*(finStreamVxB/liftDirNormalization)
                finLiftYB = finLift*(finStreamVyB/liftDirNormalization)
        # Tail lift
        # Tail absolute velocity in body frame
        tailCp = self.rocket.tailCenterOfPressure
        tailVxB = vxB + tailCp * omega2
        tailVyB = vyB - tailCp * omega1
        tailVzB = vzB
        # Wind velocity at tail
        tailZ = z + tailCp
        tailWindVx = self.env.windVelocityX.getValueOpt(tailZ)
        tailWindVy = self.env.windVelocityY.getValueOpt(tailZ)
        # Tail freestream velocity in body frame
        tailWindVxB = a11*tailWindVx + a21*tailWindVy
        tailWindVyB = a12*tailWindVx + a22*tailWindVy
        tailWindVzB = a13*tailWindVx + a23*tailWindVy
        tailStreamVxB = tailWindVxB - tailVxB
        tailStreamVyB = tailWindVyB - tailVyB
        tailStreamVzB = tailWindVzB - tailVzB
        tailStreamSpeed = (tailStreamVxB**2 +
                           tailStreamVyB**2 +
                           tailStreamVzB**2)**0.5
        # Tail attack angle and lift force
        tailAttackAngle = 0
        tailLift, tailLiftXB, tailLiftYB = 0, 0, 0
        if tailStreamVxB**2 + tailStreamVyB**2 != 0:
            # Normalize tail stream velocity in body frame
            tailStreamVzBn = tailStreamVzB/tailStreamSpeed
            if -1 * tailStreamVzBn < 1:
                tailAttackAngle = np.arccos(-tailStreamVzBn)
                # Tail lift force magnitude
                tailLift = (0.5*self.env.density(z)*(tailStreamSpeed**2)*self.rocket.area *
                            self.rocket.tailLiftCoeffDer*tailAttackAngle)
                # Tail lift force components
                liftDirNormalization = (tailStreamVxB**2+tailStreamVyB**2)**0.5
                tailLiftXB = tailLift*(tailStreamVxB/liftDirNormalization)
                tailLiftYB = tailLift*(tailStreamVyB/liftDirNormalization)
        # Total lift force
        R1 = noseLiftXB + finLiftXB + tailLiftXB
        R2 = noseLiftYB + finLiftYB + tailLiftYB
        # Determine Moments
        M1 = (-(noseCp + a)*(noseLiftYB) -
              (finCp + a)*(finLiftYB) -
              (tailCp + a)*(tailLiftYB)) + MoffCenter
        M2 = ((noseCp + a)*(noseLiftXB) +
              (finCp + a)*(finLiftXB) +
              (tailCp + a)*(tailLiftXB))
        M3 = 0
        # Not a number error check and alert
        if math.isnan(R1) or math.isnan(R2) or math.isnan(R3):
            print('Time: ' + str(t))
            print('Major error: NaN')
            print('R1: ' + str(R1) + ' R2: ' + str(R2) + ' R3: ' + str(R3))
        if math.isnan(M1) or math.isnan(M2) or math.isnan(M3):
            print('Time: ' + str(t))
            print('Major error: NaN')
            print('M1: ' + str(M1) + ' M2: ' + str(M2) + ' M3: ' + str(M3))

        # Calculate derivatives
        # Angular acceleration
        alpha1 = (M1 - (omega2*omega3*(Rz + Tz - Ri - Ti - mu*b**2) +
                  omega1*((TiDot + MtDot*(Mr - 1)*(b/M)**2) -
                          MtDot*((rN/2)**2 + (c - b*mu/Mr)**2))))/(Ri + Ti +
                                                                   mu*b**2)
        alpha2 = (M2 - (omega1*omega3*(Ri + Ti + mu*b**2 - Rz - Tz) +
                  omega2*((TiDot + MtDot*(Mr - 1)*(b/M)**2) -
                          MtDot*((rN/2)**2 + (c - b*mu/Mr)**2))))/(Ri + Ti +
                                                                   mu*b**2)
        alpha3 = (M3 - omega3*(TzDot - MtDot*(rN**2)/2))/(Rz + Tz)

        # Euler parameters derivative
        e0Dot = 0.5*(-omega1*e1 - omega2*e2 - omega3*e3)
        e1Dot = 0.5*(omega1*e0 + omega3*e2 - omega2*e3)
        e2Dot = 0.5*(omega2*e0 - omega3*e1 + omega1*e3)
        e3Dot = 0.5*(omega3*e0 + omega2*e1 - omega1*e2)

        # Linear acceleration
        L = [(R1 - b*Mt*(omega2**2 + omega3**2) - 2*c*MtDot*omega2)/M,
             (R2 + b*Mt*(alpha3 + omega1*omega2) + 2*c*MtDot*omega1)/M,
             (R3 - b*Mt*(alpha2 - omega1*omega3) + Thrust)/M]
        ax, ay, az = np.dot(K, L)
        az -= self.env.g  # Include gravity

        # Create uDot
        uDot = [vx, vy, vz, ax, ay, az, e0Dot, e1Dot, e2Dot, e3Dot,
                alpha1, alpha2, alpha3]

        if verbose:
            # Calculate rocket attackAngle for graphing porpuses
            # Get stream direction in 123 base - from xyz base
            windVelocity = np.array([self.env.windVelocityX(z),
                                     self.env.windVelocityY(z), 0])
            rocketVelocity = np.array([vx, vy, vz])
            freestreamVelocity = windVelocity - rocketVelocity
            streamDirection = freestreamVelocity/freestreamSpeed
            streamDirection = np.dot(Kt, streamDirection)
            attackAngle = (0 if -1 * streamDirection[2] >= 1 else
                           np.arccos(-streamDirection[2]))
            # Correct drag direction
            if attackAngle == np.pi:
                R3 *= -1
            if math.isnan(attackAngle):
                print('Error: NaN at t: ' + str(t))
                attackAngle = 0
            # Calculate Static Margin
            d = 2*self.rocket.radius
            R123 = np.dot(Kt, [R1, R2, 0])
            Rsquared = R1**2 + R2**2 if R1**2 + R2**2 != 0 else 0.00001
            cpPosition = np.cross(R123, [M1, M2, M3])/Rsquared
            staticMargin = (abs(cpPosition[2]) - a)/d
            staticMargin = ((M1**2 + M2**2)/Rsquared - a)/d
            staticMargin = (abs(self.rocket.cpPosition) - self.rocket.centerOfMass(t))/d
            # Store Data
            self.noseStreamSpeed.append([t, noseStreamSpeed])
            self.noseAttackAngle.append([t, noseAttackAngle])
            self.noseLift.append([t, noseLift])
            self.finStreamSpeed.append([t, finStreamSpeed])
            self.finAttackAngle.append([t, finAttackAngle])
            self.finLift.append([t, finLift])
            self.tailStreamSpeed.append([t, tailStreamSpeed])
            self.tailAttackAngle.append([t, tailAttackAngle])
            self.tailLift.append([t, tailLift])
            self.cpPosition1.append([t, cpPosition[0]])
            self.cpPosition2.append([t, cpPosition[1]])
            self.cpPosition3.append([t, cpPosition[2]])
            self.staticMargin.append([t, staticMargin])
            self.attackAngle.append([t, attackAngle*180/np.pi])
            self.freestreamSpeed.append([t, freestreamSpeed])
            self.streamVelX.append([t, freestreamVelocity[0]])
            self.streamVelY.append([t, freestreamVelocity[1]])
            self.streamVelZ.append([t, freestreamVelocity[2]])
            self.R1.append([t, R1])
            self.R2.append([t, R2])
            self.R3.append([t, R3])
            self.M1.append([t, M1])
            self.M2.append([t, M2])
            self.M3.append([t, M3])
            self.ax.append([t, ax])
            self.ay.append([t, ay])
            self.az.append([t, az])
            self.alp1.append([t, alpha1])
            self.alp2.append([t, alpha2])
            self.alp3.append([t, alpha3])

        # Return uDot
        return uDot

    def __uDotDrogue(self, t, u, verbose=False):
        # Drogue data
        S = self.rocket.drogueArea
        Cd = self.rocket.drogueCd
        ka = 1
        R = 0.5
        rho = self.env.density(u[2])
        ma = ka*rho*(4/3)*np.pi*R**3
        mp = self.rocket.mass
        # Get relevant state data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        # Get wind data
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = ((windVelocityX - vx)**2 +
                           (windVelocityY - vy)**2 +
                           (vz)**2)**0.5
        freestreamX = vx - windVelocityX
        freestreamY = vy - windVelocityY
        freestreamZ = vz
        # Determine drag force
        pseudoD = -0.5 * S * Cd * freestreamSpeed
        Dx = pseudoD * freestreamX
        Dy = pseudoD * freestreamY
        Dz = pseudoD * freestreamZ
        ax = Dx/(mp+ma)
        ay = Dy/(mp+ma)
        az = (Dz - 9.8*mp)/(mp+ma)

        if verbose:
            windVelocity = np.array([self.env.windVelocityX(z),
                                     self.env.windVelocityY(z), 0])
            rocketVelocity = np.array([vx, vy, vz])
            freestreamVelocity = windVelocity - rocketVelocity
            self.freestreamSpeed.append([t, freestreamSpeed])
            self.streamVelX.append([t, freestreamVelocity[0]])
            self.streamVelY.append([t, freestreamVelocity[1]])
            self.streamVelZ.append([t, freestreamVelocity[2]])
            self.R1.append([t, Dx])
            self.R2.append([t, Dy])
            self.R3.append([t, Dz])
            self.ax.append([t, ax])
            self.ay.append([t, ay])
            self.az.append([t, az])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def __uDotMain(self, t, u, verbose=False):
        # Main data
        S = self.rocket.mainArea
        Cd = self.rocket.mainCd
        ka = 1
        R = 1.5
        rho = self.env.density(u[2])
        to = 1.2
        ma = ka*rho*(4/3)*np.pi*R**3
        mp = self.rocket.mass
        eta = 1
        Rdot = (6*R*(1-eta)/(1.2**6))*((1-eta)*t**5 + eta*(to**3)*(t**2))
        Rdot = 0
        # Get relevant state data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        # Get wind data
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = ((windVelocityX - vx)**2 +
                           (windVelocityY - vy)**2 +
                           (vz)**2)**0.5
        freestreamX = vx - windVelocityX
        freestreamY = vy - windVelocityY
        freestreamZ = vz
        # Determine drag force
        pseudoD = -0.5 * S * Cd * freestreamSpeed - ka*rho*4*np.pi*(R**2)*Rdot
        Dx = pseudoD * freestreamX
        Dy = pseudoD * freestreamY
        Dz = pseudoD * freestreamZ
        ax = Dx/(mp+ma)
        ay = Dy/(mp+ma)
        az = (Dz - 9.8*mp)/(mp+ma)

        if verbose:
            windVelocity = np.array([self.env.windVelocityX(z),
                                     self.env.windVelocityY(z), 0])
            rocketVelocity = np.array([vx, vy, vz])
            freestreamVelocity = windVelocity - rocketVelocity
            self.freestreamSpeed.append([t, freestreamSpeed])
            self.streamVelX.append([t, freestreamVelocity[0]])
            self.streamVelY.append([t, freestreamVelocity[1]])
            self.streamVelZ.append([t, freestreamVelocity[2]])
            self.R1.append([t, Dx])
            self.R2.append([t, Dy])
            self.R3.append([t, Dz])
            self.ax.append([t, ax])
            self.ay.append([t, ay])
            self.az.append([t, az])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]