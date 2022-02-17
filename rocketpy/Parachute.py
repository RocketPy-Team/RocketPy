__author__ = "Giovani Hidalgo Ceotto, Franz Masatoshi Yuri"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"

import re
import math
import bisect
import warnings
import time
from datetime import datetime, timedelta
from inspect import signature, getsourcelines
from collections import namedtuple

import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import genfromtxt

from .Function import Function


class Parachute:
    """Keeps parachute information.

        Attributes
        ----------
    Parachute attributes:
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        CdS : float
            Drag coefficient times reference area for parachute. It is
            used to compute the drag force exerted on the parachute by
            the equation F = ((1/2)*rho*V^2)*CdS, that is, the drag
            force is the dynamic pressure computed on the parachute
            times its CdS coefficient. Has units of area and must be
            given in squared meters.
        trigger : function
            Function which defines if the parachute ejection system is
            to be triggered. It must take as input the freestream
            pressure in pascal and the state vector of the simulation,
            which is defined by [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].
            It will be called according to the sampling rate given next.
            It should return True if the parachute ejection system is
            to be triggered and False otherwise.
        samplingRate : float
            Sampling rate in which the trigger function works. It is used to
            simulate the refresh rate of onboard sensors such as barometers.
            Default value is 100. Value must be given in hertz.
        lag : float
            Time between the parachute ejection system is triggered and the
            parachute is fully opened. During this time, the simulation will
            consider the rocket as flying without a parachute. Default value
            is 0. Must be given in seconds.
        noiseBias : float
            Mean value of the noise added to the pressure signal, which is
            passed to the trigger function. Unit is in pascal.
        noiseDeviation : float
            Standard deviation of the noise added to the pressure signal,
            which is passed to the trigger function. Unit is in pascal.
        noiseCorr : tuple, list
            Tupple with the correlation between noise and time.
        noiseSignal : list
            List of (t, noise signal) corresponding to signal passed to
            trigger function. Completed after running a simulation.
        noisyPressureSignal : list
            List of (t, noisy pressure signal) that is passed to the
            trigger function. Completed after running a simulation.
        cleanPressureSignal : list
            List of (t, clean pressure signal) corresponding to signal passed to
            trigger function. Completed after running a simulation.
        noiseSignalFunction : Function
            Function of noiseSignal.
        noisyPressureSignalFunction : Function
            Function of noisyPressureSignal.
        cleanPressureSignalFunction : Function
            Function of cleanPressureSignal.
    """

    def __init__(
        self,
        name,
        CdS,
        Trigger,
        samplingRate,
        lag,
        noise=(0, 0, 0),
    ):
        """Initializes Parachute class.
        Parameters
        ----------
        name : string
            Name of the parachute.
        CdS : float
            CdS of the parachute.
        Trigger : function
            Trigger function.
        samplngRate : float
            Sampling rate, in hertz, for the Trigger function.
        lag : float
            Time, in seconds, between the parachute ejection system is triggered and the
        parachute is fully opened.
        noise : tuple, list, optional
            List in the format (mean, standard deviation, time-correlation).
            The values are used to add noise to the pressure signal which is
            passed to the trigger function. Default value is (0, 0, 0). Units
            are in pascal.
        Returns
        -------
        None
        """
        self.name = name
        self.CdS = CdS
        self.trigger = Trigger
        self.samplingRate = samplingRate
        self.lag = lag
        self.noiseSignal = [[-1e-6, np.random.normal(noise[0], noise[1])]]
        self.noisyPressureSignal = []
        self.cleanPressureSignal = []
        self.noiseBias = noise[0]
        self.noiseDeviation = noise[1]
        self.noiseCorr = (noise[2], (1 - noise[2] ** 2) ** 0.5)
        alpha, beta = self.noiseCorr
        self.noiseFunction = lambda: alpha * self.noiseSignal[-1][
            1
        ] + beta * np.random.normal(noise[0], noise[1])
        return None
