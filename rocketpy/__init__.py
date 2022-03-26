# -*- coding: utf-8 -*-

"""
RocketPy is a trajectory simulation for High-Power Rocketry built by
[Projeto Jupiter](https://www.facebook.com/ProjetoJupiter/). The code allows
for a complete 6 degrees of freedom simulation of a rocket's flight trajectory,
including high fidelity variable mass effects as well as descent under
parachutes. Weather conditions, such as wind profile, can be imported from
sophisticated datasets, allowing for realistic scenarios. Furthermore, the
implementation facilitates complex simulations, such as multi-stage rockets,
design and trajectory optimization and dispersion analysis.
"""

__author__ = "Giovani Hidalgo Ceotto"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__credits__ = ["Matheus Marques Araujo", "Rodrigo Schmitt", "Guilherme Tavares"]
__license__ = "MIT"
__version__ = "0.9.8"
__maintainer__ = "Giovani Hidalgo Ceotto"
__email__ = "ghceotto@gmail.com"
__status__ = "Production"

import re
import math
import bisect
import warnings
import time
from Function import Function
from datetime import datetime, timedelta
from inspect import signature, getsourcelines
from collections import namedtuple

import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# from Function import Function
# from Environment import Environment
from Motor import SolidMotor, HybridMotor

# from Rocket import Rocket
# from Flight import Flight

my_motor = HybridMotor(
    thrustSource="data/motors/Cesaroni_7450M2505-P.eng",
    burnOut=3,
    grainNumber=1,
    grainDensity=1815,
    grainOuterRadius=97 / 2000,
    grainInitialInnerRadius=25 / 1000,
    grainInitialHeight=230 / 1000,
    oxidizerTankRadius=62.5 / 1000,
    oxidizerTankHeight=600 / 1000,
    oxidizerInitialPresure=60e5,
    oxidizerDensity=1.98 * 1000,
    oxidizerMolarMass=44.01,
    oxidizerInitialVolume=62.5 / 1000 * 62.5 / 1000 * np.pi * 600 / 1000,
    distanceGrainToTank=200 / 1000,  # does not matter rsrs
    injectorArea=3e-05,
)

# 2.1237166338267
# 5.9215396

my_motor.info()
# print(my_motor.yCM)

# a = lambda x: x**2
# func = Function(lambda x: x**2 * a(x))
# func()

my_motor.yCM()


# my_motor.mass.plot()

# Function(my_motor.solidMass).plot()
# Function(my_motor.liquidMass).plot()

# Function(my_motor.massInChamberCM).plot()
# Function(my_motor.liquidCM).plot()

# Function.comparePlots(
#     [
#         (Function(my_motor.massInChamber), "massInChamber"),
#         (Function(my_motor.liquidMass), "liquidMass"),
#         (my_motor.mass, "mass"),
#     ],
# )

print(my_motor.yCM(-1))
