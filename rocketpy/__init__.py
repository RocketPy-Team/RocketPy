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
    thrustSource=r"C:\Users\oscar\Documents\Repositorios\RocketPy\data\motors\Cesaroni_7450M2505-P.eng",
    burnOut=3,
    grainNumber=2,
    grainDensity=1815,
    grainOuterRadius=33 / 1000,
    grainInitialInnerRadius=15 / 1000,
    grainInitialHeight=120 / 1000,
    oxidizerTankRadius=33 / 1000,
    oxidizerTankHeight=200 / 1000,
    oxidizerInitialPresure=60,
    oxidizerDensity=1.98,
    oxidizerMolarMass=44.01,
    oxidizerInitialVolume=0.000341946,
    distanceGrainToTank=120 / 1000,
    injectorArea=10 / 1000,
)

my_motor.info()
