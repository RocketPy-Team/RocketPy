# -*- coding: utf-8 -*-

__author__ = "Franz Masatoshi Yuri"
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

from rocketpy import Environment, Rocket, SolidMotor, Flight

def compute_cds_from_drop_test(final_air_density, final_speed, rocket_mass, gravity):
    """Returns the parachute's Cds calculated through its final speed, air 
    density and the rocket's mass.

    Parameters
    ----------
    final_speed : float
        Rocket's speed when landing.
    final_air_density : float
        Air density right before the rocket lands.
    rocket_mass : float
        Rocket's mass in kgpygame.examples.mask.main()

    Returns
    -------
    cds : float
        Number equal to drag coefficient times reference area for parachute.

    """
        
    cds = 2 * rocket_mass * gravity / ((final_speed ** 2) * final_air_density)
    return cds
