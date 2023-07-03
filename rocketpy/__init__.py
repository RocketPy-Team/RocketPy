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
__copyright__ = "Copyright 20XX, RocketPy Team"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__credits__ = ["Matheus Marques Araujo", "Rodrigo Schmitt", "Guilherme Tavares"]
__license__ = "MIT"
__version__ = "1.0.0a1"
__maintainer__ = "Giovani Hidalgo Ceotto"
__email__ = "ghceotto@gmail.com"
__status__ = "Production"

from .AeroSurface import (
    AeroSurface,
    EllipticalFins,
    Fins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from .Components import Components
from .Environment import Environment
from .EnvironmentAnalysis import EnvironmentAnalysis
from .Flight import Flight
from .Function import Function
from .motors import (
    CylindricalTank,
    EmptyMotor,
    Fluid,
    GenericMotor,
    HybridMotor,
    LevelBasedTank,
    LiquidMotor,
    MassBasedTank,
    MassFlowRateBasedTank,
    SolidMotor,
    SphericalTank,
    Tank,
    TankGeometry,
    UllageBasedTank,
)
from .plots import *
from .prints import *
from .Rocket import Rocket
from .utilities import *
