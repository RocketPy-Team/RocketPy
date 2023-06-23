__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


class _AeroSurfacePlots(ABC):
    """Abstract class that contains all aero surface plots."""

    def __init__(self, aero_surface):
        """Initialize the class

        Parameters
        ----------
        aero_surface : rocketpy.AeroSurface
            AeroSurface object to be plotted

        Returns
        -------
        None
        """
        self.aero_surface = aero_surface
        return None

    @abstractmethod
    def cross_section(self):
        pass

    def lift(self):
        """Plots the lift coefficient of the aero surface as a function of Mach
        and the angle of attack. A 3D plot is expected. See the rocketpy.Function
        class for more information on how this plot is made.

        Returns
        -------
        None
        """
        self.aero_surface.cl()
        return None

    def all(self):
        """Plots all aero surface plots.

        Returns
        -------
        None
        """
        self.cross_section()
        self.lift()
        return None


    def __init__(self, nosecone):
        """Initialize the class

        Parameters
        ----------
        nosecone : rocketpy.AeroSurface.NoseCone
            Nosecone object to be plotted

        Returns
        -------
        None
        """
        self.nosecone = nosecone
        return None

    def cross_section(self):
        # This will de done in the future
        return None

    def lift(self):
        """Plots the lift coefficient of the nosecone as a function of Mach and
        the angle of attack.

        Returns
        -------
        None
        """
        self.nosecone.cl()
        return None

    def all(self):
        self.cross_section()
        self.lift()
        return None
