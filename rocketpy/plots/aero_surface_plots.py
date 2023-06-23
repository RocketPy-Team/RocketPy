__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _NoseConePlots:
    """Class that contains all nosecone plots."""

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
