__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _TankGeometryPlots:
    """Class that holds plot methods for TankGeometry class.

    Attributes
    ----------
    _TankGeometryPlots.tank_geometry : TankGeometry
        TankGeometry object that will be used for the plots.

    """

    def __init__(self, tank_geometry):
        """Initializes _MotorClass class.

        Parameters
        ----------
        tank_geometry : TankGeometry
            Instance of the TankGeometry class

        Returns
        -------
        None
        """

        self.tank_geometry = tank_geometry

        return None

    def all(self):
        """Prints out all graphs available about the TankGeometry. It simply calls
        all the other plotter methods in this class.

        Parameters
        ----------
        None
        Return
        ------
        None
        """

        return None
