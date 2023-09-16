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

    def radius(self, upper=None, lower=None):
        self.tank_geometry.radius.plot(lower, upper)
        return None

    def area(self, upper=None, lower=None):
        self.tank_geometry.area.plot(lower, upper)
        return None

    def volume(self, upper=None, lower=None):
        self.tank_geometry.volume.plot(lower, upper)
        return None

    def all(self):
        """Prints out all graphs available about the TankGeometry. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.radius()
        self.area()
        self.volume()
        return None
