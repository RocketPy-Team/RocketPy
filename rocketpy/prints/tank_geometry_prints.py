import logging

logger = logging.getLogger(__name__)

class _TankGeometryPrints:
    """Class that holds prints methods for TankGeometry class.

    Attributes
    ----------
    _TankGeometryPrints.tank_geometry : TankGeometry
        TankGeometry object that will be used for the prints.

    """

    def __init__(
        self,
        tank_geometry,
    ):
        """Initializes _TankGeometryPrints class

        Parameters
        ----------
        tank_geometry: TankGeometry
            Instance of the TankGeometry class.

        Returns
        -------
        None
        """
        self.tank_geometry = tank_geometry

    def geometry(self):
        """Prints out the geometry of the tank.

        Returns
        -------
        None
        """
        logger.info("Tank Geometry:")
        logger.info(f"Average radius {self.tank_geometry.average_radius:.3f} m")
        logger.info(f"Bottom: {self.tank_geometry.bottom} m")
        logger.info(f"Top: {self.tank_geometry.top} m")
        logger.info(f"Total height: {self.tank_geometry.total_height} m")
        logger.info(f"Total volume: {self.tank_geometry.total_volume:.6f} m^3\n")

    def all(self):
        """Prints out all data available about the TankGeometry.

        Returns
        -------
        None
        """
        self.geometry()
