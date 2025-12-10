import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.prints.aero_surface_prints import _RailButtonsPrints

from .aero_surface import AeroSurface


class RailButtons(AeroSurface):
    """Class that defines a rail button pair or group.

    Attributes
    ----------
    RailButtons.buttons_distance : int, float
        Distance between the two rail buttons closest to the nozzle.
    RailButtons.angular_position : int, float
        Angular position of the rail buttons in degrees measured
        as the rotation around the symmetry axis of the rocket
        relative to one of the other principal axis.
    RailButtons.angular_position_rad : float
        Angular position of the rail buttons in radians.
    RailButtons.button_height : float, optional
        Height (standoff distance) of the rail button from the rocket
        body surface to the rail contact point, in meters. Used for
        calculating bending moments at the attachment point.
        Default is None. If not provided, bending moments cannot be
        calculated but flight dynamics remain unaffected.
    """

    def __init__(
        self,
        buttons_distance,
        angular_position=45,
        button_height=None,
        name="Rail Buttons",
        rocket_radius=None,
    ):
        """Initializes RailButtons Class.

        Parameters
        ----------
        buttons_distance : int, float
            Distance between the first and the last rail button in meters.
        angular_position : int, float, optional
            Angular position of the rail buttons in degrees measured
            as the rotation around the symmetry axis of the rocket
            relative to one of the other principal axis.
        name : string, optional
            Name of the rail buttons. Default is "Rail Buttons".
        rocket_radius : int, float, optional
            Radius of the rocket at the location of the rail buttons in meters.
            If not provided, it will be calculated when the RailButtons object
            is added to a Rocket object.
        """
        super().__init__(name, None, None)
        self.buttons_distance = buttons_distance
        self.angular_position = angular_position
        self.button_height = button_height
        self.name = name
        self.rocket_radius = rocket_radius
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

        self.prints = _RailButtonsPrints(self)

    @property
    def angular_position_rad(self):
        return np.radians(self.angular_position)

    def evaluate_center_of_pressure(self):
        """Evaluates the center of pressure of the rail buttons. Rail buttons
        do not contribute to the center of pressure of the rocket.

        Returns
        -------
        None
        """
        self.cpx = 0
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)

    def evaluate_lift_coefficient(self):
        """Evaluates the lift coefficient curve of the rail buttons. Rail
        buttons do not contribute to the lift coefficient of the rocket.

        Returns
        -------
        None
        """
        self.clalpha = Function(
            lambda mach: 0,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: 0,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

    def evaluate_geometrical_parameters(self):
        """Evaluates the geometrical parameters of the rail buttons. Rail
        buttons do not contribute to the geometrical parameters of the rocket.

        Returns
        -------
        None
        """

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "buttons_distance": self.buttons_distance,
            "angular_position": self.angular_position,
            "button_height": self.button_height,
            "name": self.name,
            "rocket_radius": self.rocket_radius,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["buttons_distance"],
            data["angular_position"],
            data.get("button_height", None),
            data["name"],
            data["rocket_radius"],
        )

    def info(self):
        """Prints out all the information about the Rail Buttons.

        Returns
        -------
        None
        """
        self.prints.geometry()

    def all_info(self):
        """Returns all info of the Rail Buttons.

        Returns
        -------
        None
        """
        self.prints.all()
