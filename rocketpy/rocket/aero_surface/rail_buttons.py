import numpy as np

from rocketpy.prints.aero_surface_prints import _RailButtonsPrints

from .generic_surface import GenericSurface


class RailButtons(GenericSurface):
    """Class that defines a rail button pair or group.

    Attributes
    ----------
    RailButtons.buttons_distance : int, float
        Distance between the two rail buttons closest to the nozzle.
    RailButtons.angular_position : int, float
        Angular position of the rail buttons in degrees measured
        as the rotation around the symmetry axis of the rocket
        relative to one of the other principal axis.
        See :ref:`Angular Position Inputs <angular_position>`
    RailButtons.angular_position_rad : float
        Angular position of the rail buttons in radians.
    RailButtons.button_height : float, optional
        Height (standoff distance) of the rail button from the rocket
        body surface to the rail contact point, in meters. Used for
        calculating bending moments at the attachment point.
        Default is None. If not provided, bending moments cannot be
        calculated but flight dynamics remain unaffected.
    """

    # Rail buttons carry no aerodynamic force, so they contribute nothing to
    # either plane: axisymmetric for the pitch/yaw check.
    is_axisymmetric = True

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
        self.buttons_distance = buttons_distance
        self.angular_position = angular_position
        self.button_height = button_height
        self.rocket_radius = rocket_radius

        # Rail buttons produce no aerodynamic force; they are modeled as a
        # generic surface with all-zero coefficients. The reference area/length
        # are placeholders (never used, since rail buttons are not part of the
        # rocket's aerodynamic_surfaces) computed from the rocket radius when
        # available.
        reference_radius = rocket_radius or 1.0
        super().__init__(
            reference_area=np.pi * reference_radius**2,
            reference_length=2 * reference_radius,
            coefficients={},
            center_of_pressure=(0, 0, 0),
            name=name,
        )

        self.prints = _RailButtonsPrints(self)

    @property
    def angular_position_rad(self):
        return np.radians(self.angular_position)

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
