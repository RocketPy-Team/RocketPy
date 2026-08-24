import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _NoseConePlots
from rocketpy.prints.aero_surface_prints import _NoseConePrints

from ._barrowman_surface import _BarrowmanSurface


class BodyTube(_BarrowmanSurface):
    """Keeps body tube information.

    A constant-radius cylindrical section of the airframe. By slender-body
    theory it produces no linear normal force (``clalpha = 0``); its entire
    in-flight normal force comes from the nonlinear Galejs body-lift term
    ``K · (A_plan / A_ref) · sin²α`` applied at the planform centroid, exactly
    like OpenRocket's ``SymmetricComponentCalc`` for a straight tube.

    Note
    ----
    Local coordinate system:
        - the origin at the top of the tube (the portion closest to the rocket's nose) and
        - the Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).

    Attributes
    ----------
    BodyTube.length : float
        Body tube length. Has units of length and must be given in meters.
    BodyTube.radius : float
        Body tube outer radius. Has units of length and must be given in meters.
    BodyTube.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization,
        in meters. Defaults to the tube radius.
    BodyTube.name : string
        Body tube name. Has no impact in simulation, as it is only used to
        display data in a more organized matter.
    BodyTube.cp : tuple
        Tuple with the x, y and z local coordinates of the body tube center of
        pressure. Has units of length and is given in meters.
    BodyTube.clalpha : float
        Normal-force coefficient slope. Identically zero for a constant-radius
        tube.
    BodyTube.plots : plots.aero_surface_plots._NoseConePlots
        This contains all the plots methods. Use help(BodyTube.plots) to know
        more about it.
    BodyTube.prints : prints.aero_surface_prints._NoseConePrints
        This contains all the prints methods. Use help(BodyTube.prints) to know
        more about it.
    """

    def __init__(
        self,
        length,
        radius,
        rocket_radius=None,
        name="Body Tube",
    ):
        """Initializes the body tube object by computing and storing the most
        important values.

        Parameters
        ----------
        length : float
            Body tube length. Has units of length and must be given in meters.
        radius : float
            Body tube outer radius. Has units of length and must be given in
            meters.
        rocket_radius : int, float, optional
            The reference rocket radius used for lift coefficient normalization.
            Defaults to the tube radius when not given.
        name : str, optional
            Body tube name. Has no impact in simulation, as it is only used to
            display data in a more organized matter.

        Returns
        -------
        None
        """
        rocket_radius = rocket_radius or radius
        self.name = name
        self.reference_area = np.pi * rocket_radius**2
        self.reference_length = 2 * rocket_radius

        self._length = length
        self._radius = radius
        self._rocket_radius = rocket_radius

        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()
        self.evaluate_body_lift_geometry()

        # Translate the Barrowman geometry into the linear generic-surface
        # coefficient model and build the shared compute path.
        super().__init__(
            reference_area=self.reference_area,
            reference_length=self.reference_length,
            coefficients={},
            center_of_pressure=(self.cpx, self.cpy, self.cpz),
            name=name,
        )

        self.plots = _NoseConePlots(self)
        self.prints = _NoseConePrints(self)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_body_lift_geometry()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.evaluate_center_of_pressure()
        self.evaluate_body_lift_geometry()

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.reference_area = np.pi * value**2
        self.reference_length = 2 * value

    def evaluate_geometrical_parameters(self):
        """Calculates and saves the body tube's surface area.

        Returns
        -------
        None
        """
        self.surface_area = 2 * np.pi * self.radius * self.length
        self.fineness_ratio = self.length / (2 * self.radius)

    def evaluate_lift_coefficient(self):
        """A constant-radius tube produces no slender-body normal force, so
        its lift-curve slope is identically zero; all normal force comes from
        the Galejs body-lift term.

        Returns
        -------
        None
        """
        self.clalpha = Function(
            lambda mach: 0.0,
            "Mach",
            f"Normal-force coefficient derivative for {self.name}",
        )

    def evaluate_center_of_pressure(self):
        """The geometric center of pressure sits at the tube midpoint in local
        coordinates. With zero slender-body lift this point carries no force;
        the Galejs term is applied at the planform centroid instead.

        Returns
        -------
        self.cp : tuple
            Tuple containing cpx, cpy, cpz.
        """
        self.cpx = 0
        self.cpy = 0
        self.cpz = self.length / 2
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def evaluate_body_lift_geometry(self):
        """Compute the planform (side-projection) geometry used by the Galejs
        body-lift term: a rectangle of width 2·R and height L, with centroid
        at the midpoint. The slender-body CP equals the same midpoint.

        Returns
        -------
        None
        """
        self._planform_area = 2 * self.radius * self.length
        self._planform_centroid = self.length / 2
        self._cp_slender = self.cpz

    def info(self):
        """Prints and plots summarized information of the body tube.

        Return
        ------
        None
        """
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        """Prints and plots all the available information of the body tube.

        Returns
        -------
        None
        """
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = {
            "length": self._length,
            "radius": self._radius,
            "rocket_radius": self._rocket_radius,
            "name": self.name,
        }

        if kwargs.get("include_outputs", False):
            clalpha = self.clalpha
            if kwargs.get("discretize", False):
                clalpha = Function(clalpha).set_discrete(0, 4, 50)

            data.update(
                {
                    "cp": self.cp,
                    "clalpha": clalpha,
                    "surface_area": self.surface_area,
                }
            )

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            length=data["length"],
            radius=data["radius"],
            rocket_radius=data["rocket_radius"],
            name=data["name"],
        )
