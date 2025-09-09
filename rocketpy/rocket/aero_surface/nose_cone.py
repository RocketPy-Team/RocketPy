import warnings

import numpy as np
from scipy.optimize import fsolve

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _NoseConePlots
from rocketpy.prints.aero_surface_prints import _NoseConePrints

from .aero_surface import AeroSurface


class NoseCone(AeroSurface):
    """Keeps nose cone information.

    Note
    ----
    Local coordinate system:
        - the origin at the tip of the nose cone and
        - the Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).

    Attributes
    ----------
    NoseCone.length : float
        Nose cone length. Has units of length and must be given in meters.
    NoseCone.kind : string
        Nose cone kind. Can be "conical", "ogive", "elliptical", "tangent",
        "von karman", "parabolic", "powerseries" or "lvhaack".
    NoseCone.bluffness : float
        Ratio between the radius of the circle on the tip of the ogive and the
        radius of the base of the ogive. Currently only used for the nose cone's
        drawing. Must be between 0 and 1. Default is None, which means that the
        nose cone will not have a sphere on the tip. If a value is given, the
        nose cone's length will be slightly reduced because of the addition of
        the sphere. Must be None or 0 if a "powerseries" nose cone kind is
        specified.
    NoseCone.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization,
        in meters.
    NoseCone.base_radius : float
        Nose cone base radius. Has units of length and must be given in meters.
    NoseCone.radius_ratio : float
        Ratio between the nose cone base radius and the rocket radius. Has no
        units. If base radius is not given, the ratio between base radius and
        rocket radius is assumed as 1, meaning that the nose cone has the same
        radius as the rocket. If base radius is given, the ratio between base
        radius and rocket radius is calculated and used for lift calculation.
    NoseCone.power : float
        Factor that controls the bluntness of the shape for a power series
        nose cone. Must be between 0 and 1. It is ignored when other nose
        cone types are used.
    NoseCone.name : string
        Nose cone name. Has no impact in simulation, as it is only used to
        display data in a more organized matter.
    NoseCone.cp : tuple
        Tuple with the x, y and z local coordinates of the nose cone center of
        pressure. Has units of length and is given in meters.
    NoseCone.cpx : float
        Nose cone local center of pressure x coordinate. Has units of length and
        is given in meters.
    NoseCone.cpy : float
        Nose cone local center of pressure y coordinate. Has units of length and
        is given in meters.
    NoseCone.cpz : float
        Nose cone local center of pressure z coordinate. Has units of length and
        is given in meters.
    NoseCone.cl : Function
        Function which defines the lift coefficient as a function of the angle
        of attack and the Mach number. Takes as input the angle of attack in
        radians and the Mach number. Returns the lift coefficient.
    NoseCone.clalpha : float
        Lift coefficient slope. Has units of 1/rad.
    NoseCone.plots : plots.aero_surface_plots._NoseConePlots
        This contains all the plots methods. Use help(NoseCone.plots) to know
        more about it.
    NoseCone.prints : prints.aero_surface_prints._NoseConePrints
        This contains all the prints methods. Use help(NoseCone.prints) to know
        more about it.
    """

    def __init__(  # pylint: disable=too-many-statements
        self,
        length,
        kind,
        base_radius=None,
        bluffness=None,
        rocket_radius=None,
        power=None,
        name="Nose Cone",
    ):
        """Initializes the nose cone. It is used to define the nose cone
        length, kind, center of pressure and lift coefficient curve.

        Parameters
        ----------
        length : float
            Nose cone length. Has units of length and must be given in meters.
        kind : string
            Nose cone kind. Can be "conical", "ogive", "elliptical", "tangent",
            "von karman", "parabolic", "powerseries" or "lvhaack". If
            "powerseries" is used, the "power" argument must be assigned to a
            value between 0 and 1.
        base_radius : float, optional
            Nose cone base radius. Has units of length and must be given in
            meters. If not given, the ratio between ``base_radius`` and
            ``rocket_radius`` will be assumed as 1.
        bluffness : float, optional
            Ratio between the radius of the circle on the tip of the ogive and
            the radius of the base of the ogive. Currently only used for the
            nose cone's drawing. Must be between 0 and 1. Default is None, which
            means that the nose cone will not have a sphere on the tip. If a
            value is given, the nose cone's length will be reduced to account
            for the addition of the sphere at the tip. Must be None or 0 if a
            "powerseries" nose cone kind is specified.
        rocket_radius : int, float, optional
            The reference rocket radius used for lift coefficient normalization.
            If not given, the ratio between ``base_radius`` and
            ``rocket_radius`` will be assumed as 1.
        power : float, optional
            Factor that controls the bluntness of the shape for a power series
            nose cone. Must be between 0 and 1. It is ignored when other nose
            cone types are used.
        name : str, optional
            Nose cone name. Has no impact in simulation, as it is only used to
            display data in a more organized matter.

        Returns
        -------
        None
        """
        rocket_radius = rocket_radius or base_radius
        super().__init__(name, np.pi * rocket_radius**2, 2 * rocket_radius)

        self._rocket_radius = rocket_radius
        self._base_radius = base_radius
        self._length = length
        if bluffness is not None:
            if bluffness > 1 or bluffness < 0:  # pragma: no cover
                raise ValueError(
                    f"Bluffness ratio of {bluffness} is out of range. "
                    "It must be between 0 and 1."
                )
        self._bluffness = bluffness
        if kind == "powerseries":
            # Checks if bluffness is not being used
            if (self.bluffness is not None) and (self.bluffness != 0):
                raise ValueError(
                    "Parameter 'bluffness' must be None or 0 when using a nose cone kind 'powerseries'."
                )

            if power is None:
                raise ValueError(
                    "Parameter 'power' cannot be None when using a nose cone kind 'powerseries'."
                )

            if power > 1 or power <= 0:
                raise ValueError(
                    f"Power value of {power} is out of range. It must be between 0 and 1."
                )
        self._power = power
        self.kind = kind

        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

        self.plots = _NoseConePlots(self)
        self.prints = _NoseConePrints(self)

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_nose_shape()

    @property
    def base_radius(self):
        return self._base_radius

    @base_radius.setter
    def base_radius(self, value):
        self._base_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_nose_shape()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.evaluate_center_of_pressure()
        self.evaluate_nose_shape()

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        if value is not None:
            if value > 1 or value <= 0:
                raise ValueError(
                    f"Power value of {value} is out of range. It must be between 0 and 1."
                )
        self._power = value
        self.evaluate_k()
        self.evaluate_center_of_pressure()
        self.evaluate_nose_shape()

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):  # pylint: disable=too-many-statements
        # Analyzes nosecone type
        # Sets the k for Cp calculation
        # Sets the function which creates the respective curve
        self._kind = value
        value = (value.replace(" ", "")).lower()

        if value == "conical":
            self.k = 2 / 3
            self.y_nosecone = Function(lambda x: x * self.base_radius / self.length)

        elif value == "lvhaack":
            self.k = 0.563

            def theta(x):
                return np.arccos(1 - 2 * max(min(x / self.length, 1), 0))

            self.y_nosecone = Function(
                lambda x: self.base_radius
                * (theta(x) - np.sin(2 * theta(x)) / 2 + (np.sin(theta(x)) ** 3) / 3)
                ** (0.5)
                / (np.pi**0.5)
            )

        elif value in ["tangent", "tangentogive", "ogive"]:
            rho = (self.base_radius**2 + self.length**2) / (2 * self.base_radius)
            volume = np.pi * (
                self.length * rho**2
                - (self.length**3) / 3
                - (rho - self.base_radius) * rho**2 * np.arcsin(self.length / rho)
            )
            area = np.pi * self.base_radius**2
            self.k = 1 - volume / (area * self.length)
            self.y_nosecone = Function(
                lambda x: np.sqrt(rho**2 - (min(x - self.length, 0)) ** 2)
                + (self.base_radius - rho)
            )

        elif value == "elliptical":
            self.k = 1 / 3
            self.y_nosecone = Function(
                lambda x: self.base_radius
                * np.sqrt(1 - ((x - self.length) / self.length) ** 2)
            )

        elif value == "vonkarman":
            self.k = 0.5

            def theta(x):
                return np.arccos(1 - 2 * max(min(x / self.length, 1), 0))

            self.y_nosecone = Function(
                lambda x: self.base_radius
                * (theta(x) - np.sin(2 * theta(x)) / 2) ** (0.5)
                / (np.pi**0.5)
            )
        elif value == "parabolic":
            self.k = 0.5
            self.y_nosecone = Function(
                lambda x: self.base_radius
                * ((2 * x / self.length - (x / self.length) ** 2) / (2 - 1))
            )
        elif value == "powerseries":
            self.k = (2 * self.power) / ((2 * self.power) + 1)
            self.y_nosecone = Function(
                lambda x: self.base_radius * np.power(x / self.length, self.power)
            )
        else:  # pragma: no cover
            raise ValueError(
                f"Nose Cone kind '{self.kind}' not found, "
                + "please use one of the following Nose Cone kinds:"
                + '\n\t"conical"'
                + '\n\t"ogive"'
                + '\n\t"lvhaack"'
                + '\n\t"tangent"'
                + '\n\t"vonkarman"'
                + '\n\t"elliptical"'
                + '\n\t"powerseries"'
                + '\n\t"parabolic"\n'
            )

        self.evaluate_center_of_pressure()
        self.evaluate_geometrical_parameters()
        self.evaluate_nose_shape()

    @property
    def bluffness(self):
        return self._bluffness

    @bluffness.setter
    def bluffness(self, value):
        # prevents from setting bluffness on "powerseries" nose cones
        if self.kind == "powerseries":
            # Checks if bluffness is not being used
            if (value is not None) and (value != 0):
                raise ValueError(
                    "Parameter 'bluffness' must be None or 0 when using a nose cone kind 'powerseries'."
                )
        if value is not None and not 0 <= value <= 1:  # pragma: no cover
            raise ValueError(
                f"Bluffness ratio of {value} is out of range. "
                "It must be between 0 and 1."
            )
        self._bluffness = value
        self.evaluate_nose_shape()

    def evaluate_geometrical_parameters(self):
        """Calculates and saves nose cone's radius ratio.

        Returns
        -------
        None
        """

        # If base radius is not given, the ratio between base radius and
        # rocket radius is assumed as 1, meaning that the nose cone has the
        # same radius as the rocket
        if self.base_radius is None and self.rocket_radius is not None:
            self.radius_ratio = 1
            self.base_radius = self.rocket_radius
        elif self.base_radius is not None and self.rocket_radius is None:
            self.radius_ratio = 1
            self.rocket_radius = self.base_radius
        # If base radius is given, the ratio between base radius and rocket
        # radius is calculated
        elif self.base_radius is not None and self.rocket_radius is not None:
            self.radius_ratio = self.base_radius / self.rocket_radius
        else:
            raise ValueError(
                "Either base radius or rocket radius must be given to "
                "calculate the nose cone radius ratio."
            )

        self.fineness_ratio = self.length / (2 * self.base_radius)

    def evaluate_nose_shape(self):  # pylint: disable=too-many-statements
        """Calculates and saves nose cone's shape as lists and re-evaluates the
        nose cone's length for a given bluffness ratio. The shape is saved as
        two vectors, one for the x coordinates and one for the y coordinates.

        Returns
        -------
        None
        """
        number_of_points = 127
        density_modifier = 3  # increase density of points to improve accuracy

        def find_x_intercept(x):
            # find the tangential intersection point between the circle and nosec curve
            return x + self.y_nosecone(x) * self.y_nosecone.differentiate_complex_step(
                x
            )

        # Calculate a function to find the radius of the nosecone curve
        def find_radius(x):
            return (self.y_nosecone(x) ** 2 + (x - find_x_intercept(x)) ** 2) ** 0.5

        # Check bluffness circle and choose whether to use it or not
        if self.bluffness is None or self.bluffness == 0:
            # Set up parameters to continue without bluffness
            r_circle, circle_center, x_init = 0, 0, 0
        else:
            # Calculate circle radius
            r_circle = self.bluffness * self.base_radius
            if self.kind == "elliptical":

                def test_circle(x):
                    # set up a circle at the starting position to test bluffness
                    return np.sqrt(r_circle**2 - (x - r_circle) ** 2)

                # Check if bluffness circle is too small
                if test_circle(1e-03) <= self.y_nosecone(1e-03):
                    # Raise a warning
                    warnings.warn(
                        "WARNING: The chosen bluffness ratio is too small for "
                        "the selected nose cone category, thereby the effective "
                        "bluffness will be 0."
                    )
                    # Set up parameters to continue without bluffness
                    r_circle, circle_center, x_init = 0, 0, 0
                else:
                    # Find the intersection point between circle and nosecone curve
                    x_init = fsolve(lambda x: find_radius(x[0]) - r_circle, r_circle)[0]
                    circle_center = find_x_intercept(x_init)
            else:
                # Find the intersection point between circle and nosecone curve
                x_init = fsolve(lambda x: find_radius(x[0]) - r_circle, r_circle)[0]
                circle_center = find_x_intercept(x_init)

        # Calculate a function to create the circle at the correct position
        def create_circle(x):
            return abs(r_circle**2 - (x - circle_center) ** 2) ** 0.5

        # Define a function for the final shape of the curve with a circle at the tip
        def final_shape(x):
            return self.y_nosecone(x) if x >= x_init else create_circle(x)

        # Vectorize the final_shape function
        final_shape_vec = np.vectorize(final_shape)

        # Create the vectors X and Y with the points of the curve
        nosecone_x = (self.length - (circle_center - r_circle)) * (
            np.linspace(0, 1, number_of_points) ** density_modifier
        )
        nosecone_y = final_shape_vec(nosecone_x + (circle_center - r_circle))

        # Evaluate final geometry parameters
        self.shape_vec = [nosecone_x, nosecone_y]
        if abs(nosecone_x[-1] - self.length) >= 0.001:  # 1 millimeter
            self._length = nosecone_x[-1]
            print(
                "Due to the chosen bluffness ratio, the nose "
                f"cone length was reduced to {self.length} m."
            )
        self.fineness_ratio = self.length / (2 * self.base_radius)

    def evaluate_lift_coefficient(self):
        """Calculates and returns nose cone's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves its lift coefficient derivative.

        Returns
        -------
        None
        """
        # Calculate clalpha
        self.clalpha = Function(
            lambda mach: 2 * self.radius_ratio**2,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

    def evaluate_k(self):
        """Updates the self.k attribute used to compute the center of
        pressure when using "powerseries" nose cones.

        Returns
        -------
        None
        """
        if self.kind == "powerseries":
            self.k = (2 * self.power) / ((2 * self.power) + 1)

    def evaluate_center_of_pressure(self):
        """Calculates and returns the center of pressure of the nose cone in
        local coordinates. The center of pressure position is saved and stored
        as a tuple. Local coordinate origin is found at the tip of the nose
        cone.

        Returns
        -------
        self.cp : tuple
            Tuple containing cpx, cpy, cpz.
        """

        self.cpz = self.k * self.length
        self.cpy = 0
        self.cpx = 0
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def draw(self, *, filename=None):
        """Draw the nosecone shape along with some important information,
        including the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        self.plots.draw(filename=filename)

    def info(self):
        """Prints and plots summarized information of the nose cone.

        Return
        ------
        None
        """
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        """Prints and plots all the available information of the nose cone.

        Returns
        -------
        None
        """
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = {
            "_length": self._length,
            "_kind": self._kind,
            "_base_radius": self._base_radius,
            "_bluffness": self._bluffness,
            "_rocket_radius": self._rocket_radius,
            "_power": self._power,
            "name": self.name,
        }
        if kwargs.get("include_outputs", False):
            clalpha = self.clalpha
            cl = self.cl
            if kwargs.get("discretize", False):
                clalpha = clalpha.set_discrete(0, 4, 50)
                cl = cl.set_discrete(
                    (-np.pi / 6, 0), (np.pi / 6, 2), (10, 10), mutate_self=False
                )
            data["cp"] = self.cp
            data["clalpha"] = clalpha
            data["cl"] = cl

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            length=data["_length"],
            kind=data["_kind"],
            base_radius=data["_base_radius"],
            bluffness=data["_bluffness"],
            rocket_radius=data["_rocket_radius"],
            power=data["_power"],
            name=data["name"],
        )
