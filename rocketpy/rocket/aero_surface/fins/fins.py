import numpy as np

from rocketpy.mathutils.function import Function

from ..aero_surface import AeroSurface


class Fins(AeroSurface):
    """Abstract class that holds common methods for the fin classes.
    Cannot be instantiated.

    Note
    ----
    Local coordinate system:
        - Origin located at the top of the root chord.
        - Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).
        - Y axis perpendicular to the Z axis, in the span direction, positive upwards.
        - X axis completes the right-handed coordinate system.

    Attributes
    ----------
    Fins.n : int
        Number of fins in fin set.
    Fins.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization,
        in meters.
    Fins.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    Fins.cant_angle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    Fins.changing_attribute_dict : dict
        Dictionary that stores the name and the values of the attributes that
        may be changed during a simulation. Useful for control systems.
    Fins.cant_angle_rad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    Fins.root_chord : float
        Fin root chord in meters.
    Fins.tip_chord : float
        Fin tip chord in meters.
    Fins.span : float
        Fin span in meters.
    Fins.name : string
        Name of fin set.
    Fins.sweep_length : float
        Fins sweep length in meters. By sweep length, understand the axial
        distance between the fin root leading edge and the fin tip leading edge
        measured parallel to the rocket centerline.
    Fins.sweep_angle : float
        Fins sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    Fins.d : float
        Reference diameter of the rocket. Has units of length and is given
        in meters.
    Fins.ref_area : float
        Reference area of the rocket.
    Fins.Af : float
        Area of the longitudinal section of each fin in the set.
    Fins.AR : float
        Aspect ratio of each fin in the set.
    Fins.gamma_c : float
        Fin mid-chord sweep angle.
    Fins.Yma : float
        Span wise position of the mean aerodynamic chord.
    Fins.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    Fins.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    Fins.lift_interference_factor : float
        Factor of Fin-Body interference in the lift coefficient.
    Fins.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of
        pressure. Has units of length and is given in meters.
    Fins.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and
        is given in meters.
    Fins.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and
        is given in meters.
    Fins.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and
        is given in meters.
    Fins.cl : Function
        Function which defines the lift coefficient as a function of the angle
        of attack and the Mach number. Takes as input the angle of attack in
        radians and the Mach number. Returns the lift coefficient.
    Fins.clalpha : float
        Lift coefficient slope. Has units of 1/rad.
    Fins.roll_parameters : list
        List containing the roll moment lift coefficient, the roll moment
        damping coefficient and the cant angle in radians.
    """

    def __init__(
        self,
        n,
        root_chord,
        span,
        rocket_radius,
        cant_angle=0,
        airfoil=None,
        name="Fins",
    ):
        """Initialize Fins class.

        Parameters
        ----------
        n : int
            Number of fins, must be larger than 2.
        root_chord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        rocket_radius : int, float
            Reference rocket radius used for lift coefficient normalization.
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files can contain a single line
            header and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".
        name : str
            Name of fin set.

        Returns
        -------
        None
        """
        # Compute auxiliary geometrical parameters
        d = 2 * rocket_radius
        ref_area = np.pi * rocket_radius**2  # Reference area

        super().__init__(name, ref_area, d)

        # Store values
        self._n = n
        self._rocket_radius = rocket_radius
        self._airfoil = airfoil
        self._cant_angle = cant_angle
        self._root_chord = root_chord
        self._span = span
        self.name = name
        self.d = d
        self.ref_area = ref_area  # Reference area

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def root_chord(self):
        return self._root_chord

    @root_chord.setter
    def root_chord(self, value):
        self._root_chord = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, value):
        self._span = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def cant_angle(self):
        return self._cant_angle

    @cant_angle.setter
    def cant_angle(self, value):
        self._cant_angle = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def airfoil(self):
        return self._airfoil

    @airfoil.setter
    def airfoil(self, value):
        self._airfoil = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    def evaluate_lift_coefficient(self):
        """Calculates and returns the fin set's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves the lift coefficient derivative
        for a single fin and the lift coefficient derivative for
        a number of n fins corrected for Fin-Body interference.

        Returns
        -------
        None
        """
        if not self.airfoil:
            # Defines clalpha2D as 2*pi for planar fins
            clalpha2D_incompressible = 2 * np.pi
        else:
            # Defines clalpha2D as the derivative of the lift coefficient curve
            # for the specific airfoil
            self.airfoil_cl = Function(
                self.airfoil[0],
                interpolation="linear",
            )

            # Differentiating at alpha = 0 to get cl_alpha
            clalpha2D_incompressible = self.airfoil_cl.differentiate_complex_step(
                x=1e-3, dx=1e-3
            )

            # Convert to radians if needed
            if self.airfoil[1] == "degrees":
                clalpha2D_incompressible *= 180 / np.pi

        # Correcting for compressible flow (apply Prandtl-Glauert correction)
        clalpha2D = Function(lambda mach: clalpha2D_incompressible / self._beta(mach))

        # Diederich's Planform Correlation Parameter
        planform_correlation_parameter = (
            2 * np.pi * self.AR / (clalpha2D * np.cos(self.gamma_c))
        )

        # Lift coefficient derivative for a single fin
        def lift_source(mach):
            return (
                clalpha2D(mach)
                * planform_correlation_parameter(mach)
                * (self.Af / self.ref_area)
                * np.cos(self.gamma_c)
            ) / (
                2
                + planform_correlation_parameter(mach)
                * np.sqrt(1 + (2 / planform_correlation_parameter(mach)) ** 2)
            )

        self.clalpha_single_fin = Function(
            lift_source,
            "Mach",
            "Lift coefficient derivative for a single fin",
        )

        # Lift coefficient derivative for n fins corrected with Fin-Body interference
        self.clalpha_multiple_fins = (
            self.lift_interference_factor
            * self.fin_num_correction(self.n)
            * self.clalpha_single_fin
        )  # Function of mach number
        self.clalpha_multiple_fins.set_inputs("Mach")
        self.clalpha_multiple_fins.set_outputs(
            f"Lift coefficient derivative for {self.n:.0f} fins"
        )
        self.clalpha = self.clalpha_multiple_fins

        # Cl = clalpha * alpha
        self.cl = Function(
            lambda alpha, mach: alpha * self.clalpha_multiple_fins(mach),
            ["Alpha (rad)", "Mach"],
            "Lift coefficient",
        )

        return self.cl

    def evaluate_roll_parameters(self):
        """Calculates and returns the fin set's roll coefficients.
        The roll coefficients are saved in a list.

        Returns
        -------
        self.roll_parameters : list
            List containing the roll moment lift coefficient, the
            roll moment damping coefficient and the cant angle in
            radians
        """

        self.cant_angle_rad = np.radians(self.cant_angle)

        clf_delta = (
            self.roll_forcing_interference_factor
            * self.n
            * (self.Yma + self.rocket_radius)
            * self.clalpha_single_fin
            / self.d
        )  # Function of mach number
        clf_delta.set_inputs("Mach")
        clf_delta.set_outputs("Roll moment forcing coefficient derivative")
        cld_omega = (
            2
            * self.roll_damping_interference_factor
            * self.n
            * self.clalpha_single_fin
            * np.cos(self.cant_angle_rad)
            * self.roll_geometrical_constant
            / (self.ref_area * self.d**2)
        )  # Function of mach number
        cld_omega.set_inputs("Mach")
        cld_omega.set_outputs("Roll moment damping coefficient derivative")
        self.roll_parameters = [clf_delta, cld_omega, self.cant_angle_rad]
        return self.roll_parameters

    @staticmethod
    def fin_num_correction(n):
        """Calculates a correction factor for the lift coefficient of multiple
        fins.
        The specifics  values are documented at:
        Niskanen, S. (2013). “OpenRocket technical documentation”.
        In: Development of an Open Source model rocket simulation software.

        Parameters
        ----------
        n : int
            Number of fins.

        Returns
        -------
        Corrector factor : int
            Factor that accounts for the number of fins.
        """
        corrector_factor = [2.37, 2.74, 2.99, 3.24]
        if 5 <= n <= 8:
            return corrector_factor[n - 5]
        else:
            return n / 2

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        *args,
    ):  # pylint: disable=arguments-differ
        """Computes the forces and moments acting on the aerodynamic surface.

        Parameters
        ----------
        stream_velocity : tuple of float
            The velocity of the airflow relative to the surface.
        stream_speed : float
            The magnitude of the airflow speed.
        stream_mach : float
            The Mach number of the airflow.
        rho : float
            Air density.
        cp : Vector
            Center of pressure coordinates in the body frame.
        omega: tuple[float, float, float]
            Tuple containing angular velocities around the x, y, z axes.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """

        R1, R2, R3, M1, M2, _ = super().compute_forces_and_moments(
            stream_velocity,
            stream_speed,
            stream_mach,
            rho,
            cp,
        )
        clf_delta, cld_omega, cant_angle_rad = self.roll_parameters
        M3_forcing = (
            (1 / 2 * rho * stream_speed**2)
            * self.reference_area
            * self.reference_length
            * clf_delta.get_value_opt(stream_mach)
            * cant_angle_rad
        )
        M3_damping = (
            (1 / 2 * rho * stream_speed)
            * self.reference_area
            * (self.reference_length) ** 2
            * cld_omega.get_value_opt(stream_mach)
            * omega[2]
            / 2
        )
        M3 = M3_forcing - M3_damping
        return R1, R2, R3, M1, M2, M3

    def to_dict(self, **kwargs):
        if self.airfoil:
            if kwargs.get("discretize", False):
                lower = -np.pi / 6 if self.airfoil[1] == "radians" else -30
                upper = np.pi / 6 if self.airfoil[1] == "radians" else 30
                airfoil = (
                    self.airfoil_cl.set_discrete(lower, upper, 50, mutate_self=False),
                    self.airfoil[1],
                )
            else:
                airfoil = (self.airfoil_cl, self.airfoil[1]) if self.airfoil else None
        else:
            airfoil = None
        data = {
            "n": self.n,
            "root_chord": self.root_chord,
            "span": self.span,
            "rocket_radius": self.rocket_radius,
            "cant_angle": self.cant_angle,
            "airfoil": airfoil,
            "name": self.name,
        }

        if kwargs.get("include_outputs", False):
            cl = self.cl
            if kwargs.get("discretize", False):
                cl = cl.set_discrete(
                    (-np.pi / 6, 0), (np.pi / 6, 2), (10, 10), mutate_self=False
                )

            data.update(
                {
                    "cp": self.cp,
                    "cl": cl,
                    "roll_parameters": self.roll_parameters,
                    "d": self.d,
                    "ref_area": self.ref_area,
                }
            )

        return data

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information, including
        the center line, the quarter line and the center of pressure position.

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
