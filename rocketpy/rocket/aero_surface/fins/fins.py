from rocketpy.rocket.aero_surface.fins._base_fin import _BaseFin


# Still abstract: each fin shape class provides the center of pressure
class Fins(_BaseFin):  # pylint: disable=abstract-method
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
    Fins.rocket_diameter : float
        Reference diameter of the rocket. Has units of length and is given
        in meters.
    Fins.reference_area : float
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
    Fins.cN : AeroCoefficient
        Normal force coefficient, the force in the pitch plane.
    Fins.cY : AeroCoefficient
        Side force coefficient, the force in the yaw plane.
    Fins.cA : AeroCoefficient
        Axial force coefficient, the force along the rocket's axis.
    Fins.cm : AeroCoefficient
        Pitching moment coefficient.
    Fins.cn : AeroCoefficient
        Yawing moment coefficient.
    Fins.cl : AeroCoefficient
        Roll moment coefficient, from the cant angle and the roll damping.
    Fins.cL : Function
        Lift coefficient, the force perpendicular to the airflow.
    Fins.cD : Function
        Drag coefficient, the force along the airflow.
    Fins.cQ : Function
        Crosswind coefficient, the side force relative to the airflow.
    Fins.cN_alpha : AeroCoefficient
        Slope of ``cN`` with the angle of attack, as a function of Mach.
        Has units of 1/rad.
    Fins.cY_beta : AeroCoefficient
        Slope of ``cY`` with the sideslip angle, as a function of Mach.
        Has units of 1/rad.
    Fins.cm_alpha : AeroCoefficient
        Slope of ``cm`` with the angle of attack, as a function of Mach.
        Has units of 1/rad.
    Fins.cn_beta : AeroCoefficient
        Slope of ``cn`` with the sideslip angle, as a function of Mach.
        Has units of 1/rad.
    Fins.cl_0 : AeroCoefficient
        Roll moment coefficient at zero roll rate, as a function of Mach.
    Fins.cl_p : AeroCoefficient
        Slope of ``cl`` with the reduced roll rate (roll damping), as a
        function of Mach.
    Fins.clalpha : float
        Normal-force coefficient slope. Has units of 1/rad.
    Fins.roll_parameters : list
        List containing the roll moment forcing coefficient, the roll moment
        damping coefficient (both for the fins at zero cant) and the cant angle
        in radians.
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
            A positive cant angle gives a negative roll moment about the
            rocket's axis (see :ref:`individual_fins`).
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
        """
        super().__init__(
            name=name,
            rocket_radius=rocket_radius,
            root_chord=root_chord,
            span=span,
            airfoil=airfoil,
            cant_angle=cant_angle,
        )

        # Store values
        self._n = n

    @property
    def n(self):
        return self._n

    @property
    def _roll_cant_angle_rad(self):
        """Cant angle, in radians, with the sign used by the roll coefficients
        ``cl_0`` and ``cl``.

        The roll forcing of a fin set uses the opposite sign convention to the
        individual fins, whose roll moment comes from the force at their center
        of pressure, so the cant angle is flipped here. The angle the user gave
        is kept unchanged as ``cant_angle``, so it reads back, saves and loads
        as given.

        Returns
        -------
        float
            Cant angle in radians, with its sign flipped.
        """
        return -self.cant_angle_rad

    @n.setter
    def n(self, value):
        self._n = value
        self._update_geometry_chain()

    def evaluate_lift_coefficient(self):
        """Calculates and returns the fin set's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves the lift coefficient derivative
        for a single fin and the lift coefficient derivative for
        a number of n fins corrected for Fin-Body interference.
        """
        self.evaluate_single_fin_lift_coefficient()

        # Normal-force coefficient derivative for n fins corrected with Fin-Body interference
        self.clalpha_multiple_fins = (
            self.fin_num_correction(self.n)
            * self.lift_interference_factor
            * self.clalpha_single_fin
        )  # Function of mach number
        self.clalpha_multiple_fins.set_inputs("Mach")
        self.clalpha_multiple_fins.set_outputs(
            f"Normal-force coefficient derivative for {self.n:.0f} fins"
        )

        self.clalpha = self.clalpha_multiple_fins

        return self.clalpha

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
        # Scaled by n, not fin_num_correction(n): every canted fin adds the same
        # roll moment, while their normal forces partly cancel in pitch and yaw
        clf_delta = (
            self.roll_forcing_interference_factor
            * self.n
            * (self.Yma + self.rocket_radius)
            * self.clalpha_single_fin
            / self.reference_length
        )  # Function of mach number
        clf_delta.set_inputs("Mach")
        clf_delta.set_outputs("Roll moment forcing coefficient derivative")
        clf_delta.set_title(
            "Roll moment forcing coefficient derivative vs. Mach number"
        )
        # Damping of the fins at zero cant; the cos(cant) factor is applied
        # when the moment is evaluated, so a new cant angle needs no rebuild
        cld_omega = -(
            2
            * self.roll_damping_interference_factor
            * self.n
            * self.clalpha_single_fin
            * self.roll_geometrical_constant
            / (self.reference_area * self.reference_length**2)
        )  # Function of mach number
        cld_omega.set_inputs("Mach")
        cld_omega.set_outputs("Roll moment damping coefficient derivative")
        cld_omega.set_title(
            "Roll moment damping coefficient derivative vs. Mach number"
        )
        self._clf_delta, self._cld_omega = clf_delta, cld_omega
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

    def to_dict(self, include_outputs=False, **kwargs):
        data = super().to_dict(include_outputs=include_outputs, **kwargs)
        data["n"] = self.n
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
        """
        self.plots.draw(filename=filename)
