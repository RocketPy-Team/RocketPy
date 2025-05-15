from rocketpy.mathutils import Function
from rocketpy.plots.aero_surface_plots import _LinearGenericSurfacePlots
from rocketpy.prints.aero_surface_prints import _LinearGenericSurfacePrints
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface


class LinearGenericSurface(GenericSurface):
    """Class that defines a generic linear aerodynamic surface. This class is
    used to define aerodynamic surfaces that have aerodynamic coefficients
    defined as linear functions of the coefficients derivatives."""

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Generic Linear Surface",
    ):
        """Create a generic linear aerodynamic surface, defined by its
        aerodynamic coefficients derivatives. This surface is used to model any
        aerodynamic surface that does not fit the predefined classes.

        Important
        ---------
        All the aerodynamic coefficients can be input as callable functions of
        angle of attack, angle of sideslip, Mach number, Reynolds number,
        pitch rate, yaw rate and roll rate. For CSV files, the header must
        contain at least one of the following: "alpha", "beta", "mach",
        "reynolds", "pitch_rate", "yaw_rate" and "roll_rate".

        See Also
        --------
        :ref:`genericsurfaces`.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface. Has the unit of meters.
            Commonly defined as the rocket's diameter.
        coefficients: dict, optional
            List of coefficients. If a coefficient is omitted, it is set to 0.
            The valid coefficients are:\n
            cL_0: callable, str, optional
                Coefficient of lift at zero angle of attack. Default is 0.\n
            cL_alpha: callable, str, optional
                Coefficient of lift derivative with respect to angle of attack.
                Default is 0.\n
            cL_beta: callable, str, optional
                Coefficient of lift derivative with respect to sideslip angle.
                Default is 0.\n
            cL_p: callable, str, optional
                Coefficient of lift derivative with respect to roll rate.
                Default is 0.\n
            cL_q: callable, str, optional
                Coefficient of lift derivative with respect to pitch rate.
                Default is 0.\n
            cL_r: callable, str, optional
                Coefficient of lift derivative with respect to yaw rate.
                Default is 0.\n
            cQ_0: callable, str, optional
                Coefficient of side force at zero angle of attack.
                Default is 0.\n
            cQ_alpha: callable, str, optional
                Coefficient of side force derivative with respect to angle of
                attack. Default is 0.\n
            cQ_beta: callable, str, optional
                Coefficient of side force derivative with respect to sideslip
                angle. Default is 0.\n
            cQ_p: callable, str, optional
                Coefficient of side force derivative with respect to roll rate.
                Default is 0.\n
            cQ_q: callable, str, optional
                Coefficient of side force derivative with respect to pitch rate.
                Default is 0.\n
            cQ_r: callable, str, optional
                Coefficient of side force derivative with respect to yaw rate.
                Default is 0.\n
            cD_0: callable, str, optional
                Coefficient of drag at zero angle of attack. Default is 0.\n
            cD_alpha: callable, str, optional
                Coefficient of drag derivative with respect to angle of attack.
                Default is 0.\n
            cD_beta: callable, str, optional
                Coefficient of drag derivative with respect to sideslip angle.
                Default is 0.\n
            cD_p: callable, str, optional
                Coefficient of drag derivative with respect to roll rate.
                Default is 0.\n
            cD_q: callable, str, optional
                Coefficient of drag derivative with respect to pitch rate.
                Default is 0.\n
            cD_r: callable, str, optional
                Coefficient of drag derivative with respect to yaw rate.
                Default is 0.\n
            cm_0: callable, str, optional
                Coefficient of pitch moment at zero angle of attack.
                Default is 0.\n
            cm_alpha: callable, str, optional
                Coefficient of pitch moment derivative with respect to angle of
                attack. Default is 0.\n
            cm_beta: callable, str, optional
                Coefficient of pitch moment derivative with respect to sideslip
                angle. Default is 0.\n
            cm_p: callable, str, optional
                Coefficient of pitch moment derivative with respect to roll rate.
                Default is 0.\n
            cm_q: callable, str, optional
                Coefficient of pitch moment derivative with respect to pitch rate.
                Default is 0.\n
            cm_r: callable, str, optional
                Coefficient of pitch moment derivative with respect to yaw rate.
                Default is 0.\n
            cn_0: callable, str, optional
                Coefficient of yaw moment at zero angle of attack.
                Default is 0.\n
            cn_alpha: callable, str, optional
                Coefficient of yaw moment derivative with respect to angle of
                attack. Default is 0.\n
            cn_beta: callable, str, optional
                Coefficient of yaw moment derivative with respect to sideslip angle.
                Default is 0.\n
            cn_p: callable, str, optional
                Coefficient of yaw moment derivative with respect to roll rate.
                Default is 0.\n
            cn_q: callable, str, optional
                Coefficient of yaw moment derivative with respect to pitch rate.
                Default is 0.\n
            cn_r: callable, str, optional
                Coefficient of yaw moment derivative with respect to yaw rate.
                Default is 0.\n
            cl_0: callable, str, optional
                Coefficient of roll moment at zero angle of attack.
                Default is 0.\n
            cl_alpha: callable, str, optional
                Coefficient of roll moment derivative with respect to angle of
                attack. Default is 0.\n
            cl_beta: callable, str, optional
                Coefficient of roll moment derivative with respect to sideslip
                angle. Default is 0.\n
            cl_p: callable, str, optional
                Coefficient of roll moment derivative with respect to roll rate.
                Default is 0.\n
            cl_q: callable, str, optional
                Coefficient of roll moment derivative with respect to pitch rate.
                Default is 0.\n
            cl_r: callable, str, optional
                Coefficient of roll moment derivative with respect to yaw rate.
                Default is 0.\n
        center_of_pressure : tuple, optional
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str
            Name of the aerodynamic surface. Default is 'GenericSurface'.
        """

        super().__init__(
            reference_area=reference_area,
            reference_length=reference_length,
            coefficients=coefficients,
            center_of_pressure=center_of_pressure,
            name=name,
        )

        self.compute_all_coefficients()

        self.prints = _LinearGenericSurfacePrints(self)
        self.plots = _LinearGenericSurfacePlots(self)

    def _get_default_coefficients(self):
        """Returns default coefficients

        Returns
        -------
        default_coefficients: dict
            Dictionary whose keys are the coefficients names and keys
            are the default values.
        """
        default_coefficients = {
            "cL_0": 0,
            "cL_alpha": 0,
            "cL_beta": 0,
            "cL_p": 0,
            "cL_q": 0,
            "cL_r": 0,
            "cQ_0": 0,
            "cQ_alpha": 0,
            "cQ_beta": 0,
            "cQ_p": 0,
            "cQ_q": 0,
            "cQ_r": 0,
            "cD_0": 0,
            "cD_alpha": 0,
            "cD_beta": 0,
            "cD_p": 0,
            "cD_q": 0,
            "cD_r": 0,
            "cm_0": 0,
            "cm_alpha": 0,
            "cm_beta": 0,
            "cm_p": 0,
            "cm_q": 0,
            "cm_r": 0,
            "cn_0": 0,
            "cn_alpha": 0,
            "cn_beta": 0,
            "cn_p": 0,
            "cn_q": 0,
            "cn_r": 0,
            "cl_0": 0,
            "cl_alpha": 0,
            "cl_beta": 0,
            "cl_p": 0,
            "cl_q": 0,
            "cl_r": 0,
        }
        return default_coefficients

    def compute_forcing_coefficient(self, c_0, c_alpha, c_beta):
        """Compute the forcing coefficient from the derivatives of the
        aerodynamic coefficients."""

        def total_coefficient(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ):
            return (
                c_0(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                + c_alpha(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * alpha
                + c_beta(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * beta
            )

        return Function(
            total_coefficient,
            [
                "alpha",
                "beta",
                "mach",
                "reynolds",
                "pitch_rate",
                "yaw_rate",
                "roll_rate",
            ],
            ["coefficient"],
        )

    def compute_damping_coefficient(self, c_p, c_q, c_r):
        """Compute the damping coefficient from the derivatives of the
        aerodynamic coefficients."""

        def total_coefficient(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ):
            return (
                c_p(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * roll_rate
                + c_q(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * pitch_rate
                + c_r(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * yaw_rate
            )

        return Function(
            total_coefficient,
            [
                "alpha",
                "beta",
                "mach",
                "reynolds",
                "pitch_rate",
                "yaw_rate",
                "roll_rate",
            ],
            ["coefficient"],
        )

    def compute_all_coefficients(self):
        """Compute all the aerodynamic coefficients from the derivatives."""
        # pylint: disable=invalid-name
        self.cLf = self.compute_forcing_coefficient(
            self.cL_0, self.cL_alpha, self.cL_beta
        )
        self.cLd = self.compute_damping_coefficient(self.cL_p, self.cL_q, self.cL_r)

        self.cQf = self.compute_forcing_coefficient(
            self.cQ_0, self.cQ_alpha, self.cQ_beta
        )
        self.cQd = self.compute_damping_coefficient(self.cQ_p, self.cQ_q, self.cQ_r)

        self.cDf = self.compute_forcing_coefficient(
            self.cD_0, self.cD_alpha, self.cD_beta
        )
        self.cDd = self.compute_damping_coefficient(self.cD_p, self.cD_q, self.cD_r)

        self.cmf = self.compute_forcing_coefficient(
            self.cm_0, self.cm_alpha, self.cm_beta
        )
        self.cmd = self.compute_damping_coefficient(self.cm_p, self.cm_q, self.cm_r)

        self.cnf = self.compute_forcing_coefficient(
            self.cn_0, self.cn_alpha, self.cn_beta
        )
        self.cnd = self.compute_damping_coefficient(self.cn_p, self.cn_q, self.cn_r)

        self.clf = self.compute_forcing_coefficient(
            self.cl_0, self.cl_alpha, self.cl_beta
        )
        self.cld = self.compute_damping_coefficient(self.cl_p, self.cl_q, self.cl_r)

    def _compute_from_coefficients(
        self,
        rho,
        stream_speed,
        alpha,
        beta,
        mach,
        reynolds,
        pitch_rate,
        yaw_rate,
        roll_rate,
    ):
        """Compute the aerodynamic forces and moments from the aerodynamic
        coefficients.

        Parameters
        ----------
        rho : float
            Air density.
        stream_speed : float
            Magnitude of the airflow speed.
        alpha : float
            Angle of attack in radians.
        beta : float
            Sideslip angle in radians.
        mach : float
            Mach number.
        reynolds : float
            Reynolds number.
        pitch_rate : float
            Pitch rate in radians per second.
        yaw_rate : float
            Yaw rate in radians per second.
        roll_rate : float
            Roll rate in radians per second.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        # Precompute common values
        dyn_pressure_area = 0.5 * rho * stream_speed**2 * self.reference_area
        dyn_pressure_area_damping = (
            0.5 * rho * stream_speed * self.reference_area * self.reference_length / 2
        )
        dyn_pressure_area_length = dyn_pressure_area * self.reference_length
        dyn_pressure_area_length_damping = (
            0.5
            * rho
            * stream_speed
            * self.reference_area
            * self.reference_length**2
            / 2
        )

        # Compute aerodynamic forces
        lift = dyn_pressure_area * self.cLf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) + dyn_pressure_area_damping * self.cLd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        side = dyn_pressure_area * self.cQf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) + dyn_pressure_area_damping * self.cQd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        drag = dyn_pressure_area * self.cDf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) + dyn_pressure_area_damping * self.cDd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        # Compute aerodynamic moments
        pitch = dyn_pressure_area_length * self.cmf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) + dyn_pressure_area_length_damping * self.cmd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        yaw = dyn_pressure_area_length * self.cnf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) + dyn_pressure_area_length_damping * self.cnd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        roll = dyn_pressure_area_length * self.clf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) + dyn_pressure_area_length_damping * self.cld(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        return lift, side, drag, pitch, yaw, roll
