from rocketpy.mathutils import Function
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface


class LinearGenericSurface(GenericSurface):
    """Class that defines a generic linear aerodynamic surface. This class is
    used to define aerodynamic surfaces that have aerodynamic coefficients
    defined as linear functions of the coefficients derivatives."""

    def __init__(
        self,
        reference_area,
        reference_length,
        cl_lift_0=0,
        cl_lift_alpha=0,
        cl_lift_beta=0,
        cl_lift_p=0,
        cl_lift_q=0,
        cl_lift_r=0,
        cq_pitch_0=0,
        cq_pitch_alpha=0,
        cq_pitch_beta=0,
        cq_pitch_p=0,
        cq_pitch_q=0,
        cq_pitch_r=0,
        cd_drag_0=0,
        cd_drag_alpha=0,
        cd_drag_beta=0,
        cd_drag_p=0,
        cd_drag_q=0,
        cd_drag_r=0,
        cm_0=0,
        cm_alpha=0,
        cm_beta=0,
        cm_p=0,
        cm_q=0,
        cm_r=0,
        cn_0=0,
        cn_alpha=0,
        cn_beta=0,
        cn_p=0,
        cn_q=0,
        cn_r=0,
        cl_0=0,
        cl_alpha=0,
        cl_beta=0,
        cl_p=0,
        cl_q=0,
        cl_r=0,
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
        For more information on how to create a custom aerodynamic surface,
        check TODO: ADD LINK TO DOCUMENTATION

        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface. Has the unit of meters.
            Commonly defined as the rocket's diameter.
        cl_lift_0 : callable, str, optional
            Coefficient of lift at zero angle of attack. Default is 0.
        cl_lift_alpha : callable, str, optional
            Coefficient of lift derivative with respect to angle of attack.
            Default is 0.
        cl_lift_beta : callable, str, optional
            Coefficient of lift derivative with respect to sideslip angle.
            Default is 0.
        cl_lift_p : callable, str, optional
            Coefficient of lift derivative with respect to roll rate.
            Default is 0.
        cl_lift_q : callable, str, optional
            Coefficient of lift derivative with respect to pitch rate.
            Default is 0.
        cl_lift_r : callable, str, optional
            Coefficient of lift derivative with respect to yaw rate.
            Default is 0.
        cq_pitch_0 : callable, str, optional
            Coefficient of pitch moment at zero angle of attack.
            Default is 0.
        cq_pitch_alpha : callable, str, optional
            Coefficient of pitch moment derivative with respect to angle of
            attack. Default is 0.
        cq_pitch_beta : callable, str, optional
            Coefficient of pitch moment derivative with respect to sideslip
            angle. Default is 0.
        cq_pitch_p : callable, str, optional
            Coefficient of pitch moment derivative with respect to roll rate.
            Default is 0.
        cq_pitch_q : callable, str, optional
            Coefficient of pitch moment derivative with respect to pitch rate.
            Default is 0.
        cq_pitch_r : callable, str, optional
            Coefficient of pitch moment derivative with respect to yaw rate.
            Default is 0.
        cd_drag_0 : callable, str, optional
            Coefficient of drag at zero angle of attack. Default is 0.
        cd_drag_alpha : callable, str, optional
            Coefficient of drag derivative with respect to angle of attack.
            Default is 0.
        cd_drag_beta : callable, str, optional
            Coefficient of drag derivative with respect to sideslip angle.
            Default is 0.
        cd_drag_p : callable, str, optional
            Coefficient of drag derivative with respect to roll rate.
            Default is 0.
        cd_drag_q : callable, str, optional
            Coefficient of drag derivative with respect to pitch rate.
            Default is 0.
        cd_drag_r : callable, str, optional
            Coefficient of drag derivative with respect to yaw rate.
            Default is 0.
        cm_0 : callable, str, optional
            Coefficient of pitch moment at zero angle of attack.
            Default is 0.
        cm_alpha : callable, str, optional
            Coefficient of pitch moment derivative with respect to angle of
            attack. Default is 0.
        cm_beta : callable, str, optional
            Coefficient of pitch moment derivative with respect to sideslip
            angle. Default is 0.
        cm_p : callable, str, optional
            Coefficient of pitch moment derivative with respect to roll rate.
            Default is 0.
        cm_q : callable, str, optional
            Coefficient of pitch moment derivative with respect to pitch rate.
            Default is 0.
        cm_r : callable, str, optional
            Coefficient of pitch moment derivative with respect to yaw rate.
            Default is 0.
        cn_0 : callable, str, optional
            Coefficient of yaw moment at zero angle of attack.
            Default is 0.
        cn_alpha : callable, str, optional
            Coefficient of yaw moment derivative with respect to angle of
            attack. Default is 0.
        cn_beta : callable, str, optional
            Coefficient of yaw moment derivative with respect to sideslip angle.
            Default is 0.
        cn_p : callable, str, optional
            Coefficient of yaw moment derivative with respect to roll rate.
            Default is 0.
        cn_q : callable, str, optional
            Coefficient of yaw moment derivative with respect to pitch rate.
            Default is 0.
        cn_r : callable, str, optional
            Coefficient of yaw moment derivative with respect to yaw rate.
            Default is 0.
        cl_0 : callable, str, optional
            Coefficient of roll moment at zero angle of attack.
            Default is 0.
        cl_alpha : callable, str, optional
            Coefficient of roll moment derivative with respect to angle of
            attack. Default is 0.
        cl_beta : callable, str, optional
            Coefficient of roll moment derivative with respect to sideslip
            angle. Default is 0.
        cl_p : callable, str, optional
            Coefficient of roll moment derivative with respect to roll rate.
            Default is 0.
        cl_q : callable, str, optional
            Coefficient of roll moment derivative with respect to pitch rate.
            Default is 0.
        cl_r : callable, str, optional
            Coefficient of roll moment derivative with respect to yaw rate.
            Default is 0.
        center_of_pressure : tuple, optional
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str
            Name of the aerodynamic surface. Default is 'GenericSurface'.
        """
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.center_of_pressure = center_of_pressure
        self.cp = center_of_pressure
        self.cpx = center_of_pressure[0]
        self.cpy = center_of_pressure[1]
        self.cpz = center_of_pressure[2]
        self.name = name

        self.cl_lift_0 = self._process_input(cl_lift_0, "cl_lift_0")
        self.cl_lift_alpha = self._process_input(cl_lift_alpha, "cl_lift_alpha")
        self.cl_lift_beta = self._process_input(cl_lift_beta, "cl_lift_beta")
        self.cl_lift_p = self._process_input(cl_lift_p, "cl_lift_p")
        self.cl_lift_q = self._process_input(cl_lift_q, "cl_lift_q")
        self.cl_lift_r = self._process_input(cl_lift_r, "cl_lift_r")
        self.cq_pitch_0 = self._process_input(cq_pitch_0, "cq_pitch_0")
        self.cq_pitch_alpha = self._process_input(cq_pitch_alpha, "cq_pitch_alpha")
        self.cq_pitch_beta = self._process_input(cq_pitch_beta, "cq_pitch_beta")
        self.cq_pitch_p = self._process_input(cq_pitch_p, "cq_pitch_p")
        self.cq_pitch_q = self._process_input(cq_pitch_q, "cq_pitch_q")
        self.cq_pitch_r = self._process_input(cq_pitch_r, "cq_pitch_r")
        self.cd_drag_0 = self._process_input(cd_drag_0, "cd_drag_0")
        self.cd_drag_alpha = self._process_input(cd_drag_alpha, "cd_drag_alpha")
        self.cd_drag_beta = self._process_input(cd_drag_beta, "cd_drag_beta")
        self.cd_drag_p = self._process_input(cd_drag_p, "cd_drag_p")
        self.cd_drag_q = self._process_input(cd_drag_q, "cd_drag_q")
        self.cd_drag_r = self._process_input(cd_drag_r, "cd_drag_r")
        self.cl_0 = self._process_input(cl_0, "cl_0")
        self.cl_alpha = self._process_input(cl_alpha, "cl_alpha")
        self.cl_beta = self._process_input(cl_beta, "cl_beta")
        self.cl_p = self._process_input(cl_p, "cl_p")
        self.cl_q = self._process_input(cl_q, "cl_q")
        self.cl_r = self._process_input(cl_r, "cl_r")
        self.cm_0 = self._process_input(cm_0, "cm_0")
        self.cm_alpha = self._process_input(cm_alpha, "cm_alpha")
        self.cm_beta = self._process_input(cm_beta, "cm_beta")
        self.cm_p = self._process_input(cm_p, "cm_p")
        self.cm_q = self._process_input(cm_q, "cm_q")
        self.cm_r = self._process_input(cm_r, "cm_r")
        self.cn_0 = self._process_input(cn_0, "cn_0")
        self.cn_alpha = self._process_input(cn_alpha, "cn_alpha")
        self.cn_beta = self._process_input(cn_beta, "cn_beta")
        self.cn_p = self._process_input(cn_p, "cn_p")
        self.cn_q = self._process_input(cn_q, "cn_q")
        self.cn_r = self._process_input(cn_r, "cn_r")

        self.compute_all_coefficients()

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
                'alpha',
                'beta',
                'mach',
                'reynolds',
                'pitch_rate',
                'yaw_rate',
                'roll_rate',
            ],
            ['coefficient'],
        )

    def compute_damping_coefficient(self, c_p, c_q, c_r):
        """Compute the damping coefficient from the derivatives of the
        aerodynamic coefficients."""

        def total_coefficient(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ):
            return (
                c_p(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * pitch_rate
                + c_q(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * yaw_rate
                + c_r(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
                * roll_rate
            )

        return Function(
            total_coefficient,
            [
                'alpha',
                'beta',
                'mach',
                'reynolds',
                'pitch_rate',
                'yaw_rate',
                'roll_rate',
            ],
            ['coefficient'],
        )

    def compute_all_coefficients(self):
        """Compute all the aerodynamic coefficients from the derivatives."""
        self.c_l_lift_f = self.compute_forcing_coefficient(
            self.cl_lift_0, self.cl_lift_alpha, self.cl_lift_beta
        )
        self.c_l_lift_d = self.compute_damping_coefficient(
            self.cl_lift_p, self.cl_lift_q, self.cl_lift_r
        )

        self.c_q_pitch_f = self.compute_forcing_coefficient(
            self.cq_pitch_0, self.cq_pitch_alpha, self.cq_pitch_beta
        )
        self.c_q_pitch_d = self.compute_damping_coefficient(
            self.cq_pitch_p, self.cq_pitch_q, self.cq_pitch_r
        )

        self.c_d_drag_f = self.compute_forcing_coefficient(
            self.cd_drag_0, self.cd_drag_alpha, self.cd_drag_beta
        )
        self.c_d_drag_d = self.compute_damping_coefficient(
            self.cd_drag_p, self.cd_drag_q, self.cd_drag_r
        )

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
            dyn_pressure_area * self.reference_length / (2 * stream_speed)
        )
        dyn_pressure_area_length = dyn_pressure_area * self.reference_length
        dyn_pressure_area_length_damping = (
            dyn_pressure_area_length * self.reference_length / (2 * stream_speed)
        )

        # Compute aerodynamic forces
        lift = dyn_pressure_area * self.c_l_lift_f(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) - dyn_pressure_area_damping * self.c_l_lift_d(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        side = dyn_pressure_area * self.c_q_pitch_f(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) - dyn_pressure_area_damping * self.c_q_pitch_d(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        drag = dyn_pressure_area * self.c_d_drag_f(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) - dyn_pressure_area_damping * self.c_d_drag_d(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        # Compute aerodynamic moments
        pitch = dyn_pressure_area_length * self.cmf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) - dyn_pressure_area_length_damping * self.cmd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        yaw = dyn_pressure_area_length * self.cnf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) - dyn_pressure_area_length_damping * self.cnd(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        roll = dyn_pressure_area_length * self.clf(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        ) - dyn_pressure_area_length_damping * self.cld(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        return lift, side, drag, pitch, yaw, roll
