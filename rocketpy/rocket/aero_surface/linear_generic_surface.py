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

    def _evaluate_derived_coefficients(self):
        """Exact override of the diagnostic cp accessors. The linear model
        already exposes the forcing derivatives ``cL_alpha``/``cm_alpha`` (pitch)
        and ``cQ_beta``/``cn_beta`` (yaw), so the slopes are read directly
        (frozen at zero alpha/beta/rates) instead of being recovered by
        numerical differentiation. Damping derivatives (``_p/_q/_r``) are
        intentionally excluded from the stability cp.
        """

        def _at_zero(coefficient, name):
            return Function(
                lambda mach: coefficient(0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0),
                "Mach",
                name,
            )

        self._set_derived_cp_accessors(
            _at_zero(self.cL_alpha, "cL_alpha"),
            _at_zero(self.cm_alpha, "cm_alpha"),
            _at_zero(self.cQ_beta, "cQ_beta"),
            _at_zero(self.cn_beta, "cn_beta"),
        )

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

    _COEFFICIENT_INPUTS = [
        "alpha",
        "beta",
        "mach",
        "reynolds",
        "pitch_rate",
        "yaw_rate",
        "roll_rate",
    ]

    def compute_forcing_coefficient(self, c_0, c_alpha, c_beta):
        """Compose the forcing coefficient ``c_0 + c_alpha*alpha + c_beta*beta``,
        evaluating only the non-zero terms.

        Two hot-loop optimizations: ``get_value_opt`` is the unvalidated fast
        evaluator (for callable-source coefficients it is the raw source), and
        terms that are identically zero are skipped entirely. For a Barrowman
        surface each forcing coefficient has at most one non-zero derivative, so
        this typically collapses to a single source call (or to a constant 0).
        """
        has_0 = not getattr(c_0, "is_zero_coefficient", False)
        has_alpha = not getattr(c_alpha, "is_zero_coefficient", False)
        has_beta = not getattr(c_beta, "is_zero_coefficient", False)
        c_0_opt = c_0.get_value_opt
        c_alpha_opt = c_alpha.get_value_opt
        c_beta_opt = c_beta.get_value_opt

        if not (has_0 or has_alpha or has_beta):

            def total_coefficient(  # pylint: disable=unused-argument
                alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
            ):
                return 0.0

        else:

            def total_coefficient(
                alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
            ):
                value = 0.0
                if has_0:
                    value += c_0_opt(
                        alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
                    )
                if has_alpha:
                    value += (
                        c_alpha_opt(
                            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
                        )
                        * alpha
                    )
                if has_beta:
                    value += (
                        c_beta_opt(
                            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
                        )
                        * beta
                    )
                return value

        return Function(total_coefficient, self._COEFFICIENT_INPUTS, ["coefficient"])

    def compute_damping_coefficient(self, c_p, c_q, c_r):
        """Compose the damping coefficient
        ``c_p*roll_rate + c_q*pitch_rate + c_r*yaw_rate``, evaluating only the
        non-zero terms (see :meth:`compute_forcing_coefficient`). For a Barrowman
        surface only ``cl_p`` (roll damping) is non-zero, so most damping
        coefficients collapse to a constant 0.
        """
        has_p = not getattr(c_p, "is_zero_coefficient", False)
        has_q = not getattr(c_q, "is_zero_coefficient", False)
        has_r = not getattr(c_r, "is_zero_coefficient", False)
        c_p_opt = c_p.get_value_opt
        c_q_opt = c_q.get_value_opt
        c_r_opt = c_r.get_value_opt

        if not (has_p or has_q or has_r):

            def total_coefficient(  # pylint: disable=unused-argument
                alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
            ):
                return 0.0

        else:

            def total_coefficient(
                alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
            ):
                value = 0.0
                if has_p:
                    value += (
                        c_p_opt(
                            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
                        )
                        * roll_rate
                    )
                if has_q:
                    value += (
                        c_q_opt(
                            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
                        )
                        * pitch_rate
                    )
                if has_r:
                    value += (
                        c_r_opt(
                            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
                        )
                        * yaw_rate
                    )
                return value

        return Function(total_coefficient, self._COEFFICIENT_INPUTS, ["coefficient"])

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

        self.cL = self.cLf
        self.cQ = self.cQf
        self.cD = self.cDf
        self.cm = self.cmf
        self.cn = self.cnf

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
        alpha_dot=0.0,  # pylint: disable=unused-argument
        beta_dot=0.0,  # pylint: disable=unused-argument
    ):
        """Compute the aerodynamic forces and moments from the aerodynamic
        coefficients.

        The linear (Barrowman) model does not use the unsteady ``alpha_dot`` /
        ``beta_dot`` terms; they are accepted for signature compatibility.

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
            Non-dimensional (reduced) pitch rate, ``q * L_ref / (2 * V)``.
        yaw_rate : float
            Non-dimensional (reduced) yaw rate, ``r * L_ref / (2 * V)``.
        roll_rate : float
            Non-dimensional (reduced) roll rate, ``p * L_ref / (2 * V)``.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        # Precompute common values. The angular rates arrive already
        # non-dimensionalized (reduced rates, e.g. ``q* = q * L_ref / (2 * V)``),
        # so the rate-damping terms use the same dynamic-pressure scaling as the
        # forcing terms: the ``L_ref / (2 * V)`` factor now lives in the rate
        # itself, not in the scaling. (Algebraically identical to the previous
        # ``0.5 * rho * V * A * L / 2`` damping scaling applied to raw rates.)
        dyn_pressure_area = 0.5 * rho * stream_speed**2 * self.reference_area
        dyn_pressure_area_length = dyn_pressure_area * self.reference_length

        # Evaluate the composed coefficients through the fast, unvalidated
        # ``get_value_opt`` path (the composed coefficients are callable-source
        # Functions, so this calls the closure directly, skipping the per-call
        # ``__call__``/``get_value`` argument validation in the hot loop).
        args = (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)

        # Compute aerodynamic forces (forcing + reduced-rate damping)
        lift = dyn_pressure_area * (
            self.cLf.get_value_opt(*args) + self.cLd.get_value_opt(*args)
        )
        side = dyn_pressure_area * (
            self.cQf.get_value_opt(*args) + self.cQd.get_value_opt(*args)
        )
        drag = dyn_pressure_area * (
            self.cDf.get_value_opt(*args) + self.cDd.get_value_opt(*args)
        )

        # Compute aerodynamic moments (forcing + reduced-rate damping)
        pitch = dyn_pressure_area_length * (
            self.cmf.get_value_opt(*args) + self.cmd.get_value_opt(*args)
        )
        yaw = dyn_pressure_area_length * (
            self.cnf.get_value_opt(*args) + self.cnd.get_value_opt(*args)
        )
        roll = dyn_pressure_area_length * (
            self.clf.get_value_opt(*args) + self.cld.get_value_opt(*args)
        )

        return lift, side, drag, pitch, yaw, roll
