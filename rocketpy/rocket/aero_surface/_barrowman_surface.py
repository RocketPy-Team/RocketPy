import numpy as np

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket.aero_surface.linear_generic_surface import LinearGenericSurface


class _BarrowmanSurface(LinearGenericSurface):
    """Intermediate base for Barrowman-defined aerodynamic surfaces
    such as nose cones, tails/transitions and fin sets.

    These surfaces expose a lift-curve slope ``clalpha`` (a ``Function`` of
    Mach), a geometric center of pressure ``cpz`` and, for fins, a pair of roll
    forcing/damping coefficients.

    The in-flight normal force and its moment are computed with the Barrowman
    method (see :meth:`compute_forces_and_moments`): the normal force uses the
    true total angle of attack, acting at the geometric centre of pressure, and
    its moment about the centre of dry mass is the geometric transport
    (``cp ^ force``).  When the subclass provides planform geometry (see
    :attr:`_planform_area`, :attr:`_planform_centroid`, :attr:`_cp_slender`), a
    non-linear Galejs body-lift term :math:`K \cdot (A_\text{plan} /
    A_\text{ref}) \cdot \sin^2\alpha` is added and the CP is blended
    accordingly.  The resultant force is reported at the blended centre of
    pressure in the body frame through :meth:`_default_surface_rotation`.

    The class also derives the linear normal-force slopes ``cN_alpha`` (pitch
    plane) and ``cY_beta`` (yaw plane), which feed the stability and
    center-of-pressure diagnostics; the geometric cp is carried by the force
    application point, so the moment slopes ``cm_alpha`` / ``cn_beta`` are zero.
    Fin roll uses the coefficient model: ``cl_0`` (cant forcing) and ``cl_p``
    (roll damping).

    Subclasses must compute ``self.clalpha`` (Function of Mach) and the geometric
    center of pressure before calling ``super().__init__`` (which passes the
    geometric cp through ``center_of_pressure``), and, for fins, set
    ``self.roll_parameters = [clf_delta, cld_omega, cant_angle_rad]``.
    Subclasses that wish to enable body lift must also set
    :attr:`_planform_area`, :attr:`_planform_centroid` and :attr:`_cp_slender`.
    """

    # Geometry-defined Barrowman surfaces are axisymmetric by construction
    # (``cY_beta = -cN_alpha``, etc.), so they contribute identically to the
    # pitch and yaw planes. The individual ``Fin`` overrides this back to False.
    is_axisymmetric = True

    # Galejs body-lift parameters.  Subclasses may override these to enable
    # the nonlinear sin²α body-lift term (see :meth:`compute_forces_and_moments`).
    _body_lift_k = 1.1          # Galejs constant K
    _planform_area = 0.0        # projected (planform) area, m²
    _planform_centroid = 0.0    # planform centroid local z, m
    _cp_slender = 0.0           # slender-body CP local z, m

    @staticmethod
    def _beta(mach):
        """Prandtl-Glauert compressibility factor used to correct subsonic
        force coefficients of the nose cone, fins and tails/transitions, as in
        Barrowman.

        Parameters
        ----------
        mach : int, float
            Mach number.

        Returns
        -------
        beta : float
            Compressibility factor based on the Mach number.

        References
        ----------
        [1] Barrowman, James S. https://arc.aiaa.org/doi/10.2514/6.1979-504
        """
        if mach < 0.8:
            return np.sqrt(1 - mach**2)
        elif mach < 1.1:
            return np.sqrt(1 - 0.8**2)
        else:
            return np.sqrt(mach**2 - 1)

    def _default_surface_rotation(self):
        """Rotation from the surface-local frame to the body frame. A Barrowman
        surface is defined in a frame flipped 180 degrees about the transverse
        axis relative to the body frame (its z axis runs from the nose toward the
        tail), so its geometric center of pressure maps to the body frame through
        this rotation. This is RocketPy's classic convention, so the surface's
        center of pressure lands at the same body-frame point as before the
        generic-surface refactor.
        """
        return Matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    def evaluate_coefficients(self):
        """Populate the coefficient slopes used by the stability diagnostics
        from the surface geometry. Called by ``GenericSurface.__init__`` and
        again whenever the geometry changes.

        Sets the normal-force slopes ``cN_alpha`` (pitch) and ``cY_beta`` (yaw)
        and the fin roll coefficients when present. The geometric center of
        pressure is carried by the force application point (not the moment
        coefficients), so ``cm_alpha`` / ``cn_beta`` are zero. The in-flight
        force and moment are computed geometrically in
        :meth:`compute_forces_and_moments`.
        """
        clalpha = self.clalpha  # normal-force-curve slope, a Function of Mach

        # Axisymmetric Barrowman normal force: equal-magnitude slopes in the
        # pitch and yaw planes. The yaw-plane (side-force) slope is opposite in
        # sign due to the body-frame axis convention.
        self.cN_alpha = self._mach_coefficient(
            lambda mach: clalpha.get_value_opt(mach), "cN_alpha"
        )
        self.cY_beta = self._mach_coefficient(
            lambda mach: -clalpha.get_value_opt(mach), "cY_beta"
        )

        # The center of pressure is carried by the force application point, so
        # the moment slopes add no further offset (the diagnostic recovers the
        # geometric cp from the application point alone).
        self.cm_alpha = self._mach_coefficient(lambda mach: 0.0, "cm_alpha")
        self.cn_beta = self._mach_coefficient(lambda mach: 0.0, "cn_beta")

        # Fin roll forcing (cant) and damping, when present.
        roll_parameters = getattr(self, "roll_parameters", None)
        if roll_parameters is not None:
            clf_delta, cld_omega, cant_angle_rad = roll_parameters
            self.cl_0 = self._mach_coefficient(
                lambda mach: clf_delta.get_value_opt(mach) * cant_angle_rad, "cl_0"
            )
            self.cl_p = self._mach_coefficient(
                lambda mach: cld_omega.get_value_opt(mach), "cl_p"
            )

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        *args,  # pylint: disable=unused-argument
    ):
        """Compute the surface's forces and moments with the Barrowman method
        plus the optional Galejs body-lift extension. Called at each
        simulation step.

        The normal force has two contributions:

        1. **Slender-body linear term**:
           ``0.5 ρ V² A_ref · clalpha(Mach) · α``

        2. **Galejs body-lift term** (nonlinear, when :attr:`_planform_area`
           > 0):
           ``0.5 ρ V² A_ref · K · (A_plan / A_ref) · sin²α``,
           with ``K = 1.1``.  At very low speed and high α (apogee) the
           term is damped by a factor ``(M / 0.05)²``.

        The total force is applied at the blended centre of pressure of the
        two contributions.  Fin sets (including canards) add their roll moment
        on top.

        Parameters
        ----------
        stream_velocity : Vector
            Velocity of the airflow relative to the surface, in the body frame.
        stream_speed : float
            Magnitude of the airflow speed.
        stream_mach : float
            Mach number of the airflow.
        rho : float
            Air density.
        cp : Vector
            Surface centre of pressure relative to the centre of dry mass, in
            the body frame (the force-application point; see
            :attr:`force_application_point`).  When body lift is active this
            is the *slender-body* CP; the blended CP is computed internally.
        omega : tuple of float
            Body angular velocity about the x, y, z axes. Only the roll
            component (``omega[2]``) is used, by fin sets.
        *args
            Extra positional arguments accepted for signature compatibility with
            the generic surface (``density``, ``dynamic_viscosity``, ``z``,
            ``alpha_dot``, ``beta_dot``); unused by the Barrowman model.

        Returns
        -------
        tuple of float
            The forces (x, y, z) and the moments about the x, y, z axes, in the
            body frame.
        """
        R1 = R2 = R3 = M1 = M2 = M3 = 0.0

        stream_vx, stream_vy, stream_vz = stream_velocity
        if stream_vx**2 + stream_vy**2 != 0:
            stream_vzn = stream_vz / stream_speed
            if -stream_vzn < 1:
                attack_angle = np.arccos(-stream_vzn)

                # --- Slender-body linear term ---
                c_lift_linear = (
                    self.clalpha.get_value_opt(stream_mach) * attack_angle
                )

                # --- Galejs body-lift term (nonlinear sin²α) ---
                c_lift_body = 0.0
                if self._planform_area > 0:
                    sin2_alpha = np.sin(attack_angle) ** 2
                    c_lift_body = (
                        self._body_lift_k
                        * self._planform_area
                        / self.reference_area
                        * sin2_alpha
                    )
                    # Low-speed / high-α damping (avoids apogee CP anomaly)
                    if stream_mach < 0.05 and attack_angle > np.pi / 4:
                        c_lift_body *= (stream_mach / 0.05) ** 2

                c_lift = c_lift_linear + c_lift_body
                lift = (
                    0.5
                    * rho
                    * stream_speed**2
                    * self.reference_area
                    * c_lift
                )
                # Normal force, perpendicular to the body axis, directed along
                # the transverse component of the flow.
                transverse_norm = (stream_vx**2 + stream_vy**2) ** 0.5
                R1 = lift * stream_vx / transverse_norm
                R2 = lift * stream_vy / transverse_norm
                # The total force acts at the blended centre of pressure:
                # slender-body CP + Galejs offset.
                force = Vector([R1, R2, R3])
                if c_lift_body > 0 and c_lift != 0:
                    # Body-frame offset between the two CPs (the
                    # _default_surface_rotation flips the local z axis, hence
                    # the minus sign).
                    dz_body = -(self._planform_centroid - self._cp_slender)
                    cp_effective = cp + Vector(
                        [0.0, 0.0, dz_body * c_lift_body / c_lift]
                    )
                else:
                    cp_effective = cp
                M1, M2, M3 = cp_effective ^ force

        # Fin roll (cant forcing + rate damping); zero for non-fin surfaces.
        M3 += self._roll_moment(stream_speed, stream_mach, rho, omega)

        return R1, R2, R3, M1, M2, M3

    def _roll_moment(self, stream_speed, mach, rho, omega):
        """Roll moment from the linear roll coefficients: cant forcing plus
        reduced-rate damping. Returns 0 for surfaces without fins, whose roll
        coefficients are identically zero.
        """
        reduced_roll_rate = (
            omega[2] * self.reference_length / (2 * stream_speed)
            if stream_speed > 0
            else 0.0
        )
        # The Barrowman roll coefficients depend only on Mach and the roll rate.
        args = (0.0, 0.0, mach, 0.0, 0.0, 0.0, reduced_roll_rate)
        cl = self.clf.get_value_opt(*args) + self.cld.get_value_opt(*args)
        return (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * self.reference_length
            * cl
        )

    def _mach_coefficient(self, func_of_mach, name="coefficient"):
        """Wrap a Mach-only callable into an :class:`AeroCoefficient` that
        depends only on Mach but is callable over the full coefficient argument
        tuple. Storing it at one dimension keeps the Mach table un-smeared and
        evaluates with a single argument in the hot loop.
        """
        return AeroCoefficient(
            func_of_mach,
            depends_on=("mach",),
            unsteady_aero=self._unsteady_aero,
            control_variables=self.control_variables,
            name=name,
        )
