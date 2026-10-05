import math

import numpy as np

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface


class _BarrowmanSurface(GenericSurface):
    """Base class for the surfaces modeled with the Barrowman method: nose
    cones, tails/transitions and fins.

    The normal force is ``clalpha(Mach)`` times the total angle of attack and
    acts at the geometric center of pressure ``cpz``. Fins add a roll moment
    from their cant angle and their roll damping.

    The coefficients ``cN``, ``cY`` and ``cl`` describe that same force, so
    what is read or plotted is what flies.
    :meth:`compute_forces_and_moments` computes it in a faster way, and a unit
    test keeps the two in agreement. The slopes ``cN_alpha`` and ``cY_beta``
    are used for the stability margin and the center of pressure.

    Subclasses must set ``self.clalpha`` (a Function of Mach) and the geometric
    center of pressure before calling ``super().__init__``. Fins must also
    provide ``roll_parameters``, ``[clf_delta, cld_omega, cant_angle_rad]``,
    with the two coefficients taken at zero cant.
    """

    # Geometry-defined Barrowman surfaces are axisymmetric by construction
    # (``cY_beta = -cN_alpha``, etc.), so they contribute identically to the
    # pitch and yaw planes. The individual ``Fin`` overrides this back to False.
    is_axisymmetric = True

    def _geometry_changed(self):
        """Rebuild the coefficients after a change to the geometry, and count
        the change so the rockets using this surface update too."""
        if not hasattr(self, "cN_alpha"):
            return  # Still being built: the constructor builds the coefficients
        self.evaluate_coefficients()
        self._evaluate_stability_derivatives()
        self._version += 1

    def _evaluate_stability_derivatives(self):
        """The slopes are set from the geometry (see
        :meth:`evaluate_coefficients`), so there is nothing to differentiate:
        only the center-of-pressure accessors are built."""
        self._set_stability_accessors()

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
        """Populate the coefficients from the surface geometry. Called by
        ``GenericSurface.__init__`` and again whenever the geometry changes.

        Sets the normal-force slopes ``cN_alpha`` (pitch) and ``cY_beta`` (yaw),
        the force coefficients ``cN`` and ``cY`` and, for fins, the roll
        coefficients ``cl_0``, ``cl_p`` and ``cl``. The geometric center of
        pressure is carried by the force application point (not the moment
        coefficients), so ``cm_alpha`` / ``cn_beta`` are zero.
        """
        clalpha = self.clalpha  # normal-force-curve slope, a Function of Mach

        # Axisymmetric Barrowman normal force: equal-magnitude slopes in the
        # pitch and yaw planes. The yaw-plane (side-force) slope is opposite in
        # sign due to the body-frame axis convention.
        self.cN_alpha = self._mach_coefficient(clalpha.get_value_opt, "cN_alpha")
        self.cY_beta = self._mach_coefficient(
            lambda mach: -clalpha.get_value_opt(mach), "cY_beta"
        )

        # The center of pressure is carried by the force application point, so
        # the moment slopes add no further offset
        self.cm_alpha = self._mach_coefficient(lambda mach: 0.0, "cm_alpha")
        self.cn_beta = self._mach_coefficient(lambda mach: 0.0, "cn_beta")

        # The classic normal force (see compute_forces_and_moments) written as
        # coefficients: ``clalpha * total angle of attack`` in the plane of the
        # wind, split between the pitch and yaw planes along the crossflow.
        total_normal = self._as_coefficient(
            lambda alpha_total, mach: clalpha.get_value_opt(mach) * alpha_total,
            "cN",
        )
        c_normal, c_side = self._split_along_crossflow(total_normal, ("cN", "cY"))
        self.cN = self._as_coefficient(c_normal, "cN")
        self.cY = self._as_coefficient(c_side, "cY")

        # Fin roll forcing (cant) and damping, when present. The cant angle is
        # read when the coefficient is evaluated, so a controller may change it
        # without the coefficients being rebuilt.
        roll_parameters = getattr(self, "roll_parameters", None)
        if roll_parameters is not None:
            clf_delta, cld_omega, _ = roll_parameters
            self.cl_0 = self._mach_coefficient(
                lambda mach: clf_delta.get_value_opt(mach) * self._roll_cant_angle_rad,
                "cl_0",
            )
            self.cl_p = self._mach_coefficient(
                lambda mach: (
                    cld_omega.get_value_opt(mach) * math.cos(self.cant_angle_rad)
                ),
                "cl_p",
            )
            self.cl = self._as_coefficient(
                lambda mach, roll_rate: (
                    clf_delta.get_value_opt(mach) * self._roll_cant_angle_rad
                    + cld_omega.get_value_opt(mach)
                    * math.cos(self.cant_angle_rad)
                    * roll_rate
                ),
                "cl",
            )

    def compute_forces_and_moments(  # pylint: disable=unused-argument
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        *args,
    ):
        """Compute the surface's forces and moments with the classic Barrowman
        method. Called at each simulation step.

        The normal force uses the true total angle of attack between the flow
        and the body axis, ``attack_angle = arccos(-v_z / |v|)``, giving
        ``0.5 * rho * V**2 * A_ref * clalpha(Mach) * attack_angle``. It is
        applied perpendicular to the body axis (along the transverse flow) at
        the geometric center of pressure, and its moment about the rocket's
        center of dry mass is the geometric transport ``cp ^ force``.
        Fin sets add their roll moment on top.

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
            Surface center of pressure relative to the center of dry mass, in
            the body frame (the force-application point; see
            :attr:`force_application_point`).
        omega : tuple of float
            Body angular velocity about the x, y, z axes. Only the roll
            component (``omega[2]``) is used, by fin sets.
        *args
            Extra positional arguments accepted for signature compatibility with
            the generic surface (``density``, ``dynamic_viscosity``, ``z``);
            unused by the Barrowman model.

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
                c_lift = self.clalpha.get_value_opt(stream_mach) * attack_angle
                lift = 0.5 * rho * stream_speed**2 * self.reference_area * c_lift
                # Normal force, perpendicular to the body axis, directed along
                # the transverse component of the flow.
                transverse_norm = (stream_vx**2 + stream_vy**2) ** 0.5
                R1 = lift * stream_vx / transverse_norm
                R2 = lift * stream_vy / transverse_norm
                # The normal force acts at the geometric center of pressure,
                # which ``cp`` already locates relative to the center of dry
                # mass; transport its moment from there.
                force = Vector([R1, R2, R3])
                M1, M2, M3 = cp ^ force

        # Fin roll (cant forcing + rate damping); zero for non-fin surfaces.
        M3 += self._roll_moment(stream_speed, stream_mach, rho, omega)

        return R1, R2, R3, M1, M2, M3

    def _roll_moment(self, stream_speed, mach, rho, omega):
        """Roll moment from the linear roll coefficients: cant forcing plus
        reduced-rate damping. Returns 0 for surfaces without fins, whose roll
        coefficients are identically zero.
        """
        if self.cl.is_zero:
            return 0.0
        reduced_roll_rate = (
            omega[2] * self.reference_length / (2 * stream_speed)
            if stream_speed > 0
            else 0.0
        )
        # The Barrowman roll coefficients depend only on Mach and the roll rate.
        args = (0.0, 0.0, mach, 0.0, 0.0, 0.0, reduced_roll_rate)
        cl = self.cl.get_value_opt(*args)
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
            control_variables=self.control_variables,
            name=name,
        )
