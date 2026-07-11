import numpy as np

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket.aero_surface.linear_generic_surface import LinearGenericSurface


class _BarrowmanSurface(LinearGenericSurface):
    """Intermediate base for geometry-defined (Barrowman) aerodynamic surfaces
    such as nose cones, tails/transitions and fin sets.

    These surfaces historically expose a lift-curve slope ``clalpha`` (a
    ``Function`` of Mach), a geometric center of pressure ``cpz`` and, for fins,
    a pair of roll forcing/damping coefficients. This class translates that
    Barrowman description into the linear generic-surface coefficient model so
    the forces and moments are computed by the single, shared
    :meth:`GenericSurface.compute_forces_and_moments`:

    - normal-force slope -> ``cL_alpha`` (pitch plane) and ``cQ_beta`` (yaw plane);
    - center-of-pressure offset -> ``cm_alpha`` / ``cn_beta`` (the moment is
      carried by the coefficients, with the force applied at the surface origin);
    - fin roll -> ``cl_0`` (cant forcing) and ``cl_p`` (roll damping).

    Subclasses must compute ``self.clalpha`` (Function of Mach) and the geometric
    center of pressure before calling ``super().__init__`` (which passes the
    geometric cp through ``center_of_pressure``), and, for fins, set
    ``self.roll_parameters = [clf_delta, cld_omega, cant_angle_rad]``.
    """

    # Geometry-defined Barrowman surfaces are axisymmetric by construction
    # (``cQ_beta = -cL_alpha``, etc.), so they contribute identically to the
    # pitch and yaw planes. The individual ``Fin`` overrides this back to False.
    is_axisymmetric = True

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

    @property
    def force_application_point(self):
        """Barrowman surfaces apply the resultant force at the surface origin;
        the whole center-of-pressure offset is carried by the ``cm``/``cn``
        moment coefficients (avoiding a double count with the ``cp ^ force``
        transport). The geometric center of pressure remains available through
        ``self.cp``/``self.cpz`` for display and through
        ``center_of_pressure_z`` as a mach-dependent diagnostic.
        """
        return Vector([0, 0, 0])

    def evaluate_coefficients(self):
        """Populate the linear generic-surface coefficient derivatives from the
        surface geometry. Called by ``GenericSurface.__init__`` and again
        whenever the geometry changes.
        """
        clalpha = self.clalpha  # Function of Mach
        cpz = self.cpz  # geometric center of pressure (set from center_of_pressure)
        reference_length = self.reference_length

        # Axisymmetric Barrowman lift: equal-magnitude slopes in the pitch and
        # yaw planes. The yaw-plane (side-force) slope is opposite in sign due to
        # the aerodynamic-to-body frame convention used by the shared compute.
        self.cL_alpha = self._mach_coefficient(
            lambda mach: clalpha.get_value_opt(mach), "cL_alpha"
        )
        self.cQ_beta = self._mach_coefficient(
            lambda mach: -clalpha.get_value_opt(mach), "cQ_beta"
        )

        # Center-of-pressure offset expressed as moment coefficients (the local
        # cp ^ force couple, with the force applied at the origin).
        self.cm_alpha = self._mach_coefficient(
            lambda mach: -clalpha.get_value_opt(mach) * cpz / reference_length,
            "cm_alpha",
        )
        self.cn_beta = self._mach_coefficient(
            lambda mach: clalpha.get_value_opt(mach) * cpz / reference_length,
            "cn_beta",
        )

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
