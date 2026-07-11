import copy
import math

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.plots.aero_surface_plots import _GenericSurfacePlots
from rocketpy.prints.aero_surface_prints import _GenericSurfacePrints
from rocketpy.rocket.aero_surface.aero_coefficient import (
    AeroCoefficient,
    build_independent_vars,
)


class GenericSurface:
    """Defines a generic aerodynamic surface with custom force and moment
    coefficients. The coefficients can be nonlinear functions of the angle of
    attack, sideslip angle, Mach number, Reynolds number, pitch rate, yaw rate
    and roll rate."""

    #: Whether this surface contributes identically to the pitch and yaw planes
    #: *by construction*. Conservatively ``False`` for a generic surface (its
    #: coefficients may differ between planes); the built-in axisymmetric
    #: surfaces override it to ``True``. The rocket uses it to skip the numeric
    #: pitch/yaw axisymmetry check when every surface is symmetric by
    #: construction.
    is_axisymmetric = False

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Generic Surface",
        unsteady_aero=False,
    ):
        """Create a generic aerodynamic surface, defined by its aerodynamic
        coefficients. This surface is used to model any aerodynamic surface
        that does not fit the predefined classes.

        Important
        ---------
        All the aerodynamic coefficients can be input as callable functions of
        angle of attack, angle of sideslip, Mach number, Reynolds number,
        pitch rate, yaw rate and roll rate. For CSV files, the header must
        contain at least one of the following: "alpha", "beta", "mach",
        "reynolds", "pitch_rate", "yaw_rate" and "roll_rate". The
        independent variable columns can be provided in any order.

        The angular-rate inputs ("pitch_rate", "yaw_rate", "roll_rate") are the
        conventional **non-dimensional reduced rates**, ``q* = q * L_ref / (2 * V)``
        (and likewise for ``r``/``p``), matching how published and tool-generated
        aerotables (Missile DATCOM, OpenVSP, CFD/wind-tunnel data) tabulate rate
        derivatives. Provide coefficient tables against the reduced rates, not the
        raw body rates in rad/s.

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
        coefficients: dict
            List of coefficients. If a coefficient is omitted, it is set to 0.
            The valid coefficients are:\n
            cL: str, callable, optional
                Lift coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cQ: str, callable, optional
                Side force coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cD: str, callable, optional
                Drag coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cm: str, callable, optional
                Pitch moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cn: str, callable, optional
                Yaw moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cl: str, callable, optional
                Roll moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
        center_of_pressure : tuple, list, optional
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str, optional
            Name of the aerodynamic surface. Default is 'GenericSurface'.
        unsteady_aero : bool, optional
            If True, the coefficients additionally depend on the time
            derivatives of the flow angles, and ``alpha_dot`` and ``beta_dot``
            are appended (in that order) to the independent variables. CSV files
            may then include "alpha_dot"/"beta_dot" columns, and callables must
            accept the two extra trailing arguments. The simulation supplies 0
            for these unless it computes them, so existing coefficient tables are
            unaffected. Default is False.
        """

        # The independent variables of the coefficients are derived (see the
        # ``independent_vars`` property) from ``unsteady_aero`` and, for
        # subclasses, ``control_variables``. When ``unsteady_aero`` is enabled,
        # the time-derivatives of the flow angles (``alpha_dot``, ``beta_dot``)
        # become extra axes (defaulting to 0 at runtime, so existing tables are
        # unaffected). Subclasses that add externally-supplied axes set
        # ``control_variables`` before calling ``super().__init__``.
        self._unsteady_aero = unsteady_aero
        # Externally-supplied axes (e.g. control deflections). Subclasses set
        # this before ``super().__init__``; defaults to none for plain surfaces.
        self.control_variables = getattr(self, "control_variables", ())
        # Ordered independent variables accepted by every coefficient: the seven
        # base axes, plus ``alpha_dot``/``beta_dot`` when ``unsteady_aero`` is
        # enabled (integrator-supplied), plus any ``control_variables`` a
        # subclass appended (externally supplied). Fixed at construction.
        self.independent_vars = build_independent_vars(
            self._unsteady_aero, self.control_variables
        )

        self.reference_area = reference_area
        self.reference_length = reference_length
        self.center_of_pressure = center_of_pressure
        self.cp = center_of_pressure
        self.cpx = center_of_pressure[0]
        self.cpy = center_of_pressure[1]
        self.cpz = center_of_pressure[2]
        self.name = name

        self._rotation_surface_to_body = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        default_coefficients = self._get_default_coefficients()
        self._check_coefficients(coefficients, default_coefficients)
        coefficients = self._complete_coefficients(coefficients, default_coefficients)
        for coeff, coeff_value in coefficients.items():
            value = AeroCoefficient(
                coeff_value,
                unsteady_aero=self._unsteady_aero,
                control_variables=self.control_variables,
                name=coeff,
            )
            setattr(self, coeff, value)

        self.evaluate_coefficients()
        self._evaluate_derived_coefficients()

        # Reporting layers. Subclasses override these with their own (more
        # specific) prints/plots after calling ``super().__init__``.
        self.prints = _GenericSurfacePrints(self)
        self.plots = _GenericSurfacePlots(self)

    @property
    def force_application_point(self):
        """Local point (surface frame) at which the resultant force is applied
        when transporting its moment to the rocket's center of dry mass. For a
        plain generic surface this is simply the center of pressure ``self.cp``;
        the residual couple is carried by the ``cm``/``cn``/``cl`` coefficients.
        Barrowman subclasses override this to the origin, because they fold the
        whole cp offset into the moment coefficients instead.
        """
        return Vector([self.cpx, self.cpy, self.cpz])

    def info(self):
        """Prints a summary of the surface's geometry and aerodynamic
        coefficients. Subclasses override this with surface-specific summaries.

        Returns
        -------
        None
        """
        self.prints.geometry()
        self.prints.coefficients()

    def all_info(self):
        """Prints and plots all available information of the surface.

        Returns
        -------
        None
        """
        self.prints.all()
        self.plots.all()

    def evaluate_coefficients(self):
        """Hook for subclasses to (re)populate the aerodynamic coefficient
        ``Function``s from their geometry. The base class builds coefficients
        directly from the user-provided dictionary, so this is a no-op here.
        Subclasses that derive coefficients from geometry (e.g. the Barrowman
        surfaces) override this and call it again whenever their geometry
        changes.

        Returns
        -------
        None
        """

    def _evaluate_derived_coefficients(self):
        """Build the mach-only diagnostic accessors used by the rocket's
        center-of-pressure / stability-margin computation, for both the pitch
        and the yaw plane.

        These reconstruct, at the linearization point ``alpha = beta = 0`` with
        zero rates, each plane's force-curve slope and the location of its
        center of pressure. The center of pressure combines the surface's
        declared local ``cpz`` with the offset implied by its moment
        coefficient (the two representations are interchangeable;
        ``cpz_eff = cpz - (dc_moment/dangle)/(dc_force/dangle) * L_ref``):

        - pitch plane: ``lift_coefficient_derivative`` (``dcL/dalpha``) and
          ``center_of_pressure_z`` (from ``cm``);
        - yaw plane: ``side_coefficient_derivative`` and
          ``center_of_pressure_z_yaw`` (from ``cn``).

        Returns
        -------
        None
        """
        cL_alpha = self._partial_slope(self.cL, axis="alpha")
        cm_alpha = self._partial_slope(self.cm, axis="alpha")
        cQ_beta = self._partial_slope(self.cQ, axis="beta")
        cn_beta = self._partial_slope(self.cn, axis="beta")
        self._set_derived_cp_accessors(cL_alpha, cm_alpha, cQ_beta, cn_beta)

    def _set_derived_cp_accessors(self, cL_alpha, cm_alpha, cQ_beta, cn_beta):
        """Store the pitch- and yaw-plane diagnostic accessors as mach-only
        ``Function``s, guarding the moment/force division for zero-force
        surfaces (which then drop out of the force-weighted cp average).

        Parameters
        ----------
        cL_alpha : Function
            Pitch-plane normal-force slope ``dcL/dalpha`` vs. mach.
        cm_alpha : Function
            Pitch-moment slope ``dcm/dalpha`` vs. mach.
        cQ_beta : Function
            Yaw-plane side-force slope ``dcQ/dbeta`` vs. mach.
        cn_beta : Function
            Yaw-moment slope ``dcn/dbeta`` vs. mach.
        """
        reference_length = self.reference_length
        local_cpz = self.force_application_point[2]

        def _cp_z(force_slope, moment_slope):
            # Recover the center of pressure from a force slope and its matching
            # moment slope, as a Function of Mach.
            def cp_z(mach):
                slope = force_slope.get_value_opt(mach)
                # No force at this Mach -> the cp is undefined; fall back to the
                # geometric application point so this surface contributes zero
                # weight to the force-weighted cp average.
                if slope == 0:
                    return local_cpz
                # cp = application point - (moment slope / force slope) * L_ref.
                return (
                    local_cpz
                    - moment_slope.get_value_opt(mach) / slope * reference_length
                )

            return Function(cp_z, "Mach", "Center of pressure to local origin (m)")

        # Pitch plane.
        self.lift_coefficient_derivative = cL_alpha
        self.center_of_pressure_z = _cp_z(cL_alpha, cm_alpha)

        # Yaw plane. The side-force slope is sign-adjusted (``-cQ_beta``) so
        # that an axisymmetric surface yields the same signed weight as the
        # pitch plane, making the two planes' margins coincide when symmetric.
        self.side_coefficient_derivative = -cQ_beta
        self.center_of_pressure_z_yaw = _cp_z(cQ_beta, cn_beta)

    def _partial_slope(self, coefficient, axis):
        """Partial derivative ``d(coefficient)/d(axis)`` at ``alpha = beta = 0``
        and zero rates, returned as a mach-only ``Function``.

        Reuses :meth:`Function.differentiate` on a single-variable slice of the
        coefficient taken along ``axis`` (``"alpha"`` or ``"beta"``) with all
        other base inputs frozen at zero. Extra axes (control deflections) are
        frozen at their current value via :meth:`_coefficient_arguments`.

        Parameters
        ----------
        coefficient : Function
            A coefficient ``Function`` over ``self.independent_vars``.
        axis : str
            Either ``"alpha"`` or ``"beta"``.

        Returns
        -------
        Function
            ``d(coefficient)/d(axis)`` evaluated at the zero point, vs. mach.
        """

        def slope(mach):
            if axis == "alpha":
                sliced = Function(
                    lambda alpha: coefficient(
                        *self._coefficient_arguments(
                            alpha, 0.0, mach, 0.0, 0.0, 0.0, 0.0
                        )
                    )
                )
            else:
                sliced = Function(
                    lambda beta: coefficient(
                        *self._coefficient_arguments(
                            0.0, beta, mach, 0.0, 0.0, 0.0, 0.0
                        )
                    )
                )
            return sliced.differentiate(0)

        return Function(slope, "Mach", "Coefficient derivative")

    def _get_default_coefficients(self):
        """Returns default coefficients

        Returns
        -------
        default_coefficients: dict
            Dictionary whose keys are the coefficients names and keys
            are the default values.
        """
        default_coefficients = {
            "cL": 0,
            "cQ": 0,
            "cD": 0,
            "cm": 0,
            "cn": 0,
            "cl": 0,
        }
        return default_coefficients

    def _complete_coefficients(self, input_coefficients, default_coefficients):
        """Creates a copy of the input coefficients dict and fill it with missing
        keys with default values

        Parameters
        ----------
        input_coefficients : str, dict
            Coefficients dictionary passed by the user. If the user only specifies some
            of the coefficients, the remaining are completed with class default
            values
        default_coefficients : dict
            Default coefficients of the class

        Returns
        -------
        coefficients : dict
            Coefficients dictionary used to setup coefficient attributes
        """
        coefficients = copy.deepcopy(input_coefficients)
        for coeff, value in default_coefficients.items():
            if coeff not in coefficients.keys():
                coefficients[coeff] = value

        return coefficients

    def _check_coefficients(self, input_coefficients, default_coefficients):
        """Check if input coefficients have only valid keys

        Parameters
        ----------
        input_coefficients : str, dict
            Coefficients dictionary passed by the user. If the user only specifies some
            of the coefficients, the remaining are completed with class default
            values
        default_coefficients : dict
            Default coefficients of the class

        Raises
        ------
        ValueError
            Raises a value error if the input coefficient has an invalid key
        """
        invalid_keys = set(input_coefficients) - set(default_coefficients)
        if invalid_keys:
            raise ValueError(
                f"Invalid coefficient name(s) used in key(s): {', '.join(invalid_keys)}. "
                "Check the documentation for valid names."
            )

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
        alpha_dot=0.0,
        beta_dot=0.0,
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
        # Precompute common values
        dyn_pressure_area = 0.5 * rho * stream_speed**2 * self.reference_area
        dyn_pressure_area_length = dyn_pressure_area * self.reference_length

        # Coefficient arguments (base 7 vars, plus any extra axes appended by
        # subclasses such as control deflections or the unsteady alpha_dot/
        # beta_dot terms).
        args = self._coefficient_arguments(
            alpha,
            beta,
            mach,
            reynolds,
            pitch_rate,
            yaw_rate,
            roll_rate,
            alpha_dot,
            beta_dot,
        )

        # Compute aerodynamic forces
        lift = dyn_pressure_area * self.cL(*args)
        side = dyn_pressure_area * self.cQ(*args)
        drag = dyn_pressure_area * self.cD(*args)

        # Compute aerodynamic moments
        pitch = dyn_pressure_area_length * self.cm(*args)
        yaw = dyn_pressure_area_length * self.cn(*args)
        roll = dyn_pressure_area_length * self.cl(*args)

        return lift, side, drag, pitch, yaw, roll

    def _coefficient_arguments(
        self,
        alpha,
        beta,
        mach,
        reynolds,
        pitch_rate,
        yaw_rate,
        roll_rate,
        alpha_dot=0.0,
        beta_dot=0.0,
    ):
        """Returns the argument tuple passed to every coefficient ``Function``,
        in ``self.independent_vars`` order. The base class provides the seven
        standard inputs, plus ``alpha_dot``/``beta_dot`` when ``unsteady_aero``
        is enabled. Subclasses (e.g. :class:`ControllableGenericSurface`)
        override this to append further axes such as control deflections.
        """
        base = (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
        if self._unsteady_aero:
            return base + (alpha_dot, beta_dot)
        return base

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        density,
        dynamic_viscosity,
        z,
        alpha_dot=0.0,
        beta_dot=0.0,
    ):
        """Computes the forces and moments acting on the aerodynamic surface.
        Used in each time step of the simulation.  This method is valid for
        both linear and nonlinear aerodynamic coefficients.

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
        density : Function
            Atmospheric density as a function of altitude. Used to compute the
            Reynolds number at the surface altitude.
        dynamic_viscosity : Function
            Atmospheric dynamic viscosity as a function of altitude. Used to
            compute the Reynolds number at the surface altitude.
        z : float
            Altitude of the surface, used to evaluate ``density`` and
            ``dynamic_viscosity``.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        # Reynolds number at the surface altitude. Computed here (rather than in
        # the flight loop) since it is only needed by generic surfaces.
        comp_density = density.get_value_opt(z)
        comp_dynamic_viscosity = dynamic_viscosity.get_value_opt(z)
        reynolds = (
            comp_density * stream_speed * self.reference_length / comp_dynamic_viscosity
            if comp_dynamic_viscosity > 0
            else 0
        )

        # Stream velocity in standard aerodynamic frame
        stream_velocity = -stream_velocity

        # Angles of attack and sideslip
        alpha = np.arctan2(stream_velocity[1], stream_velocity[2])
        beta = np.arctan2(stream_velocity[0], stream_velocity[2])

        # Non-dimensionalize the body angular rates into the conventional reduced
        # rates (e.g. ``q* = q * L_ref / (2 * V)``) before evaluating the
        # coefficients, so coefficient tables follow the standard aerotable
        # convention (Missile DATCOM, OpenVSP, CFD/wind-tunnel data tabulate rate
        # derivatives against the reduced rates). The factor is 0 at zero airspeed
        # (pad/static) to avoid division by zero; there is no aerodynamic damping
        # there anyway.
        reduced_rate_factor = (
            self.reference_length / (2 * stream_speed) if stream_speed > 0 else 0.0
        )

        # Compute aerodynamic forces and moments
        lift, side, drag, pitch, yaw, roll = self._compute_from_coefficients(
            rho,
            stream_speed,
            alpha,
            beta,
            stream_mach,
            reynolds,
            omega[0] * reduced_rate_factor,  # q*  reduced pitch rate
            omega[1] * reduced_rate_factor,  # r*  reduced yaw rate
            omega[2] * reduced_rate_factor,  # p*  reduced roll rate
            alpha_dot,
            beta_dot,
        )

        # Conversion from the aerodynamic frame to the body frame. This is the
        # direction cosine matrix (DCM) that expresses the aerodynamic-frame
        # force components in the body frame, i.e. rotations by ``-alpha`` about
        # x and ``+beta`` about y.
        rotation_matrix = Matrix(
            [
                [1, 0, 0],
                [0, math.cos(alpha), math.sin(alpha)],
                [0, -math.sin(alpha), math.cos(alpha)],
            ]
        ) @ Matrix(
            [
                [math.cos(beta), 0, math.sin(beta)],
                [0, 1, 0],
                [-math.sin(beta), 0, math.cos(beta)],
            ]
        )
        R1, R2, R3 = rotation_matrix @ Vector([side, -lift, -drag])

        # Dislocation of the aerodynamic application point to CDM
        M1, M2, M3 = Vector([pitch, yaw, roll]) + (cp ^ Vector([R1, R2, R3]))

        return R1, R2, R3, M1, M2, M3
