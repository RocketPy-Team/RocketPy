import inspect
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
from rocketpy.tools import from_hex_decode, to_hex_encode


def _as_function(func, independent_vars, name):
    """Wrap a variadic callable as a :class:`Function` over ``independent_vars``.

    ``Function`` reads its domain dimension from the callable's parameter count,
    so a variadic wrapper is given an explicit signature to advertise one
    parameter per independent variable.
    """
    func.__signature__ = inspect.Signature(
        inspect.Parameter(var, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for var in independent_vars
    )
    return Function(func, list(independent_vars), [name])


def wind_to_body_coefficients(c_lift, c_drag, c_side, independent_vars):
    """Rotate wind-frame force coefficients into the body frame.

    Given the lift, drag and side-force coefficients (each callable over the
    surface's independent-variable tuple, with the angle of attack and sideslip
    as the first two variables), return the body-frame normal, side and axial
    coefficients ``(cN, cY, cA)`` as :class:`Function`s over the same variables.
    """
    lift, drag, side = c_lift.get_value_opt, c_drag.get_value_opt, c_side.get_value_opt

    def normal(*args):
        alpha, beta = args[0], args[1]
        transverse = math.sin(beta) * side(*args) + math.cos(beta) * drag(*args)
        return math.cos(alpha) * lift(*args) + math.sin(alpha) * transverse

    def yaw_side(*args):
        beta = args[1]
        return math.cos(beta) * side(*args) - math.sin(beta) * drag(*args)

    def axial(*args):
        alpha, beta = args[0], args[1]
        transverse = math.sin(beta) * side(*args) + math.cos(beta) * drag(*args)
        return -math.sin(alpha) * lift(*args) + math.cos(alpha) * transverse

    return (
        _as_function(normal, independent_vars, "cN"),
        _as_function(yaw_side, independent_vars, "cY"),
        _as_function(axial, independent_vars, "cA"),
    )


def body_to_wind_coefficients(c_normal, c_side, c_axial, independent_vars):
    """Rotate body-frame force coefficients into the wind frame.

    Inverse of :func:`wind_to_body_coefficients`: given the body-frame normal,
    side and axial coefficients, return the wind-frame lift, drag and
    side-force coefficients ``(cL, cD, cQ)`` as :class:`Function`s.
    """
    normal = c_normal.get_value_opt
    side = c_side.get_value_opt
    axial = c_axial.get_value_opt

    def lift(*args):
        alpha = args[0]
        return math.cos(alpha) * normal(*args) - math.sin(alpha) * axial(*args)

    def drag(*args):
        alpha, beta = args[0], args[1]
        longitudinal = math.sin(alpha) * normal(*args) + math.cos(alpha) * axial(*args)
        return -math.sin(beta) * side(*args) + math.cos(beta) * longitudinal

    def yaw_side(*args):
        alpha, beta = args[0], args[1]
        longitudinal = math.sin(alpha) * normal(*args) + math.cos(alpha) * axial(*args)
        return math.cos(beta) * side(*args) + math.sin(beta) * longitudinal

    return (
        _as_function(lift, independent_vars, "cL"),
        _as_function(drag, independent_vars, "cD"),
        _as_function(yaw_side, independent_vars, "cQ"),
    )


class GenericSurface:
    """Defines a generic aerodynamic surface with custom force and moment
    coefficients. The coefficients can be nonlinear functions of the angle of
    attack, sideslip angle, Mach number, Reynolds number, pitch rate, yaw rate
    and roll rate."""

    # Whether this surface contributes identically to the pitch and yaw planes.
    # ``False`` for a generic surface (its coefficients may differ between planes)
    is_axisymmetric = False

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Generic Surface",
        reynolds_length=None,
        interpolation=None,
        extrapolation=None,
        force_convention=None,
        active_during="always",
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

        The Reynolds number ("reynolds") is by default built on the reference
        length (the rocket diameter). Published rocket data and tools often base
        Reynolds on the **body length** instead, which for a slender rocket is
        much larger (Re scales with the chosen length). If your coefficient
        table uses a different length than the reference length, pass that
        length as ``reynolds_length`` so the Reynolds number the simulation
        feeds your table matches the one it was built against.

        The angular-rate inputs ("pitch_rate", "yaw_rate", "roll_rate") are the
        conventional **non-dimensional reduced rates**,
        ``q* = q * L_ref / (2 * V)`` (and likewise for ``r``/``p``).
        Provide coefficient tables against the reduced rates, not the raw body
        rates in rad/s.

        See Also
        --------
        :ref:`genericsurfaces`.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface, in meters. Commonly the
            rocket's diameter. Used to non-dimensionalize the moment coefficients
            and the reduced rotation rates, and (unless ``reynolds_length`` is
            given) as the length scale of the Reynolds number.
        coefficients: dict
            The six force and moment coefficients, by name. Any you leave out are
            set to 0. Each one can be a constant number, a function of the flow
            variables, a list of data points, or a path to a CSV file. By default
            the force coefficients are the body-frame ones (see
            ``force_convention``); the wind-frame names ``cL``/``cQ``/``cD`` are
            also accepted. The coefficients are:\n
            cN: str, callable, optional
                Normal force coefficient (body frame). Default is 0.\n
            cY: str, callable, optional
                Side force coefficient (body frame). Default is 0.\n
            cA: str, callable, optional
                Axial force coefficient (body frame). Default is 0.\n
            cm: str, callable, optional
                Pitch moment coefficient. Default is 0.\n
            cn: str, callable, optional
                Yaw moment coefficient. Default is 0.\n
            cl: str, callable, optional
                Roll moment coefficient. Default is 0.\n
        center_of_pressure : tuple, list, optional
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str, optional
            Name of the aerodynamic surface. Default is 'Generic Surface'.
        reynolds_length : int, float, optional
            Length scale, in meters, of the Reynolds number passed to the
            coefficients. Set it to the length your Reynolds-dependent
            coefficient data was tabulated against (for example the rocket's
            body length, if your table uses a length-based Reynolds number).
            ``None`` (the default) uses ``reference_length`` (the diameter). Has
            no effect unless a coefficient actually depends on "reynolds".
        interpolation : str or dict, optional
            How tabulated coefficients interpolate between points. The accepted
            methods depend on the coefficient's dimensionality: a 1-D table
            (e.g. a Mach-only curve) accepts ``"linear"``, ``"akima"``,
            ``"spline"`` and ``"polynomial"``; a multi-dimensional scattered
            table accepts ``"linear"``, ``"shepard"`` and ``"rbf"``; and a
            multi-dimensional table on a regular Cartesian grid accepts
            ``"linear"``, ``"nearest"``, ``"slinear"``, ``"cubic"``,
            ``"quintic"`` and ``"pchip"`` (with ``"spline"`` mapped to
            ``"cubic"`` and ``"akima"`` to ``"pchip"``). Pass a single string to
            use that method for every coefficient, or a dict keyed by coefficient
            name to set them individually (coefficients left out of the dict fall
            back to the default). ``None`` (the default) uses ``"linear"`` for
            tables built here and keeps a pre-built ``Function``'s own setting.
        extrapolation : str or dict, optional
            How tabulated coefficients behave outside their data range:
            ``"constant"`` holds the value at the nearest data edge,
            ``"natural"`` keeps following the curve, and ``"zero"`` returns 0.
            Pass a single string to use that method for every coefficient, or a
            dict keyed by coefficient name to set them individually (coefficients
            left out of the dict fall back to the default). ``None`` (the
            default) uses ``"constant"`` for tables built here and keeps whatever
            a pre-built ``Function`` already carries. Only affects tabulated
            sources (constants and callables are evaluated directly).
        force_convention : str, optional
            The frame your force coefficients are given in. ``"wind"`` for the
            wind-frame coefficients ``cL`` (lift), ``cQ`` (side) and
            ``cD`` (drag); ``"body"`` for the body-frame coefficients ``cN``
            (normal), ``cY`` (side) and ``cA`` (axial), the convention used by
            DATCOM, wind tunnels and Barrowman. The moment coefficients
            (``cm``, ``cn``, ``cl``) are the same in both. ``None`` (the default)
            infers the frame from the coefficient names you pass. Whichever frame
            you use, all nine coefficients are available as attributes afterwards
            (the other frame is computed on demand).
        active_during : str or callable, optional
            When this surface produces aerodynamic force during a simulation.
            Use it to model a surface that is only present in part of the flight,
            such as jet vanes that only work while the motor burns, or a base
            drag that only appears after burnout. Accepts:

            - ``"always"`` (default): the surface always contributes force.
            - ``"power_on"``: only while the motor is burning (up to the motor's
              burn-out time).
            - ``"power_off"``: only after the motor has burned out.
            - a function ``active_during(t, flight)`` returning ``True`` when the
              surface is active at time ``t`` (in seconds) of the given
              :class:`Flight`. Use this for any custom window.
        """

        # Externally-supplied axes (e.g. control deflections). Subclasses set
        # this before ``super().__init__``. Defaults to none for plain surfaces.
        self.control_variables = getattr(self, "control_variables", ())
        # Ordered independent variables accepted by every coefficient: the seven
        # base axes, plus any ``control_variables``
        self.independent_vars = build_independent_vars(self.control_variables)

        self.reference_area = reference_area
        self.reference_length = reference_length
        self.reynolds_length = (
            reference_length if reynolds_length is None else reynolds_length
        )
        self.center_of_pressure = center_of_pressure
        self.cp = center_of_pressure
        self.cpx = center_of_pressure[0]
        self.cpy = center_of_pressure[1]
        self.cpz = center_of_pressure[2]
        self.name = name
        self.active_during = self._validate_active_during(active_during)
        self.is_active = self._build_activation_check(self.active_during)

        self._rotation_surface_to_body = self._default_surface_rotation()

        self._build_coefficients(
            coefficients, interpolation, extrapolation, force_convention
        )

        self.evaluate_coefficients()
        self._evaluate_stability_derivatives()

        # Reporting layers. Subclasses override these with their own (more
        # specific) prints/plots after calling ``super().__init__``.
        self.prints = _GenericSurfacePrints(self)
        self.plots = _GenericSurfacePlots(self)

    def _default_surface_rotation(self):
        """Rotation from the surface-local frame to the body frame. It is applied
        to the :attr:`force_application_point` when the rocket locates each
        surface's center of pressure relative to the center of dry mass. A plain
        generic surface takes its center of pressure as already body-aligned
        (the identity); geometry-defined (Barrowman) surfaces override this.
        """
        return Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @staticmethod
    def _validate_active_during(active_during):
        """Check the ``active_during`` policy and return it unchanged.

        Accepts one of the preset strings ``"always"``, ``"power_on"``,
        ``"power_off"`` or a callable ``(t, flight) -> bool``; anything else
        raises a ``ValueError`` so a typo is caught at construction rather than
        silently keeping the surface active.
        """
        if callable(active_during) or active_during in (
            "always",
            "power_on",
            "power_off",
        ):
            return active_during
        raise ValueError(
            "`active_during` must be one of 'always', 'power_on', 'power_off' "
            "or a callable(t, flight) -> bool; "
            f"got {active_during!r}."
        )

    @staticmethod
    def _build_activation_check(active_during):
        """Resolve an ``active_during`` policy into the ``is_active(t, flight)``
        function the flight integrator calls for every surface each step to skip
        the ones that are not currently active.

        Resolving it once here keeps that per-step check free of policy
        branching. A custom callable is used unchanged; each preset becomes a
        small function of the simulation time ``t`` (in seconds) and the
        ``flight`` being run, and ``"always"`` becomes a function that simply
        returns ``True``.
        """
        if callable(active_during):
            return active_during
        if active_during == "power_on":
            return lambda t, flight: t < flight.rocket.motor.burn_out_time
        if active_during == "power_off":
            return lambda t, flight: t >= flight.rocket.motor.burn_out_time
        return lambda t, flight: True  # "always"

    @property
    def force_application_point(self):
        """Local point (surface frame) at which the resultant force is applied
        when transporting its moment to the rocket's center of dry mass. This is
        the center of pressure ``self.cp``; any residual couple is carried by the
        ``cm``/``cn``/``cl`` coefficients.
        """
        return Vector([self.cpx, self.cpy, self.cpz])

    @property
    def cL(self):
        """Wind-frame lift coefficient, as a :class:`Function` of the surface's
        independent variables. Derived from the canonical body-frame ``cN``,
        ``cY`` and ``cA`` by the angle-of-attack/sideslip rotation."""
        return body_to_wind_coefficients(
            self.cN, self.cY, self.cA, self.independent_vars
        )[0]

    @property
    def cD(self):
        """Wind-frame drag coefficient (derived from ``cN``/``cY``/``cA``)."""
        return body_to_wind_coefficients(
            self.cN, self.cY, self.cA, self.independent_vars
        )[1]

    @property
    def cQ(self):
        """Wind-frame side-force coefficient (derived from ``cN``/``cY``/``cA``)."""
        return body_to_wind_coefficients(
            self.cN, self.cY, self.cA, self.independent_vars
        )[2]

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

    def _evaluate_stability_derivatives(self):
        """Compute the coefficient derivatives used for stability and store them
        as the ``cN_alpha``, ``cm_alpha``, ``cY_beta`` and ``cn_beta``
        attributes, then build the center-of-pressure accessors from them.

        A plain generic surface recovers each derivative from its body-frame
        force and moment coefficients by numerical differentiation at
        ``alpha = beta = 0`` with zero rates. The Barrowman surfaces instead set
        these four attributes directly from geometry and only reuse
        :meth:`_set_stability_accessors` (see the :class:`LinearGenericSurface`
        override).

        Returns
        -------
        None
        """
        self.cN_alpha = AeroCoefficient(
            self.cN.slope("alpha", "mach"),
            depends_on=("mach",),
            control_variables=self.control_variables,
            name="cN_alpha",
        )
        self.cm_alpha = AeroCoefficient(
            self.cm.slope("alpha", "mach"),
            depends_on=("mach",),
            control_variables=self.control_variables,
            name="cm_alpha",
        )
        self.cY_beta = AeroCoefficient(
            self.cY.slope("beta", "mach"),
            depends_on=("mach",),
            control_variables=self.control_variables,
            name="cY_beta",
        )
        self.cn_beta = AeroCoefficient(
            self.cn.slope("beta", "mach"),
            depends_on=("mach",),
            control_variables=self.control_variables,
            name="cn_beta",
        )
        self._set_stability_accessors()

    def _set_stability_accessors(self):
        """Build the pitch- and yaw-plane center-of-pressure accessors from the
        stored coefficient derivatives (``cN_alpha``/``cm_alpha`` and
        ``cY_beta``/``cn_beta``), each evaluated at ``alpha = beta = 0`` with
        zero rates.

        Each accessor is a Mach-only :class:`Function` giving the surface's
        center of pressure along the body z-axis. It combines the surface's
        local application point with the offset implied by its moment
        coefficient (``cp = application point - (moment slope / force slope) *
        L_ref``). When a surface produces no force at some Mach the center of
        pressure is undefined, so it falls back to the geometric application
        point and drops out of the force-weighted average.

        Returns
        -------
        None
        """
        reference_length = self.reference_length
        local_cpz = self.force_application_point[2]

        def _cp_z(force_coeff, moment_coeff):
            def cp_z(mach):
                slope = force_coeff.get_value_opt(0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0)
                if slope == 0:
                    return local_cpz
                moment = moment_coeff.get_value_opt(0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0)
                return local_cpz - moment / slope * reference_length

            return Function(cp_z, "Mach", "Center of pressure to local origin (m)")

        self.center_of_pressure_z = _cp_z(self.cN_alpha, self.cm_alpha)
        self.center_of_pressure_z_yaw = _cp_z(self.cY_beta, self.cn_beta)

    @staticmethod
    def _coefficient_option(option, coeff_name):
        """Resolve a per-coefficient interpolation/extrapolation setting.

        ``option`` may be a single value applied to every coefficient, a dict
        mapping coefficient names to values (coefficients absent from the dict
        fall back to the ``AeroCoefficient`` default), or ``None``.

        Parameters
        ----------
        option : str, dict, or None
            The interpolation/extrapolation argument passed to ``__init__``.
        coeff_name : str
            Name of the coefficient being built (e.g. ``"cD"``, ``"cm_alpha"``).

        Returns
        -------
        str or None
            The value to forward to :class:`AeroCoefficient` for this coefficient.
        """
        if isinstance(option, dict):
            return option.get(coeff_name)
        return option

    # Force-coefficient names in each frame. Moments (cm/cn/cl) are frame-shared.
    _WIND_FORCE_NAMES = ("cL", "cQ", "cD")
    _BODY_FORCE_NAMES = ("cN", "cY", "cA")

    def _force_frames_present(self, coefficients):
        """Report which force frames the input coefficient names belong to, as
        ``(has_wind, has_body)``.

        A generic surface matches the plain force names (``cL``/``cQ``/``cD`` for
        wind, ``cN``/``cY``/``cA`` for body). The linear model overrides this to
        match those same names as derivative prefixes (``cL_alpha`` ...).
        """
        keys = set(coefficients)
        has_wind = bool(keys & set(self._WIND_FORCE_NAMES))
        has_body = bool(keys & set(self._BODY_FORCE_NAMES))
        return has_wind, has_body

    def _resolve_force_convention(self, coefficients, force_convention):
        """Decide whether the input force coefficients are given in the wind
        frame (``cL``/``cQ``/``cD``) or the body frame (``cN``/``cY``/``cA``).

        When ``force_convention`` is ``None`` the frame is inferred from the
        coefficient names; mixing the two frames is rejected. With no force
        coefficients to infer from, the canonical body frame is assumed.
        """
        has_wind, has_body = self._force_frames_present(coefficients)
        if force_convention is None:
            if has_wind and has_body:
                raise ValueError(
                    "Mixed wind (cL/cQ/cD) and body (cN/cY/cA) force "
                    "coefficients; pass force_convention='wind' or 'body'."
                )
            return "wind" if has_wind else "body"
        if force_convention not in ("wind", "body"):
            raise ValueError(
                f"force_convention must be 'wind' or 'body', got {force_convention!r}."
            )
        return force_convention

    def _wind_input_to_body(self, coefficients):
        """Convert a wind-frame force-coefficient input (``cL``/``cQ``/``cD``)
        into the canonical body-frame coefficients (``cN``/``cY``/``cA``),
        leaving the moment coefficients untouched."""
        wind = {}
        passthrough = {}
        for name, value in coefficients.items():
            if name in self._WIND_FORCE_NAMES:
                wind[name] = value
            else:
                passthrough[name] = value

        def as_coefficient(source, name):
            return AeroCoefficient(
                source,
                control_variables=self.control_variables,
                name=name,
            )

        c_normal, c_yaw, c_axial = wind_to_body_coefficients(
            as_coefficient(wind.get("cL", 0), "cL"),
            as_coefficient(wind.get("cD", 0), "cD"),
            as_coefficient(wind.get("cQ", 0), "cQ"),
            self.independent_vars,
        )
        return {"cN": c_normal, "cY": c_yaw, "cA": c_axial, **passthrough}

    def _build_coefficients(
        self, coefficients, interpolation, extrapolation, force_convention
    ):
        """Resolve the force-coefficient frame and store the surface's
        aerodynamic coefficients as :class:`AeroCoefficient` attributes.

        Runs the full coefficient setup from the user input: picks the force
        frame, converts a wind-frame input to the canonical body frame, fills in
        any coefficient the user left out with its default (0), and stores each
        one as an attribute (``self.cN``, ``self.cm``, ...).

        Parameters
        ----------
        coefficients : dict
            The user-provided coefficients (see :meth:`__init__`).
        interpolation, extrapolation : str, dict, or None
            The interpolation/extrapolation settings (see :meth:`__init__`).
        force_convention : str or None
            The frame the input force coefficients are given in, or ``None`` to
            infer it from the coefficient names.
        """
        default_coefficients = self._get_default_coefficients()
        self.force_convention = self._resolve_force_convention(
            coefficients, force_convention
        )
        # Wind-frame force input (cL/cQ/cD) is converted once to the canonical
        # body-frame coefficients before validation. Each surface supplies the
        # conversion appropriate to its coefficients: the generic surface rotates
        # the full force coefficients, while the linear model recombines the
        # coefficient derivatives (see LinearGenericSurface._wind_input_to_body).
        # A non-dict input falls through to _check_coefficients, which rejects it.
        if self.force_convention == "wind" and isinstance(coefficients, dict):
            coefficients = self._wind_input_to_body(coefficients)
        self._check_coefficients(coefficients, default_coefficients)
        coefficients = self._complete_coefficients(coefficients, default_coefficients)

        # ``_needs_reynolds`` lets the flight loop skip the per-step atmosphere
        # lookups when no coefficient uses the Reynolds number. Only these
        # primary coefficients are checked: they are what the surface evaluates,
        # and the linear model's combined coefficients are linear combinations of
        # them, so a Reynolds dependence always shows up here.
        self._needs_reynolds = False
        for coeff, coeff_value in coefficients.items():
            value = AeroCoefficient(
                coeff_value,
                control_variables=self.control_variables,
                name=coeff,
                extrapolation=self._coefficient_option(extrapolation, coeff),
                interpolation=self._coefficient_option(interpolation, coeff),
            )
            setattr(self, coeff, value)
            if "reynolds" in value.depends_on:
                self._needs_reynolds = True

    def _get_default_coefficients(self):
        """Returns default coefficients

        Returns
        -------
        default_coefficients: dict
            Dictionary whose keys are the coefficients names and keys
            are the default values.
        """
        default_coefficients = {
            "cN": 0,
            "cY": 0,
            "cA": 0,
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
        # Shallow copy: only missing keys are added, so the user's dict is left
        # intact. The values are not mutated here (each is wrapped in an
        # AeroCoefficient, which copies it when it needs its own settings), so
        # there is no need to deep-copy potentially large tabulated coefficients.
        coefficients = dict(input_coefficients)
        for coeff, value in default_coefficients.items():
            if coeff not in coefficients:
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
            The body-frame force components ``(R1, R2, R3)`` and the moments
            ``(pitch, yaw, roll)``.
        """
        # Precompute common values
        dyn_pressure_area = 0.5 * rho * stream_speed**2 * self.reference_area
        dyn_pressure_area_length = dyn_pressure_area * self.reference_length

        # Coefficient arguments (base 7 vars, plus any extra axes appended by
        # subclasses such as control deflections).
        args = self._coefficient_arguments(
            alpha,
            beta,
            mach,
            reynolds,
            pitch_rate,
            yaw_rate,
            roll_rate,
        )

        # Body-frame force components straight from the body-frame coefficients
        # (normal cN, side cY, axial cA); no wind-to-body rotation needed.
        normal = dyn_pressure_area * self.cN(*args)
        yaw_side = dyn_pressure_area * self.cY(*args)
        axial = dyn_pressure_area * self.cA(*args)
        R1 = yaw_side
        R2 = -normal
        R3 = -axial

        # Compute aerodynamic moments
        pitch = dyn_pressure_area_length * self.cm(*args)
        yaw = dyn_pressure_area_length * self.cn(*args)
        roll = dyn_pressure_area_length * self.cl(*args)

        return R1, R2, R3, pitch, yaw, roll

    def _coefficient_arguments(
        self,
        alpha,
        beta,
        mach,
        reynolds,
        pitch_rate,
        yaw_rate,
        roll_rate,
    ):
        """Returns the argument tuple passed to every coefficient ``Function``,
        in ``self.independent_vars`` order. The base class provides the seven
        standard inputs. Subclasses (e.g. :class:`ControllableGenericSurface`)
        override this to append further axes such as control deflections.
        """
        return (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)

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
        # the flight loop) since it is only needed by generic surfaces, and only
        # when a coefficient actually depends on it -- otherwise the two
        # atmosphere lookups are skipped for every surface, every step.
        if self._needs_reynolds:
            comp_density = density.get_value_opt(z)
            comp_dynamic_viscosity = dynamic_viscosity.get_value_opt(z)
            reynolds = (
                comp_density
                * stream_speed
                * self.reynolds_length
                / comp_dynamic_viscosity
                if comp_dynamic_viscosity > 0
                else 0
            )
        else:
            reynolds = 0.0

        # Stream velocity in standard wind frame
        stream_velocity = -stream_velocity

        # Angles of attack and sideslip
        alpha = np.arctan2(stream_velocity[1], stream_velocity[2])
        beta = np.arctan2(stream_velocity[0], stream_velocity[2])

        # Non-dimensionalize the body angular rates into the conventional reduced
        # rates (e.g. ``q* = q * L_ref / (2 * V)``).
        reduced_rate_factor = (
            self.reference_length / (2 * stream_speed) if stream_speed > 0 else 0.0
        )

        # Body-frame force components and moments straight from the body-frame
        # coefficients (no wind-to-body rotation: the coefficients already live
        # in the body frame). ``alpha``/``beta`` are still passed to the
        # coefficients, they just no longer rotate the force.
        R1, R2, R3, pitch, yaw, roll = self._compute_from_coefficients(
            rho,
            stream_speed,
            alpha,
            beta,
            stream_mach,
            reynolds,
            omega[0] * reduced_rate_factor,  # q*  reduced pitch rate
            omega[1] * reduced_rate_factor,  # r*  reduced yaw rate
            omega[2] * reduced_rate_factor,  # p*  reduced roll rate
        )

        # Dislocation of the aerodynamic application point to CDM
        M1, M2, M3 = Vector([pitch, yaw, roll]) + (cp ^ Vector([R1, R2, R3]))

        return R1, R2, R3, M1, M2, M3

    def to_dict(self, include_outputs=False, **kwargs):  # pylint: disable=unused-argument
        # The stored coefficients are always the canonical body-frame set (the
        # names from ``_get_default_coefficients``: cN/cY/cA/... for a generic
        # surface, the derivative set for the linear model), so they are saved
        # with ``force_convention="body"`` and rebuilt directly on load.
        coefficients = {
            name: getattr(self, name) for name in self._get_default_coefficients()
        }
        # A preset ``active_during`` is stored as is; a custom (t, flight) -> bool
        # function is pickled to text when allowed, otherwise dropped to "always"
        # (a function cannot be restored without pickling).
        active_during = self.active_during
        if callable(active_during):
            active_during = (
                to_hex_encode(active_during)
                if kwargs.get("allow_pickle", True)
                else "always"
            )
        return {
            "reference_area": self.reference_area,
            "reference_length": self.reference_length,
            "reynolds_length": self.reynolds_length,
            "coefficients": coefficients,
            "center_of_pressure": self.center_of_pressure,
            "name": self.name,
            "force_convention": "body",
            "active_during": active_during,
        }

    @classmethod
    def from_dict(cls, data):
        # A preset ``active_during`` is used as is; anything else is unpickled
        # back into the original function (falling back to "always" if it cannot
        # be restored).
        active_during = data.get("active_during", "always")
        if active_during not in ("always", "power_on", "power_off"):
            try:
                active_during = from_hex_decode(active_during)
            except (TypeError, ValueError):
                active_during = "always"
        return cls(
            reference_area=data["reference_area"],
            reference_length=data["reference_length"],
            coefficients=data["coefficients"],
            center_of_pressure=data.get("center_of_pressure", (0, 0, 0)),
            name=data.get("name", "Generic Surface"),
            reynolds_length=data.get("reynolds_length"),
            force_convention=data.get("force_convention", "body"),
            active_during=active_during,
        )

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
