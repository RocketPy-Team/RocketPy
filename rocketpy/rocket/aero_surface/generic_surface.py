import csv

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.plots.aero_surface_plots import _GenericSurfacePlots
from rocketpy.prints.aero_surface_prints import _GenericSurfacePrints
from rocketpy.rocket.aero_surface._helpers import (
    _as_function,
    _body_to_wind_coefficients,
    _total_angle_to_body_coefficients,
    _wind_plane_lift_to_body_coefficients,
    _wind_to_body_coefficients,
)
from rocketpy.rocket.aero_surface.aero_coefficient import (
    SOURCE_ONLY_NAMES,
    AeroCoefficient,
    build_independent_vars,
)


class GenericSurface:
    """An aerodynamic surface of the rocket, defined by its force and moment
    coefficients.

    Every aerodynamic part of a rocket is a ``GenericSurface``. The nose cone,
    the fins and the tail work out their coefficients from their geometry, while
    a ``GenericSurface`` created directly takes the coefficients you give it,
    from wind-tunnel data, CFD or a model of your own. The coefficients can be
    nonlinear functions of the angle of attack, sideslip angle, Mach number,
    Reynolds number, pitch rate, yaw rate and roll rate.

    Attributes
    ----------
    GenericSurface.reference_area : float
        Reference area, in square meters.
    GenericSurface.reference_length : float
        Reference length, in meters.
    GenericSurface.reynolds_length : float
        Length scale of the Reynolds number, in meters.
    GenericSurface.name : str
        Name of the surface.
    GenericSurface.center_of_pressure : tuple
        Point where the forces are applied, ``(x, y, z)`` in meters, as given.
    GenericSurface.cp : tuple
        The same point as ``(cpx, cpy, cpz)``. ``cpz`` is 0 when the center of
        pressure varies with Mach, since the force is then applied at ``z = 0``.
    GenericSurface.cpx : float
        x coordinate of ``cp``, in meters.
    GenericSurface.cpy : float
        y coordinate of ``cp``, in meters.
    GenericSurface.cpz : float
        z coordinate of ``cp``, in meters.
    GenericSurface.active_during : str
        The motor phase the surface produces force in, as given.
    GenericSurface.active : bool
        Whether the surface starts each flight switched on.
    GenericSurface.force_convention : str
        Frame the force coefficients were given in: ``"body"`` or ``"wind"``.
    GenericSurface.independent_vars : list of str
        The variables every coefficient is called with, in order.
    GenericSurface.is_axisymmetric : bool
        Whether the surface acts the same in the pitch and yaw planes. False
        here, since given coefficients may differ between the planes.
    GenericSurface.cN : AeroCoefficient
        Normal force coefficient, the force in the pitch plane.
    GenericSurface.cY : AeroCoefficient
        Side force coefficient, the force in the yaw plane.
    GenericSurface.cA : AeroCoefficient
        Axial force coefficient, the force along the rocket's axis.
    GenericSurface.cm : AeroCoefficient
        Pitching moment coefficient, about ``center_of_pressure``.
    GenericSurface.cn : AeroCoefficient
        Yawing moment coefficient, about ``center_of_pressure``.
    GenericSurface.cl : AeroCoefficient
        Roll moment coefficient.
    GenericSurface.cL : Function
        Lift coefficient, the force perpendicular to the airflow.
    GenericSurface.cD : Function
        Drag coefficient, the force along the airflow.
    GenericSurface.cQ : Function
        Crosswind coefficient, the side force relative to the airflow.
    GenericSurface.cN_alpha : AeroCoefficient
        Slope of ``cN`` with the angle of attack, as a function of Mach.
        Has units of 1/rad.
    GenericSurface.cY_beta : AeroCoefficient
        Slope of ``cY`` with the sideslip angle, as a function of Mach.
        Has units of 1/rad.
    GenericSurface.cm_alpha : AeroCoefficient
        Slope of ``cm`` with the angle of attack, as a function of Mach.
        Has units of 1/rad.
    GenericSurface.cn_beta : AeroCoefficient
        Slope of ``cn`` with the sideslip angle, as a function of Mach.
        Has units of 1/rad.
    GenericSurface.aerodynamic_center : Function
        Pitch-plane aerodynamic center of the surface along the rocket's axis,
        in meters from the surface's position (positive toward the nose), as a
        function of Mach. It is the point where the change of the normal force
        acts when the angle of attack changes a little from zero. It is the
        same point as ``center_of_pressure`` when no moment coefficient is
        given.
    GenericSurface.aerodynamic_center_yaw : Function
        Yaw-plane aerodynamic center of the surface along the rocket's axis, in
        meters from the surface's position (positive toward the nose), as a
        function of Mach: the same for the side force and the sideslip angle.
    GenericSurface.prints : _GenericSurfacePrints
        The prints of the surface. Use help(GenericSurface.prints) to know more.
    GenericSurface.plots : _GenericSurfacePlots
        The plots of the surface. Use help(GenericSurface.plots) to know more.
    """

    # Whether this surface contributes identically to the pitch and yaw planes.
    # ``False`` for a generic surface (its coefficients may differ between planes)
    is_axisymmetric = False

    # Counts changes to the surface (its geometry, its center of pressure, its
    # orientation), so a rocket using it can tell when to update what it
    # derived from it. Every setter that changes the surface adds one.
    _version = 0

    # The frames the force coefficients can be given in
    _FORCE_CONVENTIONS = ("body", "wind")
    # Body-frame coefficients that were given in the plane of the wind (against
    # the total angle of attack) and split between the pitch and yaw planes
    _wind_plane_names = frozenset()
    # Mach numbers at which a coefficient in the plane of the wind is checked to vanish at
    # zero total angle of attack
    _ZERO_ANGLE_MACHS = (0.0, 0.3, 0.9, 2.0)
    # Step of the numerical stability slopes, in radians
    _SLOPE_STEP = 1e-6
    # Coefficients that have a direction across the rocket's axis. Given against
    # the total angle of attack alone, they act in the plane of the wind.
    _DIRECTIONAL_COEFFICIENTS = ("cN", "cY", "cm", "cn", "cL", "cQ")

    # Force-coefficient names in each frame. Moments (cm/cn/cl) are frame-shared.
    _WIND_FORCE_NAMES = ("cL", "cQ", "cD")
    _BODY_FORCE_NAMES = ("cN", "cY", "cA")
    # Rate derivatives, such as ``cm_q``: the reduced rate each suffix multiplies
    _RATE_DERIVATIVES = {"p": "roll_rate", "q": "pitch_rate", "r": "yaw_rate"}

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Generic Surface",
        *,
        reynolds_length=None,
        interpolation=None,
        extrapolation=None,
        force_convention=None,
        active_during="always",
        active=True,
    ):
        """Create an aerodynamic surface from its aerodynamic coefficients.

        Use it for a part that does not fit the predefined classes (nose cone,
        fins, tail, ...), or for a model of the whole rocket.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the surface, in square meters. Commonly the
            rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the surface, in meters. Commonly the rocket's
            diameter. Used to non-dimensionalize the moment coefficients and the
            reduced rotation rates, and (unless ``reynolds_length`` is given) as
            the length scale of the Reynolds number.
        coefficients : dict
            The force and moment coefficients, by name. Any you leave out are 0.

            - ``cN``, ``cY``, ``cA``: normal, side and axial force coefficients,
              in the body frame. The wind-frame ``cL`` (lift), ``cQ`` (side
              force) and ``cD`` (drag) can be given instead (see
              ``force_convention``).
            - ``cm``, ``cn``, ``cl``: pitch, yaw and roll moment coefficients,
              taken about ``center_of_pressure``.
            - A rate derivative, to add damping: a body-frame coefficient's name
              followed by ``_p``, ``_q`` or ``_r`` (roll, pitch or yaw rate),
              such as ``cm_q``. It is multiplied by the reduced rate and added
              to the coefficient, so damping is a negative number (see
              :ref:`generic_surface_damping`).

            Each coefficient can depend on these variables:

            - ``alpha``, ``beta``: angle of attack and sideslip angle, in
              radians, or ``alpha_deg``, ``beta_deg`` in degrees.
            - ``alpha_total``, ``phi``: total angle of attack (the angle
              between the rocket's axis and the air) and roll angle of the
              wind, in radians, or ``alpha_total_deg``, ``phi_deg`` in degrees.
              A ``cN``, ``cL`` or ``cm`` given against ``alpha_total`` alone is
              split between the pitch and yaw planes for you, and must be zero
              at zero angle (see :ref:`totalangle`).
            - ``mach``: Mach number.
            - ``reynolds``: Reynolds number (see ``reynolds_length``).
            - ``pitch_rate``, ``yaw_rate``, ``roll_rate``: angular rates in
              reduced form, such as ``q * L_ref / (2 * V)`` for the pitch rate,
              not in rad/s.

            Each coefficient can be given as:

            - a number: a constant.
            - a function whose arguments are named after the variables it uses,
              for example ``lambda alpha, mach: ...``.
            - the path to a CSV file whose header names the variables, in any
              order, with the coefficient in the last column.
            - a list or numpy array of data points, as a pair with the names of
              its variables in order, for example
              ``([[0, 0.4], [1, 0.6]], ["mach"])``.
            - a :class:`Function`, whose input names say which variables it
              uses, or as a pair like a list of data points.
            - a pair ``(axes, values)`` for values on a regular grid, for
              example ``({"alpha": alphas, "mach": machs}, values)`` with
              ``values[i, j]`` at ``alphas[i]`` and ``machs[j]``.
            - an :class:`AeroCoefficient`, used as it is.
        center_of_pressure : tuple, list, optional
            Point where the aerodynamic forces are applied and about which the
            moment coefficients are taken, as ``(x, y, z)`` in meters. It is
            measured from the position the surface is added to the rocket at:
            ``z`` runs along the rocket's centerline and is positive toward the
            nose, whichever coordinate system orientation the rocket uses. For example ``(0, 0, -0.3)`` applies
            the force 0.3 m closer to the tail. The default value is (0, 0, 0).

            The ``z`` component may instead vary with Mach, the way programs
            such as OpenRocket or RASAero report the center of pressure: give
            it as a function of Mach (``lambda mach: ...``), a one-input
            :class:`Function`, or a two-column table ``[[mach, z], ...]``. The
            force is then applied at ``z = 0`` and its moment about the given
            center of pressure is carried by the moment coefficients
            (``cm + cN * z(mach) / L_ref`` and ``cn + cY * z(mach) / L_ref``),
            so ``cm`` and ``cn`` are the moments about the center of pressure
            itself, usually 0. Such a center of pressure is fixed when the
            surface is created.
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
            The frame the force coefficients are given in:

            - ``"body"``: ``cN`` (normal), ``cY`` (side) and ``cA`` (axial), the
              convention of wind tunnels and Barrowman.
            - ``"wind"``: ``cL`` (lift), ``cQ`` (side) and ``cD`` (drag).

            The moment coefficients are the same in both frames. ``None`` (the
            default) picks the frame from the coefficient names. Whichever
            frame you use, the body-frame and wind-frame coefficients are all
            available as attributes afterwards.
        active_during : str, optional
            The motor phase this surface produces aerodynamic force in. Use it
            to model a surface that only matters in part of the flight, such as
            jet vanes that only work while the motor burns, or a base drag that
            only appears after burnout. Accepts:

            - ``"always"`` (default): the surface always contributes force.
            - ``"power_on"``: only while the motor is burning (up to the motor's
              burn-out time).
            - ``"power_off"``: only after the motor has burned out.

            To switch a surface on or off at any other moment, such as apogee,
            use an event: see ``active`` below.
        active : bool, optional
            Whether the surface starts each flight switched on. Default is
            ``True``. Use ``False`` for a surface that only appears later in
            the flight, and switch it on from an event with
            ``context.event.commands.activate_surface(surface)``. A surface
            that is on from the start is switched off the same way, with
            ``deactivate_surface``.

        Raises
        ------
        TypeError
            If ``coefficients`` is not a dict, or ``center_of_pressure`` is not
            an ``(x, y, z)`` triple.
        ValueError
            If a coefficient name, a variable name, ``force_convention`` or
            ``active_during`` is not one of the accepted values, if wind-frame
            and body-frame force coefficients are mixed without a
            ``force_convention``, or if a coefficient given against the total
            angle of attack breaks the rules above.

        See Also
        --------
        :ref:`genericsurfaces`
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
        self._set_center_of_pressure(center_of_pressure)
        self.name = name
        self.active_during = self._validate_active_during(active_during)
        self.active = bool(active)

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

    @property
    def center_of_pressure(self):
        """The point the forces are applied at and the moments are taken about,
        as ``(x, y, z)`` in meters in the surface's frame (see
        :meth:`__init__`). Setting it moves the surface's force application
        point, and a rocket holding the surface updates its aerodynamic
        center."""
        return self._center_of_pressure

    @center_of_pressure.setter
    def center_of_pressure(self, value):
        self._set_center_of_pressure(value)
        self._version += 1

    def _set_center_of_pressure(self, value):
        """Store the center of pressure without counting a change: the
        constructor and the geometry-defined surfaces, which count their own
        changes, use this."""
        try:
            x, y, z = value
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "center_of_pressure must be a tuple (x, y, z) in meters, got "
                f"{value!r}."
            ) from exc
        numeric = isinstance(z, (int, float, np.number))
        if hasattr(self, "cm") and (not numeric or getattr(self, "_xcp", None)):
            raise ValueError(
                "A center of pressure that varies with Mach is folded into the "
                "moment coefficients when the surface is built: create a new "
                "surface to change it."
            )
        xcp = (
            None
            if numeric
            else AeroCoefficient(
                z,
                control_variables=getattr(self, "control_variables", ()),
                name="center_of_pressure",
                single_var="mach",
            )
        )
        self._center_of_pressure, self._xcp = tuple(value), xcp
        self.cpx, self.cpy, self.cpz = x, y, 0.0 if xcp else z
        self.cp = (self.cpx, self.cpy, self.cpz)

    def is_active(self, t, flight):
        """Return whether the surface produces force at time ``t`` of a flight.

        Parameters
        ----------
        t : float
            Time in seconds.
        flight : Flight
            The flight being simulated.

        Returns
        -------
        bool
            ``False`` outside the motor phase given by ``active_during``.
            Otherwise the last switch an event of this flight made up to ``t``,
            or the surface's own ``active`` setting when there is none.
        """
        if self.active_during != "always":
            powered = t < flight.rocket.motor.burn_out_time
            if powered != (self.active_during == "power_on"):
                return False
        for time, active in reversed(flight._surface_switches.get(self, ())):
            if t >= time:
                return active
        return self.active

    @staticmethod
    def _validate_active_during(active_during):
        """Return ``active_during`` if it is one of the accepted values, so a
        typo is caught when the surface is built instead of leaving it active."""
        if active_during in ("always", "power_on", "power_off"):
            return active_during
        raise ValueError(
            "`active_during` must be one of 'always', 'power_on' or "
            f"'power_off'; got {active_during!r}. To switch a surface on or off "
            "at another moment of the flight, use an event with the "
            "`activate_surface` and `deactivate_surface` commands."
        )

    @property
    def force_application_point(self):
        """Local point (surface frame) at which the resultant force is applied
        when transporting its moment to the rocket's center of dry mass. This is
        the center of pressure ``self.cp``; any residual couple is carried by the
        ``cm``/``cn``/``cl`` coefficients.
        """
        return Vector([self.cpx, self.cpy, self.cpz])

    @property
    def _wind_coefficients(self):
        """The wind-frame views ``(cL, cD, cQ)`` of the body-frame ``cN``, ``cY``
        and ``cA``. Built on first use and kept until one of those three is
        replaced (for example after a change of geometry)."""
        body = (self.cN, self.cY, self.cA)
        cached_body, views = getattr(self, "_wind_coefficients_cache", (None, None))
        if cached_body is None or any(a is not b for a, b in zip(body, cached_body)):
            views = _body_to_wind_coefficients(*body, self.independent_vars)
            self._wind_coefficients_cache = (body, views)
        return views

    @property
    def cL(self):
        """Wind-frame lift coefficient, as a :class:`Function` of the surface's
        independent variables. Derived from the body-frame ``cN``, ``cY`` and
        ``cA`` (see :func:`_body_to_wind_coefficients`)."""
        return self._wind_coefficients[0]

    @property
    def cD(self):
        """Wind-frame drag coefficient (derived from ``cN``/``cY``/``cA``)."""
        return self._wind_coefficients[1]

    @property
    def cQ(self):
        """Wind-frame side-force coefficient (derived from ``cN``/``cY``/``cA``)."""
        return self._wind_coefficients[2]

    def evaluate_coefficients(self):
        """Build the coefficients from the surface's geometry. Nothing to do
        here; the Barrowman surfaces override it."""

    def _evaluate_stability_derivatives(self):
        """Compute the coefficient derivatives used for stability and store them
        as the ``cN_alpha``, ``cm_alpha``, ``cY_beta`` and ``cn_beta``
        attributes, then build the center-of-pressure accessors from them.

        A plain generic surface recovers each derivative from its body-frame
        force and moment coefficients by numerical differentiation about one
        point: ``alpha = beta = 0``, Reynolds number 0, every rotation rate 0
        and every control at 0, as a function of Mach. A coefficient that
        changes with the Reynolds number is therefore linearized at Reynolds 0
        (the edge of its table, when tabulated). The Barrowman surfaces instead
        set these four attributes directly from geometry and only reuse
        :meth:`_set_stability_accessors` (see the :class:`LinearGenericSurface`
        override).

        Returns
        -------
        None
        """
        # A coefficient given against the total angle of attack is split with
        # the sign of the angle, so its slope is taken from zero toward positive
        # angles (stencil [0, 2 step]) rather than across zero. A moment may
        # carry such a force (see _carry_moments_to_center_of_pressure), so the
        # whole surface is read that way.
        step = self._SLOPE_STEP
        one_sided = bool(self._wind_plane_names)

        def slope(coefficient, angle):
            at = {angle: step} if one_sided else None
            return coefficient.slope(angle, "mach", at=at, dx=step)

        for name, angle in (
            ("cN", "alpha"),
            ("cm", "alpha"),
            ("cY", "beta"),
            ("cn", "beta"),
        ):
            slope_name = f"{name}_{angle}"
            setattr(
                self,
                slope_name,
                AeroCoefficient(
                    slope(getattr(self, name), angle),
                    depends_on=("mach",),
                    control_variables=self.control_variables,
                    name=slope_name,
                ),
            )
        self._set_stability_accessors()

    def _set_stability_accessors(self):
        """Build the pitch- and yaw-plane aerodynamic-center accessors
        (``aerodynamic_center``, ``aerodynamic_center_yaw``) from the
        stored coefficient derivatives (``cN_alpha``/``cm_alpha`` and
        ``cY_beta``/``cn_beta``), each evaluated at ``alpha = beta = 0`` with
        zero rates.

        Each accessor is a Mach-only :class:`Function` giving the surface's
        aerodynamic center along the body z-axis (positive toward the nose),
        measured from the point the surface is positioned at. It combines the
        surface's application point, mapped into the body frame, with the offset
        implied by its moment coefficient (``application point + (moment slope
        / force slope) * L_ref``). When a surface produces no force at some
        Mach the aerodynamic center is undefined, so it falls back to the
        application point and drops out of the force-weighted average.

        Returns
        -------
        None
        """

        def _center_z(force_coeff, moment_coeff):
            def center_z(mach):
                # Same mapping the rocket uses to place the force in flight (see
                # Rocket.evaluate_surfaces_cp_to_cdm), read on each call so it
                # follows a change of geometry.
                application_z = (
                    self._rotation_surface_to_body @ self.force_application_point
                )[2]
                slope = force_coeff.get_value_opt(0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0)
                if slope == 0:
                    return application_z
                moment = moment_coeff.get_value_opt(0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0)
                return application_z + moment / slope * self.reference_length

            return Function(
                center_z, "Mach", "Aerodynamic center to surface position (m)"
            )

        self.aerodynamic_center = _center_z(self.cN_alpha, self.cm_alpha)
        self.aerodynamic_center_yaw = _center_z(self.cY_beta, self.cn_beta)

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
        if force_convention not in self._FORCE_CONVENTIONS:
            raise ValueError(
                f"force_convention must be one of {self._FORCE_CONVENTIONS}, "
                f"got {force_convention!r}."
            )
        return force_convention

    def _as_coefficient(self, source, name, single_var=None):
        """Wrap a coefficient input as an :class:`AeroCoefficient` over this
        surface's variables, with the interpolation and extrapolation the user
        asked for under that coefficient's name. ``single_var`` names the
        variable of a one-column table or headerless file that carries no name.
        """
        # A setting is either one value for every coefficient or a dict by name
        extrapolation, interpolation = (
            option.get(name) if isinstance(option, dict) else option
            for option in (self._extrapolation, self._interpolation)
        )
        coefficient = AeroCoefficient(
            source,
            control_variables=self.control_variables,
            name=name,
            extrapolation=extrapolation,
            interpolation=interpolation,
            single_var=single_var,
        )
        return coefficient

    @staticmethod
    def _is_in_wind_plane(coefficient):
        """Whether a coefficient is given against the total angle of attack
        alone, without the roll angle of the wind. A normal force, a lift or a
        pitch moment given that way acts in the plane that holds the rocket's
        axis and the wind. Given with ``phi`` too, the source says itself how
        the coefficient turns with the wind, and it is used as it is."""
        angles = set(coefficient.source_angles)
        return "alpha_total" in angles and "phi" not in angles

    def _split_along_crossflow(self, coefficient, names):
        """The pitch- and yaw-plane parts of a coefficient taken in the plane of
        the wind (see :func:`_total_angle_to_body_coefficients`), over alpha, beta
        and the variables the coefficient uses."""
        used = [
            var
            for var in self.independent_vars
            if var in ("alpha", "beta") or var in coefficient.depends_on
        ]
        return _total_angle_to_body_coefficients(coefficient, used, names)

    def _wind_plane_input_to_body(self, coefficients):
        """Convert the coefficients given in the plane of the wind (see
        :meth:`_is_in_wind_plane`) into body-frame ones, and leave the others
        as they are.

        ``cN`` and ``cm`` are split between the pitch and yaw planes (see
        :func:`_total_angle_to_body_coefficients`). ``cL`` is first turned, with
        ``cD``, into the normal force in the plane of the wind and the axial
        force. The names of the split coefficients are kept in
        ``_wind_plane_names``.
        """
        self._wind_plane_names = frozenset()
        in_plane = {
            name
            for name, coefficient in coefficients.items()
            if name in self._DIRECTIONAL_COEFFICIENTS
            and self._is_in_wind_plane(coefficient)
        }
        if not in_plane:
            return coefficients
        for name in sorted(in_plane):
            if name not in ("cN", "cm", "cL"):
                raise ValueError(self._not_in_wind_plane_message(name))
            self._check_against_total_angle(coefficients[name], name)
            self._check_zero_at_zero_total_angle(coefficients[name], name)
        body = dict(coefficients)
        split = set()

        def refuse_partner(name, partner):
            given = coefficients.get(partner)
            if given is not None and not given.is_zero:
                raise ValueError(
                    f"{partner} cannot be given together with a {name} against "
                    f"the total angle of attack: {name} then acts in the plane "
                    "of the wind and already provides the part in the other "
                    f"plane. Leave {partner} out."
                )
            body.pop(partner, None)

        for name, partner in (("cN", "cY"), ("cm", "cn")):
            if name in in_plane:
                refuse_partner(name, partner)
                names = (name, partner)
                parts = self._split_along_crossflow(coefficients[name], names)
                body.update(zip(names, parts))
                split.update(names)
        if "cL" in in_plane:
            refuse_partner("cL", "cQ")
            lift = body.pop("cL")
            drag = self._as_coefficient(body.pop("cD", 0), "cD")
            used = [
                var
                for var in self.independent_vars
                if var in ("alpha", "beta")
                or var in lift.depends_on
                or var in drag.depends_on
            ]
            names = ("cN", "cY", "cA")
            body.update(
                zip(names, _wind_plane_lift_to_body_coefficients(lift, drag, used))
            )
            split.update(("cN", "cY"))
        self._wind_plane_names = frozenset(split)
        return body

    @staticmethod
    def _not_in_wind_plane_message(name):
        """Why ``name`` cannot be given against the total angle of attack
        alone."""
        force = {"cY": "cN", "cn": "cm", "cQ": "cL"}[name]
        return (
            f"{name} is given against the total angle of attack alone. A "
            "coefficient given that way acts in the plane of the wind, "
            f"where there is no side force or yaw moment: give {force} "
            "instead, which is split between the pitch and yaw planes for "
            f"you. If {name} does depend on the direction of the wind, give "
            'it against "alpha_total" and "phi", or against "alpha" and '
            '"beta".'
        )

    def _check_against_total_angle(self, coefficient, name):
        """Reject a coefficient given against the total angle of attack that
        also reads the signed partial angles ``alpha`` or ``beta``.

        The total angle sets the size of the force and the split along the
        crossflow sets its direction, with the sign of the angle. A coefficient
        that already carries that sign would get it twice.
        """
        signed = sorted(set(coefficient.source_angles) & {"alpha", "beta"})
        if signed:
            raise ValueError(
                f"{name} is given against alpha_total together with "
                f"{' and '.join(signed)}. Against the total angle of attack it "
                "acts in the plane of the wind, and its direction comes from "
                "the split between the pitch and yaw planes, so a signed angle "
                "would put the sign on twice. Give it against alpha_total alone "
                "(with phi if it depends on the roll angle of the wind), or "
                "against alpha and beta."
            )

    def _check_zero_at_zero_total_angle(self, coefficient, name):
        """Reject a coefficient in the plane of the wind that is not zero at
        zero total angle of attack.

        There the crossflow has no direction, so a normal force or a pitch
        moment has nowhere to point: the coefficient must vanish. A table that
        starts above zero degrees fails too, since it holds its first value all
        the way down to zero. Left in, the body-frame force would flip sign
        every time the angle crosses zero, and the stability slope, taken over a
        tiny step from zero, would come out as that jump divided by the step
        (1e5 for a table starting at 2 degrees).
        """
        args = [0.0] * len(self.independent_vars)
        machs = self._ZERO_ANGLE_MACHS if "mach" in coefficient.depends_on else (0.0,)
        for mach in machs:
            args[2] = mach
            value = coefficient(*args)
            if abs(value) > 1e-9:
                raise ValueError(
                    f"{name} is {value:.4g} at zero total angle of attack (Mach "
                    f"{mach:g}) but must be zero there: with no crossflow the "
                    "force has no direction to point in. Give 0 at "
                    "alpha_total = 0; a table must start at 0 degrees, since its "
                    "first value is held down to zero otherwise."
                )

    def _wind_input_to_body(self, coefficients):
        """Convert a wind-frame force-coefficient input (``cL``/``cQ``/``cD``)
        into the canonical body-frame coefficients (``cN``/``cY``/``cA``),
        leaving the moment coefficients untouched."""
        wind = {
            name: self._as_coefficient(coefficients.get(name, 0), name)
            for name in self._WIND_FORCE_NAMES
        }
        passthrough = {
            name: value
            for name, value in coefficients.items()
            if name not in self._WIND_FORCE_NAMES
        }
        if all(coefficient.is_zero for coefficient in wind.values()):
            return passthrough

        # The conversion reads alpha and beta on top of what the inputs use.
        # Keeping only those variables stops a constant drag from looking like
        # it depends on the Reynolds number or the rotation rates.
        used = [
            var
            for var in self.independent_vars
            if var in ("alpha", "beta")
            or any(var in coefficient.depends_on for coefficient in wind.values())
        ]
        body = _wind_to_body_coefficients(wind["cL"], wind["cD"], wind["cQ"], used)
        return {**passthrough, **dict(zip(self._BODY_FORCE_NAMES, body))}

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
        self._interpolation = interpolation
        self._extrapolation = extrapolation
        if not isinstance(coefficients, dict):
            raise TypeError(
                "coefficients must be a dict from coefficient name to value, for "
                f'example {{"cN": 2.0, "cA": "cA.csv"}}; got '
                f"{type(coefficients).__name__}. For one file holding several "
                f"coefficients use {type(self).__name__}.from_csv."
            )
        default_coefficients = self._get_default_coefficients()
        self.force_convention = self._resolve_force_convention(
            coefficients, force_convention
        )
        # Kept as given, so saving the surface needs no pickling
        self._input_coefficients = {
            # A rate derivative given as an unnamed curve is against Mach
            name: self._as_coefficient(
                value, name, "mach" if self._rate_of(name) else None
            )
            for name, value in coefficients.items()
        }
        # Input in the plane of the wind, then in the wind frame, becomes
        # body-frame coefficients
        coefficients = self._wind_plane_input_to_body(self._input_coefficients)
        if self.force_convention == "wind":
            coefficients = self._wind_input_to_body(coefficients)
        coefficients = self._complete_body_coefficients(coefficients)
        self._check_coefficients(coefficients, default_coefficients)
        coefficients = {**default_coefficients, **coefficients}

        # Lets the flight skip the Reynolds number when no coefficient uses it
        self._needs_reynolds = False
        for coeff, coeff_value in coefficients.items():
            value = self._as_coefficient(coeff_value, coeff)
            setattr(self, coeff, value)
            if "reynolds" in value.depends_on:
                self._needs_reynolds = True
        if self._xcp is not None:
            self._carry_moments_to_center_of_pressure()

    def _rate_of(self, name):
        """The reduced rate a rate derivative such as ``cm_q`` multiplies, or
        ``None`` when ``name`` is not a rate derivative."""
        coefficient, _, suffix = name.partition("_")
        if coefficient in GenericSurface._get_default_coefficients():
            return self._RATE_DERIVATIVES.get(suffix)
        return None

    def _complete_body_coefficients(self, coefficients):
        """Last step before the body-frame coefficients are stored: add each
        rate derivative to its coefficient, ``cm + cm_q * pitch_rate``. (The
        linear surface keeps its derivatives and fills in its yaw plane.)"""
        coefficients = dict(coefficients)
        for name in [name for name in coefficients if self._rate_of(name)]:
            target, rate = name.partition("_")[0], self._rate_of(name)
            derivative = coefficients.pop(name)
            base = self._as_coefficient(coefficients.get(target, 0), target)
            if rate in base.depends_on:
                raise ValueError(
                    f"{target} already depends on {rate}, so {name} cannot be "
                    "given too. Give the rate dependence in one place."
                )
            used = [
                var
                for var in self.independent_vars
                if var == rate or var in base.depends_on or var in derivative.depends_on
            ]
            coefficients[target] = self._with_rate_term(base, derivative, rate, used)
        return coefficients

    @staticmethod
    def _with_rate_term(base, derivative, rate, used):
        """``base + derivative * rate`` as a Function of the variables
        ``used``."""
        read_base, read_derivative = base.evaluator(used), derivative.evaluator(used)
        at = used.index(rate)
        return _as_function(
            lambda *args: read_base(*args) + read_derivative(*args) * args[at],
            used,
            base.name,
        )

    def _carry_moments_to_center_of_pressure(self):
        """With a center of pressure that varies with Mach the force is applied
        at the surface's ``z = 0``, so its moment about the center of pressure
        is added to the moment coefficients: ``cm + cN * z(mach) / L_ref`` and
        ``cn + cY * z(mach) / L_ref`` (each derivative alike for the linear
        model, ``cm_alpha + cN_alpha * z / L_ref`` and so on)."""
        length = self.reference_length

        def carried(moment, force):
            used = [
                var
                for var in self.independent_vars
                if var == "mach" or var in moment.depends_on or var in force.depends_on
            ]
            read_moment, read_force, read_z = (
                c.evaluator(used) for c in (moment, force, self._xcp)
            )

            def total(*args):
                return read_moment(*args) + read_force(*args) * read_z(*args) / length

            return self._as_coefficient(
                _as_function(total, used, moment.name), moment.name
            )

        for name in self._get_default_coefficients():
            force_name = {"cm": "cN", "cn": "cY"}.get(name[:2])
            if force_name is not None:
                force = getattr(self, force_name + name[2:])
                if not force.is_zero:
                    setattr(self, name, carried(getattr(self, name), force))

    @classmethod
    def _get_default_coefficients(cls):
        """The coefficients the surface holds, each with its default value."""
        return {"cN": 0, "cY": 0, "cA": 0, "cm": 0, "cn": 0, "cl": 0}

    def _check_coefficients(self, input_coefficients, default_coefficients):
        """Raise a ``ValueError`` for a coefficient name the surface does not
        have."""
        invalid_keys = set(input_coefficients) - set(default_coefficients)
        if invalid_keys:
            raise ValueError(
                f"Invalid coefficient name(s) used in key(s): {', '.join(invalid_keys)}. "
                "Check the documentation for valid names."
            )

    def _coefficient_arguments(self, *state):
        """The arguments every coefficient is called with: the seven flow
        variables in ``state``, plus any a subclass adds."""
        return state

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
            The aerodynamic force components ``(R1, R2, R3)`` along the body
            x, y and z axes and the moments ``(M1, M2, M3)`` about them, taken
            about the rocket's center of dry mass.
        """
        # Reynolds number at the surface altitude. Computed here (rather than in
        # the flight loop) since it is only needed by generic surfaces
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

        args = self._coefficient_arguments(
            alpha,
            beta,
            stream_mach,
            reynolds,
            omega[0] * reduced_rate_factor,
            omega[1] * reduced_rate_factor,
            omega[2] * reduced_rate_factor,
        )
        force = 0.5 * rho * stream_speed**2 * self.reference_area
        moment = force * self.reference_length
        R1 = force * self.cY.get_value_opt(*args)
        R2 = -force * self.cN.get_value_opt(*args)
        R3 = -force * self.cA.get_value_opt(*args)
        pitch = moment * self.cm.get_value_opt(*args)
        yaw = moment * self.cn.get_value_opt(*args)
        roll = moment * self.cl.get_value_opt(*args)

        # Dislocation of the aerodynamic application point to CDM
        M1, M2, M3 = Vector([pitch, yaw, roll]) + (cp ^ Vector([R1, R2, R3]))

        return R1, R2, R3, M1, M2, M3

    @classmethod
    def _input_coefficient_names(cls):
        """Every name a coefficient can be given under: the body-frame names and
        the wind-frame ones."""
        return set(cls._get_default_coefficients()) | set(cls._WIND_FORCE_NAMES)

    @classmethod
    def _input_variable_names(cls, **kwargs):  # pylint: disable=unused-argument
        """Every name a variable can be given under, for a surface built with
        the constructor arguments ``kwargs``."""
        return [*build_independent_vars(), *SOURCE_ONLY_NAMES]

    @classmethod
    def from_csv(
        cls, file_path, reference_area, reference_length, columns=None, **kwargs
    ):
        """Create the surface from one table file holding several coefficients.

        The file has one column per variable the coefficients are tabulated
        against and one column per coefficient, in any order, for example::

            alpha_deg, mach, cN,    cA,   cm
            -2,        0.3, -0.085, 0.42,  0.260
            0,         0.3,  0.000, 0.42,  0.000
            2,         0.3,  0.085, 0.42, -0.260

        Every coefficient is read over all the variable columns. When the rows
        cover every combination of the variables' values, the table is
        interpolated as a regular grid.

        Parameters
        ----------
        file_path : str
            Path to the ``.csv`` file. Its first line names the columns.
        reference_area : int, float
            Reference area of the surface, in squared meters.
        reference_length : int, float
            Reference length of the surface, in meters.
        columns : dict, optional
            Translation from the file's column names to RocketPy's, for files
            written by other programs, for example ``{"Mach": "mach", "Alpha":
            "alpha_deg", "CN": "cN", "CA Power-Off": "cA"}``. When given, only
            the columns it lists are read and the others are ignored. When left
            as ``None`` (the default) every column of the file must carry a
            RocketPy name.

            The variable names are ``alpha`` and ``beta`` (radians) or
            ``alpha_deg`` and ``beta_deg`` (degrees), ``alpha_total`` and
            ``phi`` (total angle of attack and roll angle of the wind, also with
            ``_deg``), ``mach``, ``reynolds``,
            ``pitch_rate``, ``yaw_rate`` and ``roll_rate``, plus the control
            names of a controllable surface. The coefficient names are those of
            the class, such as ``cN``, ``cY``, ``cA`` (or ``cL``, ``cQ``,
            ``cD``), ``cm``, ``cn`` and ``cl`` for a :class:`GenericSurface`.
            Names are case sensitive: ``cN`` is the normal force and ``cn`` the
            yaw moment.
        **kwargs
            Any other argument of the class, such as ``center_of_pressure``,
            ``name``, ``interpolation`` or ``active_during``.

        Returns
        -------
        GenericSurface
            The surface, of the class this method was called on.

        Notes
        -----
        RocketPy's ``alpha`` is measured in one plane and takes both signs. For
        a table against the *total* angle of attack, which only has positive
        angles, name the column ``alpha_total`` (or ``alpha_total_deg``). A
        warning is raised when a table read
        as ``alpha`` looks like it is against the total angle.
        """
        with open(file_path, mode="r", encoding="utf-8") as file:
            header = [name.strip() for name in next(csv.reader(file))]
        data = np.atleast_2d(np.loadtxt(file_path, delimiter=",", skiprows=1))
        if columns is None:
            names = header
        else:
            missing = [name for name in columns if name not in header]
            if missing:
                raise ValueError(
                    f"Column(s) {missing} not found in {file_path}. The file "
                    f"has the columns {header}."
                )
            names = [columns.get(name) for name in header]

        variable_names = cls._input_variable_names(**kwargs)
        coefficient_names = cls._input_coefficient_names()
        unknown = [
            name
            for name in names
            if name is not None
            and name not in variable_names
            and name not in coefficient_names
        ]
        if unknown:
            raise ValueError(
                f"Column name(s) {unknown} of {file_path} are neither a variable "
                f"({', '.join(variable_names)}) nor a coefficient of "
                f"{cls.__name__}. Use the `columns` argument to translate the "
                "file's column names, which also lets the other columns be "
                "ignored."
            )
        repeated = {name for name in names if name and names.count(name) > 1}
        if repeated:
            raise ValueError(f"Column name(s) {sorted(repeated)} appear twice.")

        variables = [i for i, name in enumerate(names) if name in variable_names]
        values = [i for i, name in enumerate(names) if name in coefficient_names]
        if not variables or not values:
            raise ValueError(
                f"{file_path} needs at least one variable column and one "
                "coefficient column."
            )
        coefficients = {
            names[i]: (data[:, [*variables, i]], [names[j] for j in variables])
            for i in values
        }
        return cls(reference_area, reference_length, coefficients, **kwargs)

    @classmethod
    def _arguments_from_dict(cls, data):
        """The constructor arguments stored by :meth:`to_dict`. Subclasses extend
        it with their own arguments."""
        arguments = {
            "reference_area": data["reference_area"],
            "reference_length": data["reference_length"],
            "coefficients": data["coefficients"],
            "center_of_pressure": data.get("center_of_pressure", (0, 0, 0)),
            "name": data.get("name", "Generic Surface"),
            "reynolds_length": data.get("reynolds_length"),
            "force_convention": data.get("force_convention", "body"),
            "active_during": data.get("active_during", "always"),
            "active": data.get("active", True),
        }
        return arguments

    def to_dict(self, include_outputs=False, **kwargs):  # pylint: disable=unused-argument
        """Return the surface as a dictionary, to save it and rebuild it later.

        The coefficients are saved as they were given, so a table stays a table
        and loading the surface converts them again exactly as the constructor
        did.

        Parameters
        ----------
        include_outputs : bool, optional
            Not used: a surface has no results to save. It is accepted so that
            every RocketPy object is saved the same way. Default False.
        **kwargs
            Not used. Accepted so that every RocketPy object is saved the same
            way.

        Returns
        -------
        dict
            The arguments needed to rebuild the surface with :meth:`from_dict`.
        """
        x, y, z = self.center_of_pressure
        return {
            "reference_area": self.reference_area,
            "reference_length": self.reference_length,
            "reynolds_length": self.reynolds_length,
            "coefficients": self._input_coefficients,
            # The axial position as given: a number, or a function of Mach
            "center_of_pressure": (x, y, self._xcp or z),
            "name": self.name,
            "force_convention": self.force_convention,
            "active_during": self.active_during,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a surface saved with :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            The dictionary returned by :meth:`to_dict`.

        Returns
        -------
        GenericSurface
            The surface, of the class this method is called on.
        """
        return cls(**cls._arguments_from_dict(data))

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
