from rocketpy.rocket.aero_surface.generic_surface import GenericSurface


class ControllableGenericSurface(GenericSurface):
    """An aerodynamic surface whose coefficients also depend on one or
    more control inputs (canards, grid fins, elevons, air-brake deployment, and
    so on) set by a controller while the rocket flies.

    On top of the seven standard variables of :class:`GenericSurface`
    (``alpha``, ``beta``, ``mach``, ``reynolds``, ``pitch_rate``, ``yaw_rate``,
    ``roll_rate``), each coefficient takes one extra input per control, in the
    order listed in ``controls``. The current control values are set with
    :meth:`set_control` and passed to the coefficients automatically. During a
    flight a controller does that: build a ``rocketpy.control._Controller``
    with this surface as its ``controlled_objects`` and a function that calls
    ``set_control`` from the flight state, and bind it with
    ``rocket._add_controllers(controller)`` (the path
    :meth:`rocketpy.Rocket.add_air_brakes` uses); the flight then runs the
    controller at its sampling rate.

    Attributes
    ----------
    ControllableGenericSurface.control_variables : list of str
        Names of the controls, in the order the coefficients expect them.
    ControllableGenericSurface.control_state : dict
        Current value of each control (starts at 0).
    ControllableGenericSurface.cN : AeroCoefficient
        Normal force coefficient, the force in the pitch plane.
    ControllableGenericSurface.cY : AeroCoefficient
        Side force coefficient, the force in the yaw plane.
    ControllableGenericSurface.cA : AeroCoefficient
        Axial force coefficient, the force along the rocket's axis.
    ControllableGenericSurface.cm : AeroCoefficient
        Pitching moment coefficient, about ``center_of_pressure``.
    ControllableGenericSurface.cn : AeroCoefficient
        Yawing moment coefficient, about ``center_of_pressure``.
    ControllableGenericSurface.cl : AeroCoefficient
        Roll moment coefficient.
    ControllableGenericSurface.cL : Function
        Lift coefficient, the force perpendicular to the airflow.
    ControllableGenericSurface.cD : Function
        Drag coefficient, the force along the airflow.
    ControllableGenericSurface.cQ : Function
        Crosswind coefficient, the side force relative to the airflow.
    ControllableGenericSurface.cN_alpha : AeroCoefficient
        Slope of ``cN`` with the angle of attack, as a function of Mach.
        Has units of 1/rad.
    ControllableGenericSurface.cY_beta : AeroCoefficient
        Slope of ``cY`` with the sideslip angle, as a function of Mach.
        Has units of 1/rad.
    ControllableGenericSurface.cm_alpha : AeroCoefficient
        Slope of ``cm`` with the angle of attack, as a function of Mach.
        Has units of 1/rad.
    ControllableGenericSurface.cn_beta : AeroCoefficient
        Slope of ``cn`` with the sideslip angle, as a function of Mach.
        Has units of 1/rad.
    ControllableGenericSurface.aerodynamic_center : Function
        Pitch-plane aerodynamic center of the surface along the rocket's axis,
        in meters from the surface's position (positive toward the nose), as a
        function of Mach. It is the point where the change of the normal force
        acts when the angle of attack changes a little from zero. It is the
        same point as ``center_of_pressure`` when no moment coefficient is
        given.
    ControllableGenericSurface.aerodynamic_center_yaw : Function
        Yaw-plane aerodynamic center of the surface along the rocket's axis, in
        meters from the surface's position (positive toward the nose), as a
        function of Mach: the same for the side force and the sideslip angle.
    ControllableGenericSurface.reference_area : float
        Reference area, in square meters.
    ControllableGenericSurface.reference_length : float
        Reference length, in meters.
    ControllableGenericSurface.reynolds_length : float
        Length scale of the Reynolds number, in meters.
    ControllableGenericSurface.name : str
        Name of the surface.
    ControllableGenericSurface.center_of_pressure : tuple
        Point where the forces are applied, ``(x, y, z)`` in meters, as given.
    ControllableGenericSurface.cp : tuple
        The same point as ``(cpx, cpy, cpz)``. ``cpz`` is 0 when the center of
        pressure varies with Mach, since the force is then applied at ``z = 0``.
    ControllableGenericSurface.cpx : float
        x coordinate of ``cp``, in meters.
    ControllableGenericSurface.cpy : float
        y coordinate of ``cp``, in meters.
    ControllableGenericSurface.cpz : float
        z coordinate of ``cp``, in meters.
    ControllableGenericSurface.active_during : str
        The motor phase the surface produces force in, as given.
    ControllableGenericSurface.active : bool
        Whether the surface starts each flight switched on.
    ControllableGenericSurface.force_convention : str
        Frame the force coefficients were given in: ``"body"`` or ``"wind"``.
    ControllableGenericSurface.independent_vars : list of str
        The variables every coefficient is called with, in order.
    ControllableGenericSurface.prints : _GenericSurfacePrints
        The prints of the surface. Use help(ControllableGenericSurface.prints) to
        know more.
    ControllableGenericSurface.plots : _GenericSurfacePlots
        The plots of the surface. Use help(ControllableGenericSurface.plots) to
        know more.
    """

    # TODO: deflection-dependent static-margin diagnostics.
    #
    # The in-flight dynamics are correct: the deflection feeds the coefficient
    # functions live every step (see ``_coefficient_arguments``), and the surface
    # never physically moves, so its force-application point / ``cp_to_cdm`` cache
    # cannot go stale (unlike an individual fin's cant angle, which IS a physical
    # reconfiguration and counts up the fin's ``_version`` so the rocket
    # refreshes it).
    #
    # The gap is diagnostic-only. The surface's ``aerodynamic_center`` and the
    # rocket's come from ``cm_alpha = d(cm)/d(alpha)`` evaluated ONCE
    # (in ``_set_stability_accessors``) with the control variables frozen at their
    # value at construction (0). So if ``cm`` couples alpha and a control axis
    # (e.g. an ``alpha * deflection`` term), the reported ``static_margin`` is
    # pinned to the zero-deflection configuration and does not track ``set_control``.
    # It also is not a single well-defined number: the static margin of a deflected
    # control surface is inherently a function of the control input.
    #
    # To address this properly (not a correctness fix, defer until there is a real
    # need), likely some combination of:
    #   - an ``initial_deflection`` (per-control) argument in ``__init__`` so the
    #     derived cp accessors are built about a chosen reference deflection rather
    #     than always 0;
    #   - re-deriving the cp accessors when the deflection changes -- reuse the
    #     fin mechanism: bump ``_version`` in ``set_control`` so the rocket's
    #     stamp-based refresh re-runs the derived-cp step;
    #   - dedicated stability plots/prints that sweep the static margin (and cp)
    #     OVER the control-deflection range, since a single scalar margin is the
    #     wrong abstraction for a controllable surface.

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Controllable Generic Surface",
        controls=("deflection",),
        reynolds_length=None,
        extrapolation=None,
        interpolation=None,
        active_during="always",
        force_convention=None,
        active=True,
    ):
        """Create an aerodynamic surface whose coefficients also depend on
        controls, such as a canard deflection.

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
            The force and moment coefficients, by name. Any you leave out are 0:
            the body-frame forces ``cN`` (normal), ``cY`` (side) and ``cA``
            (axial), or the wind-frame ones ``cL`` (lift), ``cQ`` (side) and
            ``cD`` (drag), and the moments ``cm`` (pitch), ``cn`` (yaw) and
            ``cl`` (roll), taken about ``center_of_pressure``.

            Each coefficient can depend on ``alpha`` and ``beta`` (radians, or
            ``alpha_deg`` and ``beta_deg`` in degrees), ``alpha_total`` and
            ``phi`` (or ``alpha_total_deg`` and ``phi_deg``; a ``cN``, ``cL`` or
            ``cm`` against ``alpha_total`` alone is split between the pitch and
            yaw planes for you, see :class:`GenericSurface`), ``mach``, ``reynolds``, the reduced rates
            ``pitch_rate``, ``yaw_rate`` and ``roll_rate`` (such as
            ``q * L_ref / (2 * V)``), and each name in ``controls``, for example
            ``lambda alpha, mach, deflection: ...``. It can be given as a
            number, a function whose arguments are named after the variables it
            uses, the path to a CSV file whose header names the variables
            (controls included), a list or numpy array of data points as a pair
            with the names of its variables, a :class:`Function`, or a pair
            ``(axes, values)`` for values on a regular grid.
        center_of_pressure : tuple, list, optional
            Point where the aerodynamic forces are applied and about which the
            moment coefficients are taken, as ``(x, y, z)`` in meters. It is
            measured from the position the surface is added to the rocket at:
            ``z`` runs along the rocket's centerline and is positive toward the
            nose, whichever coordinate system orientation the rocket uses. The
            ``z`` component may instead vary with Mach, as a function of Mach, a
            one-input :class:`Function` or a two-column table
            ``[[mach, z], ...]``; the moments are then carried to that point by
            the moment coefficients. Default ``(0, 0, 0)``.
        name : str, optional
            Name of the surface. Default ``"Controllable Generic Surface"``.
        controls : iterable of str, optional
            Names of the controls, such as a canard deflection angle. Each name
            becomes an extra input to every coefficient, after the seven flow
            variables and in this order, and a key in :attr:`control_state`. The
            values start at 0 and are set with :meth:`set_control`. Default
            ``("deflection",)``.
        reynolds_length : int, float, optional
            Length scale, in meters, of the Reynolds number passed to the
            coefficients. Set it to the length your Reynolds-dependent data was
            tabulated against (for example the rocket's body length). ``None``
            (the default) uses ``reference_length`` (the diameter).
        extrapolation : str or dict, optional
            What tabulated coefficients do outside their data range:
            ``"constant"`` holds the nearest edge value, ``"natural"`` keeps
            following the curve, ``"zero"`` returns 0. Give one string for all
            coefficients or a dict keyed by coefficient name. ``None`` (the
            default) uses ``"constant"`` for tables built here and leaves a
            pre-built :class:`Function` unchanged.
        interpolation : str or dict, optional
            How tabulated coefficients read values between points: a 1-D table
            accepts ``"linear"``, ``"akima"``, ``"spline"`` and ``"polynomial"``;
            a multi-dimensional scattered table accepts ``"linear"``,
            ``"shepard"`` and ``"rbf"``; and a table on a regular grid accepts
            ``"linear"``, ``"nearest"``, ``"slinear"``, ``"cubic"``,
            ``"quintic"`` and ``"pchip"``. Give one string for all coefficients
            or a dict keyed by coefficient name. ``None`` (the default) uses
            ``"linear"`` for tables built here and leaves a pre-built
            :class:`Function` unchanged.
        active_during : str, optional
            The motor phase this surface produces force in: ``"always"``
            (default), ``"power_on"`` (only while the motor burns, e.g. jet
            vanes) or ``"power_off"`` (only after burnout). To switch a surface
            on or off at any other moment, use an event: see ``active`` below.
        force_convention : str, optional
            The frame the force coefficients are given in: ``"body"`` for
            ``cN``/``cY``/``cA`` or ``"wind"`` for ``cL``/``cQ``/``cD``. ``None``
            (the default) works it out from the names.
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
            ``active_during`` is not one of the accepted values, or if
            wind-frame and body-frame force coefficients are mixed without a
            ``force_convention``.

        See Also
        --------
        :ref:`genericsurfaces`
        """
        # Set before the base constructor, which reads them to build the
        # coefficients
        self.control_variables = list(controls)
        self.control_state = {name: 0.0 for name in self.control_variables}

        super().__init__(
            reference_area=reference_area,
            reference_length=reference_length,
            coefficients=coefficients,
            center_of_pressure=center_of_pressure,
            name=name,
            reynolds_length=reynolds_length,
            extrapolation=extrapolation,
            interpolation=interpolation,
            force_convention=force_convention,
            active_during=active_during,
            active=active,
        )
        # ``self.prints``/``self.plots`` are the generic ones wired by the base.

    @classmethod
    def _input_variable_names(cls, **kwargs):
        controls = kwargs.get("controls", ("deflection",))
        return [*super()._input_variable_names(), *controls]

    def _coefficient_arguments(self, *state):
        """Add the current value of each control, in ``control_variables`` order."""
        return (*state, *(self.control_state[name] for name in self.control_variables))

    def _clamp_control(self, name, value):  # pylint: disable=unused-argument
        """Hook to constrain a control value before it is stored. The base class
        applies no clamping; subclasses (e.g. ``AirBrakes``) may override."""
        return value

    def set_control(self, name, value):
        """Set the current value of a control variable (applying any clamping).

        Parameters
        ----------
        name : str
            Name of the control variable; must be one of
            :attr:`control_variables`.
        value : float
            New control value.
        """
        if name not in self.control_state:
            raise KeyError(
                f"Unknown control variable '{name}'. "
                f"Valid controls are: {self.control_variables}."
            )
        self.control_state[name] = self._clamp_control(name, value)

    def get_control(self, name):
        """Return the current value of a control variable."""
        return self.control_state[name]

    def to_dict(self, include_outputs=False, **kwargs):
        data = super().to_dict(include_outputs=include_outputs, **kwargs)
        data["controls"] = list(self.control_variables)
        return data

    @classmethod
    def _arguments_from_dict(cls, data):
        arguments = super()._arguments_from_dict(data)
        arguments["controls"] = data.get("controls", ("deflection",))
        return arguments
