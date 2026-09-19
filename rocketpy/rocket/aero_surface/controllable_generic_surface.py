from rocketpy.rocket.aero_surface.generic_surface import GenericSurface
from rocketpy.tools import from_hex_decode, to_hex_encode


class ControllableGenericSurface(GenericSurface):
    """A generic aerodynamic surface whose coefficients also depend on one or
    more control inputs (canards, grid fins, elevons, air-brake deployment, and
    so on) set by a controller while the rocket flies.

    On top of the seven standard variables of :class:`GenericSurface`
    (``alpha``, ``beta``, ``mach``, ``reynolds``, ``pitch_rate``, ``yaw_rate``,
    ``roll_rate``), each coefficient takes one extra input per control, in the
    order listed in ``controls``. A controller updates the current control
    values every simulation step (see ``Rocket.add_controllable_surface``), and
    they are passed to the coefficients automatically.

    Attributes
    ----------
    ControllableGenericSurface.control_variables : list of str
        Names of the controls, in the order the coefficients expect them.
    ControllableGenericSurface.control_state : dict
        Current value of each control (starts at 0).
    """

    # TODO: deflection-dependent static-margin diagnostics.
    #
    # The in-flight dynamics are correct: the deflection feeds the coefficient
    # functions live every step (see ``_coefficient_arguments``), and the surface
    # never physically moves, so its force-application point / ``cp_to_cdm`` cache
    # cannot go stale (unlike an individual fin's cant angle, which IS a physical
    # reconfiguration and is refreshed via ``Rocket.refresh_controlled_components``).
    #
    # The gap is diagnostic-only. The derived ``center_of_pressure_z`` /
    # ``aerodynamic_center`` come from ``cm_alpha = d(cm)/d(alpha)`` evaluated ONCE
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
    #     fin mechanism: bump ``_geometry_version`` in ``set_control`` and have
    #     ``Rocket.refresh_controlled_components`` re-run the derived-cp step;
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
    ):
        """Create a controllable generic aerodynamic surface.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the surface, in squared meters.
        reference_length : int, float
            Reference length of the surface, in meters.
        coefficients : dict
            The six force and moment coefficients (``cL``, ``cQ``, ``cD``,
            ``cm``, ``cn``, ``cl``), by name. Each one can be a constant, a
            function, or a path to a data file, and depends on the seven base
            variables **plus** the controls listed in ``controls`` (in that
            order). Any you leave out are set to 0.
        center_of_pressure : tuple, list, optional
            Application point of the aerodynamic forces and moments in the local
            surface frame. Default ``(0, 0, 0)``.
        name : str, optional
            Name of the surface. Default ``"Controllable Generic Surface"``.
        controls : iterable of str, optional
            Names of the controls, such as a canard deflection angle. Default
            ``("deflection",)``. Each name becomes an extra input to every
            coefficient and a key in :attr:`control_state`.
        reynolds_length : int, float, optional
            Length scale, in meters, of the Reynolds number passed to the
            coefficients. See :class:`GenericSurface`. ``None`` (the default)
            uses ``reference_length`` (the diameter).
        extrapolation : str or dict, optional
            What tabulated coefficients do outside their data range:
            ``"constant"`` holds the nearest edge value, ``"natural"`` keeps
            following the curve, ``"zero"`` returns 0. Give one string for all
            coefficients or a dict keyed by coefficient name. ``None`` (the
            default) uses ``"constant"`` for tables built here and leaves a
            pre-built :class:`Function` unchanged.
        interpolation : str or dict, optional
            How tabulated coefficients read values between points (for example
            ``"linear"``, ``"akima"`` or ``"spline"`` for a 1-D table; see
            :class:`rocketpy.GenericSurface` for the full list by table type).
            Give one string for all coefficients or a dict keyed by coefficient
            name. ``None`` (the default) uses ``"linear"`` for tables built here
            and leaves a pre-built :class:`Function` unchanged.
        active_during : str or callable, optional
            When this surface produces force during a simulation: ``"always"``
            (default), ``"power_on"`` (only while the motor burns, e.g. jet
            vanes), ``"power_off"`` (only after burnout), or a function
            ``active_during(t, flight)``. See :class:`GenericSurface` for details.
        """
        # These must be set before ``super().__init__`` so coefficient
        # processing (arity, CSV validation) and the derived-cp accessors see
        # the extended variable list (via the ``independent_vars`` property,
        # which appends ``control_variables``) and the current control values.
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
            active_during=active_during,
        )
        # ``self.prints``/``self.plots`` are the generic ones wired by the base.

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
        """Append the current control-variable values (in
        ``self.control_variables`` order) to the standard inputs."""
        base = super()._coefficient_arguments(
            alpha,
            beta,
            mach,
            reynolds,
            pitch_rate,
            yaw_rate,
            roll_rate,
        )
        controls = tuple(self.control_state[name] for name in self.control_variables)
        return base + controls

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

    def to_dict(  # pylint: disable=unused-argument
        self, include_outputs=False, **kwargs
    ):
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
            "coefficients": {
                "cN": self.cN,
                "cY": self.cY,
                "cA": self.cA,
                "cm": self.cm,
                "cn": self.cn,
                "cl": self.cl,
            },
            "center_of_pressure": self.center_of_pressure,
            "name": self.name,
            "controls": self.control_variables,
            "reynolds_length": self.reynolds_length,
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
            name=data.get("name", "Controllable Generic Surface"),
            controls=data.get("controls", ("deflection",)),
            reynolds_length=data.get("reynolds_length"),
            active_during=active_during,
        )
