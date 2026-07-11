from rocketpy.rocket.aero_surface.generic_surface import GenericSurface


class ControllableGenericSurface(GenericSurface):
    """A generic aerodynamic surface whose coefficients additionally depend on
    one or more **control-deflection** variables (canards, grid fins, elevons,
    air-brake deployment, …) sourced at runtime from a controller.

    On top of the seven standard independent variables of
    :class:`GenericSurface` (``alpha``, ``beta``, ``mach``, ``reynolds``,
    ``pitch_rate``, ``yaw_rate``, ``roll_rate``), the coefficient functions take
    one extra argument per entry of ``controls`` (appended in order). The
    current control values are held in :attr:`control_state` and mutated each
    simulation step by a controller (see ``Rocket.add_controllable_surface``);
    :meth:`_coefficient_arguments` appends them to every coefficient evaluation.

    Attributes
    ----------
    ControllableGenericSurface.control_variables : list of str
        Names of the control-deflection axes, in coefficient-argument order.
    ControllableGenericSurface.control_state : dict
        Current value of each control variable (defaults to 0).
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
    # (in ``_set_derived_cp_accessors``) with the control variables frozen at their
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
    ):
        """Create a controllable generic aerodynamic surface.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the surface, in squared meters.
        reference_length : int, float
            Reference length of the surface, in meters.
        coefficients : dict
            Aerodynamic coefficients (``cL``, ``cQ``, ``cD``, ``cm``, ``cn``,
            ``cl``), each a callable/CSV/Function of the seven base variables
            **plus** the control variables listed in ``controls`` (appended in
            order). Omitted coefficients default to 0.
        center_of_pressure : tuple, list, optional
            Application point of the aerodynamic forces and moments in the local
            surface frame. Default ``(0, 0, 0)``.
        name : str, optional
            Name of the surface. Default ``"Controllable Generic Surface"``.
        controls : iterable of str, optional
            Names of the control-deflection axes. Default ``("deflection",)``.
            Each name becomes an extra coefficient argument and a key in
            :attr:`control_state`.
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
        )
        # ``self.prints``/``self.plots`` are the generic ones wired by the base.
        self.initial_control_state = dict(self.control_state)

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
        """Append the current control-variable values (in
        ``self.control_variables`` order) to the standard inputs (which may
        already include the unsteady ``alpha_dot``/``beta_dot`` axes)."""
        base = super()._coefficient_arguments(
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

    def _reset(self):
        """Restore all control variables to their initial values. Ran by the
        controller at the beginning of each simulation so control state does
        not leak across flights."""
        for name, value in self.initial_control_state.items():
            self.set_control(name, value)

    def to_dict(  # pylint: disable=unused-argument
        self, include_outputs=False, **kwargs
    ):
        return {
            "reference_area": self.reference_area,
            "reference_length": self.reference_length,
            "coefficients": {
                "cL": self.cL,
                "cQ": self.cQ,
                "cD": self.cD,
                "cm": self.cm,
                "cn": self.cn,
                "cl": self.cl,
            },
            "center_of_pressure": self.center_of_pressure,
            "name": self.name,
            "controls": self.control_variables,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            reference_area=data["reference_area"],
            reference_length=data["reference_length"],
            coefficients=data["coefficients"],
            center_of_pressure=data.get("center_of_pressure", (0, 0, 0)),
            name=data.get("name", "Controllable Generic Surface"),
            controls=data.get("controls", ("deflection",)),
        )
