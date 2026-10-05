from rocketpy.plots.aero_surface_plots import _LinearGenericSurfacePlots
from rocketpy.prints.aero_surface_prints import _LinearGenericSurfacePrints
from rocketpy.rocket.aero_surface._helpers import _as_function
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface


class LinearGenericSurface(GenericSurface):
    """An aerodynamic surface whose forces and moments vary linearly.

    They vary linearly with the flow angles and the rotation rates. Instead of
    full coefficient tables, you give the coefficient *derivatives* (slopes),
    for example how much the normal force changes per radian of angle of attack,
    and the surface adds them up.

    Attributes
    ----------
    LinearGenericSurface.cN_0 : AeroCoefficient
        The 36 coefficient derivatives, ``cN_0``, ``cN_alpha``, ..., ``cl_r``,
        named ``<coefficient>_<variable>`` as given in ``coefficients``.
    LinearGenericSurface.cNf : Function
        Forcing part of ``cN``, and likewise ``cYf``, ``cAf``, ``cmf``, ``cnf``
        and ``clf``: for example ``cmf = cm_0 + cm_alpha * alpha + cm_beta * beta``.
    LinearGenericSurface.cNd : Function
        Damping part of ``cN``, and likewise ``cYd``, ``cAd``, ``cmd``, ``cnd``
        and ``cld``: for example
        ``cmd = cm_p * roll_rate + cm_q * pitch_rate + cm_r * yaw_rate``.
    LinearGenericSurface.cN : Function
        Normal force coefficient, the sum of its forcing and damping parts.
    LinearGenericSurface.cY : Function
        Side force coefficient, the sum of its forcing and damping parts.
    LinearGenericSurface.cA : Function
        Axial force coefficient, the sum of its forcing and damping parts.
    LinearGenericSurface.cm : Function
        Pitching moment coefficient, the sum of its forcing and damping parts.
    LinearGenericSurface.cn : Function
        Yawing moment coefficient, the sum of its forcing and damping parts.
    LinearGenericSurface.cl : Function
        Roll moment coefficient, the sum of its forcing and damping parts.
    LinearGenericSurface.cL : Function
        Lift coefficient, the force perpendicular to the airflow.
    LinearGenericSurface.cD : Function
        Drag coefficient, the force along the airflow.
    LinearGenericSurface.cQ : Function
        Crosswind coefficient, the side force relative to the airflow.
    LinearGenericSurface.aerodynamic_center : Function
        Pitch-plane aerodynamic center of the surface along the rocket's axis,
        in meters from the surface's position (positive toward the nose), as a
        function of Mach. It is the point where the change of the normal force
        acts when the angle of attack changes a little from zero. It is the
        same point as ``center_of_pressure`` when no moment coefficient is
        given.
    LinearGenericSurface.aerodynamic_center_yaw : Function
        Yaw-plane aerodynamic center of the surface along the rocket's axis, in
        meters from the surface's position (positive toward the nose), as a
        function of Mach: the same for the side force and the sideslip angle.
    LinearGenericSurface.reference_area : float
        Reference area, in square meters.
    LinearGenericSurface.reference_length : float
        Reference length, in meters.
    LinearGenericSurface.reynolds_length : float
        Length scale of the Reynolds number, in meters.
    LinearGenericSurface.name : str
        Name of the surface.
    LinearGenericSurface.center_of_pressure : tuple
        Point where the forces are applied, ``(x, y, z)`` in meters, as given.
    LinearGenericSurface.cp : tuple
        The same point as ``(cpx, cpy, cpz)``. ``cpz`` is 0 when the center of
        pressure varies with Mach, since the force is then applied at ``z = 0``.
    LinearGenericSurface.cpx : float
        x coordinate of ``cp``, in meters.
    LinearGenericSurface.cpy : float
        y coordinate of ``cp``, in meters.
    LinearGenericSurface.cpz : float
        z coordinate of ``cp``, in meters.
    LinearGenericSurface.active_during : str
        The motor phase the surface produces force in, as given.
    LinearGenericSurface.active : bool
        Whether the surface starts each flight switched on.
    LinearGenericSurface.force_convention : str
        Frame the force coefficients were given in: ``"body"`` or ``"wind"``.
    LinearGenericSurface.independent_vars : list of str
        The variables every coefficient is called with, in order.
    LinearGenericSurface.prints : _LinearGenericSurfacePrints
        The prints of the surface. Use help(LinearGenericSurface.prints) to know more.
    LinearGenericSurface.plots : _LinearGenericSurfacePlots
        The plots of the surface. Use help(LinearGenericSurface.plots) to know more.
    """

    # The six coefficients and their terms: derivative suffix -> the variable it
    # multiplies (``None`` for the constant term)
    _COEFFICIENTS = ("cN", "cY", "cA", "cm", "cn", "cl")
    _FORCING_TERMS = {"0": None, "alpha": "alpha", "beta": "beta"}
    _DAMPING_TERMS = {"p": "roll_rate", "q": "pitch_rate", "r": "yaw_rate"}
    # Force prefix in the body frame -> in the wind frame
    _BODY_TO_WIND_PREFIX = {"cN": "cL", "cY": "cQ", "cA": "cD"}

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Generic Linear Surface",
        reynolds_length=None,
        interpolation=None,
        extrapolation=None,
        force_convention=None,
        active_during="always",
        axisymmetric=False,
        active=True,
    ):
        """Create a linear aerodynamic surface from its coefficient derivatives.

        Instead of whole coefficients, you give how each coefficient changes with
        the angle of attack, the sideslip angle and the rotation rates, and the
        surface adds the terms up. Use it for stability-derivative data.

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
            The coefficient derivatives (slopes), by name. Any you leave out are
            set to 0. Each one can be a constant, a function, or a path to a data
            file, and says how one force or moment coefficient changes with one
            variable (angle in radians, or a non-dimensional rotation rate). The
            names follow the pattern ``<coefficient>_<variable>``: the coefficient
            is normal force ``cN``, side force ``cY``, axial force ``cA``, pitch
            moment ``cm``, yaw moment ``cn`` or roll moment ``cl``; the variable
            is ``0`` (the value at zero angle of attack, zero sideslip and zero
            rates),
            ``alpha``, ``beta``, ``p`` (roll rate), ``q`` (pitch rate) or ``r``
            (yaw rate). With ``force_convention="wind"`` the force derivatives are
            named after the wind-frame coefficients instead (lift ``cL``, side
            ``cQ``, drag ``cD`` -- e.g. ``cL_alpha``, ``cD_0``, ``cQ_beta``); the
            moment names are unchanged.

            For example ``cN_alpha`` is how much the normal force coefficient
            changes per radian of angle of attack, ``cm_q`` how much the pitch
            moment coefficient changes with the pitch rate, ``cl_p`` the roll
            damping, and ``cA_0`` the axial force (drag) coefficient at zero
            angle. That gives 36 names in all: each of ``cN``, ``cY``, ``cA``,
            ``cm``, ``cn``, ``cl`` combined with each of ``0``, ``alpha``,
            ``beta``, ``p``, ``q``, ``r``.

            Each coefficient is the sum of its six terms, for example
            ``cm = cm_0 + cm_alpha * alpha + cm_beta * beta + cm_p * roll_rate +
            cm_q * pitch_rate + cm_r * yaw_rate``. A derivative that opposes the
            motion, such as a pitch damping ``cm_q``, is given as a negative
            number. For a rocket that behaves the same in every plane, the yaw
            derivatives mirror the pitch ones with a sign flip on the force
            and moment slopes (``cY_beta = -cN_alpha``, ``cn_beta =
            -cm_alpha``) and none on the rate terms (``cY_r = cN_q``,
            ``cn_r = cm_q``); see :ref:`lineargenericsurface_axisymmetric`.

            Each derivative can itself depend on ``alpha``, ``beta``, ``mach``,
            ``reynolds``, ``pitch_rate``, ``yaw_rate`` and ``roll_rate`` (angles
            in radians, or ``alpha_deg`` and ``beta_deg`` in degrees; rates in
            reduced form, such as ``q * L_ref / (2 * V)`` for the pitch rate),
            most often on Mach alone. It can be given as:

            - a number: a constant.
            - a function whose arguments are named after the variables it uses,
              for example ``lambda mach: ...``.
            - the path to a CSV file whose header names the variables, with the
              derivative in the last column. A two-column file without a header
              is read as Mach against the derivative.
            - a list or numpy array of data points. With two columns it is read
              as Mach against the derivative; with more, give it as a pair with
              the names of its variables, for example
              ``(points, ["mach", "reynolds"])``.
            - a :class:`Function`, or a pair ``(axes, values)`` for values on a
              regular grid.
        center_of_pressure : tuple, list, optional
            Point where the aerodynamic forces are applied and about which the
            moment derivatives are taken, as ``(x, y, z)`` in meters. It is
            measured from the position the surface is added to the rocket at:
            ``z`` runs along the rocket's centerline and is positive toward the
            nose, whichever coordinate system orientation the rocket uses. The
            default value is (0, 0, 0).

            The ``z`` component may instead vary with Mach: give it as a
            function of Mach (``lambda mach: ...``), a one-input
            :class:`Function`, or a two-column table ``[[mach, z], ...]``. The
            force is then applied at ``z = 0`` and its moment about the given
            center of pressure is added to the moment derivatives
            (``cm_alpha + cN_alpha * z(mach) / L_ref`` and so on). Such a center
            of pressure is fixed when the surface is created.
        name : str, optional
            Name of the surface. Default is ``"Generic Linear Surface"``.
        reynolds_length : int, float, optional
            Length scale, in meters, of the Reynolds number passed to the
            derivatives. Set it to the length your Reynolds-dependent data was
            tabulated against (for example the rocket's body length). ``None``
            (the default) uses ``reference_length`` (the diameter). Has no
            effect unless a derivative depends on ``reynolds``.
        interpolation : str or dict, optional
            How tabulated derivatives interpolate between points. The accepted
            methods depend on the table: a 1-D table (e.g. a Mach-only curve)
            accepts ``"linear"``, ``"akima"``, ``"spline"`` and ``"polynomial"``;
            a multi-dimensional scattered table accepts ``"linear"``,
            ``"shepard"`` and ``"rbf"``; and a multi-dimensional table on a
            regular grid accepts ``"linear"``, ``"nearest"``, ``"slinear"``,
            ``"cubic"``, ``"quintic"`` and ``"pchip"`` (with ``"spline"`` read as
            ``"cubic"`` and ``"akima"`` as ``"pchip"``). Give one string for
            every derivative, or a dict keyed by derivative name (names left out
            use the default). ``None`` (the default) uses ``"linear"`` for tables
            built here and keeps a pre-built ``Function``'s own setting.
        extrapolation : str or dict, optional
            How tabulated derivatives behave outside their data range:
            ``"constant"`` holds the value at the nearest data edge,
            ``"natural"`` keeps following the curve, and ``"zero"`` returns 0.
            Give one string for every derivative, or a dict keyed by derivative
            name (names left out use the default). ``None`` (the default) uses
            ``"constant"`` for tables built here and keeps whatever a pre-built
            ``Function`` already carries. Only affects tabulated sources
            (constants and functions are evaluated directly).
        force_convention : str, optional
            The frame the force derivatives are given in:

            - ``"body"``: ``cN_*`` (normal), ``cY_*`` (side) and ``cA_*``
              (axial).
            - ``"wind"``: ``cL_*`` (lift), ``cQ_*`` (side) and ``cD_*`` (drag).

            The moment derivatives (``cm_*``, ``cn_*``, ``cl_*``) are the same
            in both. ``None`` (the default) works out the frame from the names,
            and uses ``"body"`` when there are no force derivatives. A
            wind-frame input is converted once to the body-frame derivatives the
            surface stores, by linearizing the angle-of-attack and sideslip
            rotation about zero: the straight renames ``cN_0 = cL_0``,
            ``cN_beta = cL_beta``, the rate derivatives, and the cross terms
            ``cN_alpha = cL_alpha + cD_0``, ``cY_beta = cQ_beta - cD_0``,
            ``cA_alpha = cD_alpha - cL_0`` and ``cA_beta = cD_beta + cQ_0``. At
            zero angle this reduces to ``cN = cL``, ``cY = cQ``, ``cA = cD``.
        active_during : str, optional
            The motor phase this surface produces aerodynamic force in:

            - ``"always"`` (default): the surface always contributes force.
            - ``"power_on"``: only while the motor is burning.
            - ``"power_off"``: only after the motor has burned out.

            To switch a surface on or off at any other moment, such as apogee,
            use an event: see ``active`` below.
        axisymmetric : bool, optional
            Set it to ``True`` when the data describes a rocket (or a part) that
            behaves the same in every plane through its axis, such as a rocket
            with evenly spaced fins. You then give only the pitch-plane
            derivatives (``cN_*`` or ``cL_*``, and ``cm_*``), as stability
            derivatives are usually reported, and the yaw-plane ones are filled
            in for you: ``cY_beta = -cN_alpha``, ``cn_beta = -cm_alpha``,
            ``cY_r = cN_q`` and ``cn_r = cm_q``. The axial and roll derivatives
            are used as given.

            With ``True``, do not give any yaw-plane derivative (``cY_*``,
            ``cQ_*`` or ``cn_*``), nor a sideways force or moment at zero angle
            (``cN_0``, ``cm_0``, ``cN_p``, ``cm_p``). A pitch-plane derivative may depend on
            ``mach``, ``reynolds``, the rates and ``alpha_total``, but not on
            ``alpha``, ``beta`` or ``phi``, which single out one plane.

            Default is ``False``: the two planes are used as given, and a plane
            without derivatives produces no force.
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
            If a name is not one of the 36 derivatives (a coefficient value such
            as ``cN`` included), if wind-frame and body-frame force derivatives
            are mixed without a ``force_convention``, if ``force_convention``
            or ``active_during`` is not one of the accepted values, or if
            ``axisymmetric`` is ``True`` and a derivative breaks the rules
            above.

        See Also
        --------
        :ref:`genericsurfaces`
        """
        # Read while the coefficients are built
        self._axisymmetric = bool(axisymmetric)
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

        self.compute_all_coefficients()

        self.prints = _LinearGenericSurfacePrints(self)
        self.plots = _LinearGenericSurfacePlots(self)

    @classmethod
    def _get_default_coefficients(cls):
        """Return the 36 derivative names (``cN_0`` ... ``cl_r``), each at 0."""
        return {
            f"{coefficient}_{suffix}": 0
            for coefficient in cls._COEFFICIENTS
            for suffix in (*cls._FORCING_TERMS, *cls._DAMPING_TERMS)
        }

    @classmethod
    def _wind_default_coefficient_names(cls):
        """Return the 36 derivative names with the force prefixes in the wind frame.

        ``cN_*`` becomes ``cL_*``, ``cY_*`` becomes ``cQ_*`` and ``cA_*`` becomes
        ``cD_*``; the moment names are unchanged.
        """
        names = set()
        for key in cls._get_default_coefficients():
            prefix, sep, suffix = key.partition("_")
            names.add(f"{cls._BODY_TO_WIND_PREFIX.get(prefix, prefix)}{sep}{suffix}")
        return names

    @classmethod
    def _input_coefficient_names(cls):
        """Return every name a derivative can be given under, in either frame."""
        return (
            set(cls._get_default_coefficients()) | cls._wind_default_coefficient_names()
        )

    def _force_frames_present(self, coefficients):
        """Tell which force frames the names belong to, as ``(has_wind, has_body)``.

        The frame is read from the prefix: ``cL_alpha`` is wind, ``cN_alpha`` body.
        """
        prefixes = {key.split("_", 1)[0] for key in coefficients}
        has_wind = bool(prefixes & set(self._WIND_FORCE_NAMES))
        has_body = bool(prefixes & set(self._BODY_FORCE_NAMES))
        return has_wind, has_body

    def _check_coefficients(self, input_coefficients, default_coefficients):
        """Raise a ``ValueError`` for a name that is not one of the derivatives.

        On top of the generic check, a coefficient value such as ``cN`` gets a hint
        that a derivative is expected.
        """
        values = sorted(
            set(input_coefficients) & GenericSurface._input_coefficient_names()
        )
        if values:
            raise ValueError(
                f"{', '.join(values)}: LinearGenericSurface takes derivatives, "
                f"such as {values[0]}_alpha or {values[0]}_q, not coefficient "
                "values. For a coefficient given against the angle, use "
                "GenericSurface."
            )
        super()._check_coefficients(input_coefficients, default_coefficients)

    def _as_coefficient(self, source, name, single_var=None):
        """Wrap a derivative as an :class:`AeroCoefficient`.

        A derivative's name already fixes its angle or rate, so a one-column table
        or a headerless file is read against Mach.
        """
        return super()._as_coefficient(source, name, single_var or "mach")

    def _wind_input_to_body(self, coefficients):
        """Convert wind-frame derivatives into body-frame ones.

        The wind-frame ``cL_*``, ``cQ_*`` and ``cD_*`` become ``cN_*``, ``cY_*``
        and ``cA_*``. Linearizing the rotation between the two frames about
        ``alpha = beta = 0`` makes every force derivative a straight rename
        (``cN_0 = cL_0``, ``cN_q = cL_q``, ...) except four cross terms::

            cN_alpha = cL_alpha + cD_0      cA_alpha = cD_alpha - cL_0
            cY_beta  = cQ_beta  - cD_0      cA_beta  = cD_beta  + cQ_0

        At zero angle this reduces to ``cN = cL``, ``cY = cQ``, ``cA = cD``. The
        moment derivatives are the same in both frames and pass through.
        """
        self._check_coefficients(coefficients, self._wind_default_coefficient_names())

        def combined(first_name, second_name, sign, name):
            # first + sign * second, over only the variables the two use. When
            # one is zero the other is kept as it is.
            first, second = (
                self._as_coefficient(coefficients.get(n, 0), n)
                for n in (first_name, second_name)
            )
            if second.is_zero:
                return first
            if first.is_zero:
                return second if sign > 0 else second * -1.0
            used = [
                var
                for var in self.independent_vars
                if var in first.depends_on or var in second.depends_on
            ]
            if not used:
                return first.get_value_opt() + sign * second.get_value_opt()
            evaluate_first = first.evaluator(used)
            evaluate_second = second.evaluator(used)
            return _as_function(
                lambda *args: evaluate_first(*args) + sign * evaluate_second(*args),
                used,
                name,
            )

        body = {
            f"{body_prefix}_{suffix}": coefficients.get(f"{wind_prefix}_{suffix}", 0)
            for body_prefix, wind_prefix in self._BODY_TO_WIND_PREFIX.items()
            for suffix in (*self._FORCING_TERMS, *self._DAMPING_TERMS)
        }
        body["cN_alpha"] = combined("cL_alpha", "cD_0", 1.0, "cN_alpha")
        body["cY_beta"] = combined("cQ_beta", "cD_0", -1.0, "cY_beta")
        body["cA_alpha"] = combined("cD_alpha", "cL_0", -1.0, "cA_alpha")
        body["cA_beta"] = combined("cD_beta", "cQ_0", 1.0, "cA_beta")
        for name, value in coefficients.items():
            if name.split("_", 1)[0] not in self._WIND_FORCE_NAMES:
                body[name] = value
        return body

    # Yaw-plane derivative suffix -> the pitch-plane suffix it comes from, for a
    # surface that behaves the same in every plane: a quarter turn about the
    # axis maps the pitch plane onto the yaw plane
    _YAW_FROM_PITCH = {
        "0": "0",
        "alpha": "beta",
        "beta": "alpha",
        "q": "r",
        "r": "q",
        "p": "p",
    }

    def _complete_body_coefficients(self, coefficients):
        """Fill in the yaw-plane derivatives of an axisymmetric surface from
        its pitch-plane ones: ``cY_beta = -cN_alpha``, ``cY_alpha = cN_beta``,
        ``cY_r = cN_q``, ``cY_q = -cN_r``, ``cY_0 = cN_0`` and ``cY_p = cN_p``,
        and the same from ``cm`` to ``cn``."""
        if not self._axisymmetric:
            return coefficients
        given = sorted(
            name
            for name, coefficient in self._input_coefficients.items()
            if name.split("_")[0] in ("cY", "cQ", "cn") and not coefficient.is_zero
        )
        if given:
            raise ValueError(
                f"{', '.join(given)} cannot be given with axisymmetric=True, "
                "which fills in the yaw-plane derivatives from the pitch-plane "
                "ones. Leave them out, or use axisymmetric=False and give both "
                "planes."
            )
        coefficients = dict(coefficients)
        for pitch, yaw in (("cN", "cY"), ("cm", "cn")):
            for suffix, source in self._YAW_FROM_PITCH.items():
                coefficients.pop(f"{yaw}_{suffix}", None)
                name = f"{pitch}_{source}"
                if name not in coefficients:
                    continue
                derivative = self._as_coefficient(coefficients[name], name)
                if derivative.is_zero:
                    continue
                if source in ("0", "p"):
                    raise ValueError(
                        f"{name} cannot be given with axisymmetric=True: a "
                        "sideways force or moment at zero angle of attack "
                        "points in one direction, which a rocket that behaves "
                        "the same in every plane does not have. Leave it out, "
                        "or use axisymmetric=False and give both planes."
                    )
                one_plane = sorted(
                    set(derivative.source_angles) & {"alpha", "beta", "phi"}
                )
                if one_plane:
                    raise ValueError(
                        f"{name} depends on {' and '.join(one_plane)}, which "
                        "singles out one plane, so it cannot be used with "
                        "axisymmetric=True. Give it against alpha_total, or "
                        "use axisymmetric=False and give both planes."
                    )
                sign = -1.0 if suffix in ("beta", "q") else 1.0
                coefficients[f"{yaw}_{suffix}"] = sign * derivative
        return coefficients

    def to_dict(self, include_outputs=False, **kwargs):
        data = super().to_dict(include_outputs=include_outputs, **kwargs)
        data["axisymmetric"] = self._axisymmetric
        return data

    @classmethod
    def _arguments_from_dict(cls, data):
        arguments = super()._arguments_from_dict(data)
        arguments["axisymmetric"] = data.get("axisymmetric", False)
        return arguments

    def compute_all_coefficients(self):
        """Build the six coefficients from their derivatives.

        For each coefficient (``cN``, ``cY``, ``cA``, ``cm``, ``cn``, ``cl``)
        three attributes are set, all functions of the surface's variables:

        - the whole coefficient, for example ``cm``, used in the simulation;
        - its forcing part ``cmf = cm_0 + cm_alpha * alpha + cm_beta * beta``;
        - its damping part
          ``cmd = cm_p * roll_rate + cm_q * pitch_rate + cm_r * yaw_rate``.

        The whole coefficient is the sum of the two parts. A damping derivative
        that opposes the motion is a negative number, so nothing is subtracted.
        """
        every_term = {**self._FORCING_TERMS, **self._DAMPING_TERMS}
        for coefficient in self._COEFFICIENTS:
            for suffix, terms in (
                ("f", self._FORCING_TERMS),
                ("d", self._DAMPING_TERMS),
                ("", every_term),
            ):
                setattr(
                    self,
                    coefficient + suffix,
                    self._linear_coefficient(coefficient, terms),
                )

    def _linear_coefficient(self, coefficient, terms):
        """Sum the given terms of one coefficient into a :class:`Function`.

        Each term is a derivative times its variable, such as ``cm_q * pitch_rate``;
        the ``_0`` term is used as it is. Zero derivatives are left out, which keeps
        the sum cheap during the simulation.
        """
        active = []
        for suffix, variable in terms.items():
            derivative = getattr(self, f"{coefficient}_{suffix}")
            if not derivative.is_zero:
                position = (
                    None if variable is None else self.independent_vars.index(variable)
                )
                active.append((derivative.get_value_opt, position))

        def total(*args):
            value = 0.0
            for evaluate, position in active:
                term = evaluate(*args)
                value += term if position is None else term * args[position]
            return value

        return _as_function(total, self.independent_vars, coefficient)

    def _evaluate_stability_derivatives(self):
        """Build the center-of-pressure functions.

        The slopes ``cN_alpha``, ``cm_alpha``, ``cY_beta`` and ``cn_beta`` are the
        given derivatives, so nothing is differentiated.
        """
        self._set_stability_accessors()
