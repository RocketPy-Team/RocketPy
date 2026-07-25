import copy
import csv
import inspect

from rocketpy.mathutils import Function

# Single source of truth for the seven base coefficient independent variables.
BASE_INDEPENDENT_VARS = [
    "alpha",
    "beta",
    "mach",
    "reynolds",
    "pitch_rate",
    "yaw_rate",
    "roll_rate",
]


def build_independent_vars(control_variables=()):
    """Build the ordered independent-variable list of a coefficient/surface.

    The seven base axes (``BASE_INDEPENDENT_VARS``), plus any
    ``control_variables`` (axes supplied externally, e.g. by a controller).
    Shared by :class:`AeroCoefficient` and :class:`GenericSurface` so the
    ordering is defined in exactly one place.
    """
    return list(BASE_INDEPENDENT_VARS) + list(control_variables)


class AeroCoefficient:
    """A single aerodynamic coefficient (such as lift or drag), stored using
    only the variables it actually depends on."""

    def __init__(
        self,
        source,
        depends_on=None,
        control_variables=(),
        name="coefficient",
        extrapolation=None,
        interpolation=None,
        single_var=None,
    ):
        """Build a coefficient from a value, a data table, or a function.

        A plain number is stored as a constant. Anything else is stored as a
        :class:`Function` of only the variables it depends on, so a coefficient
        that varies with Mach alone stays a simple 1-D curve instead of being
        spread across all seven variables. When the coefficient is evaluated,
        the variables it does not use are simply ignored.

        Most of the time you only pass ``source`` and leave ``depends_on`` as
        ``None``, so the variables are worked out automatically. This is the
        same input a :class:`GenericSurface` accepts. Pass ``depends_on``
        yourself only when the source and the order of its inputs are already
        known (used internally by the Barrowman surfaces and when loading a
        saved rocket).

        Parameters
        ----------
        source : int, float, str, list, tuple, callable, Function, or AeroCoefficient
            The coefficient value, given in one of these forms:

            - **number**: a constant coefficient that never changes.
            - **function or lambda**: a coefficient computed from its inputs.
              Name the arguments after the variables they use (e.g.
              ``lambda alpha, mach: ...``), or give one argument per variable,
              or a single argument together with ``single_var``.
            - **Function**: a :class:`Function` you already built, used as is.
              If ``extrapolation`` is given it is applied to a copy, so the
              Function you passed in is left unchanged.
            - **list or tuple of data points**: a table of values, read the
              same way as the same data in a CSV file. The variables it depends
              on are worked out from the table, using ``single_var`` for a
              one-input table.
            - **str**: the path to a data file. A ``.csv`` file has one column
              per variable (named in the header) and the coefficient value in
              the last column; a headerless two-column file is a table of
              ``single_var`` versus the value.
            - **AeroCoefficient**: an existing coefficient, reused as is. This
              lets one coefficient be shared by several surfaces and lets a
              rocket be saved and loaded.
        depends_on : sequence of str, optional
            The variables this coefficient actually uses, chosen from the
            surface's variables: the seven base ones ``"alpha"``, ``"beta"``,
            ``"mach"``, ``"reynolds"``, ``"pitch_rate"``, ``"yaw_rate"``,
            ``"roll_rate"``, plus any names in
            ``control_variables``. List them in the same order as the source's
            own inputs (a function's arguments, a CSV's columns). For example,
            ``()`` for a constant, ``("mach",)`` for a Mach-only curve, or the
            whole list for something that uses every variable. A name that is
            not one of the surface's variables raises a ``ValueError``. Leave it
            as ``None`` (the default) to have it worked out from ``source``.
        control_variables : sequence of str, optional
            Names of extra variables, such as control-surface deflections set by
            a controller. They are added after the seven base variables, in the
            order given. Empty for ordinary surfaces. Default ``()``.
        name : str, optional
            A readable name for the coefficient (e.g. ``"cL_alpha"`` or
            ``"Drag Coefficient with Power Off"``). It appears in error messages,
            so a clear name makes problems easier to spot. Default
            ``"coefficient"``.
        extrapolation : str, optional
            What the coefficient does outside the range of its data table:
            ``"constant"`` holds the value at the nearest edge (the safe default
            for aerodynamic coefficients, which should not shoot off to
            unrealistic values), ``"natural"`` keeps following the curve, and
            ``"zero"`` returns ``0``. ``None`` (the default) leaves a
            :class:`Function` you passed in unchanged and uses ``"constant"`` for
            a table built here. Has no effect on a constant or a function, which
            are evaluated directly.
        interpolation : str, optional
            How the coefficient reads values *between* the points of its data
            table, for example ``"linear"``, ``"akima"`` or ``"spline"`` for a
            one-input table. Only affects data tables (CSV files, lists of
            points, a :class:`Function`); it has no effect on a constant or a
            function. ``None`` (the default) leaves a :class:`Function` you
            passed in unchanged and uses ``"linear"`` for a table built here.
        single_var : str, optional
            Which variable a one-input table or function maps to. Used only when
            working out the variables of a single-input source: a headerless
            two-column CSV, a one-input :class:`Function`, or a one-argument
            function. ``None`` (the default) guesses it from the input's label
            and otherwise falls back to the first variable. Ignored when
            ``depends_on`` is given. Default ``None``.
        """
        self.name = name
        self.extrapolation = extrapolation
        self.interpolation = interpolation
        self.control_variables = tuple(control_variables)
        # ``control_variables`` completes the full ordered variable list: every
        # coefficient's argument order and each variable's position. This is a
        # surface-wide property, distinct from ``depends_on`` (the subset a
        # single coefficient reads), and it is passed in rather than derived
        # from ``depends_on``: inferring ``depends_on`` already needs this list.
        self.independent_vars = tuple(build_independent_vars(control_variables))
        # Infer the stored source and its dependencies from the raw input when
        # ``depends_on`` is not given.
        if depends_on is None:
            source, depends_on = self._resolve_input(source, single_var)
            extrapolation = self.extrapolation
            interpolation = self.interpolation
        # ``depends_on`` is kept in the given order because it matches the
        # positional argument order of the stored source (callable parameters,
        # CSV columns, …). ``_indices`` then maps the full argument tuple
        # to the source's own argument order.
        self.depends_on = tuple(depends_on)
        unknown = [var for var in self.depends_on if var not in self.independent_vars]
        if unknown:
            raise ValueError(
                f"{name} depends on unknown variable(s) {unknown}; "
                f"valid variables are {list(self.independent_vars)}."
            )
        self._indices = tuple(
            self.independent_vars.index(var) for var in self.depends_on
        )

        self.is_zero = False
        self._constant = None
        if isinstance(source, Function):
            # Only override interpolation/extrapolation when explicitly asked,
            # and always on a copy (the Function may be shared elsewhere).
            if interpolation is not None or extrapolation is not None:
                source = copy.deepcopy(source)
                # Interpolation names like "akima"/"spline" are 1-D concepts; a
                # multi-dimensional Function (e.g. a regular grid) keeps its own
                # interpolation, whose method is fixed when the grid is built, so
                # a 1-D name here would wrongly fall back to "shepard".
                if interpolation is not None and source.__dom_dim__ == 1:
                    source.set_interpolation(interpolation)
                if extrapolation is not None:
                    source.set_extrapolation(extrapolation)
            self.function = source
        elif callable(source):
            self.function = Function(
                source,
                list(self.depends_on) or ["x"],
                [name],
                interpolation=interpolation or "linear",
                extrapolation=extrapolation or "constant",
            )
        else:
            # Scalar constant.
            self._constant = float(source)
            self.is_zero = self._constant == 0.0
            self.function = Function(self._constant)

        self._evaluate = self.function.get_value_opt

    def _resolve_input(self, source, single_var):
        """Infer ``(stored source, depends_on)`` from a raw coefficient input.

        Parameters
        ----------
        source : int, float, str, list, tuple, callable, Function or AeroCoefficient
            Raw coefficient input: a scalar, a CSV file path (or any other path
            read by :class:`Function`), a list/tuple of data points, a callable,
            a pre-built :class:`Function`, or an existing ``AeroCoefficient``.
        single_var : str or None
            Name of the independent variable a one-dimensional input depends on.
            When ``None``, it is inferred from the source (see
            :meth:`_infer_single_var` / :meth:`_infer_callable_depends_on`).

        Returns
        -------
        tuple
            ``(stored_source, depends_on)`` where ``stored_source`` is the scalar
            or :class:`Function` kept internally and ``depends_on`` is the tuple
            of independent-variable names it depends on.
        """
        name = self.name
        independent_vars = self.independent_vars
        n_vars = len(independent_vars)

        if isinstance(source, AeroCoefficient):
            # An already-built coefficient passed straight through, re-keyed to
            # this surface's variable order. Adopt its extrapolation when none
            # was requested.
            if self.extrapolation is None:
                self.extrapolation = source.extrapolation
            value = (
                source._constant if source._constant is not None else source.function
            )
            return value, source.depends_on

        if isinstance(source, str):
            if source.lower().endswith(".csv"):
                return self._load_csv(
                    source,
                    name,
                    independent_vars,
                    extrapolation=self.extrapolation or "constant",
                    interpolation=self.interpolation or "linear",
                    single_var=single_var,
                )
            # Any other path is read by Function
            source = Function(
                source,
                interpolation=self.interpolation or "linear",
                extrapolation=self.extrapolation or "constant",
            )

        # A list/tuple of data points is parsed by Function and handled below
        if isinstance(source, (list, tuple)):
            try:
                source = Function(
                    list(source),
                    interpolation=self.interpolation or "linear",
                    extrapolation=self.extrapolation or "constant",
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Invalid list/tuple input for {name}: could not be parsed "
                    "into a Function of the independent variables."
                ) from exc

        if isinstance(source, Function):
            dom_dim = source.__dom_dim__
            if dom_dim == n_vars:
                return source, list(independent_vars)
            if dom_dim == 1:
                # A 1-D Function depends on ``single_var`` when given, else on
                # the first independent variable unless its input name matches.
                return source, [
                    single_var or self._infer_single_var(source, independent_vars)
                ]
            raise ValueError(
                f"{name} Function must have {n_vars} input arguments "
                f"({', '.join(independent_vars)}) or be one-dimensional."
            )

        if callable(source):
            return source, self._infer_callable_depends_on(
                source, independent_vars, name, single_var=single_var
            )

        # Anything else must be a scalar number.
        try:
            float(source)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Invalid input for {name}: must be a number, a CSV file path, "
                "a list of data points, a callable, or a Function."
            ) from exc
        return source, ()

    @staticmethod
    def _load_csv(
        file_path,
        name,
        independent_vars,
        extrapolation="constant",
        interpolation="linear",
        single_var=None,
    ):  # pylint: disable=too-many-statements
        """Load a coefficient CSV at minimal dimension.

        Expects header-based CSV data whose columns (except the last) are
        independent variables among ``independent_vars``; the last column is the
        coefficient value. The coefficient is stored over only the columns that
        are present, in their header order. A headerless two-column file is
        treated as a one-dimensional table over ``single_var``.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        name : str
            Coefficient name, used for error messages and the Function output.
        independent_vars : sequence of str
            The owning surface's ordered independent variables, used to validate
            the CSV header columns.
        extrapolation : str, optional
            Extrapolation method for the loaded ``Function``. Defaults to
            ``"constant"`` (holds the edge value past the tabulated range).
        interpolation : str, optional
            Interpolation method for the loaded ``Function``. Defaults to
            ``"linear"``. For 1-D and non-grid tables it is used directly; a
            strict Cartesian grid uses ``"regular_grid"`` with the method mapped
            from this value (see :meth:`Function.from_regular_grid_csv`).
        single_var : str, optional
            Independent variable a headerless two-column table depends on.
            Defaults to the first independent variable.

        Returns
        -------
        tuple
            ``(function, depends_on)`` where ``function`` is a low-dimensional
            ``Function`` over the present columns and ``depends_on`` lists those
            columns. Consumed by :meth:`_resolve_input`.
        """
        independent_vars = list(independent_vars)

        try:
            with open(file_path, mode="r") as file:
                reader = csv.reader(file)
                header = next(reader)
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading {name} CSV file: {e}") from e
        except StopIteration as e:
            raise ValueError(f"Invalid or empty CSV file for {name}.") from e

        if not header:
            raise ValueError(f"Invalid or empty CSV file for {name}.")

        header = [column.strip() for column in header]

        # Headerless two-column (x, coefficient) table: a 1-D table over
        # ``single_var`` (e.g. a Mach-only drag curve given as ``mach, cd``).
        def _is_numeric(value):
            try:
                float(value)
                return True
            except (TypeError, ValueError):
                return False

        if len(header) == 2 and all(_is_numeric(cell) for cell in header):
            csv_func = Function(
                file_path,
                interpolation=interpolation,
                extrapolation=extrapolation,
            )
            return csv_func, [single_var or independent_vars[0]]

        present_columns = [col for col in independent_vars if col in header]

        invalid_columns = [col for col in header[:-1] if col not in independent_vars]
        if invalid_columns:
            raise ValueError(
                f"Invalid independent variable(s) in {name} CSV: "
                f"{invalid_columns}. Valid options are: {independent_vars}."
            )

        if header[-1] in independent_vars:
            raise ValueError(
                f"Last column in {name} CSV must be the coefficient"
                " value, not an independent variable."
            )

        if not present_columns:
            raise ValueError(f"No independent variables found in {name} CSV.")

        ordered_present_columns = [
            col for col in header[:-1] if col in independent_vars
        ]

        csv_func = Function.from_regular_grid_csv(
            file_path,
            ordered_present_columns,
            name,
            extrapolation=extrapolation,
            interpolation=interpolation,
        )
        if csv_func is None:
            csv_func = Function(
                file_path,
                interpolation=interpolation,
                extrapolation=extrapolation,
            )

        return csv_func, ordered_present_columns

    @staticmethod
    def _infer_single_var(function, independent_vars):
        """Best-effort name of the variable a 1-D Function depends on."""
        try:
            label = function.__inputs__[0]
        except (AttributeError, IndexError, TypeError):
            return independent_vars[0]
        label_lower = str(label).lower()
        # Exact match first; then substring, longest variable name first, so a
        # label containing a longer variable name binds to it rather than to a
        # shorter variable that happens to be a substring of it.
        for var in independent_vars:
            if var == label_lower:
                return var
        for var in sorted(independent_vars, key=len, reverse=True):
            if var in label_lower:
                return var
        return independent_vars[0]

    @staticmethod
    def _infer_callable_depends_on(func, independent_vars, name, single_var=None):
        """Work out which variables a function coefficient uses, from its
        arguments.

        Three ways to write the function are accepted, tried in this order:

        1. One argument plus ``single_var``: the function takes a single
           argument and ``single_var`` says which variable it is, whatever the
           argument is named (e.g. a Mach-only drag curve ``lambda mach: ...``
           with ``single_var="mach"``).
        2. Arguments named after variables: every argument name matches one of
           the surface's variables, so the names themselves list what the
           function uses (e.g. ``lambda alpha, mach: ...`` uses ``alpha`` and
           ``mach``).
        3. One argument per variable: the function has exactly as many arguments
           as there are variables, so it is taken to use all of them, whatever
           the arguments are named (e.g. ``lambda a, b, m, r, p, q, rr: ...``
           for the seven base variables).

        Anything else raises ``ValueError``.
        """
        n_vars = len(independent_vars)
        try:
            params = list(inspect.signature(func).parameters.values())
        except (TypeError, ValueError):  # pragma: no cover - builtins
            params = []
        names = [p.name for p in params]

        if single_var and len(names) == 1:
            return [single_var]
        if names and set(names) <= set(independent_vars):
            return names
        if len(names) == n_vars:
            return list(independent_vars)
        raise ValueError(
            f"{name} callable must accept {n_vars} positional arguments "
            f"({', '.join(independent_vars)}) or name its parameters after the "
            "independent variables it depends on."
        )

    @property
    def is_zero_coefficient(self):
        """Kept-for-compatibility alias of ``is_zero``: whether the coefficient
        is the constant 0 (the linear model uses it to skip zero terms)."""
        return self.is_zero

    @property
    def __dom_dim__(self):
        """Number of variables the coefficient is called with."""
        return len(self.independent_vars)

    def get_value_opt(self, *args):
        """Fast evaluation without input checking (mirrors
        :meth:`Function.get_value_opt`).

        Receives every variable, passes on only the ones this coefficient uses,
        and evaluates the source. A constant is returned right away.
        """
        if self._constant is not None:
            return self._constant
        return self._evaluate(*(args[i] for i in self._indices))

    # Calling the coefficient is the same as the fast evaluator; the linear
    # model grabs ``get_value_opt`` directly for the hot loop.
    __call__ = get_value_opt

    def __mul__(self, other):
        """Scale the coefficient by ``other``, returning a new AeroCoefficient.

        Used by the Monte Carlo drag factor (``coefficient *= factor``). The
        underlying constant or :class:`Function` is scaled while ``depends_on``,
        the independent-variable axes and ``extrapolation`` are preserved.
        """
        source = self._constant if self._constant is not None else self.function
        return AeroCoefficient(
            source * other,
            self.depends_on,
            self.control_variables,
            self.name,
            extrapolation=self.extrapolation,
            interpolation=self.interpolation,
        )

    __rmul__ = __mul__

    def __repr__(self):
        """Return a concise representation showing the constant or dependencies."""
        if self._constant is not None:
            return f"AeroCoefficient({self.name}={self._constant})"
        return f"AeroCoefficient({self.name}, depends_on={self.depends_on})"

    def slice(self, *free_variables, at=None):
        """Return a :class:`Function` of only the chosen variables, holding the
        others fixed.

        This gives a lower-dimensional view of the coefficient, handy for
        inspection or plotting. For example, ``cL.slice("alpha", "mach")`` is the
        lift coefficient as a function of angle of attack and Mach, with sideslip,
        Reynolds number and the rotation rates held at zero; ``cD.slice("mach")``
        is a Mach-only drag curve.

        Parameters
        ----------
        *free_variables : str
            Names of the variables to keep as inputs, in the order you want them
            (for example ``"mach"`` or ``"alpha", "mach"``). Each must be one of
            this coefficient's independent variables.
        at : dict, optional
            Values to hold the remaining variables at, keyed by variable name.
            Any not listed are held at 0.

        Returns
        -------
        Function
            A Function of ``free_variables`` that evaluates this coefficient with
            the remaining variables held fixed.
        """
        fixed = dict(at or {})
        unknown = [
            var
            for var in list(free_variables) + list(fixed)
            if var not in self.independent_vars
        ]
        if unknown:
            raise ValueError(
                f"{self.name} has no independent variable(s) {unknown}; valid "
                f"variables are {list(self.independent_vars)}."
            )

        free_positions = [self.independent_vars.index(var) for var in free_variables]
        baseline = [fixed.get(var, 0.0) for var in self.independent_vars]

        if not free_variables:
            return Function(self.get_value_opt(*baseline))

        def sliced(*values):
            args = list(baseline)
            for position, value in zip(free_positions, values):
                args[position] = value
            return self.get_value_opt(*args)

        # Give the wrapper an explicit signature so Function reads the right
        # number of inputs (its domain dimension comes from the parameter count).
        sliced.__signature__ = inspect.Signature(
            inspect.Parameter(var, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for var in free_variables
        )
        return Function(sliced, list(free_variables), [self.name])

    def slope(self, variable, *free_variables, at=None, dx=1e-6):
        """Return the derivative of this coefficient with respect to one variable
        as a :class:`Function` of the chosen free variables.

        This is the aerodynamic slope, such as a lift-curve slope. For example,
        ``cL.slope("alpha", "mach")`` is the lift-curve slope ``d(cL)/d(alpha)``
        as a function of Mach, taken at ``alpha = 0`` with sideslip, Reynolds
        number and the rotation rates held at zero.

        Parameters
        ----------
        variable : str
            Name of the variable to differentiate with respect to (for example
            ``"alpha"`` or ``"beta"``). Must be one of this coefficient's
            independent variables, and must not also appear in
            ``free_variables``.
        *free_variables : str
            Names of the variables to keep as inputs of the resulting slope, in
            the order you want them (for example ``"mach"``). Each must be one of
            this coefficient's independent variables. Leave empty to get the
            slope at a single point.
        at : dict, optional
            Values to hold the remaining variables at, keyed by variable name.
            The value for ``variable`` is the point the derivative is taken at
            (default 0, the linearization point). Any variable not listed is held
            at 0.
        dx : float, optional
            Step size used for the numerical differentiation. Default 1e-6.

        Returns
        -------
        Function
            A Function of ``free_variables`` giving ``d(self)/d(variable)`` with
            the remaining variables held fixed.
        """
        fixed = dict(at or {})
        overlap = [var for var in free_variables if var == variable]
        if overlap:
            raise ValueError(
                f"{variable!r} cannot be both differentiated and kept free."
            )
        # Point to differentiate at, defaulting to the linearization point (0).
        diff_point = fixed.pop(variable, 0.0)
        name = f"d({self.name})/d({variable})"

        def evaluate_slope(*free_values):
            # Hold the fixed variables and the current free-variable values,
            # keep only ``variable`` free, and differentiate along it.
            slice_at = dict(fixed)
            for var, value in zip(free_variables, free_values):
                slice_at[var] = value
            return self.slice(variable, at=slice_at).differentiate(diff_point, dx=dx)

        if not free_variables:
            return Function(evaluate_slope(), name)

        evaluate_slope.__signature__ = inspect.Signature(
            inspect.Parameter(var, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for var in free_variables
        )
        return Function(evaluate_slope, list(free_variables), [name])

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        """Serialize the coefficient for :class:`rocketpy._encoders.RocketPyEncoder`."""
        return {
            "source": self._constant if self._constant is not None else self.function,
            "depends_on": list(self.depends_on),
            "control_variables": list(self.control_variables),
            "name": self.name,
            "extrapolation": self.extrapolation,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild an :class:`AeroCoefficient` from its :meth:`to_dict` form."""
        return cls(
            data["source"],
            data["depends_on"],
            data.get("control_variables", ()),
            data["name"],
            extrapolation=data.get("extrapolation"),
            interpolation=data.get("interpolation"),
        )
