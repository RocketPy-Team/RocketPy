"""Minimal-dimension aerodynamic coefficient storage.

A :class:`AeroCoefficient` stores a single aerodynamic coefficient at its
*intrinsic* dimensionality - a constant, or a :class:`Function` over only the
variables the coefficient actually depends on (its ``depends_on``) - and maps
the full coefficient argument tuple (in ``independent_vars`` order) down to that
subset on every call.

This avoids forcing a Mach-only (or constant) coefficient into a full seven
dimensional :class:`Function`: interpolation happens at the right dimension (so
a Mach-only table is not smeared across a 7-D domain) and evaluation passes only
the arguments that matter. It generalizes the per-call ``dict(zip(...))`` subset
selection that the CSV loader used to do inline.
"""

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


def build_independent_vars(unsteady_aero=False, control_variables=()):
    """Build the ordered independent-variable list of a coefficient/surface.

    The seven base axes (``BASE_INDEPENDENT_VARS``), plus ``alpha_dot`` and
    ``beta_dot`` when ``unsteady_aero`` is enabled (axes the flight integrator
    supplies automatically), plus any ``control_variables`` (axes supplied
    externally, e.g. by a controller). Shared by :class:`AeroCoefficient` and
    :class:`GenericSurface` so the ordering is defined in exactly one place.
    """
    names = list(BASE_INDEPENDENT_VARS)
    if unsteady_aero:
        names += ["alpha_dot", "beta_dot"]
    names += list(control_variables)
    return names


class AeroCoefficient:
    """A single aerodynamic coefficient stored at minimal dimensionality.

    Building goes through :meth:`__init__`: pass a raw coefficient input
    (number, callable, :class:`Function`, list/tuple of points, CSV path, or
    another :class:`AeroCoefficient`) and ``depends_on`` is inferred; pass
    ``depends_on`` explicitly only on the fast path where it is already known.
    """

    def __init__(
        self,
        source,
        depends_on=None,
        unsteady_aero=False,
        control_variables=(),
        name="coefficient",
        extrapolation=None,
        single_var=None,
    ):
        """Build a coefficient stored at minimal dimensionality.

        A number is kept as a plain constant. Anything else is wrapped in a
        :class:`Function` over only the variables it depends on (``depends_on``),
        so a Mach-only curve stays 1-D instead of being stretched across all
        seven axes. On each call the full argument tuple is mapped down to just
        those arguments (using the precomputed ``_indices``). The full, ordered
        list of variables comes from ``unsteady_aero`` and ``control_variables``
        via :func:`build_independent_vars`.

        Usually you do not pass ``depends_on``: leave it as ``None`` and it is
        worked out from ``source`` (a number, a callable, a :class:`Function`, a
        list of points, a CSV path, or another :class:`AeroCoefficient`), the
        same inputs :class:`GenericSurface` accepts (see :meth:`_resolve_input`).
        Pass ``depends_on`` yourself only on the fast path, where the source and
        its argument order are already known (the Barrowman surfaces and
        serialization).

        Parameters
        ----------
        source : number, str, list, tuple, callable, Function, or AeroCoefficient
            The coefficient value, or an input it can be worked out from when
            ``depends_on`` is ``None``. The accepted forms are:

            - **number**: kept as a constant. Calls return it directly, and
              ``is_zero`` is set when it is exactly ``0.0`` (the linear model
              uses that to skip the term). It depends on nothing.
            - **callable** (function or ``lambda``): wrapped in a
              :class:`Function`. When ``depends_on`` is worked out, the
              parameter *names* decide it: name them after the variables they
              use (e.g. ``lambda alpha, mach: ...``), give one argument per
              variable, or use one argument together with ``single_var``.
            - **Function**: used as given. If ``extrapolation`` is set, it is
              applied to a copy, never to the object you passed in (it may be
              shared elsewhere).
            - **list/tuple of points**: turned into a :class:`Function` with
              linear interpolation, so a list and the same data in a CSV give
              the same result.
            - **str**: a path to a data file. A ``.csv`` file is read by the CSV
              loader (column headers name the variables; a headerless
              two-column file is a 1-D table over ``single_var``); other files
              are read by :class:`Function`.
            - **AeroCoefficient**: an existing coefficient, re-keyed to this
              surface's variables. This is what lets a surface round-trip
              through ``to_dict``/``from_dict`` and lets one coefficient be
              reused on several surfaces.
        depends_on : sequence of str, optional
            The variables this coefficient actually uses, a (possibly empty)
            subset of the surface's full variable list (set by ``unsteady_aero``
            and ``control_variables``). Keep them in the same order as the
            source's own arguments (a callable's parameters, a CSV's columns):
            that order is used to pick the right values out of the full argument
            tuple on each call. For example, ``()`` for a constant, ``("mach",)``
            for a Mach-only curve, or the whole list for something that uses
            every variable. A name that is not one of the surface's variables
            raises a ``ValueError``. Leave it as ``None`` (the default) to have
            it worked out from ``source``; pass it only on the fast path, where
            the source and its argument order are already known.
        unsteady_aero : bool, optional
            Add the unsteady axes to this coefficient's variables. When ``True``,
            ``alpha_dot`` and ``beta_dot`` (the rates of change of the angle of
            attack and sideslip) are added after the seven base axes, so calls
            take two more arguments. The flight integrator fills these in, using
            ``0`` when it does not compute them, so ordinary tables keep working.
            Match the owning surface's setting. Default ``False``.
        control_variables : sequence of str, optional
            Names of extra axes supplied from outside, such as control-surface
            deflections from a controller. They are added after the base and
            unsteady axes, and each one becomes an extra call argument, in the
            order given. Used by :class:`ControllableGenericSurface` and air
            brakes; empty for ordinary surfaces. Default ``()``.
        name : str, optional
            A readable name for the coefficient (e.g. ``"cL_alpha"`` or
            ``"Drag Coefficient with Power Off"``). It labels the underlying
            :class:`Function` and appears in error messages, so a clear name
            makes problems easier to spot. Default ``"coefficient"``.
        extrapolation : str, optional
            How the stored :class:`Function` behaves outside its data range, one
            of the options of :meth:`Function.set_extrapolation`: ``"constant"``
            holds the edge value (used for drag, which should not run past its
            data), ``"natural"`` keeps following the curve, ``"zero"`` returns
            ``0``. ``None`` (the default) leaves a :class:`Function` you passed
            in unchanged, and uses ``"natural"`` for one built from a callable.
            An override is always applied to a copy, so your object is never
            changed.
        single_var : str, optional
            Which variable a 1-D input maps to. Used only while working out
            ``depends_on`` for a single-dimension source: a headerless
            two-column CSV, a 1-D :class:`Function`, or a one-argument callable.
            ``None`` (the default) guesses it from the input's label, falling
            back to the first variable; drag passes ``"mach"`` so a plain
            Cd-vs-Mach curve maps to Mach. Ignored when ``depends_on`` is given.
            Default ``None``.
        """
        self.name = name
        self.extrapolation = extrapolation
        self.unsteady_aero = unsteady_aero
        self.control_variables = tuple(control_variables)
        self.independent_vars = tuple(
            build_independent_vars(unsteady_aero, control_variables)
        )
        # Infer the stored source and its dependencies from the raw input when
        # ``depends_on`` is not given. ``_resolve_input`` may also adopt the
        # input's extrapolation (re-keying an AeroCoefficient), so refresh the
        # local ``extrapolation`` used by the source-storage block below.
        if depends_on is None:
            source, depends_on = self._resolve_input(source, single_var)
            extrapolation = self.extrapolation
        # ``depends_on`` is kept in the given order because it matches the
        # positional argument order of the stored source (callable parameters,
        # CSV columns, …). ``_indices`` therefore maps the full argument tuple
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
            # Only override extrapolation when explicitly asked, and on a copy:
            # the source may be a user-owned Function reused elsewhere, so
            # mutating it in place (e.g. drag forcing "constant") would change
            # its behavior everywhere the caller reuses it.
            if extrapolation is not None:
                source = copy.deepcopy(source)
                source.set_extrapolation(extrapolation)
            self.function = source
        elif callable(source):
            self.function = Function(
                source,
                list(self.depends_on) or ["x"],
                [name],
                interpolation="linear",
                extrapolation=extrapolation or "natural",
            )
        else:
            # Scalar constant.
            self._constant = float(source)
            self.is_zero = self._constant == 0.0
            self.function = Function(self._constant)

        self._evaluate = self.function.get_value_opt

    def _resolve_input(self, source, single_var):
        """Infer ``(stored source, depends_on)`` from a raw coefficient input.

        Mirrors the coefficient inputs accepted by :class:`GenericSurface`: a
        number, a callable, a :class:`Function`, a list/tuple of data points, a
        path to a CSV (or other text) file, or another :class:`AeroCoefficient`
        (re-keyed).
        Called by :meth:`__init__` when ``depends_on`` is omitted; the returned
        ``source`` is a number, a callable or a :class:`Function`, which the
        constructor's source-storage block then stores.
        """
        name = self.name
        independent_vars = self.independent_vars
        n_vars = len(independent_vars)

        if isinstance(source, AeroCoefficient):
            # An already-built coefficient passed straight through, re-keyed to
            # this surface's variable order. This is how a *surface* round-trips:
            # GenericSurface/ControllableGenericSurface store their processed
            # AeroCoefficients in ``to_dict`` and feed them back on ``from_dict``
            # (and a user may reuse one coefficient across surfaces). Adopt its
            # extrapolation when none was requested.
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
                    extrapolation=self.extrapolation or "natural",
                    single_var=single_var,
                )
            # Any other path (e.g. a whitespace-delimited ``.txt`` curve) is read
            # by Function, which auto-detects the delimiter. Linear interpolation
            # matches the CSV loader, so the same data gives identical results
            # whatever file form it is given. Falls through to the Function
            # branch below (a 1-D table keyed to ``single_var``).
            source = Function(source, interpolation="linear")

        # A list/tuple of data points is parsed by Function and handled below.
        # Linear interpolation matches the CSV loader, so the same tabular data
        # gives identical results whether supplied as a list or a CSV file
        # (Function would otherwise default to spline).
        if isinstance(source, (list, tuple)):
            try:
                source = Function(list(source), interpolation="linear")
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
        file_path, name, independent_vars, extrapolation="natural", single_var=None
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
            ``"natural"``; drag coefficients pass ``"constant"``.
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
                interpolation="linear",
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
        )
        if csv_func is None:
            csv_func = Function(
                file_path,
                interpolation="linear",
                extrapolation=extrapolation,
            )

        # The CSV columns may appear in any order; AeroCoefficient maps the full
        # argument tuple to ``ordered_present_columns`` order, so the stored
        # Function is queried directly at its own (minimal) dimensionality.
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
        # label like "alpha_dot" binds to "alpha_dot" rather than the shorter
        # substring "alpha".
        for var in independent_vars:
            if var == label_lower:
                return var
        for var in sorted(independent_vars, key=len, reverse=True):
            if var in label_lower:
                return var
        return independent_vars[0]

    @staticmethod
    def _infer_callable_depends_on(func, independent_vars, name, single_var=None):
        """Infer ``depends_on`` for a plain callable.

        Conventions are accepted in order:

        0. *Single variable* - when ``single_var`` is given and the callable
           takes a single argument, it depends on that one variable regardless
           of the parameter name (e.g. a Mach-only drag ``lambda mach: ...``).
        1. *Named subset* - every parameter name is an independent variable, so
           the parameters themselves name the dependency subset (e.g.
           ``lambda alpha, mach: ...``).
        2. *Positional full-arity* - the parameter count equals the number of
           independent variables, so the callable depends on all of them
           regardless of how its parameters are named (e.g.
           ``lambda a, b, m, r, p, q, rr: ...``).
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
        """Back-compat alias used by the linear model's hot-loop term skipping."""
        return self.is_zero

    @property
    def __dom_dim__(self):
        """Number of full independent variables (the call arity)."""
        return len(self.independent_vars)

    def get_value_opt(self, *args):
        """Fast, unvalidated evaluation (mirrors :meth:`Function.get_value_opt`).

        Maps the full ``independent_vars`` argument tuple down to the source's
        own ``depends_on`` arguments before evaluating; a constant short-circuits.
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
            self.unsteady_aero,
            self.control_variables,
            self.name,
            extrapolation=self.extrapolation,
        )

    __rmul__ = __mul__

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        """Serialize the coefficient for :class:`rocketpy._encoders.RocketPyEncoder`."""
        return {
            "source": self._constant if self._constant is not None else self.function,
            "depends_on": list(self.depends_on),
            "unsteady_aero": self.unsteady_aero,
            "control_variables": list(self.control_variables),
            "name": self.name,
            "extrapolation": self.extrapolation,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild an :class:`AeroCoefficient` from its :meth:`to_dict` form."""
        return cls(
            data["source"],
            data["depends_on"],
            data.get("unsteady_aero", False),
            data.get("control_variables", ()),
            data["name"],
            extrapolation=data.get("extrapolation"),
        )

    def __repr__(self):
        """Return a concise representation showing the constant or dependencies."""
        if self._constant is not None:
            return f"AeroCoefficient({self.name}={self._constant})"
        return f"AeroCoefficient({self.name}, depends_on={self.depends_on})"
