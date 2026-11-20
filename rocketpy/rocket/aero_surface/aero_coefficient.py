import copy
import csv
import inspect
import math
import re
import warnings

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.mathutils._regular_grid import _RegularGrid
from rocketpy.rocket.aero_surface._helpers import (
    _as_function,
    total_angle_and_roll,
)

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


# The total angle of attack (between the air and the rocket's axis, from 0 to pi)
# and the roll angle of the wind (the direction around the body the crossflow
# comes from). A source may name them as inputs.
DERIVED_ANGLES = ("alpha_total", "phi")

# Names under which an angle can be given in degrees. The simulation always
# works in radians; a coefficient whose source uses one of these names converts
# the angle before reading the source.
ANGLES_IN_DEGREES = {
    "alpha_deg": "alpha",
    "beta_deg": "beta",
    "alpha_total_deg": "alpha_total",
    "phi_deg": "phi",
}

# Names a source may use for its inputs on top of the variables themselves
SOURCE_ONLY_NAMES = (*DERIVED_ANGLES, *ANGLES_IN_DEGREES)


# Error for a one-input source whose variable cannot be told; filled with the
# name of the coefficient
_UNNAMED_INPUT_MESSAGE = (
    "Cannot tell which variable the single input of {name} is. Name it "
    "by giving the coefficient as a pair, for example "
    '(table, ["mach"]), by labelling the input of a Function after the '
    'variable (for example "mach" or "Mach Number"), by adding a header '
    "to the CSV file, or by naming the argument of a function after it "
    "(for example lambda mach: ...)."
)


def _required_arguments(func):
    """Names of the arguments ``func`` must be given by position: the ones with
    no default value. Optional arguments (``k=2.0``, keyword-only ones,
    ``*args``, ``**kwargs``) are for the function's own use and are left out."""
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return []
    return [
        parameter.name
        for parameter in parameters
        if parameter.default is parameter.empty
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
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

    # The names a Function accepts for its tables
    _INTERPOLATIONS = (
        "linear",
        "polynomial",
        "akima",
        "spline",
        "shepard",
        "rbf",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
    )
    _EXTRAPOLATIONS = ("constant", "natural", "zero")

    def __init__(  # pylint: disable=too-many-statements
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
        :class:`Function` of only the variables it depends on.

        Most of the time you only pass ``source`` and leave ``depends_on`` as
        ``None``, so the variables are worked out automatically. This is the
        same input a :class:`GenericSurface` accepts. Pass ``depends_on``
        yourself only when the source and the order of its inputs are already
        known.

        Parameters
        ----------
        source : int, float, str, list, tuple, array, callable, Function, or AeroCoefficient
            The coefficient value. It can depend on these variables, named the
            same way in every form below:

            - ``"alpha"``, ``"beta"``: angle of attack and sideslip angle, in
              radians, or ``"alpha_deg"``, ``"beta_deg"`` in degrees.
            - ``"alpha_total"``, ``"phi"``: total angle of attack (between the
              rocket's axis and the air) and roll angle of the wind (the
              direction the crossflow comes from), in radians, or
              ``"alpha_total_deg"``, ``"phi_deg"`` in degrees.
            - ``"mach"``: Mach number.
            - ``"reynolds"``: Reynolds number.
            - ``"pitch_rate"``, ``"yaw_rate"``, ``"roll_rate"``: angular rates in
              reduced form, such as ``q * L / (2 * V)`` for the pitch rate.
            - the names in ``control_variables``.

            It can be given as:

            - **number**: a constant.
            - **function or lambda**: arguments named after the variables, e.g.
              ``lambda alpha, mach: ...``, or seven arguments in the order
              ``alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate``.
              Arguments with a default value are not counted, so
              ``def cN(alpha, mach, k=2.0)`` depends on ``alpha`` and ``mach``.
              ``functools.partial`` also works.
            - **list, tuple or numpy array**: a table with one column per
              variable and the value in the last column. Name its columns with
              the ``(source, variables)`` pair below, unless it has one input
              (see ``single_var``) or all seven, in the order above.
            - **str**: path to a ``.csv`` file with one column per variable,
              named in the header, and the value in the last column. A
              headerless two-column file is ``single_var`` against the value.
            - **Function**: used as is; its input labels name the variables. If
              ``extrapolation`` is given, it is applied to a copy.
            - **AeroCoefficient**: reused as is.
            - **(axes, values)**: values on a regular grid. ``axes`` maps each
              variable to its values, and ``values`` has one dimension per
              variable, in the same order: ``({"alpha": alphas, "mach": machs},
              values)``, with ``values[i, j]`` at ``alphas[i]`` and ``machs[j]``.
            - **(source, variables)**: any of the above with the names of its
              inputs, in order, e.g. ``(points, ["alpha", "mach"])``. Use it
              when the source does not name its variables.

            A source with one input must name its variable, in one of the ways
            above or with ``single_var``: a table is never assumed to be against
            the angle of attack.
        depends_on : sequence of str, optional
            The names of the source's inputs, in the order the source takes
            them (a function's arguments, a CSV's columns). The names are
            ``"alpha"``, ``"beta"``, ``"alpha_deg"``, ``"beta_deg"``,
            ``"alpha_total"``, ``"phi"``, ``"alpha_total_deg"``, ``"phi_deg"``,
            ``"mach"``, ``"reynolds"``, ``"pitch_rate"``, ``"yaw_rate"``,
            ``"roll_rate"`` and the names in ``control_variables``. For example
            ``()`` for a constant or ``("alpha_total_deg", "mach")``. Any other
            name raises a ``ValueError``. Leave it as ``None`` (the default) to
            have it worked out from ``source``.
        control_variables : sequence of str, optional
            Names of extra variables, such as control-surface deflections set by
            a controller. They are added after the seven base variables, in the
            order given. Empty for ordinary surfaces. Default ``()``.
        name : str, optional
            A readable name for the coefficient (e.g. ``"cL_alpha"`` or
            ``"Drag Coefficient with Power Off"``). It appears in error messages
            and plots. Default ``"coefficient"``.
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
            function. A source that names its own variable (an argument or an
            input label such as ``alpha`` or ``"Alpha (deg)"``) keeps that
            name; ``single_var`` only fills in for a source that does not.
            ``None`` (the default) reads it from the input's label
            and raises a ``ValueError`` when the label names no variable.
            Ignored when ``depends_on`` is given. Default ``None``.
        """
        self.name = name
        for what, option, allowed in (
            ("interpolation", interpolation, self._INTERPOLATIONS),
            ("extrapolation", extrapolation, self._EXTRAPOLATIONS),
        ):
            if option is not None and option not in allowed:
                raise ValueError(
                    f"Unknown {what} {option!r} for {name}; the options are "
                    f"{', '.join(allowed)}."
                )
        self.extrapolation = extrapolation
        self.interpolation = interpolation
        self.control_variables = tuple(control_variables)
        # Every variable, in the order the coefficient is called with
        self.independent_vars = tuple(build_independent_vars(control_variables))
        if depends_on is None:
            source, depends_on = self._resolve_pair(source) or self._resolve_input(
                source, single_var
            )
            extrapolation = self.extrapolation
            interpolation = self.interpolation
        # The source's inputs as given (possibly in degrees or total angles),
        # kept in its own order to save and rebuild the coefficient
        self._source_variables = tuple(depends_on)
        bases = [ANGLES_IN_DEGREES.get(var, var) for var in self._source_variables]
        # The source's inputs with the degree names read as the angle they
        # stand for: ``alpha_deg`` as ``alpha``, ``alpha_total_deg`` as
        # ``alpha_total``
        self.source_angles = tuple(bases)
        self.in_degrees = tuple(
            ANGLES_IN_DEGREES[var]
            for var in self._source_variables
            if var in ANGLES_IN_DEGREES
        )
        # The derived angles come from alpha and beta; each variable is kept once
        self.depends_on = tuple(
            dict.fromkeys(
                var
                for base in bases
                for var in (("alpha", "beta") if base in DERIVED_ANGLES else (base,))
            )
        )
        if len(set(bases)) != len(bases):
            raise ValueError(
                f"{name} names the same variable more than once: "
                f"{list(self._source_variables)}."
            )
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
            # Changed only when asked, and on a copy: the Function may be shared
            if interpolation is not None or extrapolation is not None:
                source = copy.deepcopy(source)
                # Scattered points would read "akima" or "spline" as "shepard"
                if interpolation is not None and (
                    source.__dom_dim__ == 1 or source.is_regular_grid
                ):
                    source.set_interpolation(interpolation)
                if extrapolation is not None:
                    source.set_extrapolation(extrapolation)
            self.function = source
        elif callable(source):
            if len(inspect.signature(source).parameters) != len(self._source_variables):
                # Hide the optional arguments, which Function would count as inputs
                self.function = _as_function(
                    lambda *args, function=source: function(*args),
                    self._source_variables,
                    name,
                )
            else:
                self.function = Function(
                    source, list(self._source_variables) or ["x"], [name]
                )
        else:
            # Scalar constant.
            self._constant = float(source)
            self.is_zero = self._constant == 0.0
            self.function = Function(self._constant)

        self._evaluate = self.function.get_value_opt
        if self._source_variables != self.depends_on:
            self._evaluate = self._converting_evaluator()
        self._check_angle_axes()
        if callable(source) and not isinstance(source, Function):
            # Call it once, so a function that cannot be called or does not
            # return a number fails here rather than during the flight
            try:
                float(self._evaluate(*[0.0] * len(self.depends_on)))
            except Exception as exc:
                raise ValueError(
                    f"The function given for {self.name} could not be evaluated "
                    f"with every variable at zero: {exc!r}. It must accept the "
                    f"variables {list(self.depends_on)} as positional arguments "
                    "and return a number."
                ) from exc

    def _converting_evaluator(self):
        """Evaluator for a source whose inputs are not the variables as they
        are: angles in degrees, the total angle of attack or the roll angle of
        the wind. It takes the values of ``depends_on`` and builds the source's
        own inputs from them."""
        position = {var: i for i, var in enumerate(self.depends_on)}
        source_evaluate = self.function.get_value_opt

        def reader(variable):
            base = ANGLES_IN_DEGREES.get(variable, variable)
            scale = 180 / math.pi if variable in ANGLES_IN_DEGREES else 1.0
            if base in DERIVED_ANGLES:
                i_alpha, i_beta = position["alpha"], position["beta"]
                which = DERIVED_ANGLES.index(base)
                return lambda args: (
                    total_angle_and_roll(args[i_alpha], args[i_beta])[which] * scale
                )
            index = position[base]
            return lambda args: args[index] * scale

        readers = [reader(variable) for variable in self._source_variables]
        return lambda *args: source_evaluate(*[read(args) for read in readers])

    def _check_angle_axes(self):
        """Warn about two mistakes a table's angle of attack or sideslip axis
        can show.

        - Values no angle in radians can take: both angles stay within ``-pi``
          to ``pi``, so a larger value almost surely means the table is in
          degrees.
        - A table that starts at zero angle, where the coefficient is zero, and
          has no negative angles. Such a coefficient changes sign with the angle
          (a normal force or a pitch moment), but outside its range a table
          holds its edge value, so it would stay at zero for every negative
          angle. Data tabulated against the total angle of attack looks like
          this.
        """
        domain = getattr(self.function, "_domain", None)
        image = getattr(self.function, "_image", None)
        if self._constant is not None or domain is None or image is None:
            return
        if len(np.unique(domain, axis=0)) != len(domain):
            raise ValueError(
                f"The table of {self.name} lists the same input values more "
                "than once. Each point must appear a single time."
            )
        for column, variable in enumerate(self._source_variables):
            angle = ANGLES_IN_DEGREES.get(variable, variable)
            if angle not in ("alpha", "beta", *DERIVED_ANGLES):
                continue
            if column >= domain.shape[1]:
                continue
            axis = domain[:, column]
            if variable == angle and float(np.max(np.abs(axis))) > 3.2:
                warnings.warn(
                    f"The {variable} values of {self.name} go up to "
                    f"{float(np.max(np.abs(axis))):g}, which looks like degrees, "
                    "but they are read as radians. If the table is in degrees, "
                    f'name the variable "{variable}_deg" (in the CSV header, the '
                    "Function input or the list of variables given with the "
                    "table).",
                    UserWarning,
                    stacklevel=4,
                )
            at_zero = np.abs(np.ravel(image)[axis == 0])
            if (
                angle in ("alpha", "beta")
                and np.min(axis) == 0
                and at_zero.size
                and np.all(at_zero < 1e-12)
                and np.any(np.abs(image) > 1e-12)
            ):
                warnings.warn(
                    f"The table of {self.name} starts at {angle} = 0, where it is "
                    "zero, and has no negative angles. Outside its range a table "
                    f"holds its edge value, so {self.name} would be zero for every "
                    f"negative {angle}. RocketPy's {angle} is measured in one "
                    "plane and takes both signs: mirror the table to negative "
                    "angles (with the sign of the coefficient flipped). If the "
                    "data is against the total angle of attack, name the "
                    f'variable "alpha_total" instead of "{angle}".',
                    UserWarning,
                    stacklevel=4,
                )

    def _as_table(self, source):
        """Turn a file path, a list of data points or an array into a
        :class:`Function`; anything else is returned unchanged. Points that
        cover a regular grid over two or more variables are interpolated on it
        by the ``Function``, which is more accurate than treating them as
        scattered."""
        if not isinstance(source, (str, list, tuple, np.ndarray)):
            return source
        try:
            return Function(
                source if isinstance(source, str) else np.asarray(source).tolist(),
                interpolation=self.interpolation or "linear",
                extrapolation=self.extrapolation or "constant",
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Invalid input for {self.name}: could not be read as a table "
                "of data points."
            ) from exc

    def _resolve_pair(self, source):
        """Handle a source given as a pair. ``(table, ["alpha", "mach"])`` names
        the variables of a table or function, and
        ``({"alpha": alphas, "mach": machs}, values)`` gives values on a regular
        grid, with ``values[i, j]`` the value at ``alphas[i]`` and ``machs[j]``.
        Returns ``(stored source, depends_on)``, or ``None`` when ``source`` is
        not such a pair."""
        if not (isinstance(source, tuple) and len(source) == 2):
            return None
        source, variables = source
        if isinstance(source, dict) and all(isinstance(key, str) for key in source):
            shape = tuple(len(axis) for axis in source.values())
            if np.shape(variables) != shape:
                raise ValueError(
                    f"The values of {self.name} have shape {np.shape(variables)} "
                    f"but its axes {list(source)} have {shape} points: values "
                    "must have one dimension per axis, in the order of the axes "
                    "(a square grid given the other way round cannot be told "
                    "apart, so check the order)."
                )
            source, variables = (
                _RegularGrid.points_from_axes(list(source.values()), variables),
                list(source),
            )
        elif not (
            isinstance(variables, (list, tuple))
            and all(isinstance(variable, str) for variable in variables)
        ):
            return None
        if len(set(variables)) != len(variables):
            raise ValueError(
                f"{self.name} names the same variable more than once: "
                f"{list(variables)}."
            )
        source = self._as_table(source)
        if isinstance(source, Function):
            number_of_inputs = source.__dom_dim__
        elif callable(source):
            number_of_inputs = len(_required_arguments(source))
        elif variables:
            raise ValueError(
                f"{self.name} is a constant ({source}) but the variable names "
                f"{list(variables)} were given: a constant depends on nothing. "
                "Give the number alone, or a table or function of those variables."
            )
        else:
            number_of_inputs = 0
        if number_of_inputs != len(variables):
            raise ValueError(
                f"{self.name} has {number_of_inputs} input(s) but "
                f"{len(variables)} variable name(s) were given: {list(variables)}."
            )
        return source, tuple(variables)

    def _resolve_input(self, source, single_var):
        """Work out ``(stored source, depends_on)`` from a raw input: a number,
        a file path, a list of points, a function, a :class:`Function` or another
        ``AeroCoefficient``. ``single_var`` names the variable of a one-input
        source; when ``None`` it is read from the source's own labels."""
        name = self.name
        n_vars = len(self.independent_vars)
        # Every name a source may give its inputs
        independent_vars = [*self.independent_vars, *SOURCE_ONLY_NAMES]

        if isinstance(source, AeroCoefficient):
            # Reused as is, keeping its extrapolation unless another was asked
            if self.extrapolation is None:
                self.extrapolation = source.extrapolation
            value = (
                source._constant if source._constant is not None else source.function
            )
            return value, source._source_variables

        if isinstance(source, str) and source.lower().endswith(".csv"):
            return self._load_csv(
                source,
                name,
                independent_vars,
                extrapolation=self.extrapolation or "constant",
                interpolation=self.interpolation or "linear",
                single_var=single_var,
            )
        # Any other path, or a list of data points, is read as a table
        source = self._as_table(source)

        if isinstance(source, Function):
            dom_dim = source.__dom_dim__
            # Inputs named exactly after the variables say what it depends on
            inputs = [str(label) for label in source.__inputs__]
            if set(inputs) <= set(independent_vars) and len(set(inputs)) == dom_dim:
                return source, inputs
            if dom_dim == n_vars:
                return source, list(self.independent_vars)
            if dom_dim == 1:
                # No default: guessing could read a Mach curve against alpha
                variable = (
                    self._infer_single_var(source, independent_vars) or single_var
                )
                if variable is None:
                    raise ValueError(_UNNAMED_INPUT_MESSAGE.format(name=name))
                return source, [variable]
            raise ValueError(
                f"{name} Function must have {n_vars} input arguments "
                f"({', '.join(self.independent_vars)}) or be one-dimensional. To use "
                f"a table with {dom_dim} inputs, name them by giving the "
                'coefficient as a pair, for example (table, ["alpha", "mach"]).'
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
    ):
        """Read a coefficient from a CSV file. The last column is the value of
        the coefficient; the header names the variable of each of the other
        columns, among ``independent_vars``. A two-column file with no header
        is a table over ``single_var``, which must then be given. Returns
        ``(function, depends_on)``, with ``depends_on`` the columns in the order
        of the file."""
        try:
            with open(file_path, mode="r") as file:
                header = [
                    column.strip().strip("\"'")
                    for column in next(csv.reader(file, skipinitialspace=True))
                ]
        except OSError as e:
            raise ValueError(f"Error reading {name} CSV file: {e}") from e
        except StopIteration as e:
            raise ValueError(f"Invalid or empty CSV file for {name}.") from e
        if not header:
            raise ValueError(f"Invalid or empty CSV file for {name}.")

        def is_number(text):
            try:
                float(text)
                return True
            except ValueError:
                return False

        if len(header) == 2 and all(is_number(cell) for cell in header):
            # No header: a table of one variable, which has to be told
            if single_var is None:
                raise ValueError(_UNNAMED_INPUT_MESSAGE.format(name=name))
            variables = [single_var]
        else:
            variables = header[:-1]
            invalid_columns = [col for col in variables if col not in independent_vars]
            if invalid_columns:
                raise ValueError(
                    f"Invalid independent variable(s) in {name} CSV: "
                    f"{invalid_columns}. Valid options are: {list(independent_vars)}."
                )
            if header[-1] in independent_vars:
                raise ValueError(
                    f"Last column in {name} CSV must be the coefficient"
                    " value, not an independent variable."
                )
            if not variables:
                raise ValueError(f"No independent variables found in {name} CSV.")
            if len(set(variables)) != len(variables):
                raise ValueError(
                    f"{name} names the same variable more than once: {variables}."
                )

        # A table covering a regular grid is interpolated on it by Function
        function = Function(
            file_path, interpolation=interpolation, extrapolation=extrapolation
        )
        return function, variables

    @staticmethod
    def _infer_single_var(function, independent_vars):
        """The variable a one-input Function is labelled after, or ``None``.

        The label matches a variable when it contains the variable's name as a
        whole word, ignoring case and reading spaces as underscores: ``"mach"``,
        ``"Mach Number"`` and ``"Pitch Rate"`` match, ``"Alphabet"`` does not.
        An angle whose label also says ``deg`` or ``°`` (``"Alpha (deg)"``) is
        read in degrees.
        """
        try:
            label = str(function.__inputs__[0]).lower()
        except (AttributeError, IndexError, TypeError):
            return None
        # Longest name first, so "alpha_total" is not read as "alpha"
        for var in sorted(independent_vars, key=len, reverse=True):
            pattern = rf"(?<![a-z0-9_]){re.escape(var)}(?![a-z0-9_])"
            if re.search(pattern, label) or re.search(pattern, label.replace(" ", "_")):
                in_degrees = f"{var}_deg"
                if in_degrees in independent_vars and re.search(r"\bdeg|°", label):
                    return in_degrees
                return var
        return None

    @staticmethod
    def _infer_callable_depends_on(func, independent_vars, name, single_var=None):
        """Work out which variables a function coefficient uses, from its
        arguments.

        Three ways to write the function are accepted, tried in this order:

        1. Arguments named after variables: every argument name matches one of
           the surface's variables, so the names themselves list what the
           function uses (e.g. ``lambda alpha, mach: ...`` uses ``alpha`` and
           ``mach``). Naming only some of the arguments after variables is an
           error, as is giving a default value to an argument named after one
           (``mach=0.0`` would hold Mach at 0 for the whole flight).
        2. One argument plus ``single_var``: the function takes a single
           argument that is not named after a variable and ``single_var`` says
           which variable it is (e.g. a Mach-only drag curve ``lambda m: ...``
           with ``single_var="mach"``).
        3. One argument per variable: the function has exactly as many arguments
           as there are variables, so it is taken to use all of them, whatever
           the arguments are named (e.g. ``lambda a, b, m, r, p, q, rr: ...``
           for the seven base variables).

        Anything else raises ``ValueError``.
        """
        # Only the variables themselves, without their other names
        variables = [var for var in independent_vars if var not in SOURCE_ONLY_NAMES]
        n_vars = len(variables)
        names = _required_arguments(func)
        defaulted = [
            parameter
            for parameter in inspect.signature(func).parameters
            if parameter not in names and parameter in independent_vars
        ]
        if defaulted:
            raise ValueError(
                f"The function given for {name} gives a default value to "
                f"{defaulted}, which would hold that variable at the default "
                "for the whole flight. Remove the default (an argument named "
                "after a variable is read from the flow)."
            )
        matching = [argument for argument in names if argument in independent_vars]
        if matching and len(matching) == len(names):
            return names
        # One argument per variable, any named one at its own position
        if len(names) == n_vars and all(
            argument not in matching or argument == variable
            for argument, variable in zip(names, variables)
        ):
            return variables
        if matching:
            raise ValueError(
                f"The function given for {name} names {matching} after "
                f"variables but not {[n for n in names if n not in matching]}. "
                "Name every argument after a variable, or none of them (the "
                f"variables are {', '.join(variables)})."
            )
        if single_var and len(names) == 1:
            return [single_var]
        raise ValueError(
            f"Cannot tell which variables the function given for {name} uses. "
            "Name its arguments after the variables it uses, for example "
            f"lambda alpha, mach: ... (the variables are {', '.join(variables)}), "
            f"or give it all {n_vars} of them as arguments, in that order, or "
            "name them by giving the coefficient as a pair, for example "
            '(function, ["alpha", "mach"]). Arguments with a default value are '
            "not counted."
        )

    def get_value_opt(self, *args):
        """Fast evaluation without input checking (mirrors
        :meth:`Function.get_value_opt`).

        Receives every variable, passes on only the ones this coefficient uses,
        and evaluates the source. A constant is returned right away.
        """
        if self._constant is not None:
            return self._constant
        return self._evaluate(*(args[i] for i in self._indices))

    def evaluator(self, variables):
        """Return this coefficient as a plain function of ``variables`` only, in
        that order. ``variables`` must include every variable the coefficient
        depends on. Used to combine coefficients into one that is itself stored
        over only the variables it needs.
        """
        if self._constant is not None:
            constant = self._constant
            return lambda *args: constant
        positions = [variables.index(var) for var in self.depends_on]
        evaluate = self._evaluate
        return lambda *args: evaluate(*[args[i] for i in positions])

    # Calling the coefficient is the same as the fast evaluator
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
            self._source_variables,
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

    def _arguments(self, names, at):
        """Check ``names`` and the keys of ``at`` against the variables, and
        return the full argument list (``at`` values, else 0) with the position
        of each of ``names`` in it."""
        fixed = dict(at or {})
        unknown = [var for var in (*names, *fixed) if var not in self.independent_vars]
        if unknown:
            raise ValueError(
                f"{self.name} has no independent variable(s) {unknown}; valid "
                f"variables are {list(self.independent_vars)}."
            )
        if len(set(names)) != len(names):
            raise ValueError(f"{list(names)} names a variable more than once.")
        baseline = [fixed.get(var, 0.0) for var in self.independent_vars]
        return baseline, [self.independent_vars.index(var) for var in names]

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
            (for example ``"mach"`` or ``"alpha", "mach"``). The names are
            ``"alpha"``, ``"beta"``, ``"mach"``, ``"reynolds"``,
            ``"pitch_rate"``, ``"yaw_rate"``, ``"roll_rate"`` and the names in
            ``control_variables``, with the angles in radians.
        at : dict, optional
            Values to hold the other variables at, keyed by the same names. Any
            not listed are held at 0. A value given for a free variable is
            ignored.

        Returns
        -------
        Function
            A Function of ``free_variables`` that evaluates this coefficient with
            the remaining variables held fixed.
        """
        baseline, positions = self._arguments(free_variables, at)
        evaluate = self.get_value_opt
        if not free_variables:
            return Function(evaluate(*baseline))

        def sliced(*values):
            args = list(baseline)
            for position, value in zip(positions, values):
                args[position] = value
            return evaluate(*args)

        return _as_function(sliced, free_variables, self.name)

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
            ``"alpha"`` or ``"beta"``), from the names accepted by
            :meth:`slice`. It must not also appear in ``free_variables``.
        *free_variables : str
            Names of the variables to keep as inputs of the resulting slope, in
            the order you want them (for example ``"mach"``), from the names
            accepted by :meth:`slice`. Leave empty to get the slope at a single
            point.
        at : dict, optional
            Values to hold the other variables at, keyed by the same names. The
            value for ``variable`` is the point the derivative is taken at
            (default 0, the linearization point). Any variable not listed is held
            at 0, and a value given for a free variable is ignored.
        dx : float, optional
            Step size of the central difference used to take the derivative.
            Default 1e-6.

        Returns
        -------
        Function
            A Function of ``free_variables`` giving ``d(self)/d(variable)`` with
            the remaining variables held fixed.
        """
        if variable in free_variables:
            raise ValueError(
                f"{variable!r} cannot be both differentiated and kept free."
            )
        baseline, (index, *positions) = self._arguments((variable, *free_variables), at)
        point = baseline[index]
        evaluate = self.get_value_opt
        name = f"d({self.name})/d({variable})"

        def evaluate_slope(*values):
            args = list(baseline)
            for position, value in zip(positions, values):
                args[position] = value
            args[index] = point + dx
            upper = evaluate(*args)
            args[index] = point - dx
            return (upper - evaluate(*args)) / (2 * dx)

        if not free_variables:
            return Function(evaluate_slope(), name)

        return _as_function(evaluate_slope, free_variables, name)

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        """Serialize the coefficient for :class:`rocketpy._encoders.RocketPyEncoder`."""
        return {
            "source": self._constant if self._constant is not None else self.function,
            "depends_on": list(self._source_variables),
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
