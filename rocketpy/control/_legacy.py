"""The deprecated positional air brakes controller functions of v1.13,
``(time, sampling_rate, state, state_history, observed_variables,
air_brakes[, sensors[, environment]])``, and ``initial_observed_variables``:
run as controller functions that take the event context.

Remove this module, and its callers, in v1.16.
"""

import inspect
import warnings

from rocketpy.control._helpers import limited_value
from rocketpy.tools import snake_case_key


def _positional_count(function):
    """Return how many positional arguments ``function`` takes, or -1 if it
    takes any number."""
    parameters = inspect.signature(function).parameters.values()
    if any(p.kind == p.VAR_POSITIONAL for p in parameters):
        return -1
    return sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
    )


def _legacy_controller(
    function, sampling_rate, part_name=None, initial_observed_variables=None
):
    """Return the function of a :class:`Controller`: ``function`` itself when
    it takes the context, or a positional one wrapped in
    :class:`_PositionalController`, with ``initial_observed_variables`` as
    the start of its ``observed_variables``. Warn about either."""
    if initial_observed_variables is not None:
        warnings.warn(
            "`initial_observed_variables` is deprecated: record the values you "
            "want to see after the flight with `context.record(...)` and read "
            "them as `flight.records`. It will be removed in v1.16.",
            DeprecationWarning,
            stacklevel=3,
        )
    try:
        # Callable with the context alone, whatever else it accepts
        inspect.signature(function).bind(None)
        return function
    except TypeError:
        pass
    count = _positional_count(function)
    if count not in (-1, 6, 7, 8):
        raise ValueError(
            "controller_function must take the event context as its single "
            "argument, like `def controller(context): ...`. A deprecated "
            "positional controller_function must have 6, 7 or 8 arguments, (time, "
            "sampling_rate, state, state_history, observed_variables, "
            f"air_brakes[, sensors[, environment]]), but {count} were given."
        )
    warnings.warn(
        "Calling controller_function with positional arguments is deprecated; "
        "use controller_function(context) instead. It will be removed in v1.16.",
        DeprecationWarning,
        stacklevel=3,
    )
    return _PositionalController(
        function, sampling_rate, part_name, initial_observed_variables
    )


class _PositionalController:
    """Run a positional controller function as one that takes the context.
    ``state_history`` is the flight's rows so far, and ``observed_variables``
    gains each value returned other than ``None``.

    The list is kept here, not in the controller's memory: memory is copied
    before every run so a rollback can put it back, and a history that grows
    with every sample would make each copy longer than the last. As in
    v1.13, a rollback does not shorten it. It starts again at every flight
    (:meth:`reset`, called by ``Controller.reset``)."""

    def __init__(
        self, function, sampling_rate, part_name=None, initial_observed_variables=None
    ):
        self.function = function
        self.sampling_rate = sampling_rate
        # None: the rocket's only controllable with controls
        self.part_key = None if part_name is None else snake_case_key(part_name)
        self.__name__ = getattr(function, "__name__", "controller_function")
        self._count = _positional_count(function)
        self._initial_observed = (
            [] if initial_observed_variables is None else [initial_observed_variables]
        )
        self.observed_variables = list(self._initial_observed)

    def reset(self):
        """Start ``observed_variables`` again, for a new flight."""
        self.observed_variables = list(self._initial_observed)

    def __getstate__(self):
        # Saved as built: a loaded controller starts with an empty history
        return {**self.__dict__, "observed_variables": list(self._initial_observed)}

    def __call__(self, context):
        controllables = [
            controllable
            for controllable in context["rocket"].controllables
            if self.part_key in (None, snake_case_key(controllable.name))
        ]
        if len(controllables) != 1:
            raise ValueError(
                f"The controller function '{self.__name__}' moves one "
                f"controllable, but the rocket has {len(controllables)} that it "
                "could be. Write it as `controller_function(context)` and set "
                "`context.controls[...]`."
            )
        observed = self.observed_variables
        arguments = [
            context["time"],
            self.sampling_rate,
            context["canonical_state"],
            context._solution._rows,  # pylint: disable=protected-access
            observed,
            _AirBrakesView(controllables[0], context["controls"]),
            [sensor for sensor, _ in context["sensors"].values()],
            context["environment"],
        ]
        result = self.function(*arguments[: self._count if self._count >= 0 else None])
        if result is not None:
            observed.append(result)
        return result


class _AirBrakesView:
    """The air brakes as a positional controller sees them: ``deployment_level``
    is the flight's control input, kept inside its limits; anything else is
    read from the air brakes."""

    def __init__(self, air_brakes, controls):
        self._air_brakes = air_brakes
        self._controls = controls
        self._name = f"{snake_case_key(air_brakes.name)}.deployment_level"

    @property
    def deployment_level(self):
        return self._controls[self._name]

    @deployment_level.setter
    def deployment_level(self, value):
        self._controls[self._name] = limited_value(
            value, self._air_brakes.control_limits["deployment_level"], self._name
        )

    def __getattr__(self, name):
        return getattr(self._air_brakes, name)
