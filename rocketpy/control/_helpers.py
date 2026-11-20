"""Helpers of the control package: the control inputs of a flight, and what
loading a saved controller or controllable needs.

During a flight the control inputs are a plain dictionary from name to value,
such as ``{"air_brakes.deployment_level": 0.0}``: one control input per
control of each controllable on the rocket. Events and controllers write it
as ``context.controls.air_brakes.deployment_level = 0.5``, and the equations
of motion read it as ``controls``. These functions build that dictionary
(:func:`initial_controls`), keep a value inside its limits
(:func:`limited_value`) and check what an event wrote to it
(:func:`check_writes`).

Below them: what loading a saved controller or controllable can raise, what
stands for one that could not be loaded, and the checks of the functions a
controller or an effector is built from. The deprecated positional controller
functions of v1.13 are in :mod:`rocketpy.control._legacy`.
"""

import inspect
import pickle
import warnings

from rocketpy.tools import snake_case_key


def limited_value(value, limits):
    """Return ``value`` as a float, moved to the nearest of ``limits`` when it
    is outside them.

    Parameters
    ----------
    value : float
        The value written to a control.
    limits : tuple of float or None
        ``(low, high)``, or ``None`` for a control without limits.

    Returns
    -------
    float
    """
    if limits is not None:
        low, high = limits
        value = min(max(value, low), high)
    return float(value)


def initial_controls(controllables):
    """Return what a flight needs to fly the controls of ``controllables``.

    Parameters
    ----------
    controllables : sequence of Controllable
        The rocket's controllables, in the order they were added.

    Returns
    -------
    controls : dict
        Name to value, every control input at its default: the flight's
        control inputs.
    names : dict
        Controllable to the tuple of its control input names, ``"<key>.<control>"``
        in the order of its ``controls``, for the controllables that have controls.
    limits : dict
        Control input name to its ``(low, high)``, or ``None`` for no limits,
        used to keep what is written inside them.

    Raises
    ------
    ValueError
        If two controllables have names that give the same key, since
        a controller could not tell their control inputs apart.
    """
    controls, names, limits = {}, {}, {}
    owners = {}
    for controllable in controllables:
        if not controllable.controls:
            continue
        key = snake_case_key(controllable.name)
        if key in owners:
            raise ValueError(
                f"The controllables '{owners[key]}' and '{controllable.name}' "
                f"both have controls, and their control inputs would both be named "
                f"'{key}.<control>'. Give them different names."
            )
        owners[key] = controllable.name
        names[controllable] = tuple(
            f"{key}.{control}" for control in controllable.controls
        )
        for control, name in zip(controllable.controls, names[controllable]):
            controls[name] = float(controllable.control_defaults[control])
            bounds = controllable.control_limits.get(control)
            limits[name] = None if bounds is None else tuple(float(b) for b in bounds)
    return controls, names, limits


def check_writes(controls, before, limits):
    """Check the writes an event made to the control inputs.

    Called right after an event's callback, with a copy of ``controls`` taken
    before it. A written value is kept inside its limits, and counts as a
    change only if it then differs from the value before.

    Parameters
    ----------
    controls : dict
        The flight's control inputs, as the callback left them.
    before : dict
        A copy of them taken before the callback.
    limits : dict
        Control input name to its ``(low, high)``, or ``None``.

    Returns
    -------
    dict
        The control inputs whose value changed, with their new value. Empty when
        nothing changed.

    Raises
    ------
    KeyError
        If the callback added or removed a control input, such as by writing a
        misspelled name. The inputs are put back as they were.
    """
    if controls.keys() != before.keys():
        added = sorted(controls.keys() - before.keys())
        controls.clear()
        controls.update(before)
        if added:
            raise KeyError(
                f"There is no control input {added[0]!r}. The control inputs are: "
                f"{', '.join(before) or 'none'}."
            )
        raise KeyError("Control inputs cannot be removed during a flight.")
    changed = {}
    for name, value in controls.items():
        if value != before[name]:
            # Compared again once inside the limits: a write beyond a limit
            # the control already sits at changes nothing
            controls[name] = value = limited_value(value, limits[name])
            if value != before[name]:
                changed[name] = value
    return changed


# ---------------------------------------------------------------------------
# Loading a controller or a controllable from a saved file
# ---------------------------------------------------------------------------


# What loading a saved function or controllable raises when it was saved without
# pickling (a name instead of the object), or its module, class or code is not
# available here
_LOAD_ERRORS = (
    TypeError,
    ValueError,
    AttributeError,
    ImportError,
    EOFError,
    KeyError,
    pickle.UnpicklingError,
)


class UnloadedControllable:
    """Stands for a controllable of a saved rocket that could not be loaded:
    one saved without pickling, or whose class is not available here. The
    rocket leaves it out with a warning.

    Attributes
    ----------
    name : str
        The name of the controllable, as saved.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"UnloadedControllable(name={self.name!r})"


def was_loaded(part, kind, needs):
    """Return whether ``part`` of a saved rocket was loaded, warning that it
    is left out of the rocket when it was not.

    Parameters
    ----------
    part : object
        What the file gave for the part: the part itself, or what stands for
        one that could not be loaded (a dict or an
        :class:`UnloadedControllable`).
    kind : str
        What the part is, for the warning: ``"effector"``.
    needs : str
        What of the part must be available here to load it, for the warning:
        ``"class"``.

    Returns
    -------
    bool
    """
    if not isinstance(part, (dict, UnloadedControllable)):
        return True
    # Saved without pickling, or by a class not available here
    name = part.get("name") if isinstance(part, dict) else part.name
    warnings.warn(
        f"The {kind} '{name}' could not be loaded and was left out of the "
        f"rocket: it was saved without pickling, or its {needs} is not "
        "available here.",
        UserWarning,
        stacklevel=2,
    )
    return False


def _missing_controller_function(context):  # pylint: disable=unused-argument
    """Stands in for a controller function that could not be loaded."""
    return None


# ---------------------------------------------------------------------------
# Functions given in place of a method: checked when the owner is built
# ---------------------------------------------------------------------------


def check_hook(owner, what, function, default, arguments, needed=True):
    """Return ``function`` after checking it takes ``arguments``, or ``None``
    when ``owner``'s class defines the method ``what`` itself. Raise when
    neither is given and the method is ``needed``.

    A :class:`ContinuousController` or an effector can be built from functions
    (``law=``, ``force_and_moment=``) or subclassed with the same names as
    methods. The method as the base class defines it calls the function.

    Parameters
    ----------
    owner : Controllable
        The controller or effector being built. Its ``name`` names it in errors.
    what : str
        The name of the method, such as ``"law"``.
    function : callable or None
        The function given in its place, if any.
    default : function
        The method as the base class defines it. A subclass that defines its
        own replaces it, and then needs no function.
    arguments : tuple of str
        The names of the arguments the function takes, in order, for the
        check and for the error message: ``("states", "controls",
        "conditions")``, or ``("firing", "conditions")`` for an effector
        with one control.
    needed : bool, optional
        Whether the method must be given at all. Default ``True``.

    Returns
    -------
    callable or None
        ``function``, or ``None`` when the owner's class defines the method.

    Raises
    ------
    TypeError
        If neither a function nor a method is given and one is ``needed``, if
        ``function`` is not callable, or if it does not take ``arguments``.
    """
    signature = f"{what}({', '.join(arguments)})"
    if function is None:
        if needed and getattr(type(owner), what) is default:
            base = default.__qualname__.partition(".")[0]
            raise TypeError(
                f"'{owner.name}' needs a {what}: give {base} a function `{signature}`."
            )
        return None
    if not callable(function):
        raise TypeError(f"The {what} must be callable, got {function!r}.")
    try:
        inspect.signature(function).bind(*[None] * len(arguments))
    except TypeError as error:
        raise TypeError(
            f"The {what} of '{owner.name}' must take {len(arguments)} arguments, "
            f"like `def {signature}: ...`."
        ) from error
    return function


def _check_controller_function(function):
    """Raise unless ``function`` takes the context as its one argument."""
    if not callable(function):
        raise TypeError(f"The controller function must be callable, got {function!r}.")
    try:
        inspect.signature(function).bind(None)
    except TypeError as error:
        raise ValueError(
            "The controller function must take the event context as its "
            "single argument, like `def controller(context): ...`."
        ) from error
    return function
