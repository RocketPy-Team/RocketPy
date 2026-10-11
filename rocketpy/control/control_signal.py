"""The control inputs of a flight.

During a flight the control inputs are a plain dictionary, ``u``, from channel
name to value, such as ``{"air_brakes.deployment_level": 0.0}``: one channel
per control of each actuator on the rocket. Events and controllers write it as
``context.controls["air_brakes.deployment_level"] = 0.5``, and the equations of
motion read it. This module builds that dictionary and checks what is written
to it.
"""

import keyword
import re
import warnings
from typing import Protocol, runtime_checkable


@runtime_checkable
class Actuator(Protocol):
    """Anything on a rocket that a controller can move.

    An actuator declares its controls and the physics that reads them. It never
    holds the live value of a control: during a flight the values live in the
    flight's control inputs, and the physics receives them as arguments. An
    aerodynamic surface is an actuator: its controls are extra inputs to its
    coefficients.

    Attributes
    ----------
    name : str
        Name of the actuator. Its channels are named after it, see
        :func:`control_key`.
    controls : tuple of str
        Names of the controls, in the order the physics expects their values.
        Empty for an actuator with nothing to control.
    control_defaults : dict
        Value of each control at the start of every flight, and the value used
        outside a flight.
    control_limits : dict
        ``(low, high)`` for each control that has limits, or ``None``.
    clamp : bool
        What a write outside the limits does: ``True`` moves the value to the
        nearest limit, ``False`` keeps it and warns.
    """

    name: str
    controls: tuple
    control_defaults: dict
    control_limits: dict
    clamp: bool


def check_actuator(actuator):
    """Check that ``actuator`` can be flown, and raise a clear error if not.

    Called when an actuator is added to a rocket, so a wrong object fails there
    and not in the middle of a flight.

    Parameters
    ----------
    actuator : Actuator
        The object to check.

    Raises
    ------
    TypeError
        If it lacks one of the attributes of :class:`Actuator`, or a default is
        not a number.
    ValueError
        If a control name is not a valid name, is repeated, has no default, or
        has limits that are not ``(low, high)`` with ``low <= high``.
    """
    if not isinstance(actuator, Actuator):
        missing = [
            name
            for name in ("name", "controls", "control_defaults", "control_limits")
            + ("clamp",)
            if not hasattr(actuator, name)
        ]
        raise TypeError(
            f"{actuator!r} cannot be added to the rocket as an actuator: it has "
            f"no {', '.join(missing)}. See rocketpy.Actuator for what an "
            "actuator must define."
        )
    name = actuator.name
    controls = tuple(actuator.controls)
    if len(set(controls)) != len(controls):
        raise ValueError(f"'{name}' lists a control more than once: {controls}.")
    for control in controls:
        _check_control_name(control, name)
        if control not in actuator.control_defaults:
            raise ValueError(f"The control '{control}' of '{name}' has no default.")
        try:
            float(actuator.control_defaults[control])
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"The default of the control '{control}' of '{name}' must be a "
                f"number, got {actuator.control_defaults[control]!r}."
            ) from error
        _check_limits(actuator.control_limits.get(control), control, name)
    for setting in ("control_defaults", "control_limits"):
        unknown = set(getattr(actuator, setting)) - set(controls)
        if unknown:
            raise ValueError(
                f"'{name}' has {setting} for {sorted(unknown)}, which are not "
                f"among its controls {controls}."
            )


def _check_control_name(control, owner):
    """Raise if ``control`` is not a plain name."""
    if (
        not isinstance(control, str)
        or not control.isidentifier()
        or keyword.iskeyword(control)
        or control.startswith("_")
    ):
        raise ValueError(
            f"'{owner}' has a control named {control!r}. A control name must "
            "be made of letters, digits and underscores, must not start with a "
            "digit or an underscore, and must not be a Python keyword."
        )


def _check_limits(limits, control, owner):
    """Raise if ``limits`` is neither ``None`` nor ``(low, high)``."""
    if limits is None:
        return
    try:
        low, high = (float(value) for value in limits)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"The limits of the control '{control}' of '{owner}' must be "
            f"(low, high) or None, got {limits!r}."
        ) from error
    if low > high:
        raise ValueError(
            f"The limits of the control '{control}' of '{owner}' are "
            f"({low}, {high}): the first must not be larger than the second."
        )


def limited_value(value, limits, clamp, channel):
    """Return ``value`` as a float, kept inside ``limits`` or warned about.

    Parameters
    ----------
    value : float
        The value written to a control.
    limits : tuple of float or None
        ``(low, high)``, or ``None`` for a control without limits.
    clamp : bool
        ``True`` moves a value outside the limits to the nearest limit.
        ``False`` keeps it and warns.
    channel : str
        Name of the control, for the warning.

    Returns
    -------
    float
    """
    if limits is not None:
        low, high = limits
        if value < low or value > high:
            if clamp:
                value = low if value < low else high
            else:
                warnings.warn(
                    f"{channel} = {value} is outside its limits [{low}, {high}]. "
                    "The value is used as it is, so anything that depends on "
                    "it is extrapolated.",
                    UserWarning,
                )
    return float(value)


# Name boundaries in a CamelCase name: "AirBrakes" -> "Air_Brakes",
# "TVCGimbal" -> "TVC_Gimbal"
_LOWER_TO_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_END = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NOT_ALPHANUMERIC = re.compile(r"[^0-9a-zA-Z]+")


def control_key(name):
    """Return the name an actuator's channels start with: its name in
    snake_case. ``"Canards"`` gives ``canards``, ``"AirBrakes"`` gives
    ``air_brakes`` and ``"calisto_nose_cone"`` stays as it is.

    Parameters
    ----------
    name : str
        The actuator's name.

    Returns
    -------
    str
    """
    text = _ACRONYM_END.sub("_", _LOWER_TO_UPPER.sub("_", str(name)))
    key = _NOT_ALPHANUMERIC.sub("_", text).strip("_").lower()
    if not key:
        key = "actuator"
    if key[0].isdigit():
        key = f"actuator_{key}"
    return key


def control_channels(actuators):
    """Return the channel names of each actuator that has controls.

    Parameters
    ----------
    actuators : sequence of Actuator
        The rocket's actuators, in the order they were added.

    Returns
    -------
    dict
        Actuator to the tuple of its channel names, ``"<key>.<control>"`` in
        the order of its ``controls``. Actuators without controls are left out.

    Raises
    ------
    ValueError
        If two actuators with controls have names that give the same key, since
        a controller could not tell their channels apart.
    """
    channels = {}
    owners = {}
    for actuator in actuators:
        if not actuator.controls:
            continue
        key = control_key(actuator.name)
        if key in owners:
            raise ValueError(
                f"The actuators '{owners[key]}' and '{actuator.name}' both have "
                f"controls, and their channels would both be named "
                f"'{key}.<control>'. Give them different names."
            )
        owners[key] = actuator.name
        channels[actuator] = tuple(f"{key}.{name}" for name in actuator.controls)
    return channels


def initial_controls(actuators):
    """Return what a flight needs to fly the controls of ``actuators``.

    Parameters
    ----------
    actuators : sequence of Actuator
        The rocket's actuators.

    Returns
    -------
    controls : dict
        Channel name to value, every channel at its default: the flight's
        control inputs, ``u``.
    channels : dict
        Actuator to the tuple of its channel names, for the actuators that have
        controls.
    limits : dict
        Channel name to ``(limits, clamp)``, used to check what is written.
    """
    channels = control_channels(actuators)
    controls, limits = {}, {}
    for actuator, names in channels.items():
        for control, name in zip(actuator.controls, names):
            controls[name] = float(actuator.control_defaults[control])
            bounds = actuator.control_limits.get(control)
            limits[name] = (
                None if bounds is None else tuple(float(b) for b in bounds),
                bool(actuator.clamp),
            )
    return controls, channels, limits


def history_function(points, name):
    """Return a sampled history as a Function of time that holds each value
    until the next sample.

    Parameters
    ----------
    points : sequence of (float, float)
        ``(time, value)`` pairs, in time order. A later pair at the same time
        replaces an earlier one, since it is what held from that time on.
    name : str
        Name of the value, for the Function's output.

    Returns
    -------
    Function
    """
    from rocketpy.mathutils.function import (  # pylint: disable=import-outside-toplevel
        Function,
    )

    held = {}
    for time, value in points:
        held[float(time)] = float(value)
    return Function(
        [[time, value] for time, value in held.items()],
        inputs="Time (s)",
        outputs=name,
        interpolation="previous",
        extrapolation="constant",
    )


def check_writes(controls, before, limits):
    """Check the writes an event made to the control inputs.

    Called right after an event's callback, with a copy of ``controls`` taken
    before it. A changed value is kept inside its limits, or warned about.

    Parameters
    ----------
    controls : dict
        The flight's control inputs, as the callback left them.
    before : dict
        A copy of them taken before the callback.
    limits : dict
        Channel name to ``(limits, clamp)``.

    Returns
    -------
    dict
        The channels whose value changed, with their new value. Empty when
        nothing changed.

    Raises
    ------
    KeyError
        If the callback added or removed a channel, such as by writing a
        misspelled name. The inputs are put back as they were.
    """
    if controls.keys() != before.keys():
        added = sorted(controls.keys() - before.keys())
        controls.clear()
        controls.update(before)
        if added:
            raise KeyError(
                f"There is no control channel {added[0]!r}. The channels are: "
                f"{', '.join(before) or 'none'}."
            )
        raise KeyError("Control channels cannot be removed during a flight.")
    changed = {}
    for name, value in controls.items():
        if value != before[name]:
            bounds, clamp = limits[name]
            controls[name] = changed[name] = limited_value(value, bounds, clamp, name)
    return changed
