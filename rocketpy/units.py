import numpy as np

from .mathutils.function import Function

# Each value is how many of that unit make one base unit (e.g. "km": 1e-3).
# Multiply by it to leave the base unit, divide by it to reach the base unit.
UNITS_CONVERSION_DICT = {
    # Units of length. Meter "m" is the base unit.
    "mm": 1e3,
    "cm": 1e2,
    "dm": 1e1,
    "m": 1,
    "dam": 1e-1,
    "hm": 1e-2,
    "km": 1e-3,
    "ft": 1 / 0.3048,
    "in": 1 / 0.0254,
    "mi": 1 / 1609.344,
    "nmi": 1 / 1852,
    "yd": 1 / 0.9144,
    # Units of velocity. Meter per second "m/s" is the base unit.
    "m/s": 1,
    "km/h": 3.6,
    "knot": 1.9438444924406047,
    "mph": 2.2369362920544023,
    "ft/s": 1 / 0.3048,
    # Units of acceleration. Meter per square second "m/s^2" is the base unit.
    "m/s^2": 1,
    "gs": 1 / 9.80665,
    "ft/s^2": 1 / 0.3048,
    # Units of pressure. Pascal "Pa" is the base unit.
    "Pa": 1,
    "hPa": 1e-2,
    "kPa": 1e-3,
    "MPa": 1e-6,
    "bar": 1e-5,
    "atm": 1 / 101325,
    "mmHg": 1 / 133.322387415,
    "inHg": 1 / (25.4 * 133.322387415),
    # Units of time. Seconds "s" is the base unit.
    "s": 1,
    "min": 1 / 60,
    "h": 1 / 3600,
    "d": 1 / 86400,
    # Units of mass. Kilogram "kg" is the base unit.
    "mg": 1e6,
    "g": 1e3,
    "kg": 1,
    "lb": 1 / 0.45359237,
    # Units of angle. Radian "rad" is the base unit.
    "rad": 1,
    "deg": 180 / np.pi,
    "grad": 200 / np.pi,
}


def conversion_factor(from_unit, to_unit):
    """Returns the factor that multiplies a value in ``from_unit`` to give it
    in ``to_unit``. See ``UNITS_CONVERSION_DICT`` for how the table is read."""
    try:
        incoming_factor = UNITS_CONVERSION_DICT[to_unit]
    except KeyError as e:
        raise ValueError(f"Unit '{to_unit}' is not supported.") from e
    try:
        outgoing_factor = UNITS_CONVERSION_DICT[from_unit]
    except KeyError as e:
        raise ValueError(f"Unit '{from_unit}' is not supported.") from e

    return incoming_factor / outgoing_factor


def convert_units_Functions(variable, from_unit, to_unit, axis=1):
    """See units.convert_units() for documentation."""
    # Work on copies so the Function passed in is left unchanged
    variable_source = np.array(variable.get_source(), dtype=float)
    inputs = list(variable.__inputs__)
    outputs = list(variable.__outputs__)
    # Perform conversion, take special care with temperatures
    if from_unit in ["K", "degC", "degF"]:
        variable_source[:, axis] = convert_temperature(
            variable_source[:, axis], from_unit, to_unit
        )
    else:
        variable_source[:, axis] *= conversion_factor(from_unit, to_unit)
    # Rename axis labels
    match axis:
        case 0:
            inputs[0] = inputs[0].replace(from_unit, to_unit)
        case 1:
            outputs[0] = outputs[0].replace(from_unit, to_unit)
    # Create new Function instance with converted data
    return Function(
        source=variable_source,
        inputs=inputs,
        outputs=outputs,
        interpolation=variable.__interpolation__,
        extrapolation=variable.__extrapolation__,
    )


def convert_temperature(variable, from_unit, to_unit):
    """See units.convert_units() for documentation."""
    if from_unit == to_unit:
        return variable
    if from_unit == "K" and to_unit == "degC":
        return variable - 273.15
    if from_unit == "K" and to_unit == "degF":
        return (variable - 273.15) * 9 / 5 + 32
    if from_unit == "degC" and to_unit == "K":
        return variable + 273.15
    if from_unit == "degC" and to_unit == "degF":
        return variable * 9 / 5 + 32
    if from_unit == "degF" and to_unit == "K":
        return (variable - 32) * 5 / 9 + 273.15
    if from_unit == "degF" and to_unit == "degC":
        return (variable - 32) * 5 / 9
    # Conversion not supported then...
    raise ValueError(
        f"Temperature conversion from {from_unit} to {to_unit} is not supported."
    )


def convert_units(variable, from_unit, to_unit, axis=1):
    """Convert units of variable to preferred units.

    Parameters
    ----------
    variable : int, float, numpy.array, rocketpy.Function
        Variable to be converted. If Function, specify axis that should
        be converted.
    from_unit : string
        Unit of incoming data.
    to_unit : string
        Unit of returned data.
    axis : int, optional
        Axis that should be converted. 0 for x axis, 1 for y axis.
        Only applies if variable is an instance of the Function class.
        Default is 1, for the y axis.

    Returns
    -------
    variable : int, float, numpy.array, rocketpy.Function
        Variable converted from "from_unit" to "to_unit".
    """
    if from_unit == to_unit:
        # Nothing to convert, same units
        return variable
    if isinstance(variable, Function):
        # Handle Function class
        return convert_units_Functions(variable, from_unit, to_unit, axis)
    else:
        # Handle ints, floats, np.arrays
        if from_unit in ["K", "degC", "degF"]:
            return convert_temperature(variable, from_unit, to_unit)
        else:
            return variable * conversion_factor(from_unit, to_unit)
