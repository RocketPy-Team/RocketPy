"""Defines a custom JSON encoder for RocketPy objects."""

import json
from datetime import datetime
from importlib import import_module

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.prints.flight_prints import _FlightPrints
from rocketpy.plots.flight_plots import _FlightPlots


class RocketPyEncoder(json.JSONEncoder):
    """Custom JSON encoder for RocketPy objects. It defines how to encode
    different types of objects to a JSON supported format."""

    def __init__(self, *args, **kwargs):
        self.include_outputs = kwargs.pop("include_outputs", False)
        self.include_function_data = kwargs.pop("include_function_data", True)
        super().__init__(*args, **kwargs)

    def default(self, o):
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)
        elif isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, datetime):
            return [o.year, o.month, o.day, o.hour]
        elif hasattr(o, "__iter__") and not isinstance(o, str):
            return list(o)
        elif isinstance(o, Function):
            if not self.include_function_data:
                return str(o)
            else:
                encoding = o.to_dict(self.include_outputs)
                encoding["signature"] = get_class_signature(o)
                return encoding
        elif hasattr(o, "to_dict"):
            encoding = o.to_dict(self.include_outputs)
            encoding = remove_circular_references(encoding)

            encoding["signature"] = get_class_signature(o)

            return encoding

        elif hasattr(o, "__dict__"):
            encoding = remove_circular_references(o.__dict__)

            if "rocketpy" in o.__class__.__module__:
                encoding["signature"] = get_class_signature(o)

            return encoding
        else:
            return super().default(o)


class RocketPyDecoder(json.JSONDecoder):
    """Custom JSON decoder for RocketPy objects. It defines how to decode
    different types of objects from a JSON supported format."""

    def __init__(self, *args, **kwargs):
        self.resimulate = kwargs.pop("resimulate", False)
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "signature" in obj:
            signature = obj.pop("signature")

            try:
                class_ = get_class_from_signature(signature)

                if class_.__name__ == "Flight" and not self.resimulate:
                    new_flight = class_.__new__(class_)
                    new_flight.prints = _FlightPrints(new_flight)
                    new_flight.plots = _FlightPlots(new_flight)
                    new_flight.rocket = obj["rocket"]
                    new_flight.env = obj["env"]
                    new_flight.rail_length = obj["rail_length"]
                    new_flight.inclination = obj["inclination"]
                    new_flight.heading = obj["heading"]
                    new_flight.terminate_on_apogee = obj["terminate_on_apogee"]
                    new_flight.max_time = obj["max_time"]
                    new_flight.max_time_step = obj["max_time_step"]
                    new_flight.min_time_step = obj["min_time_step"]
                    new_flight.rtol = obj["rtol"]
                    new_flight.atol = obj["atol"]
                    new_flight.time_overshoot = obj["time_overshoot"]
                    new_flight.name = obj["name"]
                    new_flight.solution = obj["solution"]
                    new_flight.out_of_rail_time = obj["out_of_rail_time"]
                    new_flight.apogee_time = obj["apogee_time"]
                    new_flight.apogee = obj["apogee"]
                    new_flight.parachute_events = obj["parachute_events"]
                    new_flight.impact_state = obj["impact_state"]
                    new_flight.impact_velocity = obj["impact_velocity"]
                    new_flight.x_impact = obj["x_impact"]
                    new_flight.y_impact = obj["y_impact"]
                    new_flight.t_final = obj["t_final"]
                    new_flight.flight_phases = obj["flight_phases"]
                    new_flight.ax = obj["ax"]
                    new_flight.ay = obj["ay"]
                    new_flight.az = obj["az"]
                    new_flight.out_of_rail_time_index = obj["out_of_rail_time_index"]
                    new_flight.function_evaluations = obj["function_evaluations"]
                    new_flight.alpha1 = obj["alpha1"]
                    new_flight.alpha2 = obj["alpha2"]
                    new_flight.alpha3 = obj["alpha3"]
                    new_flight.R1 = obj["R1"]
                    new_flight.R2 = obj["R2"]
                    new_flight.R3 = obj["R3"]
                    new_flight.M1= obj["M1"]
                    new_flight.M2 = obj["M2"]
                    new_flight.M3 = obj["M3"]
                    return new_flight
                elif hasattr(class_, "from_dict"):
                    return class_.from_dict(obj)
                else:
                    # Filter keyword arguments
                    kwargs = {
                        key: value
                        for key, value in obj.items()
                        if key in class_.__init__.__code__.co_varnames
                    }

                    return class_(**kwargs)
            except (ImportError, AttributeError):
                return obj
        else:
            return obj


def get_class_signature(obj):
    """Returns the signature of a class so it can be identified on
    decoding. The signature is a dictionary with the module and
    name of the object's class as strings.


    Parameters
    ----------
    obj : object
        Object to get the signature from.

    Returns
    -------
    dict
        Signature of the class.
    """
    class_ = obj.__class__
    name = getattr(class_, "__qualname__", class_.__name__)

    return {"module": class_.__module__, "name": name}


def get_class_from_signature(signature):
    """Returns the class from its signature dictionary by
    importing the module and loading the class.

    Parameters
    ----------
    signature : dict
        Signature of the class.

    Returns
    -------
    type
        Class defined by the signature.
    """
    module = import_module(signature["module"])
    inner_class = None

    for class_ in signature["name"].split("."):
        inner_class = getattr(module, class_)

    return inner_class


def remove_circular_references(obj_dict):
    """Removes circular references from a dictionary.

    Parameters
    ----------
    obj_dict : dict
        Dictionary to remove circular references from.

    Returns
    -------
    dict
        Dictionary without circular references.
    """
    obj_dict.pop("prints", None)
    obj_dict.pop("plots", None)

    return obj_dict
