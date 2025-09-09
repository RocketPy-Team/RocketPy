"""Defines a custom JSON encoder for RocketPy objects."""

import json
from datetime import datetime
from importlib import import_module

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.flight_plots import _FlightPlots
from rocketpy.prints.flight_prints import _FlightPrints


class RocketPyEncoder(json.JSONEncoder):
    """Custom JSON encoder for RocketPy objects. It defines how to encode
    different types of objects to a JSON supported format.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the encoder with parameter options.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the parent class.
        **kwargs : dict
            Keyword arguments to configure the encoder. The following
            options are available:
            - include_outputs: bool, whether to include simulation outputs.
              Default is False.
            - include_function_data: bool, whether to include Function
              data in the encoding. If False, Functions will be encoded by their
              ``__repr__``. This is useful for reducing the size of the outputs,
              but it prevents full restoration of the object upon decoding.
              Default is True.
            - discretize: bool, whether to discretize Functions whose source
              are callables. If True, the accuracy of the decoding may be reduced.
              Default is False.
            - allow_pickle: bool, whether to pickle callable objects. If
              False, callable sources (such as user-defined functions, parachute
              triggers or simulation callable outputs) will have their name
              stored instead of the function itself. This is useful for
              reducing the size of the outputs, but it prevents full restoration
              of the object upon decoding.
              Default is True.
        """
        self.include_outputs = kwargs.pop("include_outputs", False)
        self.include_function_data = kwargs.pop("include_function_data", True)
        self.discretize = kwargs.pop("discretize", False)
        self.allow_pickle = kwargs.pop("allow_pickle", True)
        super().__init__(*args, **kwargs)

    def default(self, o):
        if isinstance(o, np.generic):
            return o.item()
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
                encoding = o.to_dict(
                    include_outputs=self.include_outputs,
                    discretize=self.discretize,
                    allow_pickle=self.allow_pickle,
                )
                encoding["signature"] = get_class_signature(o)
                return encoding
        elif hasattr(o, "to_dict"):
            encoding = o.to_dict(
                include_outputs=self.include_outputs,
                discretize=self.discretize,
                allow_pickle=self.allow_pickle,
            )
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
                    set_minimal_flight_attributes(new_flight, obj)
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


def set_minimal_flight_attributes(flight, obj):
    attributes = (
        "rocket",
        "env",
        "rail_length",
        "inclination",
        "heading",
        "initial_solution",
        "terminate_on_apogee",
        "max_time",
        "max_time_step",
        "min_time_step",
        "rtol",
        "atol",
        "time_overshoot",
        "name",
        "solution",
        "out_of_rail_time",
        "apogee_time",
        "apogee",
        "parachute_events",
        "impact_state",
        "impact_velocity",
        "x_impact",
        "y_impact",
        "t_final",
        "flight_phases",
        "ax",
        "ay",
        "az",
        "out_of_rail_time_index",
        "function_evaluations",
        "speed",
        "alpha1",
        "alpha2",
        "alpha3",
        "R1",
        "R2",
        "R3",
        "M1",
        "M2",
        "M3",
        "net_thrust",
    )

    for attribute in attributes:
        try:
            setattr(flight, attribute, obj[attribute])
        except KeyError:
            # Manual resolution of new attributes
            if attribute == "net_thrust":
                flight.net_thrust = obj["rocket"].motor.thrust
                flight.net_thrust.set_discrete_based_on_model(flight.speed)

    flight.t_initial = flight.initial_solution[0]


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
