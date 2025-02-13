"""Defines a custom JSON encoder for RocketPy objects."""

import json
from datetime import datetime
from importlib import import_module

import numpy as np

from rocketpy.mathutils.function import Function


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
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "signature" in obj:
            signature = obj.pop("signature")

            try:
                class_ = get_class_from_signature(signature)

                if hasattr(class_, "from_dict"):
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
