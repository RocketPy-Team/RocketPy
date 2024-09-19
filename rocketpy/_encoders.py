"""Defines a custom JSON encoder for RocketPy objects."""

import base64
import json
from datetime import datetime
from importlib import import_module

import dill
import numpy as np


class RocketPyEncoder(json.JSONEncoder):
    """Custom JSON encoder for RocketPy objects. It defines how to encode
    different types of objects to a JSON supported format."""

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
        elif hasattr(o, "to_dict"):
            encoding = o.to_dict()

            encoding["signature"] = get_class_signature(o)

            return encoding

        elif hasattr(o, "__dict__"):
            exception_set = {"prints", "plots"}
            encoding = {
                key: value
                for key, value in o.__dict__.items()
                if key not in exception_set
            }

            if "rocketpy" in o.__class__.__module__ and not any(
                subclass in o.__class__.__name__
                for subclass in ["FlightPhase", "TimeNode"]
            ):
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
            except ImportError:  # AttributeException
                return obj
        else:
            return obj


def get_class_signature(obj):
    class_ = obj.__class__

    return f"{class_.__module__}.{class_.__name__}"


def get_class_from_signature(signature):
    module_name, class_name = signature.rsplit(".", 1)

    module = import_module(module_name)

    return getattr(module, class_name)


def to_hex_encode(obj, encoder=base64.b85encode):
    """Converts an object to hex representation using dill.

    Parameters
    ----------
    obj : object
        Object to be converted to hex.
    encoder : callable, optional
        Function to encode the bytes. Default is base64.b85encode.

    Returns
    -------
    bytes
        Object converted to bytes.
    """
    return encoder(dill.dumps(obj)).hex()


def from_hex_decode(obj_bytes, decoder=base64.b85decode):
    """Converts an object from hex representation using dill.

    Parameters
    ----------
    obj_bytes : str
        Hex string to be converted to object.
    decoder : callable, optional
        Function to decode the bytes. Default is base64.b85decode.

    Returns
    -------
    object
        Object converted from bytes.
    """
    return dill.loads(decoder(bytes.fromhex(obj_bytes)))
