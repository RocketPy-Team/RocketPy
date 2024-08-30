"""Defines a custom JSON encoder for RocketPy objects."""

import json
from datetime import datetime

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
            return o.isoformat()
        elif hasattr(o, "__iter__") and not isinstance(o, str):
            return list(o)
        elif hasattr(o, "to_dict"):
            return o.to_dict()
        elif hasattr(o, "__dict__"):
            exception_set = {"prints", "plots"}
            return {
                key: value
                for key, value in o.__dict__.items()
                if key not in exception_set
            }
        else:
            return super().default(o)
