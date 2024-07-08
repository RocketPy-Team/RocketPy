"""Defines a custom JSON encoder for RocketPy objects."""

import json
import types

import numpy as np

from rocketpy.mathutils.function import Function


class RocketPyEncoder(json.JSONEncoder):
    """NOTE: This is still under construction, please don't use it yet."""

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
        elif hasattr(o, "to_dict"):
            return o.to_dict()
        # elif isinstance(o, Function):
        #     return o.__dict__()
        elif isinstance(o, (Function, types.FunctionType)):
            return repr(o)
        else:
            return json.JSONEncoder.default(self, o)
