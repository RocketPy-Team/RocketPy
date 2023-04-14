__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, Callable, List, Tuple, Union

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from ..Parachute import Parachute
from .DispersionModel import DispersionModel


class McParachute(DispersionModel):
    """Monte Carlo Parachute class, used to validate the input parameters of the
    parachute, based on the pydantic library. It uses the DispersionModel class
    as a base class, see its documentation for more information. The inputs
    defined here correspond to the ones defined in the Parachute class.
    """

    parachute: Parachute = Field(..., exclude=True)
    CdS: Any = 0
    trigger: List[Union[Callable, None]] = []
    samplingRate: Any = 0
    lag: Any = 0
    name: List[Union[StrictStr, None]] = []
    noise: List[
        Union[
            Tuple[
                Union[StrictInt, StrictFloat],
                Union[StrictInt, StrictFloat],
                Union[StrictInt, StrictFloat],
            ],
            None,
        ]
    ] = []
