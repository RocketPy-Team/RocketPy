__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


from typing import Any, Tuple, Union

from pydantic import Field, StrictFloat, StrictInt

from ..Flight import Flight
from .DispersionModel import DispersionModel


class McFlight(DispersionModel):
    """Monte Carlo Flight class, used to validate the input parameters of the
    flight to be used in the Dispersion class, based on the pydantic library. It
    uses the DispersionModel class as a base class, see its documentation for
    more information. The inputs defined here are the same as the ones defined
    in the Flight class, see its documentation for more information.
    """

    flight: Flight = Field(..., exclude=True)
    inclination: Any = 0
    heading: Any = 0
    initialSolution: Union[
        Flight,
        Tuple[
            Union[StrictInt, StrictFloat],  # tInitial
            Union[StrictInt, StrictFloat],  # xInit
            Union[StrictInt, StrictFloat],  # yInit
            Union[StrictInt, StrictFloat],  # zInit
            Union[StrictInt, StrictFloat],  # vxInit
            Union[StrictInt, StrictFloat],  # vyInit
            Union[StrictInt, StrictFloat],  # vzInit
            Union[StrictInt, StrictFloat],  # e0Init
            Union[StrictInt, StrictFloat],  # e1Init
            Union[StrictInt, StrictFloat],  # e2Init
            Union[StrictInt, StrictFloat],  # e3Init
            Union[StrictInt, StrictFloat],  # w1Init
            Union[StrictInt, StrictFloat],  # w2Init
            Union[StrictInt, StrictFloat],  # w3Init
        ],
    ] = None
    terminateOnApogee: bool = False
