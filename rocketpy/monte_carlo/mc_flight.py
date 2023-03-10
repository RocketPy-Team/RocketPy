from typing import Any, Tuple, Union

from pydantic import StrictFloat, StrictInt

from ..Flight import Flight
from .DispersionModel import DispersionModel


class McFlight(DispersionModel):
    """TODO: Add description

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    inclination: Any
    heading: Any
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
