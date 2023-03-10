from typing import Any, Callable, List, Tuple, Union

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from ..Parachute import Parachute
from .DispersionModel import DispersionModel


class McParachute(DispersionModel):
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

    parachute: Parachute = Field(..., repr=False)
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
