from typing import Any, List, Union

from pydantic import Field, FilePath, StrictFloat, StrictInt

from ..Motor import SolidMotor
from .DispersionModel import DispersionModel


class McSolidMotor(DispersionModel):
    """_summary_

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

    solidMotor: SolidMotor = Field(..., repr=False, exclude=True)
    thrustSource: List[Union[FilePath, None]] = []
    burnOutTime: Any = 0
    grainsCenterOfMassPosition: Any = 0
    grainNumber: List[Union[Union[StrictInt, StrictFloat], None]] = []
    grainDensity: Any = 0
    grainOuterRadius: Any = 0
    grainInitialInnerRadius: Any = 0
    grainInitialHeight: Any = 0
    grainSeparation: Any = 0
    totalImpulse: Any = 0
    nozzleRadius: Any = 0
    nozzlePosition: Any = 0
    throatRadius: Any = 0
    # TODO: why coordinateSystemOrientation is not included in this class?
