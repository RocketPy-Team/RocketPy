__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, List, Union

from pydantic import Field, FilePath, StrictFloat, StrictInt

from ..Motor import SolidMotor
from .DispersionModel import DispersionModel


class McSolidMotor(DispersionModel):
    """Monte Carlo Solid Motor class, used to validate the input parameters of
    the solid motor, based on the pydantic library. It uses the DispersionModel
    class as a base class, see its documentation for more information. The
    inputs defined here correspond to the ones defined in the SolidMotor class.
    """

    # Field(...) means it is a required field
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel root_validator
    # Fields with any typing that is not Any have special requirements
    solidMotor: SolidMotor = Field(...)
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
