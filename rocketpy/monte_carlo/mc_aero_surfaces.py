__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, List, Tuple, Union

from pydantic import Field, FilePath, StrictInt, StrictStr

from ..AeroSurfaces import EllipticalFins, NoseCone, RailButtons, Tail, TrapezoidalFins
from .DispersionModel import DispersionModel


class McNoseCone(DispersionModel):
    """Monte Carlo Nose cone class, used to validate the input parameters of the
    nose cone, based on the pydantic library. It uses the DispersionModel class
    as a base class, see its documentation for more information. The inputs
    defined here correspond to the ones defined in the NoseCone class."""

    nosecone: NoseCone = Field(..., exclude=True)
    length: Any = 0
    kind: List[Union[StrictStr, None]] = []
    baseRadius: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []


class McTrapezoidalFins(DispersionModel):
    """Monte Carlo Trapezoidal fins class, used to validate the input parameters
    of the trapezoidal fins, based on the pydantic library. It uses the
    DispersionModel class as a base class, see its documentation for more
    information.
    """

    trapezoidalFins: TrapezoidalFins = Field(..., exclude=True)
    n: List[StrictInt] = []
    rootChord: Any = 0
    tipChord: Any = 0
    span: Any = 0
    rocketRadius: Any = 0
    cantAngle: Any = 0
    sweepLength: Any = 0
    # The sweep angle is irrelevant for dispersion, use sweepLength instead
    # sweepAngle: Any = 0
    airfoil: List[Union[Tuple[FilePath, StrictStr], None]] = []
    name: List[StrictStr] = []


class McEllipticalFins(DispersionModel):
    """Monte Carlo Elliptical fins class, used to validate the input parameters
    of the elliptical fins, based on the pydantic library. It uses the
    DispersionModel class as a base class, see its documentation for more
    information.
    """

    ellipticalFins: EllipticalFins = Field(..., exclude=True)
    n: Any = 0
    rootChord: Any = 0
    span: Any = 0
    rocketRadius: Any = 0
    cantAngle: Any = 0
    airfoil: List[Union[Tuple[FilePath, StrictStr], None]] = []
    name: List[StrictStr] = []


class McTail(DispersionModel):
    """Monte Carlo Tail class, used to validate the input parameters of the tail
    based on the pydantic library. It uses the DispersionModel class as a base
    class, see its documentation for more information."""

    tail: Tail = Field(..., exclude=True)
    topRadius: Any = 0
    bottomRadius: Any = 0
    length: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []


class McRailButtons(DispersionModel):
    """Monte Carlo Rail buttons class, used to validate the input parameters of
    the rail buttons, based on the pydantic library. It uses the DispersionModel
    class as a base class, see its documentation for more information.
    """

    rail_buttons: RailButtons = Field(..., exclude=True)
    upper_button_position: Any = 0
    lower_button_position: Any = 0
    angular_position: Any = 0
