# TODO: add heading and description of the file here, as usual in RocketPy

from typing import Any, List, Tuple, Union

from pydantic import Field, FilePath, StrictInt, StrictStr

from ..AeroSurfaces import EllipticalFins, NoseCone, RailButtons, Tail, TrapezoidalFins
from .DispersionModel import DispersionModel


class McNoseCone(DispersionModel):
    """TODO: add docstring here, maybe with some examples

    Parameters
    ----------
    DispersionModel : _type_
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

    nosecone: NoseCone = Field(..., exclude=True)
    length: Any = 0
    kind: List[Union[StrictStr, None]] = []
    baseRadius: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []
    # TODO: question: how can we document the above code lines? Why are they here?


class McTrapezoidalFins(DispersionModel):
    """TODO: add docstring here, maybe with some examples

    Parameters
    ----------
    DispersionModel : _type_
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
    """TODO: add docstring here, maybe with some examples

    Parameters
    ----------
    DispersionModel : _type_
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

    ellipticalFins: EllipticalFins = Field(..., exclude=True)
    n: Any = 0
    rootChord: Any = 0
    span: Any = 0
    rocketRadius: Any = 0
    cantAngle: Any = 0
    airfoil: List[Union[Tuple[FilePath, StrictStr], None]] = []
    name: List[StrictStr] = []


class McTail(DispersionModel):
    """TODO: add docstring

    Parameters
    ----------
    DispersionModel : _type_
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

    tail: Tail = Field(..., exclude=True)
    topRadius: Any = 0
    bottomRadius: Any = 0
    length: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []


class McRailButtons(DispersionModel):
    """TODO: add docstring

    Parameters
    ----------
    DispersionModel : _type_
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

    rail_buttons: RailButtons = Field(..., exclude=True)
    upper_button_position: Any = 0
    lower_button_position: Any = 0
    angular_position: Any = 0
