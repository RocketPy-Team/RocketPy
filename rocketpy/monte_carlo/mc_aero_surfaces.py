# TODO: add heading and description of the file here, as usual in RocketPy

from typing import Any, List, Tuple, Union

from pydantic import Field, FilePath, StrictInt, StrictStr

from ..AeroSurfaces import EllipticalFins, NoseCone, Tail, TrapezoidalFins
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

    nosecone: NoseCone = Field(..., repr=False, exclude=True)
    length: Any = 0
    kind: List[Union[StrictStr, None]] = []
    baseRadius: Any = 0  # TODO: is this really necessary?
    rocketRadius: Any = 0  # TODO: is this really necessary?
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

    trapezoidalFins: TrapezoidalFins = Field(..., repr=False, exclude=True)
    n: List[StrictInt] = []
    rootChord: Any = 0
    tipChord: Any = 0
    span: Any = 0
    rocketRadius: Any = 0
    cantAngle: Any = 0
    sweepLength: Any = 0
    sweepAngle: Any = 0
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

    ellipticalFins: EllipticalFins = Field(..., repr=False, exclude=True)
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

    tail: Tail = Field(..., repr=False, exclude=True)
    topRadius: Any = 0
    bottomRadius: Any = 0
    length: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []
