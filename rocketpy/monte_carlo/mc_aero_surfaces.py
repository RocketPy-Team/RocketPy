__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, List, Tuple, Union

from pydantic import Field, FilePath, StrictInt, StrictStr

from ..AeroSurface import EllipticalFins, NoseCone, RailButtons, Tail, TrapezoidalFins
from .DispersionModel import DispersionModel


class McNoseCone(DispersionModel):
    """Monte Carlo Nose cone class, used to validate the input parameters of the
    nose cone, based on the pydantic library. It uses the DispersionModel class
    as a base class, see its documentation for more information. The inputs
    defined here correspond to the ones defined in the NoseCone class."""

    # Field(...) means it is a required field, exclude=True removes it from the
    # self.dict() method, which is used to convert the class to a dictionary
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel @root_validator
    # Fields with typing that is not Any have special requirements
    nosecone: NoseCone = Field(..., exclude=True)
    length: Any = 0
    kind: List[Union[StrictStr, None]] = []
    baseRadius: Any = 0
    rocketRadius: Any = 0
    name: List[StrictStr] = []

    def create_object(self):
        """Creates a NoseCone object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : NoseCone
            NoseCone object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = NoseCone(
            length=gen_dict["length"],
            kind=gen_dict["kind"],
            baseRadius=gen_dict["baseRadius"],
            rocketRadius=gen_dict["rocketRadius"],
            name=gen_dict["name"],
        )
        if "position" in gen_dict:
            obj.position = gen_dict["position"]
        return obj


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

    def create_object(self):
        """Creates a TrapezoidalFins object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : TrapezoidalFins
            TrapezoidalFins object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = TrapezoidalFins(
            n=gen_dict["n"],
            rootChord=gen_dict["rootChord"],
            tipChord=gen_dict["tipChord"],
            span=gen_dict["span"],
            rocketRadius=gen_dict["rocketRadius"],
            cantAngle=gen_dict["cantAngle"],
            sweepLength=gen_dict["sweepLength"],
            airfoil=gen_dict["airfoil"],
            name=gen_dict["name"],
        )
        if "position" in gen_dict:
            obj.position = gen_dict["position"]
        return obj


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

    def create_object(self):
        """Creates a EllipticalFins object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : EllipticalFins
            EllipticalFins object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = EllipticalFins(
            n=gen_dict["n"],
            rootChord=gen_dict["rootChord"],
            span=gen_dict["span"],
            rocketRadius=gen_dict["rocketRadius"],
            cantAngle=gen_dict["cantAngle"],
            airfoil=gen_dict["airfoil"],
            name=gen_dict["name"],
        )
        if "position" in gen_dict:
            obj.position = gen_dict["position"]
        return obj


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

    def create_object(self):
        """Creates a Tail object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : Tail
            Tail object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = Tail(
            topRadius=gen_dict["topRadius"],
            bottomRadius=gen_dict["bottomRadius"],
            length=gen_dict["length"],
            rocketRadius=gen_dict["rocketRadius"],
            name=gen_dict["name"],
        )
        if "position" in gen_dict:
            obj.position = gen_dict["position"]
        return obj


class McRailButtons(DispersionModel):
    """Monte Carlo Rail buttons class, used to validate the input parameters of
    the rail buttons, based on the pydantic library. It uses the DispersionModel
    class as a base class, see its documentation for more information.
    """

    rail_buttons: RailButtons = Field(..., exclude=True)
    upper_button_position: Any = 0
    lower_button_position: Any = 0
    angular_position: Any = 0

    def create_object(self):
        """Creates a RailButtons object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : RailButtons
            RailButtons object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = RailButtons(
            upper_button_position=gen_dict["upper_button_position"],
            lower_button_position=gen_dict["lower_button_position"],
            angular_position=gen_dict["angular_position"],
        )
        return obj
