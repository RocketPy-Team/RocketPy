from typing import Any, List, Tuple, Union

from pydantic import BaseModel, Field, FilePath, PrivateAttr, root_validator

from ..Rocket import Rocket
from .mc_aero_surfaces import McEllipticalFins, McNoseCone, McTail, McTrapezoidalFins
from .mc_parachute import McParachute


# TODO: make a special validator for power on and off factor since they need to have the nominal
# value inputted
class McRocket(BaseModel):
    rocket: Rocket = Field(..., repr=False)
    radius: Any = 0
    mass: Any = 0
    inertiaI: Any = 0
    inertiaZ: Any = 0
    powerOffDrag: List[Union[FilePath, None]] = []
    powerOnDrag: List[Union[FilePath, None]] = []
    centerOfDryMassPosition: Any = 0
    powerOffDragFactor: Any = 0
    powerOnDragFactor: Any = 0
    _nosecones: list = PrivateAttr()
    _fins: list = PrivateAttr()
    _tails: list = PrivateAttr()
    _parachutes: list = PrivateAttr()
    _railButtons: list = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nosecones = []
        self._fins = []
        self._tails = []
        self._parachutes = []
        self._railButtons = []

    @property
    def nosecones(self):
        return self._nosecones

    @property
    def fins(self):
        return self._fins

    @property
    def tails(self):
        return self._tails

    @property
    def parachutes(self):
        return self._parachutes

    @property
    def railButtons(self):
        return self._railButtons

    @root_validator(skip_on_failure=True)
    def set_attr(cls, values):
        """Validates inputs that can be either tuples, lists, ints or floats and
        saves them in the format (nom_val,std) or (nom_val,std,'dist_func').
        Lists are saved as they are inputted.
        Inputs can be given as floats or ints, referring to the standard deviation.
        In this case, the nominal value of that attribute will come from the rocket
        object passed. If the distribution function needs to be specified, then a
        tuple with the standard deviation as the first item, and the string containing
        the name a numpy.random distribution function can be passed.
        If a tuple with a nominal value and a standard deviation is passed, then it
        will take priority over the rocket object attribute's value. A third item
        can also be added to the tuple specifying the distribution function"""
        # TODO: add a way to check if the strings referring to the distribution func
        # are valid names for numpy.random functions

        validate_fields = [
            "radius",
            "mass",
            "inertiaI",
            "inertiaZ",
            "powerOffDrag",
            "powerOnDrag",
            "centerOfDryMassPosition",
            "powerOffDragFactor",
            "powerOnDragFactor",
        ]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first item is valid
                assert isinstance(
                    v[0], (int, float)
                ), f"\nField '{field}': \n\tFirst item of tuple must be either an int or float"
                # if len is two can either be (nom_val,std) or (std,'dist_func')
                if len(v) == 2:
                    # checks if second value is either string or int/float
                    assert isinstance(
                        v[1], (int, float, str)
                    ), f"\nField '{field}': \n\tSecond item of tuple must be either an int, float or string \n  If the first value refers to the nominal value of {field}, then the second item's value should be the desired standard deviation \n  If the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                    # if second item is not str, then (nom_val, std)
                    if not isinstance(v[1], str):
                        values[field] = v
                    # if second item is str, then (nom_val, std, str)
                    else:
                        values[field] = (getattr(values["rocket"], field), v[0], v[1])
                # if len is three, then (nom_val, std, 'dist_func')
                if len(v) == 3:
                    assert isinstance(
                        v[1], (int, float)
                    ), f"\nField '{field}': \n\tSecond item of tuple must be either an int or float \n  The second item should be the standard deviation to be used in the simulation"
                    assert isinstance(
                        v[2], str
                    ), f"\nField '{field}': \n\tThird item of tuple must be a string \n  The string should contain the name of a valid numpy.random distribution function"
                    values[field] = v
            elif isinstance(v, list):
                # checks if input list is empty, meaning nothing was inputted
                # and values should be gotten from class
                if len(v) == 0:
                    values[field] = [getattr(values["rocket"], field)]
                else:
                    # guarantee all values are valid (ints or floats)
                    assert all(
                        isinstance(item, (int, float)) for item in v
                    ), f"\nField '{field}': \n\tItems in list must be either ints or floats"
                    # all good, sets inputs
                    values[field] = v
            elif isinstance(v, (int, float)):
                # not list or tuple, must be an int or float
                # get attr and returns (nom_value, std)
                values[field] = (getattr(values["rocket"], field), v)
            else:
                raise ValueError(
                    f"\nField '{field}': \n\tMust be either a tuple, list, int or float"
                )
        return values

    def addNose(self, nosecone):
        # checks if input is a McNoseCone type
        if not isinstance(nosecone, McNoseCone):
            raise TypeError("nosecone must be of McNoseCone type")
        return self.nosecones.append(nosecone)

    def addTrapezoidalFins(self, trapezoidalFins):
        # checks if input is a McNoseCone type
        if not isinstance(trapezoidalFins, McTrapezoidalFins):
            raise TypeError("trapezoidalFins must be of McNoseCone type")
        return self.fins.append(trapezoidalFins)

    def addEllipticalFins(self, ellipticalFins):
        # checks if input is a McNoseCone type
        if not isinstance(ellipticalFins, McEllipticalFins):
            raise TypeError("ellipticalFins must be of McNoseCone type")
        return self.fins.append(ellipticalFins)

    def addTail(self, tail):
        # checks if input is a McNoseCone type
        if not isinstance(tail, McTail):
            raise TypeError("nosecone must be of McNoseCone type")
        return self.tails.append(tail)

    def addParachute(self, parachute):
        # checks if input is a McNoseCone type
        if not isinstance(parachute, McParachute):
            raise TypeError("parachute must be of McParachute type")
        return self.parachutes.append(parachute)

    def addRailButtons(self, position1, position2, angle):
        # TODO: transform rail buttons into data classes
        # TODO: currently does not vary anything just for testing
        self.railButtons = [position1, position2, angle]
