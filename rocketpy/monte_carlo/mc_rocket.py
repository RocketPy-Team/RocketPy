from typing import Any, List, Union

from pydantic import BaseModel, Field, FilePath, PrivateAttr

from ..AeroSurfaces import EllipticalFins, NoseCone, Tail, TrapezoidalFins
from ..Parachute import Parachute
from ..Rocket import Rocket
from .DispersionModel import DispersionModel
from .mc_aero_surfaces import McEllipticalFins, McNoseCone, McTail, McTrapezoidalFins
from .mc_parachute import McParachute
from .mc_solid_motor import McSolidMotor


class McRocket(DispersionModel):
    rocket: Rocket = Field(..., repr=False, exclude=True)
    radius: Any = 0
    mass: Any = 0
    inertiaI: Any = 0
    inertiaZ: Any = 0
    powerOffDrag: List[Union[FilePath, None]] = []
    powerOnDrag: List[Union[FilePath, None]] = []
    centerOfDryMassPosition: Any = 0
    powerOffDragFactor: Any = (1, 0)
    powerOnDragFactor: Any = (1, 0)
    _motors: list = PrivateAttr()
    _nosecones: list = PrivateAttr()
    _fins: list = PrivateAttr()
    _tails: list = PrivateAttr()
    _parachutes: list = PrivateAttr()
    _railButtons: list = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._motors = []
        self._nosecones = []
        self._fins = []
        self._tails = []
        self._parachutes = []
        self._railButtons = []

    @property
    def motors(self):
        return self._motors

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

    def _validate_position(self, position, obj, attr_name):
        """Checks if 'position' argument was correctly inputted in the 'add' methods.
        The logic is the same as in the set_attr root validator."""
        # checks if tuple
        if isinstance(position, tuple):
            # checks if first item is valid
            assert isinstance(
                position[0], (int, float)
            ), f"\nposition: \n\tFirst item of tuple must be either an int or float"
            # if len is two can either be (nom_val,std) or (std,'dist_func')
            if len(position) == 2:
                # checks if second value is either string or int/float
                assert isinstance(
                    position[1], (int, float, str)
                ), f"\nposition: \n\tSecond item of tuple must be either an int, float or string \n\tIf the first value refers to the nominal value of 'position', then the second item's value should be the desired standard deviation \n\tIf the first value is the standard deviation, then the second item's value should be a string containing a name of a numpy.random distribution function"
                # if second item is not str, then (nom_val, std)
                if not isinstance(position[1], str):
                    return position
                # if second item is str, then (nom_val, std, str)
                else:
                    # tries to get position from object
                    # this checks if the object has the position attribute
                    # meaning it was defined using a rocket add method
                    # if not, then position must be inputted in McRocket add methods
                    try:
                        nom_value = getattr(obj, attr_name)
                    except:
                        raise AttributeError(
                            "Attribute 'position' not found. Position should be passed in the 'position' argument of the 'add' method."
                        )
                    return (nom_value, position[0], position[1])
            # if len is three, then (nom_val, std, 'dist_func')
            if len(position) == 3:
                assert isinstance(
                    position[1], (int, float)
                ), f"\nposition: \n\tSecond item of tuple must be either an int or float \n\tThe second item should be the standard deviation to be used in the simulation"
                assert isinstance(
                    position[2], str
                ), f"\nposition: \n\tThird item of tuple must be a string \n\tThe string should contain the name of a valid numpy.random distribution function"
                return position
        elif isinstance(position, list):
            # checks if input list is empty, meaning nothing was inputted
            # and values should be gotten from class
            if len(position) == 0:
                # tries to get position from object
                # this checks if the object has the position attribute
                # meaning it was defined using a rocket add method
                # if not, then position must be inputted in McRocket add methods
                try:
                    nom_value = getattr(obj, attr_name)
                except:
                    raise AttributeError(
                        "Attribute 'position' not found. Position should be passed in the 'position' argument of the 'add' method."
                    )
                return [nom_value]
            else:
                # guarantee all values are valid (ints or floats)
                assert all(
                    isinstance(item, (int, float)) for item in position
                ), f"\nposition: \n\tItems in list must be either ints or floats"
                # all good, sets inputs
                return position
        elif isinstance(position, (int, float)):
            # not list or tuple, must be an int or float
            # get attr and returns (nom_value, std)

            # tries to get position from object
            # this checks if the object has the position attribute
            # meaning it was defined using a rocket add method
            # if not, then position must be inputted in McRocket add methods
            try:
                nom_value = getattr(obj, attr_name)
            except:
                raise AttributeError(
                    "Attribute 'position' not found. Position should be passed in the 'position' argument of the 'add' method."
                )
            return (nom_value, position)
        else:
            raise ValueError(
                f"\nposition: \n\tMust be either a tuple, list, int or float"
            )

    def addMotor(self, motor, position=[]):
        # checks if input is a McSolidMotor type
        if not isinstance(motor, McSolidMotor):
            raise TypeError("motor must be of McMotor type")
        motor.position = self._validate_position(position, self.rocket, "motorPosition")
        return self.motors.append(motor)

    def addNose(self, nose, position=[]):
        # checks if input is a McNoseCone or NoseCone type
        if not isinstance(nose, (McNoseCone, NoseCone)):
            raise TypeError(
                "nosecone must be of rocketpy.monte_carlo.McNoseCone or rocketpy.NoseCone type"
            )
        if isinstance(nose, NoseCone):
            # create McNoseCone
            nose = McNoseCone(nosecone=nose)
        nose.position = self._validate_position(position, nose.nosecone, "position")
        return self.nosecones.append(nose)

    def addTrapezoidalFins(self, fins, position=[]):
        # checks if input is a McTrapezoidalFins type
        if not isinstance(fins, (McTrapezoidalFins, TrapezoidalFins)):
            raise TypeError("fins must be of McTrapezoidalFins type")
        if isinstance(fins, TrapezoidalFins):
            # create McTrapezoidalFins
            fins = McTrapezoidalFins(trapezoidalFins=fins)
        fins.position = self._validate_position(
            position, fins.trapezoidalFins, "position"
        )
        return self.fins.append(fins)

    def addEllipticalFins(self, fins, position=[]):
        # checks if input is a McEllipticalFins type
        if not isinstance(fins, (McEllipticalFins, EllipticalFins)):
            raise TypeError("fins must be of McEllipticalFins type")
        if isinstance(fins, EllipticalFins):
            # create McEllipticalFins
            fins = McEllipticalFins(ellipticalFins=fins)
        fins.position = self._validate_position(
            position, fins.ellipticalFins, "position"
        )
        return self.fins.append(fins)

    def addTail(self, tail, position=[]):
        # checks if input is a McTail type
        if not isinstance(tail, (McTail, Tail)):
            raise TypeError("tail must be of McTail type")
        if isinstance(tail, Tail):
            # create McTail
            tail = McTail(tail=tail)
        tail.position = self._validate_position(position, tail.tail, "position")
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


class McButtons(BaseModel):
    """Class for the rail buttons"""

    position1: float
    position2: float
    angle: float

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position1 = kwargs["position1"]
        self.position2 = kwargs["position2"]
        self.angle = kwargs["angle"]
