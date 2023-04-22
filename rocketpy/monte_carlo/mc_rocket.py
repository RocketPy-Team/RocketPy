__author__ = "Mateus Stano Junqueira, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, List, Union

from pydantic import Field, FilePath, PrivateAttr

from ..AeroSurfaces import EllipticalFins, NoseCone, Tail, TrapezoidalFins
from ..Rocket import Rocket
from .DispersionModel import DispersionModel
from .mc_aero_surfaces import (
    McEllipticalFins,
    McNoseCone,
    McRailButtons,
    McTail,
    McTrapezoidalFins,
)
from .mc_parachute import McParachute
from .mc_solid_motor import McSolidMotor


class McRocket(DispersionModel):
    """Monte Carlo Rocket class, used to validate the input parameters of the
    rocket, based on the pydantic library. It uses the DispersionModel class as a
    base class, see its documentation for more information. The inputs defined
    here correspond to the ones defined in the Rocket class.
    """

    # Field(...) means it is a required field
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel root_validator
    # Fields with any typing that is not Any have special requirements
    rocket: Rocket = Field(...)
    radius: Any = 0
    mass: Any = 0
    inertiaI: Any = 0
    inertiaZ: Any = 0
    powerOffDrag: List[Union[FilePath, None]] = []
    powerOnDrag: List[Union[FilePath, None]] = []
    centerOfDryMassPosition: Any = 0
    powerOffDragFactor: Any = (1, 0)
    powerOnDragFactor: Any = (1, 0)
    # Private attributes for the add methods
    _motors: list = PrivateAttr()
    _nosecones: list = PrivateAttr()
    _fins: list = PrivateAttr()
    _tails: list = PrivateAttr()
    _parachutes: list = PrivateAttr()
    _rail_buttons: list = PrivateAttr()

    def __init__(self, **kwargs):
        """Initializes private attributes and calls DispersionModel __init__"""
        super().__init__(**kwargs)
        self._motors = []
        self._nosecones = []
        self._fins = []
        self._tails = []
        self._parachutes = []
        self._rail_buttons = []

    # getters for attributes of the add methods
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
    def rail_buttons(self):
        return self._rail_buttons

    def _validate_position(self, position, obj, attr_name):
        """Checks if 'position' argument was correctly inputted in the 'add'
        methods. The logic is the same as in the set_attr root validator."""
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
                ), f"position: second item of tuple must be an int, float or string. If the first value refers to the nominal value of 'position', then the item's second value should be the desired standard deviation. If the first value is the standard deviation, then the item's second value should be a string containing a name of a numpy.random distribution function"
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
                ), f"position: second item of tuple must be either an int or float, representing the standard deviation to be used in the simulation"
                assert isinstance(
                    position[2], str
                ), f"position: third item of tuple must be a string containing the name of a valid numpy.random distribution function"
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
                        "Attribute 'position' not found, it should be passed in the 'position' argument of the 'add' method."
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
                f"The 'position' argument must be tuple, list, int or float"
            )

    def addMotor(self, motor, position=[]):
        """Adds a motor to the McRocket object.

        Parameters
        ----------
        motor : McSolidMotor
            The motor to be added to the rocket. Must be a McSolidMotor type.
        position : int, float, tuple, list, optional
            Position of the motor in relation to rocket's coordinate system.
            If float or int, refers to the standard deviation. In this case,
            the nominal value of that attribute will come from the motor object
            passed. If the distribution function needs to be specified, then a
            tuple with the standard deviation as the first item, and the string
            containing the name a numpy.random distribution function can be
            passed e.g. (std, "dist_function").
            If a tuple with a nominal value and a standard deviation is passed,
            then it will take priority over the motor object attribute's value.
            A third item can also be added to the tuple specifying the
            distribution function e.g. (nom_value, std, "dist_function").
            If a list is passed, the code will chose a random item of the list
            in each simulation of the dispersion.
        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case motor is not a McSolidMotor type.
        """
        # checks if input is a McSolidMotor type
        if not isinstance(motor, McSolidMotor):
            raise TypeError("motor must be of McMotor type")
        motor.position = self._validate_position(position, self.rocket, "motorPosition")
        self.motors.append(motor)
        return None

    def addNose(self, nose, position=[]):
        """Adds a nose cone to the McRocket object.

        Parameters
        ----------
        nose : McNoseCone
            The nose cone to be added to the rocket. Must be a McNoseCone type.
        position : int, float, tuple, list, optional
            Position of the nose cone in relation to rocket's coordinate system.
            If float or int, refers to the standard deviation. In this case,
            the nominal value of that attribute will come from the nose cone object
            passed. If the distribution function needs to be specified, then a
            tuple with the standard deviation as the first item, and the string
            containing the name a numpy.random distribution function can be
            passed e.g. (std, "dist_function").
            If a tuple with a nominal value and a standard deviation is passed,
            then it will take priority over the nose cone object attribute's value.
            A third item can also be added to the tuple specifying the
            distribution function e.g. (nom_value, std, "dist_function").
            If a list is passed, the code will chose a random item of the list
            in each simulation of the dispersion.
        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case nose is not a McNoseCone type.
        """
        # checks if input is a McNoseCone or NoseCone type
        if not isinstance(nose, (McNoseCone, NoseCone)):
            raise TypeError(
                "nosecone must be of rocketpy.monte_carlo.McNoseCone or rocketpy.NoseCone type"
            )
        if isinstance(nose, NoseCone):
            # create McNoseCone
            nose = McNoseCone(nosecone=nose)
        nose.position = self._validate_position(position, nose.nosecone, "position")
        self.nosecones.append(nose)
        return None

    def addTrapezoidalFins(self, fins, position=[]):
        """Adds a trapezoidal fin set to the McRocket object.

        Parameters
        ----------
        fins : McTrapezoidalFins
            The trapezoidal fin set to be added to the rocket. Must be a McTrapezoidalFins type.
        position : int, float, tuple, list, optional
            Position of the trapezoidal fin set in relation to rocket's coordinate system.
            If float or int, refers to the standard deviation. In this case,
            the nominal value of that attribute will come from the trapezoidal fin set object
            passed. If the distribution function needs to be specified, then a
            tuple with the standard deviation as the first item, and the string
            containing the name a numpy.random distribution function can be
            passed e.g. (std, "dist_function").
            If a tuple with a nominal value and a standard deviation is passed,
            then it will take priority over the trapezoidal fin set object attribute's value.
            A third item can also be added to the tuple specifying the
            distribution function e.g. (nom_value, std, "dist_function").
            If a list is passed, the code will chose a random item of the list
            in each simulation of the dispersion.
        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case fins is not a McTrapezoidalFins type.
        """
        # checks if input is a McTrapezoidalFins type
        if not isinstance(fins, (McTrapezoidalFins, TrapezoidalFins)):
            raise TypeError("fins must be of McTrapezoidalFins type")
        if isinstance(fins, TrapezoidalFins):
            # create McTrapezoidalFins
            fins = McTrapezoidalFins(trapezoidalFins=fins)
        fins.position = self._validate_position(
            position, fins.trapezoidalFins, "position"
        )
        self.fins.append(fins)
        return None

    def addEllipticalFins(self, fins, position=[]):
        """Adds a elliptical fin set to the McRocket object.

        Parameters
        ----------
        fins : McEllipticalFins
            The elliptical fin set to be added to the rocket. Must be a McEllipticalFins type.
        position : int, float, tuple, list, optional
            Position of the elliptical fin set in relation to rocket's coordinate system.
            If float or int, refers to the standard deviation. In this case,
            the nominal value of that attribute will come from the elliptical fin set object
            passed. If the distribution function needs to be specified, then a
            tuple with the standard deviation as the first item, and the string
            containing the name a numpy.random distribution function can be
            passed e.g. (std, "dist_function").
            If a tuple with a nominal value and a standard deviation is passed,
            then it will take priority over the elliptical fin set object attribute's value.
            A third item can also be added to the tuple specifying the
            distribution function e.g. (nom_value, std, "dist_function").
            If a list is passed, the code will chose a random item of the list
            in each simulation of the dispersion.
        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case fins is not a McEllipticalFins type.
        """
        # checks if input is a McEllipticalFins type
        if not isinstance(fins, (McEllipticalFins, EllipticalFins)):
            raise TypeError("fins must be of McEllipticalFins type")
        if isinstance(fins, EllipticalFins):
            # create McEllipticalFins
            fins = McEllipticalFins(ellipticalFins=fins)
        fins.position = self._validate_position(
            position, fins.ellipticalFins, "position"
        )
        self.fins.append(fins)
        return None

    def addTail(self, tail, position=[]):
        """Adds a tail to the McRocket object.

        Parameters
        ----------
        tail : McTail
            The tail to be added to the rocket. Must be a McTail type.
        position : int, float, tuple, list, optional
            Position of the tail in relation to rocket's coordinate system.
            If float or int, refers to the standard deviation. In this case,
            the nominal value of that attribute will come from the tail object
            passed. If the distribution function needs to be specified, then a
            tuple with the standard deviation as the first item, and the string
            containing the name a numpy.random distribution function can be
            passed e.g. (std, "dist_function").
            If a tuple with a nominal value and a standard deviation is passed,
            then it will take priority over the tail object attribute's value.
            A third item can also be added to the tuple specifying the
            distribution function e.g. (nom_value, std, "dist_function").
            If a list is passed, the code will chose a random item of the list
            in each simulation of the dispersion.
        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case tail is not a McTail type.
        """
        # checks if input is a McTail type
        if not isinstance(tail, (McTail, Tail)):
            raise TypeError("tail must be of McTail type")
        if isinstance(tail, Tail):
            # create McTail
            tail = McTail(tail=tail)
        tail.position = self._validate_position(position, tail.tail, "position")
        self.tails.append(tail)
        return None

    def addParachute(self, parachute):
        """Adds a parachute to the McRocket object.

        Parameters
        ----------
        parachute : McParachute
            The parachute to be added to the rocket. This must be a McParachute
            type.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case the input is not a McParachute type.
        """
        # checks if input is a McParachute type
        if not isinstance(parachute, McParachute):
            raise TypeError("parachute must be of McParachute type")
        self.parachutes.append(parachute)
        return None

    def addRailButtons(
        self,
        rail_buttons,
    ):
        """Method to add rail buttons to the McRocket object.

        Parameters
        ----------
        rail_buttons : McRailButtons
            The rail buttons to be added to the rocket. This must be a
            McRailButtons type.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            In case the input is not a McRailButtons type.
        """
        if not isinstance(rail_buttons, McRailButtons):
            raise TypeError("rail_buttons must be of McRailButtons type")
        self.rail_buttons.append(rail_buttons)
        return None
