__author__ = "Mateus Stano Junqueira, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, List, Union

from pydantic import Field, FilePath, PrivateAttr
from rocketpy import Components

from rocketpy.tools import get_distribution

from ..AeroSurface import EllipticalFins, NoseCone, RailButtons, Tail, TrapezoidalFins
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

    # Field(...) means it is a required field, exclude=True removes it from the
    # self.dict() method, which is used to convert the class to a dictionary
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel @root_validator
    # Fields with typing that is not Any have special requirements
    rocket: Rocket = Field(..., exclude=True)
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
    _motors: Components = PrivateAttr()
    _nosecones: Components = PrivateAttr()
    _fins: Components = PrivateAttr()
    _tails: Components = PrivateAttr()
    _parachutes: list = PrivateAttr()
    _rail_buttons: Components = PrivateAttr()

    def __init__(self, **kwargs):
        """Initializes private attributes and calls DispersionModel __init__"""
        super().__init__(**kwargs)
        self._motors = Components()
        self._nosecones = Components()
        self._fins = Components()
        self._tails = Components()
        self._parachutes = []
        self._rail_buttons = Components()

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

    def _validate_position(self, position):
        """Checks if 'position' argument was correctly inputted in the 'add'
        methods. Position can be a tuple or a list. If it is a tuple, it must
        have length 2 or 3. If it has length 2, the first item is the nominal
        value of the position, and the second item is the standard deviation.
        If it has length 3, the first item is the nominal value, the second
        item is the standard deviation, and the third item is a string with the
        name of a numpy.random distribution function. If position is a list,
        If a list is passed, the code will chose a random item of the list
        in each simulation ran.

        Parameters
        ----------
        position : tuple, list
            Position inputted in the 'add' methods.
        component_tuple :namedtuple


        Returns
        -------
        tuple
            Tuple with the nominal value, standard deviation, and distribution
            function.

        """
        # checks if tuple
        if isinstance(position, tuple):
            # checks if tuple has acceptable length
            assert len(position) in [
                2,
                3,
            ], f"\nposition: \n\tTuple must have length 2 or 3"
            # checks if first item is valid
            assert isinstance(
                position[0], (int, float)
            ), f"\nposition: \n\tFirst item of tuple must be either an int or float"
            # if len is two can only be (nom_val,std)
            if len(position) == 2:
                # checks if second value is int/float
                assert isinstance(position[1], (int, float)), (
                    f"\nposition:"
                    + " \n\tSecond item of tuple must be an int or float."
                    + " representing the desired standard deviation."
                )
                return (position[0], position[1], get_distribution("normal"))
            # if len is three, then (nom_val, std, 'dist_func')
            if len(position) == 3:
                assert isinstance(position[1], (int, float)), (
                    f"\nposition:"
                    + " \n\tSecond item of tuple must be either an int or float,"
                    + " representing the standard deviation to be used in the"
                    + " simulation"
                )
                assert isinstance(position[2], str), (
                    f"\nposition:"
                    + " \n\tThird item of tuple must be a string containing"
                    + " the name of a valid numpy.random distribution function"
                )
                return (position[0], position[1], get_distribution(position[2]))
        elif isinstance(position, list):
            # guarantee all values are valid (ints or floats)
            assert all(
                isinstance(item, (int, float)) for item in position
            ), f"\nposition: \n\tItems in list must be either ints or floats"
            # all good, sets inputs
            return position
        else:
            raise ValueError(
                f"The 'position' argument must be tuple, list, int or float"
            )

    def addMotor(self, motor, position):
        """Adds a motor to the McRocket object.

        Parameters
        ----------
        motor : McSolidMotor
            The motor to be added to the rocket. Must be a McSolidMotor type.
        position : int, float, tuple, list
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
        self.motors.add(motor, self._validate_position(position))
        return None

    def addNose(self, nose, position):
        """Adds a nose cone to the McRocket object.

        Parameters
        ----------
        nose : McNoseCone #TODO add NoseCone type and include in the description
            The nose cone to be added to the rocket. Must be a McNoseCone type.
        position : int, float, tuple, list
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
        self.nosecones.add(nose, self._validate_position(position))
        return None

    def addTrapezoidalFins(self, fins, position):
        """Adds a trapezoidal fin set to the McRocket object.

        Parameters
        ----------
        fins : McTrapezoidalFins
            The trapezoidal fin set to be added to the rocket. Must be a McTrapezoidalFins type.
        position : int, float, tuple, list
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
        self.fins.add(fins, self._validate_position(position))
        return None

    def addEllipticalFins(self, fins, position):
        """Adds a elliptical fin set to the McRocket object.

        Parameters
        ----------
        fins : McEllipticalFins
            The elliptical fin set to be added to the rocket. Must be a McEllipticalFins type.
        position : int, float, tuple, list
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
        self.fins.add(fins, self._validate_position(position))
        return None

    def addTail(self, tail, position):
        """Adds a tail to the McRocket object.

        Parameters
        ----------
        tail : McTail
            The tail to be added to the rocket. Must be a McTail type.
        position : int, float, tuple, list
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
        self.tails.add(tail, self._validate_position(position))
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

    def setRailButtons(
        self,
        rail_buttons,
        lower_button_position,
    ):
        """Set rail buttons to the McRocket object.

        Parameters
        ----------
        rail_buttons : McRailButtons
            The rail buttons to be added to the rocket. This must be a
            McRailButtons type.
        position : int, float, tuple, list
            Position of the lower rail button (closest to the nozzle)
            in relation to rocket's coordinate system. If float or int,
            refers to the standard deviation. In this case, the nominal
            value of that attribute will come from the tail object passed.
            If the distribution function needs to be specified, then a
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
            In case the input is not a McRailButtons type.
        """
        if not isinstance(rail_buttons, (McRailButtons, RailButtons)):
            raise TypeError("rail_buttons must be of McRailButtons type")
        if isinstance(rail_buttons, RailButtons):
            # create McRailButtons
            rail_buttons = McRailButtons(rail_buttons=rail_buttons)
        self.rail_buttons.add(
            rail_buttons, self._validate_position(lower_button_position)
        )
        return None

    def create_object(self):
        """Creates a Rocket object from the randomly generated input arguments.
        If the rocket has motors, nosecones, fins or tails, they will be added
        accordingly.

        Parameters
        ----------
        None

        Returns
        -------
        obj : Rocket
            Rocket object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = Rocket(
            radius=gen_dict["radius"],
            mass=gen_dict["mass"],
            inertiaI=gen_dict["inertiaI"],
            inertiaZ=gen_dict["inertiaZ"],
            powerOffDrag=gen_dict["powerOffDrag"],
            powerOnDrag=gen_dict["powerOnDrag"],
            centerOfDryMassPosition=0,
            coordinateSystemOrientation="tailToNose",
        )
        obj.powerOffDrag *= gen_dict["powerOffDragFactor"]
        obj.powerOnDrag *= gen_dict["powerOnDragFactor"]

        if self.motors:
            for motor in self.motors:
                m = motor.component.create_object()
                position_rnd = motor.position[-1](*motor.position[:-1])
                obj.addMotor(m, position_rnd)

        if self.nosecones:
            for nosecone in self.nosecones:
                n = nosecone.component.create_object()
                position_rnd = nosecone.position[-1](*nosecone.position[:-1])
                obj.addSurfaces(n, position_rnd)

        if self.fins:
            for fin in self.fins:
                f = fin.component.create_object()
                position_rnd = fin.position[-1](*fin.position[:-1])
                obj.addSurfaces(f, position_rnd)

        if self.tails:
            for tail in self.tails:
                t = tail.component.create_object()
                position_rnd = tail.position[-1](*tail.position[:-1])
                obj.addSurfaces(t, position_rnd)

        if self.parachutes:
            for parachute in self.parachutes:
                p = parachute.create_object()
                obj.addParachute(
                    name=p.name,
                    CdS=p.CdS,
                    trigger=p.trigger,
                    samplingRate=p.samplingRate,
                    lag=p.lag,
                    noise=p.noise,
                )

        if len(self.rail_buttons) != 0:
            for rail_buttons in self.rail_buttons:
                r = rail_buttons.component.create_object()
                lower_button_position_rnd = rail_buttons.position[-1](
                    *rail_buttons.position[:-1]
                )
                upper_button_position_rnd = (
                    r.buttons_distance + lower_button_position_rnd
                )
                obj.setRailButtons(
                    upper_button_position=upper_button_position_rnd,
                    lower_button_position=lower_button_position_rnd,
                    angular_position=r.angular_position,
                )

        return obj
