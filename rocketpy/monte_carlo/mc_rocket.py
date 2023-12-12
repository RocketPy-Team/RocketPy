import warnings
from random import choice

from rocketpy.monte_carlo.motor_dispersion_model import MotorDispersionModel
from rocketpy.motors.motor import EmptyMotor, Motor
from rocketpy.rocket.aero_surface import (
    EllipticalFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from rocketpy.rocket.components import Components
from rocketpy.rocket.parachute import Parachute
from rocketpy.rocket.rocket import Rocket

from .dispersion_model import DispersionModel
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
    """A Monte Carlo Rocket class that inherits from MonteCarloModel. This
    class is used to receive a Rocket object and information about the
    dispersion of its parameters and generate a random rocket object based on
    the provided information.

    Attributes
    ----------
    object : Rocket
        The Rocket object to be used as a base for the Monte Carlo rocket.
    motors : Components
        A Components instance containing all the motors of the rocket.
    aerodynamic_surfaces : Components
        A Components instance containing all the aerodynamic surfaces of the
        rocket.
    rail_buttons : Components
        A Components instance containing all the rail buttons of the rocket.
    parachutes : list of McParachute
        A list of McParachute instances containing all the parachutes of the
        rocket.
    radius : tuple, list, int, float
        The radius of the rocket. Follows the standard input format of
        Dispersion Models.
    mass : tuple, list, int, float
        The mass of the rocket. Follows the standard input format of
        Dispersion Models.
    inertia_11 : tuple, list, int, float
        The inertia of the rocket around the x axis. Follows the standard input
        format of Dispersion Models.
    inertia_22 : tuple, list, int, float
        The inertia of the rocket around the y axis. Follows the standard input
        format of Dispersion Models.
    inertia_33 : tuple, list, int, float
        The inertia of the rocket around the z axis. Follows the standard input
        format of Dispersion Models.
    inertia_12 : tuple, list, int, float
        The inertia of the rocket around the xy axis. Follows the standard
        input format of Dispersion Models.
    inertia_13 : tuple, list, int, float
        The inertia of the rocket around the xz axis. Follows the standard
        input format of Dispersion Models.
    inertia_23 : tuple, list, int, float
        The inertia of the rocket around the yz axis. Follows the standard
        input format of Dispersion Models.
    power_off_drag : list
        The power off drag of the rocket. Follows the 1d array like input format
        of Dispersion Models.
    power_on_drag : list
        The power on drag of the rocket. Follows the 1d array like input format
        of Dispersion Models.
    power_off_drag_factor : tuple, list, int, float
        The power off drag factor of the rocket. Follows the factor input
        format of Dispersion Models.
    power_on_drag_factor : tuple, list, int, float
        The power on drag factor of the rocket. Follows the standard input
        format of Dispersion Models.
    center_of_mass_without_motor : tuple, list, int, float
        The center of mass of the rocket without the motor. Follows the
        standard input format of Dispersion Models.
    coordinate_system_orientation : list
        The orientation of the coordinate system of the rocket. This attribute
        can not be a randomized.
    """

    def __init__(
        self,
        rocket,
        radius=None,
        mass=None,
        inertia_11=None,
        inertia_22=None,
        inertia_33=None,
        inertia_12=None,
        inertia_13=None,
        inertia_23=None,
        power_off_drag=None,
        power_on_drag=None,
        power_off_drag_factor=(1, 0),
        power_on_drag_factor=(1, 0),
        center_of_mass_without_motor=None,
    ):
        """Initializes the Monte Carlo Rocket class.

        See Also
        --------
        This should link to somewhere that explains how inputs works in
        dispersion models.

        Parameters
        ----------
        rocket : Rocket
            The Rocket object to be used as a base for the Monte Carlo rocket.
        radius : int, float, tuple, list, optional
            The radius of the rocket. Follows the standard input format of
            Dispersion Models.
        mass : int, float, tuple, list, optional
            The mass of the rocket. Follows the standard input format of
            Dispersion Models.
        inertia_11 : int, float, tuple, list, optional
            The inertia of the rocket around the x axis. Follows the standard
            input format of Dispersion Models.
        inertia_22 : int, float, tuple, list, optional
            The inertia of the rocket around the y axis. Follows the standard
            input format of Dispersion Models.
        inertia_33 : int, float, tuple, list, optional
            The inertia of the rocket around the z axis. Follows the standard
            input format of Dispersion Models.
        inertia_12 : int, float, tuple, list, optional
            The inertia of the rocket around the xy axis. Follows the standard
            input format of Dispersion Models.
        inertia_13 : int, float, tuple, list, optional
            The inertia of the rocket around the xz axis. Follows the standard
            input format of Dispersion Models.
        inertia_23 : int, float, tuple, list, optional
            The inertia of the rocket around the yz axis. Follows the standard
            input format of Dispersion Models.
        power_off_drag : list, optional
            The power off drag of the rocket. Follows the 1d array like input
            format of Dispersion Models.
        power_on_drag : list, optional
            The power on drag of the rocket. Follows the 1d array like input
            format of Dispersion Models.
        power_off_drag_factor : int, float, tuple, list, optional
            The power off drag factor of the rocket. Follows the factor input
            format of Dispersion Models.
        power_on_drag_factor : int, float, tuple, list, optional
            The power on drag factor of the rocket. Follows the standard input
            format of Dispersion Models.
        center_of_mass_without_motor : int, float, tuple, list, optional
            The center of mass of the rocket without the motor. Follows the
            standard input format of Dispersion Models.
        """
        self._validate_1d_array_like("power_off_drag", power_off_drag)
        self._validate_1d_array_like("power_on_drag", power_on_drag)
        super().__init__(
            object=rocket,
            radius=radius,
            mass=mass,
            I_11_without_motor=inertia_11,
            I_22_without_motor=inertia_22,
            I_33_without_motor=inertia_33,
            I_12_without_motor=inertia_12,
            I_13_without_motor=inertia_13,
            I_23_without_motor=inertia_23,
            power_off_drag=power_off_drag,
            power_on_drag=power_on_drag,
            power_off_drag_factor=power_off_drag_factor,
            power_on_drag_factor=power_on_drag_factor,
            center_of_mass_without_motor=center_of_mass_without_motor,
            coordinate_system_orientation=None,
        )
        self.motors = Components()
        self.aerodynamic_surfaces = Components()
        self.rail_buttons = Components()
        self.parachutes = []

    def __str__(self):
        # special str for rocket because of the components and parachutes
        s = ""
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # Skip attributes starting with underscore
            if isinstance(value, tuple):
                # Format the tuple as a string with the mean and standard deviation.
                value_str = f"{value[0]:.5f} ± {value[1]:.5f} (numpy.random.{value[2].__name__})"
                s += f"{key}: {value_str}\n"
            elif isinstance(value, Components):
                # Format the components as a string with the mean and standard deviation.
                s += f"{key}:\n"
                if len(value) == 0:
                    s += "\tNone\n"
                for component in value:
                    s += f"\t{component.component.__class__.__name__} "
                    if isinstance(component.position, tuple):
                        s += f"at position: {component.position[0]:.5f} ± "
                        s += f"{component.position[1]:.5f} "
                        s += f"(numpy.random.{component.position[2].__name__})\n"
                    elif isinstance(component.position, list):
                        s += f"at position: {component.position}\n"
                    else:
                        s += f"at position: {component.position:.5f}\n"
            else:
                # Otherwise, just use the default string representation of the value.
                value_str = str(value)
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (McParachute)):
                        value_str = ""
                        for parachute in value:
                            value_str += f"\n\t{parachute.name[0]} "
                s += f"{key}: {value_str}\n"
        return s.strip()

    def add_motor(self, motor, position=None):
        """Adds a monte carlo motor to the monte carlo rocket. If a motor is
        already present, it will be replaced.

        Parameters
        ----------
        motor : McMotor or Motor
            The motor to be added to the monte carlo rocket.
        position : tuple, list, int, float, optional
            The position of the motor. Follows the standard input format of
            Dispersion Models.
        """
        # checks if there is a motor already
        if len(self.motors) > 0:
            warnings.warn(
                "Only one motor can be added to the monte carlo rocket. "
                "The previous motor will be replaced."
            )
            self.motors = Components()

        # checks if input is a Motor
        if not isinstance(motor, (Motor, MotorDispersionModel)):
            raise AssertionError("`motor` must be a McMotor or Motor type")
        if isinstance(motor, Motor):
            # create McMotor
            # TODO check motor type when hybrids and liquids are implemented
            motor = McSolidMotor(solid_motor=motor)
        self.motors.add(motor, self._validate_position(motor, position))

    def _add_surfaces(self, surfaces, positions, type, monte_carlo_type, error_message):
        """Adds a monte carlo aerodynamic surface to the monte carlo rocket. If
        an aerodynamic surface is already present, it will be replaced.

        Parameters
        ----------
        surfaces : McAeroSurface or AeroSurface
            The aerodynamic surface to be added to the monte carlo rocket.
        positions : tuple, list, int, float, optional
            The position of the aerodynamic surface. Follows the standard input
            format of Dispersion Models.
        type : type
            The type of the aerodynamic surface to be added to the monte carlo
            rocket.
        monte_carlo_type : type
            The type of the monte carlo aerodynamic surface to be added to the
            monte carlo rocket.
        error_message : str
            The error message to be raised if the input is not of the correct
            type.
        """
        if not isinstance(surfaces, (type, monte_carlo_type)):
            raise AssertionError(error_message)
        if isinstance(surfaces, type):
            # create McSurfaces
            surfaces = monte_carlo_type(component=surfaces)
        self.aerodynamic_surfaces.add(
            surfaces, self._validate_position(surfaces, positions)
        )

    def add_nose(self, nose, position=None):
        """Adds a monte carlo nose cone to the monte carlo rocket.

        Parameters
        ----------
        nose : McNoseCone or NoseCone
            The nose cone to be added to the monte carlo rocket.
        position : tuple, list, int, float, optional
            The position of the nose cone. Follows the standard input format of
            Dispersion Models.
        """
        self._add_surfaces(
            nose,
            position,
            NoseCone,
            McNoseCone,
            "`nose` must be of NoseCone or McNoseCone type",
        )

    def add_trapezoidal_fins(self, fins, position=None):
        """Adds a monte carlo trapezoidal fins to the monte carlo rocket.

        Parameters
        ----------
        fins : McTrapezoidalFins or TrapezoidalFins
            The trapezoidal fins to be added to the monte carlo rocket.
        position : tuple, list, int, float, optional
            The position of the trapezoidal fins. Follows the standard input
            format of Dispersion Models.
        """
        self._add_surfaces(
            fins,
            position,
            TrapezoidalFins,
            McTrapezoidalFins,
            "`fins` must be of TrapezoidalFins or McTrapezoidalFins type",
        )

    def add_elliptical_fins(self, fins, position=None):
        """Adds a monte carlo elliptical fins to the monte carlo rocket.

        Parameters
        ----------
        fins : McEllipticalFins or EllipticalFins
            The elliptical fins to be added to the monte carlo rocket.
        position : tuple, list, int, float, optional
            The position of the elliptical fins. Follows the standard input
            format of Dispersion Models.
        """
        self._add_surfaces(
            fins,
            position,
            EllipticalFins,
            McEllipticalFins,
            "`fins` must be of EllipticalFins or McEllipticalFins type",
        )

    def add_tail(self, tail, position=None):
        """Adds a monte carlo tail to the monte carlo rocket.

        Parameters
        ----------
        tail : McTail or Tail
            The tail to be added to the monte carlo rocket.
        position : tuple, list, int, float, optional
            The position of the tail. Follows the standard input format of
            Dispersion Models.
        """
        self._add_surfaces(
            tail,
            position,
            Tail,
            McTail,
            "`tail` must be of Tail or McTail type",
        )

    def add_parachute(self, parachute):
        """Adds a monte carlo parachute to the monte carlo rocket.

        Parameters
        ----------
        parachute : McParachute or Parachute
            The parachute to be added to the monte carlo rocket.
        """
        # checks if input is a McParachute type
        if not isinstance(parachute, (Parachute, McParachute)):
            raise AssertionError("`parachute` must be of Parachute or McParachute type")
        if isinstance(parachute, Parachute):
            # create McParachute
            parachute = McParachute(parachute=parachute)
        self.parachutes.append(parachute)

    def set_rail_buttons(
        self,
        rail_buttons,
        lower_button_position=None,
    ):
        """Sets the rail buttons of the monte carlo rocket.

        Parameters
        ----------
        rail_buttons : McRailButtons or RailButtons
            The rail buttons to be added to the monte carlo rocket.
        lower_button_position : tuple, list, int, float, optional
            The position of the lower button. Follows the standard input format
            of Dispersion Models.
        """
        if not isinstance(rail_buttons, (McRailButtons, RailButtons)):
            raise AssertionError(
                "`rail_buttons` must be of RailButtons or McRailButtons type"
            )
        if isinstance(rail_buttons, RailButtons):
            # create McRailButtons
            rail_buttons = McRailButtons(rail_buttons=rail_buttons)
        self.rail_buttons.add(
            rail_buttons, self._validate_position(rail_buttons, lower_button_position)
        )

    def _validate_position(self, validated_object, position):
        """Validate the position argument.

        Parameters
        ----------
        validated_object : object
            The object to which the position argument refers to.
        position : tuple, list, int, float
            The position argument to be validated.

        Returns
        -------
        tuple or list
            Validated position argument.

        Raises
        ------
        ValueError
            If the position argument does not conform to the specified formats.
        """

        if isinstance(position, tuple):
            return self._validate_tuple(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        elif isinstance(position, (int, float)):
            return self._validate_scalar(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        elif isinstance(position, list):
            return self._validate_list(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        elif position is None:
            position = []
            return self._validate_list(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        else:
            raise AssertionError(f"`position` must be a tuple, list, int, or float")

    def _create_get_position(self, validated_object):
        """Create a function to get the nominal position from an object.

        Parameters
        ----------
        validated_object : object
            The object to which the position argument refers to.

        Returns
        -------
        function
            Function to get the nominal position from an object. The function
            must receive two arguments.
        """

        # try to get position from object
        error_msg = (
            "`position` standard deviation was provided but the rocket does "
            f"not have the same {validated_object.object.__class__.__name__} "
            "to get the nominal position value from."
        )
        # special case for motor dispersion model
        if isinstance(validated_object, (MotorDispersionModel)):
            if isinstance(self.object.motor, EmptyMotor):
                raise AssertionError(error_msg)

            def get_motor_position(self_object, _):
                return self_object.motor_position

            return get_motor_position
        else:
            if isinstance(validated_object, McRailButtons):

                def get_surface_position(self_object, _):
                    surfaces = self_object.rail_buttons.get_tuple_by_type(
                        validated_object.object.__class__
                    )
                    if len(surfaces) == 0:
                        raise AssertionError(error_msg)
                    for surface in surfaces:
                        if surface.component == validated_object.object:
                            return surface.position
                        else:
                            raise AssertionError(error_msg)

            else:

                def get_surface_position(self_object, _):
                    surfaces = self_object.aerodynamic_surfaces.get_tuple_by_type(
                        validated_object.object.__class__
                    )
                    if len(surfaces) == 0:
                        raise AssertionError(error_msg)
                    for surface in surfaces:
                        if surface.component == validated_object.object:
                            return surface.position
                        else:
                            raise AssertionError(error_msg)

            return get_surface_position

    def _randomize_position(self, position):
        """Randomize a position provided as a tuple or list."""
        if isinstance(position, tuple):
            return position[-1](position[0], position[1])
        elif isinstance(position, list):
            return choice(position) if position else position

    def create_object(self):
        """Creates and returns a Rocket object from the randomly generated input
        arguments.

        Returns
        -------
        rocket : Rocket
            Rocket object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        rocket = Rocket(
            radius=generated_dict["radius"],
            mass=generated_dict["mass"],
            inertia=(
                generated_dict["I_11_without_motor"],
                generated_dict["I_22_without_motor"],
                generated_dict["I_33_without_motor"],
                generated_dict["I_12_without_motor"],
                generated_dict["I_13_without_motor"],
                generated_dict["I_23_without_motor"],
            ),
            power_off_drag=generated_dict["power_off_drag"],
            power_on_drag=generated_dict["power_on_drag"],
            center_of_mass_without_motor=generated_dict["center_of_mass_without_motor"],
            coordinate_system_orientation=generated_dict[
                "coordinate_system_orientation"
            ],
        )
        rocket.power_off_drag *= generated_dict["power_off_drag_factor"]
        rocket.power_on_drag *= generated_dict["power_on_drag_factor"]

        for component_motor in self.motors:
            m = component_motor.component.create_object()
            position_rnd = self._randomize_position(component_motor.position)
            rocket.add_motor(m, position_rnd)

        for component_surface in self.aerodynamic_surfaces:
            s = component_surface.component.create_object()
            position_rnd = self._randomize_position(component_surface.position)
            rocket.add_surfaces(s, position_rnd)

        for component_rail_buttons in self.rail_buttons:
            r = component_rail_buttons.component.create_object()
            lower_button_position_rnd = self._randomize_position(
                component_rail_buttons.position
            )
            upper_button_position_rnd = r.buttons_distance + lower_button_position_rnd
            rocket.set_rail_buttons(
                upper_button_position=upper_button_position_rnd,
                lower_button_position=lower_button_position_rnd,
                angular_position=r.angular_position,
            )

        for parachute in self.parachutes:
            p = parachute.create_object()
            rocket.add_parachute(
                name=p.name,
                cd_s=p.cd_s,
                trigger=p.trigger,
                sampling_rate=p.sampling_rate,
                lag=p.lag,
                noise=p.noise,
            )

        return rocket
