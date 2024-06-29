"""Defines the StochasticGenericMotor class."""

from rocketpy.motors import GenericMotor

from .stochastic_motor_model import StochasticMotorModel


# pylint: disable=too-many-arguments
class StochasticGenericMotor(StochasticMotorModel):
    """A Stochastic Generic Motor class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and
    :class:`GenericMotor <rocketpy.motors.GenericMotor>`

    Attributes
    ----------
    object : GenericMotor
        GenericMotor object to be used as a base for the stochastic model.
    thrust_source : list
        List of strings representing the thrust source to be selected.
    total_impulse : int, float, tuple, list
        Total impulse of the motor in newton seconds.
    burn_start_time : int, float, tuple, list
        Burn start time of the motor in seconds.
    burn_out_time : int, float, tuple, list
        Burn out time of the motor in seconds.
    dry_mass : int, float, tuple, list
        Dry mass of the motor in kilograms.
    dry_I_11 : int, float, tuple, list
        Dry inertia of the motor in kilograms times meters squared.
    dry_I_22 : int, float, tuple, list
        Dry inertia of the motor in kilograms times meters squared.
    dry_I_33 : int, float, tuple, list
        Dry inertia of the motor in kilograms times meters squared.
    dry_I_12 : int, float, tuple, list
        Dry inertia of the motor in kilograms times meters squared.
    dry_I_13 : int, float, tuple, list
        Dry inertia of the motor in kilograms times meters squared.
    dry_I_23 : int, float, tuple, list
        Dry inertia of the motor in kilograms times meters squared.
    chamber_radius : int, float, tuple, list
        Chamber radius of the motor in meters.
    chamber_height : int, float, tuple, list
        Chamber height of the motor in meters.
    chamber_position : int, float, tuple, list
        Chamber position of the motor in meters.
    nozzle_radius : int, float, tuple, list
        Nozzle radius of the motor in meters.
    nozzle_position : int, float, tuple, list
        Nozzle position of the motor in meters.
    center_of_dry_mass_position : int, float, tuple, list
        Center of dry mass position of the motor in meters.
    interpolation_method : str
        Interpolation method to be used. This attribute can not be randomized.
    coordinate_system_orientation : str
        Coordinate system orientation to be used. This attribute can not be
        randomized.
    """

    def __init__(
        self,
        generic_motor,
        thrust_source=None,
        total_impulse=None,
        burn_start_time=None,
        burn_out_time=None,
        propellant_initial_mass=None,
        dry_mass=None,
        dry_inertia_11=None,
        dry_inertia_22=None,
        dry_inertia_33=None,
        dry_inertia_12=None,
        dry_inertia_13=None,
        dry_inertia_23=None,
        chamber_radius=None,
        chamber_height=None,
        chamber_position=None,
        nozzle_radius=None,
        nozzle_position=None,
        center_of_dry_mass_position=None,
    ):
        """Initializes the Stochastic Generic Motor class.

        See Also
        --------
        :ref:`stochastic_model` and
        :class:`GenericMotor <rocketpy.motors.GenericMotor>`

        Parameters
        ----------
        generic_motor : GenericMotor
            GenericMotor object to be used for validation.
        thrust_source : list, optional
            List of strings representing the thrust source to be selected.
        total_impulse : int, float, tuple, list, optional
            Total impulse of the motor in newton seconds.
        burn_start_time : int, float, tuple, list, optional
            Burn start time of the motor in seconds.
        burn_out_time : int, float, tuple, list, optional
            Burn out time of the motor in seconds.
        dry_mass : int, float, tuple, list, optional
            Dry mass of the motor in kilograms.
        dry_I_11 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared.
        dry_I_22 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared.
        dry_I_33 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared.
        dry_I_12 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared.
        dry_I_13 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared.
        dry_I_23 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared.
        chamber_radius : int, float, tuple, list, optional
            Chamber radius of the motor in meters.
        chamber_height : int, float, tuple, list, optional
            Chamber height of the motor in meters.
        chamber_position : int, float, tuple, list, optional
            Position of the motor chamber in meters.
        nozzle_radius : int, float, tuple, list, optional
            Nozzle radius of the motor in meters.
        nozzle_position : int, float, tuple, list, optional
            Nozzle position of the motor in meters.
        center_of_dry_mass_position : int, float, tuple, list, optional
            Center of dry mass position of the motor in meters.
        """
        super().__init__(
            generic_motor,
            thrust_source=thrust_source,
            total_impulse=total_impulse,
            burn_start_time=burn_start_time,
            burn_out_time=burn_out_time,
            propellant_initial_mass=propellant_initial_mass,
            dry_mass=dry_mass,
            dry_I_11=dry_inertia_11,
            dry_I_22=dry_inertia_22,
            dry_I_33=dry_inertia_33,
            dry_I_12=dry_inertia_12,
            dry_I_13=dry_inertia_13,
            dry_I_23=dry_inertia_23,
            chamber_radius=chamber_radius,
            chamber_height=chamber_height,
            chamber_position=chamber_position,
            nozzle_radius=nozzle_radius,
            nozzle_position=nozzle_position,
            center_of_dry_mass_position=center_of_dry_mass_position,
            interpolate=None,
            coordinate_system_orientation=None,
        )

    def create_object(self):
        """Creates a `GenericMotor` object from the randomly generated input
        arguments.

        Returns
        -------
        GenericMotor
            GenericMotor object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        return GenericMotor(
            thrust_source=generated_dict["thrust_source"],
            burn_time=(
                generated_dict["burn_start_time"],
                generated_dict["burn_out_time"],
            ),
            propellant_initial_mass=generated_dict["propellant_initial_mass"],
            dry_mass=generated_dict["dry_mass"],
            dry_inertia=(
                generated_dict["dry_I_11"],
                generated_dict["dry_I_22"],
                generated_dict["dry_I_33"],
                generated_dict["dry_I_12"],
                generated_dict["dry_I_13"],
                generated_dict["dry_I_23"],
            ),
            chamber_radius=generated_dict["chamber_radius"],
            chamber_height=generated_dict["chamber_height"],
            chamber_position=generated_dict["chamber_position"],
            nozzle_radius=generated_dict["nozzle_radius"],
            nozzle_position=generated_dict["nozzle_position"],
            center_of_dry_mass_position=generated_dict["center_of_dry_mass_position"],
            reshape_thrust_curve=(
                (generated_dict["burn_start_time"], generated_dict["burn_out_time"]),
                generated_dict["total_impulse"],
            ),
            coordinate_system_orientation=generated_dict[
                "coordinate_system_orientation"
            ],
            interpolation_method=generated_dict["interpolate"],
        )
