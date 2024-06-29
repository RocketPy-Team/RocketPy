"""Defines the StochasticSolidMotor class."""

from rocketpy.motors import SolidMotor

from .stochastic_motor_model import StochasticMotorModel


class StochasticSolidMotor(StochasticMotorModel):
    """A Stochastic Solid Motor class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`SolidMotor <rocketpy.motors.SolidMotor>`

    Attributes
    ----------
    object : SolidMotor
        SolidMotor object to be used for validation.
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
    nozzle_radius : int, float, tuple, list
        Nozzle radius of the motor in meters.
    grain_number : list[int]
        Number of grains in the motor. This attribute can not be randomized.
    grain_density : int, float, tuple, list
        Density of the grains in the motor in kilograms per meters cubed.
    grain_outer_radius : int, float, tuple, list
        Outer radius of the grains in the motor in meters.
    grain_initial_inner_radius : int, float, tuple, list
        Initial inner radius of the grains in the motor in meters.
    grain_initial_height : int, float, tuple, list
        Initial height of the grains in the motor in meters.
    grain_separation : int, float, tuple, list
        Separation between grains in the motor in meters.
    grains_center_of_mass_position : int, float, tuple, list
        Position of the center of mass of the grains in the motor in
        meters.
    center_of_dry_mass_position : int, float, tuple, list
        Position of the center of mass of the dry mass in the motor in
        meters.
    nozzle_position : int, float, tuple, list
        Position of the nozzle in the motor in meters.
    throat_radius : int, float, tuple, list
        Radius of the throat in the motor in meters.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        solid_motor,
        thrust_source=None,
        total_impulse=None,
        burn_start_time=None,
        burn_out_time=None,
        dry_mass=None,
        dry_inertia_11=None,
        dry_inertia_22=None,
        dry_inertia_33=None,
        dry_inertia_12=None,
        dry_inertia_13=None,
        dry_inertia_23=None,
        nozzle_radius=None,
        grain_number=None,  # TODO: can't vary the number of grains.
        grain_density=None,
        grain_outer_radius=None,
        grain_initial_inner_radius=None,
        grain_initial_height=None,
        grain_separation=None,
        grains_center_of_mass_position=None,
        center_of_dry_mass_position=None,
        nozzle_position=None,
        throat_radius=None,
    ):
        """Initializes the Stochastic Solid Motor class.

        See Also
        --------
        :ref:`stochastic_model` and :class:`SolidMotor <rocketpy.motors.SolidMotor>`

        Parameters
        ----------
        solid_motor : SolidMotor
            SolidMotor object to be used for validation.
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
        nozzle_radius : int, float, tuple, list, optional
            Nozzle radius of the motor in meters.
        grain_density : int, float, tuple, list, optional
            Density of the grains in the motor in kilograms per meters cubed.
        grain_outer_radius : int, float, tuple, list, optional
            Outer radius of the grains in the motor in meters.
        grain_initial_inner_radius : int, float, tuple, list, optional
            Initial inner radius of the grains in the motor in meters.
        grain_initial_height : int, float, tuple, list, optional
            Initial height of the grains in the motor in meters.
        grain_separation : int, float, tuple, list, optional
            Separation between grains in the motor in meters.
        grains_center_of_mass_position : int, float, tuple, list, optional
            Position of the center of mass of the grains in the motor in
            meters.
        center_of_dry_mass_position : int, float, tuple, list, optional
            Position of the center of mass of the dry mass in the motor in
            meters.
        nozzle_position : int, float, tuple, list, optional
            Position of the nozzle in the motor in meters.
        throat_radius : int, float, tuple, list, optional
            Radius of the throat in the motor in meters.
        """
        super().__init__(
            solid_motor,
            thrust_source=thrust_source,
            total_impulse=total_impulse,
            burn_start_time=burn_start_time,
            burn_out_time=burn_out_time,
            dry_mass=dry_mass,
            dry_I_11=dry_inertia_11,
            dry_I_22=dry_inertia_22,
            dry_I_33=dry_inertia_33,
            dry_I_12=dry_inertia_12,
            dry_I_13=dry_inertia_13,
            dry_I_23=dry_inertia_23,
            nozzle_radius=nozzle_radius,
            grain_number=grain_number,
            grain_density=grain_density,
            grain_outer_radius=grain_outer_radius,
            grain_initial_inner_radius=grain_initial_inner_radius,
            grain_initial_height=grain_initial_height,
            grain_separation=grain_separation,
            grains_center_of_mass_position=grains_center_of_mass_position,
            center_of_dry_mass_position=center_of_dry_mass_position,
            nozzle_position=nozzle_position,
            throat_radius=throat_radius,
            interpolate=None,
            coordinate_system_orientation=None,
        )

    def create_object(self):
        """Creates and returns a SolidMotor object from the randomly generated
        input arguments.

        Returns
        -------
        SolidMotor
            SolidMotor object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        return SolidMotor(
            thrust_source=generated_dict["thrust_source"],
            dry_mass=generated_dict["dry_mass"],
            dry_inertia=(
                generated_dict["dry_I_11"],
                generated_dict["dry_I_22"],
                generated_dict["dry_I_33"],
                generated_dict["dry_I_12"],
                generated_dict["dry_I_13"],
                generated_dict["dry_I_23"],
            ),
            nozzle_radius=generated_dict["nozzle_radius"],
            grain_number=generated_dict["grain_number"],
            grain_density=generated_dict["grain_density"],
            grain_outer_radius=generated_dict["grain_outer_radius"],
            grain_initial_inner_radius=generated_dict["grain_initial_inner_radius"],
            grain_initial_height=generated_dict["grain_initial_height"],
            grain_separation=generated_dict["grain_separation"],
            grains_center_of_mass_position=generated_dict[
                "grains_center_of_mass_position"
            ],
            center_of_dry_mass_position=generated_dict["center_of_dry_mass_position"],
            nozzle_position=generated_dict["nozzle_position"],
            burn_time=(
                generated_dict["burn_start_time"],
                generated_dict["burn_out_time"],
            ),
            throat_radius=generated_dict["throat_radius"],
            reshape_thrust_curve=(
                (generated_dict["burn_start_time"], generated_dict["burn_out_time"]),
                generated_dict["total_impulse"],
            ),
            coordinate_system_orientation=generated_dict[
                "coordinate_system_orientation"
            ],
            interpolation_method=generated_dict["interpolate"],
        )
