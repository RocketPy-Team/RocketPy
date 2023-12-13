from rocketpy.motors import SolidMotor

from .motor_dispersion_model import MotorDispersionModel


class McSolidMotor(MotorDispersionModel):
    """Monte Carlo Solid Motor class, used to validate the input parameters of
    the solid motor, based on the pydantic library. It uses the DispersionModel
    class as a base class, see its documentation for more information. The
    inputs defined here correspond to the ones defined in the SolidMotor class.
    """

    # TODO:
    # - separated dry_inertias?
    # - coordinate system is not varied

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
        grain_number=None,
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
        """Initializes the Monte Carlo Solid Motor class.

        See Also
        --------
        This should link to somewhere that explains how inputs works in
        dispersion models.

        Parameters
        ----------
        solid_motor : SolidMotor
            SolidMotor object to be used for validation.
        thrust_source : list, optional
            List of strings representing the thrust source to be selected.
        total_impulse : int, float, tuple, list, optional
            Total impulse of the motor in newton seconds. Follows the standard
            input format of Dispersion Models.
        burn_time : int, float, tuple, list, optional
            Burn time of the motor in seconds. Follows the standard input
            format of Dispersion Models.
        dry_mass : int, float, tuple, list, optional
            Dry mass of the motor in kilograms. Follows the standard input
            format of Dispersion Models.
        dry_I_11 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared. Follows
            the standard input format of Dispersion Models.
        dry_I_22 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared. Follows
            the standard input format of Dispersion Models.
        dry_I_33 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared. Follows
            the standard input format of Dispersion Models.
        dry_I_12 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared. Follows
            the standard input format of Dispersion Models.
        dry_I_13 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared. Follows
            the standard input format of Dispersion Models.
        dry_I_23 : int, float, tuple, list, optional
            Dry inertia of the motor in kilograms times meters squared. Follows
            the standard input format of Dispersion Models.
        nozzle_radius : int, float, tuple, list, optional
            Nozzle radius of the motor in meters. Follows the standard input
            format of Dispersion Models.
        grain_number : int, float, tuple, list, optional
            Number of grains in the motor. Follows the standard input format of
            Dispersion Models.
        grain_density : int, float, tuple, list, optional
            Density of the grains in the motor in kilograms per meters cubed.
            Follows the standard input format of Dispersion Models.
        grain_outer_radius : int, float, tuple, list, optional
            Outer radius of the grains in the motor in meters. Follows the
            standard input format of Dispersion Models.
        grain_initial_inner_radius : int, float, tuple, list, optional
            Initial inner radius of the grains in the motor in meters. Follows
            the standard input format of Dispersion Models.
        grain_initial_height : int, float, tuple, list, optional
            Initial height of the grains in the motor in meters. Follows the
            standard input format of Dispersion Models.
        grain_separation : int, float, tuple, list, optional
            Separation between grains in the motor in meters. Follows the
            standard input format of Dispersion Models.
        grains_center_of_mass_position : int, float, tuple, list, optional
            Position of the center of mass of the grains in the motor in
            meters. Follows the standard input format of Dispersion Models.
        center_of_dry_mass_position : int, float, tuple, list, optional
            Position of the center of mass of the dry mass in the motor in
            meters. Follows the standard input format of Dispersion Models.
        nozzle_position : int, float, tuple, list, optional
            Position of the nozzle in the motor in meters. Follows the
            standard input format of Dispersion Models.
        throat_radius : int, float, tuple, list, optional
            Radius of the throat in the motor in meters. Follows the standard
            input format of Dispersion Models.
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
        )

    def create_object(self):
        """Create a randomized SolidMotor object based on the input parameters.

        Returns
        -------
        SolidMotor
            SolidMotor object with random input parameters.
        """
        generated_dict = next(self.dict_generator())
        obj = SolidMotor(
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
        )
        if "position" in generated_dict:
            obj.position = generated_dict["position"]
        return obj
