import logging

from rocketpy.rocket.aero_surface.generic_surface import GenericSurface

logger = logging.getLogger(__name__)


class _RocketPrints:
    """Class that holds prints methods for Rocket class.

    Attributes
    ----------
    _RocketPrints.rocket : rocket
        Rocket object that will be used for the prints.

    """

    def __init__(self, rocket):
        """Initializes _EnvironmentPrints class

        Parameters
        ----------
        rocket: rocketpy.rocket
            Instance of the rocket class.

        Returns
        -------
        None
        """
        self.rocket = rocket

    def inertia_details(self):
        """Print inertia details.

        Returns
        -------
        None
        """
        logger.info("\nInertia Details\n")
        logger.info(f"Rocket Mass: {self.rocket.mass:.3f} kg (without motor)")
        logger.info(f"Rocket Dry Mass: {self.rocket.dry_mass:.3f} kg (with unloaded motor)")
        logger.info(f"Rocket Loaded Mass: {self.rocket.total_mass(0):.3f} kg")
        logger.info(f"Rocket Structural Mass Ratio: {self.rocket.structural_mass_ratio:.3f}")
        logger.info(
            f"Rocket Inertia (with unloaded motor) 11: {self.rocket.dry_I_11:.3f} kg*m2"
        )
        logger.info(
            f"Rocket Inertia (with unloaded motor) 22: {self.rocket.dry_I_22:.3f} kg*m2"
        )
        logger.info(
            f"Rocket Inertia (with unloaded motor) 33: {self.rocket.dry_I_33:.3f} kg*m2"
        )
        logger.info(
            f"Rocket Inertia (with unloaded motor) 12: {self.rocket.dry_I_12:.3f} kg*m2"
        )
        logger.info(
            f"Rocket Inertia (with unloaded motor) 13: {self.rocket.dry_I_13:.3f} kg*m2"
        )
        logger.info(
            f"Rocket Inertia (with unloaded motor) 23: {self.rocket.dry_I_23:.3f} kg*m2"
        )

    def rocket_geometrical_parameters(self):
        """Print rocket geometrical parameters.

        Returns
        -------
        None
        """
        logger.info("\nGeometrical Parameters\n")
        logger.info(f"Rocket Maximum Radius: {self.rocket.radius} m")
        logger.info(f"Rocket Frontal Area: {self.rocket.area:.6f} m2")

        logger.info("\nRocket Distances")
        distance = abs(
            self.rocket.center_of_mass_without_motor
            - self.rocket.center_of_dry_mass_position
        )
        logger.info(
            "Rocket Center of Dry Mass - Center of Mass without Motor: "
            f"{distance:.3f} m"
        )
        distance = abs(
            self.rocket.center_of_dry_mass_position - self.rocket.nozzle_position
        )
        logger.info(f"Rocket Center of Dry Mass - Nozzle Exit: {distance:.3f} m")
        distance = abs(
            self.rocket.center_of_propellant_position(0)
            - self.rocket.center_of_dry_mass_position
        )
        logger.info(
            f"Rocket Center of Dry Mass - Center of Propellant Mass: {distance:.3f} m"
        )
        distance = abs(
            self.rocket.center_of_mass(0) - self.rocket.center_of_dry_mass_position
        )
        logger.info(
            f"Rocket Center of Mass - Rocket Loaded Center of Mass: {distance:.3f} m\n"
        )

    def rocket_aerodynamics_quantities(self):
        """Print rocket aerodynamics quantities.

        Returns
        -------
        None
        """
        logger.info("\nAerodynamics Lift Coefficient Derivatives\n")
        for surface, _ in self.rocket.aerodynamic_surfaces:
            if isinstance(surface, GenericSurface):
                continue
            name = surface.name
            # ref_factor corrects lift for different reference areas
            ref_factor = (surface.rocket_radius / self.rocket.radius) ** 2
            logger.info(
                f"{name} Lift Coefficient Derivative: "
                f"{ref_factor * surface.clalpha(0):.3f}/rad"
            )

        logger.info("\nCenter of Pressure\n")
        for surface, position in self.rocket.aerodynamic_surfaces:
            name = surface.name
            cpz = surface.cp[2]  # relative to the user defined coordinate system
            logger.info(
                f"{name} Center of Pressure position: "
                f"{position.z - self.rocket._csys * cpz:.3f} m"
            )
        logger.info("\nStability\n")
        logger.info(
            f"Center of Mass position (time=0): {self.rocket.center_of_mass(0):.3f} m"
        )
        logger.info(
            f"Center of Pressure position (time=0): {self.rocket.cp_position(0):.3f} m"
        )
        logger.info(
            f"Initial Static Margin (mach=0, time=0): "
            f"{self.rocket.static_margin(0):.3f} c"
        )
        logger.info(
            f"Final Static Margin (mach=0, time=burn_out): "
            f"{self.rocket.static_margin(self.rocket.motor.burn_out_time):.3f} c"
        )
        logger.info(
            f"Rocket Center of Mass (time=0) - Center of Pressure (mach=0): "
            f"{abs(self.rocket.center_of_mass(0) - self.rocket.cp_position(0)):.3f} m\n"
        )

    def parachute_data(self):
        """Print parachute data.

        Returns
        -------
        None
        """
        for chute in self.rocket.parachutes:
            chute.all_info()

    def all(self):
        """Prints all print methods about the Environment.

        Returns
        -------
        None
        """
        self.inertia_details()
        self.rocket_geometrical_parameters()
        self.rocket_aerodynamics_quantities()
        self.parachute_data()
