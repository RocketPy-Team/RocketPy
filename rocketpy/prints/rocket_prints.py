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
        print("\nInertia Details\n")
        print("Rocket Mass: {self.rocket.mass:.3f} kg")
        print(f"Rocket Dry Mass: {self.rocket.dry_mass:.3f} kg (with unloaded motor)")
        print(f"Rocket Mass: {self.rocket.total_mass(0):.3f} kg (With Propellant)")
        print(
            f"Rocket Inertia (with unloaded motor) 11: {self.rocket.dry_I_11:.3f} kg*m2"
        )
        print(
            f"Rocket Inertia (with unloaded motor) 22: {self.rocket.dry_I_22:.3f} kg*m2"
        )
        print(
            f"Rocket Inertia (with unloaded motor) 33: {self.rocket.dry_I_33:.3f} kg*m2"
        )
        print(
            f"Rocket Inertia (with unloaded motor) 12: {self.rocket.dry_I_12:.3f} kg*m2"
        )
        print(
            f"Rocket Inertia (with unloaded motor) 13: {self.rocket.dry_I_13:.3f} kg*m2"
        )
        print(
            f"Rocket Inertia (with unloaded motor) 23: {self.rocket.dry_I_23:.3f} kg*m2"
        )

    def rocket_geometrical_parameters(self):
        """Print rocket geometrical parameters.

        Returns
        -------
        None
        """
        print("\nGeometrical Parameters\n")
        print("Rocket Maximum Radius: " + str(self.rocket.radius) + " m")
        print("Rocket Frontal Area: " + "{:.6f}".format(self.rocket.area) + " m2")
        print("\nRocket Distances")
        print(
            "Rocket Center of Dry Mass - Center of Mass withour Motor: "
            + "{:.3f} m".format(
                abs(
                    self.rocket.center_of_mass_without_motor
                    - self.rocket.center_of_dry_mass_position
                )
            )
        )
        print(
            "Rocket Center of Dry Mass - Nozzle Exit Distance: "
            + "{:.3f} m".format(
                abs(
                    self.rocket.center_of_dry_mass_position
                    - self.rocket.nozzle_position
                )
            )
        )
        print(
            "Rocket Center of Dry Mass - Center of Propellant Mass: "
            + "{:.3f} m".format(
                abs(
                    self.rocket.center_of_propellant_position(0)
                    - self.rocket.center_of_dry_mass_position
                )
            )
        )
        print(
            "Rocket Center of Mass - Rocket Loaded Center of Mass: "
            + "{:.3f} m".format(
                abs(
                    self.rocket.center_of_mass(0)
                    - self.rocket.center_of_dry_mass_position
                )
            )
        )

        return None

    def rocket_aerodynamics_quantities(self):
        """Print rocket aerodynamics quantities.

        Returns
        -------
        None
        """
        print("\nAerodynamics Lift Coefficient Derivatives\n")
        for surface, position in self.rocket.aerodynamic_surfaces:
            name = surface.name
            print(
                name
                + " Lift Coefficient Derivative: {:.3f}".format(surface.clalpha(0))
                + "/rad"
            )

        print("\nAerodynamics Center of Pressure\n")
        for surface, position in self.rocket.aerodynamic_surfaces:
            name = surface.name
            cpz = surface.cp[2]
            print(
                name
                + " Center of Pressure to CM: {:.3f}".format(
                    position - self.rocket._csys * cpz
                )
                + " m"
            )
        print(
            "Distance - Center of Pressure to Center of Dry Mass: "
            + "{:.3f}".format(self.rocket.center_of_mass(0) - self.rocket.cp_position)
            + " m"
        )
        print(
            "Initial Static Margin: "
            + "{:.3f}".format(self.rocket.static_margin(0))
            + " c"
        )
        print(
            "Final Static Margin: "
            + "{:.3f}".format(
                self.rocket.static_margin(self.rocket.motor.burn_out_time)
            )
            + " c"
        )

        return None

    def parachute_data(self):
        """Print parachute data.

        Returns
        -------
        None
        """
        for chute in self.rocket.parachutes:
            chute.all_info()
        return None

    def all(self):
        """Prints all print methods about the Environment.

        Returns
        -------
        None
        """
        # Print inertia details
        self.inertia_details()
        print()

        # Print rocket geometrical parameters
        self.rocket_geometrical_parameters()
        print()

        # Print rocket aerodynamics quantities
        self.rocket_aerodynamics_quantities()
        print()

        # Print parachute data
        self.parachute_data()
        print()

        return None
