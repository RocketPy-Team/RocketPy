__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


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

        pass

    def inertia_details(self):
        """Print inertia details.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nInertia Details\n")
        print("Rocket Mass: {:.3f} kg (No Propellant)".format(self.rocket.mass))
        print(
            "Rocket Mass: {:.3f} kg (With Propellant)".format(self.rocket.total_mass(0))
        )
        print(
            "Rocket Inertia (with motor, but without propellant) 11: {:.3f} kg*m2".format(
                self.rocket.dry_I_11
            )
        )
        print(
            "Rocket Inertia (with motor, but without propellant) 22: {:.3f} kg*m2".format(
                self.rocket.dry_I_22
            )
        )
        print(
            "Rocket Inertia (with motor, but without propellant) 33: {:.3f} kg*m2".format(
                self.rocket.dry_I_33
            )
        )
        print(
            "Rocket Inertia (with motor, but without propellant) 12: {:.3f} kg*m2".format(
                self.rocket.dry_I_12
            )
        )
        print(
            "Rocket Inertia (with motor, but without propellant) 13: {:.3f} kg*m2".format(
                self.rocket.dry_I_13
            )
        )
        print(
            "Rocket Inertia (with motor, but without propellant) 23: {:.3f} kg*m2".format(
                self.rocket.dry_I_23
            )
        )

        return None

    def rocket_geometrical_parameters(self):
        """Print rocket geometrical parameters.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nGeometrical Parameters\n")
        print("Rocket Maximum Radius: " + str(self.rocket.radius) + " m")
        print("Rocket Frontal Area: " + "{:.6f}".format(self.rocket.area) + " m2")
        print("\nRocket Distances")
        print(
            "Rocket Center of Dry Mass - Nozzle Exit Distance: "
            + "{:.3f} m".format(
                abs(
                    self.rocket.center_of_dry_mass_position - self.rocket.motor_position
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

        print("\nAerodynamic Components Parameters")
        print("Currently not implemented.")

        return None

    def rocket_aerodynamics_quantities(self):
        """Print rocket aerodynamics quantities.

        Parameters
        ----------
        None

        Return
        ------
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
            print(name + " Center of Pressure to CM: {:.3f}".format(cpz) + " m")
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

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for chute in self.rocket.parachutes:
            chute.all_info()
        return None

    def all(self):
        """Prints all print methods about the Environment.

        Parameters
        ----------
        None

        Return
        ------
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
