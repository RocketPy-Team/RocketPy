__author__ = " "
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _RocketPrints:
    """Class that holds prints methods for Rocket class.

    Attributes
    ----------
    _RocketPrints.environment : rocket
        Rocket object that will be used for the prints.

    """

    def __init__(self, rocket) -> None:
        """Initializes _EnvironmentPrints class

        Parameters
        ----------
        environment: Environment
            Instance of the Environment class.

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
            "Rocket Mass: {:.3f} kg (With Propellant)".format(self.rocket.totalMass(0))
        )
        print("Rocket Inertia I: {:.3f} kg*m2".format(self.rocket.inertiaI))
        print("Rocket Inertia Z: {:.3f} kg*m2".format(self.rocket.inertiaZ))

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
                abs(self.rocket.centerOfDryMassPosition - self.rocket.motorPosition)
            )
        )
        print(
            "Rocket Center of Dry Mass - Center of Propellant Mass: "
            + "{:.3f} m".format(
                abs(
                    self.rocket.centerOfPropellantPosition(0)
                    - self.rocket.centerOfDryMassPosition
                )
            )
        )
        print(
            "Rocket Center of Mass - Rocket Loaded Center of Mass: "
            + "{:.3f} m".format(
                abs(self.rocket.centerOfMass(0) - self.rocket.centerOfDryMassPosition)
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
        for aerodynamicSurface in self.rocket.aerodynamicSurfaces:
            name = aerodynamicSurface.name
            try:
                print(
                    name
                    + " Lift Coefficient Derivative: {:.3f}".format(
                        aerodynamicSurface.clalpha(0)
                    )
                    + "/rad"
                )
            except:
                print(
                    name
                    + " Lift Coefficient Derivative: {:.3f}".format(
                        aerodynamicSurface.clalpha
                    )
                    + "/rad"
                )

        print("\nAerodynamics Center of Pressure\n")
        for aerodynamicSurface in self.rocket.aerodynamicSurfaces:
            name = aerodynamicSurface.name
            cpz = aerodynamicSurface.cp[2]
            print(name + " Center of Pressure to CM: {:.3f}".format(cpz) + " m")
        print(
            "Distance - Center of Pressure to CM: "
            + "{:.3f}".format(self.rocket.cpPosition)
            + " m"
        )
        print(
            "Initial Static Margin: "
            + "{:.3f}".format(self.rocket.staticMargin(0))
            + " c"
        )
        print(
            "Final Static Margin: "
            + "{:.3f}".format(self.rocket.staticMargin(self.rocket.motor.burnOutTime))
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
            print("\n" + chute.name.title() + " Parachute\n")
            print("CdS Coefficient: " + str(chute.CdS) + " m2")
            if chute.trigger.__name__ == "<lambda>":
                line = self.rocket.getsourcelines(chute.trigger)[0][0]
                print(
                    "Ejection signal trigger: "
                    + line.split("lambda ")[1].split(",")[0].split("\n")[0]
                )
            else:
                print("Ejection signal trigger: " + chute.trigger.__name__)
            print("Ejection system refresh rate: " + str(chute.samplingRate) + " Hz.")
            print(
                "Time between ejection signal is triggered and the "
                "parachute is fully opened: " + str(chute.lag) + " s"
            )
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
