__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _SolidMotorPrints:
    """Class that holds prints methods for SolidMotor class.

    Attributes
    ----------
    _SolidMotorPrints.solid_motor : solid_motor
        SolidMotor object that will be used for the prints.

    """

    def __init__(
        self,
        solid_motor,
    ):
        """Initializes _SolidMotorPrints class

        Parameters
        ----------
        solid_motor: SolidMotor
            Instance of the SolidMotor class.

        Returns
        -------
        None
        """
        self.solid_motor = solid_motor
        return None

    def nozzle_details(self):
        """Prints out all data available about the SolidMotor nozzle.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print nozzle details
        print("\nNozzle Details\n")
        print("Nozzle Radius: " + str(self.solid_motor.nozzleRadius) + " m")
        print("Nozzle Throat Radius: " + str(self.solid_motor.throatRadius) + " m")

    def grain_details(self):
        """Prints out all data available about the SolidMotor grain.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Print grain details
        print("\nGrain Details\n")
        print("Number of Grains: " + str(self.solid_motor.grainNumber))
        print("Grain Spacing: " + str(self.solid_motor.grainSeparation) + " m")
        print("Grain Density: " + str(self.solid_motor.grainDensity) + " kg/m3")
        print("Grain Outer Radius: " + str(self.solid_motor.grainOuterRadius) + " m")
        print(
            "Grain Inner Radius: "
            + str(self.solid_motor.grainInitialInnerRadius)
            + " m"
        )
        print("Grain Height: " + str(self.solid_motor.grainInitialHeight) + " m")
        print(
            "Grain Volume: "
            + "{:.3f}".format(self.solid_motor.grainInitialVolume)
            + " m3"
        )
        print(
            "Grain Mass: " + "{:.3f}".format(self.solid_motor.grainInitialMass) + " kg"
        )

    def motor_details(self):
        """Prints out all data available about the SolidMotor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Print motor details
        print("\nMotor Details\n")
        print("Total Burning Time: " + str(self.solid_motor.burnDuration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.solid_motor.propellantInitialMass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.solid_motor.exhaustVelocity.average(*self.solid_motor.burn_time)
            )
            + " m/s"
        )
        print(
            "Average Thrust: " + "{:.3f}".format(self.solid_motor.averageThrust) + " N"
        )
        print(
            "Maximum Thrust: "
            + str(self.solid_motor.maxThrust)
            + " N at "
            + str(self.solid_motor.maxThrustTime)
            + " s after ignition."
        )
        print(
            "Total Impulse: " + "{:.3f}".format(self.solid_motor.totalImpulse) + " Ns"
        )

    def all(self):
        """Prints out all data available about the SolidMotor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        self.nozzle_details()
        print()

        self.grain_details()
        print()

        self.motor_details()
        print()

        return None
