__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _HybridMotorPrints:
    """Class that holds prints methods for HybridMotor class.

    Attributes
    ----------
    _HybridMotorPrints.hybrid_motor : hybrid_motor
        HybridMotor object that will be used for the prints.

    """

    def __init__(
        self,
        hybrid_motor,
    ):
        """Initializes _HybridMotorPrints class

        Parameters
        ----------
        hybrid_motor: HybridMotor
            Instance of the HybridMotor class.

        Returns
        -------
        None
        """
        self.hybrid_motor = hybrid_motor
        return None

    def nozzle_details(self):
        """Prints out all data available about the Nozzle.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print nozzle details
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.hybrid_motor.nozzleRadius) + " m")
        print(
            "Nozzle Throat Radius: " + str(self.hybrid_motor.solid.throatRadius) + " m"
        )

    def grain_details(self):
        """Prints out all data available about the Grain.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.hybrid_motor.solid.grainNumber))
        print("Grain Spacing: " + str(self.hybrid_motor.solid.grainSeparation) + " m")
        print("Grain Density: " + str(self.hybrid_motor.solid.grainDensity) + " kg/m3")
        print(
            "Grain Outer Radius: "
            + str(self.hybrid_motor.solid.grainOuterRadius)
            + " m"
        )
        print(
            "Grain Inner Radius: "
            + str(self.hybrid_motor.solid.grainInitialInnerRadius)
            + " m"
        )
        print("Grain Height: " + str(self.hybrid_motor.solid.grainInitialHeight) + " m")
        print(
            "Grain Volume: "
            + "{:.3f}".format(self.hybrid_motor.solid.grainInitialVolume)
            + " m3"
        )
        print(
            "Grain Mass: "
            + "{:.3f}".format(self.hybrid_motor.solid.grainInitialMass)
            + " kg"
        )

    def motor_details(self):
        """Prints out all data available about the HybridMotor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.hybrid_motor.burnDuration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.hybrid_motor.propellantInitialMass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.hybrid_motor.exhaustVelocity.average(*self.hybrid_motor.burn_time)
            )
            + " m/s"
        )
        print(
            "Average Thrust: " + "{:.3f}".format(self.hybrid_motor.averageThrust) + " N"
        )
        print(
            "Maximum Thrust: "
            + str(self.hybrid_motor.maxThrust)
            + " N at "
            + str(self.hybrid_motor.maxThrustTime)
            + " s after ignition."
        )
        print(
            "Total Impulse: " + "{:.3f}".format(self.hybrid_motor.totalImpulse) + " Ns"
        )

    def all(self):
        """Prints out all data available about the HybridMotor.

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
