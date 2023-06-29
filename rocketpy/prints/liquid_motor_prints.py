__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _LiquidMotorPrints:
    """Class that holds prints methods for LiquidMotor class.

    Attributes
    ----------
    _LiquidMotorPrints.liquid_motor : liquid_motor
        LiquidMotor object that will be used for the prints.

    """

    def __init__(
        self,
        liquid_motor,
    ):
        """Initializes _LiquidMotorPrints class

        Parameters
        ----------
        liquid_motor: LiquidMotor
            Instance of the LiquidMotor class.

        Returns
        -------
        None
        """
        self.liquid_motor = liquid_motor
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
        print("Nozzle Radius: " + str(self.liquid_motor.nozzleRadius) + " m")

    def motor_details(self):
        """Prints out all data available about the motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.liquid_motor.burnDuration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.liquid_motor.propellantInitialMass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.liquid_motor.exhaustVelocity.average(*self.liquid_motor.burn_time)
            )
            + " m/s"
        )
        print(
            "Average Thrust: " + "{:.3f}".format(self.liquid_motor.averageThrust) + " N"
        )
        print(
            "Maximum Thrust: "
            + str(self.liquid_motor.maxThrust)
            + " N at "
            + str(self.liquid_motor.maxThrustTime)
            + " s after ignition."
        )
        print(
            "Total Impulse: " + "{:.3f}".format(self.liquid_motor.totalImpulse) + " Ns"
        )

    def all(self):
        """Prints out all data available about the LiquidMotor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.nozzle_details()
        print()

        self.motor_details()
        print()

        return None
