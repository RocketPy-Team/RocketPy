__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _MotorPrints:
    """Class that holds prints methods for Motor class.

    Attributes
    ----------
    _MotorPrints.motor : Motor
        Motor object that will be used for the prints.

    """

    def __init__(
        self,
        motor,
    ):
        """Initializes _MotorPrints class

        Parameters
        ----------
        motor: Motor
            Instance of the Motor class.

        Returns
        -------
        None
        """
        self.motor = motor
        return None

    def motor_details(self):
        """Print Motor details.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("Motor Details")
        print("Total Burning Time: " + str(self.motor.burnOutTime) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.motor.propellantInitialMass)
            + " kg"
        )
        print(
            "Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.motor.exhaustVelocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.motor.averageThrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.motor.maxThrust)
            + " N at "
            + str(self.motor.maxThrustTime)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.motor.totalImpulse) + " Ns\n")
        return None

    def all(self):
        """Prints out all data available about the Motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        self.motor_details()
        return None
