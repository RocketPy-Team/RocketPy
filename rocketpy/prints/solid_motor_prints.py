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
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.solid_motor.nozzle_radius) + " m")
        print("Nozzle Throat Radius: " + str(self.solid_motor.throat_radius) + " m\n")

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
        print("Grain Details")
        print("Number of Grains: " + str(self.solid_motor.grain_number))
        print("Grain Spacing: " + str(self.solid_motor.grain_separation) + " m")
        print("Grain Density: " + str(self.solid_motor.grain_density) + " kg/m3")
        print("Grain Outer Radius: " + str(self.solid_motor.grain_outer_radius) + " m")
        print(
            "Grain Inner Radius: "
            + str(self.solid_motor.grain_initial_inner_radius)
            + " m"
        )
        print("Grain Height: " + str(self.solid_motor.grain_initial_height) + " m")
        print(
            "Grain Volume: "
            + "{:.3f}".format(self.solid_motor.grainInitialVolume)
            + " m3"
        )
        print(
            "Grain Mass: "
            + "{:.3f}".format(self.solid_motor.grainInitialMass)
            + " kg\n"
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
        print("Motor Details")
        print("Total Burning Time: " + str(self.solid_motor.burn_duration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.solid_motor.propellant_initial_mass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.solid_motor.exhaust_velocity.average(*self.solid_motor.burn_time)
            )
            + " m/s"
        )
        print(
            "Average Thrust: " + "{:.3f}".format(self.solid_motor.average_thrust) + " N"
        )
        print(
            "Maximum Thrust: "
            + str(self.solid_motor.max_thrust)
            + " N at "
            + str(self.solid_motor.max_thrust_time)
            + " s after ignition."
        )
        print(
            "Total Impulse: "
            + "{:.3f}".format(self.solid_motor.total_impulse)
            + " Ns\n"
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
        self.grain_details()
        self.motor_details()
        return None
