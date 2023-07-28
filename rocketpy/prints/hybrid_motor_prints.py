__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np


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
        print(f"Outlet Radius: {self.hybrid_motor.nozzle_radius} m")
        print(f"Throat Radius: {self.hybrid_motor.solid.throat_radius} m")
        print(f"Outlet Area: {np.pi*self.hybrid_motor.nozzle_radius**2:.6f} m²")
        print(f"Throat Area: {np.pi*self.hybrid_motor.solid.throat_radius**2:.6f} m²")
        print(f"Position: {self.hybrid_motor.nozzle_position} m\n")
        return None

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
        print("Grain Details")
        print("Number of Grains: " + str(self.hybrid_motor.solid.grain_number))
        print("Grain Spacing: " + str(self.hybrid_motor.solid.grain_separation) + " m")
        print("Grain Density: " + str(self.hybrid_motor.solid.grain_density) + " kg/m3")
        print(
            "Grain Outer Radius: "
            + str(self.hybrid_motor.solid.grain_outer_radius)
            + " m"
        )
        print(
            "Grain Inner Radius: "
            + str(self.hybrid_motor.solid.grain_initial_inner_radius)
            + " m"
        )
        print(
            "Grain Height: " + str(self.hybrid_motor.solid.grain_initial_height) + " m"
        )
        print(
            "Grain Volume: "
            + "{:.3f}".format(self.hybrid_motor.solid.grainInitialVolume)
            + " m3"
        )
        print(
            "Grain Mass: "
            + "{:.3f}".format(self.hybrid_motor.solid.grainInitialMass)
            + " kg\n"
        )
        return None

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
        print("Motor Details")
        print("Total Burning Time: " + str(self.hybrid_motor.burn_duration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.hybrid_motor.propellant_initial_mass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.hybrid_motor.exhaust_velocity.average(*self.hybrid_motor.burn_time)
            )
            + " m/s"
        )
        print(
            "Average Thrust: "
            + "{:.3f}".format(self.hybrid_motor.average_thrust)
            + " N"
        )
        print(
            "Maximum Thrust: "
            + str(self.hybrid_motor.max_thrust)
            + " N at "
            + str(self.hybrid_motor.max_thrust_time)
            + " s after ignition."
        )
        print(
            "Total Impulse: "
            + "{:.3f}".format(self.hybrid_motor.total_impulse)
            + " Ns\n"
        )
        return None

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
        self.grain_details()
        self.motor_details()

        return None
