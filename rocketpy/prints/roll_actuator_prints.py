class _RollActuatorPrints:
    """Class that contains all roll actuator prints."""

    def __init__(self, roll_actuator):
        """Initializes _RollActuatorPrints class

        Parameters
        ----------
        roll_actuator: rocketpy.RollActuator
            Instance of the RollActuator class.

        Returns
        -------
        None
        """
        self.roll_actuator = roll_actuator

    def basics(self):
        """Prints information of the roll actuator."""
        print(f"Information of {self.roll_actuator.name}:")
        print("----------------------------------")
        print(f"Torque demand rate: {self.roll_actuator.demand_rate} Hz")
        print(
            f"Torque range: {self.roll_actuator.actuator_range[0]:.2f} to {self.roll_actuator.actuator_range[1]:.2f} N·m"
        )
        print(f"Torque rate limit: {self.roll_actuator.actuator_rate_limit} N·m/s")
        print(
            f"Torque clamping: {'Enabled' if self.roll_actuator.clamp else 'Disabled'}"
        )
        print(f"Torque time constant: {self.roll_actuator.actuator_time_constant} sec")
        print(f"Current roll torque: {self.roll_actuator.roll_torque:.2f} N·m")

    def all(self):
        """Prints all information of the roll actuator."""
        self.basics()
