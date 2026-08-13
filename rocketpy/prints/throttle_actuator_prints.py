class _ThrottleActuatorPrints:
    def __init__(self, throttle_actuator):
        """Initializes _ThrottleActuatorPrints class

        Parameters
        ----------
        throttle_actuator: rocketpy.ThrottleActuator
            Instance of the ThrottleActuator class.

        Returns
        -------
        None
        """
        self.throttle_actuator = throttle_actuator

    def basics(self):
        """Prints information of the throttle actuator."""
        print(f"Information of {self.throttle_actuator.name}:")
        print("----------------------------------")
        print(f"Throttle demand rate: {self.throttle_actuator.demand_rate} Hz")
        print(
            f"Throttle range: {self.throttle_actuator.actuator_range[0]:.2f} to {self.throttle_actuator.actuator_range[1]:.2f}"
        )
        print(f"Throttle rate limit: {self.throttle_actuator.actuator_rate_limit}")
        print(
            f"Throttle Clamping: {'Enabled' if self.throttle_actuator.clamp else 'Disabled'}"
        )
        print(
            f"Throttle time constant: {self.throttle_actuator.actuator_time_constant} sec"
        )
        print(f"Current throttle: {self.throttle_actuator.throttle:.2f}")

    def all(self):
        """Prints all information of the throttle actuator."""
        self.basics()
