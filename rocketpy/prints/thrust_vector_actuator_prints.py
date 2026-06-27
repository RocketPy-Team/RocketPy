class _ThrustVectorActuatorPrints:
    """Class that contains all thrust vector actuator prints."""

    def __init__(self, thrust_vector_actuator):
        """Initializes _ThrustVectorActuatorPrints class

        Parameters
        ----------
        thrust_vector_actuator: rocketpy.ThrustVectorActuator
            Instance of the thrust vector actuator class.

        Returns
        -------
        None
        """
        self.thrust_vector_actuator = thrust_vector_actuator

    def basics(self):
        """Prints information of the thrust vector actuator."""
        print(f"Information of {self.thrust_vector_actuator.name}:")
        print("----------------------------------")
        print(f"Gimbal demand rate: {self.thrust_vector_actuator.demand_rate} Hz")
        print(
            f"Gimbal range: {self.thrust_vector_actuator.actuator_range[0]:.2f} to {self.thrust_vector_actuator.actuator_range[1]:.2f} deg"
        )
        print(
            f"Gimbal rate limit: {self.thrust_vector_actuator.actuator_rate_limit} deg/sec"
        )
        print(
            f"Gimbal clamping: {'Enabled' if self.thrust_vector_actuator.clamp else 'Disabled'}"
        )
        print(
            f"Gimbal time constant: {self.thrust_vector_actuator.actuator_time_constant} sec"
        )
        print(
            f"Current gimbal angle: {self.thrust_vector_actuator.gimbal_angle:.2f} deg"
        )

    def all(self):
        """Prints all information of the thrust vector actuator."""
        self.basics()
