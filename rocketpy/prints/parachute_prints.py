from inspect import getsourcelines


class _ParachutePrints:
    """Class that holds prints methods for Parachute class.

    Attributes
    ----------
    _ParachutePrints.parachute : rocketpy.Parachute
        Parachute object that will be used for the prints.

    """

    def __init__(self, parachute):
        """Initializes _ParachutePrints class

        Parameters
        ----------
        parachute: rocketpy.Parachute
            Instance of the Parachute class.

        Returns
        -------
        None
        """
        self.parachute = parachute

        return None

    def trigger(self):
        """Prints trigger information.

        Returns
        -------
        None
        """

        if callable(self.parachute.trigger):
            if self.parachute.trigger.__name__ == "<lambda>":
                line = getsourcelines(self.parachute.trigger)[0][0]
                print("Ejection signal trigger: " + line.split("=")[0].strip())
            else:
                print("Ejection signal trigger: " + self.parachute.trigger.__name__)
        elif isinstance(self.parachute.trigger, (int, float)):
            print(
                "Ejection signal trigger: " + str(self.parachute.trigger) + " m (AGL)"
            )
        elif isinstance(self.parachute.trigger, str):
            print("Ejection signal trigger: At Apogee")

        print(f"Ejection system refresh rate: {self.parachute.sampling_rate:.3f} Hz")
        print(
            f"Time between ejection signal is triggered and the parachute is fully opened: {self.parachute.lag:.1f} s\n"
        )

        return None

    def noise(self):
        # Not implemented yet
        pass

    def all(self):
        """Prints all information about the parachute.

        Returns
        -------
        None
        """

        print("\nParachute Details\n")
        print(self.parachute.__str__())
        self.trigger()
        self.noise()

        return None
