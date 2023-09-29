from inspect import getsourcelines


class _ControllerPrints:
    """Class that holds prints methods for Controller class.

    Attributes
    ----------
    _ControllerPrint.controller : controller
        Controller object that will be used for the prints.

    """

    def __init__(
        self,
        controller,
    ):
        """Initializes _ControllerPrints class

        Parameters
        ----------
        controller: Controller
            Instance of the Controller class.

        Returns
        -------
        None
        """
        self.controller = controller
        return None

    def controller_function(self):
        """Prints controlle function information.

        Returns
        -------
        None
        """
        if self.controller.controller_function.__name__ == "<lambda>":
            line = getsourcelines(self.parachute.trigger)[0][0]
            print("Controller function: " + line.split("=")[0].strip())
        else:
            print(
                "Controller function: " + self.controller.controller_function.__name__
            )
        print(f"Controller refresh rate: {self.controller.sampling_rate:.3f} Hz")
        return None

    def observed_objects(self):
        """Prints observed objects."""
        print("Observed Objects")
        for obj in self.controller.observed_objects:
            if hasattr(obj, "name"):
                print(obj.name)
            else:
                print(obj)

    def all(self):
        """Prints all information about the parachute.

        Returns
        -------
        None
        """

        print("\nController Details\n")
        print(self.controller.__str__())
        self.controller_function()
        self.observed_objects()
        return None
