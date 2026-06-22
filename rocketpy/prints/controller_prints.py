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

    def controller_function(self):
        """Prints the controller function information.

        Returns
        -------
        None
        """
        if self.controller.controller_function.__name__ == "<lambda>":
            line = getsourcelines(self.controller.trigger)[0][0]
            print("Controller function: " + line.split("=")[0].strip())
        else:
            print(
                "Controller function: " + self.controller.controller_function.__name__
            )
        print(f"Controller refresh rate: {self.controller.sampling_rate:.3f} Hz")

    def interactive_objects(self):
        """Prints interactive objects."""
        print("interactive Objects")
        # check if is list
        if isinstance(self.controller.interactive_objects, list):
            for obj in self.controller.interactive_objects:
                print(getattr(obj, "name", str(obj)))
        else:
            obj = self.controller.interactive_objects
            print(getattr(obj, "name", str(obj)))

    def all(self):
        """Prints all information about the parachute.

        Returns
        -------
        None
        """

        print("\nController Details\n")
        print(self.controller)
        self.controller_function()
        self.interactive_objects()
