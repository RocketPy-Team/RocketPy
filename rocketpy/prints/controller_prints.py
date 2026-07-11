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
            line = getsourcelines(self.controller.controller_function)[0][0]
            print("Controller function: " + line.split("=")[0].strip())
        else:
            print(
                "Controller function: " + self.controller.controller_function.__name__
            )
        if self.controller.sampling_rate is None:
            print("Controller refresh rate: continuous")
        else:
            print(f"Controller refresh rate: {self.controller.sampling_rate:.3f} Hz")

    def controlled_objects(self):
        """Prints the objects controlled by the controller."""
        print("Controlled Objects")
        controlled_objects = self.controller.controlled_objects
        if not isinstance(controlled_objects, (list, tuple)):
            controlled_objects = [controlled_objects]
        for obj in controlled_objects:
            print(getattr(obj, "name", str(obj)))

    def control_history(self):
        """Prints a summary of the recorded control history, per controlled
        object and control variable."""
        history = getattr(self.controller, "control_history", {})
        if not any(
            samples for variables in history.values() for samples in variables.values()
        ):
            print("No recorded control history.")
            return
        print("Recorded Control History")
        for object_name, variables in history.items():
            for variable, samples in variables.items():
                if not samples:
                    continue
                times = [sample[0] for sample in samples]
                values = [sample[1] for sample in samples]
                print(
                    f"'{object_name}.{variable}': {len(samples)} samples from "
                    f"t = {times[0]:.3f} s to t = {times[-1]:.3f} s, "
                    f"range [{min(values):.4g}, {max(values):.4g}]"
                )

    def all(self):
        """Prints all information about the controller.

        Returns
        -------
        None
        """

        print("\nController Details\n")
        print(self.controller)
        print(f"Enabled: {self.controller.enabled}")
        if self.controller.needs:
            print(f"Declared needs: {sorted(self.controller.needs)}")
        self.controller_function()
        self.controlled_objects()
        self.control_history()


class _AirBrakesControllerPrints(_ControllerPrints):
    """Prints for AirBrakesController, adding a deployment-level summary."""

    def deployment_level(self):
        """Prints a summary of the recorded deployment level."""
        samples = self.controller.control_history.get("air_brakes", {}).get(
            "deployment_level", []
        )
        if not samples:
            print("No recorded deployment level history.")
            return
        values = [sample[1] for sample in samples]
        print(f"Final deployment level: {values[-1]:.4g}")
        print(f"Maximum deployment level: {max(values):.4g}")

    def all(self):
        """Prints all information about the air brakes controller."""
        super().all()
        self.deployment_level()
