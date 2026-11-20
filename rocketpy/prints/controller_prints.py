def print_controls(controllable):
    """Print each control of a controllable with the value it starts every
    flight at and its limits, and the controllable's own states."""
    for control in controllable.controls:
        limits = controllable.control_limits.get(control)
        limits = "no limits" if limits is None else f"from {limits[0]} to {limits[1]}"
        print(
            f"\tControl {control}: starts at "
            f"{controllable.control_defaults[control]}, {limits}"
        )
    if controllable.states:
        print(f"\tStates: {', '.join(controllable.states)}")


class _ControllablePrints:
    """Class that holds prints methods for a part with controls or states of
    its own that is not an aerodynamic surface: a continuous controller, a
    thrust effector or a force effector.

    Attributes
    ----------
    _ControllablePrints.controllable : Controllable
        The part the prints are of.
    """

    # The functions a part can be built from, by the attribute that holds them
    _FUNCTIONS = (
        ("Control law", "law_function"),
        ("State derivative", "derivative_function"),
        ("Thrust function", "thrust_function"),
        ("Force function", "force_function"),
    )

    def __init__(self, controllable):
        self.controllable = controllable

    def functions(self):
        """Print the functions the part was built from."""
        for label, attribute in self._FUNCTIONS:
            function = getattr(self.controllable, attribute, None)
            if function is not None:
                name = getattr(function, "__name__", type(function).__name__)
                print(f"\t{label}: {name}")

    def controls(self):
        """Print the part's controls, with where each starts a flight and its
        limits, and the part's own states."""
        print_controls(self.controllable)

    def all(self):
        """Print all the information about the part.

        Returns
        -------
        None
        """
        part = self.controllable
        print(f"\n{type(part).__name__}: {part.name}\n")
        self.functions()
        self.controls()


class _ControllerPrints:
    """Class that holds prints methods for Controller class.

    Attributes
    ----------
    _ControllerPrints.controller : Controller
        Controller object that will be used for the prints.
    """

    def __init__(self, controller):
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
        """Print the controller function and how often it runs."""
        function = self.controller.controller_function
        name = getattr(function, "__name__", type(function).__name__)
        print(f"Controller function: {name}")
        print(f"Sampling rate: {self.controller.sampling_rate:.3f} Hz")

    def gates(self):
        """Print when the controller runs."""
        disable_on, enable_on = self.controller._gates
        print(f"Enabled at the start: {self.controller._initial_enabled}")
        if disable_on is not None:
            print(f"Disabled on: {getattr(disable_on, '__name__', disable_on)}")
        if enable_on is not None:
            print(f"Enabled on: {getattr(enable_on, '__name__', enable_on)}")

    def memory_and_records(self):
        """Print the outputs, the names kept in memory and the names
        recorded."""
        if self.controller.outputs:
            print(f"Outputs: {', '.join(self.controller.outputs)}")
        memory = self.controller._initial_memory
        if memory:
            print(f"Memory keys: {', '.join(str(key) for key in memory)}")
        records = self.controller._records
        if records:
            print(f"Records of the last flight: {', '.join(records)}")

    def all(self):
        """Print all the information about the controller.

        Returns
        -------
        None
        """
        print("\nController Details\n")
        print(self.controller)
        self.controller_function()
        self.gates()
        self.memory_and_records()
