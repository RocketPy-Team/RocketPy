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

    def writes(self):
        """Print the controls the controller may set."""
        writes = self.controller.writes
        if writes is None:
            print("Writes: any control")
        else:
            print(f"Writes: {', '.join(writes)}")

    def gates(self):
        """Print when the controller runs."""
        disable_on, enable_on = self.controller._gates
        print(f"Enabled at the start: {self.controller._initial_enabled}")
        if disable_on is not None:
            print(f"Disabled on: {getattr(disable_on, '__name__', disable_on)}")
        if enable_on is not None:
            print(f"Enabled on: {getattr(enable_on, '__name__', enable_on)}")

    def memory_and_records(self):
        """Print the names kept in memory and the names recorded."""
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
        self.writes()
        self.gates()
        self.memory_and_records()
