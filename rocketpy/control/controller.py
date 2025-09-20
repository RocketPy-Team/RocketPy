from inspect import signature
from typing import Iterable

from rocketpy.tools import from_hex_decode, to_hex_encode

from ..prints.controller_prints import _ControllerPrints


class _Controller:
    """A class for storing and running controllers on a rocket. Controllers
    have a controller function that is called at a specified sampling rate
    during the simulation. The controller function can access and modify
    the objects that are passed to it. The controller function also stores the
    variables of interest in the objects that are passed to it."""

    def __init__(
        self,
        interactive_objects,
        controller_function,
        sampling_rate,
        initial_observed_variables=None,
        name="Controller",
    ):
        """Initialize the class with the controller function and the objects to
        be observed.

        Parameters
        ----------
        interactive_objects : list or object
            A collection of objects that the controller function can access and
            potentially modify. This can be either a list of objects or a single
            object. The objects listed here are provided to the controller function
            as the last argument, maintaining the order specified in this list if
            it's a list. The controller function gains the ability to interact with
            and make adjustments to these objects during its execution.
        controller_function : function, callable
            An user-defined function responsible for controlling the simulation.
            This function is expected to take the following arguments, in order:

            1. `time` (float): The current simulation time in seconds.
            2. `sampling_rate` (float): The rate at which the controller
               function is called, measured in Hertz (Hz).
            3. `state` (list): The state vector of the simulation, structured as
               `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.
            4. `state_history` (list): A record of the rocket's state at each
               step throughout the simulation. The state_history is organized as
               a list of lists, with each sublist containing a state vector. The
               last item in the list always corresponds to the previous state
               vector, providing a chronological sequence of the rocket's
               evolving states.
            5. `observed_variables` (list): A list containing the variables that
               the controller function returns. The return of each controller
               function call is appended to the observed_variables list. The
               initial value in the first step of the simulation of this list is
               provided by the `initial_observed_variables` argument.
            6. `interactive_objects` (list): A list containing the objects that
               the controller function can interact with. The objects are
               listed in the same order as they are provided in the
               `interactive_objects`.
            7. `sensors` (list): A list of sensors that are attached to the
                rocket. The most recent measurements of the sensors are provided
                with the ``sensor.measurement`` attribute. The sensors are
                listed in the same order as they are added to the rocket

            This function will be called during the simulation at the specified
            sampling rate. The function should evaluate and change the interactive
            objects as needed. The function return statement can be used to save
            relevant information in the `observed_variables` list.

            .. note:: The function will be called according to the sampling rate
            specified.
        sampling_rate : float
            The sampling rate of the controller function in Hertz (Hz). This
            means that the controller function will be called every
            `1/sampling_rate` seconds.
        initial_observed_variables : list, optional
            A list of the initial values of the variables that the controller
            function returns. This list is used to initialize the
            `observed_variables` argument of the controller function. The
            default value is None, which initializes the list as an empty list.
        name : str
            The name of the controller. This will be used for printing and
            plotting.

        Returns
        -------
        None
        """
        self.interactive_objects = interactive_objects
        self.base_controller_function = controller_function
        self.controller_function = self.__init_controller_function(controller_function)
        self.sampling_rate = sampling_rate
        self.initial_observed_variables = initial_observed_variables
        self.name = name
        self.prints = _ControllerPrints(self)

        if initial_observed_variables is not None:
            self.observed_variables = [initial_observed_variables]
        else:
            self.observed_variables = []

    def __init_controller_function(self, controller_function):
        """Checks number of arguments of the controller function and initializes
        it with the correct number of arguments. This is a workaround to allow
        the controller function to receive sensors without breaking changes"""
        sig = signature(controller_function)
        if len(sig.parameters) == 6:
            # pylint: disable=unused-argument
            def new_controller_function(
                time,
                sampling_rate,
                state_vector,
                state_history,
                observed_variables,
                interactive_objects,
                sensors,
            ):
                return controller_function(
                    time,
                    sampling_rate,
                    state_vector,
                    state_history,
                    observed_variables,
                    interactive_objects,
                )

        elif len(sig.parameters) == 7:
            new_controller_function = controller_function
        else:
            raise ValueError(
                "The controller function must have 6 or 7 arguments. "
                "The arguments must be in the following order: "
                "(time, sampling_rate, state_vector, state_history, "
                "observed_variables, interactive_objects, sensors)."
                "Sensors argument is optional."
            )
        return new_controller_function

    def __call__(self, time, state_vector, state_history, sensors):
        """Call the controller function. This is used by the simulation class.

        Parameters
        ----------
        time : float
            The time of the simulation in seconds.
        state_vector : list
            The state vector of the simulation, which is defined as:

            `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.
        state_history : list
            A list containing the state history of the simulation. The state
            history is a list of every state vector of every step of the
            simulation. The state history is a list of lists, where each
            sublist is a state vector and is ordered from oldest to newest.
        sensors : list
            A list of sensors that are attached to the rocket. The most recent
            measurements of the sensors are provided with the
            ``sensor.measurement`` attribute. The sensors are listed in the same
            order as they are added to the rocket.

        Returns
        -------
        None
        """
        observed_variables = self.controller_function(
            time,
            self.sampling_rate,
            state_vector,
            state_history,
            self.observed_variables,
            self.interactive_objects,
            sensors,
        )
        if observed_variables is not None:
            self.observed_variables.append(observed_variables)

    def __str__(self):
        return f"Controller '{self.name}' with sampling rate {self.sampling_rate} Hz."

    def info(self):
        """Prints out summarized information about the controller."""
        self.prints.all()

    def all_info(self):
        """Prints out all information about the controller."""
        self.info()

    def to_dict(self, **kwargs):
        allow_pickle = kwargs.get("allow_pickle", True)

        if allow_pickle:
            controller_function = to_hex_encode(self.controller_function)
        else:
            controller_function = self.controller_function.__name__

        return {
            "controller_function": controller_function,
            "sampling_rate": self.sampling_rate,
            "initial_observed_variables": self.initial_observed_variables,
            "name": self.name,
            "_interactive_objects_hash": hash(self.interactive_objects)
            if not isinstance(self.interactive_objects, Iterable)
            else [hash(obj) for obj in self.interactive_objects],
        }

    @classmethod
    def from_dict(cls, data):
        interactive_objects = data.get("interactive_objects", [])
        controller_function = data.get("controller_function")
        sampling_rate = data.get("sampling_rate")
        initial_observed_variables = data.get("initial_observed_variables")
        name = data.get("name", "Controller")

        try:
            controller_function = from_hex_decode(controller_function)
        except (TypeError, ValueError):
            pass

        obj = cls(
            interactive_objects=interactive_objects,
            controller_function=controller_function,
            sampling_rate=sampling_rate,
            initial_observed_variables=initial_observed_variables,
            name=name,
        )
        setattr(
            obj, "_interactive_objects_hash", data.get("_interactive_objects_hash", [])
        )
        return obj
