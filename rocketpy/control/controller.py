from ..prints.control_prints import _ControllerPrints


class Controller:
    """A class for storing and running controllers on a rocket."""

    def __init__(
        self,
        observed_objects,
        controller_function,
        sampling_rate,
        name="Controller",
    ):
        """Initialize the class with the controller function and the objects to
        be observed.

        Parameters
        ----------
        observed_objects : list
            A list of objects to be observed by the controller. It can be any
            python object. This list will be passed to the controller function
            as positional arguments, meaning that the order of the objects in
            this list matters. These objects will then be able to be accessed
            and modified by the controller function.
        controller_function : function, callable
            A function that takes the following arguments, in this order:

            1. Time of the simulation at the current step in seconds.
            2. The state vector of the simulation, which is defined as:

               `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.
            3. A list containing the objects to be acted upon by the controller.
               The objects in this list are the same as the objects in the
               observed_objects list, but they can be modified by the controller.

            This function will be called during the simulation at the specified
            sampling rate. The function should evaluate and change the observed
            objects as needed. The function should return None.

            Note: The function will be called according to the sampling rate
            specified.

        sampling_rate : float
            The sampling rate of the controller function in Hertz (Hz). This
            means that the controller function will be called every
            `1/sampling_rate` seconds.
        name : str
            The name of the controller. This will be used for printing and
            plotting.

        Returns
        -------
        None
        """

        self.observed_objects = observed_objects
        self.controller_function = controller_function
        self.sampling_rate = sampling_rate
        self.name = name
        self.prints = _ControllerPrints

    def __call__(self, time, state_vector):
        """Call the controller function. This is used by the simulation class.

        Parameters
        ----------
        time : float
            The time of the simulation in seconds.
        state_vector : list
            The state vector of the simulation, which is defined as:

            `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.

        Returns
        -------
        None
        """
        self.controller_function(time, state_vector, self.observed_objects)

    def __str__(self):
        return self.name

    def info(self):
        """Prints out summarized information about the controller."""
        self.prints.all()

    def all_info(self):
        """Prints out all information about the controller."""
        self.info()
