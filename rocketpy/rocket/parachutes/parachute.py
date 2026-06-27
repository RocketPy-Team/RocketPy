from abc import ABC, abstractmethod
from inspect import signature

import numpy as np

from rocketpy.tools import from_hex_decode, to_hex_encode

from ...mathutils.function import Function
from ...prints.parachute_prints import _ParachutePrints


class Parachute(ABC):
    """Abstract class to specify characteristics and useful operations for
    parachutes. Cannot be instantiated.

    Attributes
    ----------
    Parachute.name : string
        Parachute name, such as drogue and main. Has no impact in
        simulation, as it is only used to display data in a more
        organized matter.
    Parachute.parachute_type : string
        Parachute type, such as hemispherical and parafoil.
    Parachute.trigger : callable, float, str
        This parameter defines the trigger condition for the parachute ejection
        system. It can be one of the following:

        - A callable function that takes four arguments:
          1. Freestream pressure in pascals.
          2. Height in meters above ground level.
          3. The state vector of the simulation, which is defined as:

             `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.

          4. A list of sensors that are attached to the rocket. The most recent
             measurements of the sensors are provided with the
             ``sensor.measurement`` attribute. The sensors are listed in the same
             order as they are added to the rocket.

          The function should return ``True`` if the parachute ejection system
          should be triggered and False otherwise. The function will be called
          according to the specified sampling rate.

        - A float value, representing an absolute height in meters. In this
          case, the parachute will be ejected when the rocket reaches this height
          above ground level.

        - The string "apogee" which triggers the parachute at apogee, i.e.,
          when the rocket reaches its highest point and starts descending.


    Parachute.triggerfunc : function
        Trigger function created from the trigger used to evaluate the trigger
        condition for the parachute ejection system. It is a callable function
        that takes four arguments: freestream pressure in Pa, height above
        ground level in meters, the state vector of the simulation and the list
        of sensors attached to the rocket. It returns ``True`` if the parachute
        ejection system should be triggered and ``False`` otherwise.

        .. note:

            The function will be called according to the sampling rate specified.

    Parachute.sampling_rate : float
        Sampling rate, in Hz, for the trigger function.
    Parachute.lag : float
        Time, in seconds, between the parachute ejection system is triggered
        and the parachute is fully opened.
    Parachute.noise : tuple, list
        List in the format (mean, standard deviation, time-correlation).
        The values are used to add noise to the pressure signal which is passed
        to the trigger function. Default value is (0, 0, 0). Units are in Pa.
    Parachute.noise_bias : float
        Mean value of the noise added to the pressure signal, which is
        passed to the trigger function. Unit is in Pa.
    Parachute.noise_deviation : float
        Standard deviation of the noise added to the pressure signal,
        which is passed to the trigger function. Unit is in Pa.
    Parachute.noise_corr : tuple, list
        Tuple with the correlation between noise and time.
    Parachute.noise_signal : list of tuple
        List of (t, noise signal) corresponding to signal passed to
        trigger function. Completed after running a simulation.
    Parachute.noisy_pressure_signal : list of tuple
        List of (t, noisy pressure signal) that is passed to the
        trigger function. Completed after running a simulation.
    Parachute.clean_pressure_signal : list of tuple
        List of (t, clean pressure signal) corresponding to signal passed to
        trigger function. Completed after running a simulation.
    Parachute.noise_signal_function : Function
        Function of noiseSignal.
    Parachute.noisy_pressure_signal_function : Function
        Function of noisy_pressure_signal.
    Parachute.clean_pressure_signal_function : Function
        Function of clean_pressure_signal.
    """

    def __init__(
        self,
        name,
        parachute_type,
        trigger,
        sampling_rate,
        lag=0,
        noise=(0, 0, 0),
    ):
        """Initializes Parachute class.

        Parameters
        ----------
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        parachute_type : string
            Parachute type, such as hemispherical and parafoil.
        trigger : callable, float, str
            Defines the trigger condition for the parachute ejection system. It
            can be one of the following:

            - A callable function that takes three or four arguments: \

                1. Freestream pressure in pascals.
                2. Height in meters above ground level.
                3. The state vector of the simulation, which is defined as: \

                    .. code-block:: python

                        u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

                4. (optional) A list of sensors attached to the rocket.

                .. note::

                    The function should return ``True`` if the parachute \
                    ejection system should be triggered and ``False`` otherwise.
            - A float value, representing an absolute height in meters. In this \
                case, the parachute will be ejected when the rocket reaches this \
                height above ground level.
            - The string "apogee" which triggers the parachute at apogee, i.e., \
                when the rocket reaches its highest point and starts descending.

            .. note::

                The function will be called according to the sampling rate specified.
        sampling_rate : float
            Sampling rate in which the parachute trigger will be checked at.
            It is used to simulate the refresh rate of onboard sensors such
            as barometers. Default value is 100. Value must be given in hertz.
        lag : float, optional
            Time between the parachute ejection system is triggered and the
            parachute is fully opened. During this time, the simulation will
            consider the rocket as flying without a parachute. Default value
            is 0. Must be given in seconds.
        noise : tuple, list, optional
            List in the format (mean, standard deviation, time-correlation).
            The values are used to add noise to the pressure signal which is
            passed to the trigger function. Default value is ``(0, 0, 0)``.
            Units are in Pa.
        """

        # Save arguments as attributes
        self.name = name
        self.parachute_type = parachute_type
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
        self.noise = noise
        self.__init_noise(noise)
        self.__evaluate_trigger_function(trigger)

        # Prints and plots
        self.prints = _ParachutePrints(self)

    def __init_noise(self, noise):
        """Initializes all noise-related attributes.

        Parameters
        ----------
        noise : tuple, list
            List in the format (mean, standard deviation, time-correlation).
        """
        self.noise_signal = [[-1e-6, np.random.normal(noise[0], noise[1])]]
        self.noisy_pressure_signal = []
        self.clean_pressure_signal = []
        self.noise_bias = noise[0]
        self.noise_deviation = noise[1]
        self.noise_corr = (noise[2], (1 - noise[2] ** 2) ** 0.5)
        self.clean_pressure_signal_function = Function(0)
        self.noisy_pressure_signal_function = Function(0)
        self.noise_signal_function = Function(0)
        alpha, beta = self.noise_corr
        self.noise_function = lambda: (
            alpha * self.noise_signal[-1][1]
            + beta * np.random.normal(noise[0], noise[1])
        )

    def __evaluate_trigger_function(self, trigger):
        """This is used to set the triggerfunc attribute that will be used to
        interact with the Flight class.
        """
        # pylint: disable=unused-argument, function-redefined

        # Case 1: The parachute is deployed by a custom function
        if callable(trigger):
            # work around for having added sensors to parachute triggers
            # to avoid breaking changes
            triggerfunc = trigger
            sig = signature(triggerfunc)
            if len(sig.parameters) == 3:

                def triggerfunc(p, h, y, sensors):
                    return trigger(p, h, y)

            self.triggerfunc = triggerfunc

        # Case 2: The parachute is deployed at a given height
        elif isinstance(trigger, (int, float)):
            # The parachute is deployed at a given height
            def triggerfunc(p, h, y, sensors):
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return y[5] < 0 and h < trigger

            self.triggerfunc = triggerfunc

        # Case 3: The parachute is deployed at apogee
        elif isinstance(trigger, str) and trigger.lower() == "apogee":
            # The parachute is deployed at apogee
            def triggerfunc(p, h, y, sensors):
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return y[5] < 0

            self.triggerfunc = triggerfunc

        # Case 4: Invalid trigger input
        else:
            raise ValueError(
                f"Unable to set the trigger function for parachute '{self.name}'. "
                + "Trigger must be a callable, a float value or the string 'apogee'. "
                + "See the Parachute class documentation for more information."
            )

    def __str__(self):
        """Returns a string representation of the Parachute class.

        Returns
        -------
        string
            String representation of Parachute class. It is human readable.
        """
        return f"Parachute {self.name.title()} of type {self.parachute_type}"

    def __repr__(self):
        """Representation method for the class, useful when debugging."""
        return (
            f"<Parachute {self.name} of type {self.parachute_type} "
            + f"(lag = {self.lag:.4f} s, trigger = {self.trigger})>"
        )

    def info(self):
        """Prints information about the Parachute class."""
        self.prints.all()

    def all_info(self):
        """Prints all information about the Parachute class."""
        self.info()

    @abstractmethod
    def add_information_to_flight(self, flight_obj, additional_info):
        """Adds parachute information to flight"""

    @abstractmethod
    def u_dot(self, t, u, flight_information, post_processing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying under parachute. Each parachute type has


        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        flight_information : dictionary
            A dictionary containing additional information used in
            the parachute equations of motion. Examples are
            Environment and Rocket data
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack. Default is False.

        Return
        ------
        u_dot : dict
            A dictionary containing two or three keys
            1) state: State vector which depends on the parachute model.
            2) additional_information: information as dict that is added
            to the  'parachutes_info' attribute in the Flight class.
            3) post_processing_information: State vector containing
            post processing information.

        """

    def to_dict(self, **kwargs):
        allow_pickle = kwargs.get("allow_pickle", True)
        trigger = self.trigger

        if callable(self.trigger) and not isinstance(self.trigger, Function):
            if allow_pickle:
                trigger = to_hex_encode(trigger)
            else:
                trigger = trigger.__name__

        data = {
            "name": self.name,
            "parachute_type": self.parachute_type,
            "cd_s": self.cd_s,
            "trigger": trigger,
            "sampling_rate": self.sampling_rate,
            "lag": self.lag,
            "noise": self.noise,
            "radius": self.radius,
            "drag_coefficient": self.drag_coefficient,
            "height": self.height,
            "porosity": self.porosity,
        }

        if kwargs.get("include_outputs", False):
            data["noise_signal"] = self.noise_signal
            data["noise_function"] = (
                to_hex_encode(self.noise_function)
                if allow_pickle
                else self.noise_function.__name__
            )
            data["noisy_pressure_signal"] = self.noisy_pressure_signal
            data["clean_pressure_signal"] = self.clean_pressure_signal

        return data

    @classmethod
    def from_dict(cls, data):
        trigger = data["trigger"]

        try:
            trigger = from_hex_decode(trigger)
        except (TypeError, ValueError):
            pass

        parachute = cls(
            name=data["name"],
            parachute_type=data["parachute_type"],
            trigger=trigger,
            sampling_rate=data["sampling_rate"],
            lag=data["lag"],
            noise=data["noise"],
        )

        return parachute
