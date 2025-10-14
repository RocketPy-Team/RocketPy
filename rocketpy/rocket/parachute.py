from inspect import signature

import numpy as np

from rocketpy.tools import from_hex_decode, to_hex_encode

from ..mathutils.function import Function
from ..prints.parachute_prints import _ParachutePrints


class Parachute:
    """Keeps information of the parachute, which is modeled as a hemispheroid.

    Attributes
    ----------
    Parachute.name : string
        Parachute name, such as drogue and main. Has no impact in
        simulation, as it is only used to display data in a more
        organized matter.
    Parachute.cd_s : float
        Drag coefficient times reference area for parachute. It has units of
        area and must be given in squared meters.
    Parachute.trigger : callable, float, str
        This parameter defines the trigger condition for the parachute ejection
        system. It can be one of the following:

        - A callable function that takes three arguments:
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
        that takes three arguments: Freestream pressure in Pa, Height above
        ground level in meters, and the state vector of the simulation. The
        returns ``True`` if the parachute ejection system should be triggered
        and ``False`` otherwise.

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
    Parachute.radius : float
        Length of the non-unique semi-axis (radius) of the inflated hemispheroid
        parachute in meters.
    Parachute.height : float, None
        Length of the unique semi-axis (height) of the inflated hemispheroid
        parachute in meters.
    Parachute.porosity : float
        Geometric porosity of the canopy (ratio of open area to total canopy area),
        in [0, 1]. Affects only the added-mass scaling during descent; it does
        not change ``cd_s`` (drag). The default, 0.0432, yields an added-mass
        of 1.0 (“neutral” behavior).
    Parachute.added_mass_coefficient : float
        Coefficient used to calculate the added-mass due to dragged air. It is
        calculated from the porosity of the parachute.
    """

    def __init__(
        self,
        name,
        cd_s,
        trigger,
        sampling_rate,
        lag=0,
        noise=(0, 0, 0),
        radius=1.5,
        height=None,
        porosity=0.0432,
    ):
        """Initializes Parachute class.

        Parameters
        ----------
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        cd_s : float
            Drag coefficient times reference area of the parachute.
        trigger : callable, float, str
            Defines the trigger condition for the parachute ejection system. It
            can be one of the following:

            - A callable function that takes three arguments: \

                1. Freestream pressure in pascals.
                2. Height in meters above ground level.
                3. The state vector of the simulation, which is defined as: \

                    .. code-block:: python

                        u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

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
        radius : float, optional
            Length of the non-unique semi-axis (radius) of the inflated hemispheroid
            parachute. Default value is 1.5.
            Units are in meters.
        height : float, optional
            Length of the unique semi-axis (height) of the inflated hemispheroid
            parachute. Default value is the radius of the parachute.
            Units are in meters.
        porosity : float, optional
            Geometric porosity of the canopy (ratio of open area to total canopy area),
            in [0, 1]. Affects only the added-mass scaling during descent; it does
            not change ``cd_s`` (drag). The default, 0.0432, yields an added-mass
            of 1.0 (“neutral” behavior).
        """
        self.name = name
        self.cd_s = cd_s
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
        self.noise = noise
        self.noise_signal = [[-1e-6, np.random.normal(noise[0], noise[1])]]
        self.noisy_pressure_signal = []
        self.clean_pressure_signal = []
        self.noise_bias = noise[0]
        self.noise_deviation = noise[1]
        self.noise_corr = (noise[2], (1 - noise[2] ** 2) ** 0.5)
        self.clean_pressure_signal_function = Function(0)
        self.noisy_pressure_signal_function = Function(0)
        self.noise_signal_function = Function(0)
        self.radius = radius
        self.height = height or radius
        self.porosity = porosity
        self.added_mass_coefficient = 1.068 * (
            1
            - 1.465 * self.porosity
            - 0.25975 * self.porosity**2
            + 1.2626 * self.porosity**3
        )

        alpha, beta = self.noise_corr
        self.noise_function = lambda: alpha * self.noise_signal[-1][
            1
        ] + beta * np.random.normal(noise[0], noise[1])

        self.prints = _ParachutePrints(self)

        self.__evaluate_trigger_function(trigger)

    def __evaluate_trigger_function(self, trigger):
        """This is used to set the triggerfunc attribute that will be used to
        interact with the Flight class.
        """
        # pylint: disable=unused-argument, function-redefined
        # The parachute is deployed by a custom function
        if callable(trigger):
            # work around for having added sensors to parachute triggers
            # to avoid breaking changes
            triggerfunc = trigger
            sig = signature(triggerfunc)
            if len(sig.parameters) == 3:

                def triggerfunc(p, h, y, sensors):
                    return trigger(p, h, y)

            self.triggerfunc = triggerfunc

        elif isinstance(trigger, (int, float)):
            # The parachute is deployed at a given height
            def triggerfunc(p, h, y, sensors):  # pylint: disable=unused-argument
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return y[5] < 0 and h < trigger

            self.triggerfunc = triggerfunc

        elif trigger.lower() == "apogee":
            # The parachute is deployed at apogee
            def triggerfunc(p, h, y, sensors):  # pylint: disable=unused-argument
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return y[5] < 0

            self.triggerfunc = triggerfunc

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
        return f"Parachute {self.name.title()} with a cd_s of {self.cd_s:.4f} m2"

    def __repr__(self):
        """Representation method for the class, useful when debugging."""
        return (
            f"<Parachute {self.name} "
            + f"(cd_s = {self.cd_s:.4f} m2, trigger = {self.trigger})>"
        )

    def info(self):
        """Prints information about the Parachute class."""
        self.prints.all()

    def all_info(self):
        """Prints all information about the Parachute class."""
        self.info()
        # self.plots.all() # Parachutes still doesn't have plots

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
            "cd_s": self.cd_s,
            "trigger": trigger,
            "sampling_rate": self.sampling_rate,
            "lag": self.lag,
            "noise": self.noise,
            "radius": self.radius,
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
            cd_s=data["cd_s"],
            trigger=trigger,
            sampling_rate=data["sampling_rate"],
            lag=data["lag"],
            noise=data["noise"],
            radius=data.get("radius", 1.5),
            height=data.get("height", None),
            porosity=data.get("porosity", 0.0432),
        )

        return parachute
