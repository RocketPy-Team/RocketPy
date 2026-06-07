import warnings
from inspect import signature

import numpy as np

from rocketpy.tools import from_hex_decode, to_hex_encode

from ..mathutils.function import Function, funcify_method
from ..prints.parachute_prints import _ParachutePrints
from ..simulation.events.event import Event


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

        - A callable function. The recommended signature accepts ``**kwargs``
          only and returns ``True`` if the parachute ejection system should be
          triggered and ``False`` otherwise. The parachute is wrapped in an
          :class:`rocketpy.Event`, so the function receives the same keyword
          arguments as any event trigger, including ``state``, ``pressure``,
          ``height_above_ground_level``, ``sensors``, ``time``, ``flight``,
          ``rocket`` and ``environment``. See the Event documentation for the
          full list. The function is called according to the specified sampling
          rate.

          .. deprecated:: 1.13
              Defining the trigger with positional arguments
              ``(p, h, y[, sensors])`` is deprecated and emits a
              ``DeprecationWarning``; use a ``**kwargs``-only signature instead.

        - A float value, representing an absolute height in meters. In this
          case, the parachute will be ejected when the rocket reaches this height
          above ground level.

        - The string "apogee" which triggers the parachute at apogee, i.e.,
          when the rocket reaches its highest point and starts descending.


    Parachute.triggerfunc : function
        Trigger function created from the trigger used to evaluate the trigger
        condition for the parachute ejection system. It is a callable that
        receives the event keyword arguments (pressure, height above ground
        level, state vector, sensors, etc.) and returns ``True`` if the
        parachute ejection system should be triggered and ``False`` otherwise.

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
    Parachute.drag_coefficient : float
        Drag coefficient of the inflated canopy shape, used only when
        ``radius`` is not provided to estimate the parachute radius from
        ``cd_s``: ``R = sqrt(cd_s / (drag_coefficient * pi))``. Typical
        values: 1.4 for hemispherical canopies (default), 0.75 for flat
        circular canopies, 1.5 for extended-skirt canopies.
    Parachute.radius : float
        Length of the non-unique semi-axis (radius) of the inflated hemispheroid
        parachute in meters. If not provided at construction time, it is
        estimated from ``cd_s`` and ``drag_coefficient``.
    Parachute.height : float
        Length of the unique semi-axis (height) of the inflated hemispheroid
        parachute in meters.
    Parachute.porosity : float
        Geometric porosity of the canopy (ratio of open area to total canopy
        area), in [0, 1]. Affects only the added-mass scaling during descent;
        it does not change ``cd_s`` (drag). The default value of 0.0432 is
        chosen so that the resulting ``added_mass_coefficient`` equals
        approximately 1.0 ("neutral" added-mass behavior).
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
        radius=None,
        height=None,
        porosity=0.0432,
        drag_coefficient=1.4,
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

            - A callable function. The recommended signature accepts \
                ``**kwargs`` only and returns ``True`` if the parachute \
                ejection system should be triggered and ``False`` otherwise. \
                The parachute is wrapped in an :class:`rocketpy.Event`, so the \
                function receives the same keyword arguments as any event \
                trigger, including ``state`` (the state vector \
                ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``), \
                ``pressure``, ``height_above_ground_level``, ``sensors``, \
                ``time``, ``flight``, ``rocket`` and ``environment``. See the \
                Event documentation for the full list.
            - A float value, representing an absolute height in meters. In this \
                case, the parachute will be ejected when the rocket reaches this \
                height above ground level.
            - The string "apogee" which triggers the parachute at apogee, i.e., \
                when the rocket reaches its highest point and starts descending.
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
            Length of the non-unique semi-axis (radius) of the inflated
            hemispheroid parachute. If not provided, it is estimated from
            ``cd_s`` and ``drag_coefficient`` using:
            ``radius = sqrt(cd_s / (drag_coefficient * pi))``.
            Units are in meters.
        height : float, optional
            Length of the unique semi-axis (height) of the inflated hemispheroid
            parachute. Default value is the radius of the parachute.
            Units are in meters.
        porosity : float, optional
            Geometric porosity of the canopy (ratio of open area to total
            canopy area), in [0, 1]. Affects only the added-mass scaling
            during descent; it does not change ``cd_s`` (drag). The default
            value of 0.0432 is chosen so that the resulting
            ``added_mass_coefficient`` equals approximately 1.0 ("neutral"
            added-mass behavior).
        drag_coefficient : float, optional
            Drag coefficient of the inflated canopy shape, used only when
            ``radius`` is not provided. It relates the aerodynamic ``cd_s``
            to the physical canopy area via
            ``cd_s = drag_coefficient * pi * radius**2``. Typical values:

            - **1.4** — hemispherical canopy (default, NASA SP-8066)
            - **0.75** — flat circular canopy
            - **1.5** — extended-skirt canopy

            Has no effect when ``radius`` is explicitly provided.
        """

        # Save arguments as attributes
        self.name = name
        self.cd_s = cd_s
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
        self.noise = noise
        self.drag_coefficient = drag_coefficient
        self.porosity = porosity

        # Initialize derived attributes
        self.radius = self.__resolve_radius(radius, cd_s, drag_coefficient)
        self.height = height or self.radius
        self.added_mass_coefficient = 1.068 * (
            1 - 1.465 * porosity - 0.25975 * porosity**2 + 1.2626 * porosity**3
        )
        self.__init_noise(noise)
        self.__evaluate_trigger_function(trigger)
        self.event = self.to_event()

        # Prints and plots
        self.prints = _ParachutePrints(self)

    def __resolve_radius(self, radius, cd_s, drag_coefficient):
        """Resolves parachute radius from input or aerodynamic relation."""
        if radius is not None:
            return radius

        # cd_s = Cd * S = Cd * pi * R^2  =>  R = sqrt(cd_s / (Cd * pi))
        return np.sqrt(cd_s / (drag_coefficient * np.pi))

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
        alpha, beta = self.noise_corr
        self.noise_function = lambda: (
            alpha * self.noise_signal[-1][1]
            + beta * np.random.normal(noise[0], noise[1])
        )

    def __evaluate_trigger_function(self, trigger):
        """This is used to set the triggerfunc attribute that will be used to
        interact with the Flight class.
        """
        # Case 1: The parachute is deployed by a custom function
        if callable(trigger):
            sig = signature(trigger)
            parameters = list(sig.parameters.values())
            positional_param_count = sum(
                parameter.kind
                in (
                    parameter.POSITIONAL_ONLY,
                    parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in parameters
            )
            accepts_var_positional = any(
                parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
            )
            accepts_var_keyword = any(
                parameter.kind == parameter.VAR_KEYWORD for parameter in parameters
            )
            keyword_only_names = {
                parameter.name
                for parameter in parameters
                if parameter.kind == parameter.KEYWORD_ONLY
            }
            accepts_sensors_positional = (
                accepts_var_positional or positional_param_count >= 4
            )

            if positional_param_count > 0 or accepts_var_positional:
                warnings.warn(
                    "It is recommended not to use positional arguments "
                    "(e.g. `trigger(p, h, y)`) when defining a parachute "
                    "trigger. Instead, define the trigger to accept `**kwargs` "
                    "only and read values such as `kwargs['pressure']`, "
                    "`kwargs['height_above_ground_level']` and "
                    "`kwargs['state']`. See the Event documentation "
                    "for the full list of available keyword arguments.",
                    UserWarning,
                    stacklevel=2,
                )

            def triggerfunc(p, h, y, sensors, **kwargs):
                positional_args = [p, h, y]
                if accepts_sensors_positional:
                    positional_args.append(sensors)

                if accepts_var_keyword:
                    return trigger(*positional_args, **kwargs)

                forwarded_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in keyword_only_names
                }
                return trigger(*positional_args, **forwarded_kwargs)

            self.triggerfunc = triggerfunc

        # Case 2: The parachute is deployed at a given height
        elif isinstance(trigger, (int, float)):
            # The parachute is deployed at a given height
            def triggerfunc(p, h, y, sensors, **kwargs):
                _ = sensors
                _ = kwargs
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return y[5] < 0 and h < trigger

            self.triggerfunc = triggerfunc

        # Case 3: The parachute is deployed at apogee
        elif trigger.lower() == "apogee":
            def triggerfunc(p, h, y, sensors, **kwargs):
                _ = p
                _ = h
                _ = sensors
                state_history = kwargs.get("state_history")
                if not state_history:
                    return False
                previous_vz = state_history[-1][5]
                current_vz = y[5]
                return previous_vz > 0 >= current_vz

            self.triggerfunc = triggerfunc

        # Case 4: Invalid trigger input
        else:
            raise ValueError(
                f"Unable to set the trigger function for parachute '{self.name}'. "
                + "Trigger must be a callable, a float value or the string 'apogee'. "
                + "See the Parachute class documentation for more information."
            )

    def to_event(self):
        """Create an Event wrapper for this parachute trigger/callback pair.

        Returns
        -------
        Event
            Event object that encapsulates parachute trigger evaluation and
            deployment phase transitions.
        """

        def trigger_event_wrapper(**kwargs):
            """Bridge Event kwargs to parachute trigger signature."""
            pressure = kwargs.get("pressure")
            height_above_ground_level = kwargs.get("height_above_ground_level")
            state = kwargs.get("state")
            sensors = kwargs.get("sensors")
            time = kwargs.get("time")
            kwargs = {key: value for key, value in kwargs.items() if key != "sensors"}

            noise = self.noise_function()
            noisy_pressure = pressure + noise

            # TODO: can we deprecate these attributes?
            self.noise_signal.append([time, noise])
            self.clean_pressure_signal.append([time, pressure])
            self.noisy_pressure_signal.append([time, noisy_pressure])

            # TODO: include **kwargs and change docs to suggest using everything
            # via kwargs instead of specific arguments
            return self.triggerfunc(
                noisy_pressure,
                height_above_ground_level,
                state,
                sensors,
                **kwargs,
            )

        def trigger_callback(**kwargs):
            time = kwargs.get("time", None)
            flight = kwargs.get("flight", None)

            flight._active_parachute = self
            flight.parachute_events.append([time, self])

            kwargs["event"].commands.set_derivative(flight.u_dot_parachute)
            kwargs["event"].commands.start_flight_phase(
                f"{self.name}_parachute_descent",
                lag=self.lag,
                parachute=self,
            )

        # TODO: add exact time computation for parachute trigger
        # TODO: if sampling rate is none time_overshootable should be false
        return Event(
            trigger=trigger_event_wrapper,
            callback=trigger_callback,
            name=f"{self.name} Parachute Deployment",
            sampling_rate=self.sampling_rate,
            trigger_only_once=True,
            time_overshootable=True,
            priority=2,
        )

    def _reset_signals(self):
        """Resets the noise, clean pressure signal and noisy pressure signal
        attributes to their initial state. This is used when running multiple
        simulations with the same parachute object.
        """
        self.noise_signal = [
            [-1e-6, np.random.normal(self.noise_bias, self.noise_deviation)]
        ]
        self.noisy_pressure_signal = []
        self.clean_pressure_signal = []
        try:
            del self.noise_signal_function
            del self.clean_pressure_signal_function
            del self.noisy_pressure_signal_function
        except AttributeError:
            pass

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

    @funcify_method("Time (s)", "Pressure Noise (Pa)", "linear", "constant")
    def noise_signal_function(self):
        return self.noise_signal

    @funcify_method("Time (s)", "Pressure - Without Noise (Pa)", "linear", "constant")
    def clean_pressure_signal_function(self):
        return Function(self.clean_pressure_signal)

    @funcify_method("Time (s)", "Pressure - With Noise (Pa)", "linear", "constant")
    def noisy_pressure_signal_function(self):
        return self.clean_pressure_signal_function + self.noise_signal_function

    def info(self):
        """Prints information about the Parachute class."""
        self.prints.all()

    def all_info(self):
        """Prints all information about the Parachute class."""
        self.info()
        # self.plots.all() # TODO: Parachutes still doesn't have plots

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
            cd_s=data["cd_s"],
            trigger=trigger,
            sampling_rate=data["sampling_rate"],
            lag=data["lag"],
            noise=data["noise"],
            radius=data.get("radius", None),
            drag_coefficient=data.get("drag_coefficient", 1.4),
            height=data.get("height", None),
            porosity=data.get("porosity", 0.0432),
        )

        return parachute
