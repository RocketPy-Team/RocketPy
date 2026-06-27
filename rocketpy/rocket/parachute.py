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

        - A callable function that can take 3, 4, or 5 arguments:

          **3 arguments**:
            1. Freestream pressure in pascals.
            2. Height in meters above ground level.
            3. The state vector: ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``

          **4 arguments** (sensors OR acceleration):
            1. Freestream pressure in pascals.
            2. Height in meters above ground level.
            3. The state vector.
            4. Either:
               - ``sensors``: List of sensor objects attached to the rocket, OR
               - ``u_dot``: State derivative including accelerations at indices [3:6]

          **5 arguments** (sensors AND acceleration):
            1. Freestream pressure in pascals.
            2. Height in meters above ground level.
            3. The state vector.
            4. ``sensors``: List of sensor objects.
            5. ``u_dot``: State derivative with accelerations ``[vx, vy, vz, ax, ay, az, ...]``

          The function should return ``True`` to trigger deployment, ``False`` otherwise.
          The function will be called according to the specified sampling rate.

        - A float value, representing an absolute height in meters. In this
          case, the parachute will be ejected when the rocket reaches this height
          above ground level while descending.

        - The string "apogee" which triggers the parachute at apogee, i.e.,
          when the rocket reaches its highest point and starts descending.


    Parachute.triggerfunc : function
        Trigger function created from the trigger used to evaluate the trigger
        condition for the parachute ejection system. It is a callable function
        that takes five arguments: Freestream pressure in Pa, Height above
        ground level in meters, the state vector, sensors list, and u_dot.
        Returns ``True`` if the parachute ejection system should be triggered
        and ``False`` otherwise.

        .. note::

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
        self.height = self.__resolve_height(height, self.radius)
        self.added_mass_coefficient = self.__compute_added_mass_coefficient(
            self.porosity
        )
        self.__init_noise(noise)
        self.__evaluate_trigger_function(trigger)

        # Prints and plots
        self.prints = _ParachutePrints(self)

    def __resolve_radius(self, radius, cd_s, drag_coefficient):
        """Resolves parachute radius from input or aerodynamic relation."""
        if radius is not None:
            return radius

        # cd_s = Cd * S = Cd * pi * R^2  =>  R = sqrt(cd_s / (Cd * pi))
        return np.sqrt(cd_s / (drag_coefficient * np.pi))

    def __resolve_height(self, height, radius):
        """Resolves parachute height defaulting to radius when not provided."""
        return height or radius

    def __compute_added_mass_coefficient(self, porosity):
        """Computes the added-mass coefficient from canopy porosity."""
        return 1.068 * (
            1 - 1.465 * porosity - 0.25975 * porosity**2 + 1.2626 * porosity**3
        )

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

    def __evaluate_trigger_function(self, trigger):  # pylint: disable=too-many-statements
        """This is used to set the triggerfunc attribute that will be used to
        interact with the Flight class.

        Notes
        -----
        The resulting triggerfunc always has signature (p, h, y, sensors, u_dot)
        so Flight can pass both sensors and u_dot when needed.
        """
        # pylint: disable=unused-argument, function-redefined

        # Helper to wrap any callable to the internal (p, h, y, sensors, u_dot) API
        def _make_wrapper(fn):
            sig = signature(fn)
            params = list(sig.parameters.keys())

            # detect if user function expects acceleration-like argument
            expects_udot = any(
                name.lower() in ("u_dot", "udot", "acc", "acceleration")
                for name in params[3:]
            )

            def wrapper(p, h, y, sensors, u_dot):
                # Support 3, 4, and 5-arg user functions
                num_params = len(sig.parameters)
                if num_params == 3:
                    return fn(p, h, y)
                if num_params == 4:
                    # Check which 4th arg to pass
                    fourth_param = params[3].lower()
                    if fourth_param in ("u_dot", "udot", "acc", "acceleration"):
                        return fn(p, h, y, u_dot)
                    else:
                        return fn(p, h, y, sensors)
                if num_params >= 5:
                    # Pass both sensors and u_dot
                    return fn(p, h, y, sensors, u_dot)
                # If function signature is not supported, raise an error
                raise TypeError(
                    f"Trigger function '{fn.__name__}' has unsupported signature: "
                    f"expected 3, 4, or 5+ arguments, got {num_params}. "
                    "Please check the function definition."
                )

            # attach metadata so Flight can decide whether to compute u_dot
            wrapper._expects_udot = expects_udot
            return wrapper

        # Callable provided by user
        if callable(trigger):
            self.triggerfunc = _make_wrapper(trigger)
            return

        # Numeric altitude trigger
        if isinstance(trigger, (int, float)):

            def triggerfunc(p, h, y, sensors, u_dot):  # pylint: disable=unused-argument
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return y[5] < 0 and h < trigger

            triggerfunc._expects_udot = False
            self.triggerfunc = triggerfunc
            return

        # Special case: "apogee"
        if isinstance(trigger, str) and trigger.lower() == "apogee":

            def triggerfunc(p, h, y, sensors, u_dot):  # pylint: disable=unused-argument
                return y[5] < 0

            triggerfunc._expects_udot = False
            self.triggerfunc = triggerfunc
            return

        # If we reach this point, the trigger is invalid
        raise ValueError(
            f"Unable to set the trigger function for parachute '{self.name}'. "
            + "Trigger must be a callable, a float value or one of the strings "
            + "('apogee'). "
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
