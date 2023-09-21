import numpy as np

from ..mathutils.function import Function
from ..prints.parachute_prints import _ParachutePrints


class Parachute:
    """Keeps parachute information.

    Attributes
    ----------
    Parachute.name : string
        Parachute name, such as drogue and main. Has no impact in
        simulation, as it is only used to display data in a more
        organized matter.
    Parachute.cd_s : float
        Drag coefficient times reference area for parachute. It is
        used to compute the drag force exerted on the parachute by
        the equation F = ((1/2)*rho*V^2)*cd_s, that is, the drag
        force is the dynamic pressure computed on the parachute
        times its cd_s coefficient. Has units of area and must be
        given in squared meters.
    Parachute.trigger : callable, float, str
        This parameter defines the trigger condition for the parachute ejection
        system. It can be one of the following:

        - A callable function that takes three arguments:
        1. Freestream pressure in pascals.
        2. Height in meters above ground level.
        3. The state vector of the simulation, which is defined as:

           `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.

        The function should return True if the parachute ejection system should
        be triggered and False otherwise.

        - A float value, representing an absolute height in meters. In this
        case, the parachute will be ejected when the rocket reaches this height
        above ground level.

        - The string "apogee," which triggers the parachute at apogee, i.e.,
        when the rocket reaches its highest point and starts descending.

        Note: The function will be called according to the sampling rate
        specified.
    Parachute.triggerfunc : function
        This parameter defines the trigger function created from the trigger
        parameter. It is used to evaluate the trigger condition for the
        parachute ejection system. It is a callable function that takes three
        arguments:

        1. Freestream pressure in pascals.
        2. Height in meters above ground level.
        3. The state vector of the simulation, which is defined as:

           `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.

        The function should return True if the parachute ejection system should
        be triggered and False otherwise.

        Note: The function will be called according to the sampling rate
        specified.
    Parachute.sampling_rate : float
        Sampling rate, in hertz, for the trigger function.
    Parachute.lag : float
        Time, in seconds, between the parachute ejection system is triggered
        and the parachute is fully opened.
    Parachute.noise_bias : float
        Mean value of the noise added to the pressure signal, which is
        passed to the trigger function. Unit is in pascal.
    Parachute.noise_deviation : float
        Standard deviation of the noise added to the pressure signal,
        which is passed to the trigger function. Unit is in pascal.
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
        cd_s,
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
        cd_s : float
            Drag coefficient times reference area of the parachute.
        trigger : callable, float, str
            This parameter defines the trigger condition for the parachute
            ejection system. It can be one of the following:

            - A callable function that takes three arguments:
                1. Freestream pressure in pascals.
                2. Height in meters above ground level.
                3. The state vector of the simulation, which is defined as:
                    [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].

            The function should return True if the parachute ejection system
            should be triggered and False otherwise.

            - A float value, representing an absolute height in meters. In this
            case, the parachute will be ejected when the rocket reaches this
            height above ground level.

            - The string "apogee," which triggers the parachute at apogee, i.e.,
            when the rocket reaches its highest point and starts descending.

            Note: The function will be called according to the sampling rate
            specified next.
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
            passed to the trigger function. Default value is (0, 0, 0). Units
            are in pascal.
        Returns
        -------
        None
        """
        self.name = name
        self.cd_s = cd_s
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
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
        self.noise_function = lambda: alpha * self.noise_signal[-1][
            1
        ] + beta * np.random.normal(noise[0], noise[1])

        self.prints = _ParachutePrints(self)

        # evaluate the trigger
        if callable(trigger):
            self.triggerfunc = trigger
        elif isinstance(trigger, (int, float)):
            # trigger is interpreted as the absolute height at which the parachute will be ejected
            def triggerfunc(p, h, y):
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return True if y[5] < 0 and h < trigger else False

            self.triggerfunc = triggerfunc

        elif trigger == "apogee":
            # trigger for apogee
            def triggerfunc(p, h, y):
                # p = pressure considering parachute noise signal
                # h = height above ground level considering parachute noise signal
                # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
                return True if y[5] < 0 else False

            self.triggerfunc = triggerfunc

        return None

    def __str__(self):
        """Returns a string representation of the Parachute class.

        Returns
        -------
        string
            String representation of Parachute class. It is human readable.
        """
        return "Parachute {} with a cd_s of {:.4f} m2".format(
            self.name.title(),
            self.cd_s,
        )

    def info(self):
        """Prints information about the Parachute class."""
        self.prints.all()

        return None

    def all_info(self):
        """Prints all information about the Parachute class."""
        self.info()
        # self.plots.all() # Parachutes still doesn't have plots

        return None
