from rocketpy.rocket import Parachute

from .stochastic_model import StochasticModel


class StochasticParachute(StochasticModel):
    """A Stochastic Parachute class that inherits from StochasticModel. This
    class is used to receive a Parachute object and information about the
    dispersion of its parameters and generate a random parachute object based
    on the provided information.

    See Also
    --------
    :ref:`stochastic_model`

    Attributes
    ----------
    object : Parachute
        Parachute object to be used for validation.
    cd_s : tuple, list, int, float
        Drag coefficient of the parachute. Follows the standard input format of
        Stochastic Models.
    trigger : list
        List of callables, string "apogee" or ints/floats. Follows the standard
        input format of Stochastic Models.
    sampling_rate : tuple, list, int, float
        Sampling rate of the parachute in seconds. Follows the standard input
        format of Stochastic Models.
    lag : tuple, list, int, float
        Lag of the parachute in seconds. Follows the standard input format of
        Stochastic Models.
    noise : list
        List of tuples in the form of (mean, standard deviation,
        time-correlation). Follows the standard input format of Stochastic
        Models.
    name : list
        List of names. This attribute can not be randomized.
    """

    def __init__(
        self,
        parachute,
        cd_s=None,
        trigger=None,
        sampling_rate=None,
        lag=None,
        noise=None,
    ):
        """Initializes the Stochastic Parachute class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        parachute : Parachute
            Parachute object to be used for validation.
        cd_s : tuple, list, int, float
            Drag coefficient of the parachute. Follows the standard input
            format of Stochastic Models.
        trigger : list
            List of callables, string "apogee" or ints/floats. Follows the
            standard input format of Stochastic Models.
        sampling_rate : tuple, list, int, float
            Sampling rate of the parachute in seconds. Follows the standard
            input format of Stochastic Models.
        lag : tuple, list, int, float
            Lag of the parachute in seconds. Follows the standard input format
            of Stochastic Models.
        noise : list
            List of tuples in the form of (mean, standard deviation,
            time-correlation). Follows the standard input format of Stochastic
            Models.
        """
        self.parachute = parachute
        self.cd_s = cd_s
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
        self.noise = noise

        self._validate_trigger(trigger)
        self._validate_noise(noise)
        super().__init__(
            parachute,
            cd_s=cd_s,
            trigger=trigger,
            sampling_rate=sampling_rate,
            lag=lag,
            noise=noise,
            name=None,
        )

    def __repr__(self):
        return (
            f"StochasticParachute("
            f"parachute={self.object}, "
            f"cd_s={self.cd_s}, "
            f"trigger={self.trigger}, "
            f"sampling_rate={self.sampling_rate}, "
            f"lag={self.lag}, "
            f"noise={self.noise})"
        )

    def _validate_trigger(self, trigger):
        """Validates the trigger input. If the trigger input argument is not
        None, it must be:
        - a list of callables, string "apogee" or ints/floats
        - a tuple that will be further validated in the StochasticModel class
        """
        if trigger is not None:
            assert isinstance(trigger, list) and all(
                isinstance(member, (str, int, float) or callable(member))
                for member in trigger
            ), "`trigger` must be a list of callables, string 'apogee' or ints/floats"

    def _validate_noise(self, noise):
        """Validates the noise input. If the noise input argument is not
        None, it must be a list of tuples in the form of
        (mean, standard deviation, time-correlation)
        """
        if noise is not None:
            assert isinstance(noise, list) and all(
                isinstance(member, tuple) for member in noise
            ), (
                "`noise` must be a list of tuples in the form of "
                "(mean, standard deviation, time-correlation)"
            )

    def create_object(self):
        """Creates and returns a Parachute object from the randomly generated
        input arguments.

        Returns
        -------
        parachute : Parachute
            Parachute object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        return Parachute(**generated_dict)
