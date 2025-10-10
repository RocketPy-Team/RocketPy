"""Defines the StochasticParachute class."""

from rocketpy.rocket import Parachute

from .stochastic_model import StochasticModel


class StochasticParachute(StochasticModel):
    """A Stochastic Parachute class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`Parachute <rocketpy.rocket.Parachute>`

    Attributes
    ----------
    object : Parachute
        Parachute object to be used for validation.
    cd_s : tuple, list, int, float
        Drag coefficient of the parachute.
    trigger : list
        List of callables, string "apogee" or ints/floats.
    sampling_rate : tuple, list, int, float
        Sampling rate of the parachute in seconds.
    lag : tuple, list, int, float
        Lag of the parachute in seconds.
    noise : list[tuple]
        List of tuples in the form of (mean, standard deviation,
        time-correlation).
    name : list[str]
        List with the name of the parachute object. This cannot be randomized.
    radius : tuple, list, int, float
        Radius of the parachute in meters.
    height : tuple, list, int, float
        Height of the parachute in meters.
    porosity : tuple, list, int, float
        Porosity of the parachute.
    """

    def __init__(
        self,
        parachute,
        cd_s=None,
        trigger=None,
        sampling_rate=None,
        lag=None,
        noise=None,
        radius=None,
        height=None,
        porosity=None,
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
            Drag coefficient of the parachute.
        trigger : list
            List of callables, string "apogee" or ints/floats.
        sampling_rate : tuple, list, int, float
            Sampling rate of the parachute in seconds.
        lag : tuple, list, int, float
            Lag of the parachute in seconds. Pay special attention to ensure
            the lag will not assume negative values based on its mean and
            standard deviation.
        noise : list
            List of tuples in the form of (mean, standard deviation,
            time-correlation).
        radius : tuple, list, int, float
            Radius of the parachute in meters.
        height : tuple, list, int, float
            Height of the parachute in meters.
        porosity : tuple, list, int, float
            Porosity of the parachute.
        """
        self.parachute = parachute
        self.cd_s = cd_s
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
        self.noise = noise
        self.radius = radius
        self.height = height
        self.porosity = porosity

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
            radius=radius,
            height=height,
            porosity=porosity,
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
