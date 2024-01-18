from rocketpy.rocket import Parachute

from .dispersion_model import DispersionModel


class McParachute(DispersionModel):
    """A Monte Carlo Parachute class that inherits from MonteCarloModel. This
    class is used to receive a Parachute object and information about the
    dispersion of its parameters and generate a random parachute object based
    on the provided information.

    Attributes
    ----------
    object : Parachute
        Parachute object to be used for validation.
    cd_s : tuple, list, int, float
        Drag coefficient of the parachute. Follows the standard input format of
        Dispersion Models.
    trigger : list
        List of callables, string "apogee" or ints/floats. Follows the standard
        input format of Dispersion Models.
    sampling_rate : tuple, list, int, float
        Sampling rate of the parachute in seconds. Follows the standard input
        format of Dispersion Models.
    lag : tuple, list, int, float
        Lag of the parachute in seconds. Follows the standard input format of
        Dispersion Models.
    noise : list
        List of tuples in the form of (mean, standard deviation,
        time-correlation). Follows the standard input format of Dispersion
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
        """Initializes the Monte Carlo Parachute class.

        See Also
        --------
        This should link to somewhere that explains how inputs works in
        dispersion models.

        Parameters
        ----------
        parachute : Parachute
            Parachute object to be used for validation.
        cd_s : tuple, list, int, float
            Drag coefficient of the parachute. Follows the standard input
            format of Dispersion Models.
        trigger : list
            List of callables, string "apogee" or ints/floats. Follows the
            standard input format of Dispersion Models.
        sampling_rate : tuple, list, int, float
            Sampling rate of the parachute in seconds. Follows the standard
            input format of Dispersion Models.
        lag : tuple, list, int, float
            Lag of the parachute in seconds. Follows the standard input format
            of Dispersion Models.
        noise : list
            List of tuples in the form of (mean, standard deviation,
            time-correlation). Follows the standard input format of Dispersion
            Models.
        """
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

    def _validate_trigger(self, trigger):
        """Validates the trigger input. If the trigger input argument is not
        None, it must be:
        - a list of callables, string "apogee" or ints/floats
        - a tuple that will be further validated in the DispersionModel class
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
        parachute = Parachute(
            name=generated_dict["name"],
            cd_s=generated_dict["cd_s"],
            trigger=generated_dict["trigger"],
            sampling_rate=generated_dict["sampling_rate"],
            lag=generated_dict["lag"],
            noise=generated_dict["noise"],
        )
        return parachute
