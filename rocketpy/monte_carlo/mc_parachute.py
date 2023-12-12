from rocketpy.rocket import Parachute

from .dispersion_model import DispersionModel


class McParachute(DispersionModel):
    def __init__(
        self,
        parachute,
        cd_s=None,
        trigger=None,
        sampling_rate=None,
        lag=None,
        noise=None,
    ):
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
