from .dispersion_model import DispersionModel


class MotorDispersionModel(DispersionModel):
    """Motor Dispersion Model class, used to validate the input parameters of
    the motor. It uses the DispersionModel class as a base class, see its
    documentation for more information.
    """

    def __init__(self, object, **kwargs):
        self._validate_1d_array_like("thrust_source", kwargs.get("thrust_source"))
        self._validate_grain_number(kwargs.get("grain_number"))
        super().__init__(object, **kwargs)

    def _validate_grain_number(self, grain_number):
        """Validates the grain number input. If the grain number input argument
        is not None, it must be a list of positive integers.

        Parameters
        ----------
        grain_number : list
            List of integers representing the grain number to be selected.

        Raises
        ------
        AssertionError
            If `grain_number` is not None and is not a list of positive integers.
        """
        if grain_number is not None:
            assert isinstance(grain_number, list) and all(
                isinstance(member, int) and member >= 0 for member in grain_number
            ), "`grain_number` must be a list of positive integers"

    # TODO: how to validate this? seems that burn_time has to start at 0
    # to work with reshape thrust curve
    def _validate_burn_time(self, burn_time):
        pass
