from .dispersion_model import DispersionModel


class MotorDispersionModel(DispersionModel):
    """Motor Dispersion Model class, used to validate the input parameters of
    the motor. It uses the DispersionModel class as a base class, see its
    documentation for more information.
    """

    def __init__(self, object, **kwargs):
        self._validate_1d_array_like("thrust_source", kwargs.get("thrust_source"))
        self._validate_positive_int_list("grain_number", kwargs.get("grain_number"))
        super().__init__(object, **kwargs)

    # TODO: how to validate this? seems that burn_time has to start at 0
    # to work with reshape thrust curve
    def _validate_burn_time(self, burn_time):
        pass
