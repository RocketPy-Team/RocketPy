from .dispersion_model import DispersionModel


class MotorDispersionModel(DispersionModel):
    """Monte Carlo Motor Model class that inherits from MonteCarloModel. This
    class is used to standardize the input of the motor dispersion model.
    """

    def __init__(self, object, **kwargs):
        self._validate_1d_array_like("thrust_source", kwargs.get("thrust_source"))
        self._validate_positive_int_list("grain_number", kwargs.get("grain_number"))
        super().__init__(object, **kwargs)
