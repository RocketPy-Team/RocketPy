"""Defines the StochasticMotorModel class."""

from .stochastic_model import StochasticModel


class StochasticMotorModel(StochasticModel):
    """Stochastic Motor Model class that inherits from StochasticModel. This
    class makes a common ground for other stochastic motor classes.

    See Also
    --------
    :ref:`stochastic_model`
    """

    def __init__(self, obj, **kwargs):
        self._validate_1d_array_like("thrust_source", kwargs.get("thrust_source"))
        # TODO: never vary the grain_number
        self._validate_positive_int_list("grain_number", kwargs.get("grain_number"))
        super().__init__(obj, **kwargs)
