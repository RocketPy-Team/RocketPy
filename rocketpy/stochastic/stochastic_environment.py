"""Defines the StochasticEnvironment class."""

from .stochastic_model import StochasticModel


class StochasticEnvironment(StochasticModel):
    """A Stochastic Environment class that inherits from StochasticModel. This
    class is used to store an Environment object and the uncertainty of their
    parameters.

    See Also
    --------
    :ref:`stochastic_model` and :class:`Environment <rocketpy.environment.Environment>`

    Attributes
    ----------
    object : Environment
        Environment object to be used as a base for the stochastic model.
    elevation : tuple, list, int, float
        Elevation of the launch site in meters.
    gravity : tuple, list, int, float
        Gravitational acceleration in meters per second squared.
    latitude : tuple, list, int, float
        Latitude of the launch site in degrees.
    longitude : tuple, list, int, float
        Longitude of the launch site in degrees.
    ensemble_member : list[int]
        List of integers representing the ensemble members to be selected.
    date : list[tuple]
        List of dates, which are tuples of four elements
        (year, month, day, hour). This attribute can not be randomized.
    datum : list[str]
        List of datum. This attribute can not be randomized.
    timezone : list[pytz.timezone]
        List with the timezone. This attribute can not be randomized.
    wind_velocity_x_factor : tuple, list, int, float
        Factor to multiply the wind velocity in the x direction. This should
        be used only when the wind velocity is defined as a constant value.
    wind_velocity_y_factor : tuple, list, int, float
        Factor to multiply the wind velocity in the y direction. This should
        be used only when the wind velocity is defined as a constant value.
    """

    def __init__(
        self,
        environment,
        elevation=None,
        gravity=None,
        latitude=None,
        longitude=None,
        ensemble_member=None,
        wind_velocity_x_factor=(1, 0),
        wind_velocity_y_factor=(1, 0),
    ):
        """Initializes the Stochastic Environment class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        environment : Environment
            Environment object to be used for validation.
        date : list, optional
            List of dates, which are tuples of four elements
            (year, month, day, hour).
        elevation : int, float, tuple, list, optional
            Elevation of the launch site in meters.
        gravity : int, float, tuple, list, optional
            Gravitational acceleration in meters per second squared.
        latitude : int, float, tuple, list, optional
            Latitude of the launch site in degrees.
        longitude : int, float, tuple, list, optional
            Longitude of the launch site in degrees.
        ensemble_member : list, optional
            List of integers representing the ensemble member to be selected.
        wind_velocity_x_factor : tuple, list, int, float, optional
            Factor to multiply the wind velocity in the x direction. This should
            be used only when the wind velocity is defined as a constant value.
        wind_velocity_y_factor : tuple, list, int, float, optional
            Factor to multiply the wind velocity in the y direction. This should
            be used only when the wind velocity is defined as a constant value.
        """

        super().__init__(
            environment,
            date=None,
            elevation=elevation,
            gravity=gravity,
            latitude=latitude,
            longitude=longitude,
            ensemble_member=ensemble_member,
            wind_velocity_x_factor=wind_velocity_x_factor,
            wind_velocity_y_factor=wind_velocity_y_factor,
            datum=None,
            timezone=None,
        )
        self._validate_ensemble(ensemble_member, environment)

    def _validate_ensemble(self, ensemble_member, environment):
        """Validates the ensemble member input argument. If the environment
        does not have ensemble members, the ensemble member input argument
        must be None. If the environment has ensemble members, the ensemble
        member input argument must be a list of positive integers, and the
        integers must be in the range from 0 to the number of ensemble members
        minus one.

        Parameters
        ----------
        ensemble_member : list
            List of integers representing the ensemble members to be selected.
        environment : Environment
            Environment object to be used for validation.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the environment does not have ensemble members and the
            ensemble_member input argument is not None.
        """
        valid_atmospheric_types = ["Ensemble", "Reanalysis"]

        if environment.atmospheric_model_type not in valid_atmospheric_types:
            if ensemble_member is not None:
                raise AssertionError(
                    f"Environment with {environment.atmospheric_model_type} "
                    "does not have ensemble members"
                )
            return

        if ensemble_member is not None:
            assert isinstance(ensemble_member, list), "`ensemble_member` must be a list"
            assert all(
                isinstance(member, int) and member >= 0 for member in ensemble_member
            ), "`ensemble_member` must be a list of positive integers"
            assert (
                0
                <= min(ensemble_member)
                <= max(ensemble_member)
                < environment.num_ensemble_members
            ), (
                "`ensemble_member` must be in the range from 0 to "
                + f"{environment.num_ensemble_members - 1}"
            )
            setattr(self, "ensemble_member", ensemble_member)
        else:
            # if no ensemble member is provided, get it from the environment
            setattr(self, "ensemble_member", environment.ensemble_member)

    def create_object(self):
        """Creates an Environment object from the randomly generated input
        arguments. The environment object is not recreated to avoid having to
        reestablish the atmospheric model. Instead, attributes are changed
        directly.

        Parameters
        ----------
        None

        Returns
        -------
        Environment
            Environment object with the randomly generated input arguments.

        Notes
        -----
        This method is overwriting the create_object method from the
        `StochasticModel` class to handle the special case of the ensemble
        member attribute.
        """
        generated_dict = next(self.dict_generator())
        for key, value in generated_dict.items():
            # special case for ensemble member
            # TODO: Generalize create_object() with a env.ensemble_member setter
            if key == "ensemble_member":
                self.obj.select_ensemble_member(value)
            else:
                if "factor" in key:
                    # get original attribute value and multiply by factor
                    attribute_name = f"_{key.replace('_factor', '')}"
                    value = getattr(self, attribute_name) * value
                    key = f"{key.replace('_factor', '')}"
                setattr(self.obj, key, value)
        return self.obj
