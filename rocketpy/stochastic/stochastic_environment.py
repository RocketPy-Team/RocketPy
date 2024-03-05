from .stochastic_model import StochasticModel


class StochasticEnvironment(StochasticModel):
    """A Stochastic Environment class that inherits from StochasticModel. This
    class is used to receive a Environment object and information about the
    dispersion of its parameters and generate a random environment object based
    on the provided information.

    Attributes
    ----------
    object : Environment
        Environment object to be used for validation.
    elevation : tuple, list, int, float
        Elevation of the launch site in meters. Follows the standard input
        format of Stochastic Models.
    gravity : tuple, list, int, float
        Gravitational acceleration in meters per second squared. Follows the
        standard input format of Stochastic Models.
    latitude : tuple, list, int, float
        Latitude of the launch site in degrees. Follows the standard input
        format of Stochastic Models.
    longitude : tuple, list, int, float
        Longitude of the launch site in degrees. Follows the standard input
        format of Stochastic Models.
    ensemble_member : list
        List of integers representing the ensemble member to be selected.
    wind_velocity_x_factor : tuple, list, int, float
        Factor to be multiplied by the wind velocity in the x direction.
    wind_velocity_y_factor : tuple, list, int, float
        Factor to be multiplied by the wind velocity in the y direction.
    date : list
        List of dates, which are tuples of four elements
        (year, month, day, hour). This attribute can not be randomized.
    datum : list
        List of datum. This attribute can not be randomized.
    timezone : list
        List of timezones. This attribute can not be randomized.
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
        This should link to somewhere that explains how inputs works in
        Stochastic models.

        Parameters
        ----------
        environment : Environment
            Environment object to be used for validation.
        date : list, optional
            List of dates, which are tuples of four elements
            (year, month, day, hour).
        elevation : int, float, tuple, list, optional
            Elevation of the launch site in meters. Follows the standard
            input format of Stochastic Models.
        gravity : int, float, tuple, list, optional
            Gravitational acceleration in meters per second squared. Follows
            the standard input format of Stochastic Models.
        latitude : int, float, tuple, list, optional
            Latitude of the launch site in degrees. Follows the standard
            input format of Stochastic Models.
        longitude : int, float, tuple, list, optional
            Longitude of the launch site in degrees. Follows the standard
            input format of Stochastic Models.
        ensemble_member : list, optional
            List of integers representing the ensemble member to be selected.
        wind_velocity_x_factor : int, float, tuple, list, optional
            Factor to be multiplied by the wind velocity in the x direction.
            Follows the factor input format of Stochastic Models.
        wind_velocity_y_factor : int, float, tuple, list, optional
            Factor to be multiplied by the wind velocity in the y direction.
            Follows the factor input format of Stochastic Models.
        """
        # Validate in StochasticModel
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

    def __str__(self):
        # special str for environment because of datetime
        s = ""
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # Skip attributes starting with underscore
            if isinstance(value, tuple):
                try:
                    # Format the tuple as a string with the mean and standard deviation.
                    value_str = f"{value[0]:.5f} ± {value[1]:.5f} (numpy.random.{value[2].__name__})"
                except AttributeError:
                    # treats date attribute
                    value_str = str(value)
            else:
                # Otherwise, just use the default string representation of the value.
                value_str = str(value)
            s += f"{key}: {value_str}\n"
        return s.strip()

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
            List of integers representing the ensemble member to be selected.
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
            ), f"`ensemble_member` must be in the range from 0 to {environment.num_ensemble_members - 1}"
            setattr(self, "ensemble_member", ensemble_member)
        else:
            # if no ensemble member is provided, get it from the environment
            setattr(self, "ensemble_member", environment.ensemble_member)

    def create_object(self):
        """Creates a Environment object from the randomly generated input
        arguments.The environment object is not recreated to avoid having to
        reestablish the atmospheric model. Instead, attributes are changed
        directly.

        Parameters
        ----------
        None

        Returns
        -------
        obj : Environment
            Environment object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        for key, value in generated_dict.items():
            # special case for ensemble member
            # TODO if env.ensemble_member had a setter this create_object method
            # could be generalized
            if key == "ensemble_member":
                self.object.select_ensemble_member(value)
            else:
                if "factor" in key:
                    # get original attribute value and multiply by factor
                    attribute_name = f"_{key.replace('_factor', '')}"
                    value = getattr(self, attribute_name) * value
                setattr(self.object, key, value)
        return self.object
