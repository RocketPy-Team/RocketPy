from rocketpy.environment import Environment

from .dispersion_model import DispersionModel

# TODO: name suggestions: `DispersionEnvironment`, `DispEnvironment`, EnvironmentDispersion`, EnvironmentDisp`, `MonteCarloEnvironment`, `EnvironmentMonteCarlo`,


class McEnvironment(DispersionModel):
    """Monte Carlo Environment class, used to validate the input parameters of
    the environment. It uses the DispersionModel class as a base class, see its
    documentation for more information. The inputs defined here correspond to
    the ones defined in the Environment class.
    """

    # TODO: Since we are not recreating the environment object, to avoid having
    # to reestablish the atmospheric model, I believe date, datum, time_zone
    # and maybe elevation do not do anything?
    def __init__(
        self,
        environment,
        date=None,
        elevation=None,
        gravity=None,
        latitude=None,
        longitude=None,
        ensemble_member=None,
        wind_velocity_x_factor=None,
        wind_velocity_y_factor=None,
    ):
        """Initializes the Monte Carlo Environment class.

        See Also
        --------
        This should link to somewhere that explains how inputs works in
        dispersion models.

        Parameters
        ----------
        environment : Environment
            Environment object to be used for validation.
        date : list, optional
            List of dates, which are tuples of four elements
            (year, month, day, hour).
        elevation : int, float, tuple, list, optional
            Elevation of the launch site in meters. Follows the standard
            input format of Dispersion Models.
        gravity : int, float, tuple, list, optional
            Gravitational acceleration in meters per second squared. Follows
            the standard input format of Dispersion Models.
        latitude : int, float, tuple, list, optional
            Latitude of the launch site in degrees. Follows the standard
            input format of Dispersion Models.
        longitude : int, float, tuple, list, optional
            Longitude of the launch site in degrees. Follows the standard
            input format of Dispersion Models.
        ensemble_member : list, optional
            List of integers representing the ensemble member to be selected.
        wind_velocity_x_factor : int, float, tuple, list, optional
            Factor to be multiplied by the wind velocity in the x direction.
        wind_velocity_y_factor : int, float, tuple, list, optional
            Factor to be multiplied by the wind velocity in the y direction.
        """
        # Validate in DispersionModel
        super().__init__(
            environment,
            date=date,
            elevation=elevation,
            gravity=gravity,
            latitude=latitude,
            longitude=longitude,
            ensemble_member=ensemble_member,
            wind_velocity_x_factor=wind_velocity_x_factor,
            wind_velocity_y_factor=wind_velocity_y_factor,
        )
        # Special validation
        self._validate_date(date, environment)
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
                    # treats date atribute
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

    def _validate_date(self, date, environment):
        """Validates the date input argument. If the date input argument is
        None, gets the date from the environment and saves it as a list of
        one element. Else, the input argument must be a list with tuples of four
        elements (year, month, day, hour)

        Parameters
        ----------
        date : list
            Date to be used for validation.
        environment : Environment
            Environment object to be used for validation.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the date input argument is not None, a datetime object or a list
            of datetime objects.
        """
        if date is None:
            date = [environment.date]
        else:
            assert isinstance(date, list) and all(
                isinstance(member, tuple) and len(member) == 4 for member in date
            ), "`date` must be a list of tuples of four elements "
            "(year, month, day, hour)"
        setattr(self, "date", date)

    def create_object(self):
        """Creates a Environment object from the randomly generated input
        arguments.The environment object is not recreatead to avoid having to
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
