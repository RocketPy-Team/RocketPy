from copy import deepcopy

from rocketpy.environment import Environment

from .DispersionModel import DispersionModel

# TODO: name suggestions: `DispersionEnvironment`, `DispEnvironment`, EnvironmentDispersion`, EnvironmentDisp`, `MonteCarloEnvironment`, `EnvironmentMonteCarlo`,


class McEnvironment(DispersionModel):
    """Monte Carlo Environment class, it holds information about an Environment
    to be used in the Dispersion class based on the pydantic class. It uses the
    DispersionModel as a base model, see its docs for more information. The
    inputs defined here are the same as the ones defined in the Environment, see
    its docs for more information. Only environment field is required.
    """

    # TODO: missing special validation from pydantic version:
    # - ensemble_member must be a list of integers
    # - date should be validated as a datetime object
    # TODO: Should datum and timezone even be here?
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
        datum=None,
        time_zone=None,
    ):
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
            datum=datum,
            time_zone=time_zone,
        )
        # run special validators
        if ensemble_member:
            setattr(self, "ensemble_member", self.validate_ensemble(ensemble_member))

    def validate_ensemble(self, ensemble_member):
        """Special validator for the ensembleMember argument. It checks if the
        environment has the correct atmospheric model type and if the list does
        not overflow the ensemble members.
        """
        assert self.object.atmospheric_model_type in [
            "Ensemble",
            "Reanalysis",
        ], (
            f"Environment with {self.object.atmospheric_model_type} "
            "does not have ensemble members"
        )
        assert (
            max(ensemble_member) < self.object.num_ensemble_members
            and min(ensemble_member) >= 0
        ), f"`ensemble_member` must be in between from 0 to {self.object.num_ensemble_members - 1}"
        return ensemble_member

    def create_object(self):
        """Creates a Environment object from the randomly generated input arguments.
        The environment object is not recreatead to avoid having to reestablish
        the atmospheric model. Intead, a copy of the original environment is
        made, and its attributes changed.

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
