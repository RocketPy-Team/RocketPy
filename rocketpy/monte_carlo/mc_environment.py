__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from copy import deepcopy
from typing import Any, List, Tuple, Union

from pydantic import Field, StrictInt, StrictStr, validator

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

    # Field(...) means it is a required field, exclude=True removes it from the
    # self.dict() method, which is used to convert the class to a dictionary
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel @root_validator
    # Fields with typing that is not Any have special requirements
    environment: Environment = Field(..., exclude=True)
    railLength: Any = 0
    date: List[Union[Tuple[int, int, int, int], None]] = []
    elevation: Any = 0
    gravity: Any = 0
    latitude: Any = 0
    longitude: Any = 0
    ensembleMember: List[StrictInt] = []
    windXFactor: Any = (1, 0)
    windYFactor: Any = (1, 0)
    datum: List[Union[StrictStr, None]] = []
    timeZone: List[Union[StrictStr, None]] = []

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("ensembleMember")
    def val_ensemble(cls, v, values):
        """Special validator for the ensembleMember argument. It checks if the
        environment has the correct atmospheric model type and if the list does
        not overflow the ensemble members.
        """
        if v:
            assert values["environment"].atmosphericModelType in [
                "Ensemble",
                "Reanalysis",
            ], f"Environment with {values['environment'].atmosphericModelType} does not have ensemble members"
            assert (
                max(v) < values["environment"].numEnsembleMembers and min(v) >= 0
            ), f"Please choose ensembleMember from 0 to {values['environment'].numEnsembleMembers - 1}"
        return v

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
        gen_dict = deepcopy(next(self.dict_generator()))
        obj = self.environment
        obj.railLength = gen_dict["railLength"]
        obj.date = gen_dict["date"]
        obj.elevation = gen_dict["elevation"]
        obj.gravity = gen_dict["gravity"]
        obj.latitude = gen_dict["latitude"]
        obj.longitude = gen_dict["longitude"]
        obj.datum = gen_dict["datum"]
        obj.timeZone = gen_dict["timeZone"]
        # Apply Factors
        obj.windVelocityX *= gen_dict["windXFactor"]
        obj.windVelocityY *= gen_dict["windYFactor"]
        if gen_dict["ensembleMember"]:
            obj.selectEnsembleMember(gen_dict["ensembleMember"])
        return obj
