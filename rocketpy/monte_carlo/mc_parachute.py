from typing import Any, Callable, List, Tuple, Union

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from rocketpy.rocket import Parachute

from .DispersionModel import DispersionModel


class McParachute(DispersionModel):
    """Monte Carlo Parachute class, used to validate the input parameters of the
    parachute, based on the pydantic library. It uses the DispersionModel class
    as a base class, see its documentation for more information. The inputs
    defined here correspond to the ones defined in the Parachute class.
    """

    # Field(...) means it is a required field, exclude=True removes it from the
    # self.dict() method, which is used to convert the class to a dictionary
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel @root_validator
    # Fields with typing that is not Any have special requirements
    parachute: Parachute = Field(..., exclude=True)
    CdS: Any = 0
    trigger: List[Union[Callable, None]] = []
    samplingRate: Any = 0
    lag: Any = 0
    name: List[Union[StrictStr, None]] = []
    noise: List[
        Union[
            Tuple[
                Union[StrictInt, StrictFloat],
                Union[StrictInt, StrictFloat],
                Union[StrictInt, StrictFloat],
            ],
            None,
        ]
    ] = []

    def create_object(self):
        """Creates a Parachute object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : Parachute
            Parachute object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = Parachute(
            CdS=gen_dict["CdS"],
            Trigger=gen_dict["trigger"],
            samplingRate=gen_dict["samplingRate"],
            lag=gen_dict["lag"],
            name=gen_dict["name"],
            noise=gen_dict["noise"],
        )
        return obj
