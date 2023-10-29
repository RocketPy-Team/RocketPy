from typing import Any, Tuple, Union

from pydantic import Field, StrictFloat, StrictInt

from rocketpy.simulation import Flight

from .DispersionModel import DispersionModel


class McFlight(DispersionModel):
    """Monte Carlo Flight class, used to validate the input parameters of the
    flight to be used in the Dispersion class, based on the pydantic library. It
    uses the DispersionModel class as a base class, see its documentation for
    more information. The inputs defined here are the same as the ones defined
    in the Flight class, see its documentation for more information.
    """

    # Field(...) means it is a required field, exclude=True removes it from the
    # self.dict() method, which is used to convert the class to a dictionary
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel @root_validator
    # Fields with typing that is not Any have special requirements
    flight: Flight = Field(..., exclude=True)
    inclination: Any = 0
    heading: Any = 0
    initialSolution: Union[
        Flight,
        Tuple[
            Union[StrictInt, StrictFloat],  # tInitial
            Union[StrictInt, StrictFloat],  # xInit
            Union[StrictInt, StrictFloat],  # yInit
            Union[StrictInt, StrictFloat],  # zInit
            Union[StrictInt, StrictFloat],  # vxInit
            Union[StrictInt, StrictFloat],  # vyInit
            Union[StrictInt, StrictFloat],  # vzInit
            Union[StrictInt, StrictFloat],  # e0Init
            Union[StrictInt, StrictFloat],  # e1Init
            Union[StrictInt, StrictFloat],  # e2Init
            Union[StrictInt, StrictFloat],  # e3Init
            Union[StrictInt, StrictFloat],  # w1Init
            Union[StrictInt, StrictFloat],  # w2Init
            Union[StrictInt, StrictFloat],  # w3Init
        ],
    ] = None
    terminateOnApogee: bool = False

    def rnd_inclination(self):
        """Creates a random inclination

        Returns
        -------
        inclination : float
            Random inclination
        """
        gen_dict = next(self.dict_generator())
        return gen_dict["inclination"]

    def rnd_heading(self):
        """Creates a random heading

        Returns
        -------
        heading : float
            Random heading
        """
        gen_dict = next(self.dict_generator())
        return gen_dict["heading"]

    def create_object(self):
        """Creates a Flight object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : Flight
            Flight object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = Flight(
            environment=self.flight.env,
            rocket=self.flight.rocket,
            inclination=gen_dict["inclination"],
            heading=gen_dict["heading"],
            initialSolution=self.initialSolution,
            terminateOnApogee=self.terminateOnApogee,
        )
        return obj
