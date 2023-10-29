__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from typing import Any, List, Union

from pydantic import Field, FilePath, StrictFloat, StrictInt

from rocketpy.motors import SolidMotor

from .DispersionModel import DispersionModel


class McSolidMotor(DispersionModel):
    """Monte Carlo Solid Motor class, used to validate the input parameters of
    the solid motor, based on the pydantic library. It uses the DispersionModel
    class as a base class, see its documentation for more information. The
    inputs defined here correspond to the ones defined in the SolidMotor class.
    """

    # Field(...) means it is a required field, exclude=True removes it from the
    # self.dict() method, which is used to convert the class to a dictionary
    # Fields with typing Any must have the standard dispersion form of tuple or
    # list. This is checked in the DispersionModel @root_validator
    # Fields with typing that is not Any have special requirements
    solidMotor: SolidMotor = Field(..., exclude=True)
    thrustSource: List[Union[FilePath, None]] = []
    burnOutTime: Any = 0
    grainsCenterOfMassPosition: Any = 0
    grainNumber: List[Union[Union[StrictInt, StrictFloat], None]] = []
    grainDensity: Any = 0
    grainOuterRadius: Any = 0
    grainInitialInnerRadius: Any = 0
    grainInitialHeight: Any = 0
    grainSeparation: Any = 0
    totalImpulse: Any = 0
    nozzleRadius: Any = 0
    nozzlePosition: Any = 0
    throatRadius: Any = 0

    def create_object(self):
        """Creates a SolidMotor object from the randomly generated input arguments.

        Parameters
        ----------
        None

        Returns
        -------
        obj : SolidMotor
            SolidMotor object with the randomly generated input arguments.
        """
        gen_dict = next(self.dict_generator())
        obj = SolidMotor(
            throatRadius=gen_dict["throatRadius"],
            burnOutTime=gen_dict["burnOutTime"],
            grainNumber=gen_dict["grainNumber"],
            grainDensity=gen_dict["grainDensity"],
            grainOuterRadius=gen_dict["grainOuterRadius"],
            grainInitialInnerRadius=gen_dict["grainInitialInnerRadius"],
            grainInitialHeight=gen_dict["grainInitialHeight"],
            grainSeparation=gen_dict["grainSeparation"],
            nozzleRadius=gen_dict["nozzleRadius"],
            nozzlePosition=gen_dict["nozzlePosition"],
            thrustSource=gen_dict["thrustSource"],
            grainsCenterOfMassPosition=gen_dict["grainsCenterOfMassPosition"],
            reshapeThrustCurve=(gen_dict["burnOutTime"], gen_dict["totalImpulse"]),
        )
        if "position" in gen_dict:
            obj.position = gen_dict["position"]
        return obj
