from typing import Any, List, Tuple, Union

from pydantic import BaseModel, StrictFloat, StrictInt, root_validator

from ..Flight import Flight


class McFlight(BaseModel):
    """TODO: Add description

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    inclination: Any
    heading: Any
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

    class Config:
        arbitrary_types_allowed = True

    @root_validator(skip_on_failure=True)
    def val_basic(cls, values):
        """Validates inputs that can be either tuples or lists.
        Tuples can have either 2 or 3 items. First two must be either float or int,
        representing the nominal value and standard deviation. Third item must be
        a string containing the name of a numpy.random distribution function"""

        validate_fields = ["inclination", "heading"]
        for field in validate_fields:
            v = values[field]
            # checks if tuple
            if isinstance(v, tuple):
                # checks if first two items are valid
                assert isinstance(v[0], (int, float)) and isinstance(
                    v[1], (int, float)
                ), f"\nField '{field}': \n\tFirst two items of tuple must be either an int or float \n\tFirst item refers to nominal value, and the second to the standard deviation"
                # extra check for third item if passed
                if len(v) == 3:
                    assert isinstance(
                        v[2], str
                    ), f"\nField '{field}': \n\tThird item of tuple must be either a string \n\tThe third item must be a string containing the name of a numpy.random distribution function"
                # all good, sets inputs
                values[field] = v
            elif isinstance(v, list):
                # guarantee all values are valid (ints or floats)
                assert all(
                    isinstance(item, (int, float)) for item in v
                ), f"\nField '{field}': \n\tItems in list must be either ints or floats"
                # all good, sets inputs
                values[field] = v
            else:
                raise ValueError(
                    f"\nField '{field}': \n\tMust be either a tuple or list"
                )
        return values
