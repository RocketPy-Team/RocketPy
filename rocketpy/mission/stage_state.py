"""StageState – lifecycle states for a Stage."""

from enum import Enum, auto


class StageState(Enum):
    """Enumeration of the lifecycle states a :class:`Stage` can occupy.

    Typical lifecycle progressions are:

    - Powered stage: ``ATTACHED → IGNITED → SEPARATED → SPENT``
    - Unpowered drop stage (e.g. fairing): ``ATTACHED → SEPARATED``
    - Sustainer (never physically separated): ``ATTACHED → IGNITED → SPENT``

    Attributes
    ----------
    ATTACHED :
        The stage is attached to the parent body and has not yet ignited.
    IGNITED :
        The stage motor has ignited but the stage is still mechanically
        coupled to the parent.
    SEPARATED :
        The stage has mechanically separated from the parent and is flying
        independently.
    SPENT :
        The stage motor has burned out.  The stage may still be attached
        (e.g. coast phase) or may have separated already.
    """

    ATTACHED = auto()
    IGNITED = auto()
    SEPARATED = auto()
    SPENT = auto()
