"""ParentUpdate – interface for modifying a parent body after separation."""

from abc import ABC, abstractmethod


class ParentUpdate(ABC):
    """Interface for updating a parent body when a child separates.

    When a :class:`~rocketpy.mission.Deployable` or
    :class:`~rocketpy.mission.Stage` separates from its parent, the
    parent body may need to be updated to account for the mass and
    inertia change caused by the separation.  Concrete implementations
    of this interface perform that update in place.
    """