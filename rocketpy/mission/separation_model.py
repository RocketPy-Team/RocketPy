"""SeparationModel – interface for child/parent separation dynamics."""

from abc import ABC, abstractmethod


class SeparationModel(ABC):
    """Interface for modelling the separation impulse between two bodies.

    When a :class:`~rocketpy.mission.Deployable` or
    :class:`~rocketpy.mission.Stage` separates from its parent, a
    :class:`SeparationModel` is responsible for computing any
    velocity/orientation perturbations applied to both the parent and
    child bodies at the instant of separation.

    Implementors should override :meth:`apply` and may store any physical
    parameters (e.g. spring constant, separation charge energy) as
    instance attributes.
    """