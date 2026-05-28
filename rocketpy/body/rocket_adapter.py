"""RocketAdapter – wraps a legacy :class:`Rocket` as a :class:`BodyLike`.

This module provides a minimal adapter to allow legacy ``Rocket``
instances to be consumed by code expecting a ``BodyLike`` interface.
"""

from copy import deepcopy


class RocketAdapter:
    """Adapts a legacy :class:`~rocketpy.rocket.Rocket` to the
    :class:`BodyLike` interface.

    This allows the existing :class:`~rocketpy.rocket.Rocket` builder to be
    consumed by the new multistage simulation infrastructure without any
    changes to the ``Rocket`` class itself.

    Parameters
    ----------
    rocket : :class:`~rocketpy.rocket.Rocket`
        The rocket instance to adapt.

    Attributes
    ----------
    rocket : :class:`~rocketpy.rocket.Rocket`
        The wrapped rocket instance.
    """

    def __init__(self, rocket):
        # store a deep copy to avoid accidental mutation of the original
        self.rocket = deepcopy(rocket)

    def __repr__(self):
        return f"<RocketAdapter rocket={getattr(self.rocket, 'name', None)!r}>\n"
