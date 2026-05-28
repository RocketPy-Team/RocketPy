"""FlyingBody – a first-class, fully configurable flight body."""

from copy import deepcopy


class FlyingBody:
    """A fully configurable flight body composed of interchangeable models.

    Unlike :class:`~rocketpy.rocket.Rocket`, which is a rich user-facing
    builder object, :class:`FlyingBody` is designed as a clean value object
    that holds just enough information for the simulation engine to integrate
    the equations of motion.  Users who need the full Rocket builder
    experience can convert a :class:`~rocketpy.rocket.Rocket` to a
    :class:`FlyingBody` via :class:`RocketAdapter`.  It satisfies the
    :class:`~rocketpy.body.BodyLike` protocol.
    """