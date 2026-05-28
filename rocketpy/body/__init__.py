"""rocketpy.body – body abstraction layer for multistage simulation."""

from rocketpy.body.body_like import BodyLike
from rocketpy.body.flying_body import FlightBody
from rocketpy.body.rocket_adapter import RocketAdapter

__all__ = ["BodyLike", "FlyingBody", "RocketAdapter"]