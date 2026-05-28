"""BodyLike protocol for the multistage mission architecture.

This module defines a structural typing contract so bodies can satisfy the
interface without inheriting from a shared base class.
"""

from typing import Protocol, runtime_checkable
@runtime_checkable
class BodyLike(Protocol):
    """Structural interface that every flight-ready body must satisfy.

    Any object that can be integrated by the Flight simulation engine
    must implement this interface.  Both the native :class:`FlightBody`
    and the :class:`RocketAdapter` (which wraps a legacy :class:`Rocket`)
    satisfy it, so that the simulation layer can treat them uniformly.
    This uses structural typing, so concrete bodies do not need to inherit
    from :class:`BodyLike` to be accepted by the simulation layer.

    Attributes
    ----------
    name : str
        Human-readable identifier for the body.
    """