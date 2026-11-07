"""Axial geometry of a rocket, as the stacking of stages needs it."""

from rocketpy.rocket.aero_surface.fins.fins import Fins
from rocketpy.rocket.aero_surface.nose_cone import NoseCone
from rocketpy.rocket.aero_surface.tail import Tail


def axial_extent(rocket):
    """Axial extent (bottom, top) spanned by a rocket's aerodynamic
    surfaces, in the rocket's own coordinate system.

    NoseCone, Tail and Fins each occupy a span (their own ``length`` or
    ``root_chord``, starting from their own reference position); every
    other surface type (GenericSurface, RailButtons, individual Fin,
    TubeFins, ...) is treated as a single point at its own position. This
    is an approximation for those types, not an exact geometric fit.

    Parameters
    ----------
    rocket : Rocket
        Must have at least one aerodynamic surface.

    Returns
    -------
    tuple of float
        (bottom, top) - the lowest and highest axial coordinates spanned.
    """
    if not rocket.aerodynamic_surfaces:
        raise ValueError(
            "Rocket must have at least one aerodynamic surface to compute "
            "its axial extent."
        )
    bounds = []
    for surface, position, *_ in rocket.aerodynamic_surfaces:
        z = position.z
        bounds.append(z)
        if isinstance(surface, NoseCone):
            bounds.append(z - rocket._csys * surface.length)
        elif isinstance(surface, Tail):
            bounds.append(z - rocket._csys * surface.length)
        elif isinstance(surface, Fins):
            bounds.append(z - rocket._csys * surface.root_chord)
    return min(bounds), max(bounds)
