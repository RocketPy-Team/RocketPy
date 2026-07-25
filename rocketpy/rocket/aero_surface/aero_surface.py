import warnings
from abc import ABC

from rocketpy.rocket.aero_surface.generic_surface import GenericSurface

_AEROSURFACE_DEPRECATION_MESSAGE = (
    "`AeroSurface` is deprecated and will be removed in a future major "
    "release. RocketPy's aerodynamic surfaces now derive from `GenericSurface`; "
    "use `GenericSurface` (or a concrete surface class such as `NoseCone`, "
    "`TrapezoidalFins`, `Tail`, ...) instead. Note that `isinstance(surface, "
    "AeroSurface)` still returns True for all surfaces."
)


class AeroSurface(ABC):
    """Deprecated base class for aerodynamic surfaces.

    .. deprecated::
        ``AeroSurface`` is no longer the base of RocketPy's aerodynamic
        surfaces, which now all derive from :class:`GenericSurface`. It is kept
        only as a deprecated compatibility shim and will be removed in a future
        major release. Importing, instantiating or subclassing it emits a
        ``DeprecationWarning``.

        For backward compatibility, :class:`GenericSurface` (and therefore every
        concrete surface) is registered as a *virtual subclass*, so existing
        ``isinstance(surface, AeroSurface)`` and
        ``issubclass(type(surface), AeroSurface)`` checks keep working.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            _AEROSURFACE_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2
        )

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        warnings.warn(
            _AEROSURFACE_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2
        )


# Register GenericSurface (and thus all concrete surfaces) as a virtual
# subclass so that ``isinstance(surface, AeroSurface)`` remains True during the
# deprecation period. Virtual registration does not trigger ``__init_subclass__``.
AeroSurface.register(GenericSurface)
