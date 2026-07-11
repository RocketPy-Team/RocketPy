"""Backward-compatibility shim for the relocated parachute module.

The parachute classes moved from ``rocketpy.rocket.parachute`` to the
``rocketpy.rocket.parachutes`` subpackage, and the old concrete ``Parachute``
class became an abstract base with the hemispherical model split out into
``HemisphericalParachute``. Importing from this module still works for backward
compatibility but is deprecated and will be removed in v1.14.0.
"""

import warnings

from .parachutes.hemispherical_parachute import HemisphericalParachute
from .parachutes.parachute import Parachute

warnings.warn(
    "Importing from 'rocketpy.rocket.parachute' is deprecated and will be "
    "removed in v1.14.0. Import from 'rocketpy.rocket.parachutes' instead, "
    "e.g. 'from rocketpy.rocket.parachutes import HemisphericalParachute'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Parachute", "HemisphericalParachute"]
