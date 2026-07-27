"""The rocketpy.environment module is responsible for the Atmospheric and Earth
models. The methods and classes not listed in the __all__ variable will be
considered private and should be used with caution.
"""

from .albedo import EarthRadiationPressure
from .atmosphere import (
    Atmosphere,
    AtmosphereLayer,
    AtmosphericState,
    ExponentialAtmosphereLayer,
    FunctionAtmosphereLayer,
    HarrisPriesterAtmosphereLayer,
    NRLMSISE00AtmosphereLayer,
    ZeroAtmosphereLayer,
)
from .environment import Environment
from .environment_analysis import EnvironmentAnalysis
from .gravity import (
    DefaultGravity,
    Gravity,
    SomiglianaGravity,
    SphericalGravity,
    SphericalHarmonicGravity,
    VerticalGravity,
    ZeroGravity,
    ZonalGravity,
)
from .models import Earth, Space
from .third_body import (
    SpiceEphemeris,
    ThirdBody,
)

__all__ = [
    "Atmosphere",
    "AtmosphereLayer",
    "AtmosphericState",
    "DefaultGravity",
    "Environment",
    "Earth",
    "EarthRadiationPressure",
    "EnvironmentAnalysis",
    "ExponentialAtmosphereLayer",
    "FunctionAtmosphereLayer",
    "Gravity",
    "HarrisPriesterAtmosphereLayer",
    "NRLMSISE00AtmosphereLayer",
    "SphericalGravity",
    "SomiglianaGravity",
    "SphericalHarmonicGravity",
    "SpiceEphemeris",
    "Space",
    "ThirdBody",
    "ZeroAtmosphereLayer",
    "VerticalGravity",
    "ZeroGravity",
    "ZonalGravity",
]
