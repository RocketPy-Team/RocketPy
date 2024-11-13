import pytest

from rocketpy import CylindricalTank, SphericalTank


@pytest.fixture
def pressurant_tank_geometry():
    """An example of a pressurant cylindrical tank with spherical
    caps.

    Returns
    -------
    rocketpy.CylindricalTank
        An object of the CylindricalTank class.
    """
    return CylindricalTank(0.135 / 2, 0.981, spherical_caps=True)


@pytest.fixture
def propellant_tank_geometry():
    """An example of a cylindrical tank with spherical
    caps.

    Returns
    -------
    rocketpy.CylindricalTank
        An object of the CylindricalTank class.
    """
    return CylindricalTank(0.0744, 0.8068, spherical_caps=True)


@pytest.fixture
def spherical_oxidizer_geometry():
    """An example of a spherical tank.

    Returns
    -------
    rocketpy.SphericalTank
        An object of the SphericalTank class.
    """
    return SphericalTank(0.05)
