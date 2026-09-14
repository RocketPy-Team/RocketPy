import numericalunits
import pytest


@pytest.fixture
def m():
    """Create a simple object of the numericalunits.m class to be used in the
    tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    numericalunits.m
        A simple object of the numericalunits.m class
    """
    return numericalunits.m


@pytest.fixture
def kg():
    """Create a simple object of the numericalunits.kg class to be used in the
    tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    numericalunits.kg
        A simple object of the numericalunits.kg class
    """
    return numericalunits.kg
