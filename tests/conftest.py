import pytest


# Pytest configuration

pytest_plugins = [
    "tests.fixtures.environment.environment_fixtures",
    "tests.fixtures.flight.flight_fixtures",
    "tests.fixtures.function.function_fixtures",
    "tests.fixtures.hybrid.hybrid_fixtures",
    "tests.fixtures.motor.motor_fixtures",
    "tests.fixtures.parachutes.parachute_fixtures",
    "tests.fixtures.rockets.rocket_fixtures",
    "tests.fixtures.surfaces.surface_fixtures",
    "tests.fixtures.units.numerical_fixtures"
]

#

def pytest_addoption(parser):
    """Add option to run slow tests. This is used to skip slow tests by default.

    Parameters
    ----------
    parser : _pytest.config.argparsing.Parser
        Parser object to which the option is added.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add marker to run slow tests. This is used to skip slow tests by default.

    Parameters
    ----------
    config : _pytest.config.Config
        Config object to which the marker is added.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default. This is used to skip slow tests by default.

    Parameters
    ----------
    config : _pytest.config.Config
        Config object to which the marker is added.
    items : list
        List of tests to be modified.
    """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


## AeroSurfaces


## Parachutes


## Flights


## Dimensionless motors and rockets


## Environment


## Functions


