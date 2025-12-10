import matplotlib
import netCDF4
import numpy as np
import pytest

# Configure matplotlib to use non-interactive backend for tests
matplotlib.use("Agg")

# Pytest configuration
pytest_plugins = [
    "tests.fixtures.environment.environment_fixtures",
    "tests.fixtures.flight.flight_fixtures",
    "tests.fixtures.function.function_fixtures",
    "tests.fixtures.motor.liquid_fixtures",
    "tests.fixtures.motor.hybrid_fixtures",
    "tests.fixtures.motor.solid_motor_fixtures",
    "tests.fixtures.motor.empty_motor_fixtures",
    "tests.fixtures.motor.tanks_fixtures",
    "tests.fixtures.motor.fluid_fixtures",
    "tests.fixtures.motor.tank_geometry_fixtures",
    "tests.fixtures.motor.generic_motor_fixtures",
    "tests.fixtures.parachutes.parachute_fixtures",
    "tests.fixtures.rockets.rocket_fixtures",
    "tests.fixtures.surfaces.surface_fixtures",
    "tests.fixtures.units.numerical_fixtures",
    "tests.fixtures.monte_carlo.monte_carlo_fixtures",
    "tests.fixtures.monte_carlo.stochastic_fixtures",
    "tests.fixtures.monte_carlo.custom_sampler_fixtures",
    "tests.fixtures.monte_carlo.stochastic_motors_fixtures",
    "tests.fixtures.sensors.sensors_fixtures",
    "tests.fixtures.generic_surfaces.generic_surfaces_fixtures",
    "tests.fixtures.generic_surfaces.linear_generic_surfaces_fixtures",
]


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


# TODO: move this to Environment fixtures when possible
@pytest.fixture
def merra2_file_path(tmp_path):  # pylint: disable=too-many-statements
    """
    Generates a temporary NetCDF file that STRICTLY mimics the structure of a
    NASA MERRA-2 'inst3_3d_asm_Np' file (Assimilated Meteorological Fields)
    because MERRA-2 files are too large.

    """
    file_path = tmp_path / "MERRA2_300.inst3_3d_asm_Np.20230620.nc4"

    with netCDF4.Dataset(file_path, "w", format="NETCDF4") as nc:
        # Define Dimensions
        nc.createDimension("lon", 5)
        nc.createDimension("lat", 5)
        nc.createDimension("lev", 5)
        nc.createDimension("time", None)

        # Define Coordinates
        lon = nc.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon[:] = np.linspace(-180, 180, 5)

        lat = nc.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = np.linspace(-90, 90, 5)

        lev = nc.createVariable("lev", "f8", ("lev",))
        lev.units = "hPa"
        lev[:] = np.linspace(1000, 100, 5)

        time = nc.createVariable("time", "i4", ("time",))
        time.units = "minutes since 2023-06-20 00:00:00"
        time[:] = [720]

        # Define Data Variables
        # NetCDF names are uppercase ("T") to match NASA specs.

        t_var = nc.createVariable("T", "f4", ("time", "lev", "lat", "lon"))
        t_var.units = "K"
        t_var[:] = np.full((1, 5, 5, 5), 300.0)

        u_var = nc.createVariable("U", "f4", ("time", "lev", "lat", "lon"))
        u_var.units = "m s-1"
        u_var[:] = np.full((1, 5, 5, 5), 10.0)

        v_var = nc.createVariable("V", "f4", ("time", "lev", "lat", "lon"))
        v_var.units = "m s-1"
        v_var[:] = np.full((1, 5, 5, 5), 5.0)

        h_var = nc.createVariable("H", "f4", ("time", "lev", "lat", "lon"))
        h_var.units = "m"
        h_var[:] = np.linspace(0, 10000, 5).reshape(1, 5, 1, 1) * np.ones((1, 5, 5, 5))

        # PHIS: Surface Geopotential Height [m2 s-2]
        phis = nc.createVariable("PHIS", "f4", ("time", "lat", "lon"))
        phis.units = "m2 s-2"

        # We set PHIS to 9806.65 (Energy).
        # We expect the code to divide by ~9.8 and get ~1000.0 (Height).
        phis[:] = np.full((1, 5, 5), 9806.65)

    return str(file_path)
