import datetime
import os
import time
from unittest.mock import patch

import numpy as np
import pytest
import pytz

from rocketpy import Environment


def test_env_set_date(example_env):
    """Test that the date is set correctly in the environment object. This
    basically takes a date and time and converts it to a datetime object, then
    set the date to the example environment object. The test checks if the
    datetime object is the same as the one in the example environment object.

    Parameters
    ----------
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    example_env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
    assert example_env.datetime_date == datetime.datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, 12, tzinfo=pytz.utc
    )


def test_env_set_date_time_zone(example_env):
    """Test that the time zone is set correctly in the environment object

    Parameters
    ----------
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    example_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12), timezone="America/New_York"
    )
    date_naive = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 12)
    timezone = pytz.timezone("America/New_York")
    date_aware_local_date = timezone.localize(date_naive)
    date_aware_utc = date_aware_local_date.astimezone(pytz.UTC)
    assert example_env.datetime_date == date_aware_utc


def test_env_set_location(example_env):
    """Test that the location is set correctly in the environment object. This
    one basically set the location using the set_location() method and then
    checks if the latitude and longitude are the same as the ones in the
    example environment object.

    Parameters
    ----------
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env.set_location(-21.960641, -47.482122)
    assert example_env.latitude == -21.960641
    assert example_env.longitude == -47.482122


def test_set_elevation(example_env):
    """Tests if the elevation is set correctly in the environment object. This
    one basically set the elevation using the set_elevation() method and then
    checks if the elevation is the same as the one in the example environment
    object.

    Parameters
    ----------
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env.set_elevation(elevation=200)
    assert example_env.elevation == 200


def test_set_topographic_profile(example_env):
    """Tests the topographic profile in the environment object.

    Parameters
    ----------
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env.set_location(46.90479, 8.07575)
    example_env.set_topographic_profile(
        type="NASADEM_HGT",
        file="data/sites/switzerland/NASADEM_NC_n46e008.nc",
        dictionary="netCDF4",
    )
    assert (
        example_env.get_elevation_from_topographic_profile(
            example_env.latitude, example_env.longitude
        )
        == 1565
    )


def test_export_environment(example_env_robust):
    """Tests the export_environment() method of the Environment class.

    Parameters
    ----------
    example_env_robust : rocketpy.Environment
        Example environment object to be tested.
    """
    assert example_env_robust.export_environment(filename="environment") == None
    os.remove("environment.json")


def test_geodesic_to_utm():
    """Tests the conversion from geodesic to UTM coordinates."""
    x, y, utm_zone, utm_letter, hemis, EW = Environment.geodesic_to_utm(
        lat=32.990254,
        lon=-106.974998,
        semi_major_axis=6378137.0,  # WGS84
        flattening=1 / 298.257223563,  # WGS84
    )
    assert np.isclose(x, 315468.64, atol=1e-5) == True
    assert np.isclose(y, 3651938.65, atol=1e-5) == True
    assert utm_zone == 13
    assert utm_letter == "S"
    assert hemis == "N"
    assert EW == "W"


def test_utm_to_geodesic():
    """Tests the conversion from UTM to geodesic coordinates."""

    lat, lon = Environment.utm_to_geodesic(
        x=315468.64,
        y=3651938.65,
        utm_zone=13,
        hemis="N",
        semi_major_axis=6378137.0,  # WGS84
        flattening=1 / 298.257223563,  # WGS84
    )
    assert np.isclose(lat, 32.99025, atol=1e-5) == True
    assert np.isclose(lon, -106.9750, atol=1e-5) == True


@pytest.mark.parametrize(
    "lat, radius", [(0, 6378137.0), (90, 6356752.31424518), (-90, 6356752.31424518)]
)
def test_earth_radius_calculation(lat, radius):
    """Tests if the earth radius calculation is correct. It takes 3 different
    latitudes and their expected results and compares them with the results
    from the method.

    Parameters
    ----------
    lat : float
        The latitude in decimal degrees.
    radius : float
        The expected radius in meters at the given latitude.
    """
    semi_major_axis = 6378137.0  # WGS84
    flattening = 1 / 298.257223563  # WGS84
    res = Environment.calculate_earth_radius(lat, semi_major_axis, flattening)
    assert pytest.approx(res, abs=1e-8) == radius


@pytest.mark.parametrize(
    "angle, deg, arc_min, arc_sec",
    [
        (-106.974998, -106.0, 58, 29.9928),
        (32.990254, 32, 59.0, 24.9144),
        (90.0, 90, 0, 0),
    ],
)
def test_decimal_degrees_to_arc_seconds(angle, deg, arc_min, arc_sec):
    """Tests if the conversion from decimal degrees to arc seconds is correct.
    It takes 3 different angles and their expected results and compares them
    with the results from the method.

    Parameters
    ----------
    angle : float
        Angle in decimal degrees.
    deg : int
        Expected degrees.
    arc_min : int
        Expected arc minutes.
    arc_sec : float
        Expected arc seconds.
    """
    res = Environment.decimal_degrees_to_arc_seconds(angle)
    assert pytest.approx(res[0], abs=1e-8) == deg
    assert pytest.approx(res[1], abs=1e-8) == arc_min
    assert pytest.approx(res[2], abs=1e-8) == arc_sec
