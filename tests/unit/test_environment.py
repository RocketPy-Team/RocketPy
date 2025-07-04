import json
import os

import numpy as np
import pytest
import pytz

from rocketpy import Environment
from rocketpy.environment.tools import geodesic_to_utm, utm_to_geodesic


@pytest.mark.parametrize(
    "latitude, longitude", [(-21.960641, -47.482122), (0, 0), (21.960641, 47.482122)]
)
def test_location_set_location_saves_location(latitude, longitude, example_plain_env):
    """Tests location is saved correctly in the environment obj.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
    latitude: float
        The latitude in decimal degrees.
    longitude: float
        The longitude in decimal degrees.
    """
    example_plain_env.set_location(latitude, longitude)
    assert example_plain_env.latitude == latitude
    assert example_plain_env.longitude == longitude


@pytest.mark.parametrize("elevation", [(0), (100), (1000), (100000)])
def test_elevation_set_elevation_saves_elevation(elevation, example_plain_env):
    """Tests the wether the 'set_elevation' method within the Environment class
    sets the elevation correctly.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
    """

    example_plain_env.set_elevation(elevation=elevation)
    assert example_plain_env.elevation == elevation


@pytest.mark.parametrize(
    "latitude, longitude, theoretical_elevation",
    [(46.90479, 8.07575, 1565), (46.00001, 8.00001, 2562), (46.99999, 8.99999, 2832)],
)
def test_location_set_topographic_profile_computes_elevation(
    latitude, longitude, theoretical_elevation, example_plain_env
):
    """Tests elevation computation given topographic profile in the environment obj.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
    latitude: float
        The latitude in decimal degrees.
    longitude: float
        The longitude in decimal degrees.
    """
    example_plain_env.set_topographic_profile(
        type="NASADEM_HGT",
        file="data/sites/switzerland/NASADEM_NC_n46e008.nc",
        dictionary="netCDF4",
    )
    computed_elevation = example_plain_env.get_elevation_from_topographic_profile(
        latitude, longitude
    )
    assert computed_elevation == theoretical_elevation


def test_geodesic_coordinate_geodesic_to_utm_converts_coordinate():
    """Tests the conversion from geodesic to UTM coordinates."""
    (
        x,
        y,
        utm_zone,
        utm_letter,
        north_south_hemis,
        east_west_hemis,
    ) = geodesic_to_utm(
        lat=32.990254,
        lon=-106.974998,
        semi_major_axis=6378137.0,  # WGS84
        flattening=1 / 298.257223563,  # WGS84
    )
    assert np.isclose(x, 315468.64, atol=1e-5)
    assert np.isclose(y, 3651938.65, atol=1e-5)
    assert utm_zone == 13
    assert utm_letter == "S"
    assert north_south_hemis == "N"
    assert east_west_hemis == "W"


def test_utm_to_geodesic_converts_coordinates():
    """Tests the utm_to_geodesic method within the Environment
    class and checks the conversion results from UTM to geodesic
    coordinates.
    """

    lat, lon = utm_to_geodesic(
        x=315468.64,
        y=3651938.65,
        utm_zone=13,
        hemis="N",
        semi_major_axis=6378137.0,  # WGS84
        flattening=1 / 298.257223563,  # WGS84
    )
    assert np.isclose(lat, 32.99025, atol=1e-5)
    assert np.isclose(lon, -106.9750, atol=1e-5)


@pytest.mark.parametrize(
    "latitude, theoretical_radius",
    [(0, 6378137.0), (90, 6356752.31424518), (-90, 6356752.31424518)],
)
def test_latitude_calculate_earth_radius_computes_radius(latitude, theoretical_radius):
    """Tests earth radius calculation.

    Parameters
    ----------
    latitude : float
        The latitude in decimal degrees.
    theoretical_radius : float
        The expected radius in meters at the given latitude.
    """
    semi_major_axis = 6378137.0  # WGS84
    flattening = 1 / 298.257223563  # WGS84
    computed_radius = Environment.calculate_earth_radius(
        latitude, semi_major_axis, flattening
    )
    assert pytest.approx(computed_radius, abs=1e-8) == theoretical_radius


@pytest.mark.parametrize(
    "angle, theoretical_degree, theoretical_arc_minutes, theoretical_arc_seconds",
    [
        (-106.974998, -106.0, 58, 29.9928),
        (32.990254, 32, 59, 24.9144),
        (90.0, 90, 0, 0),
    ],
)
def test_decimal_degrees_to_arc_seconds_computes_correct_values(
    angle, theoretical_degree, theoretical_arc_minutes, theoretical_arc_seconds
):
    """Tests the conversion from decimal degrees to arc minutes and arc seconds.

    Parameters
    ----------
    angle : float
        Angle in decimal degrees.
    theoretical_degree : int
        Expected computed integer degrees.
    theoretical_arc_minutes : int
        Expected computed arc minutes.
    theoretical_arc_seconds : float
        Expected computed arc seconds.
    """
    computed_data = Environment.decimal_degrees_to_arc_seconds(angle)

    assert pytest.approx(computed_data[0], abs=1e-8) == theoretical_degree
    assert pytest.approx(computed_data[1], abs=1e-8) == theoretical_arc_minutes
    assert pytest.approx(computed_data[2], abs=1e-8) == theoretical_arc_seconds


def test_date_naive_set_date_saves_utc_timezone_by_default(
    example_plain_env, example_date_naive
):
    """Tests environment.set_date sets timezone to UTC by default

    Parameters
    ----------
    example_plain_env: rocketpy.Environment
    example_date_naive: datetime.datetime
    """
    example_plain_env.set_date(example_date_naive)
    assert example_plain_env.datetime_date == pytz.utc.localize(example_date_naive)


def test_date_aware_set_date_saves_custom_timezone(
    example_plain_env, example_date_naive
):
    """Tests time zone is set accordingly in environment obj given a date_aware input

    Parameters
    ----------
    example_plain_env: rocketpy.Environment
    example_date_naive: datetime.datetime
    """
    example_plain_env.set_date(example_date_naive, timezone="America/New_York")
    example_date_aware = pytz.timezone("America/New_York").localize(example_date_naive)
    assert example_plain_env.datetime_date == example_date_aware


@pytest.mark.parametrize("env_name", ["example_spaceport_env", "example_euroc_env"])
def test_environment_export_environment_exports_valid_environment_json(
    request, env_name
):
    """Tests the export_environment() method of the Environment class.

    Parameters
    ----------
    env_name : str
        The name of the environment fixture to be tested.
    """
    # get the fixture with the name in the string
    env = request.getfixturevalue(env_name)
    # Check file creation
    assert env.export_environment(filename="environment") is None
    with open("environment.json", "r") as json_file:
        exported_env = json.load(json_file)
    assert os.path.isfile("environment.json")

    # Check file content
    assert exported_env["gravity"] == env.gravity(env.elevation)
    assert exported_env["date"] == [
        env.datetime_date.year,
        env.datetime_date.month,
        env.datetime_date.day,
        env.datetime_date.hour,
    ]
    assert exported_env["latitude"] == env.latitude
    assert exported_env["longitude"] == env.longitude
    assert exported_env["elevation"] == env.elevation
    assert exported_env["datum"] == env.datum
    assert exported_env["timezone"] == env.timezone
    assert exported_env["max_expected_height"] == float(env.max_expected_height)
    assert exported_env["atmospheric_model_type"] == env.atmospheric_model_type
    assert exported_env["atmospheric_model_file"] is None
    assert exported_env["atmospheric_model_dict"] is None
    assert exported_env["atmospheric_model_pressure_profile"] == str(
        env.pressure.get_source()
    )
    assert exported_env["atmospheric_model_temperature_profile"] == str(
        env.temperature.get_source()
    )
    assert exported_env["atmospheric_model_wind_velocity_x_profile"] == str(
        env.wind_velocity_x.get_source()
    )
    assert exported_env["atmospheric_model_wind_velocity_y_profile"] == str(
        env.wind_velocity_y.get_source()
    )

    os.remove("environment.json")
