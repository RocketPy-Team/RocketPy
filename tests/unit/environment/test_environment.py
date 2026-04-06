import json
import os

import numpy as np
import pytest
import pytz

from rocketpy import Environment
from rocketpy.environment.tools import (
    find_longitude_index,
    geodesic_to_lambert_conformal,
    geodesic_to_utm,
    utm_to_geodesic,
)
from rocketpy.environment.weather_model_mapping import WeatherModelMapping


class DummyLambertProjection:
    """Minimal projection metadata container for unit tests."""

    latitude_of_projection_origin = 40.0
    longitude_of_central_meridian = 263.0
    standard_parallel = np.array([30.0, 60.0])
    earth_radius = 6371229.0


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


def test_geodesic_to_lambert_conformal_projection_origin_maps_to_zero():
    """Tests wrapped central meridian maps to coordinate origin in Lambert conformal."""
    projection = DummyLambertProjection()

    x, y = geodesic_to_lambert_conformal(
        lat=projection.latitude_of_projection_origin,
        lon=projection.longitude_of_central_meridian % 360,
        projection_variable=projection,
        x_units="m",
    )

    assert np.isclose(x, 0.0, atol=1e-8)
    assert np.isclose(y, 0.0, atol=1e-8)


def test_geodesic_to_lambert_conformal_km_units_scale_from_meters():
    """Tests Lambert conformal conversion scales outputs from meters to km."""
    projection = DummyLambertProjection()

    x_meters, y_meters = geodesic_to_lambert_conformal(
        lat=39.0,
        lon=-96.0,
        projection_variable=projection,
        x_units="m",
    )
    x_km, y_km = geodesic_to_lambert_conformal(
        lat=39.0,
        lon=-96.0,
        projection_variable=projection,
        x_units="km",
    )

    assert np.isclose(x_km, x_meters / 1000.0, atol=1e-8)
    assert np.isclose(y_km, y_meters / 1000.0, atol=1e-8)


def test_find_longitude_index_accepts_lower_grid_boundary():
    """Tests longitude equal to first grid value is accepted as in-range."""
    lon_list = [0.0, 0.25, 0.5]

    lon, lon_index = find_longitude_index(0.0, lon_list)

    assert lon == 0.0
    assert lon_index == 1


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


class _DummyDataset:
    """Small test double that mimics a netCDF dataset variables mapping."""

    def __init__(self, variable_names):
        self.variables = {name: object() for name in variable_names}


def test_resolve_dictionary_keeps_compatible_mapping(example_plain_env):
    """Keep the user-selected mapping when it already matches dataset keys."""
    gfs_mapping = example_plain_env._Environment__weather_model_map.get("GFS")
    dataset = _DummyDataset(
        [
            "time",
            "lat",
            "lon",
            "isobaric",
            "Temperature_isobaric",
            "Geopotential_height_isobaric",
            "u-component_of_wind_isobaric",
            "v-component_of_wind_isobaric",
        ]
    )

    resolved = example_plain_env._Environment__resolve_dictionary_for_dataset(
        gfs_mapping, dataset
    )

    assert resolved is gfs_mapping


def test_resolve_dictionary_falls_back_to_legacy_mapping(example_plain_env):
    """Fallback to a compatible built-in mapping for legacy NOMADS-style files."""
    thredds_gfs_mapping = example_plain_env._Environment__weather_model_map.get("GFS")
    dataset = _DummyDataset(
        [
            "time",
            "lat",
            "lon",
            "lev",
            "tmpprs",
            "hgtprs",
            "ugrdprs",
            "vgrdprs",
        ]
    )

    resolved = example_plain_env._Environment__resolve_dictionary_for_dataset(
        thredds_gfs_mapping, dataset
    )

    # Explicit legacy mappings should be preferred over unrelated model mappings.
    assert resolved == example_plain_env._Environment__weather_model_map.get(
        "GFS_LEGACY"
    )
    assert resolved["level"] == "lev"
    assert resolved["temperature"] == "tmpprs"
    assert resolved["geopotential_height"] == "hgtprs"


def test_weather_model_mapping_exposes_legacy_aliases():
    """Legacy mapping names should be available and case-insensitive."""
    mapping = WeatherModelMapping()

    assert mapping.get("GFS_LEGACY")["temperature"] == "tmpprs"
    assert mapping.get("gfs_legacy")["temperature"] == "tmpprs"


def test_dictionary_matches_dataset_rejects_missing_projection(example_plain_env):
    """Reject mapping when projection key is declared but variable is missing."""
    # Arrange
    mapping = {
        "time": "time",
        "latitude": "y",
        "longitude": "x",
        "projection": "LambertConformal_Projection",
        "level": "isobaric",
        "temperature": "Temperature_isobaric",
        "geopotential_height": "Geopotential_height_isobaric",
        "geopotential": None,
        "u_wind": "u-component_of_wind_isobaric",
        "v_wind": "v-component_of_wind_isobaric",
    }
    dataset = _DummyDataset(
        [
            "time",
            "y",
            "x",
            "isobaric",
            "Temperature_isobaric",
            "Geopotential_height_isobaric",
            "u-component_of_wind_isobaric",
            "v-component_of_wind_isobaric",
        ]
    )

    # Act
    is_compatible = example_plain_env._Environment__dictionary_matches_dataset(
        mapping, dataset
    )

    # Assert
    assert not is_compatible


def test_dictionary_matches_dataset_accepts_geopotential_only(example_plain_env):
    """Accept mapping when geopotential exists and geopotential height is absent."""
    # Arrange
    mapping = {
        "time": "time",
        "latitude": "latitude",
        "longitude": "longitude",
        "level": "level",
        "temperature": "t",
        "geopotential_height": None,
        "geopotential": "z",
        "u_wind": "u",
        "v_wind": "v",
    }
    dataset = _DummyDataset(
        [
            "time",
            "latitude",
            "longitude",
            "level",
            "t",
            "z",
            "u",
            "v",
        ]
    )

    # Act
    is_compatible = example_plain_env._Environment__dictionary_matches_dataset(
        mapping, dataset
    )

    # Assert
    assert is_compatible


def test_resolve_dictionary_warns_when_falling_back(example_plain_env):
    """Emit warning and return a built-in mapping when fallback is required."""
    # Arrange
    incompatible_mapping = {
        "time": "bad_time",
        "latitude": "bad_lat",
        "longitude": "bad_lon",
        "level": "bad_level",
        "temperature": "bad_temp",
        "geopotential_height": "bad_height",
        "geopotential": None,
        "u_wind": "bad_u",
        "v_wind": "bad_v",
    }
    dataset = _DummyDataset(
        [
            "time",
            "lat",
            "lon",
            "isobaric",
            "Temperature_isobaric",
            "Geopotential_height_isobaric",
            "u-component_of_wind_isobaric",
            "v-component_of_wind_isobaric",
        ]
    )

    # Act
    with pytest.warns(UserWarning, match="Falling back to built-in mapping"):
        resolved = example_plain_env._Environment__resolve_dictionary_for_dataset(
            incompatible_mapping, dataset
        )

    # Assert
    assert resolved == example_plain_env._Environment__weather_model_map.get("GFS")


def test_resolve_dictionary_returns_original_when_no_compatible_builtin(
    example_plain_env,
):
    """Return original mapping unchanged when no built-in mapping can match."""
    # Arrange
    original_mapping = {
        "time": "a",
        "latitude": "b",
        "longitude": "c",
        "level": "d",
        "temperature": "e",
        "geopotential_height": "f",
        "geopotential": None,
        "u_wind": "g",
        "v_wind": "h",
    }
    dataset = _DummyDataset(["foo", "bar"])

    # Act
    resolved = example_plain_env._Environment__resolve_dictionary_for_dataset(
        original_mapping, dataset
    )

    # Assert
    assert resolved is original_mapping


@pytest.mark.parametrize(
    "model_type,file_name,error_message",
    [
        (
            "Forecast",
            "hiresw",
            "HIRESW latest-model shortcut is currently unavailable",
        ),
        (
            "Ensemble",
            "gefs",
            "GEFS latest-model shortcut is currently unavailable",
        ),
    ],
)
def test_set_atmospheric_model_blocks_deactivated_shortcuts_case_insensitive(
    example_plain_env,
    model_type,
    file_name,
    error_message,
):
    """Reject deactivated shortcut aliases regardless of input string case."""
    # Arrange
    environment = example_plain_env

    # Act / Assert
    with pytest.raises(ValueError, match=error_message):
        environment.set_atmospheric_model(type=model_type, file=file_name)


def test_validate_dictionary_uses_case_insensitive_file_shortcut(example_plain_env):
    """Infer built-in mapping from file shortcut even when shortcut is lowercase."""
    # Arrange
    environment = example_plain_env

    # Act
    mapping = environment._Environment__validate_dictionary("gfs", None)

    # Assert
    assert mapping == environment._Environment__weather_model_map.get("GFS")


def test_validate_dictionary_raises_type_error_for_invalid_dictionary(
    example_plain_env,
):
    """Raise TypeError when no valid dictionary can be inferred."""
    # Arrange
    environment = example_plain_env

    # Act / Assert
    with pytest.raises(TypeError, match="Please specify a dictionary"):
        environment._Environment__validate_dictionary("not_a_model", None)


def test_set_atmospheric_model_normalizes_shortcut_case_for_forecast(example_plain_env):
    """Normalize shortcut name before lookup and process forecast data."""
    # Arrange
    environment = example_plain_env

    environment._Environment__atm_type_file_to_function_map = {
        "forecast": {
            "GFS": lambda: "fake-dataset",
        },
        "ensemble": {},
    }

    called_arguments = {}

    def fake_process_forecast_reanalysis(dataset, dictionary):
        called_arguments["dataset"] = dataset
        called_arguments["dictionary"] = dictionary

    environment.process_forecast_reanalysis = fake_process_forecast_reanalysis

    # Act
    environment.set_atmospheric_model(type="Forecast", file="gfs")

    # Assert
    assert called_arguments["dataset"] == "fake-dataset"
    assert called_arguments[
        "dictionary"
    ] == environment._Environment__weather_model_map.get("GFS")


def test_set_atmospheric_model_raises_for_unknown_model_type(example_plain_env):
    """Raise ValueError for unknown atmospheric model selector."""
    # Arrange
    environment = example_plain_env

    # Act / Assert
    with pytest.raises(ValueError, match="Unknown model type"):
        environment.set_atmospheric_model(type="unknown_type")
