import json
import os
from datetime import datetime

import netCDF4
import numpy as np
import numpy.testing as npt
import pytest
import pytz

from rocketpy import Environment, Function
from rocketpy.environment.tools import (
    find_longitude_index,
    geodesic_to_lambert_conformal,
    geodesic_to_utm,
    get_final_date_from_time_array,
    get_initial_date_from_time_array,
    get_interval_date_from_time_array,
    get_pressure_levels_from_file,
    pressure_unit_to_factor,
    utm_to_geodesic,
)
from rocketpy.environment.weather_model_mapping import WeatherModelMapping
from rocketpy.tools import geopotential_height_to_geometric_height


def _user_defined_ensemble_profiles():
    """Return two members with a shared isobaric grid."""
    pressure = np.array([101325.0, 90000.0, 80000.0])
    member_0_height = np.array([0.0, 1000.0, 2000.0])
    member_1_height = np.array([100.0, 1100.0, 2100.0])
    return [
        {
            "pressure": np.column_stack((member_0_height, pressure)),
            "temperature": np.column_stack((member_0_height, [288.0, 281.0, 275.0])),
            "wind_u": np.column_stack((member_0_height, [1.0, 2.0, 3.0])),
            "wind_v": np.column_stack((member_0_height, [-1.0, -2.0, -3.0])),
        },
        {
            "pressure": np.column_stack((member_1_height, pressure)),
            "temperature": np.column_stack((member_1_height, [290.0, 283.0, 277.0])),
            "wind_u": np.column_stack((member_1_height, [4.0, 5.0, 6.0])),
            "wind_v": np.column_stack((member_1_height, [-4.0, -5.0, -6.0])),
        },
    ]


def test_time_array_interval_helper_accepts_a_single_time():
    """A static user ensemble has no forecast interval."""

    class SingleTimeArray:
        """Minimal single-value NetCDF-like time coordinate."""

        units = "hours since 2025-06-01 12:00:00"

        def __len__(self):
            return 1

    assert get_interval_date_from_time_array(SingleTimeArray()) == 0


def test_create_ensemble_exports_and_activates_profiles(tmp_path):
    """Export user profiles and expose each member through Environment."""
    # Arrange
    env = Environment(
        date=(2025, 6, 1, 12),
        latitude=32.99,
        longitude=-106.97,
        elevation=0,
    )
    output = tmp_path / "test_ensemble"

    # Act
    file_path = env.create_ensemble(_user_defined_ensemble_profiles(), file_name=output)

    # Assert
    assert file_path == str(output) + ".nc"
    assert env.atmospheric_model_type == "Ensemble"
    assert env.num_ensemble_members == 2
    assert env.ensemble_member == 0
    assert env.pressure(1000) == pytest.approx(90000)
    assert env.temperature(1000) == pytest.approx(281)
    assert env.wind_velocity_x(1000) == pytest.approx(2)

    env.select_ensemble_member(1)
    assert env.pressure(1100) == pytest.approx(90000)
    assert env.temperature(1100) == pytest.approx(283)
    assert env.wind_velocity_x(1100) == pytest.approx(5)
    assert env.wind_velocity_y(1100) == pytest.approx(-5)

    with netCDF4.Dataset(file_path) as dataset:
        assert dataset.Conventions == "CF-1.8"
        assert dataset.source == "RocketPy Environment.create_ensemble"
        assert dataset.variables["time"].long_name == "profile valid time"
        assert {
            name: len(dataset.dimensions[name]) for name in ("ens", "lev", "time")
        } == {"ens": 2, "lev": 3, "time": 1}
        assert dataset.variables["lev"].units == "hPa"
        assert dataset.variables["tmpprs"].standard_name == "air_temperature"
        npt.assert_allclose(dataset.variables["lev"][:], [1013.25, 900, 800])


def test_create_ensemble_file_round_trip(tmp_path):
    """Reload the exported file using the existing GEFS ensemble mapping."""
    # Arrange
    source_env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)
    file_path = source_env.create_ensemble(
        _user_defined_ensemble_profiles(), file_name=tmp_path / "round_trip.nc"
    )
    loaded_env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act
    loaded_env.set_atmospheric_model(type="Ensemble", file=file_path, dictionary="GEFS")
    loaded_env.select_ensemble_member(1)

    # Assert
    assert loaded_env.num_ensemble_members == 2
    assert loaded_env.pressure(1100) == pytest.approx(90000)
    assert loaded_env.temperature(1100) == pytest.approx(283)
    assert loaded_env.wind_velocity_x(1100) == pytest.approx(5)
    assert loaded_env.wind_velocity_y(1100) == pytest.approx(-5)


def test_create_ensemble_rejects_non_overlapping_pressure_profiles(tmp_path):
    """Reject members that cannot be sampled on a common pressure grid."""
    # Arrange
    profiles = _user_defined_ensemble_profiles()
    heights = profiles[1]["pressure"][:, 0]
    profiles[1]["pressure"] = np.column_stack((heights, [70000.0, 60000.0, 50000.0]))

    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act / Assert
    with pytest.raises(ValueError, match="no common pressure range"):
        env.create_ensemble(profiles, file_name=tmp_path / "invalid.nc")


def test_create_ensemble_does_not_overwrite_by_default(tmp_path):
    """Preserve an existing ensemble file unless overwrite is explicit."""
    # Arrange
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)
    file_path = env.create_ensemble(
        _user_defined_ensemble_profiles(), file_name=tmp_path / "existing.nc"
    )

    # Act / Assert
    with pytest.raises(FileExistsError, match="overwrite=True"):
        env.create_ensemble(_user_defined_ensemble_profiles(), file_name=file_path)


def test_create_ensemble_accepts_array_functions_and_explicit_levels(tmp_path):
    """Accept array-backed Functions and sort explicit pressure levels."""
    # Arrange
    profiles = _user_defined_ensemble_profiles()
    for member in profiles:
        for variable, source in member.items():
            member[variable] = Function(source)
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act
    file_path = env.create_ensemble(
        profiles,
        file_name=tmp_path / "function_profiles.nc",
        pressure_levels=[80000, 101325, 90000],
    )

    # Assert
    with netCDF4.Dataset(file_path) as dataset:
        npt.assert_allclose(dataset.variables["lev"][:], [1013.25, 900, 800])


@pytest.mark.parametrize(
    "source, error, match",
    [
        (Function(lambda height: height), TypeError, "array-backed Function"),
        (object(), TypeError, "two-column numeric array"),
        (np.array([0.0, 1.0]), ValueError, "at least two"),
        (np.array([[0.0, 1.0], [1.0, np.inf]]), ValueError, "non-finite"),
        (np.array([[0.0, 1.0], [0.0, 2.0]]), ValueError, "heights must be unique"),
    ],
)
def test_create_ensemble_rejects_invalid_profile_sources(
    tmp_path, source, error, match
):
    """Reject profile sources that cannot define a finite height-value curve."""
    # Arrange
    profiles = _user_defined_ensemble_profiles()
    profiles[0]["wind_u"] = source
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act / Assert
    with pytest.raises(error, match=match):
        env.create_ensemble(profiles, file_name=tmp_path / "invalid_source.nc")


def test_create_ensemble_rejects_invalid_profile_collections(tmp_path):
    """Reject invalid ensemble containers and incomplete members."""
    # Arrange
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)
    profiles = _user_defined_ensemble_profiles()
    output = tmp_path / "invalid_collection.nc"

    # Act / Assert
    with pytest.raises(TypeError, match="sequence of member mappings"):
        env.create_ensemble(profiles[0], file_name=output)
    with pytest.raises(TypeError, match="sequence of member mappings"):
        env.create_ensemble(1, file_name=output)
    with pytest.raises(ValueError, match="At least two"):
        env.create_ensemble(profiles[:1], file_name=output)
    with pytest.raises(TypeError, match="Member 1 must be a mapping"):
        env.create_ensemble([profiles[0], None], file_name=output)

    incomplete_profiles = _user_defined_ensemble_profiles()
    incomplete_profiles[1].pop("wind_v")
    with pytest.raises(ValueError, match="missing required profile.*wind_v"):
        env.create_ensemble(incomplete_profiles, file_name=output)


@pytest.mark.parametrize(
    "variable, values, match",
    [
        ("pressure", [101325.0, 0.0, 80000.0], "pressure values must be positive"),
        (
            "pressure",
            [101325.0, 80000.0, 90000.0],
            "pressure must decrease strictly",
        ),
        ("temperature", [288.0, 0.0, 275.0], "temperature values must be positive"),
    ],
)
def test_create_ensemble_rejects_invalid_profile_values(
    tmp_path, variable, values, match
):
    """Reject nonphysical pressure and temperature profile values."""
    # Arrange
    profiles = _user_defined_ensemble_profiles()
    profiles[0][variable][:, 1] = values
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act / Assert
    with pytest.raises(ValueError, match=match):
        env.create_ensemble(profiles, file_name=tmp_path / "invalid_values.nc")


@pytest.mark.parametrize(
    "pressure_levels, error, match",
    [
        (["invalid", "values"], TypeError, "numeric array"),
        ([[101325.0, 90000.0]], ValueError, "one-dimensional"),
        ([101325.0, np.nan], ValueError, "finite, positive"),
        ([101325.0, 101325.0], ValueError, "duplicates"),
        ([90000.0], ValueError, "At least two pressure levels"),
        ([110000.0, 90000.0], ValueError, "inside the pressure range"),
    ],
)
def test_create_ensemble_rejects_invalid_pressure_levels(
    tmp_path, pressure_levels, error, match
):
    """Reject explicit pressure grids that cannot be shared by all members."""
    # Arrange
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act / Assert
    with pytest.raises(error, match=match):
        env.create_ensemble(
            _user_defined_ensemble_profiles(),
            file_name=tmp_path / "invalid_levels.nc",
            pressure_levels=pressure_levels,
        )


def test_create_ensemble_rejects_profiles_without_height_coverage(tmp_path):
    """Require every variable to span the common pressure-grid heights."""
    # Arrange
    profiles = _user_defined_ensemble_profiles()
    profiles[0]["temperature"] = profiles[0]["temperature"][1:]
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act / Assert
    with pytest.raises(ValueError, match="temperature.*does not cover all heights"):
        env.create_ensemble(profiles, file_name=tmp_path / "incomplete_height.nc")


def test_create_ensemble_rejects_heights_below_earth_center(tmp_path):
    """Reject geometric heights at or below the coordinate singularity."""
    # Arrange
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)
    profiles = _user_defined_ensemble_profiles()
    invalid_heights = np.array(
        [-env.earth_radius - 2000, -env.earth_radius - 1000, -env.earth_radius - 1]
    )
    for member in profiles:
        for source in member.values():
            source[:, 0] = invalid_heights

    # Act / Assert
    with pytest.raises(ValueError, match="greater than -Earth's radius"):
        env.create_ensemble(profiles, file_name=tmp_path / "invalid_height.nc")


def test_create_ensemble_rejects_invalid_file_name():
    """Require the NetCDF output name to implement the path protocol."""
    # Arrange
    env = Environment(date=(2025, 6, 1, 12), latitude=32.99, longitude=-106.97)

    # Act / Assert
    with pytest.raises(TypeError, match="string or path-like"):
        env.create_ensemble(_user_defined_ensemble_profiles(), file_name=object())


class DummyLambertProjection:
    """Minimal projection metadata container for unit tests."""

    latitude_of_projection_origin = 40.0
    longitude_of_central_meridian = 263.0
    standard_parallel = np.array([30.0, 60.0])
    earth_radius = 6371229.0


@pytest.mark.parametrize(
    "date_helper", [get_initial_date_from_time_array, get_final_date_from_time_array]
)
def test_time_array_date_helpers_convert_cftime_dates(
    monkeypatch, date_helper, dummy_time_array, dummy_cftime_date
):
    """Convert NetCDF/cftime date objects to JSON-serializable datetimes."""

    # Arrange
    def fake_num2date(*_args, **_kwargs):
        return dummy_cftime_date

    monkeypatch.setattr("rocketpy.environment.tools.netCDF4.num2date", fake_num2date)

    # Act
    converted_date = date_helper(dummy_time_array)

    # Assert
    assert converted_date == datetime(2023, 6, 24, 9, 30, 15, 123456)


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
    assert str(exported_env["atmospheric_model_pressure_profile"]) == str(
        env.pressure.get_source()
    )
    assert str(exported_env["atmospheric_model_temperature_profile"]) == str(
        env.temperature.get_source()
    )
    assert str(exported_env["atmospheric_model_wind_velocity_x_profile"]) == str(
        env.wind_velocity_x.get_source()
    )
    assert str(exported_env["atmospheric_model_wind_velocity_y_profile"]) == str(
        env.wind_velocity_y.get_source()
    )

    os.remove("environment.json")


@pytest.mark.parametrize(
    "atmospheric_model_type",
    ["windy", "meteomatics", "forecast", "reanalysis", "ensemble"],
)
def test_environment_to_dict_from_dict_round_trip_preserves_weather_metadata(
    example_plain_env, atmospheric_model_type
):
    """Round-trip weather-model environments without losing metadata.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
        Baseline environment used to build the serialized state.
    atmospheric_model_type : str
        Weather-model label stored in the serialized payload.
    """
    # Arrange
    env = example_plain_env

    weather_metadata = {
        "atmospheric_model_type": atmospheric_model_type,
        "atmospheric_model_file": None,
        "atmospheric_model_dict": {"time": "time"},
        "atmospheric_model_init_date": datetime(2024, 1, 1, 0),
        "atmospheric_model_end_date": datetime(2024, 1, 1, 6),
        "atmospheric_model_interval": 6,
        "atmospheric_model_init_lat": -10.0,
        "atmospheric_model_end_lat": 10.0,
        "atmospheric_model_init_lon": -20.0,
        "atmospheric_model_end_lon": 20.0,
    }

    ensemble_metadata = {
        "level_ensemble": None,
        "height_ensemble": None,
        "temperature_ensemble": None,
        "wind_u_ensemble": None,
        "wind_v_ensemble": None,
        "wind_heading_ensemble": None,
        "wind_direction_ensemble": None,
        "wind_speed_ensemble": None,
        "num_ensemble_members": None,
    }

    if atmospheric_model_type == "ensemble":
        ensemble_metadata.update(
            {
                "level_ensemble": np.array([1000.0, 900.0]),
                "height_ensemble": np.array([[0.0, 1000.0], [0.0, 1000.0]]),
                "temperature_ensemble": np.array([[288.15, 281.15], [288.15, 281.15]]),
                "wind_u_ensemble": np.array([[2.0, 3.0], [2.0, 3.0]]),
                "wind_v_ensemble": np.array([[4.0, 5.0], [4.0, 5.0]]),
                "wind_heading_ensemble": np.array(
                    [[26.565051, 30.963757], [26.565051, 30.963757]]
                ),
                "wind_direction_ensemble": np.array(
                    [[206.565051, 210.963757], [206.565051, 210.963757]]
                ),
                "wind_speed_ensemble": np.array(
                    [[4.472136, 5.830952], [4.472136, 5.830952]]
                ),
                "num_ensemble_members": 2,
                "ensemble_member": 1,
            }
        )

    for metadata in (weather_metadata, ensemble_metadata):
        for attribute, value in metadata.items():
            setattr(env, attribute, value)

    env_dict = env.to_dict()

    # The serialized payload should be self-contained and not depend on files.
    assert "atmospheric_model_file" not in env_dict
    assert "atmospheric_model_dict" not in env_dict

    # Act
    restored_env = Environment.from_dict(env_dict)

    # Assert
    assert restored_env.atmospheric_model_type == atmospheric_model_type
    assert restored_env.atmospheric_model_init_date == env.atmospheric_model_init_date
    assert restored_env.atmospheric_model_end_date == env.atmospheric_model_end_date
    assert restored_env.atmospheric_model_interval == env.atmospheric_model_interval
    assert restored_env.atmospheric_model_init_lat == env.atmospheric_model_init_lat
    assert restored_env.atmospheric_model_end_lat == env.atmospheric_model_end_lat
    assert restored_env.atmospheric_model_init_lon == env.atmospheric_model_init_lon
    assert restored_env.atmospheric_model_end_lon == env.atmospheric_model_end_lon

    if atmospheric_model_type == "ensemble":
        npt.assert_allclose(restored_env.level_ensemble, env.level_ensemble)
        npt.assert_allclose(restored_env.height_ensemble, env.height_ensemble)
        assert restored_env.num_ensemble_members == env.num_ensemble_members
        assert restored_env.ensemble_member == env.ensemble_member == 1


_METEOMATICS_FAKE_PROFILES = {
    "temperature": {0: 288.15, 1000: 281.65, 5000: 255.65},
    "pressure": {0: 101325.0, 1000: 89876.0, 5000: 54048.0},
    "wind_u": {0: 1.0, 1000: 3.0, 5000: 8.0},
    "wind_v": {0: -1.0, 1000: -2.0, 5000: -4.0},
}


def _patch_meteomatics_fetcher(monkeypatch, profiles=None, recorder=None):
    """Replace the Meteomatics fetcher with an offline fake (no API calls)."""
    profiles = _METEOMATICS_FAKE_PROFILES if profiles is None else profiles

    def fake_fetch(**kwargs):
        if recorder is not None:
            recorder.update(kwargs)
        return profiles

    monkeypatch.setattr(
        "rocketpy.environment.environment.fetch_atmospheric_data_from_meteomatics",
        fake_fetch,
    )


def test_meteomatics_atmosphere_sets_profiles(example_euroc_env, monkeypatch):
    """Build pressure, temperature and wind profiles from Meteomatics data.

    The fake profiles are indexed by height above ground level, so the
    Environment elevation (100 m for the EuRoC fixture) must be added to obtain
    heights above sea level.
    """
    recorder = {}
    _patch_meteomatics_fetcher(monkeypatch, recorder=recorder)

    example_euroc_env.set_atmospheric_model(
        type="Meteomatics", file="mix", username="user", password="pass"
    )

    assert example_euroc_env.atmospheric_model_type == "Meteomatics"
    # AGL 0 m -> ASL 100 m (the fixture elevation)
    assert pytest.approx(101325.0, rel=1e-6) == example_euroc_env.pressure(100)
    assert pytest.approx(288.15, rel=1e-6) == example_euroc_env.temperature(100)
    assert pytest.approx(1.0) == example_euroc_env.wind_velocity_x(100)
    assert pytest.approx(-1.0) == example_euroc_env.wind_velocity_y(100)
    assert pytest.approx(np.sqrt(2.0)) == example_euroc_env.wind_speed(100)
    assert example_euroc_env.max_expected_height == pytest.approx(5100.0)
    # Credentials and model are forwarded to the fetcher.
    assert recorder["username"] == "user"
    assert recorder["password"] == "pass"
    assert recorder["model"] == "mix"


def test_meteomatics_non_string_model_raises(example_euroc_env, monkeypatch):
    """Reject a non-string model instead of silently querying the default.

    Passing a Dataset or a path as ``file`` by accident must not be coerced to
    ``"mix"``, which would quietly query (and charge for) the wrong model.
    """
    _patch_meteomatics_fetcher(monkeypatch)

    with pytest.raises(ValueError, match="Invalid Meteomatics model"):
        example_euroc_env.set_atmospheric_model(
            type="Meteomatics", file=123, username="user", password="pass"
        )


def test_meteomatics_reads_credentials_from_environment(example_euroc_env, monkeypatch):
    """Fall back to the METEOMATICS_* environment variables for credentials."""
    recorder = {}
    _patch_meteomatics_fetcher(monkeypatch, recorder=recorder)
    monkeypatch.setenv("METEOMATICS_USERNAME", "env-user")
    monkeypatch.setenv("METEOMATICS_PASSWORD", "env-pass")

    example_euroc_env.set_atmospheric_model(type="Meteomatics")

    assert recorder["username"] == "env-user"
    assert recorder["password"] == "env-pass"
    assert recorder["model"] == "mix"  # default model when file is omitted
    assert pytest.approx(288.15, rel=1e-6) == example_euroc_env.temperature(100)


def test_meteomatics_missing_credentials_raises(example_euroc_env, monkeypatch):
    """Raise a clear error when no credentials are available."""
    _patch_meteomatics_fetcher(monkeypatch)
    monkeypatch.delenv("METEOMATICS_USERNAME", raising=False)
    monkeypatch.delenv("METEOMATICS_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="username and password"):
        example_euroc_env.set_atmospheric_model(type="Meteomatics")


def test_meteomatics_missing_date_raises(example_plain_env, monkeypatch):
    """Raise when the Environment has no launch date set."""
    _patch_meteomatics_fetcher(monkeypatch)

    with pytest.raises(ValueError, match="launch date"):
        example_plain_env.set_atmospheric_model(
            type="Meteomatics", username="user", password="pass"
        )


def test_meteomatics_drops_missing_values_and_intersects_wind_grid(
    example_euroc_env, monkeypatch
):
    """Drop ``None`` values and keep only wind levels present in both u and v.

    Temperature at 1000 m is ``None`` (dropped), and the wind grids disagree at
    5000 m (only ``wind_u`` has it), so the wind profile must keep only the
    common, non-null levels {0, 1000} m AGL.
    """
    profiles = {
        "temperature": {0: 288.15, 1000: None, 5000: 255.65},
        "pressure": {0: 101325.0, 5000: 54048.0},
        "wind_u": {0: 1.0, 1000: 3.0, 5000: 8.0},
        "wind_v": {0: -1.0, 1000: -2.0},  # missing 5000 -> intersection drops it
    }
    _patch_meteomatics_fetcher(monkeypatch, profiles=profiles)

    example_euroc_env.set_atmospheric_model(
        type="Meteomatics", username="user", password="pass"
    )

    # Wind kept only the two common non-null AGL levels {0, 1000} -> ASL {100, 1100}.
    npt.assert_array_equal(example_euroc_env.height, [100.0, 1100.0])
    assert len(example_euroc_env.wind_us) == 2
    # Temperature dropped the None level: {0, 5000} AGL -> ASL {100, 5100}.
    assert len(example_euroc_env.temperatures) == 2
    assert pytest.approx(255.65, rel=1e-6) == example_euroc_env.temperature(5100)
    assert example_euroc_env.max_expected_height == pytest.approx(5100.0)


def test_meteomatics_no_usable_data_raises(example_euroc_env, monkeypatch):
    """Raise a clear error when the API returns no usable wind data."""
    profiles = {
        "temperature": {0: 288.15},
        "pressure": {0: 101325.0},
        "wind_u": {},
        "wind_v": {},
    }
    _patch_meteomatics_fetcher(monkeypatch, profiles=profiles)

    with pytest.raises(ValueError, match="usable atmospheric data"):
        example_euroc_env.set_atmospheric_model(
            type="Meteomatics", username="user", password="pass"
        )


def test_meteomatics_single_level_profile_raises(example_euroc_env, monkeypatch):
    """Reject a collapsed grid (one level per profile) up front.

    A single altitude level builds a Function that cannot be evaluated at its
    own node, so ``set_atmospheric_model`` must fail immediately with a clear
    message rather than succeed and crash later at ``pressure``/``density``.
    """
    profiles = {
        "temperature": {0: 288.15},
        "pressure": {0: 101325.0},
        "wind_u": {0: 1.0},
        "wind_v": {0: -1.0},
    }
    _patch_meteomatics_fetcher(monkeypatch, profiles=profiles)

    with pytest.raises(ValueError, match="at least two valid altitude levels"):
        example_euroc_env.set_atmospheric_model(
            type="Meteomatics", username="user", password="pass"
        )


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


def test_resolve_dictionary_falls_back_to_first_compatible_mapping(example_plain_env):
    """Fallback to the first compatible built-in mapping for legacy-style files."""
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

    def fake_process_forecast_reanalysis(dataset, dictionary, conversion_factor):
        called_arguments["dataset"] = dataset
        called_arguments["dictionary"] = dictionary
        called_arguments["conversion_factr"] = conversion_factor

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


def test_wind_heading_direction_wraparound_interpolation(example_plain_env):
    """Test that wind heading and direction interpolation wraps around correctly
    across the 360°/0° boundary when initialized with a 2D array.
    """
    # Create discrete points at 1000m and 1100m
    # 350 deg at 1000m, 10 deg at 1100m.
    # Midpoint should be 360 deg or 0 deg, NOT 180 deg.
    heading_data = np.array([[1000, 350], [1100, 10]])
    direction_data = np.array([[1000, 350], [1100, 10]])

    example_plain_env._Environment__set_wind_heading_function(heading_data)
    example_plain_env._Environment__set_wind_direction_function(direction_data)

    # Evaluate at midpoint (1050m)
    mid_heading = example_plain_env.wind_heading(1050)
    mid_direction = example_plain_env.wind_direction(1050)

    # Check that it's close to 0 or 360 (which is also 0 modulo 360)
    assert np.isclose(mid_heading, 0.0) or np.isclose(mid_heading, 360.0)
    assert np.isclose(mid_direction, 0.0) or np.isclose(mid_direction, 360.0)

    # Also test another wrap-around case, e.g. 10 to 350
    heading_data2 = np.array([[1000, 10], [1100, 350]])
    example_plain_env._Environment__set_wind_heading_function(heading_data2)
    mid_heading2 = example_plain_env.wind_heading(1050)
    assert np.isclose(mid_heading2, 0.0) or np.isclose(mid_heading2, 360.0)


@pytest.mark.parametrize("shortcut_name", ["AIGFS", "HRRR"])
def test_forecast_shortcut_and_dictionary_are_case_insensitive(
    monkeypatch, shortcut_name
):
    """Ensure forecast shortcuts and built-in dictionaries ignore input casing."""
    # Arrange
    env = Environment(date=(2026, 3, 17, 12), latitude=32.99, longitude=-106.97)

    sentinel_dataset = object()
    env._Environment__atm_type_file_to_function_map["forecast"][shortcut_name] = (
        lambda: sentinel_dataset
    )

    captured = {}

    def fake_process_forecast_reanalysis(file, dictionary, conversion_factor):
        captured["file"] = file
        captured["dictionary"] = dictionary
        captured["conversion_factor"] = conversion_factor

    monkeypatch.setattr(
        env, "process_forecast_reanalysis", fake_process_forecast_reanalysis
    )
    monkeypatch.setattr(env, "calculate_density_profile", lambda: None)
    monkeypatch.setattr(env, "calculate_speed_of_sound_profile", lambda: None)
    monkeypatch.setattr(env, "calculate_dynamic_viscosity", lambda: None)

    # Act
    env.set_atmospheric_model(
        type="forecast",
        file=shortcut_name.lower(),
        dictionary=shortcut_name.lower(),
    )

    # Assert
    expected_dictionary = env._Environment__weather_model_map.get(shortcut_name)
    assert captured["file"] is sentinel_dataset
    assert captured["dictionary"] == expected_dictionary
    assert env.atmospheric_model_file == shortcut_name
    assert env.atmospheric_model_dict == expected_dictionary


def test_weather_model_mapping_get_is_case_insensitive():
    """Ensure built-in mapping names are resolved regardless of casing."""
    mapping = WeatherModelMapping()
    assert mapping.get("aigfs") == mapping.get("AIGFS")
    assert mapping.get("ecmwf_v0") == mapping.get("ECMWF_v0")


@pytest.mark.parametrize(
    "model, expected_factor",
    [
        # NOMADS-GrADS models expose pressure on the 'lev' coordinate in hPa
        # and MUST be scaled by 100 (regression: they were forced to factor 1).
        ("GEFS", 100),
        ("HIRESW", 100),
        # THREDDS (UCAR) models expose pressure on 'isobaric' already in Pa.
        ("GFS", 1),
        ("NAM", 1),
        ("RAP", 1),
        ("HRRR", 1),
        ("AIGFS", 1),
    ],
)
def test_pressure_conversion_factor_autodetect_by_model(
    example_plain_env, model, expected_factor
):
    """Regression test for the GEFS/HIRESW pressure-unit bug: NOMADS-GrADS
    models report pressure in hPa (factor 100), THREDDS models in Pa (factor 1).
    A wrong factor silently corrupts the whole atmospheric profile (100x)."""
    factor = example_plain_env._Environment__determine_pressure_conversion_factor(
        None, None, model
    )
    assert factor == expected_factor


def test_pressure_isa_discretization_bounds(example_plain_env):
    """The pressure_ISA discretization must span the full range of the
    Standard Atmosphere model: from the lowest geopotential layer (-2000 m) up
    to the highest (80000 m), both converted to geometric height. It must also
    be a physically sane pressure curve: altitude strictly increasing, pressure
    strictly decreasing, and sea level (0 m) sampled exactly.
    """

    # Act
    pressure_isa_function = example_plain_env.pressure_ISA
    source_array = pressure_isa_function.source
    altitudes = source_array[:, 0]
    pressures = source_array[:, 1]

    # Expected min/max geometric heights
    earth_radius = example_plain_env.earth_radius
    expected_min_height = geopotential_height_to_geometric_height(-2000, earth_radius)
    expected_max_height = geopotential_height_to_geometric_height(80000, earth_radius)

    # Assert
    assert len(altitudes) == 100
    assert np.isclose(altitudes[0], expected_min_height)
    assert np.isclose(altitudes[-1], expected_max_height)
    assert expected_min_height < 0 < expected_max_height
    # Sea level must be one of the sampled points (split boundary)
    assert np.any(np.isclose(altitudes, 0.0))
    # Physical sanity: altitude increasing, pressure decreasing monotonically
    assert np.all(np.diff(altitudes) > 0)
    assert np.all(np.diff(pressures) < 0)


@pytest.mark.parametrize(
    "model, expected_factor",
    [("GEFS", 100), ("HIRESW", 100), ("GFS", 1), ("AIGFS", 1)],
)
def test_pressure_conversion_factor_autodetect_by_dictionary(
    example_plain_env, model, expected_factor
):
    """Model shortcuts arriving via ``dictionary`` (the realistic download
    path) must map to the same factor as when they arrive via ``file``."""
    factor = example_plain_env._Environment__determine_pressure_conversion_factor(
        None, model, None
    )
    assert factor == expected_factor


@pytest.mark.parametrize(
    "units, expected_levels",
    [
        ("mb", [100000.0, 85000.0]),
        ("millibar", [100000.0, 85000.0]),
        ("millibars", [100000.0, 85000.0]),
        ("hPa", [100000.0, 85000.0]),
        ("mbar", [100000.0, 85000.0]),
        ("Pa", [1000.0, 850.0]),
    ],
)
def test_get_pressure_levels_from_file_unit_synonyms(units, expected_levels):
    """hPa/millibar unit synonyms auto-scale by 100; Pa by 1."""

    class _Var:
        def __init__(self, values, units):
            self._values = np.asarray(values)
            self.units = units

        def __getitem__(self, key):
            return self._values[key]

    class _DS:
        def __init__(self, var):
            self.variables = {"lev": var}

    dataset = _DS(_Var([1000.0, 850.0], units))
    levels = get_pressure_levels_from_file(dataset, {"level": "lev"}, None)
    npt.assert_allclose(levels, expected_levels)


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("mbar", 100),
        ("mb", 100),
        ("hPa", 100),
        ("millibar", 100),
        ("millibars", 100),
        ("hectopascal", 100),
        ("Pa", 1),
        ("pascal", 1),
        ("parsecs", None),
        ("", None),
    ],
)
def test_pressure_unit_to_factor(unit, expected):
    """The shared unit->factor helper: hPa synonyms ->100, Pa ->1, else None."""

    assert pressure_unit_to_factor(unit) == expected


@pytest.mark.parametrize("unit, expected", [("mb", 100), ("millibar", 100), ("Pa", 1)])
def test_pressure_conversion_factor_explicit_unit_synonyms(
    example_plain_env, unit, expected
):
    """An explicit string ``pressure_conversion_factor`` accepts the same unit
    synonyms as file auto-detection (Copilot review consistency fix)."""
    factor = example_plain_env._Environment__determine_pressure_conversion_factor(
        unit, None, None
    )
    assert factor == expected


def test_set_atmospheric_model_rejects_unknown_pressure_unit(example_plain_env):
    """An unrecognized ``pressure_conversion_factor`` unit is rejected during
    validation, before any file access."""
    with pytest.raises(ValueError, match="pressure_conversion_factor"):
        example_plain_env.set_atmospheric_model(
            type="Forecast", file="dummy", pressure_conversion_factor="parsecs"
        )
