import json
import os

import numpy as np
import pytest
from pytest import approx

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.tools import euler313_to_quaternions

# calisto standard simulation no wind solution index 200
TIME = 3.338513236767685
U = [
    0.02856482783411794,
    50.919436628139216,
    1898.9056294848442,
    0.021620542063162787,
    30.468683793837055,
    284.19140267225384,
    -0.0076008223256743114,
    0.0004430927976100488,
    0.05330950836930627,
    0.9985245671704497,
    0.0026388673982115224,
    0.00010697759229808481,
    19.72526891699468,
]
U_DOT = [
    0.021620542063162787,
    30.468683793837055,
    284.19140267225384,
    0.0009380154986373648,
    1.4853035773069556,
    4.377014845613867,
    -9.848086239924413,
    0.5257087555505318,
    -0.0030529818895471124,
    -0.07503444684343626,
    0.028008532884449017,
    -0.052789015849051935,
    2.276425320359305,
]
GRAVITY = 9.81


@pytest.mark.parametrize(
    "sensor",
    [
        "noisy_rotated_accelerometer",
        "quantized_accelerometer",
        "noisy_rotated_gyroscope",
        "quantized_gyroscope",
        "noisy_barometer",
        "quantized_barometer",
        "noisy_gnss",
    ],
)
def test_sensors_prints(sensor, request):
    """Test the print methods of the Sensor class. Checks if all attributes are
    printed correctly.
    """
    sensor = request.getfixturevalue(sensor)
    sensor.prints.all()
    assert True


def test_rotation_matrix(noisy_rotated_accelerometer):
    """Test the rotation_matrix property of the InertialSensor class. Checks if
    the rotation matrix is correctly calculated.
    """
    # values from external source
    expected_matrix = np.array(
        [
            [-0.125, -0.6495190528383292, 0.7499999999999999],
            [0.6495190528383292, -0.625, -0.43301270189221946],
            [0.7499999999999999, 0.43301270189221946, 0.5000000000000001],
        ]
    )
    rotation_matrix = np.array(
        noisy_rotated_accelerometer.rotation_sensor_to_body.components
    )
    assert np.allclose(expected_matrix, rotation_matrix, atol=1e-8)


def test_inertial_quantization(quantized_accelerometer):
    """Test the quantize method of the InertialSensor class. Checks if returned values
    are as expected.
    """
    # expected values calculated by hand
    assert quantized_accelerometer.quantize(Vector([3, 3, 3])) == Vector(
        [1.9528, 1.9528, 1.9528]
    )
    assert quantized_accelerometer.quantize(Vector([-3, -3, -3])) == Vector(
        [-1.9528, -1.9528, -1.9528]
    )
    assert quantized_accelerometer.quantize(Vector([1, 1, 1])) == Vector(
        [0.9764, 0.9764, 0.9764]
    )


def test_scalar_quantization(quantized_barometer):
    """Test the quantize method of the ScalarSensor class. Checks if returned values
    are as expected.
    """
    # expected values calculated by hand
    assert quantized_barometer.quantize(7e5) == 7e4
    assert quantized_barometer.quantize(-7e5) == -7e4
    assert quantized_barometer.quantize(1001) == 1000.96


@pytest.mark.parametrize(
    "sensor, input_value, expected_output",
    [
        (
            "quantized_accelerometer",
            Vector([3, 3, 3]),
            Vector([1.9528, 1.9528, 1.9528]),
        ),
        (
            "quantized_accelerometer",
            Vector([-3, -3, -3]),
            Vector([-1.9528, -1.9528, -1.9528]),
        ),
        (
            "quantized_accelerometer",
            Vector([1, 1, 1]),
            Vector([0.9764, 0.9764, 0.9764]),
        ),
        ("quantized_barometer", 7e5, 7e4),
        ("quantized_barometer", -7e5, -7e4),
        ("quantized_barometer", 1001, 1000.96),
    ],
)
def test_quantization(sensor, input_value, expected_output, request):
    """Test the quantize method of various sensor classes. Checks if returned values
    are as expected.

    Parameters
    ----------
    sensor : str
        Fixture name of the sensor to be tested.
    input_value : any
        Input value to be quantized by the sensor.
    expected_output : any
        Expected output value after quantization.
    """
    sensor = request.getfixturevalue(sensor)
    result = sensor.quantize(input_value)
    assert result == expected_output


@pytest.mark.parametrize(
    "sensor",
    [
        "ideal_accelerometer",
        "ideal_gyroscope",
    ],
)
def test_inertial_measured_data(sensor, request, example_plain_env):
    """Test the measured_data property of the Sensor class. Checks if
    the measured data is treated properly when the sensor is added once or more
    than once to the rocket.
    """
    sensor = request.getfixturevalue(sensor)

    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data) == 1
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data) == 2
    assert all(isinstance(i, tuple) for i in sensor.measured_data)

    # check case when sensor is added more than once to the rocket
    sensor.measured_data = [
        sensor.measured_data[:],
        sensor.measured_data[:],
    ]
    sensor._save_data = sensor._save_data_multiple
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data) == 2
    assert len(sensor.measured_data[0]) == 3
    assert len(sensor.measured_data[1]) == 2
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data[0]) == 3
    assert len(sensor.measured_data[1]) == 3


@pytest.mark.parametrize(
    "sensor",
    [
        "ideal_barometer",
        "ideal_gnss",
    ],
)
def test_scalar_measured_data(sensor, request, example_plain_env):
    """Test the measure method of ScalarSensor. Checks if saved
    measurement is (P) and if measured_data is [(t, P), ...]
    """
    sensor = request.getfixturevalue(sensor)

    t = TIME
    u = U

    sensor.measure(
        t,
        u=u,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data) == 1
    sensor.measure(
        t,
        u=u,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data) == 2
    assert all(isinstance(i, tuple) for i in sensor.measured_data)

    # check case when sensor is added more than once to the rocket
    sensor.measured_data = [
        sensor.measured_data[:],
        sensor.measured_data[:],
    ]
    sensor._save_data = sensor._save_data_multiple
    sensor.measure(
        t,
        u=u,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data) == 2
    assert len(sensor.measured_data[0]) == 3
    assert len(sensor.measured_data[1]) == 2
    sensor.measure(
        t,
        u=u,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(sensor.measured_data[0]) == 3
    assert len(sensor.measured_data[1]) == 3


def test_noisy_rotated_accelerometer(noisy_rotated_accelerometer, example_plain_env):
    """Test the measure method of the Accelerometer class. Checks if saved
    measurement is (ax,ay,az) and if measured_data is [(t, (ax,ay,az)), ...]
    """

    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0.4, 0.4, 1])
    inertial_acceleration = Vector(U_DOT[3:6]) + Vector([0, 0, -GRAVITY])
    omega = Vector(U[10:13])
    omega_dot = Vector(U_DOT[10:13])
    acceleration = (
        inertial_acceleration
        + Vector.cross(omega_dot, relative_position)
        + Vector.cross(omega, Vector.cross(omega, relative_position))
    )

    # calculate total rotation matrix
    cross_axis_sensitivity = Matrix(
        [
            [1, 0.005, 0.005],
            [0.005, 1, 0.005],
            [0.005, 0.005, 1],
        ]
    )
    sensor_rotation = Matrix.transformation(
        euler313_to_quaternions(*np.deg2rad([60, 60, 60]))
    )
    total_rotation = sensor_rotation @ cross_axis_sensitivity
    rocket_rotation = Matrix.transformation(U[6:10]).transpose
    # expected measurement without noise
    ax, ay, az = total_rotation @ (rocket_rotation @ acceleration)
    # expected measurement with constant bias
    ax += 0.5
    ay += 0.5
    az += 0.5

    # check last measurement considering noise error bounds
    noisy_rotated_accelerometer.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=relative_position,
        environment=example_plain_env,
    )
    assert noisy_rotated_accelerometer.measurement == approx([ax, ay, az], rel=0.1)
    assert len(noisy_rotated_accelerometer.measurement) == 3
    assert noisy_rotated_accelerometer.measured_data[0][1:] == approx(
        [ax, ay, az], rel=0.1
    )
    assert noisy_rotated_accelerometer.measured_data[0][0] == TIME


def test_noisy_rotated_gyroscope(noisy_rotated_gyroscope, example_plain_env):
    """Test the measure method of the Gyroscope class. Checks if saved
    measurement is (wx,wy,wz) and if measured_data is [(t, (wx,wy,wz)), ...]
    """
    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0.4, 0.4, 1])
    omega = Vector(U[10:13])
    # calculate total rotation matrix
    cross_axis_sensitivity = Matrix(
        [
            [1, 0.005, 0.005],
            [0.005, 1, 0.005],
            [0.005, 0.005, 1],
        ]
    )
    sensor_rotation = Matrix.transformation(
        euler313_to_quaternions(*np.deg2rad([-60, -60, -60]))
    )
    total_rotation = sensor_rotation @ cross_axis_sensitivity
    # expected measurement without noise
    wx, wy, wz = total_rotation @ omega
    # expected measurement with constant bias
    wx += 0.5
    wy += 0.5
    wz += 0.5

    # check last measurement considering noise error bounds
    noisy_rotated_gyroscope.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=relative_position,
        environment=example_plain_env,
    )
    assert noisy_rotated_gyroscope.measurement == approx([wx, wy, wz], rel=0.3)
    assert len(noisy_rotated_gyroscope.measurement) == 3
    assert noisy_rotated_gyroscope.measured_data[0][1:] == approx([wx, wy, wz], rel=0.3)
    assert noisy_rotated_gyroscope.measured_data[0][0] == TIME


def test_noisy_barometer(noisy_barometer, example_plain_env):
    """Test the measure method of the Barometer class. Checks if saved
    measurement is (P) and if measured_data is [(t, P), ...]
    """
    # expected measurement without noise
    relative_position = Vector([0.4, 0.4, 1])
    relative_altitude = (Matrix.transformation(U[6:10]) @ relative_position).z
    P = example_plain_env.pressure(relative_altitude + U[2])
    # expected measurement with constant bias
    P += 0.5

    noisy_barometer.measure(
        time=TIME,
        u=U,
        relative_position=relative_position,
        environment=example_plain_env,
    )
    assert noisy_barometer.measurement == approx(P, rel=0.03)
    assert noisy_barometer.measured_data[0][1] == approx(P, rel=0.03)
    assert noisy_barometer.measured_data[0][0] == TIME


def test_noisy_gnss(noisy_gnss, example_plain_env):
    """Test the measure method of the GnssReceiver class. Checks if saved
    measurement is (latitude, longitude, altitude) and if measured_data is [(t, (latitude, longitude, altitude)), ...]
    """
    # expected measurement without noise
    relative_position = Vector([0.4, 0.4, 1])
    lat, lon = example_plain_env.latitude, example_plain_env.longitude
    earth_radius = example_plain_env.earth_radius
    x, y, z = (Matrix.transformation(U[6:10]) @ relative_position) + Vector(U[0:3])
    drift = (x**2 + y**2) ** 0.5
    bearing = (2 * np.pi - np.arctan2(-x, y)) * (180 / np.pi)
    latitude = np.degrees(
        np.arcsin(
            np.sin(np.radians(lat)) * np.cos(drift / earth_radius)
            + np.cos(np.radians(lat))
            * np.sin(drift / earth_radius)
            * np.cos(np.radians(bearing))
        )
    )
    longitude = np.degrees(
        np.radians(lon)
        + np.arctan2(
            np.sin(np.radians(bearing))
            * np.sin(drift / earth_radius)
            * np.cos(np.radians(lat)),
            np.cos(drift / earth_radius)
            - np.sin(np.radians(lat)) * np.sin(np.radians(latitude)),
        )
    )
    altitude = z

    noisy_gnss.measure(
        time=TIME,
        u=U,
        relative_position=relative_position,
        environment=example_plain_env,
    )
    assert noisy_gnss.measurement == approx([latitude, longitude, altitude], abs=3.2)
    assert len(noisy_gnss.measurement) == 3
    assert noisy_gnss.measured_data[0][1:] == approx(
        [latitude, longitude, altitude], abs=3.2
    )
    assert noisy_gnss.measured_data[0][0] == TIME

    # check last measurement considering noise error bounds
    noisy_gnss.measure(
        time=TIME,
        u=U,
        relative_position=relative_position,
        environment=example_plain_env,
    )
    assert noisy_gnss.measurement == approx([latitude, longitude, altitude], abs=3.2)
    assert len(noisy_gnss.measurement) == 3
    assert noisy_gnss.measured_data[1][1:] == approx(
        [latitude, longitude, altitude], abs=3.2
    )
    assert noisy_gnss.measured_data[1][0] == TIME


@pytest.mark.parametrize(
    "sensor, file_format, expected_header",
    [
        ("ideal_accelerometer", "csv", "t,ax,ay,az\n"),
        ("ideal_gyroscope", "csv", "t,wx,wy,wz\n"),
        ("ideal_barometer", "csv", "t,pressure\n"),
        ("ideal_gnss", "csv", "t,latitude,longitude,altitude\n"),
    ],
)
def test_export_data_csv(
    sensor, file_format, expected_header, request, example_plain_env
):
    """Test the export_data method for CSV format."""
    sensor = request.getfixturevalue(sensor)

    # Perform measurement
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    file_name = f"sensors.{file_format}"

    # Export data
    sensor.export_measured_data(file_name, file_format=file_format)

    # Check CSV contents
    with open(file_name, "r") as file:
        contents = file.read()

    expected_data = expected_header
    for data in sensor.measured_data:
        expected_data += ",".join(map(str, data)) + "\n"

    assert contents == expected_data

    os.remove(file_name)


@pytest.mark.parametrize(
    "sensor, file_format, expected_keys",
    [
        ("ideal_accelerometer", "json", ("ax", "ay", "az")),
        ("ideal_gyroscope", "json", ("wx", "wy", "wz")),
        ("ideal_barometer", "json", ("pressure",)),
        ("ideal_gnss", "json", ("latitude", "longitude", "altitude")),
    ],
)
def test_export_data_json(
    sensor, file_format, expected_keys, request, example_plain_env
):
    """Test the export_data method for JSON format."""
    sensor = request.getfixturevalue(sensor)

    # Perform measurement
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    file_name = f"sensors.{file_format}"

    # Export data
    sensor.export_measured_data(file_name, file_format=file_format)

    # Check JSON contents
    with open(file_name, "r") as file:
        contents = json.load(file)

    expected_data = {"t": []}
    for key in expected_keys:
        expected_data[key] = []

    for data in sensor.measured_data:
        expected_data["t"].append(data[0])
        for i, key in enumerate(expected_keys):
            expected_data[key].append(data[i + 1])

    assert contents == expected_data

    os.remove(file_name)


@pytest.mark.parametrize(
    "sensor, file_format, expected_header",
    [
        ("ideal_accelerometer", "csv", "t,ax,ay,az\n"),
        ("ideal_gyroscope", "csv", "t,wx,wy,wz\n"),
    ],
)
def test_export_multiple_sensors_csv(
    sensor, file_format, expected_header, request, example_plain_env
):
    """Test exporting data for multiple instances in CSV format."""
    sensor = request.getfixturevalue(sensor)

    # Perform measurement
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    sensor.measured_data = [sensor.measured_data[:], sensor.measured_data[:]]
    file_name = f"sensors.{file_format}"

    # Export multiple data
    sensor.export_measured_data(file_name, file_format=file_format)

    # Check CSV for both instances
    with open(f"{file_name}_1", "r") as file:
        contents_1 = file.read()

    with open(f"{file_name}_2", "r") as file:
        contents_2 = file.read()

    expected_data = expected_header
    for data in sensor.measured_data[0]:
        expected_data += ",".join(map(str, data)) + "\n"

    assert contents_1 == expected_data
    assert contents_2 == expected_data

    os.remove(f"{file_name}_1")
    os.remove(f"{file_name}_2")


@pytest.mark.parametrize(
    "sensor, file_format, expected_keys",
    [
        ("ideal_accelerometer", "json", ("ax", "ay", "az")),
        ("ideal_gyroscope", "json", ("wx", "wy", "wz")),
    ],
)
def test_export_multiple_sensors_json(
    sensor, file_format, expected_keys, request, example_plain_env
):
    """Test exporting data for multiple instances in JSON format."""
    sensor = request.getfixturevalue(sensor)

    # Perform measurement
    sensor.measure(
        time=TIME,
        u=U,
        u_dot=U_DOT,
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    sensor.measured_data = [sensor.measured_data[:], sensor.measured_data[:]]
    file_name = f"sensors.{file_format}"

    # Export multiple data
    sensor.export_measured_data(file_name, file_format=file_format)

    # Check JSON for both instances
    with open(f"{file_name}_1", "r") as file:
        contents_1 = json.load(file)

    with open(f"{file_name}_2", "r") as file:
        contents_2 = json.load(file)

    expected_data = {"t": []}
    for key in expected_keys:
        expected_data[key] = []

    for data in sensor.measured_data[0]:
        expected_data["t"].append(data[0])
        for i, key in enumerate(expected_keys):
            expected_data[key].append(data[i + 1])

    assert contents_1 == expected_data
    assert contents_2 == expected_data

    os.remove(f"{file_name}_1")
    os.remove(f"{file_name}_2")
