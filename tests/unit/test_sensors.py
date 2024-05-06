import json
import os

import numpy as np
from pytest import approx
import pytest

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.tools import euler_to_quaternions

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
    """Test the rotation_matrix property of the Accelerometer class. Checks if
    the rotation matrix is correctly calculated.
    """
    # values from external source
    expected_matrix = np.array(
        [
            [0.2500000, -0.0580127, 0.9665064],
            [0.4330127, 0.8995190, -0.0580127],
            [-0.8660254, 0.4330127, 0.2500000],
        ]
    )
    rotation_matrix = np.array(noisy_rotated_accelerometer.rotation_matrix.components)
    assert np.allclose(expected_matrix, rotation_matrix, atol=1e-8)


def test_quantization(quantized_accelerometer):
    """Test the quantize method of the Sensor class. Checks if returned values
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


@pytest.mark.parametrize(
    "sensor",
    [
        "ideal_accelerometer",
        "ideal_gyroscope",
    ],
)
def test_measured_data(sensor, request):
    """Test the measured_data property of the Sensors class. Checks if
    the measured data is treated properly when the sensor is added once or more
    than once to the rocket.
    """
    sensor = request.getfixturevalue(sensor)

    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)
    assert len(sensor.measured_data) == 1
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)
    assert len(sensor.measured_data) == 2
    assert all(isinstance(i, tuple) for i in sensor.measured_data)

    # check case when sensor is added more than once to the rocket
    sensor.measured_data = [
        sensor.measured_data[:],
        sensor.measured_data[:],
    ]
    sensor._save_data = sensor._save_data_multiple
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)
    assert len(sensor.measured_data) == 2
    assert len(sensor.measured_data[0]) == 3
    assert len(sensor.measured_data[1]) == 2
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)
    assert len(sensor.measured_data[0]) == 3
    assert len(sensor.measured_data[1]) == 3


def test_noisy_rotated_accelerometer(noisy_rotated_accelerometer):
    """Test the measure method of the Accelerometer class. Checks if saved
    measurement is (ax,ay,az) and if measured_data is [(t, (ax,ay,az)), ...]
    """

    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0.4, 0.4, 1])
    a_I = Vector(U_DOT[3:6]) + Vector([0, 0, -GRAVITY])
    omega = Vector(U[10:13])
    omega_dot = Vector(U_DOT[10:13])
    accel = (
        a_I
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
    sensor_rotation = Matrix.transformation(euler_to_quaternions(60, 60, 60))
    total_rotation = sensor_rotation @ cross_axis_sensitivity
    rocket_rotation = Matrix.transformation(U[6:10])
    # expected measurement without noise
    ax, ay, az = total_rotation @ (rocket_rotation @ accel)
    # expected measurement with constant bias
    ax += 0.5
    ay += 0.5
    az += 0.5

    # check last measurement considering noise error bounds
    noisy_rotated_accelerometer.measure(TIME, U, U_DOT, relative_position, GRAVITY)
    assert noisy_rotated_accelerometer.measurement == approx([ax, ay, az], rel=0.1)
    assert len(noisy_rotated_accelerometer.measurement) == 3
    assert noisy_rotated_accelerometer.measured_data[0][1:] == approx(
        [ax, ay, az], rel=0.1
    )
    assert noisy_rotated_accelerometer.measured_data[0][0] == TIME


def test_noisy_rotated_gyroscope(noisy_rotated_gyroscope):
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
    sensor_rotation = Matrix.transformation(euler_to_quaternions(-60, -60, -60))
    total_rotation = sensor_rotation @ cross_axis_sensitivity
    rocket_rotation = Matrix.transformation(U[6:10])
    # expected measurement without noise
    wx, wy, wz = total_rotation @ (rocket_rotation @ omega)
    # expected measurement with constant bias
    wx += 0.5
    wy += 0.5
    wz += 0.5

    # check last measurement considering noise error bounds
    noisy_rotated_gyroscope.measure(TIME, U, U_DOT, relative_position, GRAVITY)
    assert noisy_rotated_gyroscope.measurement == approx([wx, wy, wz], rel=0.2)
    assert len(noisy_rotated_gyroscope.measurement) == 3
    assert noisy_rotated_gyroscope.measured_data[0][1:] == approx([wx, wy, wz], rel=0.2)
    assert noisy_rotated_gyroscope.measured_data[0][0] == TIME


@pytest.mark.parametrize(
    "sensor, expected_string",
    [
        ("ideal_accelerometer", "t,ax,ay,az\n"),
        ("ideal_gyroscope", "t,wx,wy,wz\n"),
    ],
)
def test_export_data_csv(sensor, expected_string, request):
    """Test the export_data method of accelerometer. Checks if the data is
    exported correctly.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal accelerometer and a gyroscope.
    """
    sensor = request.getfixturevalue(sensor)
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)

    file_name = "sensors.csv"

    sensor.export_measured_data(file_name, format="csv")

    with open(file_name, "r") as file:
        contents = file.read()

    expected_data = expected_string
    for t, x, y, z in sensor.measured_data:
        expected_data += f"{t},{x},{y},{z}\n"

    assert contents == expected_data

    # check exports for accelerometers added more than once to the rocket
    sensor.measured_data = [
        sensor.measured_data[:],
        sensor.measured_data[:],
    ]
    sensor.export_measured_data(file_name, format="csv")
    with open(file_name + "_1", "r") as file:
        contents = file.read()
    assert contents == expected_data

    with open(file_name + "_2", "r") as file:
        contents = file.read()
    assert contents == expected_data

    os.remove(file_name)
    os.remove(file_name + "_1")
    os.remove(file_name + "_2")


@pytest.mark.parametrize(
    "sensor, expected_string",
    [
        ("ideal_accelerometer", ("ax", "ay", "az")),
        ("ideal_gyroscope", ("wx", "wy", "wz")),
    ],
)
def test_export_data_json(sensor, expected_string, request):
    """Test the export_data method of the accelerometer. Checks if the data is
    exported correctly.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal
        accelerometer and a gyroscope.
    """
    sensor = request.getfixturevalue(sensor)
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)
    sensor.measure(TIME, U, U_DOT, Vector([0, 0, 0]), GRAVITY)

    file_name = "sensors.json"

    sensor.export_measured_data(file_name, format="json")

    contents = json.load(open(file_name, "r"))

    expected_data = {
        "t": [],
        expected_string[0]: [],
        expected_string[1]: [],
        expected_string[2]: [],
    }
    for t, x, y, z in sensor.measured_data:
        expected_data["t"].append(t)
        expected_data[expected_string[0]].append(x)
        expected_data[expected_string[1]].append(y)
        expected_data[expected_string[2]].append(z)

    assert contents == expected_data

    # check exports for accelerometers added more than once to the rocket
    sensor.measured_data = [
        sensor.measured_data[:],
        sensor.measured_data[:],
    ]
    sensor.export_measured_data(file_name, format="json")
    contents = json.load(open(file_name + "_1", "r"))
    assert contents == expected_data

    contents = json.load(open(file_name + "_2", "r"))
    assert contents == expected_data

    os.remove(file_name)
    os.remove(file_name + "_1")
    os.remove(file_name + "_2")
