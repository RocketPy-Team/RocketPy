import json
import os

import numpy as np
from pytest import approx

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.tools import euler_to_quaternions

# calisto standard simulation no wind solution index 200
SOLUTION = [
    3.338513236767685,
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
UDOT = [
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


def test_accelerometer_prints(noisy_rotated_accelerometer, quantized_accelerometer):
    """Test the print methods of the Accelerometer class. Checks if all
    attributes are printed correctly.
    """
    noisy_rotated_accelerometer.prints.all()
    quantized_accelerometer.prints.all()
    assert True


def test_gyroscope_prints(noisy_rotated_gyroscope, quantized_gyroscope):
    """Test the print methods of the Gyroscope class. Checks if all
    attributes are printed correctly.
    """
    noisy_rotated_gyroscope.prints.all()
    quantized_gyroscope.prints.all()
    assert True


def test_rotation_matrix(noisy_rotated_accelerometer):
    """Test the rotation_matrix property of the Accelerometer class. Checks if
    the rotation matrix is correctly calculated.
    """
    expected_matrix = np.array(
        [
            [0.2500000, -0.0580127, 0.9665064],
            [0.4330127, 0.8995190, -0.0580127],
            [-0.8660254, 0.4330127, 0.2500000],
        ]
    )
    rotation_matrix = np.array(noisy_rotated_accelerometer.rotation_matrix.components)
    assert np.allclose(expected_matrix, rotation_matrix, atol=1e-8)


def test_ideal_accelerometer_measure(ideal_accelerometer):
    """Test the measure method of the Accelerometer class. Checks if saved
    measurement is (ax,ay,az) and if measured_data is [(t, (ax,ay,az)), ...]
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]

    relative_position = Vector([0, 0, 0])
    gravity = 9.81
    a_I = Vector(UDOT[3:6])
    omega = Vector(u[10:13])
    omega_dot = Vector(UDOT[10:13])
    accel = (
        a_I
        + Vector.cross(omega_dot, relative_position)
        + Vector.cross(omega, Vector.cross(omega, relative_position))
    )
    ax, ay, az = Matrix.transformation(u[6:10]) @ accel
    ideal_accelerometer.measure(t, u, UDOT, relative_position, gravity)

    # check last measurement
    assert len(ideal_accelerometer.measurement) == 3
    assert all(isinstance(i, float) for i in ideal_accelerometer.measurement)
    assert ideal_accelerometer.measurement == approx([ax, ay, az], abs=1e-10)

    # check measured values
    assert len(ideal_accelerometer.measured_data) == 1
    ideal_accelerometer.measure(t, u, UDOT, relative_position, gravity)
    assert len(ideal_accelerometer.measured_data) == 2

    assert all(isinstance(i, tuple) for i in ideal_accelerometer.measured_data)
    assert ideal_accelerometer.measured_data[0][0] == t
    assert ideal_accelerometer.measured_data[0][1:] == approx([ax, ay, az], abs=1e-10)


def test_ideal_gyroscope_measure(ideal_gyroscope):
    """Test the measure method of the Gyroscope class. Checks if saved
    measurement is (wx,wy,wz) and if measured_data is [(t, (wx,wy,wz)), ...]
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    relative_position = Vector(
        [np.random.randint(-1, 1), np.random.randint(-1, 1), np.random.randint(-1, 1)]
    )

    rot = Matrix.transformation(u[6:10])
    ax, ay, az = rot @ Vector(u[10:13])

    ideal_gyroscope.measure(t, u, UDOT, relative_position)

    # check last measurement
    assert len(ideal_gyroscope.measurement) == 3
    assert all(isinstance(i, float) for i in ideal_gyroscope.measurement)
    assert ideal_gyroscope.measurement == approx([ax, ay, az], abs=1e-10)

    # check measured values
    assert len(ideal_gyroscope.measured_data) == 1
    ideal_gyroscope.measure(t, u, UDOT, relative_position)
    assert len(ideal_gyroscope.measured_data) == 2

    assert all(isinstance(i, tuple) for i in ideal_gyroscope.measured_data)
    assert ideal_gyroscope.measured_data[0][0] == t
    assert ideal_gyroscope.measured_data[0][1:] == approx([ax, ay, az], abs=1e-10)


def test_noisy_rotated_accelerometer(noisy_rotated_accelerometer):
    """Test the measure method of the Accelerometer class. Checks if saved
    measurement is (ax,ay,az) and if measured_data is [(t, (ax,ay,az)), ...]
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]

    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0.4, 0.4, 1])
    gravity = 9.81
    a_I = Vector(UDOT[3:6]) + Vector([0, 0, -gravity])
    omega = Vector(u[10:13])
    omega_dot = Vector(UDOT[10:13])
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
    rocket_rotation = Matrix.transformation(u[6:10])
    # expected measurement without noise
    ax, ay, az = total_rotation @ (rocket_rotation @ accel)
    # expected measurement with constant bias
    ax += 0.5
    ay += 0.5
    az += 0.5

    # check last measurement considering noise error bounds
    noisy_rotated_accelerometer.measure(t, u, UDOT, relative_position, gravity)
    assert noisy_rotated_accelerometer.measurement == approx([ax, ay, az], rel=0.5)


def test_noisy_rotated_gyroscope(noisy_rotated_gyroscope):
    """Test the measure method of the Gyroscope class. Checks if saved
    measurement is (wx,wy,wz) and if measured_data is [(t, (wx,wy,wz)), ...]
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0.4, 0.4, 1])
    gravity = 9.81
    omega = Vector(u[10:13])
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
    rocket_rotation = Matrix.transformation(u[6:10])
    # expected measurement without noise
    wx, wy, wz = total_rotation @ (rocket_rotation @ omega)
    # expected measurement with constant bias
    wx += 0.5
    wy += 0.5
    wz += 0.5

    # check last measurement considering noise error bounds
    noisy_rotated_gyroscope.measure(t, u, UDOT, relative_position, gravity)
    assert noisy_rotated_gyroscope.measurement == approx([wx, wy, wz], rel=0.5)


def test_quatization_accelerometer(quantized_accelerometer):
    """Test the measure method of the Accelerometer class. Checks if saved
    measurement is (ax,ay,az) and if measured_data is [(t, (ax,ay,az)), ...]
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0, 0, 0])
    gravity = 9.81
    a_I = Vector(UDOT[3:6])
    omega = Vector(u[10:13])
    omega_dot = Vector(UDOT[10:13])
    accel = (
        a_I
        + Vector.cross(omega_dot, relative_position)
        + Vector.cross(omega, Vector.cross(omega, relative_position))
    )

    # calculate total rotation matrix
    rocket_rotation = Matrix.transformation(u[6:10])
    # expected measurement without noise
    ax, ay, az = rocket_rotation @ accel
    # expected measurement with quantization
    az = 2  # saturated
    ax = round(ax / 0.4882) * 0.4882
    ay = round(ay / 0.4882) * 0.4882
    az = round(az / 0.4882) * 0.4882

    # check last measurement considering noise error bounds
    quantized_accelerometer.measure(t, u, UDOT, relative_position, gravity)
    assert quantized_accelerometer.measurement == approx([ax, ay, az], abs=1e-10)


def test_quatization_gyroscope(quantized_gyroscope):
    """Test the measure method of the Gyroscope class. Checks if saved
    measurement is (wx,wy,wz) and if measured_data is [(t, (wx,wy,wz)), ...]
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    # calculate acceleration at sensor position in inertial frame
    relative_position = Vector([0.4, 0.4, 1])
    gravity = 9.81
    omega = Vector(u[10:13])
    # calculate total rotation matrix
    rocket_rotation = Matrix.transformation(u[6:10])
    # expected measurement without noise
    wx, wy, wz = rocket_rotation @ omega
    # expected measurement with quantization
    wz = 15  # saturated
    wx = round(wx / 0.4882) * 0.4882
    wy = round(wy / 0.4882) * 0.4882
    wz = round(wz / 0.4882) * 0.4882

    # check last measurement considering noise error bounds
    quantized_gyroscope.measure(t, u, UDOT, relative_position, gravity)
    assert quantized_gyroscope.measurement == approx([wx, wy, wz], abs=1e-10)


def test_export_accel_data_csv(ideal_accelerometer):
    """Test the export_data method of accelerometer. Checks if the data is
    exported correctly.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal accelerometer and a gyroscope.
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    relative_position = Vector([0, 0, 0])
    gravity = 9.81
    ideal_accelerometer.measure(t, u, UDOT, relative_position, gravity)
    ideal_accelerometer.measure(t, u, UDOT, relative_position, gravity)

    file_name = "sensors.csv"

    ideal_accelerometer.export_measured_data(file_name, format="csv")

    with open(file_name, "r") as file:
        contents = file.read()

    expected_data = "t,ax,ay,az\n"
    for t, ax, ay, az in ideal_accelerometer.measured_data:
        expected_data += f"{t},{ax},{ay},{az}\n"

    assert contents == expected_data
    os.remove(file_name)


def test_export_accel_data_json(ideal_accelerometer):
    """Test the export_data method of the accelerometer. Checks if the data is
    exported correctly.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal
        accelerometer and a gyroscope.
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    relative_position = Vector([0, 0, 0])
    gravity = 9.81
    ideal_accelerometer.measure(t, u, UDOT, relative_position, gravity)
    ideal_accelerometer.measure(t, u, UDOT, relative_position, gravity)

    file_name = "sensors.json"

    ideal_accelerometer.export_measured_data(file_name, format="json")

    contents = json.load(open(file_name, "r"))

    expected_data = {"t": [], "ax": [], "ay": [], "az": []}
    for t, ax, ay, az in ideal_accelerometer.measured_data:
        expected_data["t"].append(t)
        expected_data["ax"].append(ax)
        expected_data["ay"].append(ay)
        expected_data["az"].append(az)

    assert contents == expected_data
    os.remove(file_name)


def test_export_gyro_data_csv(ideal_gyroscope):
    """Test the export_data method of the gyroscope. Checks if the data is
    exported correctly.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal
        accelerometer and a gyroscope.
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    relative_position = Vector([0, 0, 0])
    ideal_gyroscope.measure(t, u, UDOT, relative_position)
    ideal_gyroscope.measure(t, u, UDOT, relative_position)

    file_name = "sensors.csv"

    ideal_gyroscope.export_measured_data(file_name, format="csv")

    with open(file_name, "r") as file:
        contents = file.read()

    expected_data = "t,wx,wy,wz\n"
    for t, wx, wy, wz in ideal_gyroscope.measured_data:
        expected_data += f"{t},{wx},{wy},{wz}\n"

    assert contents == expected_data
    os.remove(file_name)


def test_export_gyro_data_json(ideal_gyroscope):
    """Test the export_data method of the gyroscope. Checks if the data is
    exported correctly.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal accelerometer and a gyroscope.
    """
    t = SOLUTION[0]
    u = SOLUTION[1:]
    relative_position = Vector([0, 0, 0])
    ideal_gyroscope.measure(t, u, UDOT, relative_position)
    ideal_gyroscope.measure(t, u, UDOT, relative_position)

    file_name = "sensors.json"

    ideal_gyroscope.export_measured_data(file_name, format="json")

    contents = json.load(open(file_name, "r"))

    expected_data = {"t": [], "wx": [], "wy": [], "wz": []}
    for t, wx, wy, wz in ideal_gyroscope.measured_data:
        expected_data["t"].append(t)
        expected_data["wx"].append(wx)
        expected_data["wy"].append(wy)
        expected_data["wz"].append(wz)

    assert contents == expected_data
    os.remove(file_name)
