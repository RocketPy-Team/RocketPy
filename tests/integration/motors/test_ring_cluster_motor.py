# pylint: disable=invalid-name
import numpy as np
import pytest

from rocketpy import Function, RingClusterMotor, SolidMotor


@pytest.fixture
def base_motor():
    """
    Creates a simplified SolidMotor for testing purposes.
    Properties:
    - Constant Thrust: 1000 N
    - Burn time: 5 s
    - Dry mass: 10 kg
    - Dry Inertia: (1.0, 1.0, 0.1)
    """
    thrust_curve = Function(lambda t: 1000 if t < 5 else 0, "Time (s)", "Thrust (N)")

    return SolidMotor(
        thrust_source=thrust_curve,
        burn_time=5,
        dry_mass=10.0,
        dry_inertia=(1.0, 1.0, 0.1),  # Ixx, Iyy, Izz
        grain_number=1,
        grain_density=1000,
        grain_outer_radius=0.05,
        grain_initial_inner_radius=0.02,
        grain_initial_height=0.5,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        nozzle_radius=0.02,
        grain_separation=0.001,
        grains_center_of_mass_position=0.25,
        center_of_dry_mass_position=0.25,
    )


def test_cluster_initialization(base_motor):
    """
    Tests if the RingClusterMotor initializes basic attributes correctly.
    """
    N = 3
    R = 0.5
    cluster = RingClusterMotor(motor=base_motor, number=N, radius=R)

    assert cluster.number == N
    assert cluster.radius == R
    assert cluster.grain_outer_radius == base_motor.grain_outer_radius


def test_cluster_mass_and_thrust_scaling(base_motor):
    """
    Tests if scalar and derived properties are correctly multiplied by N and that functional properties preserve their Function behavior
    """
    N = 4
    R = 0.2
    cluster = RingClusterMotor(motor=base_motor, number=N, radius=R)

    assert np.isclose(cluster.thrust(1), base_motor.thrust(1) * N)

    assert np.isclose(cluster.dry_mass, base_motor.dry_mass * N)

    assert np.isclose(cluster.propellant_mass(0), base_motor.propellant_mass(0) * N)  # pylint: disable=not-callable
    assert np.isclose(cluster.total_impulse, base_motor.total_impulse * N)
    assert np.isclose(cluster.average_thrust, base_motor.average_thrust * N)


def test_cluster_dry_inertia_steiner_theorem(base_motor):
    """
    Tests the implementation of the Parallel Axis Theorem (Huygens-Steiner)
    for the static (dry) mass of the cluster.

    Theoretical Formulas:
    I_zz_cluster = N * I_zz_local + N * m * R^2
    I_xx_cluster = N * I_xx_local + (N/2) * m * R^2  (Radial symmetry approximation)
    """
    N = 3
    R = 1.0
    cluster = RingClusterMotor(motor=base_motor, number=N, radius=R)

    m_dry = base_motor.dry_mass
    Ixx_loc = base_motor.dry_I_11
    Izz_loc = base_motor.dry_I_33

    expected_Izz = N * Izz_loc + N * m_dry * (R**2)

    expected_I_trans = N * Ixx_loc + (N / 2) * m_dry * (R**2)

    assert np.isclose(cluster.dry_I_33, expected_Izz)
    assert np.isclose(cluster.dry_I_11, expected_I_trans)
    assert np.isclose(cluster.dry_I_22, expected_I_trans)


def test_cluster_invalid_inputs(base_motor):
    """Tests if the validation raises errors for bad inputs."""
    with pytest.raises(ValueError):
        RingClusterMotor(motor=base_motor, number=1, radius=0.5)
    with pytest.raises(ValueError):
        RingClusterMotor(motor=base_motor, number=2, radius=-1.0)
    with pytest.raises(TypeError):
        RingClusterMotor(motor=base_motor, number="two", radius=0.5)


def test_cluster_methods_and_setters(base_motor):
    """Touches the display methods and setters to ensure coverage."""
    cluster = RingClusterMotor(motor=base_motor, number=2, radius=0.5)

    cluster.info()

    cluster.draw_cluster_layout(show=False)
    cluster.draw_cluster_layout(rocket_radius=0.1, show=False)

    cluster.propellant_mass = 50.0
    assert cluster.propellant_mass == 50.0
    cluster.propellant_I_11 = 2.0
    assert cluster.propellant_I_11 == 2.0


def test_cluster_propellant_inertia_dynamic(base_motor):
    """
    Tests if the Steiner theorem is correctly applied dynamically
    via exact geometric summation, especially for N=2.
    """
    N = 2
    R = 0.5
    cluster = RingClusterMotor(motor=base_motor, number=N, radius=R)

    t = 0

    m_prop = base_motor.propellant_mass(t)
    Ixx_prop_loc = base_motor.propellant_I_11(t)
    Izz_prop_loc = base_motor.propellant_I_33(t)

    expected_ixx = (Ixx_prop_loc * N) + 0

    expected_izz = (Izz_prop_loc * N) + (m_prop * N * R**2)

    assert np.isclose(cluster.propellant_I_11(t), expected_ixx)  # pylint: disable=not-callable
    assert np.isclose(cluster.propellant_I_33(t), expected_izz)
