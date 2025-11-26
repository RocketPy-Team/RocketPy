import warnings
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import Function, NoseCone, Rocket, SolidMotor
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.motors.empty_motor import EmptyMotor
from rocketpy.motors.motor import Motor


@patch("matplotlib.pyplot.show")
def test_elliptical_fins(mock_show, calisto_robust, calisto_trapezoidal_fins):  # pylint: disable=unused-argument
    test_rocket = calisto_robust
    calisto_robust.aerodynamic_surfaces.remove(calisto_trapezoidal_fins)
    test_rocket.add_elliptical_fins(4, span=0.100, root_chord=0.120, position=-1.168)
    static_margin = test_rocket.static_margin(0)
    assert test_rocket.all_info() is None or not abs(static_margin - 2.30) < 0.01


def test_evaluate_static_margin_assert_cp_equals_cm(dimensionless_calisto):
    rocket = dimensionless_calisto
    rocket.evaluate_center_of_pressure()
    rocket.evaluate_static_margin()

    burn_time = rocket.motor.burn_time

    assert pytest.approx(
        rocket.center_of_mass(0) / (2 * rocket.radius), 1e-8
    ) == pytest.approx(rocket.static_margin(0), 1e-8)
    assert pytest.approx(
        rocket.center_of_mass(burn_time[1]) / (2 * rocket.radius), 1e-8
    ) == pytest.approx(rocket.static_margin(burn_time[1]), 1e-8)
    assert pytest.approx(rocket.total_lift_coeff_der(0), 1e-8) == pytest.approx(0, 1e-8)
    assert pytest.approx(rocket.cp_position(0), 1e-8) == pytest.approx(0, 1e-8)


@pytest.mark.parametrize(
    "k, type_",
    ([2 / 3, "conical"], [0.46469957130675876, "ogive"], [0.563, "lvhaack"]),
)
def test_add_nose_assert_cp_cm_plus_nose(k, type_, calisto, dimensionless_calisto, m):
    calisto.add_nose(length=0.55829, kind=type_, position=1.160)
    cpz = (1.160) - k * 0.55829  # Relative to the center of dry mass
    clalpha = 2

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)

    assert clalpha == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert calisto.cp_position(0) == pytest.approx(cpz, 1e-8)

    dimensionless_calisto.add_nose(length=0.55829 * m, kind=type_, position=(1.160) * m)
    assert pytest.approx(dimensionless_calisto.static_margin(0), 1e-8) == pytest.approx(
        calisto.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_calisto.static_margin(np.inf), 1e-8
    ) == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.total_lift_coeff_der(0), 1e-8
    ) == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.cp_position(0) / m, 1e-8
    ) == pytest.approx(calisto.cp_position(0), 1e-8)


def test_add_tail_assert_cp_cm_plus_tail(calisto, dimensionless_calisto, m):
    calisto.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.313,
    )

    clalpha = -2 * (1 - (0.0635 / 0.0435) ** (-2)) * (0.0635 / (calisto.radius)) ** 2
    cpz = (-1.313) - (0.06 / 3) * (
        1 + (1 - (0.0635 / 0.0435)) / (1 - (0.0635 / 0.0435) ** 2)
    )

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert np.abs(clalpha) == pytest.approx(
        np.abs(calisto.total_lift_coeff_der(0)), 1e-8
    )
    assert calisto.cp_position(0) == cpz

    dimensionless_calisto.add_tail(
        top_radius=0.0635 * m,
        bottom_radius=0.0435 * m,
        length=0.060 * m,
        position=(-1.313) * m,
    )
    assert pytest.approx(dimensionless_calisto.static_margin(0), 1e-8) == pytest.approx(
        calisto.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_calisto.static_margin(np.inf), 1e-8
    ) == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.total_lift_coeff_der(0), 1e-8
    ) == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.cp_position(0) / m, 1e-8
    ) == pytest.approx(calisto.cp_position(0), 1e-8)


@pytest.mark.parametrize(
    "sweep_angle, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [(39.8, 2.51, 3.16, 1.50), (-10, 2.47, 3.21, 1.49), (29.1, 2.50, 3.28, 1.52)],
)
def test_add_trapezoidal_fins_sweep_angle(
    calisto,
    sweep_angle,
    expected_fin_cpz,
    expected_clalpha,
    expected_cpz_cm,
    calisto_nose_cone,
):
    # Reference values from OpenRocket
    calisto.add_surfaces(calisto_nose_cone, Vector([0, 0, 1.160]))
    fin_set = calisto.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_angle=sweep_angle,
        position=-1.064,
    )

    # Check center of pressure
    translate = 1.160
    cpz = -1.300 - fin_set.cpz
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = fin_set.cl(1, 0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - calisto.cp_position(0) == pytest.approx(expected_cpz_cm, 0.01)


@pytest.mark.parametrize(
    "sweep_length, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [
        (0.075, 2.28, 3.16, 1.502),
        (-0.0159, 2.24, 3.21, 1.485),
        (0.05, 2.27, 3.28, 1.513),
    ],
)
def test_add_trapezoidal_fins_sweep_length(
    calisto,
    sweep_length,
    expected_fin_cpz,
    expected_clalpha,
    expected_cpz_cm,
    calisto_nose_cone,
):
    # Reference values from OpenRocket
    calisto.add_surfaces(calisto_nose_cone, Vector([0, 0, 1.160]))
    fin_set = calisto.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_length=sweep_length,
        position=-1.064,
    )

    # Check center of pressure
    translate = 1.160
    cpz = -fin_set.cp[2] - 1.064
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = fin_set.cl(1, 0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - calisto.cp_position(0) == pytest.approx(expected_cpz_cm, 0.01)

    assert isinstance(calisto.aerodynamic_surfaces[0].component, NoseCone)


def test_add_fins_assert_cp_cm_plus_fins(calisto, dimensionless_calisto, m):
    calisto.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
    )

    cpz = (-1.168) - (
        ((0.120 - 0.040) / 3) * ((0.120 + 2 * 0.040) / (0.120 + 0.040))
        + (1 / 6) * (0.120 + 0.040 - 0.120 * 0.040 / (0.120 + 0.040))
    )

    clalpha = (4 * 4 * (0.1 / (2 * calisto.radius)) ** 2) / (
        1
        + np.sqrt(
            1
            + (2 * np.sqrt((0.12 / 2 - 0.04 / 2) ** 2 + 0.1**2) / (0.120 + 0.040)) ** 2
        )
    )
    clalpha *= 1 + calisto.radius / (0.1 + calisto.radius)

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)

    assert np.abs(clalpha) == pytest.approx(
        np.abs(calisto.total_lift_coeff_der(0)), 1e-8
    )
    assert calisto.cp_position(0) == pytest.approx(cpz, 1e-8)

    dimensionless_calisto.add_trapezoidal_fins(
        4,
        span=0.100 * m,
        root_chord=0.120 * m,
        tip_chord=0.040 * m,
        position=(-1.168) * m,
    )
    assert pytest.approx(dimensionless_calisto.static_margin(0), 1e-8) == pytest.approx(
        calisto.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_calisto.static_margin(np.inf), 1e-8
    ) == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.total_lift_coeff_der(0), 1e-8
    ) == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.cp_position(0) / m, 1e-8
    ) == pytest.approx(calisto.cp_position(0), 1e-8)


@pytest.mark.parametrize(
    """cdm_position, grain_cm_position, nozzle_position, coord_direction,
    motor_position, expected_motor_cdm, expected_motor_cpp""",
    [
        (0.317, 0.397, 0, "nozzle_to_combustion_chamber", -1.373, -1.056, -0.976),
        (0, 0.08, -0.317, "nozzle_to_combustion_chamber", -1, -1, -0.92),
        (-0.317, -0.397, 0, "combustion_chamber_to_nozzle", -1.373, -1.056, -0.976),
        (0, -0.08, 0.317, "combustion_chamber_to_nozzle", -1, -1, -0.92),
        (1.317, 1.397, 1, "nozzle_to_combustion_chamber", -2.373, -1.056, -0.976),
    ],
)
def test_add_motor_coordinates(
    calisto_motorless,
    cdm_position,
    grain_cm_position,
    nozzle_position,
    coord_direction,
    motor_position,
    expected_motor_cdm,
    expected_motor_cpp,
):
    """Test the method add_motor and related position properties in a Rocket
    instance.

    This test checks the correctness of the `add_motor` method and the computed
    `motor_center_of_dry_mass_position` and `center_of_propellant_position`
    properties in the `Rocket` class using various parameters related to the
    motor's position, nozzle's position, and other related coordinates.
    Different scenarios are tested using parameterization, checking scenarios
    moving from the nozzle to the combustion chamber and vice versa, and with
    various specific physical and geometrical characteristics of the motor.

    Parameters
    ----------
    calisto_motorless : Rocket instance
        A predefined instance of a Rocket without a motor, used as a base for testing.
    cdm_position : float
        Position of the center of dry mass of the motor.
    grain_cm_position : float
        Position of the grains' center of mass.
    nozzle_position : float
        Position of the nozzle.
    coord_direction : str
        Direction for coordinate system orientation;
        it can be "nozzle_to_combustion_chamber" or "combustion_chamber_to_nozzle".
    motor_position : float
        Position where the motor should be added to the rocket.
    expected_motor_cdm : float
        Expected position of the motor's center of dry mass after addition.
    expected_motor_cpp : float
        Expected position of the center of propellant after addition.
    """
    example_motor = SolidMotor(
        thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=0,
        dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=cdm_position,
        nozzle_position=nozzle_position,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=grain_cm_position,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation=coord_direction,
    )
    calisto = calisto_motorless
    calisto.add_motor(example_motor, position=motor_position)

    calculated_motor_cdm = calisto.motor_center_of_dry_mass_position
    calculated_motor_cpp = calisto.center_of_propellant_position

    assert pytest.approx(expected_motor_cdm) == calculated_motor_cdm
    assert pytest.approx(expected_motor_cpp) == calculated_motor_cpp(0)


def test_add_cm_eccentricity_assert_properties_set(calisto):
    calisto.add_cm_eccentricity(x=4, y=5)

    assert calisto.cp_eccentricity_x == -4
    assert calisto.cp_eccentricity_y == -5

    assert calisto.thrust_eccentricity_x == -4
    assert calisto.thrust_eccentricity_y == -5


def test_add_thrust_eccentricity_assert_properties_set(calisto):
    calisto.add_thrust_eccentricity(x=4, y=5)

    assert calisto.thrust_eccentricity_x == 4
    assert calisto.thrust_eccentricity_y == 5


def test_add_cp_eccentricity_assert_properties_set(calisto):
    calisto.add_cp_eccentricity(x=4, y=5)

    assert calisto.cp_eccentricity_x == 4
    assert calisto.cp_eccentricity_y == 5


def test_add_motor(calisto_motorless, cesaroni_m1670):
    """Tests the add_motor method of the Rocket class.
    Both with respect to return instances and expected behaviour.
    Parameters
    ----------
    calisto_motorless : Rocket instance
        A predefined instance of a Rocket without a motor, used as a base for testing.
    cesaroni_m1670 : rocketpy.SolidMotor
        Cesaroni M1670 motor
    """

    assert isinstance(calisto_motorless.motor, EmptyMotor)
    center_of_mass_motorless = calisto_motorless.center_of_mass
    calisto_motorless.add_motor(cesaroni_m1670, 0)

    assert isinstance(calisto_motorless.motor, Motor)
    center_of_mass_with_motor = calisto_motorless.center_of_mass

    assert center_of_mass_motorless is not center_of_mass_with_motor


def test_check_missing_all_components(calisto_motorless):
    """Tests the _check_missing_components method for a Rocket with no components."""
    with pytest.warns(UserWarning) as record:
        calisto_motorless._check_missing_components()

    assert len(record) == 1
    msg = str(record[0].message)
    assert "motor" in msg
    assert "aerodynamic surfaces" in msg


def test_check_missing_some_components(calisto):
    """Tests the _check_missing_components method for a Rocket missing some components."""
    calisto.aerodynamic_surfaces = []

    with pytest.warns(UserWarning) as record:
        calisto._check_missing_components()

    assert len(record) == 1
    msg = str(record[0].message)
    assert "aerodynamic surfaces" in msg


def test_check_missing_no_components_missing(calisto_robust):
    """Tests the _check_missing_components method for a complete Rocket."""
    # Catch all warnings that occur inside this 'with' block.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calisto_robust._check_missing_components()
        # For a complete rocket, this method should NOT issue any warnings.
    assert len(w) == 0


def test_set_rail_button(calisto):
    rail_buttons = calisto.set_rail_buttons(0.2, -0.5, 30)
    # assert buttons_distance
    assert (
        rail_buttons.buttons_distance
        == calisto.rail_buttons[0].component.buttons_distance
        == pytest.approx(0.7, 1e-12)
    )
    # assert buttons position on rocket
    assert calisto.rail_buttons[0].position.z == -0.5
    # assert angular position
    assert (
        rail_buttons.angular_position
        == calisto.rail_buttons[0].component.angular_position
        == 30
    )
    # assert upper button position
    assert calisto.rail_buttons[0].component.buttons_distance + calisto.rail_buttons[
        0
    ].position.z == pytest.approx(0.2, 1e-12)


def test_evaluate_total_mass(calisto_motorless):
    """Tests the evaluate_total_mass method of the Rocket class.
    Both with respect to return instances and expected behaviour.

    Parameters
    ----------
    calisto_motorless : Rocket instance
        A predefined instance of a Rocket without a motor, used as a base for testing.
    """
    assert isinstance(calisto_motorless.evaluate_total_mass(), Function)


def test_evaluate_center_of_mass(calisto):
    """Tests the evaluate_center_of_mass method of the Rocket class.
    Both with respect to return instances and expected behaviour.
    Parameters
    ----------
    calisto : Rocket instance
        A predefined instance of the calisto Rocket with a motor, used as a base for testing.
    """
    assert isinstance(calisto.evaluate_center_of_mass(), Function)


def test_evaluate_nozzle_to_cdm(calisto):
    expected_distance = 1.255
    atol = 1e-3  # Equivalent to 1mm
    assert pytest.approx(expected_distance, atol) == calisto.nozzle_to_cdm
    # Test if calling the function returns the same result
    res = calisto.evaluate_nozzle_to_cdm()
    assert pytest.approx(expected_distance, atol) == res


def test_evaluate_nozzle_gyration_tensor(calisto):
    expected_gyration_tensor = np.array(
        [[0.3940207, 0, 0], [0, 0.3940207, 0], [0, 0, 0.0005445]]
    )
    atol = 1e-3 * 1e-2 * 1e-2  # Equivalent to 1g * 1cm^2
    assert np.allclose(
        expected_gyration_tensor, np.array(calisto.nozzle_gyration_tensor), atol=atol
    )
    # Test if calling the function returns the same result
    res = calisto.evaluate_nozzle_gyration_tensor()
    assert np.allclose(expected_gyration_tensor, np.array(res), atol=atol)


def test_evaluate_com_to_cdm_function(calisto):
    atol = 1e-3  # Equivalent to 1mm
    assert np.allclose(
        (calisto.center_of_dry_mass_position - calisto.center_of_mass).source,
        calisto.com_to_cdm_function.source,
        atol=atol,
    )


def test_get_inertia_tensor_at_time(calisto):
    # Expected values (for t = 0)
    # TODO: compute these values by hand or using CAD.
    I_11 = 10.516647727227216
    I_22 = 10.516647727227216
    I_33 = 0.0379420341586346

    # Set tolerance threshold
    atol = 1e-5

    # Get inertia tensor at t = 0
    inertia_tensor = calisto.get_inertia_tensor_at_time(0)

    # Check if the values are close to the expected ones
    assert pytest.approx(I_11, atol) == inertia_tensor.x[0]
    assert pytest.approx(I_22, atol) == inertia_tensor.y[1]
    assert pytest.approx(I_33, atol) == inertia_tensor.z[2]
    # Check if products of inertia are zero
    assert pytest.approx(0, atol) == inertia_tensor.x[1]
    assert pytest.approx(0, atol) == inertia_tensor.x[2]
    assert pytest.approx(0, atol) == inertia_tensor.y[0]
    assert pytest.approx(0, atol) == inertia_tensor.y[2]
    assert pytest.approx(0, atol) == inertia_tensor.z[0]
    assert pytest.approx(0, atol) == inertia_tensor.z[1]


def test_get_inertia_tensor_derivative_at_time(calisto):
    # Expected values (for t = 2s)
    # TODO: compute these values by hand or using CAD.
    I_11_dot = -0.7164327431607691
    I_22_dot = -0.7164327431607691
    I_33_dot = -0.0006714936623050

    # Set tolerance threshold
    atol = 1e-3

    # Get inertia tensor at t = 2s
    inertia_tensor = calisto.get_inertia_tensor_derivative_at_time(2)

    # Check if the values are close to the expected ones
    assert pytest.approx(I_11_dot, atol) == inertia_tensor.x[0]
    assert pytest.approx(I_22_dot, atol) == inertia_tensor.y[1]
    assert pytest.approx(I_33_dot, atol) == inertia_tensor.z[2]
    # Check if products of inertia are zero
    assert pytest.approx(0, atol) == inertia_tensor.x[1]
    assert pytest.approx(0, atol) == inertia_tensor.x[2]
    assert pytest.approx(0, atol) == inertia_tensor.y[0]
    assert pytest.approx(0, atol) == inertia_tensor.y[2]
    assert pytest.approx(0, atol) == inertia_tensor.z[0]
    assert pytest.approx(0, atol) == inertia_tensor.z[1]


def test_add_thrust_eccentricity(calisto):
    """Test add_thrust_eccentricity method of the Rocket class."""
    calisto.add_thrust_eccentricity(0.1, 0.1)
    assert calisto.thrust_eccentricity_x == 0.1
    assert calisto.thrust_eccentricity_y == 0.1


def test_add_cm_eccentricity(calisto):
    """Test add_cm_eccentricity method of the Rocket class."""
    calisto.add_cm_eccentricity(-0.1, -0.1)
    assert calisto.cp_eccentricity_x == 0.1
    assert calisto.cp_eccentricity_y == 0.1
    assert calisto.thrust_eccentricity_x == 0.1
    assert calisto.thrust_eccentricity_y == 0.1


class TestAddSurfaces:
    """Test the add_surfaces method with different nose cone configurations.
    More specifically, this will check the static margin of the rocket with
    different nose cone configurations."""

    @pytest.fixture(autouse=True)
    def setup(self, calisto):
        self.calisto = calisto
        self.length = 0.55829
        self.kind = "vonkarman"
        self.position = 1.16
        self.bluffness = 0
        self.base_radius = 0.0635
        self.rocket_radius = 0.0635

    def test_add_surfaces_base_equals_rocket_radius(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=self.base_radius,
            bluffness=self.bluffness,
            rocket_radius=self.rocket_radius,
            name="Nose Cone 1",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(1, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)

    def test_add_surfaces_base_half_rocket_radius(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=self.base_radius / 2,
            bluffness=self.bluffness,
            rocket_radius=self.rocket_radius,
            name="Nose Cone 2",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(0.5, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)

    def test_add_surfaces_base_radius_none(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=None,
            bluffness=self.bluffness,
            rocket_radius=self.rocket_radius * 2,
            name="Nose Cone 3",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(1, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)

    def test_add_surfaces_rocket_radius_none(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=self.base_radius,
            bluffness=self.bluffness,
            rocket_radius=None,
            name="Nose Cone 4",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(1, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)


def test_coordinate_system_orientation(
    calisto_nose_cone, cesaroni_m1670, calisto_trapezoidal_fins
):
    """Test if the coordinate system orientation is working properly. This test
    basically checks if the static margin is the same for the same rocket with
    different coordinate system orientation.

    Parameters
    ----------
    calisto_nose_cone : rocketpy.NoseCone
        Nose cone of the rocket
    cesaroni_m1670 : rocketpy.SolidMotor
        Cesaroni M1670 motor
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        Trapezoidal fins of the rocket
    """
    motor_nozzle_to_combustion_chamber = cesaroni_m1670

    motor_combustion_chamber_to_nozzle = SolidMotor(
        thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=-0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=-0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="combustion_chamber_to_nozzle",
    )

    rocket_tail_to_nose = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket_tail_to_nose.add_motor(motor_nozzle_to_combustion_chamber, position=-1.373)

    rocket_tail_to_nose.add_surfaces(calisto_nose_cone, 1.160)
    rocket_tail_to_nose.add_surfaces(calisto_trapezoidal_fins, -1.168)

    static_margin_tail_to_nose = rocket_tail_to_nose.static_margin

    rocket_nose_to_tail = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )

    rocket_nose_to_tail.add_motor(motor_combustion_chamber_to_nozzle, position=1.373)

    rocket_nose_to_tail.add_surfaces(calisto_nose_cone, -1.160)
    rocket_nose_to_tail.add_surfaces(calisto_trapezoidal_fins, 1.168)

    static_margin_nose_to_tail = rocket_nose_to_tail.static_margin

    assert np.array_equal(static_margin_tail_to_nose, static_margin_nose_to_tail)
