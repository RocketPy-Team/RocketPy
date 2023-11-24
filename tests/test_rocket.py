from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import NoseCone, Rocket, SolidMotor, Function
from rocketpy.motors.motor import Motor, EmptyMotor


@patch("matplotlib.pyplot.show")
def test_rocket(mock_show, calisto_robust):
    test_rocket = calisto_robust
    static_margin = test_rocket.static_margin(0)
    # Check if all_info and static_method methods are working properly
    assert test_rocket.all_info() == None or not abs(static_margin - 2.05) < 0.01


@patch("matplotlib.pyplot.show")
def test_aero_surfaces_infos(
    mock_show, calisto_nose_cone, calisto_tail, calisto_trapezoidal_fins
):
    assert calisto_nose_cone.all_info() == None
    assert calisto_trapezoidal_fins.all_info() == None
    assert calisto_tail.all_info() == None
    assert calisto_trapezoidal_fins.draw() == None


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
        thrust_source="data/motors/Cesaroni_M1670.eng",
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
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket_tail_to_nose.add_motor(motor_nozzle_to_combustion_chamber, position=-1.373)

    rocket_tail_to_nose.aerodynamic_surfaces.add(calisto_nose_cone, 1.160)
    rocket_tail_to_nose.aerodynamic_surfaces.add(calisto_trapezoidal_fins, -1.168)

    static_margin_tail_to_nose = rocket_tail_to_nose.static_margin

    rocket_nose_to_tail = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )

    rocket_nose_to_tail.add_motor(motor_combustion_chamber_to_nozzle, position=1.373)

    rocket_nose_to_tail.aerodynamic_surfaces.add(calisto_nose_cone, -1.160)
    rocket_nose_to_tail.aerodynamic_surfaces.add(calisto_trapezoidal_fins, 1.168)

    static_margin_nose_to_tail = rocket_nose_to_tail.static_margin

    assert np.array_equal(static_margin_tail_to_nose, static_margin_nose_to_tail)


@patch("matplotlib.pyplot.show")
def test_elliptical_fins(mock_show, calisto_robust, calisto_trapezoidal_fins):
    test_rocket = calisto_robust
    calisto_robust.aerodynamic_surfaces.remove(calisto_trapezoidal_fins)
    fin_set = test_rocket.add_elliptical_fins(
        4, span=0.100, root_chord=0.120, position=-1.168
    )
    static_margin = test_rocket.static_margin(0)
    assert test_rocket.all_info() == None or not abs(static_margin - 2.30) < 0.01


@patch("matplotlib.pyplot.show")
def test_airfoil(
    mock_show,
    calisto,
    calisto_main_chute,
    calisto_drogue_chute,
    calisto_nose_cone,
    calisto_tail,
):
    test_rocket = calisto
    test_rocket.set_rail_buttons(0.082, -0.618)
    calisto.aerodynamic_surfaces.add(calisto_nose_cone, 1.160)
    calisto.aerodynamic_surfaces.add(calisto_tail, -1.313)

    fin_set_NACA = test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
        airfoil=("tests/fixtures/airfoils/NACA0012-radians.txt", "radians"),
        name="NACA0012",
    )
    fin_set_E473 = test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
        airfoil=("tests/fixtures/airfoils/e473-10e6-degrees.csv", "degrees"),
        name="E473",
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)

    static_margin = test_rocket.static_margin(0)

    assert test_rocket.all_info() == None or not abs(static_margin - 2.03) < 0.01


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
    "k, type",
    ([2 / 3, "conical"], [0.46469957130675876, "ogive"], [0.563, "lvhaack"]),
)
def test_add_nose_assert_cp_cm_plus_nose(k, type, calisto, dimensionless_calisto, m):
    calisto.add_nose(length=0.55829, kind=type, position=1.160)
    cpz = (1.160) - k * 0.55829  # Relative to the center of dry mass
    clalpha = 2

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)

    assert clalpha == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert calisto.cp_position(0) == pytest.approx(cpz, 1e-8)

    dimensionless_calisto.add_nose(length=0.55829 * m, kind=type, position=(1.160) * m)
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
    calisto.aerodynamic_surfaces.add(calisto_nose_cone, 1.160)
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
    calisto.aerodynamic_surfaces.add(calisto_nose_cone, 1.160)
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
            + (2 * np.sqrt((0.12 / 2 - 0.04 / 2) ** 2 + 0.1**2) / (0.120 + 0.040))
            ** 2
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
        thrust_source="data/motors/Cesaroni_M1670.eng",
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

    assert calisto.thrust_eccentricity_y == -4
    assert calisto.thrust_eccentricity_x == -5


def test_add_thrust_eccentricity_assert_properties_set(calisto):
    calisto.add_thrust_eccentricity(x=4, y=5)

    assert calisto.thrust_eccentricity_y == 4
    assert calisto.thrust_eccentricity_x == 5


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


def test_set_rail_button(calisto):
    rail_buttons = calisto.set_rail_buttons(0.2, -0.5, 30)
    # assert buttons_distance
    assert (
        rail_buttons.buttons_distance
        == calisto.rail_buttons[0].component.buttons_distance
        == pytest.approx(0.7, 1e-12)
    )
    # assert buttons position on rocket
    assert calisto.rail_buttons[0].position == -0.5
    # assert angular position
    assert (
        rail_buttons.angular_position
        == calisto.rail_buttons[0].component.angular_position
        == 30
    )
    # assert upper button position
    assert calisto.rail_buttons[0].component.buttons_distance + calisto.rail_buttons[
        0
    ].position == pytest.approx(0.2, 1e-12)


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
