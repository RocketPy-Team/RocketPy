from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import Rocket, SolidMotor
from rocketpy.AeroSurface import NoseCone


@patch("matplotlib.pyplot.show")
def test_rocket(mock_show, solid_motor):
    test_motor = solid_motor
    test_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956 - 1.815,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=1.278 - 0.1182359460624346,
        name="NoseCone",
    )
    fin_set = test_rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
    )
    tail = test_rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656 - 0.1182359460624346,
    )

    def drogue_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and h < 800 else False

    main = test_rocket.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    drogue = test_rocket.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    static_margin = test_rocket.static_margin(0)

    # Check if all_info and static_method methods are working properly
    assert test_rocket.all_info() == None or not abs(static_margin - 2.05) < 0.01
    # Check if NoseCone all_info() is working properly
    assert nose_cone.all_info() == None
    # Check if fin_set all_info() is working properly
    assert fin_set.all_info() == None
    # Check if tail all_info() is working properly
    assert tail.all_info() == None
    # Check if draw method is working properly
    assert fin_set.draw() == None


@patch("matplotlib.pyplot.show")
def test_coordinate_system_orientation(mock_show):
    motor_nozzleToCombustionChamber = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    motor_combustionChamberToNozzle = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="combustion_chamber_to_nozzle",
    )

    rocket_tail_to_nose = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956 - 1.815,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket_tail_to_nose.add_motor(
        motor_nozzleToCombustionChamber, position=-1.255 - 0.1182359460624346
    )

    nose_cone = rocket_tail_to_nose.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=1.278 - 0.1182359460624346,
        name="NoseCone",
    )
    fin_set = rocket_tail_to_nose.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
    )

    static_margin_tail_to_nose = rocket_tail_to_nose.static_margin(0)

    rocket_nose_to_tail = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956 - 1.815,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )

    rocket_nose_to_tail.add_motor(
        motor_combustionChamberToNozzle, position=1.255 + 0.1182359460624346
    )

    NoseCone = rocket_nose_to_tail.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=-1.278 + 0.1182359460624346,
        name="NoseCone",
    )
    fin_set = rocket_nose_to_tail.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=1.04956 + 0.1182359460624346,
    )

    static_margin_nose_to_tail = rocket_nose_to_tail.static_margin(0)

    assert (
        rocket_tail_to_nose.all_info() == None
        or rocket_nose_to_tail.all_info() == None
        or not abs(static_margin_tail_to_nose - static_margin_nose_to_tail) < 0.0001
    )


@patch("matplotlib.pyplot.show")
def test_elliptical_fins(mock_show):
    test_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    test_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956 - 1.815,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=1.278 - 0.1182359460624346,
        name="NoseCone",
    )
    fin_set = test_rocket.add_elliptical_fins(
        4, span=0.100, root_chord=0.120, position=-1.04956 - 0.1182359460624346
    )
    Tail = test_rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656 - 0.1182359460624346,
    )

    def drogue_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and h < 800 else False

    Main = test_rocket.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    static_margin = test_rocket.static_margin(0)

    assert test_rocket.all_info() == None or not abs(static_margin - 2.30) < 0.01


@patch("matplotlib.pyplot.show")
def test_airfoil(mock_show):
    test_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    test_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956 - 1.815,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    NoseCone = test_rocket.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=1.278 - 0.1182359460624346,
        name="NoseCone",
    )
    fin_set_NACA = test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
        airfoil=("tests/fixtures/airfoils/NACA0012-radians.txt", "radians"),
    )
    fin_set_E473 = test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
        airfoil=("tests/fixtures/airfoils/e473-10e6-degrees.csv", "degrees"),
    )
    Tail = test_rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656 - 0.1182359460624346,
    )

    def drogue_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and h < 800 else False

    Main = test_rocket.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    static_margin = test_rocket.static_margin(0)

    assert test_rocket.all_info() == None or not abs(static_margin - 2.03) < 0.01


def test_evaluate_static_margin_assert_cp_equals_cm(kg, m, dimensionless_rocket):
    rocket = dimensionless_rocket
    rocket.evaluate_static_margin()

    burn_time = rocket.motor.burn_time

    assert pytest.approx(
        rocket.center_of_mass(0) / (2 * rocket.radius), 1e-8
    ) == pytest.approx(rocket.static_margin(0), 1e-8)
    assert pytest.approx(
        rocket.center_of_mass(burn_time[1]) / (2 * rocket.radius), 1e-8
    ) == pytest.approx(rocket.static_margin(burn_time[1]), 1e-8)
    assert rocket.total_lift_coeff_der == 0
    assert rocket.cp_position == 0


@pytest.mark.parametrize(
    "k, type",
    (
        [2 / 3, "conical"],
        [0.466, "ogive"],
        [0.563, "lvhaack"],
        [0.5, "default"],
        [0.5, "not a mapped string, to show default case"],
    ),
)
def test_add_nose_assert_cp_cm_plus_nose(k, type, rocket, dimensionless_rocket, m):
    rocket.add_nose(length=0.55829, kind=type, position=1.278 - 0.1182359460624346)
    cpz = (
        1.278 - 0.1182359460624346
    ) - k * 0.55829  # Relative to the center of dry mass
    clalpha = 2

    static_margin_initial = (rocket.center_of_mass(0) - cpz) / (2 * rocket.radius)
    assert static_margin_initial == pytest.approx(rocket.static_margin(0), 1e-8)

    static_margin_final = (rocket.center_of_mass(np.inf) - cpz) / (2 * rocket.radius)
    assert static_margin_final == pytest.approx(rocket.static_margin(np.inf), 1e-8)

    assert clalpha == pytest.approx(rocket.total_lift_coeff_der, 1e-8)
    assert rocket.cp_position == pytest.approx(cpz, 1e-8)

    dimensionless_rocket.add_nose(
        length=0.55829 * m, kind=type, position=(1.278 - 0.1182359460624346) * m
    )
    assert pytest.approx(dimensionless_rocket.static_margin(0), 1e-8) == pytest.approx(
        rocket.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_rocket.static_margin(np.inf), 1e-8
    ) == pytest.approx(rocket.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_rocket.total_lift_coeff_der, 1e-8
    ) == pytest.approx(rocket.total_lift_coeff_der, 1e-8)
    assert pytest.approx(dimensionless_rocket.cp_position / m, 1e-8) == pytest.approx(
        rocket.cp_position, 1e-8
    )


def test_add_tail_assert_cp_cm_plus_tail(rocket, dimensionless_rocket, m):
    rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656 - 0.1182359460624346,
    )

    clalpha = -2 * (1 - (0.0635 / 0.0435) ** (-2)) * (0.0635 / (rocket.radius)) ** 2
    cpz = (-1.194656 - 0.1182359460624346) - (0.06 / 3) * (
        1 + (1 - (0.0635 / 0.0435)) / (1 - (0.0635 / 0.0435) ** 2)
    )

    static_margin_initial = (rocket.center_of_mass(0) - cpz) / (2 * rocket.radius)
    assert static_margin_initial == pytest.approx(rocket.static_margin(0), 1e-8)

    static_margin_final = (rocket.center_of_mass(np.inf) - cpz) / (2 * rocket.radius)
    assert static_margin_final == pytest.approx(rocket.static_margin(np.inf), 1e-8)
    assert np.abs(clalpha) == pytest.approx(np.abs(rocket.total_lift_coeff_der), 1e-8)
    assert rocket.cp_position == cpz

    dimensionless_rocket.add_tail(
        top_radius=0.0635 * m,
        bottom_radius=0.0435 * m,
        length=0.060 * m,
        position=(-1.194656 - 0.1182359460624346) * m,
    )
    assert pytest.approx(dimensionless_rocket.static_margin(0), 1e-8) == pytest.approx(
        rocket.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_rocket.static_margin(np.inf), 1e-8
    ) == pytest.approx(rocket.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_rocket.total_lift_coeff_der, 1e-8
    ) == pytest.approx(rocket.total_lift_coeff_der, 1e-8)
    assert pytest.approx(dimensionless_rocket.cp_position / m, 1e-8) == pytest.approx(
        rocket.cp_position, 1e-8
    )


@pytest.mark.parametrize(
    "sweep_angle, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [(39.8, 2.51, 3.16, 1.65), (-10, 2.47, 3.21, 1.63), (29.1, 2.50, 3.28, 1.66)],
)
def test_add_trapezoidal_fins_sweep_angle(
    rocket, sweep_angle, expected_fin_cpz, expected_clalpha, expected_cpz_cm
):
    # Reference values from OpenRocket
    Nose = rocket.add_nose(
        length=0.55829, kind="vonKarman", position=1.278 - 0.1182359460624346
    )

    fin_set = rocket.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_angle=sweep_angle,
        position=-1.182 - 0.1182359460624346,
    )

    # Check center of pressure
    translate = 0.55829 + 0.71971 - 0.1182359460624346
    cpz = -1.182 - fin_set.cpz - 0.1182359460624346
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = fin_set.cl(1, 0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - rocket.cp_position == pytest.approx(expected_cpz_cm, 0.01)


@pytest.mark.parametrize(
    "sweep_length, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [(0.075, 2.51, 3.16, 1.65), (-0.0159, 2.47, 3.21, 1.63), (0.05, 2.50, 3.28, 1.66)],
)
def test_add_trapezoidal_fins_sweep_length(
    rocket, sweep_length, expected_fin_cpz, expected_clalpha, expected_cpz_cm
):
    # Reference values from OpenRocket
    Nose = rocket.add_nose(
        length=0.55829, kind="vonKarman", position=1.278 - 0.1182359460624346
    )

    fin_set = rocket.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_length=sweep_length,
        position=-1.182 - 0.1182359460624346,
    )

    # Check center of pressure
    translate = 1.278 - 0.1182359460624346
    cpz = -fin_set.cp[2] - 1.182 - 0.1182359460624346
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = fin_set.cl(1, 0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - rocket.cp_position == pytest.approx(expected_cpz_cm, 0.01)

    assert isinstance(rocket.aerodynamic_surfaces[0].component, NoseCone)


def test_add_fins_assert_cp_cm_plus_fins(rocket, dimensionless_rocket, m):
    rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
    )

    cpz = (-1.04956 - 0.1182359460624346) - (
        ((0.120 - 0.040) / 3) * ((0.120 + 2 * 0.040) / (0.120 + 0.040))
        + (1 / 6) * (0.120 + 0.040 - 0.120 * 0.040 / (0.120 + 0.040))
    )

    clalpha = (4 * 4 * (0.1 / (2 * rocket.radius)) ** 2) / (
        1
        + np.sqrt(
            1
            + (2 * np.sqrt((0.12 / 2 - 0.04 / 2) ** 2 + 0.1**2) / (0.120 + 0.040))
            ** 2
        )
    )
    clalpha *= 1 + rocket.radius / (0.1 + rocket.radius)

    static_margin_initial = (rocket.center_of_mass(0) - cpz) / (2 * rocket.radius)
    assert static_margin_initial == pytest.approx(rocket.static_margin(0), 1e-8)

    static_margin_final = (rocket.center_of_mass(np.inf) - cpz) / (2 * rocket.radius)
    assert static_margin_final == pytest.approx(rocket.static_margin(np.inf), 1e-8)

    assert np.abs(clalpha) == pytest.approx(np.abs(rocket.total_lift_coeff_der), 1e-8)
    assert rocket.cp_position == pytest.approx(cpz, 1e-8)

    dimensionless_rocket.add_trapezoidal_fins(
        4,
        span=0.100 * m,
        root_chord=0.120 * m,
        tip_chord=0.040 * m,
        position=(-1.04956 - 0.1182359460624346) * m,
    )
    assert pytest.approx(dimensionless_rocket.static_margin(0), 1e-8) == pytest.approx(
        rocket.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_rocket.static_margin(np.inf), 1e-8
    ) == pytest.approx(rocket.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_rocket.total_lift_coeff_der, 1e-8
    ) == pytest.approx(rocket.total_lift_coeff_der, 1e-8)
    assert pytest.approx(dimensionless_rocket.cp_position / m, 1e-8) == pytest.approx(
        rocket.cp_position, 1e-8
    )


def test_add_cm_eccentricity_assert_properties_set(rocket):
    rocket.add_cm_eccentricity(x=4, y=5)

    assert rocket.cp_eccentricity_x == -4
    assert rocket.cp_eccentricity_y == -5

    assert rocket.thrust_eccentricity_y == -4
    assert rocket.thrust_eccentricity_x == -5


def test_add_thrust_eccentricity_assert_properties_set(rocket):
    rocket.add_thrust_eccentricity(x=4, y=5)

    assert rocket.thrust_eccentricity_y == 4
    assert rocket.thrust_eccentricity_x == 5


def test_add_cp_eccentricity_assert_properties_set(rocket):
    rocket.add_cp_eccentricity(x=4, y=5)

    assert rocket.cp_eccentricity_x == 4
    assert rocket.cp_eccentricity_y == 5


def test_set_rail_button(rocket):
    rail_buttons = rocket.set_rail_buttons(0.2, -0.5, 30)
    # assert buttons_distance
    assert (
        rail_buttons.buttons_distance
        == rocket.rail_buttons[0].component.buttons_distance
        == pytest.approx(0.7, 1e-12)
    )
    # assert buttons position on rocket
    assert rocket.rail_buttons[0].position == -0.5
    # assert angular position
    assert (
        rail_buttons.angular_position
        == rocket.rail_buttons[0].component.angular_position
        == 30
    )
    # assert upper button position
    assert rocket.rail_buttons[0].component.buttons_distance + rocket.rail_buttons[
        0
    ].position == pytest.approx(0.2, 1e-12)
