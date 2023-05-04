from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import Rocket, SolidMotor
from rocketpy.AeroSurfaces import NoseCone


@patch("matplotlib.pyplot.show")
def test_rocket(mock_show):
    test_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        grains_center_of_mass_position=0.39796,
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    test_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia_i=6.60,
        inertia_z=0.0351,
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_dry_mass_position=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.add_motor(test_motor, position=-1.255)

    test_rocket.set_rail_buttons([0.2, -0.5])

    nosecone = test_rocket.add_nose(
        length=0.55829, kind="vonKarman", position=1.278, name="NoseCone"
    )
    finset = test_rocket.add_trapezoidal_fins(
        4, span=0.100, root_chord=0.120, tip_chord=0.040, position=-1.04956
    )
    tail = test_rocket.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

    def drogue_trigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.add_parachute(
        "Main",
        CdS=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.add_parachute(
        "Drogue",
        CdS=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    static_margin = test_rocket.static_margin(0)

    # Check if allinfo and static_method methods are working properly
    assert test_rocket.allinfo() == None or not abs(static_margin - 2.05) < 0.01
    # Check if NoseCone allinfo() is working properly
    assert nosecone.allinfo() == None
    # Check if finset allinfo() is working properly
    assert finset.allinfo() == None
    # Check if tail allinfo() is working properly
    assert tail.allinfo() == None
    # Check if draw method is working properly
    assert finset.draw() == None


@patch("matplotlib.pyplot.show")
def test_coordinate_system_orientation(mock_show):
    motor_nozzle_to_combustion_chamber = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        grains_center_of_mass_position=0.39796,
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    motor_combustion_chamber_to_nozzle = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        grains_center_of_mass_position=-0.39796,
        nozzle_position=0,
        coordinate_system_orientation="combustion_chamber_to_nozzle",
    )

    rocket_tail_to_nose = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia_i=6.60,
        inertia_z=0.0351,
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_dry_mass_position=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket_tail_to_nose.add_motor(motor_nozzle_to_combustion_chamber, position=-1.255)

    nosecone = rocket_tail_to_nose.add_nose(
        length=0.55829, kind="vonKarman", position=1.278, name="NoseCone"
    )
    finset = rocket_tail_to_nose.add_trapezoidal_fins(
        4, span=0.100, root_chord=0.120, tip_chord=0.040, position=-1.04956
    )

    static_margin_tail_to_nose = rocket_tail_to_nose.static_margin(0)

    rocket_nose_to_tail = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia_i=6.60,
        inertia_z=0.0351,
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_dry_mass_position=0,
        coordinate_system_orientation="nose_to_tail",
    )

    rocket_nose_to_tail.add_motor(motor_combustion_chamber_to_nozzle, position=1.255)

    nosecone = rocket_nose_to_tail.add_nose(
        length=0.55829, kind="vonKarman", position=-1.278, name="NoseCone"
    )
    finset = rocket_nose_to_tail.add_trapezoidal_fins(
        4, span=0.100, root_chord=0.120, tip_chord=0.040, position=1.04956
    )

    static_margin_nose_to_tail = rocket_nose_to_tail.static_margin(0)

    assert (
        rocket_tail_to_nose.allinfo() == None
        or rocket_nose_to_tail.allinfo() == None
        or not abs(static_margin_tail_to_nose - static_margin_nose_to_tail) < 0.0001
    )


@patch("matplotlib.pyplot.show")
def test_elliptical_fins(mock_show):
    test_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        grains_center_of_mass_position=0.39796,
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    test_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia_i=6.60,
        inertia_z=0.0351,
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_dry_mass_position=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.add_motor(test_motor, position=-1.255)

    test_rocket.set_rail_buttons([0.2, -0.5])

    nosecone = test_rocket.add_nose(
        length=0.55829, kind="vonKarman", position=1.278, name="NoseCone"
    )
    finset = test_rocket.add_elliptical_fins(
        4, span=0.100, root_chord=0.120, position=-1.04956
    )
    tail = test_rocket.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

    def drogue_trigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.add_parachute(
        "Main",
        CdS=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.add_parachute(
        "Drogue",
        CdS=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    static_margin = test_rocket.static_margin(0)

    assert test_rocket.allinfo() == None or not abs(static_margin - 2.30) < 0.01
    assert finset.draw() == None


@patch("matplotlib.pyplot.show")
def test_airfoil(mock_show):
    test_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        grains_center_of_mass_position=0.39796,
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    test_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia_i=6.60,
        inertia_z=0.0351,
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_dry_mass_position=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.add_motor(test_motor, position=-1.255)

    test_rocket.set_rail_buttons([0.2, -0.5])

    nosecone = test_rocket.add_nose(
        length=0.55829, kind="vonKarman", position=1.278, name="NoseCone"
    )
    finsetNACA = test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956,
        airfoil=("tests/fixtures/airfoils/NACA0012-radians.txt", "radians"),
    )
    finsetE473 = test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956,
        airfoil=("tests/fixtures/airfoils/e473-10e6-degrees.csv", "degrees"),
    )
    tail = test_rocket.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

    def drogue_trigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.add_parachute(
        "Main",
        CdS=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.add_parachute(
        "Drogue",
        CdS=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    static_margin = test_rocket.static_margin(0)

    assert test_rocket.allinfo() == None or not abs(static_margin - 2.03) < 0.01


def test_evaluate_static_margin_assert_cp_equals_cm(kg, m, dimensionless_rocket):
    rocket = dimensionless_rocket
    rocket.evaluate_static_margin()

    burn_out_time = rocket.motor.burn_out_time

    assert rocket.center_of_mass(0) / (2 * rocket.radius) == rocket.static_margin(0)
    assert pytest.approx(
        rocket.center_of_mass(burn_out_time) / (2 * rocket.radius), 1e-12
    ) == pytest.approx(rocket.static_margin(burn_out_time), 1e-12)
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
    rocket.add_nose(length=0.55829, kind=type, position=1.278)
    cpz = 1.278 - k * 0.55829  # Relative to the center of dry mass
    clalpha = 2

    static_margin_initial = (rocket.center_of_mass(0) - cpz) / (2 * rocket.radius)
    assert static_margin_initial == pytest.approx(rocket.static_margin(0), 1e-12)

    static_margin_final = (rocket.center_of_mass(np.inf) - cpz) / (2 * rocket.radius)
    assert static_margin_final == pytest.approx(rocket.static_margin(np.inf), 1e-12)

    assert clalpha == pytest.approx(rocket.total_lift_coeff_der, 1e-12)
    assert rocket.cp_position == pytest.approx(cpz, 1e-12)

    dimensionless_rocket.add_nose(length=0.55829 * m, kind=type, position=1.278 * m)
    assert pytest.approx(dimensionless_rocket.static_margin(0), 1e-12) == pytest.approx(
        rocket.static_margin(0), 1e-12
    )
    assert pytest.approx(
        dimensionless_rocket.static_margin(np.inf), 1e-12
    ) == pytest.approx(rocket.static_margin(np.inf), 1e-12)
    assert pytest.approx(
        dimensionless_rocket.total_lift_coeff_der, 1e-12
    ) == pytest.approx(rocket.total_lift_coeff_der, 1e-12)
    assert pytest.approx(dimensionless_rocket.cp_position / m, 1e-12) == pytest.approx(
        rocket.cp_position, 1e-12
    )


def test_add_tail_assert_cp_cm_plus_tail(rocket, dimensionless_rocket, m):
    rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656,
    )

    clalpha = -2 * (1 - (0.0635 / 0.0435) ** (-2)) * (0.0635 / (rocket.radius)) ** 2
    cpz = -1.194656 - (0.06 / 3) * (
        1 + (1 - (0.0635 / 0.0435)) / (1 - (0.0635 / 0.0435) ** 2)
    )

    static_margin_initial = (rocket.center_of_mass(0) - cpz) / (2 * rocket.radius)
    assert static_margin_initial == pytest.approx(rocket.static_margin(0), 1e-12)

    static_margin_final = (rocket.center_of_mass(np.inf) - cpz) / (2 * rocket.radius)
    assert static_margin_final == pytest.approx(rocket.static_margin(np.inf), 1e-12)
    assert np.abs(clalpha) == pytest.approx(np.abs(rocket.total_lift_coeff_der), 1e-8)
    assert rocket.cp_position == cpz

    dimensionless_rocket.add_tail(
        top_radius=0.0635 * m,
        bottom_radius=0.0435 * m,
        length=0.060 * m,
        position=-1.194656 * m,
    )
    assert pytest.approx(dimensionless_rocket.static_margin(0), 1e-12) == pytest.approx(
        rocket.static_margin(0), 1e-12
    )
    assert pytest.approx(
        dimensionless_rocket.static_margin(np.inf), 1e-12
    ) == pytest.approx(rocket.static_margin(np.inf), 1e-12)
    assert pytest.approx(
        dimensionless_rocket.total_lift_coeff_der, 1e-12
    ) == pytest.approx(rocket.total_lift_coeff_der, 1e-12)
    assert pytest.approx(dimensionless_rocket.cp_position / m, 1e-12) == pytest.approx(
        rocket.cp_position, 1e-12
    )


@pytest.mark.parametrize(
    "sweep_angle, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [(39.8, 2.51, 3.16, 1.65), (-10, 2.47, 3.21, 1.63), (29.1, 2.50, 3.28, 1.66)],
)
def test_add_trapezoidal_fins_sweep_angle(
    rocket, sweep_angle, expected_fin_cpz, expected_clalpha, expected_cpz_cm
):
    # Reference values from OpenRocket
    Nose = rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)

    finset = rocket.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_angle=sweep_angle,
        position=-1.182,
    )

    # Check center of pressure
    translate = 0.55829 + 0.71971
    cpz = -1.182 - finset.cpz  # Should be - 1.232
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = finset.cl(1, 0.0)
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
    Nose = rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)

    finset = rocket.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_length=sweep_length,
        position=-1.182,
    )

    # Check center of pressure
    translate = 0.55829 + 0.71971
    cpz = -finset.cp[2] - 1.182
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = finset.cl(1, 0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - rocket.cp_position == pytest.approx(expected_cpz_cm, 0.01)

    # Check if AeroSurfaces.__getitem__() works
    assert isinstance(rocket.aerodynamic_surfaces.__getitem__(0)[0], NoseCone)


def test_add_fins_assert_cp_cm_plus_fins(rocket, dimensionless_rocket, m):
    rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956,
    )

    cpz = -1.04956 - (
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
    assert static_margin_initial == pytest.approx(rocket.static_margin(0), 1e-12)

    static_margin_final = (rocket.center_of_mass(np.inf) - cpz) / (2 * rocket.radius)
    assert static_margin_final == pytest.approx(rocket.static_margin(np.inf), 1e-12)

    assert np.abs(clalpha) == pytest.approx(np.abs(rocket.total_lift_coeff_der), 1e-12)
    assert rocket.cp_position == pytest.approx(cpz, 1e-12)

    dimensionless_rocket.add_trapezoidal_fins(
        4,
        span=0.100 * m,
        root_chord=0.120 * m,
        tip_chord=0.040 * m,
        position=-1.04956 * m,
    )
    assert pytest.approx(dimensionless_rocket.static_margin(0), 1e-12) == pytest.approx(
        rocket.static_margin(0), 1e-12
    )
    assert pytest.approx(
        dimensionless_rocket.static_margin(np.inf), 1e-12
    ) == pytest.approx(rocket.static_margin(np.inf), 1e-12)
    assert pytest.approx(
        dimensionless_rocket.total_lift_coeff_der, 1e-12
    ) == pytest.approx(rocket.total_lift_coeff_der, 1e-12)
    assert pytest.approx(dimensionless_rocket.cp_position / m, 1e-12) == pytest.approx(
        rocket.cp_position, 1e-12
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


def test_set_rail_button_assert_distance_reverse(rocket):
    rocket.set_rail_buttons([-0.5, 0.2])
    assert rocket.rail_buttons.upper_button_position == 0.2
    assert rocket.rail_buttons.lower_button_position == -0.5
    assert rocket.rail_buttons.angular_position == 45
