from unittest.mock import patch

import numpy as np

from rocketpy import Rocket, SolidMotor


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

@patch("matplotlib.pyplot.show")
def test_air_brakes_clamp_on(mock_show, calisto_air_brakes_clamp_on):
    """Test the air brakes class with clamp on configuration. This test checks
    the basic attributes and the set_deployed_level method. It also checks the
    all_info method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show method.
    calisto_air_brakes_clamp_on : Rocket instance
        A predefined instance of the calisto with air brakes in clamp on
        configuration.
    """
    air_brakes_clamp_on = calisto_air_brakes_clamp_on.air_brakes[0]

    # test basic attributes
    assert air_brakes_clamp_on.drag_coefficient.__dom_dim__ == 2
    assert (
        air_brakes_clamp_on.reference_area
        == calisto_air_brakes_clamp_on.radius**2 * np.pi
    )
    air_brakes_clamp_on.set_deployed_level(0.5)
    assert air_brakes_clamp_on.deployed_level == 0.5
    air_brakes_clamp_on.set_deployed_level(1.5)
    assert air_brakes_clamp_on.deployed_level == 1
    air_brakes_clamp_on.set_deployed_level(-1)
    assert air_brakes_clamp_on.deployed_level == 0
    air_brakes_clamp_on.set_deployed_level(0)
    assert air_brakes_clamp_on.deployed_level == 0

    assert air_brakes_clamp_on.all_info() == None


@patch("matplotlib.pyplot.show")
def test_air_brakes_clamp_off(mock_show, calisto_air_brakes_clamp_off):
    """Test the air brakes class with clamp off configuration. This test checks
    the basic attributes and the set_deployed_level method. It also checks the
    all_info method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show method.
    calisto_air_brakes_clamp_off : Rocket instance
        A predefined instance of the calisto with air brakes in clamp off
        configuration.
    """
    air_brakes_clamp_off = calisto_air_brakes_clamp_off.air_brakes[0]

    # test basic attributes
    assert air_brakes_clamp_off.drag_coefficient.__dom_dim__ == 2
    assert (
        air_brakes_clamp_off.reference_area
        == calisto_air_brakes_clamp_off.radius**2 * np.pi
    )

    air_brakes_clamp_off.set_deployed_level(0.5)
    assert air_brakes_clamp_off.deployed_level == 0.5
    air_brakes_clamp_off.set_deployed_level(1.5)
    assert air_brakes_clamp_off.deployed_level == 1.5
    air_brakes_clamp_off.set_deployed_level(-1)
    assert air_brakes_clamp_off.deployed_level == -1
    air_brakes_clamp_off.set_deployed_level(0)
    assert air_brakes_clamp_off.deployed_level == 0

    assert air_brakes_clamp_off.all_info() == None
