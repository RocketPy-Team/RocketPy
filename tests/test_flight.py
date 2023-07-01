import datetime
import os
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest
from scipy import optimize

from rocketpy import Environment, Flight, Function, Rocket, SolidMotor, Components

plt.rcParams.update({"figure.max_open_warning": 0})

# Helper functions


def setup_rocket_with_given_static_margin(rocket, static_margin):
    """Takes any rocket, removes its aerodynamic surfaces and adds a set of
    nose, fins and tail specially designed to have a given static margin.
    The rocket is modified in place.

    Parameters
    ----------
    rocket : Rocket
        Rocket to be modified
    static_margin : float
        Static margin that the given rocket shall have

    Returns
    -------
    rocket : Rocket
        Rocket with the given static margin.
    """

    def compute_static_margin_error_given_distance(position, static_margin, rocket):
        rocket.aerodynamic_surfaces = Components()
        rocket.add_nose(length=0.5, kind="vonKarman", position=1.0 + 0.5)
        rocket.add_trapezoidal_fins(
            4,
            span=0.100,
            root_chord=0.100,
            tip_chord=0.100,
            position=position,
        )
        rocket.add_tail(
            top_radius=0.0635,
            bottom_radius=0.0435,
            length=0.060,
            position=-1.194656,
        )
        return rocket.static_margin(0) - static_margin

    sol = optimize.root_scalar(
        compute_static_margin_error_given_distance,
        bracket=[-2.0, 2.0],
        method="brentq",
        args=(static_margin, rocket),
    )

    return rocket


@patch("matplotlib.pyplot.show")
def test_flight(mock_show):
    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time

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

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
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

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    assert test_flight.all_info() == None


@patch("matplotlib.pyplot.show")
def test_initial_solution(mock_show):
    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time

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

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
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
        return true if y[5] < 0 else false

    def main_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return true if y[5] < 0 and h < 800 else false

    main = test_rocket.add_parachute(
        "main",
        cd_s=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    drogue = test_rocket.add_parachute(
        "drogue",
        cd_s=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        railLength=5,
        inclination=85,
        heading=0,
    )

    assert test_flight.all_info() == None


@patch("matplotlib.pyplot.show")
def test_initial_solution(mock_show):
    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time

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

    test_rocket.set_rail_buttons(0.2, -0.5)

    test_rocket.add_motor(test_motor, position=-1.255)

    nosecone = test_rocket.add_nose(
        length=0.55829, kind="vonKarman", position=0.71971 + 0.558291
    )
    fin_set = test_rocket.add_trapezoidal_fins(
        4, span=0.100, root_chord=0.120, tip_chord=0.040, position=-1.04956
    )
    tail = test_rocket.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
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

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
        # max_time=300*60,
        # minTimeStep=0.1,
        # max_time_step=10,
        rtol=1e-8,
        atol=1e-6,
        verbose=True,
        initial_solution=[
            0.0,
            0.0,
            0.0,
            1.5e3,
            10,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    assert test_flight.all_info() == None


@pytest.mark.parametrize("wind_u, wind_v", [(0, 10), (0, -10), (10, 0), (-10, 0)])
@pytest.mark.parametrize(
    "static_margin, max_time",
    [(-0.1, 2), (-0.01, 5), (0, 5), (0.01, 20), (0.1, 20), (1.0, 20)],
)
def test_stability_static_margins(wind_u, wind_v, static_margin, max_time):
    """Test stability margins for a constant velocity flight, 100 m/s, wind a
    lateral wind speed of 10 m/s. Rocket has infinite mass to prevent side motion.
    Check if a restoring moment exists depending on static margins."""

    # Create an environment with ZERO gravity to keep the rocket's speed constant
    env = Environment(gravity=0, latitude=0, longitude=0, elevation=0)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=wind_u,
        wind_v=wind_v,
        pressure=101325,
        temperature=300,
    )
    # Make sure that the free_stream_mach will always be 0, so that the rocket
    # behaves as the STATIC (free_stream_mach=0) margin predicts
    env.speed_of_sound = Function(1e16)

    # Create a motor with ZERO thrust and ZERO mass to keep the rocket's speed constant
    dummy_motor = SolidMotor(
        thrust_source=1e-300,
        burn_time=1e-10,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        grains_center_of_mass_position=0.397,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1e-300,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        nozzle_position=0,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    # create a rocket with zero drag and huge mass to keep the rocket's speed constant
    dummy_rocket = Rocket(
        radius=127 / 2000,
        mass=1e16,
        inertia=(1, 1, 0.034),
        power_off_drag=0,
        power_on_drag=0,
        center_of_mass_without_motor=0,
    )
    dummy_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)
    dummy_rocket.add_motor(dummy_motor, position=-1.255 - 0.1182359460624346)

    setup_rocket_with_given_static_margin(dummy_rocket, static_margin)

    # Simulate
    init_pos = [0, 0, 100]  # Start at 100 m of altitude
    init_vel = [0, 0, 100]  # Start at 100 m/s
    init_att = [1, 0, 0, 0]  # Inclination of 90 deg and heading of 0 deg
    init_angvel = [0, 0, 0]
    initial_solution = [0] + init_pos + init_vel + init_att + init_angvel
    test_flight = Flight(
        rocket=dummy_rocket,
        rail_length=1,
        environment=env,
        initial_solution=initial_solution,
        max_time=max_time,
        max_time_step=1e-2,
        verbose=False,
    )
    test_flight.post_process(interpolation="linear")

    # Check stability according to static margin
    if wind_u == 0:
        moments = test_flight.M1.get_source()[:, 1]
        wind_sign = np.sign(wind_v)
    else:  # wind_v == 0
        moments = test_flight.M2.get_source()[:, 1]
        wind_sign = -np.sign(wind_u)

    assert (
        (static_margin > 0 and np.max(moments) * np.min(moments) < 0)
        or (static_margin < 0 and np.all(moments / wind_sign <= 0))
        or (static_margin == 0 and np.all(np.abs(moments) <= 1e-10))
    )


@patch("matplotlib.pyplot.show")
def test_rolling_flight(mock_show):
    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time
    test_env.set_atmospheric_model(type="standard_atmosphere")

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

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
    )
    fin_set = test_rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956,
        cant_angle=0.5 - 0.1182359460624346,
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

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    assert test_flight.all_info() == None


@patch("matplotlib.pyplot.show")
def test_simpler_parachute_triggers(mock_show):
    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time

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

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
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

    main = test_rocket.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=800,
        sampling_rate=105,
        lag=0,
    )

    drogue = test_rocket.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger="apogee",
        sampling_rate=105,
        lag=0,
    )

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    assert (
        abs(test_flight.z(test_flight.parachute_events[0][0]) - test_flight.apogee) <= 1
    )
    assert (
        abs(
            test_flight.z(test_flight.parachute_events[1][0])
            - (800 + test_env.elevation)
        )
        <= 1
    )

    assert test_flight.all_info() == None


def test_export_data():
    "Tests weather the method Flight.export_data is working as intended"

    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )

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
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
    )
    fin_set = test_rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
    )

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    # Basic export
    test_flight.export_data("test_export_data_1.csv")

    # Custom export
    test_flight.export_data(
        "test_export_data_2.csv",
        "z",
        "vz",
        "e1",
        "w3",
        "angle_of_attack",
        time_step=0.1,
    )

    # Load exported files and fixtures and compare them

    test_1 = np.loadtxt("test_export_data_1.csv", delimiter=",")
    test_2 = np.loadtxt("test_export_data_2.csv", delimiter=",")

    # Delete files
    os.remove("test_export_data_1.csv")
    os.remove("test_export_data_2.csv")

    # Check if basic exported content matches data
    assert np.allclose(test_flight.x[:, 0], test_1[:, 0], atol=1e-5) == True
    assert np.allclose(test_flight.x[:, 1], test_1[:, 1], atol=1e-5) == True
    assert np.allclose(test_flight.y[:, 1], test_1[:, 2], atol=1e-5) == True
    assert np.allclose(test_flight.z[:, 1], test_1[:, 3], atol=1e-5) == True
    assert np.allclose(test_flight.vx[:, 1], test_1[:, 4], atol=1e-5) == True
    assert np.allclose(test_flight.vy[:, 1], test_1[:, 5], atol=1e-5) == True
    assert np.allclose(test_flight.vz[:, 1], test_1[:, 6], atol=1e-5) == True
    assert np.allclose(test_flight.e0[:, 1], test_1[:, 7], atol=1e-5) == True
    assert np.allclose(test_flight.e1[:, 1], test_1[:, 8], atol=1e-5) == True
    assert np.allclose(test_flight.e2[:, 1], test_1[:, 9], atol=1e-5) == True
    assert np.allclose(test_flight.e3[:, 1], test_1[:, 10], atol=1e-5) == True
    assert np.allclose(test_flight.w1[:, 1], test_1[:, 11], atol=1e-5) == True
    assert np.allclose(test_flight.w2[:, 1], test_1[:, 12], atol=1e-5) == True
    assert np.allclose(test_flight.w3[:, 1], test_1[:, 13], atol=1e-5) == True

    # Check if custom exported content matches data
    timePoints = np.arange(test_flight.t_initial, test_flight.t_final, 0.1)
    assert np.allclose(timePoints, test_2[:, 0], atol=1e-5) == True
    assert np.allclose(test_flight.z(timePoints), test_2[:, 1], atol=1e-5) == True
    assert np.allclose(test_flight.vz(timePoints), test_2[:, 2], atol=1e-5) == True
    assert np.allclose(test_flight.e1(timePoints), test_2[:, 3], atol=1e-5) == True
    assert np.allclose(test_flight.w3(timePoints), test_2[:, 4], atol=1e-5) == True
    assert (
        np.allclose(test_flight.angle_of_attack(timePoints), test_2[:, 5], atol=1e-5)
        == True
    )


def test_export_kml():
    "Tests weather the method Flight.export_kml is working as intended"

    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )

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
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
    )
    fin_set = test_rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
    )

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    # Basic export
    test_flight.export_kml(
        "test_export_data_1.kml", time_step=None, extrude=True, altitude_mode="absolute"
    )

    # Load exported files and fixtures and compare them
    test_1 = open("test_export_data_1.kml", "r")

    for row in test_1:
        if row[:29] == "                <coordinates>":
            r = row[29:-15]
            r = r.split(",")
            for i, j in enumerate(r):
                r[i] = j.split(" ")
    lon, lat, z, coords = [], [], [], []
    for i in r:
        for j in i:
            coords.append(j)
    for i in range(0, len(coords), 3):
        lon.append(float(coords[i]))
        lat.append(float(coords[i + 1]))
        z.append(float(coords[i + 2]))

    # Delete temporary test file
    test_1.close()
    os.remove("test_export_data_1.kml")

    assert np.allclose(test_flight.latitude[:, 1], lat, atol=1e-3) == True
    assert np.allclose(test_flight.longitude[:, 1], lon, atol=1e-3) == True
    assert np.allclose(test_flight.z[:, 1], z, atol=1e-3) == True


@patch("matplotlib.pyplot.show")
def test_latlon_conversions(mock_show):
    test_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )

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
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
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

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=45,
    )

    # Check for initial and final lat/lon coordinates based on launch pad coordinates
    test_flight.post_process()
    assert abs(test_flight.latitude(0)) - abs(test_flight.env.latitude) < 1e-6
    assert abs(test_flight.longitude(0)) - abs(test_flight.env.longitude) < 1e-6
    assert test_flight.latitude(test_flight.t_final) > test_flight.env.latitude
    assert test_flight.longitude(test_flight.t_final) > test_flight.env.longitude


@patch("matplotlib.pyplot.show")
def test_latlon_conversions2(mock_show):
    "additional tests to capture incorrect behaviors during lat/lon conversions"
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
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    test_rocket.set_rail_buttons(0.2 - 0.1182359460624346, -0.5 - 0.1182359460624346)

    test_rocket.add_motor(test_motor, position=-1.255 - 0.1182359460624346)

    nose_cone = test_rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=0.71971 + 0.558291 - 0.1182359460624346,
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

    test_env = Environment(
        latitude=0,
        longitude=0,
        elevation=1400,
    )

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    assert abs(test_flight.longitude(test_flight.t_final) - 0) < 1e-12
    assert test_flight.latitude(test_flight.t_final) > 0


@pytest.mark.skip(reason="legacy tests")
@pytest.mark.parametrize(
    "rail_length, out_of_rail_time",
    [
        (0.52, 0.5180212542878443),
        (5.2, 5.180378138072207),
        (50.2, 50.00897551720473),
        (100000, 100003.35594050681),
    ],
)
def test_rail_length(rocket, example_env, rail_length, out_of_rail_time):
    """Test the rail length parameter of the Flight class. This test simply
    simulate the flight using different rail lengths and checks if the expected
    out of rail altitude is achieved. Four different rail lengths are
    tested: 0.001, 1, 10, and 100000 meters. This provides a good test range.
    Currently, if a rail length of 0 is used, the simulation will fail in a
    ZeroDivisionError, which is not being tested here.

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket to be simulated. In this case, the fixture rocket is used.
        See the conftest.py file for more information.
    example_env : rocketpy.Environment
        The environment to be simulated. In this case, the fixture environment
        is used. See the conftest.py file for more information.
    rail_length : float, int
        The length of the rail in meters. It must be a positive number. See the
        Flight class documentation for more information.
    out_of_rail_time : float, int
        The expected time at which the rocket leaves the rail in seconds.
    """
    test_flight = Flight(
        rocket=rocket,
        environment=example_env,
        rail_length=rail_length,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    assert (test_flight.z(test_flight.out_of_rail_time) - out_of_rail_time) < 1e-6
