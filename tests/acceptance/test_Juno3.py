import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from rocketpy import Environment, Flight, Function, Rocket, SolidMotor


def test_ndrt_2020_rocket_data_asserts_acceptance():
    # Juno III flight (2023, by Projeto Jupiter)
    # Launched at Spaceport America Cup 2023 edition
    # Date: June 23rd, 2023. around 17hrs local time
    # (Team Number 138)
    # Permission to use flight data given by 
    
    # IMPORTANT RESULTS
    # Last simulated apogee before flight: 3026.054 m
    # Official recorded apogee: 3213 m
    # Relative error of 8.08 %

    # Defining all parameters

    parameters = {
        # Mass Details
        "rocket_mass": (24.05, 0.010),
        # Propulsion details
        "impulse": (8800, 0.033 * 8800),
        "burn_time": (5.8, 0.1),
        "nozzle_radius": (0.0335, 0.001),
        "throat_radius": (0.0114, 0.001),
        "grain_separation": (0.006, 0.001),
        "grain_density": (1748.9, 30),
        "grain_outer_radius": (0.0465, 0.001),
        "grain_initial_inner_radius": (0.016, 0.002),
        "grain_initial_height": (0.156, 0.001),
        # Aerodynamics
        "drag_coefficient": (, ),
        "inertia_i": (, ),
        "inertia_z": (, ),
        "radius": (0.0655, 0.001),
        "distance_rocket_nozzle": (, ),
        "distance_rocket_propellant": (, ),
        "power_off_drag": (, ),
        "power_on_drag": (, ),
        "nose_length": (0.565, 0.001),
        "nose_distance_to_cm": (, ),
        "fin_span": (0.130, 0.001),
        "fin_root_chord": (0.20, 0.001),
        "fin_tip_chord": (0.12, 0.0001),
        "fin_distance_to_cm": (, ),
        "transition_top_radius": (, ),
        "transition_bottom_radius": (,),
        "transition_length": (, ),
        "transition_distance_to_cm": (, ),
        # Launch and environment details
        "wind_direction": (, ),
        "wind_speed": (, ),
        "inclination": (85, 1),
        "heading": (105, 3),
        "rail_length": (5.2, 0.001),
        # Parachute details
        "cd_s_drogue": (0.885, ),
        "cd_s_main": (, ),
        "lag_rec": (0.5, 0.5 / 2),
    }

    # Environment conditions
    env = Environment(
        gravity=9.81,
        latitude=32.939377,
        longitude=-106.911986,
        date=(2023, 6, 23, 17),
        elevation=1480,
    )

    env.set_atmospheric_model(
        type="Reanalysis",
        file="tests/fixtures/acceptance/Juno3/spaceport_america_pressure_levels_2023_hourly.nc",
        dictionary="ECMWF",
    )

    env.max_expected_height = 6000

    # motor information
    # add mandioca to fixtures.
    L1395 = SolidMotor(
        thrust_source="tests/fixtures/acceptance/NDRT_2020/ndrt_2020_motor_Cesaroni_4895L1395-P.eng",
        burn_time=parameters.get("burn_time")[0],
        dry_mass=1,
        dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=0,
        grains_center_of_mass_position=parameters.get("distance_rocket_propellant")[0],
        grain_number=5,
        grain_separation=parameters.get("grain_separation")[0],
        grain_density=parameters.get("grain_density")[0],
        grain_outer_radius=parameters.get("grain_outer_radius")[0],
        grain_initial_inner_radius=parameters.get("grain_initial_inner_radius")[0],
        grain_initial_height=parameters.get("grain_initial_height")[0],
        nozzle_radius=parameters.get("nozzle_radius")[0],
        throat_radius=parameters.get("throat_radius")[0],
        interpolation_method="linear",
        nozzle_position=parameters.get("distance_rocket_nozzle")[0],
    )

