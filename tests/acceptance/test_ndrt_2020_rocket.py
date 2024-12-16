import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from rocketpy import Environment, Flight, Rocket, SolidMotor


@pytest.mark.parametrize(
    "env_file",
    [
        "data/weather/ndrt_2020_weather_data_ERA5.nc",
        "data/weather/ndrt_2020_weather_data_ERA5_new.nc",
    ],
)
def test_ndrt_2020_rocket_data_asserts_acceptance(env_file):
    # Notre Dame Rocket Team 2020 Flight
    # Launched at 19045-18879 Avery Rd, Three Oaks, MI 49128
    # Permission to use flight data given by Brooke Mumma, 2020
    #
    # IMPORTANT RESULTS  (23rd feb)
    # Measured Stability Margin 2.875 cal
    # Official Target Altitude 4,444 ft
    # Measured Altitude 4,320 ft or 1316.736 m
    # Drift: 2275 ft

    # Importing libraries

    # Defining all parameters
    parameters = {
        # Mass Details
        "rocket_mass": (23.321 - 2.475 - 1, 0.010),
        # propulsion details
        "impulse": (4895.050, 0.033 * 4895.050),
        "burn_time": (3.45, 0.1),
        "nozzle_radius": (49.5 / 2000, 0.001),
        "throat_radius": (21.5 / 2000, 0.001),
        "grain_separation": (3 / 1000, 0.001),
        "grain_density": (1519.708, 30),
        "grain_outer_radius": (33 / 1000, 0.001),
        "grain_initial_inner_radius": (15 / 1000, 0.002),
        "grain_initial_height": (120 / 1000, 0.001),
        # aerodynamic details
        "drag_coefficient": (0.44, 0.1),
        "inertia_i": (83.351, 0.3 * 83.351),
        "inertia_z": (0.15982, 0.3 * 0.15982),
        "radius": (203 / 2000, 0.001),
        "distance_rocket_nozzle": (-1.255, 0.100),
        "distance_rocket_propellant": (-0.85704, 0.100),
        "power_off_drag": (1, 0.033),
        "power_on_drag": (1, 0.033),
        "nose_length": (0.610, 0.001),
        "nose_distance_to_cm": (0.71971, 0.100),
        "fin_span": (0.165, 0.001),
        "fin_root_chord": (0.152, 0.001),
        "fin_tip_chord": (0.0762, 0.001),
        "fin_distance_to_cm": (-1.04956, 0.100),
        "transition_top_radius": (203 / 2000, 0.010),
        "transition_bottom_radius": (155 / 2000, 0.010),
        "transition_length": (0.127, 0.010),
        "transition_distance_to_cm": (-1.194656, 0.010),
        # launch and environment details
        "wind_direction": (0, 3),
        "wind_speed": (1, 0.30),
        "inclination": (90, 1),
        "heading": (181, 3),
        "rail_length": (3.353, 0.001),
        # parachute details
        "cd_s_drogue": (1.5 * np.pi * (24 * 25.4 / 1000) * (24 * 25.4 / 1000) / 4, 0.1),
        "cd_s_main": (2.2 * np.pi * (120 * 25.4 / 1000) * (120 * 25.4 / 1000) / 4, 0.1),
        "lag_rec": (1, 0.5),
    }

    # Environment conditions
    env = Environment(
        gravity=9.81,
        latitude=41.775447,
        longitude=-86.572467,
        date=(2020, 2, 23, 16),
        elevation=206,
    )
    env.set_atmospheric_model(
        type="Reanalysis",
        file=env_file,
        dictionary="ECMWF",
    )
    env.max_expected_height = 2000

    # motor information
    L1395 = SolidMotor(
        thrust_source="data/motors/cesaroni/Cesaroni_4895L1395-P.eng",
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

    # Rocket information
    NDRT2020 = Rocket(
        radius=parameters.get("radius")[0],
        mass=parameters.get("rocket_mass")[0],
        inertia=(
            parameters.get("inertia_i")[0],
            parameters.get("inertia_i")[0],
            parameters.get("inertia_z")[0],
        ),
        power_off_drag=parameters.get("drag_coefficient")[0],
        power_on_drag=parameters.get("drag_coefficient")[0],
        center_of_mass_without_motor=0,
    )
    NDRT2020.set_rail_buttons(0.2, -0.5, 45)
    NDRT2020.add_motor(L1395, parameters.get("distance_rocket_nozzle")[0])
    NDRT2020.add_nose(
        length=parameters.get("nose_length")[0],
        kind="tangent",
        position=parameters.get("nose_distance_to_cm")[0]
        + parameters.get("nose_length")[0],
    )
    NDRT2020.add_trapezoidal_fins(
        3,
        span=parameters.get("fin_span")[0],
        root_chord=parameters.get("fin_root_chord")[0],
        tip_chord=parameters.get("fin_tip_chord")[0],
        position=parameters.get("fin_distance_to_cm")[0],
    )
    NDRT2020.add_tail(
        top_radius=parameters.get("transition_top_radius")[0],
        bottom_radius=parameters.get("transition_bottom_radius")[0],
        length=parameters.get("transition_length")[0],
        position=parameters.get("transition_distance_to_cm")[0],
    )

    # Parachute set-up
    def drogue_trigger(p, h, y):  # pylint: disable=unused-argument
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def main_trigger(p, h, y):  # pylint: disable=unused-argument
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 167.64 m (AGL) or 550 ft (AGL)
        return True if y[5] < 0 and h < 167.64 else False

    NDRT2020.add_parachute(
        "Drogue",
        cd_s=parameters.get("cd_s_drogue")[0],
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=parameters.get("lag_rec")[0],
        noise=(0, 8.3, 0.5),
    )
    NDRT2020.add_parachute(
        "Main",
        cd_s=parameters.get("cd_s_main")[0],
        trigger=main_trigger,
        sampling_rate=105,
        lag=parameters.get("lag_rec")[0],
        noise=(0, 8.3, 0.5),
    )

    # Flight
    rocketpy_flight = Flight(
        rocket=NDRT2020,
        environment=env,
        rail_length=parameters.get("rail_length")[0],
        inclination=parameters.get("inclination")[0],
        heading=parameters.get("heading")[0],
    )
    df_ndrt_rocketpy = pd.DataFrame(
        rocketpy_flight.z[:, :], columns=["Time", "Altitude"]
    )
    df_ndrt_rocketpy["Vertical Velocity"] = rocketpy_flight.vz[:, 1]
    # df_ndrt_rocketpy["Vertical Acceleration"] = rocketpy_flight.az[:, 1]
    df_ndrt_rocketpy["Altitude"] -= env.elevation

    # Reading data from the flightData (sensors: Raven)
    df_ndrt_raven = pd.read_csv("data/rockets/NDRT_2020/ndrt_2020_flight_data.csv")
    # convert feet to meters
    df_ndrt_raven[" Altitude (m-AGL)"] = df_ndrt_raven[" Altitude (Ft-AGL)"] / 3.28084
    # Calculate the vertical velocity as a derivative of the altitude
    velocity_raven = [0]
    for i in range(1, len(df_ndrt_raven[" Altitude (m-AGL)"]), 1):
        v = (
            df_ndrt_raven[" Altitude (m-AGL)"][i]
            - df_ndrt_raven[" Altitude (m-AGL)"][i - 1]
        ) / (df_ndrt_raven[" Time (s)"][i] - df_ndrt_raven[" Time (s)"][i - 1])
        if (
            v != 92.85844059786486
            and v != -376.85000927682034
            and v != -57.00530169566588
            and v != -52.752200796647145
            and v != 63.41561104540437
        ):
            # This way we remove the outliers
            velocity_raven.append(v)
        else:
            velocity_raven.append(velocity_raven[-1])
    velocity_raven_filt = savgol_filter(velocity_raven, 51, 3)

    apogee_time_measured = df_ndrt_raven.loc[
        df_ndrt_raven[" Altitude (Ft-AGL)"].idxmax(), " Time (s)"
    ]
    apogee_time_simulated = rocketpy_flight.apogee_time

    assert (
        abs(max(df_ndrt_raven[" Altitude (m-AGL)"]) - max(df_ndrt_rocketpy["Altitude"]))
        / max(df_ndrt_raven[" Altitude (m-AGL)"])
        < 0.015
    )
    assert (max(velocity_raven_filt) - rocketpy_flight.max_speed) / max(
        velocity_raven_filt
    ) < 0.06
    assert (
        abs(apogee_time_measured - apogee_time_simulated) / apogee_time_simulated < 0.02
    )
