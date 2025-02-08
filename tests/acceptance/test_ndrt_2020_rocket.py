import pandas as pd
import pytest
from scipy.signal import savgol_filter

from rocketpy import Environment, Flight


@pytest.mark.parametrize(
    "env_file",
    [
        "data/weather/ndrt_2020_weather_data_ERA5.nc",
        "data/weather/ndrt_2020_weather_data_ERA5_new.nc",
    ],
)
def test_ndrt_2020_rocket_data_asserts_acceptance(env_file, ndrt_2020_rocket):
    """
    Notre Dame Rocket Team 2020 Flight
    - Launched at 19045-18879 Avery Rd, Three Oaks, MI 49128
    - Permission to use flight data given by Brooke Mumma, 2020

    IMPORTANT RESULTS  (23rd feb)
    - Measured Stability Margin 2.875 cal
    - Official Target Altitude 4,444 ft
    - Measured Altitude 4,320 ft or 1316.736 m
    - Drift: 2275 ft
    """

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

    # Flight
    rocketpy_flight = Flight(
        rocket=ndrt_2020_rocket,
        environment=env,
        rail_length=3.353,
        inclination=90,
        heading=181,
    )

    # Reading data from the flightData (sensors: Raven)
    df = pd.read_csv("data/rockets/NDRT_2020/ndrt_2020_flight_data.csv")

    # convert feet to meters
    df[" Altitude (m-AGL)"] = df[" Altitude (Ft-AGL)"] / 3.28084

    # Calculate the vertical velocity as a derivative of the altitude
    velocity_raven = [0]
    for i in range(1, len(df[" Altitude (m-AGL)"]), 1):
        v = (df[" Altitude (m-AGL)"][i] - df[" Altitude (m-AGL)"][i - 1]) / (
            df[" Time (s)"][i] - df[" Time (s)"][i - 1]
        )
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

    # Filter using Savitzky-Golay filter
    velocity_raven_filt = savgol_filter(velocity_raven, 51, 3)

    # Apogee

    apogee_measured = max(df[" Altitude (m-AGL)"])
    apogee_rocketpy = rocketpy_flight.apogee - rocketpy_flight.env.elevation
    apogee_error = abs(apogee_measured - apogee_rocketpy) / apogee_measured
    assert apogee_error < 0.02  # historical threshold for this flight

    # Max Speed

    max_speed_measured = max(velocity_raven_filt)
    max_speed_rocketpy = rocketpy_flight.max_speed
    max_speed_error = abs(max_speed_measured - max_speed_rocketpy) / max_speed_measured
    assert (max_speed_error) < 0.06

    # Apogee Time

    apogee_time_measured = df.loc[df[" Altitude (Ft-AGL)"].idxmax(), " Time (s)"]
    apogee_time_rocketpy = rocketpy_flight.apogee_time
    apogee_time_error = (
        abs(apogee_time_measured - apogee_time_rocketpy) / apogee_time_rocketpy
    )
    assert apogee_time_error < 0.025
