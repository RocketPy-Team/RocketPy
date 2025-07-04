# Bella Lui Kaltbrunn Mission from ERT (EPFL Rocket Team)
# Permission to use flight data given by Antoine Scardigli, 2020

# Importing libraries
import matplotlib as mpl
import numpy as np
from scipy.signal import savgol_filter

from rocketpy import Environment, Flight, Function, Rocket, SolidMotor

mpl.rc("figure", max_open_warning=0)  # Prevent matplotlib warnings


def test_bella_lui_rocket_data_asserts_acceptance():
    # Defining all parameters
    parameters = {
        # Mass Details
        "rocket_mass": (18.227 - 1, 0.010),  # 1.373 = propellant mass
        # propulsion details
        "impulse": (2157, 0.03 * 2157),
        "burn_time": (2.43, 0.1),
        "nozzle_radius": (44.45 / 1000, 0.001),
        "throat_radius": (21.4376 / 1000, 0.001),
        "grain_separation": (3 / 1000, 1 / 1000),
        "grain_density": (782.4, 30),
        "grain_outer_radius": (85.598 / 2000, 0.001),
        "grain_initial_inner_radius": (33.147 / 1000, 0.002),
        "grain_initial_height": (152.4 / 1000, 0.001),
        # Aerodynamic Details
        "inertia_i": (0.78267, 0.03 * 0.78267),
        "inertia_z": (0.064244, 0.03 * 0.064244),
        "radius": (156 / 2000, 0.001),
        "distance_rocket_nozzle": (-1.1356, 0.100),
        "distance_rocket_propellant": (-1, 0.100),
        "power_off_drag": (1, 0.05),
        "power_on_drag": (1, 0.05),
        "nose_length": (0.242, 0.001),
        "nose_distance_to_cm": (1.3, 0.100),
        "fin_span": (0.200, 0.001),
        "fin_root_chord": (0.280, 0.001),
        "fin_tip_chord": (0.125, 0.001),
        "fin_distance_to_cm": (-0.75, 0.100),
        "tail_top_radius": (156 / 2000, 0.001),
        "tail_bottom_radius": (135 / 2000, 0.001),
        "tail_length": (0.050, 0.001),
        "tail_distance_to_cm": (-1.0856, 0.001),
        # Launch and Environment Details
        "wind_direction": (0, 5),
        "wind_speed": (1, 0.05),
        "inclination": (89, 1),
        "heading": (45, 5),
        "rail_length": (4.2, 0.001),
        # Parachute Details
        "CdS_drogue": (np.pi / 4, 0.20 * np.pi / 4),
        "lag_rec": (1, 0.020),
    }

    # Environment conditions
    env = Environment(
        gravity=9.81,
        latitude=47.213476,
        longitude=9.003336,
        date=(2020, 2, 22, 13),
        elevation=407,
    )
    env.set_atmospheric_model(
        type="Reanalysis",
        file="data/weather/bella_lui_weather_data_ERA5.nc",
        dictionary="ECMWF",
    )
    env.max_expected_height = 2000

    # Motor Information
    K828FJ = SolidMotor(
        thrust_source="data/motors/aerotech/AeroTech_K828FJ.eng",
        burn_time=parameters.get("burn_time")[0],
        dry_mass=1,
        dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=0,
        grains_center_of_mass_position=parameters.get("distance_rocket_propellant")[0],
        grain_number=3,
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
    # rocket information
    BellaLui = Rocket(
        radius=parameters.get("radius")[0],
        mass=parameters.get("rocket_mass")[0],
        inertia=(
            parameters.get("inertia_i")[0],
            parameters.get("inertia_i")[0],
            parameters.get("inertia_z")[0],
        ),
        power_off_drag=0.43,
        power_on_drag=0.43,
        center_of_mass_without_motor=0,
    )
    BellaLui.set_rail_buttons(0.1, -0.5)
    BellaLui.add_motor(K828FJ, parameters.get("distance_rocket_nozzle")[0])
    BellaLui.add_nose(
        length=parameters.get("nose_length")[0],
        kind="tangent",
        position=parameters.get("nose_distance_to_cm")[0]
        + parameters.get("nose_length")[0],
    )
    BellaLui.add_trapezoidal_fins(
        3,
        span=parameters.get("fin_span")[0],
        root_chord=parameters.get("fin_root_chord")[0],
        tip_chord=parameters.get("fin_tip_chord")[0],
        position=parameters.get("fin_distance_to_cm")[0],
    )
    BellaLui.add_tail(
        top_radius=parameters.get("tail_top_radius")[0],
        bottom_radius=parameters.get("tail_bottom_radius")[0],
        length=parameters.get("tail_length")[0],
        position=parameters.get("tail_distance_to_cm")[0],
    )

    # Parachute set-up
    def drogue_trigger(p, h, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    BellaLui.add_parachute(
        "Drogue",
        cd_s=parameters.get("CdS_drogue")[0],
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=parameters.get("lag_rec")[0],
        noise=(0, 8.3, 0.5),
    )

    # Define aerodynamic drag coefficients
    BellaLui.power_off_drag = Function(
        [
            (0.01, 0.51),
            (0.02, 0.46),
            (0.04, 0.43),
            (0.28, 0.43),
            (0.29, 0.44),
            (0.45, 0.44),
            (0.49, 0.46),
        ],
        "Mach Number",
        "Drag Coefficient with Power Off",
        "linear",
        "constant",
    )
    BellaLui.power_on_drag = Function(
        [
            (0.01, 0.51),
            (0.02, 0.46),
            (0.04, 0.43),
            (0.28, 0.43),
            (0.29, 0.44),
            (0.45, 0.44),
            (0.49, 0.46),
        ],
        "Mach Number",
        "Drag Coefficient with Power On",
        "linear",
        "constant",
    )
    BellaLui.power_off_drag *= parameters.get("power_off_drag")[0]
    BellaLui.power_on_drag *= parameters.get("power_on_drag")[0]

    # Flight
    test_flight = Flight(
        rocket=BellaLui,
        environment=env,
        rail_length=parameters.get("rail_length")[0],
        inclination=parameters.get("inclination")[0],
        heading=parameters.get("heading")[0],
    )

    # Comparison with Real Data
    flight_data = np.loadtxt(
        "data/rockets/EPFL_Bella_Lui/bella_lui_flight_data_filtered.csv",
        skiprows=1,
        delimiter=",",
        usecols=(2, 3, 4),
    )
    time_kalt = flight_data[:573, 0]
    altitude_kalt = flight_data[:573, 1]
    vert_vel_kalt = flight_data[:573, 2]

    # Make sure that all vectors have the same length
    time_rcp = []
    altitude_rcp = []
    velocity_rcp = []
    acceleration_rcp = []
    i = 0
    while i <= int(test_flight.t_final):
        time_rcp.append(i)
        altitude_rcp.append(test_flight.z(i) - test_flight.env.elevation)
        velocity_rcp.append(test_flight.vz(i))
        acceleration_rcp.append(test_flight.az(i))
        i += 0.005

    time_rcp.append(test_flight.t_final)
    altitude_rcp.append(0)
    velocity_rcp.append(test_flight.vz(test_flight.t_final))
    acceleration_rcp.append(test_flight.az(test_flight.t_final))

    # Acceleration comparison (will not be used in our publication)

    # Calculate the acceleration as a velocity derivative
    acceleration_kalt = [0]
    for i in range(1, len(vert_vel_kalt), 1):
        acc = (vert_vel_kalt[i] - vert_vel_kalt[i - 1]) / (
            time_kalt[i] - time_kalt[i - 1]
        )
        acceleration_kalt.append(acc)

    acceleration_kalt_filt = savgol_filter(acceleration_kalt, 51, 3)  # Filter our data

    apogee_time_measured = time_kalt[np.argmax(altitude_kalt)]
    apogee_time_simulated = test_flight.apogee_time

    apogee_error_threshold = 0.015
    apogee_error = abs(
        max(altitude_kalt) - test_flight.apogee + test_flight.env.elevation
    ) / max(altitude_kalt)
    assert apogee_error < apogee_error_threshold, (
        f"Apogee altitude error exceeded the threshold. "
        f"Expected the error to be less than {apogee_error_threshold * 100}%, "
        f"but got an error of {apogee_error * 100:.1f}%."
    )
    assert abs(max(velocity_rcp) - max(vert_vel_kalt)) / max(vert_vel_kalt) < 0.06
    assert (
        abs(max(acceleration_rcp) - max(acceleration_kalt_filt))
        / max(acceleration_kalt_filt)
        < 0.05
    )
    assert (
        abs(apogee_time_measured - apogee_time_simulated) / apogee_time_simulated < 0.02
    )
    # Guarantee the impact velocity is within 30% of the real data.
    # Use the last 5 real points to avoid outliers
    assert (
        abs(test_flight.impact_velocity - np.mean(vert_vel_kalt[-5:]))
        / abs(test_flight.impact_velocity)
        < 0.30
    )
