# Valetudo Flight from Projeto Jupiter
# Launched at LASC 2019
# Permission to use flight data given by Projeto Jupiter, 2020

# Importing libraries
from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
from scipy.signal import savgol_filter
import numpy as np
import os

# Import parachute trigger algorithm named SisRec
if os.name == "nt":
    from SisRecWindows import SisRec
else:
    from SisRecLinux import SisRec

# Defining all parameters
analysis_parameters = {
    # Mass Details
    "rocket_mass": (8.257, 0.001),
    # Propulsion Details
    "impulse": (1415.15, 35.3),
    "burn_time": (5.274, 1),
    "nozzle_radius": (21.642 / 1000, 0.5 / 1000),
    "throat_radius": (8 / 1000, 0.5 / 1000),
    "grain_separation": (6 / 1000, 1 / 1000),
    "grain_density": (1707, 50),
    "grain_outer_radius": (21.4 / 1000, 0.375 / 1000),
    "grain_initial_inner_radius": (9.65 / 1000, 0.375 / 1000),
    "grain_initial_height": (120 / 1000, 1 / 1000),
    # Aerodynamic Details
    "inertia_i": (3.675, 0.03675),
    "inertia_z": (0.007, 0.00007),
    "radius": (40.45 / 1000, 0.001),
    "distance_rocket_nozzle": (-1.024, 0.001),
    "distance_rocket_propellant": (-0.571, 0.001),
    "power_off_drag": (0.9081 / 1.05, 0.033),
    "power_on_drag": (0.9081 / 1.05, 0.033),
    "nose_length": (0.274, 0.001),
    "nose_distance_to_cm": (1.134, 0.001),
    "fin_span": (0.077, 0.0005),
    "fin_root_chord": (0.058, 0.0005),
    "fin_tip_chord": (0.018, 0.0005),
    "fin_distance_to_cm": (-0.906, 0.001),
    # Launch and Environment Details
    "wind_direction": (0, 2),
    "wind_speed": (1, 0.033),
    "inclination": (84.7, 1),
    "heading": (53, 2),
    "rail_length": (5.7, 0.0005),
    # "ensemble_member": list(range(10)),
    # Parachute Details
    "CdS_drogue": (0.349 * 1.3, 0.07),
    "lag_rec": (1, 0.5),
    # Electronic Systems Details
    "lag_se": (0.73, 0.16),
}

# Environment conditions
env = Environment(
    gravity=9.8,
    date=(2019, 8, 10, 21),
    latitude=-23.363611,
    longitude=-48.011389,
)
env.set_elevation(668)
env.max_expected_height = 1500
env.set_atmospheric_model(
    type="Reanalysis",
    file="tests/fixtures/acceptance/PJ_Valetudo/valetudo_weather_data_ERA5.nc",
    dictionary="ECMWF",
)

# Create motor
Keron = SolidMotor(
    thrust_source="tests/fixtures/acceptance/PJ_Valetudo/valetudo_motor_Keron.csv",
    burn_time=5.274,
    grains_center_of_mass_position=analysis_parameters.get(
        "distance_rocket_propellant"
    )[0],
    reshape_thrust_curve=(
        analysis_parameters.get("burn_time")[0],
        analysis_parameters.get("impulse")[0],
    ),
    nozzle_radius=analysis_parameters.get("nozzle_radius")[0],
    throat_radius=analysis_parameters.get("throat_radius")[0],
    grain_number=6,
    grain_separation=analysis_parameters.get("grain_separation")[0],
    grain_density=analysis_parameters.get("grain_density")[0],
    grain_outer_radius=analysis_parameters.get("grain_outer_radius")[0],
    grain_initial_inner_radius=analysis_parameters.get("grain_initial_inner_radius")[0],
    grain_initial_height=analysis_parameters.get("grain_initial_height")[0],
    interpolation_method="linear",
    nozzle_position=analysis_parameters.get("distance_rocket_nozzle")[0],
)

# Create rocket
Valetudo = Rocket(
    motor=Keron,
    radius=analysis_parameters.get("radius")[0],
    mass=analysis_parameters.get("rocket_mass")[0],
    inertia=(
        analysis_parameters.get("inertia_i")[0],
        analysis_parameters.get("inertia_i")[0],
        analysis_parameters.get("inertia_z")[0],
    ),
    power_off_drag="tests/fixtures/acceptance/PJ_Valetudo/valetudo_drag_power_off.csv",
    power_on_drag="tests/fixtures/acceptance/PJ_Valetudo/valetudo_drag_power_on.csv",
)
Valetudo.power_off_drag *= analysis_parameters.get("power_off_drag")[0]
Valetudo.power_on_drag *= analysis_parameters.get("power_on_drag")[0]
Valetudo.add_motor(Keron, analysis_parameters.get("distance_rocket_nozzle")[0])
nosecone = Valetudo.add_nose(
    length=analysis_parameters.get("nose_length")[0],
    kind="vonKarman",
    position=analysis_parameters.get("nose_distance_to_cm")[0]
    + analysis_parameters.get("nose_length")[0],
)
fin_set = Valetudo.add_trapezoidal_fins(
    n=3,
    root_chord=analysis_parameters.get("fin_root_chord")[0],
    tip_chord=analysis_parameters.get("fin_tip_chord")[0],
    span=analysis_parameters.get("fin_span")[0],
    position=analysis_parameters.get("fin_distance_to_cm")[0],
)
Valetudo.set_rail_buttons(0.224, -0.93, 30)

# Set up parachutes
sis_rec_drogue = SisRec.SisRecSt(0.8998194205245451, 0.2)


def drogue_trigger(p, h, y):
    return True if sis_rec_drogue.update(p / 100000) == 2 else False


drogue = Valetudo.add_parachute(
    "Drogue",
    cd_s=analysis_parameters["CdS_drogue"][0],
    trigger=drogue_trigger,
    sampling_rate=105,
    lag=analysis_parameters["lag_rec"][0] + analysis_parameters["lag_se"][0],
    noise=(0, 8.3, 0.5),
)
# Prepare parachutes
sis_rec_drogue.reset()
sis_rec_drogue.enable()

test_flight = Flight(
    rocket=Valetudo,
    environment=env,
    rail_length=analysis_parameters.get("rail_length")[0],
    inclination=analysis_parameters.get("inclination")[0],
    heading=analysis_parameters.get("heading")[0],
    max_time=600,
)
test_flight.post_process()

# Print summary
test_flight.info()
