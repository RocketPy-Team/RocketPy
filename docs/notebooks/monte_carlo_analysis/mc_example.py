import datetime

from rocketpy import Environment, Flight, MonteCarlo, Rocket, SolidMotor, Function
from rocketpy.stochastic import (StochasticNoseCone, StochasticParachute,
                                 StochasticRailButtons, StochasticTail,
                                 StochasticTrapezoidalFins, StochasticSolidMotor)

##### Environment
tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env = Environment(latitude=39.389700, longitude=-8.288964, elevation=113)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
env.set_atmospheric_model(type="Ensemble", file="GEFS")

# mc_env = StochasticEnvironment(
#     environment=env,
#     ensemble_member=list(range(env.num_ensemble_members)),
#     wind_velocity_x_factor=(1.0, 0.33, "normal"),
#     wind_velocity_y_factor=(1.0, 0.33, "normal"),
# )

sto_env_parameters = {
    "environment": env,
    "ensemble_member": list(range(env.num_ensemble_members)),
    "wind_velocity_x_factor": (1.0, 0.33, "normal"),
    "wind_velocity_y_factor": (1.0, 0.33, "normal"),
}

##### Motor
motor = SolidMotor(
    thrust_source="data/motors/Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

mc_motor = StochasticSolidMotor(
    solid_motor=motor,
    thrust_source=[
        "data/motors/Cesaroni_M1670.eng",
        [[0, 6000], [1, 6000], [2, 6000], [3, 6000], [4, 6000]],
        Function([[0, 6000], [1, 6000], [2, 6000], [3, 6000], [4, 6000]]),
    ],
    burn_start_time=(0, 0.1),
    grains_center_of_mass_position=0.001,
    grain_density=50,
    grain_separation=1 / 1000,
    grain_initial_height=1 / 1000,
    grain_initial_inner_radius=0.375 / 1000,
    grain_outer_radius=0.375 / 1000,
    total_impulse=(6500, 1000),
    throat_radius=0.5 / 1000,
    nozzle_radius=0.5 / 1000,
    nozzle_position=0.001,
)


# sto_motor_parameters = {
#     "solid_motor": motor,
#     "thrust_source": [
#         "data/motors/Cesaroni_M1670.eng",
#         [[0, 6000], [1, 6000], [2, 6000], [3, 6000], [4, 6000]],
#     ],
#     "burn_start_time": (0, 0.1),
#     "grains_center_of_mass_position": 0.001,
#     "grain_density": 50,
#     "grain_separation": 1 / 1000,
#     "grain_initial_height": 1 / 1000,
#     "grain_initial_inner_radius": 0.375 / 1000,
#     "grain_outer_radius": 0.375 / 1000,
#     "total_impulse": (6500, 1000),
#     "throat_radius": 0.5 / 1000,
#     "nozzle_radius": 0.5 / 1000,
#     "nozzle_position": 0.001,
# }

##### Rocket
rocket = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag="data/calisto/powerOffDragCurve.csv",
    power_on_drag="data/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

rail_buttons = rocket.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.618,
    angular_position=45,
)

rocket.add_motor(motor, position=-1.255)

nose_cone = rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)

fin_set = rocket.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0.5,
    airfoil=("data/calisto/NACA0012-radians.csv", "radians"),
)

tail = rocket.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)

Main = rocket.add_parachute(
    "Main",
    cd_s=10.0,
    trigger=800,
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

Drogue = rocket.add_parachute(
    "Drogue",
    cd_s=1.0,
    trigger="apogee",
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

sto_rocket_parameters = {
    "rocket": rocket,
    "radius": 0.0127 / 2000,
    "mass": (15.426, 0.5, "normal"),
    "inertia_11": (6.321, 0),
    "inertia_22": 0.01,
    "inertia_33": 0.01,
    "center_of_mass_without_motor": 0,
}

mc_nose_cone = StochasticNoseCone(
    nosecone=nose_cone,
    length=0.001,
)

mc_fin_set = StochasticTrapezoidalFins(
    trapezoidal_fins=fin_set,
    root_chord=0.0005,
    tip_chord=0.0005,
    span=0.0005,
)

mc_tail = StochasticTail(
    tail=tail,
    top_radius=0.001,
    bottom_radius=0.001,
    length=0.001,
)

mc_rail_buttons = StochasticRailButtons(
    rail_buttons=rail_buttons, buttons_distance=0.001
)

mc_main = StochasticParachute(
    parachute=Main,
    cd_s=0.1,
    lag=0.1,
)

mc_drogue = StochasticParachute(
    parachute=Drogue,
    cd_s=0.07,
    lag=0.2,
)

sto_rocket_parameters["motor"] = (mc_motor, 0.001)
sto_rocket_parameters["nose"] = (mc_nose_cone, (1.134, 0.001))
sto_rocket_parameters["trapezoidal_fins"] = (mc_fin_set, (0.001, "normal"))
sto_rocket_parameters["rail_buttons"] = (mc_rail_buttons, (0.001, "normal")) 
sto_rocket_parameters["tail"] = mc_tail
sto_rocket_parameters["parachute_main"] = mc_main
sto_rocket_parameters["parachute_drogue"] = mc_drogue


##### Flight
test_flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=5,
    inclination=84,
    heading=133,
)

sto_flight_parameters = {
    "flight": test_flight,
    "inclination": (84.7, 1),
    "heading": (53, 2),
}


##### Step 2: Starting the Monte Carlo Simulations
test_dispersion = MonteCarlo(
    filename="monte_carlo_analysis_outputs/monte_carlo_class_example",
    environment_params=sto_env_parameters,
    rocket_params=sto_rocket_parameters,
    flight_params=sto_flight_parameters,
)

##### Running the Monte Carlo Simulations
test_dispersion.simulate(number_of_simulations=50, append=False, parallel=True)
