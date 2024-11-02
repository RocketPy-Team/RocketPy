# %% [markdown]
# # Monte Carlo sensitivity analysis simulation

# %% [markdown]
# This notebook shows how execute Monte Carlo simulations to create
# datasets used in the sensitivity analysis.

# %% [markdown]
# First, let's import the necessary libraries

import datetime

# %%
from rocketpy import Environment, Flight, MonteCarlo, Rocket, SolidMotor
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticNoseCone,
    StochasticParachute,
    StochasticRailButtons,
    StochasticRocket,
    StochasticSolidMotor,
    StochasticTail,
    StochasticTrapezoidalFins,
)

# %% [markdown]
# ## Set Distributions

# %% [markdown]
# The Monte Carlo class allows us to express the parameters uncertainty
# by specifying a probability distribution. We consider two possibilities: either the
# parameter is constant and there is no uncertainty about it, or we propose a normal
# distribution and specify its mean and standard deviation.
#
# In this example, the goal of the sensitivity analysis is to study the rocket, motor, flight and parachute
# parameters influence in the flight outputs (e.g. apogee). The dictionary below defines
# the stochastic parameters along with their mean and standard deviation.

# %%
analysis_parameters = {
    # Rocket properties
    "rocket_mass": {"mean": 14.426, "std": 0.5},
    "rocket_radius": {"mean": 127 / 2000, "std": 1 / 1000},
    # Motor Properties
    "motors_dry_mass": {"mean": 1.815, "std": 1 / 100},
    "motors_grain_density": {"mean": 1815, "std": 50},
    "motors_total_impulse": {"mean": 5700, "std": 50},
    "motors_burn_out_time": {"mean": 3.9, "std": 0.2},
    "motors_nozzle_radius": {"mean": 33 / 1000, "std": 0.5 / 1000},
    "motors_grain_separation": {"mean": 5 / 1000, "std": 1 / 1000},
    "motors_grain_initial_height": {"mean": 120 / 1000, "std": 1 / 100},
    "motors_grain_initial_inner_radius": {"mean": 15 / 1000, "std": 0.375 / 1000},
    "motors_grain_outer_radius": {"mean": 33 / 1000, "std": 0.375 / 1000},
    # Parachutes
    "parachutes_main_cd_s": {"mean": 10, "std": 0.1},
    "parachutes_main_lag": {"mean": 1.5, "std": 0.1},
    "parachutes_drogue_cd_s": {"mean": 1, "std": 0.07},
    "parachutes_drogue_lag": {"mean": 1.5, "std": 0.2},
    # Flight
    "heading": {"mean": 53, "std": 2},
    "inclination": {"mean": 84.7, "std": 1},
}

# %% [markdown]
# ## Create Standard Objects
#

# %% [markdown]
# We will first create a standard RocketPy simulation objects (e.g. Environment, SolidMotor, etc.) to then create the Stochastic objects. All
# deterministic parameters are set to its values, and the stochastic ones are set to the `mean` value defined in the dictionary above.
#

# %%
# Environment

env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
env.set_atmospheric_model(type="Forecast", file="GFS")

# Motor
motor = SolidMotor(
    thrust_source="../../../data/motors/cesaroni/Cesaroni_M1670.eng",
    dry_mass=analysis_parameters["motors_dry_mass"]["mean"],
    nozzle_radius=analysis_parameters["motors_nozzle_radius"]["mean"],
    grain_density=analysis_parameters["motors_grain_density"]["mean"],
    burn_time=analysis_parameters["motors_burn_out_time"]["mean"],
    grain_outer_radius=analysis_parameters["motors_grain_outer_radius"]["mean"],
    grain_initial_inner_radius=analysis_parameters["motors_grain_initial_inner_radius"][
        "mean"
    ],
    grain_initial_height=analysis_parameters["motors_grain_initial_height"]["mean"],
    grain_separation=analysis_parameters["motors_grain_separation"]["mean"],
    dry_inertia=(0.125, 0.125, 0.002),
    grain_number=5,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

# Rocket
rocket = Rocket(
    radius=analysis_parameters["rocket_radius"]["mean"],
    mass=analysis_parameters["rocket_mass"]["mean"],
    inertia=(6.321, 6.321, 0.034),
    power_off_drag="../../../data/rockets/calisto/powerOffDragCurve.csv",
    power_on_drag="../../../data/rockets/calisto/powerOnDragCurve.csv",
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
    airfoil=("../../../data/airfoils/NACA0012-radians.txt", "radians"),
)

tail = rocket.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)
Main = rocket.add_parachute(
    "Main",
    cd_s=analysis_parameters["parachutes_main_cd_s"]["mean"],
    lag=analysis_parameters["parachutes_main_lag"]["mean"],
    trigger=800,
    sampling_rate=105,
    noise=(0, 8.3, 0.5),
)

Drogue = rocket.add_parachute(
    "Drogue",
    cd_s=analysis_parameters["parachutes_drogue_cd_s"]["mean"],
    lag=analysis_parameters["parachutes_drogue_lag"]["mean"],
    trigger="apogee",
    sampling_rate=105,
    noise=(0, 8.3, 0.5),
)

# Flight
test_flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=5,
    inclination=analysis_parameters["inclination"]["mean"],
    heading=analysis_parameters["heading"]["mean"],
)

# %% [markdown]
# ## Create Stochastic Objects

# %% [markdown]
# For each RocketPy object, we will create a ``Stochastic`` counterpart that extends the initial model, allowing us to define the uncertainties of each input parameter. The uncertainty is set as the `std` of the uncertainty dictionary.

# %% [markdown]
# ### Stochastic Environment

# %% [markdown]
# We create a `StochasticEnvironment` to pass to the Monte Carlo class. Our initial goal
# in the sensitivity analysis is to study the influence of motor, rocket, flight
# and parachute parameters in the flight variables. Therefore, the environment is kept
# constant and equals to the prediction made for tomorrow. Note we do not take into
# account the  uncertainty of the prediction.

# %%
stochastic_env = StochasticEnvironment(
    environment=env,
)

stochastic_env.visualize_attributes()

# %% [markdown]
# ### Motor
#

# %% [markdown]
# We can now create a `StochasticSolidMotor` object to define the uncertainties associated with the motor.

# %%
stochastic_motor = StochasticSolidMotor(
    solid_motor=motor,
    dry_mass=analysis_parameters["motors_dry_mass"]["std"],
    grain_density=analysis_parameters["motors_grain_density"]["std"],
    burn_out_time=analysis_parameters["motors_burn_out_time"]["std"],
    nozzle_radius=analysis_parameters["motors_nozzle_radius"]["std"],
    grain_separation=analysis_parameters["motors_grain_separation"]["std"],
    grain_initial_height=analysis_parameters["motors_grain_initial_height"]["std"],
    grain_initial_inner_radius=analysis_parameters["motors_grain_initial_inner_radius"][
        "std"
    ],
    grain_outer_radius=analysis_parameters["motors_grain_outer_radius"]["std"],
    total_impulse=(
        analysis_parameters["motors_total_impulse"]["mean"],
        analysis_parameters["motors_total_impulse"]["std"],
    ),
)
stochastic_motor.visualize_attributes()

# %% [markdown]
# ### Rocket
#

# %% [markdown]
# We can now create a `StochasticRocket` object to define the uncertainties associated with the rocket.

# %%
stochastic_rocket = StochasticRocket(
    rocket=rocket,
    radius=analysis_parameters["rocket_radius"]["std"],
    mass=analysis_parameters["rocket_mass"]["std"],
)
stochastic_rocket.visualize_attributes()

# %% [markdown]
# The `StochasticRocket` still needs to have its aerodynamic surfaces and parachutes added.
# As discussed, we need to set the uncertainties in parachute parameters.

# %%
stochastic_nose_cone = StochasticNoseCone(
    nosecone=nose_cone,
)

stochastic_fin_set = StochasticTrapezoidalFins(
    trapezoidal_fins=fin_set,
)

stochastic_tail = StochasticTail(
    tail=tail,
)

stochastic_rail_buttons = StochasticRailButtons(
    rail_buttons=rail_buttons,
)

stochastic_main = StochasticParachute(
    parachute=Main,
    cd_s=analysis_parameters["parachutes_main_cd_s"]["std"],
    lag=analysis_parameters["parachutes_main_lag"]["std"],
)

stochastic_drogue = StochasticParachute(
    parachute=Drogue,
    cd_s=analysis_parameters["parachutes_drogue_cd_s"]["std"],
    lag=analysis_parameters["parachutes_drogue_lag"]["std"],
)

# %% [markdown]
# Then we must add them to our stochastic rocket, much like we do in the normal Rocket.
#
#

# %%
stochastic_rocket.add_motor(stochastic_motor)
stochastic_rocket.add_nose(stochastic_nose_cone)
stochastic_rocket.add_trapezoidal_fins(
    stochastic_fin_set,
)
stochastic_rocket.add_tail(stochastic_tail)
stochastic_rocket.set_rail_buttons(stochastic_rail_buttons)
stochastic_rocket.add_parachute(stochastic_main)
stochastic_rocket.add_parachute(stochastic_drogue)
stochastic_rocket.visualize_attributes()

# %% [markdown]
#
# ### Flight
#

# %% [markdown]
# The setup is concluded by creating the `StochasticFlight`.

# %%
stochastic_flight = StochasticFlight(
    flight=test_flight,
    inclination=analysis_parameters["inclination"]["std"],
    heading=analysis_parameters["heading"]["std"],
)
stochastic_flight.visualize_attributes()

# %% [markdown]
# ### Run the Monte Carlo Simulations
#

# %% [markdown]
# Finally, we simulate our flights and save the data.

# %%
test_dispersion = MonteCarlo(
    filename="monte_carlo_analysis_outputs/sensitivity_analysis_data",
    environment=stochastic_env,
    rocket=stochastic_rocket,
    flight=stochastic_flight,
)
test_dispersion.simulate(number_of_simulations=100, append=False)

# %% [markdown]
# We give a last check on the variables summary results.

# %%
test_dispersion.prints.all()

# %%
