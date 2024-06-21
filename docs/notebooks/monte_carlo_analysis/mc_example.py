import datetime

from rocketpy import (Environment, Flight, Function, MonteCarlo, Rocket,
                      SolidMotor)
from rocketpy.stochastic import (StochasticEnvironment, StochasticFlight,
                                 StochasticNoseCone, StochasticParachute,
                                 StochasticRailButtons, StochasticRocket,
                                 StochasticSolidMotor, StochasticTail,
                                 StochasticTrapezoidalFins)
def test_run(n_workers, n_sim, append_mode, light_mode):
    initial_time = datetime.datetime.now()
    
    env = Environment(latitude=39.389700, longitude=-8.288964, elevation=113)
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
    env.set_atmospheric_model(type="Ensemble", file="GEFS")

    mc_env = StochasticEnvironment(
        environment=env,
        ensemble_member=list(range(env.num_ensemble_members)),
        wind_velocity_x_factor=(1.0, 0.33, "normal"),
        wind_velocity_y_factor=(1.0, 0.33, "normal"),
    )

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

    mc_rocket = StochasticRocket(
        rocket=rocket,
        radius=0.0127 / 2000,
        mass=(15.426, 0.5, "normal"),
        inertia_11=(6.321, 0),
        inertia_22=0.01,
        inertia_33=0.01,
        center_of_mass_without_motor=0,
    )

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

    mc_rocket.add_motor(mc_motor, position=0.001)
    mc_rocket.add_nose(mc_nose_cone, position=(1.134, 0.001))
    mc_rocket.add_trapezoidal_fins(mc_fin_set, position=(0.001, "normal"))
    mc_rocket.add_tail(mc_tail)
    mc_rocket.set_rail_buttons(mc_rail_buttons, lower_button_position=(0.001, "normal"))
    mc_rocket.add_parachute(mc_main)
    mc_rocket.add_parachute(mc_drogue)

    test_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5,
        inclination=84,
        heading=133,
    )

    mc_flight = StochasticFlight(
        flight=test_flight,
        inclination=(84.7, 1),
        heading=(53, 2),
    )

    test_dispersion = MonteCarlo(
        filename="monte_carlo_class_example",
        environment=mc_env,
        rocket=mc_rocket,
        flight=mc_flight,
        batch_path="/home/sorban/Documents/code/RocketPy/mc_simulations",
    )

    if n_workers == 0:
        test_dispersion.simulate(
            number_of_simulations=n_sim, append=append_mode, light_mode=light_mode, parallel=False
        )
    else:
        test_dispersion.simulate(
            number_of_simulations=n_sim, append=append_mode, light_mode=light_mode, parallel=True, n_workers=n_workers
        )
        
    end_time = datetime.datetime.now()
    elapsed_time = end_time - initial_time
    return elapsed_time.total_seconds()