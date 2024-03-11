import pandas as pd
from numpy.random import normal
import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class SimulationInfo:
    """Dataclass containing sensitivity analysis simulation information"""

    nominal: dict
    parameters_df: pd.DataFrame
    trajectories_df: pd.DataFrame
    target_variables_df: pd.DataFrame


def stochastic_simulation(
    stochastic_parameters: dict,
    number_of_simulations: int = 100,
    ensemble_number: int = 0,
    target_variables: tuple = (
        "out_of_rail_time",
        "out_of_rail_velocity",
        "apogee_time",
        "apogee",
        "apogee_x",
        "apogee_y",
        "t_final",
        "x_impact",
        "y_impact",
        "impact_velocity",
    ),
    random_seed=None,
) -> SimulationInfo:
    """

    Parameters
    ----------
        stochastic_parameters (dict): Dictionary whose keys are the parameters to be
            varied in the sensitivity analysis. The values are also dictionaries
            containing two keys, "mean" and "sd", representing the mean and standard
            value of the parameter for the analysis.

        number_of_simulations (int, optional): Number of simulations. Defaults to 100.

        ensemble_number (int, optional): Ensemble number used in the enviroment.
            Defaults to 0.

        target_variables (tuple, optional): Flight variables saved after each simulation
            for the.
            Defaults to (
                "out_of_rail_time",
                "out_of_rail_velocity",
                "apogee_time",
                "apogee",
                "apogee_x",
                "apogee_y",
                "t_final",
                "x_impact",
                "y_impact",
                "impact_velocity",
            ).

    Returns
    -------
        SimulationInfo: A SimulationInfo dataclass contaning the simulation data
            required in the sensitivity analysis.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    ### Create Parameters dataframe
    parameters_mean = {
        parameter: stochastic_parameters[parameter]["mean"]
        for parameter in stochastic_parameters.keys()
    }
    parameters_sd = {
        parameter: stochastic_parameters[parameter]["sd"]
        for parameter in stochastic_parameters.keys()
    }
    parameters_df = pd.DataFrame(columns=stochastic_parameters.keys())
    n_samples = 0
    while n_samples < number_of_simulations:
        for parameter in stochastic_parameters.keys():
            mean, sd = (
                stochastic_parameters[parameter]["mean"],
                stochastic_parameters[parameter]["sd"],
            )
            parameters_df.loc[n_samples, parameter] = normal(mean, sd)

        # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
        if (
            parameters_df.loc[n_samples, "lag_rec"] < 0
            or parameters_df.loc[n_samples, "lag_se"] < 0
        ):
            continue

        # Update counter
        n_samples += 1

    ###

    ### Setup parachute trigger and enviroment
    # Set up parachutes. This rocket, named Valetudo, only has a drogue chute.
    def drogue_trigger(p, h, y):
        # Check if rocket is going down, i.e. if it has passed the apogee
        vertical_velocity = y[5]
        # Return true to activate parachute once the vertical velocity is negative
        return True if vertical_velocity < 0 else False

    # Update environment object
    # Define basic Environment object
    Env = Environment(date=(2019, 8, 10, 21), latitude=-23.363611, longitude=-48.011389)
    Env.set_elevation(668)
    Env.max_expected_height = 1500
    Env.set_atmospheric_model(
        type="Ensemble",
        file="dispersion_analysis_inputs/LASC2019_reanalysis.nc",
        dictionary="ECMWF",
    )
    Env.select_ensemble_member(ensemble_number)
    ###

    ### Simulate nominal Flight
    nominal_target_variables = {
        target_variable: None for target_variable in target_variables
    }
    # Create motor
    Keron = SolidMotor(
        thrust_source="dispersion_analysis_inputs/thrustCurve.csv",
        burn_time=5.274,
        reshape_thrust_curve=(
            stochastic_parameters["burn_time"]["mean"],
            stochastic_parameters["impulse"]["mean"],
        ),
        nozzle_radius=stochastic_parameters["nozzle_radius"]["mean"],
        throat_radius=stochastic_parameters["throat_radius"]["mean"],
        grain_number=6,
        grain_separation=stochastic_parameters["grain_separation"]["mean"],
        grain_density=stochastic_parameters["grain_density"]["mean"],
        grain_outer_radius=stochastic_parameters["grain_outer_radius"]["mean"],
        grain_initial_inner_radius=stochastic_parameters["grain_initial_inner_radius"][
            "mean"
        ],
        grain_initial_height=stochastic_parameters["grain_initial_height"]["mean"],
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        nozzle_position=stochastic_parameters["nozzle_position"]["mean"],
        grains_center_of_mass_position=stochastic_parameters[
            "grains_center_of_mass_position"
        ]["mean"],
        dry_mass=stochastic_parameters["motor_dry_mass"]["mean"],
        dry_inertia=(
            stochastic_parameters["motor_inertia_11"]["mean"],
            stochastic_parameters["motor_inertia_11"]["mean"],
            stochastic_parameters["motor_inertia_33"]["mean"],
        ),
        center_of_dry_mass_position=stochastic_parameters["motor_dry_mass_position"][
            "mean"
        ],
    )
    # Create rocket
    Valetudo = Rocket(
        radius=stochastic_parameters["radius"]["mean"],
        mass=stochastic_parameters["rocket_mass"]["mean"],
        inertia=(
            stochastic_parameters["rocket_inertia_11"]["mean"],
            stochastic_parameters["rocket_inertia_11"]["mean"],
            stochastic_parameters["rocket_inertia_33"]["mean"],
        ),
        power_off_drag="dispersion_analysis_inputs/Cd_PowerOff.csv",
        power_on_drag="dispersion_analysis_inputs/Cd_PowerOn.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    Valetudo.set_rail_buttons(0.224, -0.93, 30)

    Valetudo.add_motor(Keron, position=0)

    # Edit rocket drag
    Valetudo.power_off_drag *= stochastic_parameters["power_off_drag"]["mean"]
    Valetudo.power_on_drag *= stochastic_parameters["power_on_drag"]["mean"]
    # Add rocket nose, fins and tail
    _ = Valetudo.add_nose(
        length=stochastic_parameters["nose_length"]["mean"],
        kind="vonKarman",
        position=stochastic_parameters["nose_distance_to_CM"]["mean"]
        + stochastic_parameters["nose_length"]["mean"],
    )
    _ = Valetudo.add_trapezoidal_fins(
        n=3,
        span=stochastic_parameters["fin_span"]["mean"],
        root_chord=stochastic_parameters["fin_root_chord"]["mean"],
        tip_chord=stochastic_parameters["fin_tip_chord"]["mean"],
        position=stochastic_parameters["fin_distance_to_CM"]["mean"],
        cant_angle=0,
        airfoil=None,
    )
    # Add parachute
    _ = Valetudo.add_parachute(
        "Drogue",
        cd_s=stochastic_parameters["cd_s_drogue"]["mean"],
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=stochastic_parameters["lag_rec"]["mean"]
        + stochastic_parameters["lag_se"]["mean"],
        noise=(0, 8.3, 0.5),
    )

    # Run trajectory simulation
    try:
        nominal_flight = Flight(
            rocket=Valetudo,
            environment=Env,
            rail_length=stochastic_parameters["rail_length"]["mean"],
            inclination=stochastic_parameters["inclination"]["mean"],
            heading=stochastic_parameters["heading"]["mean"],
            max_time=600,
        )
        for target_variable in target_variables:
            nominal_target_variables[target_variable] = getattr(
                nominal_flight, target_variable
            )
        nominal_target_variables["apogee"] -= Env.elevation
    except Exception as E:
        print(E)
    ###

    ### Create target variables DataFrame
    target_variables_df = pd.DataFrame(columns=target_variables)

    ### Create simulation trajectory DataFrame
    time_delta = 0.05
    n_time_points = (
        int((nominal_flight.t_final - nominal_flight.t_initial) / time_delta) + 1
    )
    nominal_time_steps = np.linspace(
        nominal_flight.t_initial, nominal_flight.t_final, n_time_points
    )
    number_of_time_steps = len(nominal_time_steps)

    simulation_trajectory_df = pd.DataFrame(
        np.full((number_of_time_steps, 1 + 3 + (3 * number_of_simulations)), np.nan),
        columns=[
            "Time",
            "nominal_Flight_x",
            "nominal_Flight_y",
            "nominal_Flight_z",
            *[
                f"Sim_{sim_number}_Flight_{coord}"
                for sim_number in range(1, 1 + number_of_simulations)
                for coord in ["x", "y", "z"]
            ],
        ],
    )
    simulation_trajectory_df["Time"] = nominal_time_steps
    simulation_trajectory_df["nominal_Flight_x"] = nominal_flight.x(nominal_time_steps)
    simulation_trajectory_df["nominal_Flight_y"] = nominal_flight.y(nominal_time_steps)
    simulation_trajectory_df["nominal_Flight_z"] = nominal_flight.z(nominal_time_steps)
    ###

    ### Simulation
    # Iterate over flight rows
    for sim_number, row in tqdm(parameters_df.iterrows(), total=number_of_simulations):
        sim_number += 1
        # Create motor
        Keron = SolidMotor(
            thrust_source="dispersion_analysis_inputs/thrustCurve.csv",
            burn_time=5.274,
            reshape_thrust_curve=(row["burn_time"], row["impulse"]),
            nozzle_radius=row["nozzle_radius"],
            throat_radius=row["throat_radius"],
            grain_number=6,
            grain_separation=row["grain_separation"],
            grain_density=row["grain_density"],
            grain_outer_radius=row["grain_outer_radius"],
            grain_initial_inner_radius=row["grain_initial_inner_radius"],
            grain_initial_height=row["grain_initial_height"],
            interpolation_method="linear",
            coordinate_system_orientation="nozzle_to_combustion_chamber",
            nozzle_position=row["nozzle_position"],
            grains_center_of_mass_position=row["grains_center_of_mass_position"],
            dry_mass=row["motor_dry_mass"],
            dry_inertia=(
                row["motor_inertia_11"],
                row["motor_inertia_11"],
                row["motor_inertia_33"],
            ),
            center_of_dry_mass_position=row["motor_dry_mass_position"],
        )
        # Create rocket
        Valetudo = Rocket(
            radius=row["radius"],
            mass=row["rocket_mass"],
            inertia=(
                row["rocket_inertia_11"],
                row["rocket_inertia_11"],
                row["rocket_inertia_33"],
            ),
            power_off_drag="dispersion_analysis_inputs/Cd_PowerOff.csv",
            power_on_drag="dispersion_analysis_inputs/Cd_PowerOn.csv",
            center_of_mass_without_motor=0,
            coordinate_system_orientation="tail_to_nose",
        )
        Valetudo.set_rail_buttons(0.224, -0.93, 30)

        Valetudo.add_motor(Keron, position=0)

        # Edit rocket drag
        Valetudo.power_off_drag *= row["power_off_drag"]
        Valetudo.power_on_drag *= row["power_on_drag"]
        # Add rocket nose, fins and tail
        _ = Valetudo.add_nose(
            length=row["nose_length"],
            kind="vonKarman",
            position=row["nose_distance_to_CM"] + row["nose_length"],
        )
        _ = Valetudo.add_trapezoidal_fins(
            n=3,
            span=row["fin_span"],
            root_chord=row["fin_root_chord"],
            tip_chord=row["fin_tip_chord"],
            position=row["fin_distance_to_CM"],
            cant_angle=0,
            airfoil=None,
        )
        # Add parachute
        _ = Valetudo.add_parachute(
            "Drogue",
            cd_s=row["cd_s_drogue"],
            trigger=drogue_trigger,
            sampling_rate=105,
            lag=row["lag_rec"] + row["lag_se"],
            noise=(0, 8.3, 0.5),
        )

        # Run trajectory simulation
        try:
            test_flight = Flight(
                rocket=Valetudo,
                environment=Env,
                rail_length=row["rail_length"],
                inclination=row["inclination"],
                heading=row["heading"],
                max_time=600,
            )
            simulation_trajectory_df[f"Sim_{sim_number}_Flight_x"] = test_flight.x(
                nominal_time_steps
            )
            simulation_trajectory_df[f"Sim_{sim_number}_Flight_y"] = test_flight.y(
                nominal_time_steps
            )
            simulation_trajectory_df[f"Sim_{sim_number}_Flight_z"] = test_flight.z(
                nominal_time_steps
            )
            for target_variable in target_variables:
                target_variables_df.loc[sim_number, target_variable] = getattr(
                    test_flight, target_variable
                )
            target_variables_df.loc[sim_number, "apogee"] -= Env.elevation
        except Exception as E:
            print(E)
    ###

    nominal = {
        "parameters_mean": parameters_mean,
        "parameters_sd": parameters_sd,
        "target_variables": nominal_target_variables,
    }
    simulation_info = SimulationInfo(
        nominal=nominal,
        parameters_df=parameters_df,
        trajectories_df=simulation_trajectory_df,
        target_variables_df=target_variables_df,
    )

    return simulation_info
