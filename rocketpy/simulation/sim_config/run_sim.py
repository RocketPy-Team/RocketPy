import warnings

from rocketpy.simulation.flight import Flight
from rocketpy import Environment, Rocket

from rocketpy.simulation.sim_config.deserializer import motor_deserializer


def run_flight(parameters: dict, env: Environment) -> Flight:
    motor = motor_deserializer(parameters["motors"])

    rocket = Rocket(
        radius=parameters["rocket"]["radius"],
        mass=parameters["rocket"]["mass"],
        inertia=parameters["rocket"]["inertia"],
        power_off_drag=parameters["rocket"]["drag_coefficient_power_off"],
        power_on_drag=parameters["rocket"]["drag_coefficient_power_on"],
        center_of_mass_without_motor=parameters["rocket"][
            "center_of_mass_without_motor"
        ],
        coordinate_system_orientation=parameters["rocket"][
            "coordinate_system_orientation"
        ],
    )

    # Add rocket nose, fins and tail and motor
    # adopting 'tail_to_nose' coordinate system
    rocket.add_motor(motor, parameters["motors"]["position"])

    try:
        rocket.add_nose(
            length=parameters["nosecones"]["nose_length"],
            kind=parameters["nosecones"]["nose_kind"],
            position=parameters["nosecones"]["nose_distance_to_tail"],
        )
    except KeyError as e:
        print(f"No nosecone found: {e}. Skipping nosecone addition.")

    try:
        rocket.set_rail_buttons(
            upper_button_position=parameters["rail_buttons"][
                "rail_button_dist_to_tail1"
            ],
            lower_button_position=parameters["rail_buttons"][
                "rail_button_dist_to_tail2"
            ],
        )
    except KeyError as e:
        print(f"No rail buttons found: {e}. Skipping rail buttons addition.")

    try:
        for idx, _ in enumerate(
            filter(lambda x: "tail_top_radius" in x, parameters["tails"].keys())
        ):
            rocket.add_tail(
                top_radius=parameters["tails"][f"tail_top_radius{idx}"],
                bottom_radius=parameters["tails"][f"tail_bottom_radius{idx}"],
                length=parameters["tails"][f"tail_length{idx}"],
                position=parameters["tails"][f"tail_distance_to_tail{idx}"],
            )
    except KeyError as e:
        print(f"No tail found: {e}. Skipping tail addition.")

    if len(parameters["trapezoidal_fins"]) != 0:
        for idx, _ in enumerate(
            filter(lambda x: "fin_set_" in x, parameters["trapezoidal_fins"].keys())
        ):
            rocket.add_trapezoidal_fins(
                n=parameters["trapezoidal_fins"][f"fin_set_{idx}"][f"fin_n"],
                span=parameters["trapezoidal_fins"][f"fin_set_{idx}"][f"fin_span"],
                root_chord=parameters["trapezoidal_fins"][f"fin_set_{idx}"][
                    f"fin_root_chord"
                ],
                tip_chord=parameters["trapezoidal_fins"][f"fin_set_{idx}"][
                    f"fin_tip_chord"
                ],
                position=parameters["trapezoidal_fins"][f"fin_set_{idx}"][
                    f"fin_distance_to_tail"
                ],
                sweep_length=parameters["trapezoidal_fins"][f"fin_set_{idx}"][
                    f"fin_sweep_length"
                ],
                # sweep_angle=parameters["trapezoidal_fins"][f"fin_set_{idx}"][f"fin_sweep_angle"],
                # radius=0,
                cant_angle=0,
                airfoil=None,
            )

    if len(parameters["elliptical_fins"]) != 0:
        raise ValueError("Elliptical fins not implemented yet.")

    if len(parameters["parachutes"]) == 0:
        raise Warning("No parachutes found. Ballistic flight")

    else:
        count_chute = 0

        for idx, _ in enumerate(
            filter(lambda x: "chute_" in x, parameters["parachutes"].keys())
        ):

            # Add parachute
            if (
                parameters["parachutes"][f"chute_{idx}"]["name"] == "Drogue"
                or parameters["parachutes"][f"chute_{idx}"]["name"] == "drogue"
            ):
                drogue_trigger = parameters["parachutes"][f"chute_{idx}"]["trigger"]
                Drogue = rocket.add_parachute(
                    parameters["parachutes"][f"chute_{idx}"]["name"],
                    cd_s=parameters["parachutes"][f"chute_{idx}"][f"cds"],
                    trigger=drogue_trigger,  # it will be added by exec()
                    sampling_rate=parameters["parachutes"][f"chute_{idx}"][
                        "sampling_rate"
                    ],
                    lag=parameters["parachutes"][f"chute_{idx}"]["lag"]
                    + 0.001,  # 0.001 adicionado para evitar warnings
                    noise=parameters["parachutes"][f"chute_{idx}"]["noise"],
                )

            elif (
                parameters["parachutes"][f"chute_{idx}"]["name"] == "Main"
                or parameters["parachutes"][f"chute_{idx}"]["name"] == "main"
            ):
                Main = rocket.add_parachute(
                    parameters["parachutes"][f"chute_{idx}"]["name"],
                    cd_s=parameters["parachutes"][f"chute_{idx}"][f"cds"],
                    trigger=parameters["parachutes"][f"chute_{idx}"][
                        "trigger"
                    ],  # it will be added by exec()
                    sampling_rate=parameters["parachutes"][f"chute_{idx}"][
                        "sampling_rate"
                    ],
                    lag=parameters["parachutes"][f"chute_{idx}"]["lag"]
                    + 0.001,  # 0.001 adicionado para evitar warnings
                    noise=parameters["parachutes"][f"chute_{idx}"]["noise"],
                )
                main_idx = idx

            else:
                Chute = rocket.add_parachute(
                    parameters["parachutes"][f"chute_{idx}"]["name"],
                    cd_s=parameters["parachutes"][f"chute_{idx}"][f"cds"],
                    trigger=parameters["parachutes"][f"chute_{idx}"][
                        "trigger"
                    ],  # it will be added by exec()
                    sampling_rate=parameters["parachutes"][f"chute_{idx}"][
                        "sampling_rate"
                    ],
                    lag=parameters["parachutes"][f"chute_{idx}"]["lag"]
                    + 0.001,  # 0.001 adicionado para evitar warnings
                    noise=parameters["parachutes"][f"chute_{idx}"]["noise"],
                )
                warn_string = f"Parachute name '{parameters['parachutes'][f'chute_{idx}']['name']}' not recognized, but still added to the rocket."
                warnings.warn(warn_string)

            count_chute += 1

    TestFlight = Flight(
        rocket=rocket,
        environment=env,
        inclination=parameters["flight"]["inclination"],
        heading=parameters["flight"]["heading"],
        rail_length=parameters["flight"]["rail_length"],
    )

    return TestFlight
