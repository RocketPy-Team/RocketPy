import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from rocketpy.mathutils.function import Function
from rocketpy.rocket.aero_surface import (
    EllipticalFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)

from rocketpy.simulation.sim_config.serializer import (
    function_serializer,
    motor_serializer,
)


def flight2json_v1(flights, name, path):
    path -= Path(path)
    if not isinstance(flights, Iterable) or len(flights) == 1:
        flightv1_serializer(flights, name, path)
    else:
        for n_stage, flight in enumerate(flights):
            flightv1_serializer(flight, name, path / f"stage{n_stage}")


def flightv1_serializer(flight, name, save_path: str = None, return_dict=False):
    save_path = Path(save_path) if save_path is not None else Path.cwd()
    """Converts a flight object to a dictionary"""

    ################# Create dictionary #################
    # Create dictionary based on new structure
    obj_dict = {
        "environment": {},
        "flight": {},
        "id": {},
        "motors": {},
        "nosecones": {},
        "parachutes": {},
        "rail_buttons": {},
        "rocket": {},
        "tails": {},
        "trapezoidal_fins": {},
        "elliptical_fins": {},
    }

    # Save environment parameters
    obj_dict["environment"]["elevation"] = flight.env.elevation
    obj_dict["flight"]["rail_length"] = flight.rail_length
    obj_dict["environment"]["date"] = str(flight.env.local_date)[0:19]
    obj_dict["environment"]["latitude"] = flight.env.latitude
    obj_dict["environment"]["longitude"] = flight.env.longitude

    obj_dict["motors"] = motor_serializer(flight.rocket.motor)
    obj_dict["motors"]["position"] = flight.rocket.motor_position

    obj_dict["rocket"]["mass"] = flight.rocket.mass
    obj_dict["rocket"]["inertia"] = (
        flight.rocket.I_11_without_motor,
        flight.rocket.I_22_without_motor,
        flight.rocket.I_33_without_motor,
        flight.rocket.I_12_without_motor,
        flight.rocket.I_13_without_motor,
        flight.rocket.I_23_without_motor,
    )
    obj_dict["rocket"]["radius"] = flight.rocket.radius
    obj_dict["rocket"][
        "center_of_mass_without_motor"
    ] = flight.rocket.center_of_mass_without_motor
    obj_dict["rocket"][
        "coordinate_system_orientation"
    ] = flight.rocket.coordinate_system_orientation

    rb2_pos = flight.rocket.rail_buttons.get_tuple_by_type(RailButtons)[0].position
    rb1_pos = (
        rb2_pos
        + flight.rocket.rail_buttons.get_tuple_by_type(RailButtons)[0][
            0
        ].buttons_distance
    )

    obj_dict["rail_buttons"]["rail_button_dist_to_tail1"] = rb1_pos
    obj_dict["rail_buttons"]["rail_button_dist_to_tail2"] = rb2_pos

    count_trapezoidal_fin = 0
    count_elliptical_fin = 0
    count_tail = 0
    # add aerodynamic surfaces
    for surface in flight.rocket.aerodynamic_surfaces:
        # Treat each surface differently
        if isinstance(surface[0], NoseCone):
            pos = flight.rocket.aerodynamic_surfaces.get_tuple_by_type(NoseCone)[
                0
            ].position

            obj_dict["nosecones"]["nose_length"] = surface[0]._length
            obj_dict["nosecones"]["nose_kind"] = surface[0].kind
            obj_dict["nosecones"]["nose_distance_to_tail"] = pos
            obj_dict["nosecones"]["nose_rocket_radius"] = surface[0]._rocket_radius

        elif isinstance(surface[0], TrapezoidalFins):
            pos = flight.rocket.aerodynamic_surfaces.get_tuple_by_type(TrapezoidalFins)[
                count_trapezoidal_fin
            ].position

            obj_dict["trapezoidal_fins"][
                "fin_set_{}".format(count_trapezoidal_fin)
            ] = {}
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_n"
            ] = surface[0].n
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_root_chord"
            ] = surface[0].root_chord
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_tip_chord"
            ] = surface[0].tip_chord
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_span"
            ] = surface[0].span
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_distance_to_tail"
            ] = pos
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_rocket_radius"
            ] = surface[0].rocket_radius
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_cant_angle"
            ] = surface[0].cant_angle
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_sweep_angle"
            ] = surface[0].sweep_angle
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_sweep_length"
            ] = surface[0].sweep_length
            obj_dict["trapezoidal_fins"]["fin_set_{}".format(count_trapezoidal_fin)][
                "fin_airfoil"
            ] = surface[0].airfoil
            count_trapezoidal_fin += 1

        elif isinstance(surface[0], EllipticalFins):
            pos = flight.rocket.aerodynamic_surfaces.get_tuple_by_type(EllipticalFins)[
                count_elliptical_fin
            ].position

            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)] = {}
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_n"
            ] = surface[0].n
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_root_chord"
            ] = surface[0].rootChord
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_span"
            ] = surface[0].span
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_distance_to_tail"
            ] = pos
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_rocket_radius"
            ] = surface[0].rocketRadius
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_cant_angle"
            ] = surface[0].cantAngle
            obj_dict["elliptical_fins"]["fin_set_{}".format(count_elliptical_fin)][
                "fin_airfoil"
            ] = surface[0].airfoil
            count_elliptical_fin += 1

        elif isinstance(surface[0], Tail):
            pos = flight.rocket.aerodynamic_surfaces.get_tuple_by_type(Tail)[
                count_tail
            ].position

            obj_dict["tails"]["tail_top_radius{}".format(count_tail)] = surface[
                0
            ].top_radius
            obj_dict["tails"]["tail_bottom_radius{}".format(count_tail)] = surface[
                0
            ].bottom_radius
            obj_dict["tails"]["tail_length{}".format(count_tail)] = surface[0].length
            obj_dict["tails"]["tail_distance_to_tail{}".format(count_tail)] = pos
            obj_dict["tails"]["tail_rocket_radius{}".format(count_tail)] = surface[
                0
            ].rocket_radius
            count_tail += 1

        else:
            raise ValueError("Invalid surface type")

    count_chute = 0
    # add parachutes
    for chute in flight.rocket.parachutes:
        obj_dict["parachutes"]["chute_{}".format(count_chute)] = {}
        obj_dict["parachutes"]["chute_{}".format(count_chute)]["name"] = chute.name
        obj_dict["parachutes"]["chute_{}".format(count_chute)]["cds"] = chute.cd_s
        obj_dict["parachutes"]["chute_{}".format(count_chute)][
            "sampling_rate"
        ] = chute.sampling_rate
        obj_dict["parachutes"]["chute_{}".format(count_chute)]["lag"] = chute.lag
        obj_dict["parachutes"]["chute_{}".format(count_chute)]["noise"] = (
            chute.noise_bias,
            chute.noise_deviation,
            chute.noise_corr[0],
        )
        obj_dict["parachutes"]["chute_{}".format(count_chute)][
            "trigger"
        ] = chute.trigger
        count_chute += 1

    # add Flight parameters
    obj_dict["flight"]["initial_solution"] = flight.initial_solution
    obj_dict["flight"]["inclination"] = flight.inclination
    obj_dict["flight"]["heading"] = flight.heading
    obj_dict["flight"]["name"] = name

    ################# Write CSV files #################
    # Write the thrust data to the CSV file
    obj_dict["motors"]["thrust_source"] = function_serializer(
        flight.rocket.motor.thrust
    )

    # Write the power off drag data to the CSV file
    obj_dict["rocket"]["drag_coefficient_power_off"] = function_serializer(
        flight.rocket.power_off_drag
    )

    # Write the power on drag data to the CSV file
    obj_dict["rocket"]["drag_coefficient_power_on"] = function_serializer(
        flight.rocket.power_on_drag
    )

    if return_dict:
        return obj_dict

    else:  ################# Write JSON file #################
        # Save the dictionary to a JSON file
        with open(save_path / "data_inputs" / "parameters.json", "w") as json_file:
            json.dump(obj_dict, json_file, indent=4, default=serialize)


def serialize(obj):
    """Function to handle numpy objects during json.dump"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.complex_):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, Function):
        return obj.source
    else:
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")
