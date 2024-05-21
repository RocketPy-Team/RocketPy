import csv
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from packaging.version import parse
from rocketpy import Function, SolidMotor

if parse(version("rocketpy")) >= parse("1.0.0a1"):
    from rocketpy import (
        HybridMotor,
        LevelBasedTank,
        LiquidMotor,
        MassBasedTank,
        MassFlowRateBasedTank,
        UllageBasedTank,
    )


def function_serializer(function_object: Function, t_range=None):
    func = deepcopy(function_object)
    if callable(function_object.source):
        if t_range is not None:
            func.set_discrete(*t_range, 100)
        else:
            raise ValueError("t_range must be specified for callable functions")

    return func.get_source()

    # with open(csv_path, "w+", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(func.source)


def motor_serializer(motor):
    result = {}

    result["burn_time"] = motor.burn_time
    result["nozzle_radius"] = motor.nozzle_radius
    result["dry_mass"] = motor.dry_mass
    result["dry_inertia"] = (
        motor.dry_I_11,
        motor.dry_I_22,
        motor.dry_I_33,
        motor.dry_I_12,
        motor.dry_I_13,
        motor.dry_I_23,
    )
    result[
        "center_of_dry_mass"
    ] = (
        motor.center_of_dry_mass_position
    )  # it considered the motor in CM calculation before

    if isinstance(motor, SolidMotor):
        result.update(_solid_motor_serializer(motor))
        result["kind"] = "solid"
    elif isinstance(motor, LiquidMotor):
        result.update(_liquid_motor_serializer(motor))
        result["kind"] = "liquid"
    elif isinstance(motor, HybridMotor):
        result.update(_hybrid_motor_serializer(motor))
        result["kind"] = "hybrid"
    else:
        raise ValueError("Invalid motor type")

    return result


def _solid_motor_serializer(solid_motor):
    result = {}

    result["grain_number"] = solid_motor.grain_number
    result["grain_density"] = solid_motor.grain_density
    result["grain_initial_inner_radius"] = solid_motor.grain_initial_inner_radius
    result["grain_outer_radius"] = solid_motor.grain_outer_radius
    result["grain_initial_height"] = solid_motor.grain_initial_height
    result["throat_radius"] = solid_motor.throat_radius
    result["grain_separation"] = solid_motor.grain_separation
    result[
        "grains_center_of_mass_position"
    ] = solid_motor.grains_center_of_mass_position

    return result


def _liquid_motor_serializer(liquid_motor):
    result = {}

    tanks_ser = []

    for tank_dict in liquid_motor.positioned_tanks:
        tank, position = tank_dict.values()
        tank_ser = tank_serializer(tank)
        tank_ser.update({"position": position})
        tanks_ser.append(tank_ser)

    result["tanks"] = tanks_ser

    return result


def _hybrid_motor_serializer(hybrid_motor):
    result = {}

    result.update(_solid_motor_serializer(hybrid_motor.solid))
    result.update(_liquid_motor_serializer(hybrid_motor.liquid))

    return result


def tank_serializer(tank):
    result = {}
    result["name"] = tank.name
    result[f"{tank.name}_radius"] = function_serializer(tank.geometry.radius)
    result["flux_time"] = tank.flux_time
    result["gas"] = fluid_serializer(tank.gas)
    result["liquid"] = fluid_serializer(tank.liquid)
    result["discretize"] = tank.discretize

    if isinstance(tank, LevelBasedTank):
        result.update(_level_based_tank_serializer(tank))
        result.update({"kind": "level"})
    elif isinstance(tank, MassBasedTank):
        result.update(_mass_based_tank_serializer(tank))
        result.update({"kind": "mass"})
    elif isinstance(tank, MassFlowRateBasedTank):
        result.update(_mass_flow_rate_based_tank_serializer(tank))
        result.update({"kind": "mass_flow"})
    elif isinstance(tank, UllageBasedTank):
        result.update(_ullage_based_tank_serializer(tank))
        result.update({"kind": "ullage"})

    return result


def _mass_flow_rate_based_tank_serializer(mass_flow_rate_based_tank):
    result = {}
    tank_name = mass_flow_rate_based_tank.name

    result["initial_liquid_mass"] = mass_flow_rate_based_tank.initial_liquid_mass
    result["initial_gas_mass"] = mass_flow_rate_based_tank.initial_gas_mass
    result[f"{tank_name}_liquid_mass_flow_rate_in"] = function_serializer(
        mass_flow_rate_based_tank.liquid_mass_flow_rate_in
    )
    result[f"{tank_name}_gas_mass_flow_rate_in"] = function_serializer(
        mass_flow_rate_based_tank.gas_mass_flow_rate_in
    )
    result[f"{tank_name}_liquid_mass_flow_rate_out"] = function_serializer(
        mass_flow_rate_based_tank.liquid_mass_flow_rate_out
    )
    result[f"{tank_name}_gas_mass_flow_rate_out"] = function_serializer(
        mass_flow_rate_based_tank.gas_mass_flow_rate_out
    )

    return result


def _mass_based_tank_serializer(mass_based_tank):
    result = {}
    tank_name = mass_based_tank.name

    result[f"{tank_name}_liquid_mass"] = function_serializer(
        mass_based_tank.liquid_mass
    )
    result[f"{tank_name}_gas_mass"] = function_serializer(mass_based_tank.gas_mass)

    return result


def _level_based_tank_serializer(level_based_tank):
    result = {}
    tank_name = level_based_tank.name

    result[f"{tank_name}_liquid_height"] = function_serializer(
        level_based_tank.liquid_height
    )

    return result


def _ullage_based_tank_serializer(ullage_based_tank):
    result = {}
    tank_name = ullage_based_tank.name

    result[f"{tank_name}_ullage"] = function_serializer(ullage_based_tank.ullage)

    return result


def fluid_serializer(fluid):
    result = {}

    result["name"] = fluid.name
    result["density"] = fluid.density

    return result
