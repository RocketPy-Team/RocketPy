from importlib.metadata import version
from pathlib import Path

from packaging.version import parse
from rocketpy import Function, SolidMotor

if parse(version("rocketpy")) >= parse("1.0.0a1"):
    from rocketpy import (
        Fluid,
        HybridMotor,
        LevelBasedTank,
        LiquidMotor,
        MassBasedTank,
        MassFlowRateBasedTank,
        TankGeometry,
        UllageBasedTank,
    )


def motor_deserializer(params):
    motor_kind = params["kind"]

    thrust = params["thrust_source"]
    burn_time = params["burn_time"]
    nozzle_radius = params["nozzle_radius"]
    dry_mass = params["dry_mass"]
    dry_inertia = params["dry_inertia"]
    center_of_dry_mass_position = params["center_of_dry_mass"]

    if motor_kind == "solid":
        motor = SolidMotor(
            thrust_source=thrust,
            burn_time=burn_time,
            nozzle_radius=nozzle_radius,
            throat_radius=params["throat_radius"],
            dry_mass=dry_mass,
            dry_inertia=dry_inertia,
            center_of_dry_mass_position=center_of_dry_mass_position,
            grain_number=params["grain_number"],
            grain_density=params["grain_density"],
            grain_outer_radius=params["grain_outer_radius"],
            grain_initial_inner_radius=params["grain_initial_inner_radius"],
            grain_initial_height=params["grain_initial_height"],
            grain_separation=params["grain_separation"],
            grains_center_of_mass_position=params["grains_center_of_mass_position"],
        )

    elif motor_kind == "liquid":
        motor = LiquidMotor(
            thrust_source=thrust,
            burn_time=burn_time,
            nozzle_radius=nozzle_radius,
            dry_mass=dry_mass,
            dry_inertia=dry_inertia,
            center_of_dry_mass_position=center_of_dry_mass_position,
        )

        for tank_ser in params["tanks"]:
            tank = tank_deserializer(tank_ser)
            motor.add_tank(tank, tank_ser["position"])

    elif motor_kind == "hybrid":
        motor = HybridMotor(
            thrust_source=thrust,
            burn_time=burn_time,
            nozzle_radius=nozzle_radius,
            dry_mass=dry_mass,
            dry_inertia=dry_inertia,
            center_of_dry_mass_position=center_of_dry_mass_position,
            throat_radius=params["throat_radius"],
            grain_number=params["grain_number"],
            grain_density=params["grain_density"],
            grain_outer_radius=params["grain_outer_radius"],
            grain_initial_inner_radius=params["grain_initial_inner_radius"],
            grain_initial_height=params["grain_initial_height"],
            grain_separation=params["grain_separation"],
            grains_center_of_mass_position=params["grains_center_of_mass_position"],
        )

        for tank_ser in params["tanks"]:
            tank = tank_deserializer(tank_ser)
            motor.add_tank(tank, tank_ser["position"])

    return motor


def tank_deserializer(params):
    kind = params["kind"]

    if kind == "level":
        tank = _level_based_tank_deserializer(params)
    elif kind == "mass":
        tank = _mass_based_tank_deserializer(params)
    elif kind == "mass_flow":
        tank = _mass_flow_rate_based_tank_deserializer(params)
    elif kind == "ullage":
        tank = _ullage_based_tank_deserializer(params)
    else:
        raise ValueError("Invalid tank type")

    return tank


def _level_based_tank_deserializer(params):
    liquid_level = Function(params[f"{params['name']}_liquid_level"])

    return LevelBasedTank(
        name=params["name"],
        geometry=tank_geometry_deserializer(params),
        flux_time=params["flux_time"],
        gas=fluid_deserializer(params["gas"]),
        liquid=fluid_deserializer(params["liquid"]),
        discretize=params["discretize"],
        liquid_height=params["liquid_height"],
    )


def _mass_based_tank_deserializer(
    params,
):
    gas_mass = Function(params[f"{params['name']}_gas_mass"])
    liquid_mass = Function(params[f"{params['name']}_liquid_mass"])

    return MassBasedTank(
        name=params["name"],
        geometry=tank_geometry_deserializer(
            params,
        ),
        flux_time=params["flux_time"],
        gas=fluid_deserializer(params["gas"]),
        liquid=fluid_deserializer(params["liquid"]),
        discretize=params["discretize"],
        gas_mass=gas_mass,
        liquid_mass=liquid_mass,
    )


def _mass_flow_rate_based_tank_deserializer(
    params,
):
    gas_mass_flow_rate_in = Function(params[f"{params['name']}_gas_mass_flow_rate_in"])
    gas_mass_flow_rate_out = Function(
        params[f"{params['name']}_gas_mass_flow_rate_out"]
    )
    liquid_mass_flow_rate_in = Function(
        params[f"{params['name']}_liquid_mass_flow_rate_in"]
    )
    liquid_mass_flow_rate_out = Function(
        params[f"{params['name']}_liquid_mass_flow_rate_out"]
    )

    return MassFlowRateBasedTank(
        name=params["name"],
        geometry=tank_geometry_deserializer(
            params,
        ),
        flux_time=params["flux_time"],
        gas=fluid_deserializer(params["gas"]),
        liquid=fluid_deserializer(params["liquid"]),
        discretize=params["discretize"],
        initial_gas_mass=params["initial_gas_mass"],
        initial_liquid_mass=params["initial_liquid_mass"],
        gas_mass_flow_rate_in=gas_mass_flow_rate_in,
        gas_mass_flow_rate_out=gas_mass_flow_rate_out,
        liquid_mass_flow_rate_in=liquid_mass_flow_rate_in,
        liquid_mass_flow_rate_out=liquid_mass_flow_rate_out,
    )


def _ullage_based_tank_deserializer(
    params,
):
    ullage = Function(params[f"{params['name']}_ullage"])

    return UllageBasedTank(
        name=params["name"],
        geometry=tank_geometry_deserializer(
            params,
        ),
        flux_time=params["flux_time"],
        gas=fluid_deserializer(params["gas"]),
        liquid=fluid_deserializer(params["liquid"]),
        discretize=params["discretize"],
        ullage=ullage,
    )


def tank_geometry_deserializer(params):
    radius_function = Function(params[f"{params['name']}_radius"])
    geometry_dict = {
        (
            radius_function.x_array[0],
            radius_function.x_array[-1],
        ): radius_function.source
    }

    return TankGeometry(geometry_dict)


def fluid_deserializer(params):
    return Fluid(params["name"], params["density"])
