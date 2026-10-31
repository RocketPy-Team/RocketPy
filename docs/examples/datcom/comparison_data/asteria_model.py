"""Build AST-HER from the saved current Asteria configuration, without plots."""

import json
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from rocketpy import (
    Rocket,
    Environment,
    LiquidMotor,
    MassFlowRateBasedTank,
    CylindricalTank,
    Fluid,
    Function,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "rocketpy").is_dir())
DATA = ROOT / "data/rockets/ast_her"


def build(native, executable):
    cfg = json.loads((DATA / "configuration.json").read_text())
    r, m, f = cfg["rocket"], cfg["motors"], cfg["fins"]
    thrust = np.loadtxt(DATA / "thrust_curve.csv", delimiter=",")
    performance = np.genfromtxt(DATA / "performance.csv", delimiter=",", names=True)
    isp = np.interp(
        thrust[:, 1],
        performance["Thrust_sea_level_kN"] * 1000,
        performance["Specific_impulse_sea_level_s"] * m["isp_degradation_factor"],
    )
    mdot = thrust[:, 1] / (isp * 9.80665)
    fuel_flow = mdot / (1 + m["of"])
    oxidizer_flow = mdot - fuel_flow
    fuel_gas_flow = fuel_flow * m["pressurant_density_60bar"] / m["eth_density"]
    oxidizer_gas_flow = (
        oxidizer_flow * m["pressurant_density_40bar_cold"] / m["lox_density"]
    )

    def curve(values):
        return np.column_stack((thrust[:, 0], values))

    fuel_liquid = Fluid("Ethanol", m["eth_density"])
    oxidizer_liquid = Fluid("LOX", m["lox_density"])
    nitrogen_fuel = Fluid("Nitrogen, fuel tank", m["pressurant_density_60bar"])
    nitrogen_oxidizer = Fluid(
        "Nitrogen, oxidizer tank", m["pressurant_density_40bar_cold"]
    )
    nitrogen_supply = Fluid("Nitrogen, supply tank", m["pressurant_density_300bar"])
    burn = (float(thrust[0, 0]), float(thrust[-1, 0]))
    motor = LiquidMotor(
        thrust_source=thrust,
        burn_time=burn,
        dry_mass=0,
        dry_inertia=m["motor_dry_inertia"],
        center_of_dry_mass_position=m["center_of_dry_mass_position"],
        nozzle_position=m["nozzle_position"],
        nozzle_radius=m["nozzle_exit_radius"],
        reference_pressure=m["reference_pressure"],
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    tanks = []
    for label, height_key, position_key, fluid, gas, liquid_out, gas_in in [
        (
            "Fuel",
            "eth_tank_height",
            "eth_tank_position",
            fuel_liquid,
            nitrogen_fuel,
            fuel_flow,
            fuel_gas_flow,
        ),
        (
            "Oxidizer",
            "lox_tank_height",
            "lox_tank_position",
            oxidizer_liquid,
            nitrogen_oxidizer,
            oxidizer_flow,
            oxidizer_gas_flow,
        ),
    ]:
        tank = MassFlowRateBasedTank(
            name=label,
            geometry=CylindricalTank(
                m["tank_radius"], m[height_key], spherical_caps=m["tank_spherical_caps"]
            ),
            flux_time=burn,
            initial_liquid_mass=trapezoid(liquid_out, x=thrust[:, 0])
            + m["numeric_stab_mass"],
            initial_gas_mass=0.001,
            liquid_mass_flow_rate_in=0,
            liquid_mass_flow_rate_out=curve(liquid_out),
            gas_mass_flow_rate_in=curve(gas_in),
            gas_mass_flow_rate_out=0,
            liquid=fluid,
            gas=gas,
        )
        motor.add_tank(tank, position=m[position_key])
        tanks.append(tank)
    pr = m["prnt_tank_radius"]
    supply_tank = MassFlowRateBasedTank(
        name="Pressurant",
        geometry=CylindricalTank(
            pr,
            m["pressurant_tank_volume"] / (np.pi * pr**2) + 0.667 * pr,
            spherical_caps=True,
        ),
        flux_time=burn,
        initial_liquid_mass=0,
        initial_gas_mass=m["pressurant_tank_volume"] * m["pressurant_density_300bar"],
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=0,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=curve(fuel_gas_flow + oxidizer_gas_flow),
        liquid=nitrogen_supply,
        gas=nitrogen_supply,
    )
    motor.add_tank(supply_tank, position=m["pressure_tank_position"])
    tanks.append(supply_tank)
    print(
        f"Burn duration: {burn[1]:.3f} s; total impulse: {motor.total_impulse:.1f} N s"
    )

    drag = Function(
        str(Path(__file__).with_name("asteria_drag_curve.csv")),
        interpolation="linear",
        extrapolation="constant",
    )
    rocket = Rocket(
        radius=r["radius"],
        mass=r["rocket_dry_mass"],
        inertia=r["inertia"],
        center_of_mass_without_motor=r["center_of_mass_without_motor"],
        power_on_drag=drag,
        power_off_drag=drag,
        coordinate_system_orientation=r["coordinate_system_orientation"],
    )
    rocket.add_motor(motor, position=m["position"])
    rocket.add_nose(
        length=cfg["nosecones"]["length"],
        kind=cfg["nosecones"]["kind"],
        position=cfg["nosecones"]["position"],
    )
    geometry = dict(
        n=f["NPANEL"],
        root_chord=f["CHORD"][0],
        tip_chord=f["CHORD"][1],
        span=f["SSPAN"][1] - f["SSPAN"][0],
        position=f["XLE"][0],
        sweep_length=f["XLE"][1] - f["XLE"][0],
        cant_angle=f["cant_angle"],
    )
    if native:
        rocket.add_datcom_hex_fins(
            **geometry,
            path=executable,
            upper_thickness_to_chord=f["ZUPPER"],
            lower_thickness_to_chord=f["ZLOWER"],
            max_thickness_position=f["LMAXU"],
            flat_length=f["LFLATU"],
            leading_edge_radius=f["LER"],
            datcom_options={"beta": [-10, 0, 10], "powered_base_drag": "nozzle"},
        )
    else:
        rocket.add_trapezoidal_fins(**geometry)
    rocket.add_tail(
        top_radius=r["radius"],
        bottom_radius=cfg["tails"]["DAFT"] / 2,
        length=cfg["tails"]["LAFT"],
        position=cfg["tails"]["position"],
        kind="ogive" if native else "conical",
    )
    rocket.set_rail_buttons(
        upper_button_position=cfg["rail_buttons"]["upper_position"],
        lower_button_position=cfg["rail_buttons"]["lower_position"],
        angular_position=cfg["rail_buttons"]["angular_position"],
    )
    saved = json.loads((DATA / "atmosphere.json").read_text())
    e = cfg["environment"]
    env = Environment(
        latitude=e["latitude"],
        longitude=e["longitude"],
        elevation=e["elevation"],
        timezone=e["timezone"],
    )
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=saved["pressure"]["source"],
        temperature=saved["temperature"]["source"],
        wind_u=saved["wind_velocity_x"]["source"],
        wind_v=saved["wind_velocity_y"]["source"],
    )
    launch = dict(
        rail_length=cfg["flight"]["rail_length"],
        inclination=cfg["flight"]["inclination"],
        heading=cfg["flight"]["heading"],
        terminate_on_apogee=True,
        max_time=90,
        max_time_step=0.1,
    )
    return rocket, env, launch
