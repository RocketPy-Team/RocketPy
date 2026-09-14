"""Acceptance test for the 2024 Defiance example flight."""

import pytest

from rocketpy import Environment, Flight, Rocket
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank


MEASURED_APOGEE_AGL = 9308.32
MAX_RELATIVE_APOGEE_ERROR = 0.01
REFERENCE_MAX_SPEED = 444.24
REFERENCE_MAX_ACCELERATION = 10400.76
REFERENCE_IMPACT_X = 1609.40
REFERENCE_IMPACT_Y = 87.03
REFERENCE_METRIC_RELATIVE_TOLERANCE = 0.01
REFERENCE_IMPACT_ABSOLUTE_TOLERANCE = 3.0


def _build_defiance_flight():
    """Build the deterministic Defiance example flight."""
    environment = Environment(
        latitude=47.966527,
        longitude=-81.87413,
        elevation=1383.4,
        date=(2024, 8, 24, 0),
    )
    environment.set_atmospheric_model(type="custom_atmosphere", wind_v=1.0, wind_u=-2.9)

    liquid_oxidizer = Fluid(name="N2O_l", density=960)
    gaseous_oxidizer = Fluid(name="N2O_g", density=1.9277)
    oxidizer_tank = MassFlowRateBasedTank(
        name="oxidizer_tank",
        geometry=CylindricalTank(radius_function=0.0665, height=1.79),
        flux_time=6.5,
        liquid=liquid_oxidizer,
        gas=gaseous_oxidizer,
        initial_liquid_mass=17,
        initial_gas_mass=0,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=17 / 6.5,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
    )

    motor = HybridMotor(
        thrust_source="data/rockets/defiance/Thrust_curve.csv",
        dry_mass=13.832,
        dry_inertia=(1.801, 1.801, 0.0305),
        center_of_dry_mass_position=0.780,
        grain_number=1,
        grain_separation=0,
        grain_outer_radius=0.0665,
        grain_initial_inner_radius=0.061,
        grain_initial_height=1.25,
        grain_density=920,
        nozzle_radius=0.0447,
        throat_radius=0.0234,
        grains_center_of_mass_position=0.377,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    motor.add_tank(tank=oxidizer_tank, position=2.2)

    rocket = Rocket(
        radius=0.07,
        mass=37.211,
        inertia=(94.14, 94.14, 0.09),
        center_of_mass_without_motor=3.29,
        power_off_drag="data/rockets/defiance/DragCurve.csv",
        power_on_drag="data/rockets/defiance/DragCurve.csv",
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=0.2)
    rocket.add_nose(length=0.563, kind="vonKarman", position=4.947)
    rocket.add_trapezoidal_fins(
        n=3,
        span=0.115,
        root_chord=0.4,
        tip_chord=0.2,
        position=0.175,
    )
    rocket.add_tail(
        top_radius=0.07,
        bottom_radius=0.064,
        length=0.0597,
        position=0.1,
    )
    rocket.add_parachute(name="main", cd_s=2.2, trigger=305, sampling_rate=100, lag=0)
    rocket.add_parachute(
        name="drogue",
        cd_s=1.55,
        trigger="apogee",
        sampling_rate=100,
        lag=0,
    )

    return Flight(
        rocket=rocket,
        environment=environment,
        inclination=85,
        heading=90,
        rail_length=10,
    )


@pytest.fixture(scope="module")
def defiance_flight():
    """Return one deterministic Defiance flight for the acceptance checks."""
    return _build_defiance_flight()


def test_defiance_rocket_apogee_matches_measured_flight(defiance_flight):
    """Compare the Defiance example simulation with its measured apogee."""
    simulated_apogee_agl = defiance_flight.apogee - defiance_flight.env.elevation
    relative_error = (
        abs(MEASURED_APOGEE_AGL - simulated_apogee_agl) / MEASURED_APOGEE_AGL
    )

    assert relative_error < MAX_RELATIVE_APOGEE_ERROR, (
        f"Defiance apogee relative error is {relative_error:.2%}; "
        f"expected less than {MAX_RELATIVE_APOGEE_ERROR:.2%}."
    )


def test_defiance_rocket_matches_reference_flight_metrics(defiance_flight):
    """Guard the deterministic example's peak and impact metrics."""
    tolerance = {"rel": REFERENCE_METRIC_RELATIVE_TOLERANCE}
    impact_tolerance = {
        **tolerance,
        "abs": REFERENCE_IMPACT_ABSOLUTE_TOLERANCE,
    }

    assert defiance_flight.max_speed == pytest.approx(REFERENCE_MAX_SPEED, **tolerance)
    assert defiance_flight.max_acceleration == pytest.approx(
        REFERENCE_MAX_ACCELERATION, **tolerance
    )
    assert defiance_flight.x_impact == pytest.approx(
        REFERENCE_IMPACT_X, **impact_tolerance
    )
    assert defiance_flight.y_impact == pytest.approx(
        REFERENCE_IMPACT_Y, **impact_tolerance
    )
