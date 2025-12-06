from rocketpy import Parachute
import pytest


# TODO: Analyse if, instead of a new file, test could be included in a existing one
def test_knacke_opening_shock_example():
    """
    Verifies the opening shock force calculation comparing to a textbook example.

    Reference: Knacke, T. W. (1992). Parachute Recovery Systems Design Manual.
    (Page 5-51, Figure 5-21)
    """
    # Setup
    knacke_cd = 0.49
    knacke_s = 17.76  # m^2
    knacke_cx = 1.088
    knacke_x1 = 1.0

    knacke_density = 0.458  # kg/m^3
    knacke_velocity = 123.2  # m/s

    # Expected result
    knacke_force = 32916.8  # N

    # Defining example parachute
    parachute = Parachute(
        name="B-47 Test Chute",
        cd_s=knacke_cd * knacke_s,
        trigger="apogee",
        sampling_rate=100,
        opening_shock_coefficient=knacke_cx * knacke_x1,
    )

    # Calculating the shock force
    calculated_force = parachute.calculate_opening_shock(
        knacke_density, knacke_velocity
    )

    # Analysing results
    assert calculated_force == pytest.approx(knacke_force, rel=1e-2)
