"""Unit tests for the AirBrakes surface: drag coefficient rules, clamping,
reset protocol, and serialization schema."""

import numpy as np
import pytest

from rocketpy import AirBrakes

DRAG_CURVE = "data/rockets/calisto/air_brakes_cd.csv"
REFERENCE_AREA = 0.01


def make_air_brakes(**kwargs):
    defaults = {
        "drag_coefficient_curve": DRAG_CURVE,
        "reference_area": REFERENCE_AREA,
    }
    defaults.update(kwargs)
    return AirBrakes(**defaults)


class TestDragCoefficient:
    def test_drag_coefficient_is_two_dimensional(self):
        air_brakes = make_air_brakes()
        assert air_brakes.drag_coefficient.__dom_dim__ == 2

    @pytest.mark.parametrize("override", [False, True])
    def test_zero_deployment_gives_zero_cd(self, override):
        air_brakes = make_air_brakes(override_rocket_drag=override)
        air_brakes.deployment_level = 0
        cd = air_brakes.cD.get_value_opt(
            *air_brakes._coefficient_arguments(0, 0, 0.5, 1e6, 0, 0, 0)
        )
        assert cd == 0.0

    def test_deployed_cd_matches_curve(self):
        air_brakes = make_air_brakes()
        air_brakes.deployment_level = 0.5
        for mach in (0.2, 0.5, 0.8):
            cd = air_brakes.cD.get_value_opt(
                *air_brakes._coefficient_arguments(0, 0, mach, 1e6, 0, 0, 0)
            )
            assert cd == pytest.approx(
                air_brakes.drag_coefficient.get_value_opt(0.5, mach)
            )

    def test_cd_independent_of_incidence_and_rates(self):
        air_brakes = make_air_brakes()
        air_brakes.deployment_level = 0.7
        reference = air_brakes.cD.get_value_opt(
            *air_brakes._coefficient_arguments(0, 0, 0.5, 1e6, 0, 0, 0)
        )
        perturbed = air_brakes.cD.get_value_opt(
            *air_brakes._coefficient_arguments(0.1, -0.05, 0.5, 5e5, 0.01, 0.02, 0.03)
        )
        assert perturbed == pytest.approx(reference)


class TestClamping:
    def test_clamp_on_bounds_deployment_level(self):
        air_brakes = make_air_brakes(clamp=True)
        air_brakes.deployment_level = 1.5
        assert air_brakes.deployment_level == 1
        air_brakes.deployment_level = -1
        assert air_brakes.deployment_level == 0
        air_brakes.deployment_level = 0.5
        assert air_brakes.deployment_level == 0.5

    def test_clamp_off_warns_and_keeps_value(self):
        air_brakes = make_air_brakes(clamp=False)
        with pytest.warns(UserWarning, match="Extrapolation"):
            air_brakes.deployment_level = 1.5
        assert air_brakes.deployment_level == 1.5
        with pytest.warns(UserWarning, match="Extrapolation"):
            air_brakes.deployment_level = -1
        assert air_brakes.deployment_level == -1


class TestResetProtocol:
    def test_reset_restores_initial_deployment_level(self):
        air_brakes = make_air_brakes(deployment_level=0.3)
        assert air_brakes.initial_control_state == {"deployment_level": 0.3}
        air_brakes.deployment_level = 0.9
        air_brakes._reset()
        assert air_brakes.deployment_level == 0.3

    def test_reset_reapplies_clamped_initial_value(self):
        air_brakes = make_air_brakes(deployment_level=1.5, clamp=True)
        assert air_brakes.deployment_level == 1
        air_brakes.deployment_level = 0.2
        air_brakes._reset()
        assert air_brakes.deployment_level == 1

    def test_control_state_backs_deployment_level(self):
        air_brakes = make_air_brakes()
        air_brakes.set_control("deployment_level", 0.4)
        assert air_brakes.deployment_level == 0.4
        assert air_brakes.get_control("deployment_level") == 0.4
        assert air_brakes.control_variables == ["deployment_level"]


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        air_brakes = make_air_brakes(
            clamp=False,
            override_rocket_drag=True,
            deployment_level=0.2,
            name="Brakes",
        )
        data = air_brakes.to_dict()
        rebuilt = AirBrakes.from_dict(data)

        assert rebuilt.name == "Brakes"
        assert rebuilt.clamp is False
        assert rebuilt.override_rocket_drag is True
        assert rebuilt.reference_area == REFERENCE_AREA
        assert rebuilt.initial_deployment_level == 0.2
        assert rebuilt.deployment_level == 0.2
        for mach in np.linspace(0.1, 0.9, 5):
            assert rebuilt.drag_coefficient.get_value_opt(0.5, mach) == pytest.approx(
                air_brakes.drag_coefficient.get_value_opt(0.5, mach)
            )
