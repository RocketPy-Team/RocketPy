"""Unit tests for ScheduledController: schedule normalization, open-loop
replay with clamping, from_controller fidelity, and the serialization
fallback when a controller function cannot be restored."""

import numpy as np
import pytest

from rocketpy import (
    AirBrakes,
    ControllableGenericSurface,
    Controller,
    Function,
    ScheduledController,
)

STATE = [0.0] * 13


def make_surface(name="canard"):
    return ControllableGenericSurface(
        reference_area=0.01, reference_length=0.1, coefficients={}, name=name
    )


class TestScheduleNormalization:
    def test_flat_schedule_maps_to_single_object(self):
        surface = make_surface()
        controller = ScheduledController(
            {"deflection": [(0, 0.0), (10, 1.0)]}, surface, sampling_rate=10
        )
        assert list(controller.schedule) == ["canard"]
        assert isinstance(controller.schedule["canard"]["deflection"], Function)

    def test_flat_schedule_with_multiple_objects_rejected(self):
        surfaces = [make_surface("a"), make_surface("b")]
        with pytest.raises(ValueError, match="exactly one"):
            ScheduledController({"deflection": [(0, 0.0)]}, surfaces, sampling_rate=10)

    def test_nested_schedule_and_source_forms(self):
        surfaces = [make_surface("a"), make_surface("b")]
        controller = ScheduledController(
            {
                "a": {"deflection": Function(lambda t: 0.5)},
                "b": {"deflection": 0.3},
            },
            surfaces,
            sampling_rate=10,
        )
        controller(time=1.0, state=STATE)
        assert surfaces[0].get_control("deflection") == 0.5
        assert surfaces[1].get_control("deflection") == 0.3

    def test_non_dict_schedule_rejected(self):
        with pytest.raises(TypeError, match="schedule must be a dict"):
            ScheduledController([(0, 0.0)], make_surface(), sampling_rate=10)


class TestReplay:
    def test_interpolated_replay(self):
        surface = make_surface()
        controller = ScheduledController(
            {"deflection": [(0, 0.0), (10, 1.0)]}, surface, sampling_rate=10
        )
        controller(time=2.5, state=STATE)
        assert surface.get_control("deflection") == pytest.approx(0.25)
        # constant extrapolation past the schedule
        controller(time=20.0, state=STATE)
        assert surface.get_control("deflection") == pytest.approx(1.0)

    def test_replay_applies_clamping(self):
        air_brakes = AirBrakes(
            drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
            reference_area=0.01,
            clamp=True,
        )
        controller = ScheduledController(
            {"deployment_level": [(0, 0.0), (1, 5.0)]}, air_brakes, sampling_rate=10
        )
        controller(time=1.0, state=STATE)
        assert air_brakes.deployment_level == 1  # clamped from 5.0

    def test_from_controller_replays_recorded_history(self):
        surface = make_surface()

        def law(**kwargs):
            kwargs["controlled_objects"].set_control("deflection", 0.1 * kwargs["time"])

        source = Controller(law, surface, sampling_rate=10)
        for time in (1.0, 2.0, 3.0):
            source(time=time, state=STATE)

        replay = ScheduledController.from_controller(source)
        assert replay.sampling_rate == 10
        assert replay.name == "Controller (replay)"
        for time in (1.0, 1.5, 3.0):
            replay(time=time, state=STATE)
            assert surface.get_control("deflection") == pytest.approx(0.1 * time)

    def test_from_controller_without_history_rejected(self):
        source = Controller(lambda **kwargs: None, make_surface(), sampling_rate=10)
        with pytest.raises(ValueError, match="no recorded control history"):
            ScheduledController.from_controller(source)


class TestSerializationFallback:
    def build_recorded_controller(self):
        surface = make_surface()

        def law(**kwargs):
            kwargs["controlled_objects"].set_control("deflection", 0.1 * kwargs["time"])

        controller = Controller(law, surface, sampling_rate=10)
        for time in (1.0, 2.0, 3.0):
            controller(time=time, state=STATE)
        return controller

    def test_undecodable_function_with_history_falls_back_to_replay(self):
        data = self.build_recorded_controller().to_dict(include_outputs=True)
        data["controller_function"] = "<not decodable>"

        with pytest.warns(UserWarning, match="ScheduledController"):
            fallback = Controller.from_dict(data)
        assert isinstance(fallback, ScheduledController)

        surface = make_surface()
        fallback.bind_controlled_objects(surface, fallback.controlled_objects_name)
        fallback(time=2.5, state=STATE)
        assert surface.get_control("deflection") == pytest.approx(0.25)

    def test_undecodable_function_without_history_raises(self):
        data = self.build_recorded_controller().to_dict(include_outputs=False)
        data["controller_function"] = "<not decodable>"
        with pytest.raises(ValueError, match="Could not restore"):
            Controller.from_dict(data)

    def test_history_restored_from_dict(self):
        data = self.build_recorded_controller().to_dict(include_outputs=True)
        rebuilt = Controller.from_dict(data)
        schedule = rebuilt.recorded_schedule["canard"]["deflection"]
        assert schedule(2.0) == pytest.approx(0.2)
        assert rebuilt._controlled_objects_ref == ["canard"]

    def test_scheduled_controller_round_trip(self):
        surface = make_surface()
        controller = ScheduledController(
            {"deflection": np.array([[0, 0.0], [10, 1.0]])},
            surface,
            sampling_rate=10,
            name="Replay",
        )
        data = controller.to_dict()
        assert data["controller_function"] is None

        rebuilt = ScheduledController.from_dict(data)
        fresh = make_surface()
        rebuilt.bind_controlled_objects(fresh)
        rebuilt(time=5.0, state=STATE)
        assert fresh.get_control("deflection") == pytest.approx(0.5)
