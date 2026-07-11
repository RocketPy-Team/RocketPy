"""Unit tests for the public Controller class: event integration, callback
kwargs, automatic control-state tracking, and the reset lifecycle."""

import pytest

from rocketpy import ControllableGenericSurface, ControlledObject, Controller
from rocketpy.simulation.events.event import Event

STATE = [0.0] * 13


def make_surface(name="canard", controls=("deflection",)):
    return ControllableGenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={},
        name=name,
        controls=controls,
    )


def make_controller(surface=None, **kwargs):
    surface = surface if surface is not None else make_surface()

    def controller_function(**kwargs):
        return None

    defaults = {
        "controller_function": controller_function,
        "controlled_objects": surface,
        "sampling_rate": 10,
    }
    defaults.update(kwargs)
    return Controller(**defaults), surface


class TestControllerConstruction:
    def test_controller_is_an_event(self):
        controller, _ = make_controller()
        assert isinstance(controller, Event)
        assert controller.changes_dynamics is True
        assert controller.priority == 3
        assert controller.trigger is None
        assert controller.trigger_only_once is False

    def test_controllable_surface_satisfies_controlled_object_protocol(self):
        assert isinstance(make_surface(), ControlledObject)

    def test_positional_controller_function_rejected(self):
        with pytest.raises(ValueError, match="keyword arguments only"):
            Controller(
                lambda time, state: None,
                controlled_objects=make_surface(),
                sampling_rate=10,
            )

    def test_var_positional_controller_function_rejected(self):
        def controller_function(*args, **kwargs):
            return None

        with pytest.raises(ValueError, match="keyword arguments only"):
            Controller(
                controller_function,
                controlled_objects=make_surface(),
                sampling_rate=10,
            )

    def test_non_callable_controller_function_rejected(self):
        with pytest.raises(ValueError, match="callable"):
            Controller(
                "not a function", controlled_objects=make_surface(), sampling_rate=10
            )

    def test_invalid_needs_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown needs keys"):
            make_controller(needs=["not_a_need"])

    def test_reserved_controlled_objects_name_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            make_controller(controlled_objects_name="time")

    def test_mismatched_controlled_objects_name_length_rejected(self):
        surfaces = [make_surface("a"), make_surface("b")]
        with pytest.raises(ValueError, match="match number"):
            Controller(
                lambda **kwargs: None,
                controlled_objects=surfaces,
                sampling_rate=10,
                controlled_objects_name=["only_one"],
            )


class TestControllerCallback:
    def test_callback_kwargs_injection(self):
        received = {}

        def controller_function(**kwargs):
            received.update(kwargs)

        surface = make_surface()
        controller = Controller(
            controller_function,
            controlled_objects=surface,
            sampling_rate=10,
            controlled_objects_name="canard_surface",
        )
        controller(time=0.5, state=STATE)

        assert received["controller"] is controller
        assert received["event"] is controller
        assert received["controlled_objects"] is surface
        assert received["canard_surface"] is surface
        assert received["time"] == 0.5
        assert received["sampling_rate"] == 10

    def test_return_value_appended_to_log(self):
        def controller_function(**kwargs):
            return {"time": kwargs["time"]}

        controller = Controller(
            controller_function, controlled_objects=make_surface(), sampling_rate=10
        )
        controller(time=0.1, state=STATE)
        controller(time=0.2, state=STATE)

        assert controller.log == [{"time": 0.1}, {"time": 0.2}]
        assert controller.log is controller.callback_log
        assert controller.return_log is controller.log


class TestControlStateTracking:
    def test_control_history_records_every_execution(self):
        def controller_function(**kwargs):
            kwargs["controlled_objects"].set_control("deflection", 0.1 * kwargs["time"])

        controller, _ = make_controller(controller_function=controller_function)
        controller(time=1.0, state=STATE)
        controller(time=2.0, state=STATE)

        history = controller.control_history["canard"]["deflection"]
        assert history == [
            (1.0, pytest.approx(0.1)),
            (2.0, pytest.approx(0.2)),
        ]

    def test_history_recorded_even_when_function_does_not_actuate(self):
        controller, _ = make_controller()
        controller(time=1.0, state=STATE)
        assert controller.control_history["canard"]["deflection"] == [(1.0, 0.0)]

    def test_friendly_name_used_as_tracking_key(self):
        controller, _ = make_controller(controlled_objects_name="my_canard")
        assert list(controller.control_history) == ["my_canard"]

    def test_multiple_objects_tracked_with_deduplicated_names(self):
        surfaces = [make_surface("fin"), make_surface("fin")]
        controller = Controller(
            lambda **kwargs: None, controlled_objects=surfaces, sampling_rate=10
        )
        assert list(controller.control_history) == ["fin", "fin_2"]

    def test_untrackable_objects_are_ignored(self):
        controller = Controller(
            lambda **kwargs: None, controlled_objects=object(), sampling_rate=10
        )
        assert controller.control_history == {}
        controller(time=1.0, state=STATE)  # must not raise

    def test_recorded_schedule_interpolates_history(self):
        def controller_function(**kwargs):
            kwargs["controlled_objects"].set_control("deflection", kwargs["time"])

        controller, _ = make_controller(controller_function=controller_function)
        controller(time=1.0, state=STATE)
        controller(time=2.0, state=STATE)

        schedule = controller.recorded_schedule["canard"]["deflection"]
        assert schedule(1.0) == pytest.approx(1.0)
        assert schedule(1.5) == pytest.approx(1.5)
        # constant extrapolation outside the recorded window
        assert schedule(5.0) == pytest.approx(2.0)
        assert schedule(0.0) == pytest.approx(1.0)

    def test_recorded_schedule_single_sample_is_constant(self):
        controller, _ = make_controller()
        controller(time=1.0, state=STATE)
        schedule = controller.recorded_schedule["canard"]["deflection"]
        assert schedule(10.0) == pytest.approx(0.0)

    def test_recorded_schedule_empty_when_no_history(self):
        controller, _ = make_controller()
        assert controller.recorded_schedule == {}


class TestResetLifecycle:
    def test_reset_restores_context_snapshot(self):
        def controller_function(**kwargs):
            kwargs["controller"].context["observed"].append(kwargs["time"])

        controller, _ = make_controller(
            controller_function=controller_function, context={"observed": []}
        )
        controller(time=1.0, state=STATE)
        assert controller.context == {"observed": [1.0]}

        controller.reset()
        assert controller.context == {"observed": []}
        # deepcopy semantics: resetting twice must not alias the same list
        controller.context["observed"].append("junk")
        controller.reset()
        assert controller.context == {"observed": []}

    def test_reset_clears_history_and_log(self):
        def controller_function(**kwargs):
            kwargs["controlled_objects"].set_control("deflection", 1.0)
            return "entry"

        controller, surface = make_controller(controller_function=controller_function)
        controller(time=1.0, state=STATE)
        controller.reset()

        assert controller.control_history == {"canard": {"deflection": []}}
        assert controller.log == []
        assert controller.triggered_times == []

    def test_reset_restores_controlled_object_initial_state(self):
        controller, surface = make_controller()
        surface.set_control("deflection", 0.7)
        controller.reset()
        assert surface.control_state == surface.initial_control_state
        assert surface.get_control("deflection") == 0.0

    def test_controllable_surface_reset_restores_all_controls(self):
        surface = make_surface(controls=("delta_pitch", "delta_yaw"))
        surface.set_control("delta_pitch", 0.3)
        surface.set_control("delta_yaw", -0.1)
        surface._reset()
        assert surface.control_state == {"delta_pitch": 0.0, "delta_yaw": 0.0}


class TestBindControlledObjects:
    def test_rebinding_switches_tracked_object(self):
        controller, _ = make_controller()
        replacement = make_surface("replacement")
        controller.bind_controlled_objects(replacement)

        controller(time=1.0, state=STATE)
        assert list(controller.control_history) == ["replacement"]
        assert controller.controlled_objects is replacement

    def test_rebinding_restores_friendly_name_bindings(self):
        received = {}

        def controller_function(**kwargs):
            received.update(kwargs)

        controller = Controller(
            controller_function, controlled_objects=[], sampling_rate=10
        )
        surface = make_surface()
        controller.bind_controlled_objects(surface, "air_brakes")
        controller(time=1.0, state=STATE)
        assert received["air_brakes"] is surface


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        def controller_function(**kwargs):
            return kwargs["time"]

        controller, _ = make_controller(
            controller_function=controller_function,
            name="Roundtrip",
            context={"gain": 2.0},
            controlled_objects_name="canard_surface",
        )
        data = controller.to_dict()
        surface = make_surface()
        rebuilt = Controller.from_dict(data, controlled_objects=surface)

        assert rebuilt.name == "Roundtrip"
        assert rebuilt.sampling_rate == 10
        assert rebuilt.context == {"gain": 2.0}
        assert rebuilt.controlled_objects is surface
        rebuilt(time=3.0, state=STATE)
        assert rebuilt.log == [3.0]

    def test_from_dict_without_objects_keeps_pending_name(self):
        controller, _ = make_controller(controlled_objects_name="canard_surface")
        rebuilt = Controller.from_dict(controller.to_dict())

        assert rebuilt.controlled_objects == []
        assert rebuilt.controlled_objects_name == "canard_surface"

        surface = make_surface()
        rebuilt.bind_controlled_objects(surface, rebuilt.controlled_objects_name)
        assert list(rebuilt.control_history) == ["canard_surface"]
