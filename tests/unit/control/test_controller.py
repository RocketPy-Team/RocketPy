from dataclasses import dataclass

from rocketpy.control.controller import _Controller


@dataclass(frozen=True)
class InteractiveObject:
    name: str


def controller_function(
    time,
    sampling_rate,
    state_vector,
    state_history,
    observed_variables,
    interactive_objects,
    sensors,
    environment,
):
    return {
        "time": time,
        "sampling_rate": sampling_rate,
        "state": state_vector,
        "history": state_history,
        "previous": observed_variables[-1],
        "objects": interactive_objects,
        "sensors": sensors,
        "environment": environment,
    }


def test_controller_preserves_initial_observation_and_appends_return_value():
    initial_observation = {"deployment": 0.0}
    interactive_object = InteractiveObject("Air brakes")
    controller = _Controller(
        interactive_objects=[interactive_object],
        controller_function=controller_function,
        sampling_rate=20,
        initial_observed_variables=initial_observation,
        name="Air-brake controller",
    )

    state = [0.0] * 13
    history = [[0.0] + state]
    sensors = [object()]
    environment = object()
    controller(0.25, state, history, sensors, environment)

    assert controller.observed_variables == [
        initial_observation,
        {
            "time": 0.25,
            "sampling_rate": 20,
            "state": state,
            "history": history,
            "previous": initial_observation,
            "objects": [interactive_object],
            "sensors": sensors,
            "environment": environment,
        },
    ]


def test_controller_info_reports_discrete_rate_and_each_interactive_object(capsys):
    controller = _Controller(
        interactive_objects=[
            InteractiveObject("Left air brake"),
            InteractiveObject("Right air brake"),
        ],
        controller_function=controller_function,
        sampling_rate=4,
        name="Roll controller",
    )

    controller.all_info()

    assert capsys.readouterr().out == (
        "\nController Details\n\n"
        "Controller 'Roll controller' with sampling rate 4 Hz.\n"
        "Controller function: controller_function\n"
        "Controller refresh rate: 4.000 Hz\n"
        "interactive Objects\n"
        "Left air brake\n"
        "Right air brake\n"
    )


def test_controller_info_reports_continuous_rate_and_single_object(capsys):
    controller = _Controller(
        interactive_objects=InteractiveObject("Thrust vector actuator"),
        controller_function=controller_function,
        sampling_rate=None,
        name="Pitch controller",
    )

    controller.info()

    assert capsys.readouterr().out == (
        "\nController Details\n\n"
        "Controller 'Pitch controller' with continuous sampling.\n"
        "Controller function: controller_function\n"
        "Controller refresh rate: continuous (every solver step)\n"
        "interactive Objects\n"
        "Thrust vector actuator\n"
    )


def test_controller_to_dict_without_pickle_uses_function_name_and_object_hashes():
    interactive_objects = [InteractiveObject("A"), InteractiveObject("B")]
    controller = _Controller(
        interactive_objects=interactive_objects,
        controller_function=controller_function,
        sampling_rate=8,
        initial_observed_variables=[0.0],
        name="Test controller",
    )

    data = controller.to_dict(allow_pickle=False)

    assert data == {
        "controller_function": "controller_function",
        "sampling_rate": 8,
        "initial_observed_variables": [0.0],
        "name": "Test controller",
        "_interactive_objects_hash": [hash(obj) for obj in interactive_objects],
    }


def test_controller_to_dict_hashes_a_single_interactive_object():
    interactive_object = InteractiveObject("Actuator")
    controller = _Controller(
        interactive_objects=interactive_object,
        controller_function=controller_function,
    )

    assert controller.to_dict(allow_pickle=False)["_interactive_objects_hash"] == hash(
        interactive_object
    )


def test_controller_from_dict_accepts_an_existing_callable():
    data = {
        "interactive_objects": [InteractiveObject("Actuator")],
        "controller_function": controller_function,
        "sampling_rate": 5,
        "initial_observed_variables": [1.0],
        "name": "Restored controller",
        "_interactive_objects_hash": [1234],
    }

    restored = _Controller.from_dict(data)

    assert restored.base_controller_function is controller_function
    assert restored.sampling_rate == 5
    assert restored.initial_observed_variables == [1.0]
    assert restored.name == "Restored controller"
    assert restored._interactive_objects_hash == [1234]
