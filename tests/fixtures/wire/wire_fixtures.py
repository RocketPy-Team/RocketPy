import pytest

from rocketpy.rocket import Wire


@pytest.fixture
def test_communications_wire():
    return Wire(
        current=0.01, wire_type="communications", name="test_communications_wire"
    )


@pytest.fixture()
def test_ignition_wire_motor():
    return Wire(
        current=3,
        wire_type="ignition",
        ignition_wire_function="motor_ignition",
        extra_ignition_time=0.5,
        name="test_ignition_wire_motor",
    )


@pytest.fixture()
def test_ignition_wire_parachute():
    return Wire(
        current=3,
        wire_type="ignition",
        ignition_wire_function="parachute_deployment",
        extra_ignition_time=0.5,
        name="test_ignition_wire_parachute",
    )
