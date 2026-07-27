import pytest

from rocketpy import Epoch, FlightState, FrameVector, PointMassRocket


def test_rocket_orbital_properties_accept_state_dependent_callables():
    """Rocket evaluates drag and radiation properties through one public model."""
    # Arrange
    rocket = PointMassRocket(0.1, 10.0, 0.0, 0.4, 0.4)
    state = FlightState.cartesian(
        epoch=Epoch.relative_origin(),
        position=[7.0e6, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
        frame="gcrf",
    )
    rocket.set_orbital_drag_model(
        coefficient=lambda epoch, state, direction: 2.0 + direction[1] / 7500.0,
        projected_area=3.0,
    )
    rocket.set_radiation_properties(coefficient=1.3, projected_area=2.0)
    velocity = FrameVector(state.velocity, state.frame)
    earth_direction = FrameVector(-state.position, state.frame)

    # Act
    drag_coefficient = rocket.evaluate_orbital_drag_coefficient(
        state.epoch, state, velocity
    )

    # Assert
    assert drag_coefficient == pytest.approx(3.0)
    assert rocket.evaluate_orbital_drag_area(
        state.epoch, state, velocity
    ) == pytest.approx(3.0)
    assert rocket.evaluate_radiation_coefficient(
        state.epoch, state, earth_direction
    ) == pytest.approx(1.3)
    assert rocket.evaluate_radiation_area(
        state.epoch, state, earth_direction
    ) == pytest.approx(2.0)
    assert velocity.frame is state.frame
