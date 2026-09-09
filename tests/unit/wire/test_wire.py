import pytest

from rocketpy.exceptions import InvalidParameterError
from rocketpy.mathutils import Vector
from rocketpy.plots.wire_plots import _WirePlots
from rocketpy.rocket import Wire


@pytest.mark.parametrize(
    "wire_type, ignition_wire_funciton, current, extra_ignition_time",
    [
        (0, "motor_ignition", 1, 0.4),
        (  # wrong wire type: number
            "ignition",
            None,
            1,
            0.4,
        ),
        (  # wrong ignition_wire_type: None
            "ignition",
            0,
            1,
            0.4,
        ),
        (  # wrong ignition_wire_type: number
            "ignition",
            "motor",
            "1",
            0.4,
        ),
        (  # wrong current: str
            "ignition",
            "motor",
            1,
            "0.4",
        ),
        (  # wrong extra_igntion_time: str
            "ignition",
            "motor",
            1,
            -0.4,
        ),  # wrong extra_igntion_time: -
    ],
)
def validate_wire_type(wire_type, ignition_wire_function, current, extra_ignition_time):
    """Ensures that InvalidParameterError are raised with incorrect entry parameters."""
    with pytest.raises(InvalidParameterError):
        Wire(current, wire_type, ignition_wire_function, extra_ignition_time)


def test_define_magnetic_field(test_communications_wire):
    """Tests proper handling of the magnetic field calculated
    in the _magnetic_field dictionary."""
    position_vector = [0.003, 0.002, 1]
    magnetic_field = [10, 25, 10]
    test_communications_wire.define_magnetic_field(position_vector, magnetic_field)
    list_key = list(test_communications_wire._magnetic_field)
    assert list_key[0] == tuple(position_vector)
    assert (
        test_communications_wire._magnetic_field[tuple(position_vector)]
        == magnetic_field
    )
    assert test_communications_wire._magnetic_field[tuple(position_vector)] == Vector(
        magnetic_field
    )


def test_current_directon(calisto_with_sensors):
    """Tests that changing direction of the current results in a magnetic field vector
    of the same magnitude but with opposite components and the magnetic field components
    have a correct physical meaning."""
    horizontal_wire_1 = Wire(10, wire_type="communications", name="horizontal_wire")
    horizontal_wire_2 = Wire(10, wire_type="communications", name="horizontal_wire")
    calisto_with_sensors.add_wire(
        horizontal_wire_1, [[0.001, 0.001, 0], [0.001, -0.001, 0]]
    )
    calisto_with_sensors.add_wire(
        horizontal_wire_2, [[0.001, -0.001, 0], [0.001, 0.001, 0]]
    )

    horizontal_wire_1.measure_magnetic_field([0, 0, 0], frame="bacs")
    horizontal_wire_2.measure_magnetic_field([0, 0, 0], frame="bacs")

    b_field_1 = horizontal_wire_1._magnetic_field[(0, 0, 0)]
    b_field_2 = horizontal_wire_2._magnetic_field[(0, 0, 0)]

    # given the calisto_robust frame, [0, 0, 0] is the center of dry mass WITHOUT THE MOTOR,
    # However the bacs frame in which the magnetic field is calculated is slighlty bellow along the z axis
    # As a result, following right-hand rule the x component won't be 0, but a really small number, and in opposite directions
    # In y axis since they are symmetrical component of both field will be 0
    # In the z axis they will oppose since the direciton is opposite
    assert b_field_1[0] == pytest.approx(0, abs=1e-6)
    assert b_field_1[1] == 0
    assert b_field_2[0] == pytest.approx(0, abs=1e-6)
    assert b_field_2[1] == 0
    assert b_field_1[0] == -b_field_2[0]
    assert b_field_2[2] != 0
    assert b_field_1[2] != 0
    assert b_field_2[2] == -b_field_1[2]
    assert abs(b_field_2) == abs(b_field_1)


def test_dimension_increase(calisto_with_sensors):
    """Tests that if the wire is closer it has a higher magnitude for the magnetic field"""
    closer_wire = Wire(10, wire_type="communications", name="horizontal_wire")
    farther_wire = Wire(10, wire_type="communications", name="horizontal_wire")
    calisto_with_sensors.add_wire(closer_wire, [[0.001, 0.001, 0], [0.001, -0.001, 0]])
    calisto_with_sensors.add_wire(farther_wire, [[0.001, 0.001, 0], [0.001, -0.001, 0]])

    closer_wire.measure_magnetic_field([0, 0, 0], frame="bacs")
    farther_wire.measure_magnetic_field([-0.4, 0, 0], frame="bacs")

    b_field_closer = closer_wire._magnetic_field[(0, 0, 0)]
    b_field_farther = farther_wire._magnetic_field[(-0.4, 0, 0)]
    assert abs(b_field_closer) > abs(
        b_field_farther
    )  # closer wire must have a higher magnetic field


@pytest.mark.parametrize(
    "position_vector",
    [
        [0, 0, 0],
        [0.002, 0.002, 0],
        [0.001, 0.001, 0],
        [-0.001, -0.001, 0],
    ],
)
def test_raise_warning(test_communications_wire, calisto_robust, position_vector):
    """Ensures that a warning is raised when the magnetic field measurement function
    of the wire has as a parameter a point located along the wire line."""
    calisto_robust.add_wire(
        test_communications_wire,
        position_endpoints=[[-0.003, -0.003, 0], [0.003, 0.003, 0]],
    )
    with pytest.warns(UserWarning):
        test_communications_wire.measure_magnetic_field(
            position_vector=position_vector, frame="ucs"
        )


def test_from_dict():
    """Ensures from_dict returns a Wire object, whose attributes
    are the ones passed in the dict."""
    wire_dict = {
        "current": 3,
        "wire_type": "ignition",
        "ignition_wire_function": "motor_ignition",
        "extra_ignition_time": 2,
    }
    wire_from_dict = Wire.from_dict(wire_dict)

    assert isinstance(wire_from_dict, Wire)
    assert wire_from_dict.current == pytest.approx(3)
    assert wire_from_dict.wire_type == "ignition"
    assert wire_from_dict.ignition_wire_function == "motor_ignition"
    assert wire_from_dict.extra_ignition_time == pytest.approx(2)


def test_rocket_belonging(test_communications_wire, calisto_robust):
    """Tests the belonging method of the wire works as intended."""
    calisto_robust.add_wire(
        test_communications_wire,
        position_endpoints=[[0.001, 0.002, -0.3], [0.001, 0.002, 0.3]],
    )
    assert isinstance(test_communications_wire.plots, _WirePlots)
    assert not isinstance(test_communications_wire._csys, type(None))
    assert not isinstance(
        test_communications_wire.center_of_dry_mass_position, type(None)
    )


def test_wire_prints_and_plots(test_communications_wire, calisto_robust):
    """Tests the print methods of the Wire class work properly. Checks if all attributes are
    printed and plotted correctly.
    """
    with pytest.raises(InvalidParameterError):
        test_communications_wire.plots.draw()
    calisto_robust.add_wire(
        test_communications_wire, [[0.001, 0.002, -0.3], [0.001, 0.002, 0.3]]
    )
    test_communications_wire.prints.all()
    test_communications_wire.plots.all()
    assert True
