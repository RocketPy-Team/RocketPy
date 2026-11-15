"""The readings of a rocket's sensors: read during a flight through
``context.sensors``, kept in the flight's solution as ``solution.sensors`` and
as Functions in ``flight.sensor_history``, and taken back when the flight goes
back in time, without flying."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Flight
from rocketpy.sensors.barometer import Barometer
from rocketpy.simulation.events.event_execution import rewind
from rocketpy.simulation.events.views import SensorsView, sensor_positions
from rocketpy.simulation.flight_data_exporter import _sensor_rows
from rocketpy.simulation.solution import Solution

# A rocket at rest 100 m above the ground, upright
STATE = [0, 0, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
STATE_DOT = [0] * 13


class _Context(dict):
    """The part of the event context a sensor's event reads."""

    def __init__(self, solution, **values):
        super().__init__(**values)
        self._solution = solution


def _read(solution, event, rocket, environment, time):
    """Run a sensor's event once at ``time``."""
    event.callback(
        _Context(
            solution,
            event=event,
            time=time,
            canonical_state=STATE,
            canonical_state_dot=STATE_DOT,
            rocket=rocket,
            environment=environment,
        )
    )


def _positions_by_key(rocket):
    """Each sensor position by its key, as the flight stores it. Each sensor
    event is given the name its readings go by."""
    return sensor_positions(rocket._sensor_events)


def _flight_of(rocket, solution, t=1.0):
    """A stand-in flight holding what the sensor results read."""
    flight = SimpleNamespace(
        rocket=rocket,
        solution=solution,
        t=t,
        events=[event for _, event in rocket._sensor_events],
        _controls={},
        _step_log={},
        _sensors_by_key=_positions_by_key(rocket),
    )
    return flight


def _history(flight):
    """``flight.sensor_history`` of the stand-in flight."""
    return Flight.sensor_history.func(flight)


def _view(rocket, solution):
    """``context.sensors`` for a flight of ``rocket``."""
    return SensorsView(_positions_by_key(rocket), solution)


@pytest.fixture
def solution():
    return Solution()


@pytest.fixture
def rocket(calisto_with_sensors, example_plain_env, solution):
    """Calisto with its ideal sensors (the accelerometer added twice), each
    position having read at 0.1 s and 0.2 s."""
    for sensor, _ in calisto_with_sensors._sensor_events:
        sensor._reset(calisto_with_sensors)
    _positions_by_key(calisto_with_sensors)
    # Each position reads both times before the next one reads: the position
    # a measurement belongs to does not depend on the order they run in
    for _, event in calisto_with_sensors._sensor_events:
        event.reset()
        for time in (0.1, 0.2):
            _read(solution, event, calisto_with_sensors, example_plain_env, time)
    return calisto_with_sensors


def _sensor(rocket, index):
    return rocket._sensor_events[index][0]


def _event(rocket, index):
    return rocket._sensor_events[index][1]


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_sensors_are_named_by_name_and_numbered_at_later_positions(rocket, solution):
    keys = list(_flight_of(rocket, solution)._sensors_by_key)
    assert keys == [
        "accelerometer",
        "accelerometer_2",
        "gyroscope",
        "barometer",
        "gnss_receiver",
    ]
    assert list(solution.sensors) == [
        "Accelerometer",
        "Accelerometer 2",
        "Gyroscope",
        "Barometer",
        "GnssReceiver",
    ]


def test_two_sensors_with_one_name_are_numbered(calisto, solution):
    first, second = Barometer(sampling_rate=10), Barometer(sampling_rate=10)
    calisto.add_sensor(first, 0.0)
    calisto.add_sensor(second, 0.5)
    view = _view(calisto, solution)

    assert list(view) == ["Barometer", "Barometer 2"]
    assert dir(view) == ["barometer", "barometer_2"]
    assert len(calisto.sensors_by_name["Barometer"]) == 2


def test_a_name_in_snake_case_counts_as_the_same_name(calisto, solution):
    calisto.add_sensor(Barometer(sampling_rate=10), 0.0)
    calisto.add_sensor(Barometer(sampling_rate=10, name="barometer"), 0.5)
    assert list(_view(calisto, solution)) == ["Barometer", "barometer 2"]


def test_a_number_taken_by_a_name_is_skipped(calisto, solution):
    calisto.add_sensor(Barometer(sampling_rate=10, name="Barometer 2"), 0.0)
    barometer = Barometer(sampling_rate=10)
    calisto.add_sensor(barometer, 0.1)
    calisto.add_sensor(barometer, 0.5)
    assert list(_view(calisto, solution)) == ["Barometer 2", "Barometer", "Barometer 3"]


# ---------------------------------------------------------------------------
# context.sensors
# ---------------------------------------------------------------------------


def test_context_sensors_read_the_latest_measurement(rocket, solution):
    sensors = _view(rocket, solution)
    barometer = _sensor(rocket, 3)

    assert sensors.barometer.pressure == barometer.measurement
    assert sensors["Gyroscope"].wz == _sensor(rocket, 2).measurement[2]
    assert sensors.gnss_receiver.altitude == _sensor(rocket, 4).measurement[2]
    assert list(sensors) == [
        "Accelerometer",
        "Accelerometer 2",
        "Gyroscope",
        "Barometer",
        "GnssReceiver",
    ]


def test_each_position_of_a_sensor_is_read_on_its_own(rocket, solution):
    """One sensor object added twice has one ``measurement``; the context
    gives the reading made at each position."""
    sensors = _view(rocket, solution)
    # a different reading at position 1
    solution._sensors["Accelerometer"]["az"][-1] = (0.2, -1.0)

    assert sensors.accelerometer.az == -1.0
    assert (
        sensors.accelerometer_2.az == solution._sensors["Accelerometer 2"]["az"][-1][1]
    )


def test_context_sensors_are_none_before_the_first_measurement(calisto_with_sensors):
    sensors = _view(calisto_with_sensors, Solution())
    assert sensors.barometer.pressure is None
    assert sensors["Accelerometer 2"].ax is None


def test_context_sensors_name_what_is_missing_and_cannot_be_written(rocket, solution):
    sensors = _view(rocket, solution)
    with pytest.raises(AttributeError, match="Its sensors are: accelerometer,"):
        _ = sensors.accelerometer_3
    with pytest.raises(AttributeError, match="Its channels are: ax, ay, az"):
        _ = sensors.accelerometer.jerk
    with pytest.raises(KeyError, match="no sensor named 'barometer'.*'Barometer'"):
        _ = sensors["barometer"]  # the name as given, not in snake_case
    with pytest.raises(TypeError, match="read-only"):
        sensors["Barometer"] = 0.0
    with pytest.raises(AttributeError, match="read-only"):
        sensors.barometer.pressure = 0.0


# ---------------------------------------------------------------------------
# The flight's readings
# ---------------------------------------------------------------------------


def test_a_sensor_event_stores_each_reading_by_channel(rocket, solution):
    pressure = solution.sensors.barometer.pressure
    assert list(solution.sensors.barometer) == ["pressure"]
    assert list(pressure[:, 0]) == [0.1, 0.2]
    assert pressure[-1, 1] == _sensor(rocket, 3).measurement
    assert np.array_equal(solution.sensors["Barometer"]["pressure"], pressure)
    assert list(solution.sensors["Barometer"].items())[0][0] == "pressure"


def test_each_position_of_a_sensor_keeps_its_own_measurements(rocket, solution):
    assert list(solution.sensors["Accelerometer 2"].az[:, 0]) == [0.1, 0.2]
    rows = _sensor_rows(_flight_of(rocket, solution))
    accelerometer = _sensor(rocket, 0)
    assert [[row[0] for row in position] for position in rows[accelerometer]] == [
        [0.1, 0.2],
        [0.1, 0.2],
    ]


def test_the_sensor_history_uses_the_names_of_context_sensors(rocket, solution):
    history = _history(_flight_of(rocket, solution))

    view = _view(rocket, solution)
    assert list(history) == list(view)
    assert [list(history[name]) for name in history] == [
        list(view[name]) for name in view
    ]
    pressure = history["Barometer"].pressure
    assert list(history.barometer.pressure.x_array) == list(pressure.x_array)
    assert list(pressure.x_array) == [0.1, 0.2]
    # between two readings the earlier one holds
    assert pressure(0.15) == pressure.y_array[0]
    assert pressure(0.2) == solution.sensors.barometer.pressure[-1, 1]


def test_the_sensor_rows_give_the_same_readings_as_tuples(rocket, solution):
    flight = _flight_of(rocket, solution)
    history = _history(flight)
    rows = _sensor_rows(flight)

    # added twice: one list per position
    assert len(rows[_sensor(rocket, 0)]) == 2
    assert rows[_sensor(rocket, 0)][1][0] == (
        0.1,
        history["Accelerometer 2"].ax(0.1),
        history["Accelerometer 2"].ay(0.1),
        history.accelerometer_2.az(0.1),
    )
    assert [row[0] for row in rows[_sensor(rocket, 2)]] == [0.1, 0.2]


def test_flight_records_leave_out_the_sensors(rocket, solution):
    assert Flight.records.fget(_flight_of(rocket, solution)) == {}


def test_flight_sensor_data_is_deprecated_and_gives_the_sensor_rows(rocket, solution):
    """``flight.sensor_data`` is kept as it was in v1.13, sensor object to its
    list of tuples, with a deprecation warning."""
    flight = _flight_of(rocket, solution)
    with pytest.warns(DeprecationWarning, match="sensor_data is deprecated"):
        data = Flight.sensor_data.fget(flight)
    assert data == _sensor_rows(flight)


def test_a_flight_without_readings_has_empty_sensor_rows(calisto_with_sensors):
    _positions_by_key(calisto_with_sensors)
    flight = _flight_of(calisto_with_sensors, Solution())
    assert len(_history(flight)) == 0
    rows = _sensor_rows(flight)
    accelerometer = _sensor(calisto_with_sensors, 0)
    assert rows[accelerometer] == [[], []]
    assert all(
        position == []
        for sensor, position in rows.items()
        if sensor is not accelerometer
    )


def test_the_readings_are_saved_with_the_solution(rocket, solution):  # pylint: disable=unused-argument
    loaded = Solution.from_dict(solution.to_dict())
    assert list(loaded.sensors) == list(solution.sensors)
    assert np.array_equal(
        loaded.sensors.accelerometer_2.az, solution.sensors.accelerometer_2.az
    )


# ---------------------------------------------------------------------------
# Going back in time
# ---------------------------------------------------------------------------


def test_a_rollback_drops_the_readings_after_its_time(
    rocket, solution, example_plain_env
):
    flight = _flight_of(rocket, solution, t=0.3)
    rewind(flight, 0.15)

    assert list(solution.sensors.barometer.pressure[:, 0]) == [0.1]
    assert list(_history(flight)["Barometer"].pressure.x_array) == [0.1]
    # the sensor reads that time again on the trajectory flown from there
    _read(solution, _event(rocket, 3), rocket, example_plain_env, 0.2)
    assert list(solution.sensors.barometer.pressure[:, 0]) == [0.1, 0.2]


def test_a_rollback_takes_the_sensors_back(rocket, solution):
    """What a controller reads of a sensor during the flight: its measured
    data loses the measurements after the time the flight went back to, and
    its measurement is again the last one kept."""
    flight = _flight_of(rocket, solution, t=0.3)
    barometer = _sensor(rocket, 3)
    barometer.measurement = -1.0  # a measurement on the trajectory left behind

    rewind(flight, 0.15)

    assert barometer.measurement == solution.sensors.barometer.pressure[-1, 1]
    assert [row[0] for row in barometer.measured_data] == [0.1]
    accelerometer = _sensor(rocket, 0)
    assert [[row[0] for row in rows] for rows in accelerometer.measured_data] == [
        [0.1],
        [0.1],
    ]
    assert accelerometer.measurement == accelerometer.measured_data[1][0][1:]
    assert _view(rocket, solution).barometer.pressure == barometer.measurement


def test_a_new_flight_starts_without_a_measurement(rocket):
    barometer = _sensor(rocket, 3)
    assert barometer.measurement is not None
    barometer._reset(rocket)
    assert barometer.measurement is None
    assert barometer.measured_data == []


def test_a_rollback_before_any_measurement_leaves_none(rocket, solution):
    rewind(_flight_of(rocket, solution, t=0.3), 0.05)
    assert all(sensor.measurement is None for sensor, _ in rocket._sensor_events)
    assert _sensor(rocket, 3).measured_data == []
    assert _view(rocket, solution).accelerometer.ax is None
