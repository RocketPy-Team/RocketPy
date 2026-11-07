import ast
import inspect
import json
import os

import pytest

from rocketpy.simulation import monte_carlo as mc_module
from rocketpy.simulation.monte_carlo import (
    MonteCarlo,
    _refuse_logs_missing_a_simulation,
)


def _a_log(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(rows), encoding="utf-8")
    return str(path)


def _row(index):
    return json.dumps({"index": index, "mass": 1.0}) + "\n"


def _complete(tmp_path, count=3, name="ok"):
    rows = [_row(index) for index in range(count)]
    return (
        _a_log(tmp_path, f"{name}.inputs.txt", rows),
        _a_log(tmp_path, f"{name}.outputs.txt", rows),
    )


def test_a_run_that_recorded_everything_is_accepted(tmp_path):
    """Logs holding every index the run asked for raise nothing."""
    inputs, outputs = _complete(tmp_path)

    _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_blank_lines_between_rows_are_not_simulations(tmp_path):
    """A blank line is skipped rather than counted as an unreadable row."""
    # An interrupted write leaves them, and reading one as a row would report
    # a damaged log for a run that lost nothing.
    rows = [_row(0), "\n", _row(1), "   \n", _row(2)]
    inputs = _a_log(tmp_path, "gappy.inputs.txt", rows)
    outputs = _a_log(tmp_path, "gappy.outputs.txt", rows)

    _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_a_missing_simulation_is_refused(tmp_path):
    """A gap in the output log names the first index that is missing."""
    inputs, outputs = _complete(tmp_path)
    _a_log(tmp_path, "ok.outputs.txt", [_row(0), _row(2)])

    with pytest.raises(RuntimeError, match=r"output log.*missing.*being 1"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_a_simulation_recorded_twice_is_refused(tmp_path):
    """A duplicated index is refused, since a set alone would hide it."""
    inputs, outputs = _complete(tmp_path)
    _a_log(tmp_path, "ok.inputs.txt", [_row(0), _row(1), _row(1), _row(2)])

    with pytest.raises(RuntimeError, match="more than once"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_a_row_that_cannot_be_read_is_refused(tmp_path):
    """A torn row means the log's contents cannot be established."""
    inputs, outputs = _complete(tmp_path)
    _a_log(tmp_path, "ok.outputs.txt", [_row(0), "{half a row\n", _row(2)])

    with pytest.raises(RuntimeError, match="cannot be read"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_rows_numbered_past_the_run_are_left_alone(tmp_path):
    """An append below what a checkpoint already holds loses no simulation."""
    # Refusing these said rows were missing when they were extra, and moved
    # what append means inside a change about worker failure.
    rows = [_row(index) for index in range(4)]
    inputs = _a_log(tmp_path, "big.inputs.txt", rows)
    outputs = _a_log(tmp_path, "big.outputs.txt", rows)

    _refuse_logs_missing_a_simulation(inputs, outputs, 2)


def test_logs_that_hold_different_simulations_are_refused(tmp_path):
    """The input and output logs have to hold the same indices."""
    inputs = _a_log(tmp_path, "a.inputs.txt", [_row(0), _row(1)])
    outputs = _a_log(tmp_path, "a.outputs.txt", [_row(0), _row(2)])

    with pytest.raises(RuntimeError):
        _refuse_logs_missing_a_simulation(inputs, outputs, 2)


@pytest.mark.parametrize("index", [True, False, 1.0, -1, [], "1", None])
def test_an_index_that_is_not_a_whole_number_is_refused(tmp_path, index):
    """Only a non-negative int names a simulation, whatever compares equal."""
    # True and 1.0 both equal 1, so either could stand in for a simulation that
    # was never run. An unhashable one used to escape as a raw TypeError.
    rows = [_row(0), _row(index)]
    inputs = _a_log(tmp_path, "odd.inputs.txt", rows)
    outputs = _a_log(tmp_path, "odd.outputs.txt", rows)

    with pytest.raises(RuntimeError, match="cannot be read"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 2)


def test_a_stray_index_past_the_rows_is_refused(tmp_path):
    """A number no run of this length could have produced is not a simulation."""
    # A worker that claimed one index too many writes exactly this, and the
    # target alone cannot tell it from a checkpoint that is legitimately longer.
    rows = [_row(0), _row(1), _row(99)]
    inputs = _a_log(tmp_path, "stray.inputs.txt", rows)
    outputs = _a_log(tmp_path, "stray.outputs.txt", rows)

    with pytest.raises(RuntimeError, match="past the 3 it holds"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 2)


def test_the_order_a_parallel_run_finishes_in_is_not_a_hole(tmp_path):
    """Rows arrive in completion order, which is not sorted and not wrong."""
    rows = [_row(1), _row(0), _row(2)]
    inputs = _a_log(tmp_path, "para.inputs.txt", rows)
    outputs = _a_log(tmp_path, "para.outputs.txt", rows)

    _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_logs_that_disagree_on_the_order_are_refused(tmp_path):
    """A record goes into both logs under one lock, so the order is the same."""
    inputs = _a_log(tmp_path, "order.inputs.txt", [_row(0), _row(1)])
    outputs = _a_log(tmp_path, "order.outputs.txt", [_row(1), _row(0)])

    with pytest.raises(RuntimeError, match="same order"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 2)


def _leave_cleanly_without_recording(_flight):
    # A worker that ends the way an out-of-memory kill ends it, but with the
    # status of one that finished. Nothing about the process says otherwise.
    os._exit(0)


def test_a_worker_that_leaves_cleanly_without_recording_is_not_a_success(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """A zero exit with no row written makes ``simulate`` raise."""
    analysis = MonteCarlo(
        filename=str(tmp_path / "study"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
        data_collector={"leave": _leave_cleanly_without_recording},
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        analysis.simulate(
            number_of_simulations=6, append=False, parallel=True, n_workers=2
        )


def test_no_failure_path_waits_on_a_worker_without_a_bound():
    """No ``join`` in the parallel path is called without a timeout."""
    # An unbounded join anywhere in the parallel path puts back the hang that
    # the bounded teardown exists to end, and it does so where it is hardest
    # to notice: only when a worker is already stuck.
    tree = ast.parse(inspect.getsource(mc_module))
    run_in_parallel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__run_in_parallel"
    )

    unbounded = [
        node.lineno
        for node in ast.walk(run_in_parallel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and not node.args
        and not node.keywords
    ]

    assert not unbounded, f"join() with no timeout at lines {unbounded}"


def test_starting_a_worker_happens_inside_the_cleanup_scope():
    """A start that fails has to leave the workers before it accounted for."""
    # Structural for the same reason the unbounded-join check is: it only shows
    # when a start has already failed, which a unit test cannot make a real
    # process do. Outside the try, an interrupt there left children running.
    tree = ast.parse(inspect.getsource(mc_module))
    run_in_parallel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__run_in_parallel"
    )
    guarded = [
        node
        for handler in ast.walk(run_in_parallel)
        if isinstance(handler, ast.Try)
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
    ]

    assert guarded, "Process.start() is not inside a try in __run_in_parallel"


def test_a_worker_is_recorded_only_once_it_has_started():
    """A process that never started cannot be joined or terminated."""
    tree = ast.parse(inspect.getsource(mc_module))
    run_in_parallel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__run_in_parallel"
    )
    lines = {"start": None, "append": None}
    for node in ast.walk(run_in_parallel):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "start":
                lines["start"] = node.lineno
            if node.func.attr == "append":
                lines["append"] = node.lineno

    assert lines["start"] is not None and lines["append"] is not None
    assert lines["start"] < lines["append"], "appended before it started"


def test_a_collector_cannot_take_over_the_simulation_index(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """A collector key called index is refused before the run touches a file."""
    # Measured before this was refused: every row was written with the
    # collector's value, so the log said 999 twice for a two-simulation run
    # and every check that reads an index was reading the wrong thing.
    # Refused when the collector is handed over, which is before any file
    # is opened, rather than at the end of a run that is already spoilt.
    with pytest.raises(ValueError, match="index"):
        MonteCarlo(
            filename=str(tmp_path / "study"),
            environment=stochastic_environment,
            rocket=stochastic_calisto,
            flight=stochastic_flight,
            data_collector={"index": lambda flight: 999},
        )


def test_a_collector_key_of_its_own_is_still_welcome(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """The control. Only the one reserved name is refused."""
    analysis = MonteCarlo(
        filename=str(tmp_path / "ok"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
        data_collector={"apogee_twice": lambda flight: 2 * flight.apogee},
    )

    analysis.simulate(number_of_simulations=1, append=False)

    with open(analysis.output_file, "r", encoding="utf-8") as written:
        row = json.loads(next(line for line in written if line.strip()))
    # Not the value of the index: how a run numbers its simulations is
    # settled elsewhere, and pinning it here would tie this to that.
    assert "index" in row
    assert "apogee_twice" in row
