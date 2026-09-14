import json
import os
from contextlib import suppress
from types import SimpleNamespace

import pytest

from rocketpy.simulation import monte_carlo as mc_module
from rocketpy.simulation.monte_carlo import MonteCarlo


class _Mutex:
    def __init__(self):
        self.held = False
        self.acquired = 0
        self.blocking = None
        self.timeout = None

    def acquire(self, blocking=True, timeout=None):  # the proxy's signature
        self.acquired += 1
        self.blocking, self.timeout = blocking, timeout
        self.held = True
        return True

    def release(self):
        self.held = False


class _MutexThatCannotBeTaken(_Mutex):
    """A lock a dead holder never gave back: acquire waits out its bound."""

    def acquire(self, blocking=True, timeout=None):
        super().acquire(blocking, timeout)
        self.held = False
        return False


class _MutexThatBreaksOnAcquire(_Mutex):
    """The manager is gone, so asking for the lock raises instead."""

    def acquire(self, blocking=True, timeout=None):
        super().acquire(blocking, timeout)
        self.held = False
        raise OSError("the manager is gone")


class _MutexThatBreaksOnRelease(_Mutex):
    """The manager goes while the lock is held, so giving it back raises."""

    def release(self):
        raise OSError("the manager is gone")


class _ErrorEvent:
    def __init__(self, refuse=False):
        self.was_set = False
        self.refuse = refuse

    def is_set(self):
        return self.was_set

    def set(self):
        if self.refuse:
            raise OSError("the manager is gone")
        self.was_set = True


def _raise_instead(message):
    def refuse(*_args, **_kwargs):
        raise OSError(message)

    return refuse


def _refusing_model():
    def refuse(_seed):
        raise RuntimeError("the models would not reseed")

    return SimpleNamespace(last_rnd_dict={}, _set_stochastic=refuse)


def _a_worker(tmp_path, model, event=None):
    study = MonteCarlo(
        filename=str(tmp_path / "study"),
        environment=model,
        rocket=model,
        flight=model,
    )
    return study, event or _ErrorEvent()


def _run(study, monitor, error_event, mutex=None):
    # Name-mangled: the producer is what each worker process runs, and nothing
    # else in the suite calls it.
    mutex = mutex or _Mutex()
    study._MonteCarlo__sim_producer(42, monitor, mutex, error_event)
    return mutex


def test_a_worker_that_fails_before_seeding_finishes_says_so(tmp_path, capsys):
    """A failure above the loop is reported against worker startup."""
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, _refusing_model())

    _run(study, monitor, error_event)

    assert error_event.was_set
    reported = capsys.readouterr().out
    assert "worker startup" in reported
    assert "the models would not reseed" in reported


def test_a_worker_that_fails_before_claiming_an_index_says_so(tmp_path, capsys):
    """A failed claim is startup too, since no index was taken."""

    def refuse():
        raise RuntimeError("the monitor would not hand out an index")

    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _seed: None)
    monitor = SimpleNamespace(keep_simulating=lambda: True, increment=refuse)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set
    assert "worker startup" in capsys.readouterr().out


def test_a_worker_that_fails_inside_a_simulation_names_the_index(
    tmp_path, capsys, monkeypatch
):
    """A failure after a claim is reported against that index."""

    # The control. An index is claimed and the simulation then fails, which is
    # the path that already worked, so the report still has to name it.
    def refuse(_self):
        raise RuntimeError("the simulation would not run")

    monkeypatch.setattr(
        MonteCarlo, "_MonteCarlo__run_single_simulation", refuse, raising=True
    )
    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _seed: None)
    monitor = SimpleNamespace(keep_simulating=lambda: True, increment=lambda: 8)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set
    assert "iteration 7" in capsys.readouterr().out


def test_a_startup_failure_is_written_down_and_not_only_printed(tmp_path):
    """The error log gets a row even when no inputs were drawn."""
    # The caller is told to read the error file, and a traceback the worker
    # printed is not there to be read once its output has been redirected.
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, _refusing_model())

    _run(study, monitor, error_event)

    with open(study.error_file, "r", encoding="utf-8") as recorded:
        rows = [json.loads(line) for line in recorded if line.strip()]
    assert len(rows) == 1
    assert rows[0]["index"] is None
    assert rows[0]["stage"] == "worker startup"
    assert "the models would not reseed" in rows[0]["error"]


@pytest.mark.parametrize("failing", ["_set_stochastic", "increment"])
def test_a_worker_failure_never_raises_out_of_the_producer(tmp_path, failing):
    """A reported failure leaves the producer without an exception."""

    # The handler used to reach for names the loop had not bound yet, so the
    # process died with UnboundLocalError and the parent waited forever.
    def refuse(*_args):
        raise RuntimeError("boom")

    model = SimpleNamespace(
        last_rnd_dict={},
        _set_stochastic=refuse if failing == "_set_stochastic" else lambda _s: None,
    )
    monitor = SimpleNamespace(
        keep_simulating=lambda: True,
        increment=refuse if failing == "increment" else (lambda: 1),
    )
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set


@pytest.mark.parametrize("breaking", ["error_file", "reprint", "event"])
def test_reporting_a_failure_never_keeps_the_mutex(tmp_path, monkeypatch, breaking):
    """The manager lock is released however the reporting goes."""
    # The mutex is the manager's, so a worker that ends while holding it leaves
    # the next one waiting on a process that is gone, and the parent never
    # reaches the join that would have noticed.
    if breaking == "error_file":
        monkeypatch.setattr(
            mc_module, "_worker_failure_record", _raise_instead("no disk")
        )
    if breaking == "reprint":
        monkeypatch.setattr(
            mc_module._SimMonitor, "reprint", _raise_instead("no stdout")
        )
    event = _ErrorEvent(refuse=breaking == "event")
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, _refusing_model(), event)
    mutex = _Mutex()

    # A worker that could not announce its failure re-raises on the way out, so
    # that its exit code carries what the event could not. The lock still has
    # to be back either way, which is what this is about.
    with suppress(RuntimeError):
        _run(study, monitor, error_event, mutex)

    assert mutex.acquired == 1
    assert not mutex.held


def test_a_reporting_failure_does_not_replace_the_simulation_failure(
    tmp_path, monkeypatch, capsys
):
    """An unwritable log does not hide what actually failed."""
    monkeypatch.setattr(mc_module, "_worker_failure_record", _raise_instead("no disk"))
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, _refusing_model())

    _run(study, monitor, error_event)

    assert error_event.was_set
    assert "the models would not reseed" in capsys.readouterr().out
    assert not os.path.getsize(study.error_file)


def _committing_producer(monkeypatch):
    """Make one simulation run start to finish without a real flight."""
    monkeypatch.setattr(
        MonteCarlo, "_MonteCarlo__run_single_simulation", lambda self: None
    )
    monkeypatch.setattr(
        MonteCarlo,
        "_MonteCarlo__evaluate_flight_inputs",
        lambda self, index: json.dumps({"index": index, "committed": True}) + "\n",
    )
    monkeypatch.setattr(
        MonteCarlo,
        "_MonteCarlo__evaluate_flight_outputs",
        lambda self, flight, index: json.dumps({"index": index}) + "\n",
    )


def _one_then_broken():
    calls = {"count": 0}

    def keep_simulating():
        calls["count"] += 1
        if calls["count"] == 1:
            return True
        raise RuntimeError("the monitor died between simulations")

    return SimpleNamespace(
        keep_simulating=keep_simulating,
        increment=lambda: 1,
        print_update_status=lambda: None,
    )


def test_a_failure_between_simulations_is_not_blamed_on_the_last_one(
    tmp_path, capsys, monkeypatch
):
    """A failure after a committed row is not reported against it."""
    # Simulation 0 finishes and its row is committed. The next claim then
    # fails, which is not simulation 0's doing and must not be recorded as it.
    _committing_producer(monkeypatch)
    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _seed: None)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, _one_then_broken(), error_event)

    assert "worker startup" in capsys.readouterr().out


def test_a_committed_row_is_not_written_to_the_error_log_as_well(tmp_path, monkeypatch):
    """A row that succeeded appears in one log, not in both."""
    _committing_producer(monkeypatch)
    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _seed: None)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, _one_then_broken(), error_event)

    with open(study.output_file, "r", encoding="utf-8") as written:
        committed = [json.loads(line) for line in written if line.strip()]
    with open(study.error_file, "r", encoding="utf-8") as recorded:
        errored = [json.loads(line) for line in recorded if line.strip()]

    assert committed == [{"index": 0}]
    assert all(row.get("committed") is None for row in errored)


def test_a_worker_that_cannot_announce_its_failure_does_not_exit_cleanly(tmp_path):
    """With the event unreachable the producer raises, so the exit is not zero."""
    # The event is how a worker reaches the parent. With it unreachable, the
    # only signal left is how the process ends, so it must not end well.
    model = _refusing_model()
    study, error_event = _a_worker(tmp_path, model, _ErrorEvent(refuse=True))

    with pytest.raises(RuntimeError, match="the models would not reseed"):
        _run(study, SimpleNamespace(keep_simulating=lambda: True), error_event)


def test_a_worker_that_did_announce_its_failure_returns(tmp_path):
    """With the event delivered the producer returns on purpose."""
    # The control. With the event delivered the parent already knows, so the
    # producer returns and the process exits cleanly on purpose.
    study, error_event = _a_worker(tmp_path, _refusing_model())

    _run(study, SimpleNamespace(keep_simulating=lambda: True), error_event)

    assert error_event.was_set


@pytest.mark.parametrize(
    "mutex_class", [_MutexThatCannotBeTaken, _MutexThatBreaksOnAcquire]
)
def test_a_lock_the_reporter_cannot_take_does_not_stop_it(
    tmp_path, capsys, mutex_class
):
    """A lock that times out or is gone still leaves the failure announced."""
    # Asking for it without a bound is how a worker whose sibling died holding
    # the lock waits forever, with nothing recorded and no exit code to read.
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, _refusing_model())
    mutex = mutex_class()

    _run(study, monitor, error_event, mutex)

    assert error_event.was_set
    assert "the models would not reseed" in capsys.readouterr().out
    assert not mutex.held
    assert mutex.timeout is not None  # asked for with a bound


def test_a_lock_that_breaks_on_release_does_not_hide_the_failure(tmp_path, capsys):
    """Giving the lock back can raise, and must not replace what failed."""
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, _refusing_model())

    _run(study, monitor, error_event, _MutexThatBreaksOnRelease())

    assert error_event.was_set
    assert "the models would not reseed" in capsys.readouterr().out


def test_a_failure_after_the_inputs_were_drawn_still_records_why(tmp_path, monkeypatch):
    """The error file is where the caller is sent, so it has to say what broke."""
    # Writing the input row on its own left no stage and no traceback there,
    # for every failure past the point the inputs had been built.
    _committing_producer(monkeypatch)
    monkeypatch.setattr(
        MonteCarlo,
        "_MonteCarlo__evaluate_flight_outputs",
        _raise_instead("the outputs would not serialize"),
    )
    monitor = SimpleNamespace(keep_simulating=lambda: True, increment=lambda: 1)
    study, error_event = _a_worker(
        tmp_path, SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _s: None)
    )

    _run(study, monitor, error_event)

    with open(study.error_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    assert len(rows) == 1
    assert "the outputs would not serialize" in rows[0]["error"]
    assert rows[0]["stage"] == "iteration 0"
    assert rows[0]["inputs"]["committed"] is True


def test_an_input_row_that_is_not_an_object_does_not_break_the_reporter(
    tmp_path, monkeypatch
):
    """Reading the row is best effort: the failure being reported comes first."""
    _committing_producer(monkeypatch)
    monkeypatch.setattr(
        MonteCarlo,
        "_MonteCarlo__evaluate_flight_inputs",
        lambda self, index: json.dumps([index]) + "\n",
    )
    monkeypatch.setattr(
        MonteCarlo,
        "_MonteCarlo__evaluate_flight_outputs",
        _raise_instead("the outputs would not serialize"),
    )
    monitor = SimpleNamespace(keep_simulating=lambda: True, increment=lambda: 1)
    study, error_event = _a_worker(
        tmp_path, SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _s: None)
    )

    _run(study, monitor, error_event)

    with open(study.error_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    assert "the outputs would not serialize" in rows[0]["error"]
    assert "inputs" not in rows[0]
