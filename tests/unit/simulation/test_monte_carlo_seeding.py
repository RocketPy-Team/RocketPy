import json
import os
from time import time

import multiprocess
import numpy as np
import pytest

from rocketpy.simulation.monte_carlo import (
    MonteCarlo,
    _seed_sequence_to_int,
    _SimMonitor,
    _study_seed,
)
from rocketpy.stochastic import StochasticEnvironment


def _what_was_drawn(value):
    """The same record without the parts that say which process wrote it.

    Two of the fields a serialized ``Function`` carries are properties of the
    writer rather than of the draw. ``hash`` is the object's identity in that
    process. A callable ``source`` is its pickle, and the same callable pickles
    to different bytes under ``spawn``: measured on one drag curve, the parent
    and a forked child agree and a spawned child does not, for a value the
    fixture does not vary at all. Everything a run actually draws is numeric
    and stays.
    """
    if isinstance(value, dict):
        return {
            key: _what_was_drawn(item)
            for key, item in value.items()
            if key != "hash" and not (key == "source" and isinstance(item, str))
        }
    if isinstance(value, list):
        return [_what_was_drawn(item) for item in value]
    return value


def _seed_for(index):
    """The seed MonteCarlo gives one index, without running the others."""
    return _seed_sequence_to_int(np.random.SeedSequence([42, index]))


def _sampled_inputs(analysis):
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    return {row["index"]: _what_was_drawn(row) for row in rows}


def _a_run(tmp_path, stem, models, *, parallel=False, workers=None, seed=None, count=4):
    environment, rocket, flight = models
    analysis = MonteCarlo(
        filename=str(tmp_path / stem),
        environment=environment,
        rocket=rocket,
        flight=flight,
        export_list=["max_speed", "t_final"],
    )
    analysis.simulate(
        number_of_simulations=count,
        append=False,
        parallel=parallel,
        n_workers=workers,
        random_seed=seed,
        include_function_data=False,
    )
    return analysis


@pytest.fixture(name="models")
def _models(stochastic_environment, stochastic_calisto, stochastic_flight):
    stochastic_flight.max_time = 1
    stochastic_flight.obj.equations_of_motion = "generalized"
    return stochastic_environment, stochastic_calisto, stochastic_flight


@pytest.mark.parametrize("seed", [0, 42, np.int64(7), 2**128])
def test_study_seed_accepts_non_negative_integers(seed):
    """A study records a plain integer even when given a NumPy integer."""
    assert _study_seed(seed) == seed
    assert isinstance(_study_seed(seed), int)


@pytest.mark.parametrize("seed", [True, 1.5, "42", [1, 2], np.random.SeedSequence(7)])
def test_study_seed_rejects_non_integer_values(seed):
    """Stateful generators and sequences are not integer study seeds."""
    with pytest.raises(TypeError, match="random_seed must be"):
        _study_seed(seed)


def test_study_seed_rejects_negative_values():
    """Reject invalid seeds before any simulation log can be truncated."""
    with pytest.raises(ValueError, match="non-negative"):
        _study_seed(-1)


@pytest.mark.slow
def test_an_unusable_seed_costs_the_previous_run_nothing(models, tmp_path):
    analysis = _a_run(tmp_path, "study", models, seed=42, count=2)
    kept = _sampled_inputs(analysis)

    with pytest.raises(TypeError):
        analysis.simulate(2, append=False, random_seed=np.random.default_rng(1))

    assert _sampled_inputs(analysis) == kept


# ------------------------------------------------------------- what a run gives


@pytest.mark.slow
def test_one_seed_gives_one_set_of_inputs(models, tmp_path):
    first = _a_run(tmp_path, "first", models, seed=42)
    again = _a_run(tmp_path, "again", models, seed=42)

    assert _sampled_inputs(first) == _sampled_inputs(again)


@pytest.mark.slow
def test_another_seed_gives_another_set(models, tmp_path):
    # The control. Without this the test above passes on a run that ignores
    # the seed entirely.
    first = _a_run(tmp_path, "first", models, seed=42)
    other = _a_run(tmp_path, "other", models, seed=7)

    assert _sampled_inputs(first) != _sampled_inputs(other)


@pytest.mark.skipif(os.cpu_count() < 4, reason="needs four workers to be four")
@pytest.mark.slow
def test_an_index_gets_the_same_inputs_however_the_run_was_split(models, tmp_path):
    # This is the guarantee. Splitting the work differently must not change
    # what any one simulation drew.
    serial = _a_run(tmp_path, "serial", models, seed=42)
    two = _a_run(tmp_path, "two", models, parallel=True, workers=2, seed=42)
    four = _a_run(tmp_path, "four", models, parallel=True, workers=4, seed=42)

    assert _sampled_inputs(serial) == _sampled_inputs(two)
    assert _sampled_inputs(serial) == _sampled_inputs(four)


def test_an_index_drawn_last_draws_what_it_draws_alone(example_spaceport_env):
    """A worker reaches an index without running the ones before it.

    The split test above cannot see this: every input in its fixture is given
    an explicit nominal, and only one read off the wrapped object can move.
    """
    # A bare standard deviation, so the nominal is read from the environment,
    # which is also where create_object writes the draw back. Both are built
    # before anything is drawn, the way every worker starts from one model.
    walked = StochasticEnvironment(example_spaceport_env, elevation=10)
    alone = StochasticEnvironment(example_spaceport_env, elevation=10)

    for index in range(4):
        walked._set_stochastic(_seed_for(index))
        walked.create_object()
    reached_last = walked.last_rnd_dict["elevation"]

    alone._set_stochastic(_seed_for(3))
    alone.create_object()

    assert reached_last == alone.last_rnd_dict["elevation"]


def test_the_indices_of_that_run_are_not_all_one_value(example_spaceport_env):
    # The control. Without it the test above holds on a model that draws a
    # constant, which is what an elevation of nominal zero would give.
    walked = StochasticEnvironment(example_spaceport_env, elevation=10)
    drawn = []
    for index in range(4):
        walked._set_stochastic(_seed_for(index))
        walked.create_object()
        drawn.append(walked.last_rnd_dict["elevation"])

    assert len(set(drawn)) == len(drawn)


@pytest.mark.slow
def test_an_appended_run_carries_the_same_stream_on(models, tmp_path):
    whole = _a_run(tmp_path, "whole", models, seed=42, count=4)

    part = _a_run(tmp_path, "part", models, seed=42, count=2)
    part.simulate(4, append=True, random_seed=42, include_function_data=False)

    assert _sampled_inputs(part) == _sampled_inputs(whole)


@pytest.fixture(name="spawned_workers")
def _spawned_workers():
    """Start workers the way Windows does, wherever the test happens to run."""
    was = multiprocess.get_start_method()
    multiprocess.set_start_method("spawn", force=True)
    yield
    multiprocess.set_start_method(was, force=True)


@pytest.mark.usefixtures("spawned_workers")
@pytest.mark.skipif(os.cpu_count() < 4, reason="needs four workers to be four")
@pytest.mark.slow
def test_an_index_keeps_its_inputs_when_the_workers_are_spawned(models, tmp_path):
    """The same guarantee on the start method Windows uses.

    A spawned child rebuilds the models by unpickling rather than inheriting
    them, so this is a different question from the one above and was worth
    asking separately: the first version of these tests compared serialized
    callables, which differ between processes for a value nothing varies.
    """
    serial = _a_run(tmp_path, "serial", models, seed=42)
    two = _a_run(tmp_path, "two", models, parallel=True, workers=2, seed=42)
    four = _a_run(tmp_path, "four", models, parallel=True, workers=4, seed=42)

    assert _sampled_inputs(serial) == _sampled_inputs(two)
    assert _sampled_inputs(serial) == _sampled_inputs(four)


class _ALockNobodyContends:
    """The manager's mutex, with only this process to hand it to."""

    def acquire(self):
        pass

    def release(self):
        pass


class _AnEventNobodySet:
    """The failure event, which nothing in a clean run reaches for."""

    def is_set(self):
        return False

    def set(self):
        raise AssertionError("the producer reported a failure")


@pytest.mark.slow
def test_a_worker_loop_draws_the_study_serial_draws(models, tmp_path):
    """The loop a worker runs, driven here, produces the run serial produces."""
    # Every other check of this starts a child process, where the same code
    # runs and nothing in the test can watch it.
    serial = _a_run(tmp_path, "serial", models, seed=42, count=4)
    environment, rocket, flight = models
    worker = MonteCarlo(
        filename=str(tmp_path / "worker"),
        environment=environment,
        rocket=rocket,
        flight=flight,
        export_list=["max_speed", "t_final"],
    )
    worker.simulate(0, append=False, random_seed=42, include_function_data=False)

    worker._MonteCarlo__sim_producer(
        _SimMonitor(initial_count=0, n_simulations=4, start_time=time()),
        _ALockNobodyContends(),
        _AnEventNobodySet(),
    )

    assert _sampled_inputs(worker) == _sampled_inputs(serial)
