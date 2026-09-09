import json
import os
from time import time

import multiprocess
import numpy as np
import pytest

from rocketpy.stochastic import StochasticEnvironment
from rocketpy.simulation.monte_carlo import (
    MonteCarlo,
    _root_seed_sequence,
    _root_state_of,
    _seed_of_simulation,
    _seed_sequence_to_int,
    _SimMonitor,
)


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
    root = _root_state_of(_root_seed_sequence(42))
    return _seed_sequence_to_int(_seed_of_simulation(root, index))


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
    )
    analysis.simulate(
        number_of_simulations=count,
        append=False,
        parallel=parallel,
        n_workers=workers,
        random_seed=seed,
    )
    return analysis


@pytest.fixture(name="models")
def _models(stochastic_environment, stochastic_calisto, stochastic_flight):
    return stochastic_environment, stochastic_calisto, stochastic_flight


# --------------------------------------------------------------- the derivation


@pytest.mark.parametrize("seed", [42, None, [1, 2, 3], np.random.SeedSequence(7)])
def test_a_simulation_gets_the_child_spawn_would_have_given_it(seed):
    # The whole point of deriving one index directly: it has to be the same
    # child, bit for bit, as spawning every index before it would produce.
    root = _root_seed_sequence(seed)
    state = _root_state_of(root)
    spawned = np.random.SeedSequence(**root.state).spawn(6)

    for index, expected in enumerate(spawned):
        assert np.array_equal(
            expected.generate_state(4),
            _seed_of_simulation(state, index).generate_state(4),
        )


def test_deriving_an_index_costs_nothing_for_the_ones_before_it():
    state = _root_state_of(_root_seed_sequence(42))

    far = _seed_of_simulation(state, 1_000_000)

    assert far.spawn_key == (1_000_000,)


def test_two_indices_do_not_share_a_stream():
    state = _root_state_of(_root_seed_sequence(42))

    first = _seed_of_simulation(state, 0).generate_state(4)
    second = _seed_of_simulation(state, 1).generate_state(4)

    assert not np.array_equal(first, second)


# ------------------------------------------------------------------- the root


def test_a_caller_seed_sequence_is_not_consumed():
    given = np.random.SeedSequence(42)

    _root_seed_sequence(given).spawn(5)

    assert given.n_children_spawned == 0


def test_a_caller_cannot_move_the_run_by_editing_the_list_it_passed():
    # SeedSequence keeps a sequence entropy by reference, so without a copy of
    # its own the run would follow whatever the caller did to that list next.
    given = [1, 2, 3]
    state = _root_state_of(_root_seed_sequence(given))

    before = _seed_of_simulation(state, 0).generate_state(4)
    given[0] = 999

    assert np.array_equal(_seed_of_simulation(state, 0).generate_state(4), before)


@pytest.mark.parametrize("given", [np.random.default_rng(42), np.random.PCG64(42)])
def test_a_generator_is_refused_rather_than_read(given):
    # Using a consume-on-use object as an immutable seed cannot mean what it
    # says, so it is refused instead of quietly meaning something else.
    with pytest.raises(TypeError, match="random_seed must be"):
        _root_seed_sequence(given)


def test_an_unusable_seed_costs_the_previous_run_nothing(models, tmp_path):
    analysis = _a_run(tmp_path, "study", models, seed=42, count=2)
    kept = _sampled_inputs(analysis)

    with pytest.raises(TypeError):
        analysis.simulate(2, append=False, random_seed=np.random.default_rng(1))

    assert _sampled_inputs(analysis) == kept


# ------------------------------------------------------------- what a run gives


def test_one_seed_gives_one_set_of_inputs(models, tmp_path):
    first = _a_run(tmp_path, "first", models, seed=42)
    again = _a_run(tmp_path, "again", models, seed=42)

    assert _sampled_inputs(first) == _sampled_inputs(again)


def test_another_seed_gives_another_set(models, tmp_path):
    # The control. Without this the test above passes on a run that ignores
    # the seed entirely.
    first = _a_run(tmp_path, "first", models, seed=42)
    other = _a_run(tmp_path, "other", models, seed=7)

    assert _sampled_inputs(first) != _sampled_inputs(other)


@pytest.mark.skipif(os.cpu_count() < 4, reason="needs four workers to be four")
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


def test_an_appended_run_carries_the_same_stream_on(models, tmp_path):
    whole = _a_run(tmp_path, "whole", models, seed=42, count=4)

    part = _a_run(tmp_path, "part", models, seed=42, count=2)
    part.simulate(4, append=True, random_seed=42)

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
    )
    worker.simulate(number_of_simulations=0, append=False, random_seed=42)

    worker._MonteCarlo__sim_producer(
        _SimMonitor(initial_count=0, n_simulations=4, start_time=time()),
        _ALockNobodyContends(),
        _AnEventNobodySet(),
    )

    assert _sampled_inputs(worker) == _sampled_inputs(serial)
