import json

import pytest

from rocketpy.simulation.monte_carlo import (
    MonteCarlo,
    _root_digest,
    _SIMULATION_ROOT_KEY,
)


def _study(tmp_path, stem, models):
    environment, rocket, flight = models
    return MonteCarlo(
        filename=str(tmp_path / stem),
        environment=environment,
        rocket=rocket,
        flight=flight,
    )


def _drawn(analysis):
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    return {row["index"]: row.get("mass") for row in rows}


@pytest.fixture(name="models")
def _models(stochastic_environment, stochastic_calisto, stochastic_flight):
    return stochastic_environment, stochastic_calisto, stochastic_flight


def test_a_row_says_which_root_drew_it(models, tmp_path):
    """Every input row carries the root, so the log describes itself."""
    analysis = _study(tmp_path, "study", models)

    analysis.simulate(2, append=False, random_seed=42)

    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    assert all(row[_SIMULATION_ROOT_KEY]["entropy"] == 42 for row in rows)


def test_an_append_with_no_seed_carries_on_from_the_rows(models, tmp_path):
    """The ordinary resume: no seed given, the stream continues anyway."""
    whole = _study(tmp_path, "whole", models)
    whole.simulate(4, append=False, random_seed=42)

    part = _study(tmp_path, "part", models)
    part.simulate(2, append=False, random_seed=42)
    part.simulate(4, append=True)

    assert _drawn(part) == _drawn(whole)


def test_a_fresh_object_can_continue_a_study(models, tmp_path):
    """A notebook restart is a new object over the same files."""
    first = _study(tmp_path, "study", models)
    first.simulate(2, append=False, random_seed=42)

    second = _study(tmp_path, "study", models)
    second.simulate(4, append=True)

    assert sorted(_drawn(second)) == [0, 1, 2, 3]


def test_appending_with_another_seed_is_refused(models, tmp_path):
    """Two roots in one file is the thing this exists to prevent."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)

    with pytest.raises(ValueError, match="different root"):
        analysis.simulate(4, append=True, random_seed=7)


def test_appending_with_the_same_seed_is_allowed(models, tmp_path):
    """The control. Saying the seed again is not a mismatch."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)

    analysis.simulate(4, append=True, random_seed=42)

    assert sorted(_drawn(analysis)) == [0, 1, 2, 3]


def test_a_log_holding_two_studies_is_refused(models, tmp_path):
    """Rows that disagree are two studies, and neither is safe to continue."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    rows[1][_SIMULATION_ROOT_KEY]["entropy"] = 999
    with open(analysis.input_file, "w", encoding="utf-8") as rewritten:
        for row in rows:
            rewritten.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="more than one study"):
        analysis.simulate(4, append=True)


def test_a_log_whose_rows_carry_no_root_is_refused(models, tmp_path):
    """A study from before this release cannot be shown to be one study."""
    # Measured before this was refused: the append read the log as empty,
    # started a root of its own, and left two lineages in the one file.
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    for row in rows:
        row.pop(_SIMULATION_ROOT_KEY)
    with open(analysis.input_file, "w", encoding="utf-8") as rewritten:
        for row in rows:
            rewritten.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="does not say which root"):
        analysis.simulate(4, append=True)


def _rewrite(path, replacement):
    """Put ``replacement`` under the root key of every row of one log."""
    with open(path, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    for row in rows:
        row[_SIMULATION_ROOT_KEY] = replacement(row[_SIMULATION_ROOT_KEY])
    with open(path, "w", encoding="utf-8") as rewritten:
        for row in rows:
            rewritten.write(json.dumps(row) + "\n")


def _rewrite_roots(analysis, damage):
    """Damage the recorded root, leaving the two logs still naming one study.

    The output rows hold the root's digest, so they are restamped rather than
    damaged: otherwise the two logs disagree and that is refused first.
    """
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        first = json.loads(next(line for line in written if line.strip()))
    damaged = damage(first[_SIMULATION_ROOT_KEY])
    _rewrite(analysis.input_file, lambda _root: damaged)
    if isinstance(damaged, dict):
        _rewrite(analysis.output_file, lambda _digest: _root_digest(damaged))


def test_a_log_whose_root_is_null_is_refused(models, tmp_path):
    """A null root is no more of a lineage than no root at all."""
    # It read as an empty log, so an append started a second root behind the
    # damaged rows, which is the case this check exists to prevent.
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)

    _rewrite_roots(analysis, lambda root: None)

    with pytest.raises(ValueError, match="does not say which root"):
        analysis.simulate(4, append=True)


@pytest.mark.parametrize(
    "damage",
    [
        lambda root: {**root, "entropy": None},
        lambda root: {**root, "pool_size": 0},
        lambda root: {**root, "pool_size": 2},
        lambda root: {**root, "n_children_spawned": -1},
        lambda root: {**root, "spawn_key": [True]},
        lambda root: {key: value for key, value in root.items() if key != "entropy"},
        lambda root: {**root, "unexpected": 1},
    ],
    ids=[
        "null entropy",
        "no pool",
        "pool numpy refuses",
        "negative base",
        "bool key",
        "short",
        "long",
    ],
)
def test_a_root_that_no_stream_can_be_rebuilt_from_is_refused(models, tmp_path, damage):
    """Rows agreeing on a root does not make it a root that can be resumed."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)

    _rewrite_roots(analysis, damage)

    with pytest.raises(ValueError, match="rebuilt from"):
        analysis.simulate(4, append=True)


def test_an_empty_log_is_not_a_log_that_cannot_be_checked(models, tmp_path):
    """The control. Nothing recorded yet is a fresh start, not a refusal."""
    analysis = _study(tmp_path, "study", models)

    analysis.simulate(2, append=True, random_seed=42)

    assert sorted(_drawn(analysis)) == [0, 1]


def test_a_seed_given_as_a_sequence_is_recorded_as_one(models, tmp_path):
    """A sequence is a documented seed, so the row has to carry it whole."""
    analysis = _study(tmp_path, "study", models)

    analysis.simulate(2, append=False, random_seed=[1, 2, 3])

    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    assert all(row[_SIMULATION_ROOT_KEY]["entropy"] == [1, 2, 3] for row in rows)


def test_a_sequence_seed_still_continues_the_same_study(models, tmp_path):
    """And reads back, since a root that cannot be compared refuses the append."""
    whole = _study(tmp_path, "whole", models)
    whole.simulate(4, append=False, random_seed=[1, 2, 3])

    part = _study(tmp_path, "part", models)
    part.simulate(2, append=False, random_seed=[1, 2, 3])
    part.simulate(4, append=True)

    assert _drawn(part) == _drawn(whole)


def test_blank_lines_between_rows_do_not_hide_the_root(models, tmp_path):
    """A gap in the file is not a row, and not a study without a root either."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = written.read().splitlines()
    with open(analysis.input_file, "w", encoding="utf-8") as spaced:
        for row in rows:
            spaced.write(row + "\n\n")

    analysis.simulate(4, append=True)

    assert sorted(_drawn(analysis)) == [0, 1, 2, 3]


def test_a_row_that_cannot_be_read_is_refused(models, tmp_path):
    """A row that will not parse leaves nothing to check the root against."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "a", encoding="utf-8") as damaged:
        damaged.write("{ this was cut off\n")

    with pytest.raises(ValueError, match="cannot be read"):
        analysis.simulate(4, append=True)


def test_an_output_log_from_another_study_is_refused(models, tmp_path):
    """Matching indices do not make two files the two halves of one run."""
    # Nothing in the indices tells them apart: both studies number their rows
    # from zero, so the root is what says they belong together.
    one = _study(tmp_path, "one", models)
    one.simulate(2, append=False, random_seed=42)
    other = _study(tmp_path, "other", models)
    other.simulate(2, append=False, random_seed=7)
    with open(other.output_file, "r", encoding="utf-8") as theirs:
        stolen = theirs.read()
    with open(one.output_file, "w", encoding="utf-8") as ours:
        ours.write(stolen)

    with pytest.raises(ValueError, match="same root"):
        one.simulate(4, append=True)


def test_a_checkpoint_whose_halves_disagree_is_refused(models, tmp_path):
    """Where to carry on from is not established by one log alone."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.output_file, "r", encoding="utf-8") as written:
        rows = [line for line in written if line.strip()]
    with open(analysis.output_file, "w", encoding="utf-8") as trimmed:
        trimmed.write(rows[0])

    with pytest.raises(ValueError, match="do not record the same simulations"):
        analysis.simulate(4, append=True)


def test_a_collector_cannot_take_over_the_root(models, tmp_path):
    """The root binds the two logs, so nothing outside the run writes it."""
    # Set past the constructor, so this pins the row and not the validation.
    analysis = _study(tmp_path, "study", models)
    analysis.data_collector = {_SIMULATION_ROOT_KEY: lambda _flight: "forged"}

    analysis.simulate(2, append=False, random_seed=42)

    with open(analysis.output_file, "r", encoding="utf-8") as written:
        roots = {
            json.dumps(json.loads(line)[_SIMULATION_ROOT_KEY], sort_keys=True)
            for line in written
            if line.strip()
        }
    assert roots != {'"forged"'}
    assert len(roots) == 1


def test_a_collector_named_after_the_root_is_refused(models, tmp_path):
    """Overwriting it afterwards protects the log but discards the callback."""
    with pytest.raises(ValueError, match="run after the collectors|discarded"):
        MonteCarlo(
            filename=str(tmp_path / "study"),
            environment=models[0],
            rocket=models[1],
            flight=models[2],
            data_collector={_SIMULATION_ROOT_KEY: lambda _flight: 1},
        )


def _rows_of(path):
    with open(path, "r", encoding="utf-8") as written:
        return [line for line in written if line.strip()]


def _rewrite_indices(path, numbers):
    """Renumber a log's rows, leaving everything else in them alone."""
    rows = [json.loads(line) for line in _rows_of(path)]
    for row, number in zip(rows, numbers):
        row["index"] = number
    with open(path, "w", encoding="utf-8") as rewritten:
        for row in rows:
            rewritten.write(json.dumps(row) + "\n")


def test_a_blank_line_in_the_output_log_does_not_move_the_continuation(
    models, tmp_path
):
    """Where to carry on from comes from the records, not the line count."""
    # A fresh object counts physical output lines when it opens the file, so
    # one blank line made an append start at 3 and leave index 2 missing.
    written = _study(tmp_path, "study", models)
    written.simulate(2, append=False, random_seed=42)
    with open(written.output_file, "a", encoding="utf-8") as padded:
        padded.write("\n")

    resumed = _study(tmp_path, "study", models)
    assert resumed.num_of_loaded_sims == 3  # the count this must not trust
    resumed.simulate(4, append=True)

    assert sorted(_drawn(resumed)) == [0, 1, 2, 3]


def test_halves_that_number_the_same_count_differently_are_refused(models, tmp_path):
    """Equal row counts do not make two logs the two halves of one run."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    _rewrite_indices(analysis.output_file, [0, 2])

    with pytest.raises(ValueError, match="do not record the same simulations"):
        analysis.simulate(4, append=True)


@pytest.mark.parametrize(
    "numbers", [[0, 0], [0, 2], [1, 2]], ids=["duplicate", "hole", "no zero"]
)
def test_a_checkpoint_that_is_not_the_runs_first_simulations_is_refused(
    models, tmp_path, numbers
):
    """An append continues a run, so what it continues has to be its start."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    for path in (analysis.input_file, analysis.output_file):
        _rewrite_indices(path, numbers)

    with pytest.raises(ValueError, match="not the run's first"):
        analysis.simulate(4, append=True)


def test_the_order_a_parallel_run_finished_in_is_carried_on_from(models, tmp_path):
    """Rows arrive in completion order, which is not sorted and not wrong."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    for path in (analysis.input_file, analysis.output_file):
        _rewrite_indices(path, [1, 0])

    analysis.simulate(4, append=True)

    assert sorted(_drawn(analysis)) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "number", [True, 1.0, "1", None, -1], ids=["bool", "float", "str", "null", "neg"]
)
def test_a_row_that_does_not_number_its_simulation_is_refused(models, tmp_path, number):
    """Only a non-negative int says which simulation a row holds."""
    # True and 1.0 both equal 1, so either would pass for a record that is not
    # there, and the count they contribute to decides where an append starts.
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    for path in (analysis.input_file, analysis.output_file):
        _rewrite_indices(path, [0, number])

    with pytest.raises(ValueError, match="does not number"):
        analysis.simulate(4, append=True)
