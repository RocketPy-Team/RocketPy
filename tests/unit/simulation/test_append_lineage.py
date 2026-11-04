"""Validate checkpoints before allowing Monte Carlo to append any rows."""

import json
from types import SimpleNamespace

import pytest

from rocketpy import MonteCarlo


@pytest.fixture
def checkpoint(tmp_path):
    """Create paired logs without executing a flight."""
    study = MonteCarlo(
        str(tmp_path / "study"), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    study._MonteCarlo__study_seed = 42
    study._completed_indices = set()
    study._index_offset = 0
    _write(study.input_file, [{"index": 0, "seed": 42}, {"index": 1, "seed": 42}])
    _write(study.output_file, [{"index": 0, "apogee": 10}, {"index": 1, "apogee": 20}])
    return study


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_checkpoint_recovers_the_recorded_seed(checkpoint):
    """A fresh object's entropy must not replace the checkpoint's seed."""
    checkpoint._MonteCarlo__study_seed = 99

    checkpoint._MonteCarlo__continue_study(None)

    assert checkpoint._MonteCarlo__study_seed == 42
    assert checkpoint._completed_indices == {0, 1}


@pytest.mark.parametrize("seed", [42, None])
def test_checkpoint_accepts_a_matching_or_omitted_seed(checkpoint, seed):
    """Explicitly repeating the seed and reading it back are equivalent."""
    checkpoint._MonteCarlo__continue_study(seed)

    assert checkpoint._completed_indices == {0, 1}


def test_checkpoint_rejects_a_different_seed(checkpoint):
    """Appending cannot mix two seeded studies."""
    checkpoint._MonteCarlo__study_seed = 7

    with pytest.raises(ValueError, match="different root seed"):
        checkpoint._MonteCarlo__continue_study(7)


@pytest.mark.parametrize("seed", [None, True, -1, 1.5, "42", [42], {}])
def test_checkpoint_rejects_invalid_recorded_seeds(checkpoint, seed):
    """A damaged seed must be rejected before opening files for append."""
    _write(
        checkpoint.input_file, [{"index": 0, "seed": seed}, {"index": 1, "seed": seed}]
    )

    with pytest.raises(ValueError, match="invalid seed"):
        checkpoint._MonteCarlo__continue_study(None)


def test_checkpoint_rejects_mixed_seeds(checkpoint):
    """Matching indices do not justify mixing different input streams."""
    _write(checkpoint.input_file, [{"index": 0, "seed": 42}, {"index": 1, "seed": 7}])

    with pytest.raises(ValueError, match="more than one study"):
        checkpoint._MonteCarlo__continue_study(None)


def test_checkpoint_rejects_missing_seeds_after_seeded_rows(checkpoint):
    """Only an unchanged legacy prefix may omit the study seed."""
    _write(checkpoint.input_file, [{"index": 0, "seed": 42}, {"index": 1}])

    with pytest.raises(ValueError, match="Legacy rows must precede"):
        checkpoint._MonteCarlo__continue_study(None)


@pytest.mark.parametrize("numbers", [[0, 2], [2, 0], [1, 2]])
def test_checkpoint_accepts_holes_and_completion_order(checkpoint, numbers):
    """An interrupted parallel run may have completed later indices first."""
    _write(checkpoint.input_file, [{"index": index, "seed": 42} for index in numbers])
    _write(checkpoint.output_file, [{"index": index} for index in numbers])

    checkpoint._MonteCarlo__continue_study(None)

    assert checkpoint._completed_indices == set(numbers)
    assert checkpoint._index_offset == 0


def test_checkpoint_rejects_duplicate_indices(checkpoint):
    """Repeated indices cannot be silently counted as separate flights."""
    for path in (checkpoint.input_file, checkpoint.output_file):
        _write(path, [{"index": 0, "seed": 42}] * 2)

    with pytest.raises(ValueError, match="duplicate"):
        checkpoint._MonteCarlo__continue_study(None)


@pytest.mark.parametrize("index", [True, 1.0, "1", None, -1])
def test_checkpoint_rejects_invalid_indices(checkpoint, index):
    """Only non-negative integer indices can identify completed simulations."""
    _write(checkpoint.input_file, [{"index": index, "seed": 42}])

    with pytest.raises(ValueError, match="does not number"):
        checkpoint._MonteCarlo__continue_study(None)


def test_checkpoint_rejects_mismatched_logs(checkpoint):
    """A missing half of a committed record needs repair before resuming."""
    _write(checkpoint.output_file, [{"index": 0}])

    with pytest.raises(ValueError, match="same simulations"):
        checkpoint._MonteCarlo__continue_study(None)


def test_checkpoint_rejects_unreadable_rows(checkpoint):
    """A truncated row cannot be mistaken for an empty checkpoint."""
    checkpoint.input_file.write_text('{"index":', encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be read"):
        checkpoint._MonteCarlo__continue_study(None)


def test_blank_lines_do_not_count_as_simulations(checkpoint):
    """Whitespace does not change which completed indices are skipped."""
    with checkpoint.input_file.open("a", encoding="utf-8") as log:
        log.write("\n   \n")

    checkpoint._MonteCarlo__continue_study(None)

    assert checkpoint._completed_indices == {0, 1}


@pytest.mark.parametrize("key", ["index", "seed", "run_root"])
def test_collectors_cannot_overwrite_checkpoint_metadata(tmp_path, key):
    """Reserved names must not silently replace a collector's output."""
    with pytest.raises(ValueError, match="Invalid"):
        MonteCarlo(
            str(tmp_path / "study"),
            None,
            None,
            None,
            data_collector={key: lambda flight: 1},
        )
