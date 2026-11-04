from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fixture_name", ["monte_carlo_calisto", "monte_carlo_calisto_pre_loaded"]
)
def test_the_shared_fixtures_write_outside_the_working_directory(fixture_name, request):
    """A bare filename resolves against wherever pytest was started.

    Both shared fixtures passed one, so a run left three zero-byte logs in the
    repository root with nothing ignoring them. ``filename`` is what the logs
    derive from; ``import_results`` repoints the three file attributes at what
    it read, which is why they are not what this checks.
    """
    study = request.getfixturevalue(fixture_name)
    stem = Path(study.filename).resolve()

    assert stem.is_absolute()
    assert Path.cwd().resolve() not in stem.parents, f"{stem} is under the tree"
