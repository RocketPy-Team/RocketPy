from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fixture_name", ["monte_carlo_calisto", "monte_carlo_calisto_pre_loaded"]
)
def test_the_shared_fixtures_keep_their_logs_in_tmp_path(fixture_name, request):
    """The shared Monte Carlo fixtures write their logs inside ``tmp_path``."""
    study = request.getfixturevalue(fixture_name)
    tmp_path = request.getfixturevalue("tmp_path").resolve()

    for log in (study.filename, study.input_file, study.output_file, study.error_file):
        assert Path(log).resolve().parent == tmp_path, log
