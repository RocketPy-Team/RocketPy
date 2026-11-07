import json
import threading
from time import time

import pytest

from rocketpy.simulation.monte_carlo import MonteCarlo, _SimMonitor


def _indices(analysis):
    with open(analysis.output_file, "r", encoding="utf-8") as written:
        return sorted(json.loads(line)["index"] for line in written if line.strip())


def _a_study(tmp_path, stem, environment, rocket, flight):
    return MonteCarlo(
        filename=str(tmp_path / stem),
        environment=environment,
        rocket=rocket,
        flight=flight,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_a_run_numbers_its_simulations_from_zero(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path, parallel
):
    # Serial used to write 1, 2, 3 while parallel wrote 0, 1, 2, so the same
    # simulation had two names depending on how the run was started.
    analysis = _a_study(
        tmp_path,
        f"study-{parallel}",
        stochastic_environment,
        stochastic_calisto,
        stochastic_flight,
    )

    analysis.simulate(
        number_of_simulations=3,
        append=False,
        parallel=parallel,
        n_workers=2 if parallel else None,
    )

    assert _indices(analysis) == [0, 1, 2]


def test_both_modes_agree_on_what_a_simulation_is_called(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    one = _a_study(
        tmp_path,
        "serial",
        stochastic_environment,
        stochastic_calisto,
        stochastic_flight,
    )
    other = _a_study(
        tmp_path, "para", stochastic_environment, stochastic_calisto, stochastic_flight
    )

    one.simulate(number_of_simulations=3, append=False, parallel=False)
    other.simulate(number_of_simulations=3, append=False, parallel=True, n_workers=2)

    assert _indices(one) == _indices(other)


def test_an_appended_run_carries_on_from_the_last_index(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    analysis = _a_study(
        tmp_path, "study", stochastic_environment, stochastic_calisto, stochastic_flight
    )

    analysis.simulate(number_of_simulations=2, append=False)
    analysis.simulate(number_of_simulations=4, append=True)

    assert _indices(analysis) == [0, 1, 2, 3]


def test_a_serial_failure_names_the_simulation_the_way_parallel_would(
    stochastic_environment,
    stochastic_calisto,
    stochastic_flight,
    tmp_path,
    monkeypatch,
    capsys,
):
    """A run that fails reports the index the other mode would have used."""
    # The message is the only place the numbering reaches whoever ran it, so
    # it can disagree with the logs without anything else noticing.
    analysis = _a_study(
        tmp_path,
        "failing",
        stochastic_environment,
        stochastic_calisto,
        stochastic_flight,
    )
    attempts = []
    ran = MonteCarlo._MonteCarlo__run_single_simulation

    def fails_on_the_second(self):
        attempts.append(None)
        if len(attempts) == 2:
            raise RuntimeError("the flight would not run")
        return ran(self)

    monkeypatch.setattr(
        MonteCarlo, "_MonteCarlo__run_single_simulation", fails_on_the_second
    )

    with pytest.raises(RuntimeError, match="the flight would not run"):
        analysis.simulate(number_of_simulations=3, append=False)

    assert "Error on iteration 1:" in capsys.readouterr().out


def test_claim_next_index_hands_out_each_index_once():
    """Claimers racing each other get range(n) between them and nothing past it."""
    n_simulations, n_claimers = 200, 8
    monitor = _SimMonitor(0, n_simulations, time())
    ready = threading.Barrier(n_claimers)
    guard = threading.Lock()
    claimed = []

    def claim_until_empty():
        ready.wait()
        mine = []
        while (index := monitor.claim_next_index()) is not None:
            mine.append(index)
        with guard:
            claimed.extend(mine)

    workers = [threading.Thread(target=claim_until_empty) for _ in range(n_claimers)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert len(claimed) == n_simulations
    assert len(set(claimed)) == n_simulations
    assert sorted(claimed) == list(range(n_simulations))
    assert max(claimed) < n_simulations


def test_claim_next_index_is_empty_once_the_target_is_reached():
    """The control: a checkpoint that already holds the target claims nothing."""
    monitor = _SimMonitor(3, 3, time())

    assert monitor.claim_next_index() is None
    assert monitor.count == 3


def test_a_collector_cannot_take_over_the_simulation_index(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """The index names the seed a row was drawn with, so nothing else sets it."""
    # Set past the constructor, so this pins the row and not the validation.
    analysis = _a_study(
        tmp_path, "study", stochastic_environment, stochastic_calisto, stochastic_flight
    )
    analysis.export_list = []
    analysis._export_config = {}
    # The row also carries which root drew it, which simulate() would have set.
    analysis._MonteCarlo__root_state = (42, (), 4, 0)
    analysis.data_collector = {"index": lambda _flight: 999}

    row = json.loads(analysis._MonteCarlo__evaluate_flight_outputs(None, 7))

    assert row["index"] == 7
