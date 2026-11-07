import os
from types import SimpleNamespace

import pytest

from rocketpy.simulation.monte_carlo import (
    MonteCarlo,
    _refuse_a_worker_that_did_not_finish,
)


def _worker(exitcode):
    return SimpleNamespace(exitcode=exitcode)


def test_workers_that_all_exited_cleanly_are_accepted():
    """A fleet that all exited zero raises nothing."""
    _refuse_a_worker_that_did_not_finish([_worker(0), _worker(0)])


def test_a_worker_killed_by_a_signal_is_refused():
    """A negative exit code names the worker and the signal that ended it."""
    with pytest.raises(RuntimeError, match=r"worker 1 with exit code -9"):
        _refuse_a_worker_that_did_not_finish([_worker(0), _worker(-9)])


def test_a_worker_that_exited_nonzero_is_refused():
    """A positive exit code is refused the same way a signal is."""
    with pytest.raises(RuntimeError, match=r"worker 0 with exit code 1"):
        _refuse_a_worker_that_did_not_finish([_worker(1), _worker(0)])


def test_every_unfinished_worker_is_named():
    """The message names each unfinished worker and leaves the clean ones out."""
    with pytest.raises(RuntimeError) as raised:
        _refuse_a_worker_that_did_not_finish([_worker(-9), _worker(0), _worker(3)])

    assert "worker 0" in str(raised.value)
    assert "worker 2" in str(raised.value)
    assert "worker 1" not in str(raised.value)


@pytest.mark.parametrize("exitcode", [None, -15, 2])
def test_anything_but_a_clean_exit_is_refused(exitcode):
    """``None`` counts as unfinished, not as finished."""
    with pytest.raises(RuntimeError):
        _refuse_a_worker_that_did_not_finish([_worker(exitcode), _worker(0)])


def _leave_without_recording(_flight):
    """Ends the worker the way a kill or an out-of-memory exit does.

    ``os._exit`` rather than a signal, since ``SIGKILL`` is POSIX-only, and
    reached through the data collector rather than a patched method, since a
    ``spawn`` platform re-imports the module and would not see the patch.
    """
    os._exit(1)


def test_a_worker_that_leaves_early_does_not_pass_as_a_finished_run(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """A worker leaving through ``os._exit`` makes ``simulate`` raise."""
    # The event the workers report through is set by their own handler, and
    # this one leaves without running it, so the run used to return as though
    # it had done every simulation it was asked for.
    analysis = MonteCarlo(
        filename=str(tmp_path / "study"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
        data_collector={"leave": _leave_without_recording},
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        analysis.simulate(
            number_of_simulations=6, append=False, parallel=True, n_workers=2
        )
