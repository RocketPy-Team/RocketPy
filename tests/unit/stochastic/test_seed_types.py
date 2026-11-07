import numpy as np
import pytest

from rocketpy.stochastic.stochastic_model import (
    _names_as_spawn_key,
    _sampler_seed,
)
from rocketpy.tools import _seed_sequence_to_int


def _a_worker_seed(index=0, workers=2):
    # What MonteCarlo.__run_in_parallel spawns and hands to each worker, which
    # passes it straight to environment/rocket/flight._set_stochastic.
    return np.random.SeedSequence().spawn(workers)[index]


def test_the_seed_type_a_worker_is_handed_is_accepted(stochastic_calisto):
    stochastic_calisto._set_stochastic(_a_worker_seed())

    stochastic_calisto.create_object()


def test_a_parachute_derives_its_noise_seed_from_a_worker_seed(
    stochastic_main_parachute,
):
    stochastic_main_parachute._set_stochastic(_a_worker_seed())

    assert stochastic_main_parachute.create_object().noise[2] is not None


def test_two_workers_do_not_share_a_sampler_stream():
    first, second = np.random.SeedSequence(7).spawn(2)
    # They come off one root, so they carry the same entropy and differ only in
    # spawn_key. Reading the entropy alone would put both on one stream.
    assert first.entropy == second.entropy

    assert _sampler_seed(first, ("__list_choice__",)) != _sampler_seed(
        second, ("__list_choice__",)
    )


def test_a_caller_seed_sequence_is_not_consumed():
    root = np.random.SeedSequence(42)

    _sampler_seed(root, ("__list_choice__",))

    assert root.n_children_spawned == 0
    assert root.spawn(1)[0].spawn_key == (0,)


def test_the_same_seed_sequence_twice_gives_the_same_sampler_seed():
    root = np.random.SeedSequence(42)

    first = _sampler_seed(root, ("pressure_noise", "main"))
    second = _sampler_seed(root, ("pressure_noise", "main"))

    assert first == second


@pytest.mark.parametrize("seed", [42, 7, [1, 2, 3]])
@pytest.mark.parametrize("names", [("__list_choice__",), ("pressure_noise", "main")])
def test_a_seed_that_is_not_a_sequence_reaches_numpy_untouched(seed, names):
    # The control. Every fixed-seed baseline in the suite was recorded through
    # this path, so anything but a SeedSequence has to arrive as it always did.
    # Compared with the expression rather than with a recorded number, which
    # would go red on a NumPy release instead of on a change of ours.
    unchanged = np.random.SeedSequence(
        entropy=seed, spawn_key=_names_as_spawn_key(tuple(sorted(names)))
    )

    assert _sampler_seed(seed, names) == _seed_sequence_to_int(unchanged)


def test_no_seed_still_means_no_seed():
    # None is left out above on purpose: it asks NumPy for fresh entropy, so
    # two calls must not agree, and comparing one against another would be
    # asserting the opposite of what an unseeded run promises.
    first = _sampler_seed(None, ("__list_choice__",))
    second = _sampler_seed(None, ("__list_choice__",))

    assert first != second
