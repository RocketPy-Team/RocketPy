"""Determinism tests for seeded parachute pressure noise (#1091).

Pressure noise is drawn from a per-instance ``numpy.random.Generator`` instead
of the process-global ``numpy.random``. The seed is private: only
``StochasticParachute`` and ``from_dict`` set it, after construction.
"""

import numpy as np
import pytest

from rocketpy import Parachute
from rocketpy.stochastic import StochasticParachute


def _parachute(seed, noise=(0, 8.3, 0.5), name="main"):
    parachute = Parachute(
        name=name,
        cd_s=10.0,
        trigger="apogee",
        sampling_rate=100,
        noise=noise,
    )
    if seed is not None:
        parachute._set_noise_seed(seed)
    return parachute


def _noise_sequence(parachute, n=16):
    # Include the initial sample stored at construction, then draw from
    # ``noise_function`` the way Flight does while sampling the trigger.
    samples = [parachute.noise_signal[0][1]]
    for _ in range(n):
        value = parachute.noise_function()
        parachute.noise_signal.append([0.0, value])
        samples.append(value)
    return samples


def test_same_seed_is_reproducible():
    assert _noise_sequence(_parachute(42)) == _noise_sequence(_parachute(42))


def test_different_seeds_decorrelate():
    assert _noise_sequence(_parachute(1)) != _noise_sequence(_parachute(2))


def test_default_unseeded_still_draws_noise():
    """seed=None keeps the default path working with non-zero noise."""
    parachute = _parachute(None)
    samples = _noise_sequence(parachute, n=8)
    assert any(sample != 0.0 for sample in samples)


def test_noise_independent_of_global_numpy_rng():
    np.random.seed(0)
    first = _noise_sequence(_parachute(7))
    np.random.seed(999)
    _ = [np.random.random() for _ in range(1000)]
    second = _noise_sequence(_parachute(7))
    assert first == second


def test_seeded_parachute_does_not_consume_global_rng():
    np.random.seed(0)
    position_before = np.random.get_state()[2]
    _noise_sequence(_parachute(7))
    position_after = np.random.get_state()[2]
    assert position_before == position_after


def test_zero_noise_still_returns_zero():
    parachute = _parachute(42, noise=(0, 0, 0))
    assert parachute.noise_function() == 0.0


def test_seed_survives_serialization_round_trip():
    original = _parachute(11)
    restored = Parachute.from_dict(original.to_dict())
    assert restored.to_dict()["seed"] == 11
    assert _noise_sequence(restored) == _noise_sequence(_parachute(11))


def test_from_dict_defaults_seed_to_none_when_absent():
    data = _parachute(11).to_dict()
    del data["seed"]
    restored = Parachute.from_dict(data)
    assert restored.to_dict()["seed"] is None
    assert any(sample != 0.0 for sample in _noise_sequence(restored))


def test_seed_is_not_a_constructor_argument():
    with pytest.raises(TypeError, match="seed"):
        Parachute(  # pylint: disable=unexpected-keyword-arg
            name="main",
            cd_s=10.0,
            trigger="apogee",
            sampling_rate=100,
            seed=1,
        )


def test_set_noise_seed_redraws_the_initial_noise_sample():
    """The sample taken at construction came from an unseeded generator."""
    seeded = _parachute(5)
    assert seeded.noise_signal[0][1] == _parachute(5).noise_signal[0][1]
    assert len(seeded.noise_signal) == 1


def test_stochastic_parachute_threads_seed_into_created_object():
    template = _parachute(None, noise=(0, 8.3, 0.5))
    stochastic = StochasticParachute(template)
    stochastic._set_stochastic(seed=123)
    first = stochastic.create_object()
    stochastic._set_stochastic(seed=123)
    second = stochastic.create_object()
    assert first._seed is not None
    assert first._seed == second._seed
    assert _noise_sequence(first) == _noise_sequence(second)

    stochastic._set_stochastic(seed=456)
    other = stochastic.create_object()
    assert other._seed != first._seed
    assert _noise_sequence(other) != _noise_sequence(first)


# Pinned from a StochasticParachute seeded with 123, so a change to how the
# noise seed is derived or applied moves them. Same seed, same sequence.
_PINNED = {
    "main": (
        6639568465033922193,
        [
            -3.78616199388814,
            -13.452710525868634,
            -4.673764102705483,
            -10.801711039361278,
            -8.053692181461434,
            -16.713116227374677,
            -21.497333383014304,
        ],
    ),
    "drogue": (
        6152077226054995121,
        [
            -1.1998384111241425,
            5.348258635026273,
            2.8774014645733943,
            4.406981770690811,
            12.553121676963602,
            7.348730317452494,
            -4.59114813843395,
        ],
    ),
}


@pytest.mark.parametrize("name", ["main", "drogue"])
def test_stochastic_parachute_noise_sequence_is_pinned(name):
    stochastic = StochasticParachute(
        Parachute(
            name=name,
            cd_s=10.0,
            trigger="apogee",
            sampling_rate=100,
            noise=(0, 8.3, 0.5),
        )
    )
    stochastic._set_stochastic(seed=123)

    built = stochastic.create_object()

    seed, sequence = _PINNED[name]
    assert built._seed == seed
    assert _noise_sequence(built, n=6) == pytest.approx(sequence, rel=1e-12)
