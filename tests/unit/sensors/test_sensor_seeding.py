"""Determinism tests for seeded sensor noise (additive, isolated).

Sensor measurement noise is drawn from a per-instance ``numpy.random.Generator``
created from the new ``seed`` argument, instead of the process-global
``numpy.random``. A seed makes the noise reproducible for a given input and keeps
it independent of the global RNG state, so it stays deterministic under parallel
or forked execution. This addresses #1042.

The tests build sensors directly and do not touch the existing fixtures or the
inherited tests. Noise is sampled on a fixed grid, so a fixed number of
sequential draws is a faithful stand-in for a run of a given length.
"""

import json
from types import SimpleNamespace

import numpy as np

from rocketpy._encoders import RocketPyEncoder
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.barometer import Barometer
from rocketpy.sensors.gnss_receiver import GnssReceiver
from rocketpy.sensors.gyroscope import Gyroscope


def _accelerometer(seed):
    # Non-zero white noise and random walk so the draws actually exercise the RNG.
    return Accelerometer(
        sampling_rate=10,
        noise_density=1.0,
        noise_variance=1.0,
        random_walk_density=0.5,
        random_walk_variance=1.0,
        seed=seed,
    )


def _noise_sequence(sensor, n=16):
    return [tuple(sensor.apply_noise(Vector([0.0, 0.0, 0.0]))) for _ in range(n)]


def test_same_seed_is_reproducible():
    assert _noise_sequence(_accelerometer(42)) == _noise_sequence(_accelerometer(42))


def test_different_seeds_decorrelate():
    assert _noise_sequence(_accelerometer(1)) != _noise_sequence(_accelerometer(2))


def test_noise_independent_of_global_numpy_rng():
    # Perturbing the process-global RNG must not change a seeded sensor's noise.
    # This is the regression guard for the original bug (noise drawn from the
    # global ``np.random``).
    np.random.seed(0)
    first = _noise_sequence(_accelerometer(7))
    np.random.seed(999)
    _ = [np.random.random() for _ in range(1000)]
    second = _noise_sequence(_accelerometer(7))
    assert first == second


def test_seeded_sensor_does_not_consume_global_rng():
    # A seeded sensor draws only from its own generator, leaving the global RNG
    # position untouched, so it cannot contaminate other code or a forked worker.
    np.random.seed(0)
    position_before = np.random.get_state()[2]
    _noise_sequence(_accelerometer(7))
    position_after = np.random.get_state()[2]
    assert position_before == position_after


def _gnss_measurements(seed, n=8):
    gnss = GnssReceiver(
        sampling_rate=1,
        position_accuracy=5.0,
        altitude_accuracy=5.0,
        seed=seed,
    )
    # Minimal launch-frame state: 100 m up with an identity attitude quaternion.
    state = np.array([0, 0, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    environment = SimpleNamespace(latitude=0.0, longitude=0.0, earth_radius=6.371e6)
    measurements = []
    for _ in range(n):
        gnss.measure(
            0.0,
            u=state,
            relative_position=Vector([0.0, 0.0, 0.0]),
            environment=environment,
        )
        measurements.append(gnss.measurement)
    return measurements


def test_gnss_noise_is_seeded_and_reproducible():
    assert _gnss_measurements(5) == _gnss_measurements(5)
    assert _gnss_measurements(5) != _gnss_measurements(6)


def test_seed_survives_serialization_round_trip():
    """to_dict exposes the seed and from_dict restores it, across all sensor types.

    Sensors serialize through the JSON encoder, which turns the inertial sensors'
    Vector fields into lists, so this exercises the round trip the same way the
    library actually saves and loads them.
    """
    cases = [
        (
            Accelerometer(
                sampling_rate=10, noise_density=1.0, noise_variance=1.0, seed=11
            ),
            11,
        ),
        (
            Gyroscope(sampling_rate=10, noise_density=1.0, noise_variance=1.0, seed=22),
            22,
        ),
        (
            Barometer(sampling_rate=10, noise_density=1.0, noise_variance=1.0, seed=33),
            33,
        ),
        (
            GnssReceiver(
                sampling_rate=1, position_accuracy=5.0, altitude_accuracy=5.0, seed=44
            ),
            44,
        ),
    ]
    for sensor, seed in cases:
        assert sensor.to_dict()["seed"] == seed
        data = json.loads(json.dumps(sensor.to_dict(), cls=RocketPyEncoder))
        assert type(sensor).from_dict(data).to_dict()["seed"] == seed


def test_from_dict_defaults_seed_to_none_when_absent():
    """Dicts serialized before this change (no seed key) still load, seed None."""
    data = GnssReceiver(
        sampling_rate=1, position_accuracy=5.0, altitude_accuracy=5.0, seed=44
    ).to_dict()
    del data["seed"]
    assert GnssReceiver.from_dict(data).to_dict()["seed"] is None
