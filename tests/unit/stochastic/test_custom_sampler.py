from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.environment.environment import Environment
from rocketpy.stochastic import StochasticRocket
from rocketpy.stochastic.custom_sampler import CustomSampler
from rocketpy.stochastic.stochastic_model import StochasticModel, _sampler_seed


def test_create_object(stochastic_environment_custom_sampler):
    """Test create object method of StochasticEnvironment class.

    This test checks if the create_object method of the StochasticEnvironment
    class creates a StochasticEnvironment object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        StochasticEnvironment object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_environment_custom_sampler.create_object()
    assert isinstance(obj, Environment)


class _Gaussian(CustomSampler):
    """A sampler of the shape the documentation teaches."""

    def __init__(self, mean, sd):
        self.mean, self.sd = mean, sd
        self.rng = np.random.default_rng()

    def sample(self, n_samples=1):
        return list(self.rng.normal(self.mean, self.sd, n_samples))

    def reset_seed(self, seed=None):
        self.rng = np.random.default_rng(seed)


def _deviates(drawn):
    """The standard normal behind each draw, so samplers with different means
    and spreads can still be compared."""
    return (
        (drawn["mass"] - 14.426) / 0.5,
        (drawn["radius"] - 0.0635) / 0.001,
    )


def _two_sampler_model(calisto_robust):
    return StochasticRocket(
        rocket=calisto_robust,
        mass=_Gaussian(14.426, 0.5),
        radius=_Gaussian(0.0635, 0.001),
    )


def test_two_samplers_do_not_draw_the_same_deviate(calisto_robust):
    """Every sampler on a model used to be reset with the model's own seed, so
    two backed by ``default_rng`` started from the same state and drew the same
    underlying value. Not nearly identical: the same, to every digit."""
    model = _two_sampler_model(calisto_robust)
    model._set_stochastic(4242)

    mass, radius = _deviates(next(model.dict_generator()))

    assert mass != pytest.approx(radius, abs=1e-12)


def test_a_seed_still_reproduces_the_same_samples(calisto_robust):
    """The control. Independence must not have been bought with fresh entropy
    per reseed, which would decorrelate the samplers and lose the seed."""
    model = _two_sampler_model(calisto_robust)

    model._set_stochastic(4242)
    first = _deviates(next(model.dict_generator()))
    model._set_stochastic(4242)
    again = _deviates(next(model.dict_generator()))
    model._set_stochastic(99)
    other = _deviates(next(model.dict_generator()))

    assert first == again
    assert first != other


def test_adding_a_parameter_leaves_the_others_where_they_were(calisto_robust):
    """Seeds are keyed by the input's name, not its position, so declaring one
    more sampler does not move the streams of the ones already there."""
    model = _two_sampler_model(calisto_robust)
    model._set_stochastic(4242)
    before = _deviates(next(model.dict_generator()))

    wider = StochasticRocket(
        rocket=calisto_robust,
        mass=_Gaussian(14.426, 0.5),
        radius=_Gaussian(0.0635, 0.001),
        inertia_11=_Gaussian(6.321, 0.1),
    )
    wider._set_stochastic(4242)
    after = _deviates(next(wider.dict_generator()))

    assert after == before


class _SharedPair:
    """Two wrappers over one generator, as the wind example in the docs does."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.last_seed = None
        self.reset_count = 0

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.last_seed = seed
        self.reset_count += 1

    def draw(self):
        return float(self.rng.normal())


class _SharedWrapper(CustomSampler):
    def __init__(self, shared):
        self.shared = shared

    def sample(self, n_samples=1):
        return [self.shared.draw() for _ in range(n_samples)]

    def reset_seed(self, seed=None):
        self.shared.reset(seed)


def _seed_the_shared_generator_received(declare_second_first):
    shared = _SharedPair()
    first, second = _SharedWrapper(shared), _SharedWrapper(shared)
    inputs = (
        {"wind_y": second, "wind_x": first}
        if declare_second_first
        else {"wind_x": first, "wind_y": second}
    )
    model = StochasticModel(SimpleNamespace(wind_x=0.0, wind_y=0.0), **inputs)
    shared.reset_count = 0  # the constructor has already seeded once
    model._set_stochastic(4242)
    return shared.last_seed, shared.reset_count


def test_a_shared_generator_lands_on_the_same_seed_whatever_the_order():
    """Samplers may share one generator on purpose, and each reset overwrites
    the last, so whichever is reset last decides the stream. Seeding runs in
    sorted order for that reason: the same seed has to mean the same stream
    whichever order the model was written in.

    On the values drawn, not the stream: two wrappers reading one generator
    take successive values, so swapping the declaration swaps which wrapper
    gets which. That is inherent to sharing a generator and is not seeding.
    """
    ordered, reversed_ = (
        _seed_the_shared_generator_received(False),
        _seed_the_shared_generator_received(True),
    )

    assert ordered == reversed_
    assert ordered[1] == 2, "each wrapper still resets the generator it wraps"


def test_two_names_that_a_hash_would_collide_get_different_streams():
    """Keying by a 32-bit hash put these two back on one stream, which is the
    bug this keying exists to prevent. Both are valid identifiers and their
    CRC32 is 1560575156."""
    assert _sampler_seed(4242, "wd4s4xka50") != _sampler_seed(4242, "p56cjcee10")


def test_the_sampler_seed_keeps_the_full_width():
    """128 bits, matching the width the Monte Carlo seeding uses, so a study
    spawning many streams does not run into birthday collisions."""
    assert _sampler_seed(4242, "mass").bit_length() > 64


def test_declaring_a_sampler_does_not_reorder_the_other_inputs(calisto_robust):
    """Seeding is sorted; the validation loop below it is not. Sorting that one
    too would set __dict__ alphabetically and move every tuple's draw."""
    plain = StochasticRocket(rocket=calisto_robust, mass=(14.426, 0.5))
    plain._set_stochastic(42)
    before = next(plain.dict_generator())

    with_sampler = StochasticRocket(
        rocket=calisto_robust, mass=(14.426, 0.5), radius=_Gaussian(0.0635, 0.001)
    )
    with_sampler._set_stochastic(42)
    after = next(with_sampler.dict_generator())

    # `radius` is declared either way, so it is in both. Only its kind changed.
    assert list(after) == list(before)
    assert after["mass"] == before["mass"]


class _GroupedWrapper(_SharedWrapper):
    """A wrapper that says which generator it shares, as the docs now do."""

    @property
    def seed_group(self):
        return self.shared


def _grouped_model(extra_independent=False):
    shared = _SharedPair()
    inputs = {"wind_x": _GroupedWrapper(shared), "wind_y": _GroupedWrapper(shared)}
    if extra_independent:
        inputs["mass"] = _Gaussian(14.426, 0.5)
    obj = SimpleNamespace(**{name: 0.0 for name in inputs})
    model = StochasticModel(obj, **inputs)
    shared.reset_count = 0  # the constructor has already seeded once
    model._set_stochastic(4242)
    return next(model.dict_generator()), shared


def test_a_shared_group_is_seeded_once_between_its_members():
    """Resetting each member in turn threw away every seed but the last, and
    left the group's stream decided by whichever member went last. It is one
    generator, so it gets one seed."""
    _, shared = _grouped_model()

    assert shared.reset_count == 1


def test_an_independent_sampler_does_not_move_a_shared_group():
    """Keying by name protects independent samplers from each other. The group
    has to be protected the same way, and keying it by the member that sorts
    last would not have been."""
    alone, _ = _grouped_model()
    alongside, _ = _grouped_model(extra_independent=True)

    assert (alone["wind_x"], alone["wind_y"]) == (
        alongside["wind_x"],
        alongside["wind_y"],
    )


class _RefusesTheSeed(_Gaussian):
    """A sampler whose generator will not take the seed it is given.

    `numpy.random.RandomState` is the real case: it refuses anything above
    2**32-1 with a ValueError, and the seeds handed out here are 128 bits.
    """

    def __init__(self, mean, sd, failure):
        super().__init__(mean, sd)
        self.failure = failure

    def reset_seed(self, seed=None):
        raise self.failure


@pytest.mark.parametrize(
    "failure",
    [ValueError("out of range"), TypeError("wrong type"), RuntimeError("boom")],
    ids=lambda f: type(f).__name__,
)
def test_a_sampler_that_refuses_its_seed_is_named_in_the_error(failure):
    """Only RuntimeError used to be caught, so a legacy RandomState sampler
    raised a bare ValueError with nothing to say which input it came from."""
    with pytest.raises(RuntimeError, match="mass") as raised:
        StochasticModel(
            SimpleNamespace(mass=0.0), mass=_RefusesTheSeed(0.0, 1.0, failure)
        )

    assert raised.value.__cause__ is failure


def test_a_legacy_random_state_sampler_is_named_rather_than_raising_bare():
    """The concrete case, not a stand-in: RandomState really does refuse the
    128-bit seed this hands out."""

    class LegacySampler(CustomSampler):
        """Built on RandomState rather than default_rng."""

        def sample(self, n_samples=1):
            return list(self.rng.normal(size=n_samples))

        def reset_seed(self, seed=None):
            self.rng = np.random.RandomState(seed)

    with pytest.raises(RuntimeError, match="mass") as raised:
        StochasticModel(SimpleNamespace(mass=0.0), mass=LegacySampler())

    assert isinstance(raised.value.__cause__, ValueError)


class _CountingShared(_SharedPair):
    """Records how it was reset, so the dispatch can be checked."""

    def __init__(self):
        super().__init__()
        self.reset_seed_calls = 0

    def reset_seed(self, seed=None):
        self.reset_seed_calls += 1
        self.reset(seed)


class _WrapperOverGroup(CustomSampler):
    """A wrapper whose own reset_seed would be the wrong thing to call."""

    def __init__(self, shared):
        self.shared = shared
        self.own_resets = 0

    @property
    def seed_group(self):
        return self.shared

    def sample(self, n_samples=1):
        return [self.shared.draw() for _ in range(n_samples)]

    def reset_seed(self, seed=None):
        self.own_resets += 1
        self.shared.reset(seed)


def test_a_group_that_can_reset_itself_is_reset_directly():
    """Dispatching through one member assumes every member resets the same way
    and holds no state of its own. The group owns the shared generator, so it
    is the thing to reset when it knows how."""
    shared = _CountingShared()
    first, second = _WrapperOverGroup(shared), _WrapperOverGroup(shared)
    model = StochasticModel(
        SimpleNamespace(wind_x=0.0, wind_y=0.0), wind_x=first, wind_y=second
    )
    shared.reset_seed_calls = 0
    first.own_resets = second.own_resets = 0

    model._set_stochastic(4242)

    assert shared.reset_seed_calls == 1
    assert (first.own_resets, second.own_resets) == (0, 0)


def test_a_group_key_does_not_depend_on_the_order_it_is_given():
    """The caller sorts today. The helper sorts too, so a future call site
    cannot hand one group two different seeds by listing it another way."""
    assert _sampler_seed(4242, ("wind_x", "wind_y")) == _sampler_seed(
        4242, ("wind_y", "wind_x")
    )


def test_a_non_sampler_is_refused_even_under_optimisation():
    """`python -O` strips an assert, so the check that keeps a non-sampler out
    of the model has to be a raise. The documented AssertionError is kept, so a
    caller already catching it is unaffected."""
    model = StochasticModel(SimpleNamespace(mass=0.0))

    with pytest.raises(AssertionError, match="must be a CustomSampler"):
        model._validate_custom_sampler("mass", object())


def test_the_refusal_survives_python_dash_o():
    """The mechanism, not just the behaviour: run it in a child with -O and
    check the exception still arrives."""
    import subprocess
    import sys

    program = (
        "from types import SimpleNamespace;"
        "from rocketpy.stochastic.stochastic_model import StochasticModel;"
        "m = StochasticModel(SimpleNamespace(mass=0.0));"
        "\ntry:\n"
        "    m._validate_custom_sampler('mass', object())\n"
        "except AssertionError:\n"
        "    print('refused')\n"
    )
    done = subprocess.run(
        [sys.executable, "-O", "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "refused" in done.stdout, done.stderr
