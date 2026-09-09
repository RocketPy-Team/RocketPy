"""Every nested component of a StochasticRocket is reseeded from its own child
of the run's seed, so components that sample the same distribution do not draw
the same values, and one seed still reproduces the whole rocket.
"""

import ast
import inspect

import numpy as np

from rocketpy.rocket.components import Components
from rocketpy.tools import _seed_sequence_from
from rocketpy.stochastic import (
    StochasticAirBrakes,
    StochasticParachute,
    StochasticRocket,
)
from rocketpy.stochastic.stochastic_model import StochasticModel

# Captured before any patching, so wrapping it twice in one test does not stack.
_REAL_SET_STOCHASTIC = StochasticModel._set_stochastic


def _seeds_handed_out(monkeypatch, rocket, seed):
    """The seeds every nested component received during one reseed."""
    recorded = []

    def recording(self, seed=None):
        recorded.append(seed)
        return _REAL_SET_STOCHASTIC(self, seed)

    monkeypatch.setattr(StochasticModel, "_set_stochastic", recording)
    rocket._set_stochastic(seed)
    return recorded


def _drawn(component):
    return next(component.dict_generator())


def _members_of(collection):
    """Components yields (component, position) pairs; a plain list does not."""
    if isinstance(collection, Components):
        return [component for component, _ in collection]
    return list(collection)


def test_two_components_with_one_spec_do_not_share_one_stream(
    stochastic_calisto, calisto_main_chute
):
    """The whole rocket shared one seed, so two parachutes built from the same
    spec drew the same ``cd_s`` and the same ``lag``, every time. A study of a
    main and a drogue was really a study of one chute counted twice.
    """
    stochastic_calisto.parachutes = []
    for _ in range(2):
        stochastic_calisto.add_parachute(
            StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
        )

    stochastic_calisto._set_stochastic(99)
    first, second = (_drawn(chute) for chute in stochastic_calisto.parachutes)

    assert first["cd_s"] != second["cd_s"]
    assert first["lag"] != second["lag"]


def test_one_seed_reproduces_every_component(stochastic_calisto, calisto_main_chute):
    """Independent is not enough on its own; it still has to follow the seed."""
    stochastic_calisto.parachutes = []
    stochastic_calisto.add_parachute(
        StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
    )

    def drawn_with(seed):
        stochastic_calisto._set_stochastic(seed)
        return [_drawn(chute) for chute in stochastic_calisto.parachutes] + [
            _drawn(stochastic_calisto)["mass"]
        ]

    assert drawn_with(2718) == drawn_with(2718)
    assert drawn_with(2718) != drawn_with(2719)


def test_component_seeds_do_not_collide(monkeypatch, stochastic_calisto):
    """The same statement across every component type, not only the parachutes:
    the body, each aerodynamic surface, the motor and the rail buttons.
    """
    seeds = _seeds_handed_out(monkeypatch, stochastic_calisto, 42)

    assert len(seeds) > 3, "expected the rocket body and several components"
    assert len(seeds) == len(set(seeds)), (
        "components share a seed, so they draw perfectly correlated samples"
    )


def test_the_reseed_covers_every_collection_create_object_uses(stochastic_calisto):
    """Whatever ``create_object`` iterates has to be reseeded too.

    Read off the source rather than off a fixture, because the collection no
    fixture populates is exactly the one that gets missed: air brakes were built
    and sampled and never reseeded, and every seeding test passed anyway.
    """
    rocket = stochastic_calisto
    tree = ast.parse(inspect.getsource(type(rocket).create_object).lstrip())
    iterated = {
        node.iter.attr
        for node in ast.walk(tree)
        # Comprehensions too. This scan exists to catch a collection added
        # later, and one written as a comprehension would slip past a For walk.
        if isinstance(node, (ast.For, ast.comprehension))
        and isinstance(node.iter, ast.Attribute)
        and isinstance(node.iter.value, ast.Name)
        and node.iter.value.id == "self"
        and not node.iter.attr.startswith("_")
    }
    declared = set(type(rocket)._stochastic_collections())

    assert iterated, "found no collections in create_object, so the scan is broken"
    # Both directions. One left in the reseed after create_object stopped using
    # it still spawns a child and moves every stream that follows.
    assert iterated == declared, {
        "sampled but never reseeded": sorted(iterated - declared),
        "reseeded but never sampled": sorted(declared - iterated),
    }


def test_an_air_brake_answers_to_the_seed(
    stochastic_calisto, calisto_air_brakes_clamp_on
):
    """Air brakes were in ``create_object`` and not in the reseed, so a fixed
    seed did not reproduce them: 0.683, then 0.586, then 0.488 for one seed
    asked three times.

    Built here rather than taken from the fixture, since wrapping an
    ``AirBrakes`` with no arguments gives every parameter a zero standard
    deviation, and that draws the same value under any seed whether or not
    anything reseeds it.
    """
    stochastic_calisto.add_air_brakes(
        StochasticAirBrakes(
            air_brakes=calisto_air_brakes_clamp_on.air_brakes[0],
            deployment_level=(0.5, 0.1),
        ),
        calisto_air_brakes_clamp_on._controllers[0],
    )
    air_brake = stochastic_calisto.air_brakes[0]

    def drawn(seed):
        stochastic_calisto._set_stochastic(seed)
        return _drawn(air_brake)["deployment_level"]

    first = drawn(1234)

    assert drawn(1234) == first, "the same seed drew a different air brake"
    assert drawn(1235) != first, "a different seed drew the same air brake"


def test_adding_a_surface_leaves_the_other_collections_alone(
    stochastic_calisto, calisto_main_chute, stochastic_nose_cone
):
    """Each collection has a root of its own, so an unrelated component in one
    of them does not move the streams in the others."""
    stochastic_calisto.parachutes = []
    stochastic_calisto.add_parachute(
        StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
    )

    def parachute_draw():
        stochastic_calisto._set_stochastic(5)
        return _drawn(stochastic_calisto.parachutes[0])

    before = parachute_draw()
    # The deterministic nose, so add_nose builds a wrapper of its own. Adding
    # the fixture again would store one wrapper twice, which is #1172.
    stochastic_calisto.add_nose(stochastic_nose_cone.obj, position=1.1)

    assert parachute_draw() == before


def test_every_entry_is_reseeded_exactly_once(
    monkeypatch, stochastic_calisto, calisto_main_chute, calisto_air_brakes_clamp_on
):
    """Counted rather than read off the source.

    The scan above reads ``create_object`` for ``for x in self.collection``, so
    a helper, a local alias or a ``getattr`` would hide a collection from it.
    This counts what actually happens.
    """
    stochastic_calisto.add_parachute(
        StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
    )
    stochastic_calisto.add_air_brakes(
        StochasticAirBrakes(
            air_brakes=calisto_air_brakes_clamp_on.air_brakes[0],
            deployment_level=(0.5, 0.1),
        ),
        calisto_air_brakes_clamp_on._controllers[0],
    )

    counted = {}

    def recording(self, seed=None):
        counted[id(self)] = counted.get(id(self), 0) + 1
        return _REAL_SET_STOCHASTIC(self, seed)

    monkeypatch.setattr(StochasticModel, "_set_stochastic", recording)
    stochastic_calisto._set_stochastic(3)

    entries = [
        component
        for name in type(stochastic_calisto)._stochastic_collections()
        for component in _members_of(getattr(stochastic_calisto, name))
    ]

    assert entries, "no components to count"
    assert all(counted.get(id(entry)) == 1 for entry in entries), {
        type(entry).__name__: counted.get(id(entry)) for entry in entries
    }


def test_two_air_brakes_with_one_spec_stay_independent(
    stochastic_calisto, calisto_air_brakes_clamp_on
):
    """The air brakes are a plain list, so they take a different route through
    the reseed than the positioned collections do.

    About the streams only. Both are added with one controller because the
    rocket keeps a single one, which is a separate problem recorded in #1172.
    """
    for _ in range(2):
        stochastic_calisto.add_air_brakes(
            StochasticAirBrakes(
                air_brakes=calisto_air_brakes_clamp_on.air_brakes[0],
                deployment_level=(0.5, 0.1),
            ),
            calisto_air_brakes_clamp_on._controllers[0],
        )

    def drawn(seed):
        stochastic_calisto._set_stochastic(seed)
        return [
            _drawn(brake)["deployment_level"] for brake in stochastic_calisto.air_brakes
        ]

    first = drawn(808)

    assert first[0] != first[1], "two air brakes drew the same value"
    assert drawn(808) == first
    assert drawn(809) != first


def test_the_rocket_body_keeps_the_seed_as_given(monkeypatch, stochastic_calisto):
    """Fixing the nested components did not need the body's stream to move.

    Reproducibility and seed uniqueness both hold with the body on a spawned
    child, so neither of them would notice it going back there and taking every
    fixed-seed mass and radius baseline with it.
    """
    seeds = _seeds_handed_out(monkeypatch, stochastic_calisto, 42)

    assert seeds[0] == 42


def test_the_tree_is_built_by_the_reset_not_by_the_add(calisto, calisto_main_chute):
    """A rocket resets itself while being constructed, with nothing attached.

    So a component added afterwards keeps the generator it was built with until
    the next reset, which is the guarantee the documentation states and the one
    a Monte Carlo relies on.
    """
    rocket = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    chute = StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
    rocket.add_parachute(chute)

    assert getattr(chute, "_seed", None) is None

    rocket._set_stochastic(99)

    assert chute._seed is not None


def test_a_worker_seed_sequence_is_copied_rather_than_spawned_from():
    # A parallel run hands each worker a SeedSequence. Spawning from it would
    # advance a counter the caller still holds, so the next use of the same
    # object would build a different tree.
    worker = np.random.SeedSequence(7).spawn(2)[0]

    _seed_sequence_from(worker).spawn(3)

    assert worker.n_children_spawned == 0


def test_a_worker_seed_sequence_keeps_its_place_in_the_tree():
    worker = np.random.SeedSequence(7).spawn(2)[0]

    children = _seed_sequence_from(worker).spawn(2)

    assert [child.spawn_key for child in children] == [(0, 0), (0, 1)]


def test_two_workers_do_not_get_the_same_collection_roots():
    first, second = np.random.SeedSequence(7).spawn(2)

    one = _seed_sequence_from(first).spawn(1)[0].generate_state(4)
    other = _seed_sequence_from(second).spawn(1)[0].generate_state(4)

    assert not np.array_equal(one, other)


def test_an_integer_seed_still_roots_the_collections():
    assert _seed_sequence_from(42).spawn(1)[0].spawn_key == (0,)
    assert _seed_sequence_from(None).spawn(1)[0].spawn_key == (0,)
