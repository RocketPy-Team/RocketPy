import ast
import inspect
import textwrap

import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.mathutils.function import Function
from rocketpy.rocket.aero_surface import FreeFormFins
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFreeFormFins,
    StochasticParachute,
    StochasticRocket,
)
from rocketpy.stochastic.stochastic_model import StochasticModel, _snapshot_of


@pytest.mark.parametrize(
    "fixture_name",
    [
        "stochastic_rail_buttons",
        "stochastic_main_parachute",
        "stochastic_environment",
        "stochastic_environment_custom_sampler",
        "stochastic_tail",
        "stochastic_calisto",
        "stochastic_free_form_fins",
    ],
)
def test_visualize_attributes(request, fixture_name):
    """Tests the visualize_attributes method of the StochasticModel class. It
    must run without breaking and return the formatted report string (which is
    also printed), so the report is never silently lost.
    """
    fixture = request.getfixturevalue(fixture_name)
    report = fixture.visualize_attributes()
    assert isinstance(report, str)
    assert report


def test_list_choices_are_reproducible(calisto_free_form_fins):
    """Choosing between the candidate values of a list input must come from the
    model's own generator, so that the same seed replays the same choices.

    The interpreter-wide ``random.choice`` was used, which ``_set_stochastic``
    does not reseed: a fixed-seed run picked different values every time, and
    Monte Carlo workers forked from one process walked a single shared stream
    instead of sampling independently.
    """
    taller = [(0, 0), (0.06, 0.12), (0.12, 0.12), (0.12, 0)]
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins,
        shape_points=[calisto_free_form_fins.shape_points, taller],
    )

    def spans(seed):
        stochastic._set_stochastic(seed)
        return [round(stochastic.create_object().span, 4) for _ in range(20)]

    assert spans(7) == spans(7)
    assert spans(7) != spans(8)
    # Both candidates must stay reachable, or the assertions above would also
    # hold for a generator that always returned the same one.
    assert set(spans(7)) == {0.1, 0.12}


def _windy_environment():
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    environment.wind_velocity_x = 10.0
    return environment


def _effective_wind_x(environment):
    """The wind the Environment would actually fly with."""
    wind = environment.wind_velocity_x
    return float(wind(0)) if callable(wind) else float(wind)


def test_a_factor_does_not_compound_across_reseeds():
    """Reseeding with the same seed has to give the same inputs.

    ``StochasticEnvironment.create_object`` writes the sampled value back onto
    the Environment rather than building a copy, so re-reading the nominal from
    it compounded: 10 -> 8.576 -> 7.355 -> 6.308, each the last one multiplied
    by the same factor again.
    """
    stochastic = StochasticEnvironment(
        environment=_windy_environment(), wind_velocity_x_factor=(1.0, 0.1)
    )

    winds = []
    for _ in range(4):
        stochastic._set_stochastic(12345)
        winds.append(_effective_wind_x(stochastic.create_object()))

    assert len(set(winds)) == 1, f"the same seed drifted across reseeds: {winds}"


def test_a_seed_gives_the_same_input_whatever_was_sampled_before_it():
    """Caching the nominal per model, not per seed, is what this pins.

    Reseeding to 103 has to give what it gives on a fresh model, whether or not
    102 and 101 ran first. A cache keyed by the seed would satisfy the test
    above and still fail here, because each new seed would re-read a nominal
    the previous ``create_object`` had already moved.
    """

    def wind_after(seeds):
        stochastic = StochasticEnvironment(
            environment=_windy_environment(), wind_velocity_x_factor=(1.0, 0.1)
        )
        wind = None
        for seed in seeds:
            stochastic._set_stochastic(seed)
            wind = _effective_wind_x(stochastic.create_object())
        return wind

    assert wind_after([101, 102, 103]) == wind_after([103])


def test_a_scalar_nominal_does_not_drift_across_reseeds():
    """Not only the factors.

    ``_validate_scalar`` and the ``(std, "distribution")`` tuple both take their
    nominal from the wrapped object, so a plain scalar spec drifts the same way
    a factor compounds.
    """
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    stochastic = StochasticEnvironment(environment=environment, elevation=100.0)

    elevations = []
    for _ in range(4):
        stochastic._set_stochastic(2024)
        elevations.append(float(stochastic.create_object().elevation))

    assert len(set(elevations)) == 1, f"the nominal elevation drifted: {elevations}"


def test_the_nominal_is_the_one_the_input_was_configured_with(example_plain_env):
    """Snapshot semantics, stated once and pinned here.

    A model samples around what the wrapped object held when the input was
    configured, so a later change to that object deliberately does not move what
    is sampled around. That is the same rule the drift above depends on.
    """
    example_plain_env.elevation = 1000
    # A scalar is a spread around the object's own value, so this is the form
    # that reads the nominal. A tuple carries its own centre and would not.
    model = StochasticEnvironment(environment=example_plain_env, elevation=5)

    model._set_stochastic(4242)
    around_first = model.elevation[0]

    example_plain_env.elevation = 9000
    model._set_stochastic(4242)

    assert model.elevation[0] == around_first == 1000, (
        "the model followed the object instead of the value it was configured with"
    )


def test_a_mutable_nominal_survives_a_write_through_the_object(
    calisto_free_form_fins,
):
    """Holding the object itself would let ``obj.shape_points[:] = ...`` reach
    the nominal and move a value the model is supposed to sample around."""
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=0.001
    )
    stochastic._set_stochastic(7)
    expected = np.array(stochastic.shape_points[0], copy=True)

    stochastic.obj.shape_points[:] = [(9.9, 9.9)] * len(stochastic.obj.shape_points)
    stochastic._set_stochastic(7)

    assert np.array_equal(stochastic.shape_points[0], expected)


def test_a_generated_object_cannot_move_the_kept_nominal(calisto_free_form_fins):
    """The kept value has to stay private, not only be copied on the way in.

    An empty spec keeps the object's own outline, and that one object reached
    the model attribute, ``last_rnd_dict`` and the fins ``create_object``
    returns, so a write through any of them moved the next reseed.
    """
    expected = [tuple(point) for point in calisto_free_form_fins.shape_points]
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=None
    )

    generated = stochastic.create_object()
    generated.shape_points[1] = (9.9, 9.9)
    stochastic._set_stochastic(7)

    assert [tuple(point) for point in stochastic.shape_points[0]] == expected


def test_writing_through_the_model_attribute_cannot_move_it_either():
    """The same, from the other public surface.

    ``numpy.asarray(value, dtype=float)`` hands back what it was given when that
    is already a float array, so the fin is built from one here.
    """
    fins = FreeFormFins(
        n=4,
        shape_points=np.array(
            [(0, 0), (0.08, 0.1), (0.12, 0.1), (0.12, 0)], dtype=float
        ),
        rocket_radius=0.0635,
    )
    stochastic = StochasticFreeFormFins(free_form_fins=fins, shape_points=0.001)
    stochastic._set_stochastic(7)
    expected = np.array(stochastic.shape_points[0], copy=True)

    stochastic.shape_points[0][1] = (9.9, 9.9)
    stochastic._set_stochastic(7)

    assert np.array_equal(stochastic.shape_points[0], expected)


def test_a_spread_and_distribution_tuple_does_not_drift():
    """The ``(std, "distribution")`` form takes its centre from the object too.

    The scalar test above goes through ``_validate_scalar`` and this through
    ``_validate_tuple_length_two``, so one says nothing about the other. Each
    run gets its own Environment, since ``create_object`` writes onto it.
    """

    def elevation_after(seeds):
        environment = Environment()
        environment.set_atmospheric_model(type="standard_atmosphere")
        environment.elevation = 100.0
        stochastic = StochasticEnvironment(
            environment=environment, elevation=(5.0, "normal")
        )
        drawn = None
        for seed in seeds:
            stochastic._set_stochastic(seed)
            drawn = float(stochastic.create_object().elevation)
        return drawn

    assert elevation_after([2024, 2024]) == elevation_after([2024])
    assert elevation_after([7, 11, 2024]) == elevation_after([2024])


def test_an_input_added_after_the_model_takes_its_nominal_then(calisto):
    """An ``add_*`` input is configured after ``__init__``, so it is read then.

    A scalar spec centres on the object's own value, and ``add_cp_eccentricity``
    is the first thing to ask for it, so what the rocket holds at that moment is
    what gets kept.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)

    calisto.cp_eccentricity_x = 0.5
    stochastic.add_cp_eccentricity(x=0.001)
    calisto.cp_eccentricity_x = 9.0

    stochastic._set_stochastic(11)
    drawn = float(next(stochastic.dict_generator())["cp_eccentricity_x"])

    assert abs(drawn - 0.5) < 0.05, f"centred on {drawn}, not on the add-time 0.5"


def test_the_snapshot_keeps_by_reference_what_it_does_not_copy():
    """Only built-in containers are copied, so the rule behaves as stated."""
    array = np.array([1.0, 2.0])
    listed = [[1.0], [2.0]]
    function = Function(lambda x: x)

    mapped = {"a": [1.0]}
    grouped = {("a", 1), ("b", 2)}

    assert _snapshot_of(array) is not array
    assert _snapshot_of(listed) is not listed
    assert _snapshot_of(listed)[0] is not listed[0]  # deep, not shallow
    assert _snapshot_of(mapped) is not mapped
    assert _snapshot_of(mapped)["a"] is not mapped["a"]
    assert _snapshot_of(grouped) is not grouped
    assert _snapshot_of(grouped) == grouped
    assert _snapshot_of(function) is function
    assert _snapshot_of(3.0) == 3.0


def test_a_function_nominal_is_held_as_it_was_given(calisto):
    """The documented exception, pinned rather than only written down.

    A ``Function`` is mutable through its own API, so ``set_source`` on the
    rocket's drag curve does move the baseline. Copying it would mean
    ``deepcopy`` of whatever a user passed, on a path that runs on every reseed
    and today cannot fail, which is not a trade this change should make.
    """
    stochastic = StochasticRocket(
        rocket=calisto, radius=0.0127 / 2, power_off_drag_factor=(1.0, 0.1)
    )
    stochastic._set_stochastic(4242)
    before = float(stochastic.create_object().power_off_drag(0.5))

    calisto.power_off_drag.set_source(lambda mach: 0.9)
    stochastic._set_stochastic(4242)
    after = float(stochastic.create_object().power_off_drag(0.5))

    assert 0.3 < before < 0.5, before
    assert 0.7 < after < 1.1, after


def test_two_components_do_not_share_one_position_nominal(stochastic_calisto):
    """Component positions are read live, through an injected getter.

    Every one of them arrives under the name ``position``, so keeping them the
    way the other inputs are kept would hand the second component the first
    one's place. The report test notices the getter going missing, but only
    because reading ``position`` off the rocket raises; it would not notice a
    key that quietly collides.
    """
    stochastic_calisto._set_stochastic(5)

    places = {}
    for component, position in stochastic_calisto.aerodynamic_surfaces:
        nominal = position[0]
        places[type(component).__name__] = float(getattr(nominal, "z", nominal))

    assert len(places) > 1, "need more than one surface for this to say anything"
    assert len(set(places.values())) == len(places), places


def test_the_snapshot_reaches_a_mutable_nested_in_a_tuple():
    """An ``airfoil`` is ``(source, unit)`` and the source may be an array, so
    stopping at the tuple would leave that array shared with the object it came
    from. A ``Function`` nested the same way still travels by reference.
    """
    source = np.array([[0.0, 0.0], [1.0, 1.0]])
    function = Function(lambda x: x)

    copied = _snapshot_of((source, "degrees"))
    source[0, 1] = 99.0

    assert copied[0][0, 1] == 0.0
    assert _snapshot_of((function, "degrees"))[0] is function


def test_configuring_a_late_input_again_reads_the_nominal_again(calisto):
    """``configured`` has to mean the second call as well as the first.

    The kept nominal had no replacement path, so a second
    ``add_cp_eccentricity`` went on sampling around the value the rocket held
    at the first one. Reproducible, and around the wrong centre.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)

    calisto.cp_eccentricity_x = 0.5
    stochastic.add_cp_eccentricity(x=0.001)
    calisto.cp_eccentricity_x = 0.8
    stochastic.add_cp_eccentricity(x=0.001)

    assert stochastic.cp_eccentricity_x[0] == 0.8

    stochastic._set_stochastic(11)

    assert stochastic.cp_eccentricity_x[0] == 0.8


def test_a_refused_reconfiguration_leaves_the_previous_one(calisto):
    """Dropping the kept nominal before validation must not outlive a failure."""
    calisto.cp_eccentricity_x = 0.5
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(x=0.001)

    calisto.cp_eccentricity_x = 0.8
    with pytest.raises(AssertionError):
        stochastic.add_cp_eccentricity(x=object())

    stochastic._set_stochastic(11)

    assert stochastic.cp_eccentricity_x[0] == 0.5


@pytest.mark.parametrize(
    ("add_them", "kept"),
    [
        ("add_cp_eccentricity", "cp_eccentricity_y"),
        ("add_thrust_eccentricity", "thrust_eccentricity_y"),
    ],
)
def test_an_axis_left_out_keeps_everything_it_had(calisto, add_them, kept):
    """Not only its declaration: its distribution and its kept nominal too.

    Keeping the declaration alone left the two disagreeing. The private side
    still said the axis was random while the validated attribute had been
    replaced by a lone nominal, and ``dict_generator`` reads the attribute, so
    the axis stopped varying until something reset the model. A serial Monte
    Carlo never does, so a whole study would have run with it switched off.
    """
    setattr(calisto, kept, 0.0)
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    getattr(stochastic, add_them)(x=0.001, y=0.002)
    stochastic._set_stochastic(11)
    centre = getattr(stochastic, kept)[0]

    # Moved between the calls, so re-reading the nominal would show up.
    setattr(calisto, kept, 9.0)
    getattr(stochastic, add_them)(x=0.005)

    drawn = {next(stochastic.dict_generator())[kept] for _ in range(8)}
    assert len(drawn) == 8, "the axis stopped varying before any reset"

    stochastic._set_stochastic(11)

    assert getattr(stochastic, kept)[0] == centre
    assert kept in next(stochastic.dict_generator())


def test_leaving_an_axis_out_does_not_take_away_what_it_declared(calisto):
    """``None`` reads the same whether it was written or the argument was
    left out, so a call that mentions only x has to leave y where it was.

    Removing on the second reading would take away an axis nobody mentioned,
    which is why removal needs an argument the caller cannot supply by
    omission. That is #1171 rather than this change.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(x=0.001, y=0.002)

    stochastic.add_cp_eccentricity(x=0.005)
    stochastic._set_stochastic(11)

    generated = next(stochastic.dict_generator())
    assert "cp_eccentricity_x" in generated
    assert "cp_eccentricity_y" in generated


def test_a_half_that_was_never_given_is_not_declared(calisto):
    """The control for the one above.

    ``None`` also means an axis the caller never mentioned, and removing what
    was never there has to stay a no-op rather than an error.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(x=0.001)

    generated = next(stochastic.dict_generator())

    assert "cp_eccentricity_x" in generated
    assert "cp_eccentricity_y" not in generated


def test_a_pair_of_late_inputs_is_replaced_together(calisto):
    """``add_cp_eccentricity`` takes x and y in one call, so a y that will not
    validate must leave x exactly as it was, declaration included.

    Only y is given first, so x is undeclared going in and a partial commit
    shows up as an eccentricity the caller never successfully asked for.
    """
    calisto.cp_eccentricity_x = 0.5
    calisto.cp_eccentricity_y = 0.6
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(y=0.002)

    calisto.cp_eccentricity_x = 0.8
    with pytest.raises(AssertionError):
        stochastic.add_cp_eccentricity(x=0.001, y=object())

    stochastic._set_stochastic(11)
    generated = next(stochastic.dict_generator())

    assert "cp_eccentricity_x" not in generated
    assert generated["cp_eccentricity_y"] is not None
    assert stochastic.cp_eccentricity_y[0] == 0.6


def test_one_generated_object_cannot_change_the_next_one(calisto_free_form_fins):
    """``create_object`` can be called again without a reseed in between.

    The list branch handed back the candidate itself, which for an empty spec is
    the model's own working outline, and ``FreeFormFins`` keeps it by reference.
    Writing through the first fins therefore reached the second ones.
    """
    expected = [tuple(point) for point in calisto_free_form_fins.shape_points]
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=None
    )

    first = stochastic.create_object()
    first.shape_points[1] = (9.9, 9.9)
    second = stochastic.create_object()

    assert [tuple(point) for point in second.shape_points] == expected


def test_the_record_of_a_draw_is_not_a_window_onto_the_object(
    calisto_free_form_fins,
):
    """``last_rnd_dict`` says what was drawn.

    A Monte Carlo writes it out after the flight has run, so a value the flight
    edited in place would be logged instead of the one that was sampled.
    """
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=None
    )

    generated = stochastic.create_object()
    recorded = np.array(stochastic.last_rnd_dict["shape_points"], copy=True)
    generated.shape_points[1] = (9.9, 9.9)

    assert np.array_equal(stochastic.last_rnd_dict["shape_points"], recorded)


def test_a_subclass_that_adjusts_a_draw_records_it_again():
    """The base class records before a subclass has had its turn.

    ``StochasticFreeFormFins`` corrects the outline in ``dict_generator`` and
    ``StochasticParachute`` adds the pressure noise seed in ``create_object``,
    both after the record was taken. Read off the source, because the subclass
    that gets this wrong is the one nobody wrote a fixture for.
    """
    # Walked from the base class rather than from what the package exports,
    # since StochasticMotorModel is a subclass the exports do not reach.
    models, stack = set(), [StochasticModel]
    while stack:
        for subclass in stack.pop().__subclasses__():
            models.add(subclass)
            stack.append(subclass)

    offenders = []
    for model in sorted(models, key=lambda cls: cls.__name__):
        name = model.__name__
        for method in ("dict_generator", "create_object"):
            if method not in vars(model):
                continue
            source = textwrap.dedent(inspect.getsource(vars(model)[method]))
            body = ast.parse(source).body[0].body
            # Only the names the method binds from a draw. A local it fills
            # in for its own use, such as the factors StochasticEnvironment
            # collects, is not a record of anything.
            drawn = {
                target.id
                for node in ast.walk(ast.Module(body=body, type_ignores=[]))
                if isinstance(node, ast.Assign)
                and (
                    "dict_generator" in ast.dump(node.value)
                    or method == "dict_generator"
                )
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            writes = [
                node
                for node in ast.walk(ast.Module(body=body, type_ignores=[]))
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in drawn
                    for target in node.targets
                )
            ]
            if writes and "_record_draw" not in source:
                offenders.append(f"{name}.{method}")

    assert not offenders, (
        f"these override dict_generator, write into the drawn dictionary and "
        f"never call _record_draw, so the record keeps what they replaced: "
        f"{sorted(offenders)}"
    )


def test_the_record_holds_the_outline_the_fins_were_built_from(
    calisto_free_form_fins,
):
    """A perturbed outline is pulled back onto the body line after it is drawn.

    Under seed 7 the two root points drift to 0.000299 and 0.001340 and the
    correction returns them to zero, so a record taken before it reports an
    outline the fins were never built from.
    """
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=0.001
    )
    stochastic._set_stochastic(7)

    built = stochastic.create_object()

    assert np.array_equal(
        np.asarray(stochastic.last_rnd_dict["shape_points"], dtype=float),
        np.asarray(built.shape_points, dtype=float),
    )


def test_a_rocket_records_the_outline_its_fins_were_built_from(
    stochastic_calisto, stochastic_free_form_fins
):
    """The same, once the fins are nested in a rocket.

    ``_create_surface`` copies each component's own record into the rocket's,
    so a component that recorded too early reaches the Monte Carlo input log.
    """
    # A tuple position carries its own centre, so it does not go looking for
    # matching fins on a deterministic rocket that has none.
    stochastic_calisto.add_free_form_fins(
        stochastic_free_form_fins, position=(-1.05, 0.001)
    )
    stochastic_calisto._set_stochastic(7)

    rocket = stochastic_calisto.create_object()

    recorded = next(
        entry["shape_points"]
        for entry in stochastic_calisto.last_rnd_dict["aerodynamic_surfaces"]
        if "shape_points" in entry
    )
    built = next(
        surface
        for surface in rocket.aerodynamic_surfaces.get_components()
        if isinstance(surface, FreeFormFins)
    )

    assert np.array_equal(
        np.asarray(recorded, dtype=float),
        np.asarray(built.shape_points, dtype=float),
    )


def test_choosing_between_no_candidates_gives_back_an_empty_copy(example_plain_env):
    """Validation never produces an empty candidate list, so this is the guard
    that keeps ``integers(0)`` from raising if one ever reaches here."""
    model = StochasticEnvironment(environment=example_plain_env)
    empty = []

    chosen = model._choose(empty)

    assert chosen == []
    assert chosen is not empty


def test_a_parachute_records_the_noise_seed_it_was_built_with(calisto_main_chute):
    """``create_object`` derives the pressure noise seed after the draw.

    The parachute is built with it either way, so a record taken before it
    describes a parachute whose noise nobody can reproduce.
    """
    stochastic = StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
    stochastic._set_stochastic(42)

    built = stochastic.create_object()

    assert stochastic.last_rnd_dict["seed"] == built._seed


def test_a_rocket_records_the_noise_seed_its_parachute_was_built_with(
    stochastic_calisto, calisto_main_chute
):
    """The same once nested, which is the shape a Monte Carlo writes out."""
    stochastic_calisto.parachutes = []
    stochastic_calisto.add_parachute(
        StochasticParachute(parachute=calisto_main_chute, cd_s=0.1, lag=0.2)
    )
    stochastic_calisto._set_stochastic(42)

    rocket = stochastic_calisto.create_object()

    recorded = stochastic_calisto.last_rnd_dict["parachutes"][0]["seed"]
    assert recorded == rocket.parachutes[0]._seed
