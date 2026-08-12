import numpy as np
import pytest

from rocketpy.rocket.aero_surface import (
    FreeFormFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from rocketpy.stochastic import StochasticFreeFormFins

## NOSE CONE


def test_stochastic_nose_cone_create_object(stochastic_nose_cone):
    """Test create object method of StochasticNoseCone class.

    This test checks if the create_object method of the StochasticNoseCone
    class creates a StochasticNoseCone object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_nose_cone : StochasticNoseCone
        StochasticNoseCone object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_nose_cone.create_object()
    assert isinstance(obj, NoseCone)


## TRAPEZOIDAL FINS


def test_stochastic_trapezoidal_fins_create_object(stochastic_trapezoidal_fins):
    """Test create object method of StochasticTrapezoidalFins class.

    This test checks if the create_object method of the StochasticTrapezoidalFins
    class creates a StochasticTrapezoidalFins object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_trapezoidal_fins : StochasticTrapezoidalFins
        StochasticTrapezoidalFins object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_trapezoidal_fins.create_object()
    assert isinstance(obj, TrapezoidalFins)


## FREE FORM FINS

NOMINAL_SHAPE = [(0, 0), (0.08, 0.1), (0.12, 0.1), (0.12, 0)]


def test_stochastic_free_form_fins_create_object(stochastic_free_form_fins):
    """Test create object method of StochasticFreeFormFins class.

    This test checks if the create_object method of the StochasticFreeFormFins
    class creates a FreeFormFins object from the randomly generated input
    arguments.

    Parameters
    ----------
    stochastic_free_form_fins : StochasticFreeFormFins
        StochasticFreeFormFins object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_free_form_fins.create_object()
    assert isinstance(obj, FreeFormFins)


def test_stochastic_free_form_fins_nominal_shape_is_preserved(calisto_free_form_fins):
    """With nothing to randomize, the created fin set must keep the outline of
    the object it was built from."""
    stochastic = StochasticFreeFormFins(free_form_fins=calisto_free_form_fins)

    created = stochastic.create_object()

    assert np.allclose(
        np.asarray(created.shape_points, dtype=float),
        np.asarray(calisto_free_form_fins.shape_points, dtype=float),
    )


@pytest.mark.parametrize(
    "shape_points",
    [
        0.001,
        (0.001, "normal"),
        (NOMINAL_SHAPE, 0.001),
        (NOMINAL_SHAPE, 0.001, "normal"),
    ],
    ids=["scalar", "std_and_dist", "outline_and_std", "outline_std_and_dist"],
)
def test_stochastic_free_form_fins_perturbs_the_whole_outline(
    calisto_free_form_fins, shape_points
):
    """A fin outline is only meaningful as a complete set of points, so every
    accepted format must randomize all of them and keep the (n, 2) shape."""
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=shape_points
    )
    stochastic._set_stochastic(42)

    created = stochastic.create_object()

    nominal = np.asarray(NOMINAL_SHAPE, dtype=float)
    sampled = np.asarray(created.shape_points, dtype=float)
    assert sampled.shape == nominal.shape
    # Every coordinate is drawn on its own, so none of the four points is left
    # exactly where it was, apart from the root's y (see the test below).
    assert not np.allclose(sampled[:, 0], nominal[:, 0])
    assert not np.allclose(sampled[1:3, 1], nominal[1:3, 1])
    # A standard deviation of a millimetre must not turn into a new fin.
    assert np.abs(sampled - nominal).max() < 0.01


@pytest.mark.parametrize(
    "shape_points",
    [0.001, (0.001, "normal"), (NOMINAL_SHAPE, 0.001, "laplace")],
    ids=["scalar", "std_and_dist", "outline_std_and_dist"],
)
def test_stochastic_free_form_fins_keeps_the_root_on_the_body_line(
    calisto_free_form_fins, shape_points
):
    """FreeFormFins measures the span from y = 0 and slices the chords over that
    interval, so a perturbed root point must not drift off the body line: it
    would put part of the fin inside the airframe and inflate the span the
    chords are measured against.
    """
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=shape_points
    )
    stochastic._set_stochastic(3)

    for _ in range(50):
        sampled = np.asarray(stochastic.create_object().shape_points, dtype=float)
        # The first and last points of the nominal outline are on the body line.
        assert sampled[0, 1] == 0
        assert sampled[-1, 1] == 0
        assert (sampled[:, 1] >= 0).all()


def test_stochastic_free_form_fins_bare_outline_is_a_single_candidate(
    calisto_free_form_fins,
):
    """A bare outline is a list, which the base class would otherwise read as a
    list of candidate values and sample a single (x, y) point from."""
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=NOMINAL_SHAPE
    )

    created = stochastic.create_object()

    assert np.allclose(
        np.asarray(created.shape_points, dtype=float),
        np.asarray(NOMINAL_SHAPE, dtype=float),
    )


def test_stochastic_free_form_fins_chooses_between_outlines(calisto_free_form_fins):
    """A list of outlines is a set of candidate shapes to choose from."""
    taller = [(0, 0), (0.06, 0.12), (0.12, 0.12), (0.12, 0)]
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins,
        shape_points=[NOMINAL_SHAPE, taller],
    )
    stochastic._set_stochastic(42)

    spans = {round(stochastic.create_object().span, 4) for _ in range(50)}

    assert spans == {0.1, 0.12}


def test_stochastic_free_form_fins_chooses_between_outlines_of_different_lengths(
    calisto_free_form_fins,
):
    """Candidate outlines need not have the same number of points: choosing
    between a three-point and a four-point fin is the plainest form of choosing
    between shapes, and numpy raises on that ragged list if it is converted
    whole instead of one candidate at a time.
    """
    triangle = [(0, 0), (0.08, 0.1), (0.12, 0)]
    quadrilateral = [(0, 0), (0.06, 0.12), (0.12, 0.12), (0.12, 0)]
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins,
        shape_points=[triangle, quadrilateral],
    )
    stochastic._set_stochastic(42)

    point_counts = {len(stochastic.create_object().shape_points) for _ in range(50)}

    assert point_counts == {3, 4}


def test_stochastic_free_form_fins_accepts_an_array_outline(calisto_free_form_fins):
    """A sampled outline comes back as an array, so feeding one straight back in
    as the nominal outline must work."""
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins,
        shape_points=np.asarray(NOMINAL_SHAPE, dtype=float),
    )

    created = stochastic.create_object()

    assert np.allclose(
        np.asarray(created.shape_points, dtype=float),
        np.asarray(NOMINAL_SHAPE, dtype=float),
    )


def test_stochastic_free_form_fins_empty_list_means_the_nominal_outline(
    calisto_free_form_fins,
):
    """An empty list means "take the nominal value and do not randomize" for
    every other stochastic input, and this one is no different."""
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=[]
    )

    created = stochastic.create_object()

    assert np.allclose(
        np.asarray(created.shape_points, dtype=float),
        np.asarray(calisto_free_form_fins.shape_points, dtype=float),
    )


@pytest.mark.parametrize(
    "shape_points",
    [
        "not_an_outline",
        [[(0, 0), (0.1, 0.1)]],
        [(0, 0), (0.1, 0.1)],
        [(0, 0, 0), (0.1, 0.1, 0), (0.1, 0, 0)],
        [(0, 0), (1, 1, 1), (2, 0)],
        [[("a", "b"), ("c", "d"), ("e", "f")]],
        (0.001,),
        (NOMINAL_SHAPE, 0.001, "normal", 1),
        (NOMINAL_SHAPE, "normal"),
        (0.001, 5),
        (NOMINAL_SHAPE, 0.001, 7),
        (0.001, "uniform"),
        (NOMINAL_SHAPE, 0.001, "wald"),
    ],
    ids=[
        "string",
        "too_few_points",
        "bare_outline_too_few_points",
        "three_dimensional_points",
        "ragged_outline",
        "non_numeric_points",
        "tuple_too_short",
        "tuple_too_long",
        "outline_with_string_std",
        "std_with_non_string_dist",
        "outline_with_non_string_dist",
        "bounded_distribution",
        "shape_parameter_distribution",
    ],
)
def test_stochastic_free_form_fins_rejects_invalid_shape_points(
    calisto_free_form_fins, shape_points
):
    """An outline that cannot mean a fin shape, or a distribution that cannot
    mean a deviation around one, must fail during validation rather than
    reaching FreeFormFins or the sampler."""
    with pytest.raises(AssertionError):
        StochasticFreeFormFins(
            free_form_fins=calisto_free_form_fins, shape_points=shape_points
        )


## TAIL


def test_stochastic_tail_create_object(stochastic_tail):
    """Test create object method of StochasticTail class.

    This test checks if the create_object method of the StochasticTail
    class creates a StochasticTail object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_tail : StochasticTail
        StochasticTail object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_tail.create_object()
    assert isinstance(obj, Tail)


## RAIL BUTTONS


def test_stochastic_rail_buttons_create_object(stochastic_rail_buttons):
    """Test create object method of StochasticRailButtons class.

    This test checks if the create_object method of the StochasticRailButtons
    class creates a StochasticRailButtons object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_rail_buttons : StochasticRailButtons
        StochasticRailButtons object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_rail_buttons.create_object()
    assert isinstance(obj, RailButtons)
