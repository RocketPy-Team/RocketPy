from numbers import Real

import numpy as np
import pytest

from rocketpy.rocket.aero_surface import FreeFormFins
from rocketpy.rocket.parachute import Parachute
from rocketpy.rocket.rocket import Rocket
from rocketpy.stochastic import (
    StochasticFreeFormFins,
    StochasticParachute,
    StochasticRocket,
    StochasticTrapezoidalFins,
)


def test_str(stochastic_calisto):
    assert isinstance(str(stochastic_calisto), str)


def test_create_object(stochastic_calisto):
    """Test create object method of StochasticRocket class.

    This test checks if the create_object method of the StochasticCalisto
    class creates a StochasticCalisto object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_calisto : StochasticCalisto
        StochasticCalisto object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_calisto.create_object()
    assert isinstance(obj, Rocket)


def test_sampled_parachute_geometry_reaches_the_created_rocket(
    stochastic_calisto, calisto_main_chute
):
    """The sampled Parachute used to be discarded and a second one built from
    six of its ten fields, so radius, height, porosity and the drag coefficient
    never left `last_rnd_dict`. Parachute re-derived radius from cd_s and the
    default drag coefficient, and height fell back to that radius.
    """
    stochastic_calisto.parachutes = []
    stochastic_calisto.add_parachute(
        StochasticParachute(
            parachute=calisto_main_chute,
            cd_s=0.1,
            radius=0.3,
            height=0.2,
            porosity=0.01,
            drag_coefficient=0.2,
        )
    )
    stochastic_calisto._set_stochastic(42)

    rocket = stochastic_calisto.create_object()

    built = rocket.parachutes[0]
    sampled = stochastic_calisto.last_rnd_dict["parachutes"][0]
    for field in ("cd_s", "radius", "height", "porosity", "drag_coefficient"):
        assert getattr(built, field) == sampled[field], (
            f"the rocket flies a {field} the run never sampled"
        )


def test_the_parachute_is_attached_exactly_once(
    stochastic_calisto, stochastic_main_parachute, stochastic_drogue_parachute
):
    """The control. Without it the test above would pass on a create_object
    that attaches nothing at all, which would silently fly every Monte Carlo
    rocket without its parachutes."""
    stochastic_calisto.parachutes = []
    for parachute in (stochastic_main_parachute, stochastic_drogue_parachute):
        stochastic_calisto.add_parachute(parachute)
    stochastic_calisto._set_stochastic(42)

    rocket = stochastic_calisto.create_object()

    assert len(rocket.parachutes) == 2
    assert [p.name for p in rocket.parachutes] == [
        stochastic_main_parachute.obj.name,
        stochastic_drogue_parachute.obj.name,
    ]


def test_a_parachute_is_built_once_per_simulation(
    stochastic_calisto, stochastic_main_parachute, monkeypatch
):
    """Building it twice drew the initial pressure noise from the global NumPy
    RNG twice, which is state no seed here controls. See #1091."""
    built = []
    real = Parachute.__init__

    def counting(self, *args, **kwargs):
        built.append(self)
        return real(self, *args, **kwargs)

    stochastic_calisto.parachutes = []
    stochastic_calisto.add_parachute(stochastic_main_parachute)
    stochastic_calisto._set_stochastic(42)
    monkeypatch.setattr(Parachute, "__init__", counting)

    stochastic_calisto.create_object()

    assert len(built) == 1


def test_configured_geometry_survives_without_being_randomized(calisto_robust):
    """The wider case. Dropping the four fields did not need anyone to
    randomize them: a parachute built with an explicit radius flew a radius
    re-derived from cd_s instead, in every Monte Carlo simulation."""
    chute = calisto_robust.add_parachute(
        "geometric",
        cd_s=10.0,
        trigger="apogee",
        sampling_rate=105,
        lag=1.5,
        radius=2.0,
        height=1.5,
        porosity=0.05,
        drag_coefficient=1.4,
    )
    stochastic = StochasticRocket(rocket=calisto_robust, mass=(14.426, 0.5))
    stochastic.parachutes = []
    stochastic.add_parachute(StochasticParachute(chute, cd_s=(10.0, 0.5)))
    stochastic._set_stochastic(42)

    flown = stochastic.create_object().parachutes[0]

    assert (flown.radius, flown.height, flown.porosity) == (2.0, 1.5, 0.05)


def test_a_deterministic_surface_is_wrapped_in_its_stochastic_model(
    calisto_robust, calisto_trapezoidal_fins
):
    """`_add_surfaces` used to wrap deterministic surfaces with a `component=`
    keyword none of the stochastic classes accept, so passing any plain
    aerodynamic surface raised a TypeError instead of being wrapped."""
    stochastic = StochasticRocket(rocket=calisto_robust)

    stochastic.add_trapezoidal_fins(calisto_trapezoidal_fins)

    added = stochastic.aerodynamic_surfaces.get_tuple_by_type(StochasticTrapezoidalFins)
    assert len(added) == 1
    assert added[0].component.obj is calisto_trapezoidal_fins


def test_add_free_form_fins_reaches_the_created_rocket(
    calisto_robust, stochastic_free_form_fins
):
    """The fin set added to the stochastic rocket must be the one the created
    rocket flies, with the outline randomized as a block."""
    stochastic = StochasticRocket(rocket=calisto_robust)
    stochastic.add_free_form_fins(stochastic_free_form_fins, position=(-1.04956, 0.001))
    stochastic._set_stochastic(42)

    rocket = stochastic.create_object()

    fin_sets = rocket.aerodynamic_surfaces.get_tuple_by_type(FreeFormFins)
    assert len(fin_sets) == 1
    flown = fin_sets[0].component
    nominal = np.asarray(stochastic_free_form_fins.obj.shape_points, dtype=float)
    sampled = np.asarray(flown.shape_points, dtype=float)
    assert sampled.shape == nominal.shape
    assert not np.allclose(sampled, nominal)


def test_add_free_form_fins_rejects_other_surfaces(calisto_robust, calisto_tail):
    stochastic = StochasticRocket(rocket=calisto_robust)

    with pytest.raises(AssertionError):
        stochastic.add_free_form_fins(calisto_tail)


def test_add_free_form_fins_wraps_a_deterministic_fin_set(calisto_robust):
    """A plain FreeFormFins must be wrapped in its own stochastic model, the
    same way the other surfaces are."""
    fins = calisto_robust.add_free_form_fins(
        n=4,
        shape_points=[(0, 0), (0.08, 0.1), (0.12, 0.1), (0.12, 0)],
        position=-1.04956,
    )
    stochastic = StochasticRocket(rocket=calisto_robust)

    stochastic.add_free_form_fins(fins)

    added = stochastic.aerodynamic_surfaces.get_tuple_by_type(StochasticFreeFormFins)
    assert len(added) == 1
    assert added[0].component.obj is fins


@pytest.mark.parametrize(
    "add_them, names",
    [
        (
            "add_cp_eccentricity",
            ("cp_eccentricity_x", "cp_eccentricity_y"),
        ),
        (
            "add_thrust_eccentricity",
            ("thrust_eccentricity_x", "thrust_eccentricity_y"),
        ),
    ],
)
def test_an_eccentricity_added_after_init_is_still_drawn(calisto, add_them, names):
    """``dict_generator`` walks the declared inputs, and these arrive later.

    The list is built in ``__init__``, so a distribution installed by an
    ``add_*`` method afterwards was never re-validated on a reseed and stayed
    bound to the unseeded generator: a fixed seed did not reproduce it.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    getattr(stochastic, add_them)(x=(0.0, 0.001), y=(0.0, 0.001))
    stochastic._set_stochastic(42)

    generated = next(stochastic.dict_generator())

    assert set(names) <= set(generated), f"{add_them} was set but never sampled"
    assert all(isinstance(generated[name], Real) for name in names)


def test_two_seeds_move_an_eccentricity_that_was_added_late(calisto):
    """Being present is not enough; it has to follow the seed."""
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(x=(0.0, 0.01), y=(0.0, 0.01))

    def drawn(seed):
        stochastic._set_stochastic(seed)
        return next(stochastic.dict_generator())["cp_eccentricity_x"]

    assert drawn(7) == drawn(7)
    assert drawn(7) != drawn(8)


def test_a_declared_eccentricity_is_not_drawn_a_second_time(calisto):
    """``create_object`` applies the draw ``dict_generator`` already made.

    A second draw spends another value out of the same stream, which moves
    every component position ``create_object`` places after it.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(x=(0.0, 0.01), y=(0.0, 0.01))
    stochastic.add_thrust_eccentricity(x=(0.0, 0.01), y=(0.0, 0.01))

    stochastic._set_stochastic(42)
    declared = next(stochastic.dict_generator())
    expected = {name: declared[name] for name in declared if "eccentricity" in name}
    assert len(expected) == 4

    stochastic._set_stochastic(42)
    rocket = stochastic.create_object()

    applied = {
        "cp_eccentricity_x": rocket.cp_eccentricity_x,
        "cp_eccentricity_y": rocket.cp_eccentricity_y,
        "thrust_eccentricity_x": rocket.thrust_eccentricity_x,
        "thrust_eccentricity_y": rocket.thrust_eccentricity_y,
    }
    assert applied == expected
    assert {name: stochastic.last_rnd_dict[name] for name in expected} == expected


def test_an_eccentricity_half_that_was_left_out_is_still_drawn(calisto):
    """Only a half that was given is a declared input, so the other is not.

    ``create_object`` has to keep drawing it, and keep reporting it, or the
    inputs it writes stop describing the rocket it built.
    """
    stochastic = StochasticRocket(rocket=calisto, radius=0.0127 / 2)
    stochastic.add_cp_eccentricity(x=(0.0, 0.01))

    stochastic._set_stochastic(42)
    rocket = stochastic.create_object()

    assert "cp_eccentricity_y" in stochastic.last_rnd_dict
    assert stochastic.last_rnd_dict["cp_eccentricity_y"] == rocket.cp_eccentricity_y
