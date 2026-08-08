from rocketpy.rocket.parachute import Parachute
from rocketpy.rocket.rocket import Rocket
from rocketpy.stochastic import StochasticParachute, StochasticRocket


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
