"""Defines the StochasticParachute class."""

from rocketpy.rocket import Parachute
from rocketpy.rocket.parachute import _is_a_height_trigger

from .stochastic_model import StochasticModel, _sampler_seed


def _is_a_trigger(member):
    """One of the forms ``Parachute`` accepts, and no more.

    The numeric forms defer to ``Parachute``'s own predicate instead of
    restating it. Both were written out separately before and drifted: this one
    kept ``(int, float)`` while ``Parachute`` widened to ``numbers.Real``, so a
    ``numpy.int64`` height was refused here even though the ``Parachute`` it
    would have built accepts it. Calling the same function is what keeps the
    promise that what this accepts is what a parachute accepts.

    That applies to the delay of a ``("time", t_deploy)`` trigger too, so a
    string is refused rather than quietly coerced by ``float()``.
    """
    if callable(member):
        return True
    if isinstance(member, str):
        return member.lower() == "apogee"
    if (
        isinstance(member, (tuple, list))
        and len(member) == 2
        and isinstance(member[0], str)
        and member[0].lower() == "time"
    ):
        return bool(_is_a_height_trigger(member[1]) and member[1] >= 0)
    return _is_a_height_trigger(member)


class StochasticParachute(StochasticModel):
    """A Stochastic Parachute class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`Parachute <rocketpy.rocket.Parachute>`

    Attributes
    ----------
    object : Parachute
        Parachute object to be used for validation.
    cd_s : tuple, list, int, float
        Drag coefficient of the parachute.
    trigger : list
        List of callables, string "apogee", ints/floats, or
        ``("time", t_deploy)`` tuples.
    sampling_rate : tuple, list, int, float
        Sampling rate of the parachute in seconds.
    lag : tuple, list, int, float
        Lag of the parachute in seconds.
    noise : list[tuple]
        List of tuples in the form of (mean, standard deviation,
        time-correlation).
    name : list[str]
        List with the name of the parachute object. This cannot be randomized.
    radius : tuple, list, int, float
        Radius of the parachute in meters.
    drag_coefficient : tuple, list, int, float
        Drag coefficient of the inflated canopy shape, used only when
        ``radius`` is not provided.
    height : tuple, list, int, float
        Height of the parachute in meters.
    porosity : tuple, list, int, float
        Porosity of the parachute.
    """

    def __init__(
        self,
        parachute,
        cd_s=None,
        trigger=None,
        sampling_rate=None,
        lag=None,
        noise=None,
        radius=None,
        drag_coefficient=None,
        height=None,
        porosity=None,
    ):
        """Initializes the Stochastic Parachute class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        parachute : Parachute
            Parachute object to be used for validation.
        cd_s : tuple, list, int, float
            Drag coefficient of the parachute.
        trigger : list
            List of callables, string "apogee", ints/floats, or
            ``("time", t_deploy)`` tuples.
        sampling_rate : tuple, list, int, float
            Sampling rate of the parachute in seconds.
        lag : tuple, list, int, float
            Lag of the parachute in seconds. Pay special attention to ensure
            the lag will not assume negative values based on its mean and
            standard deviation.
        noise : list
            List of tuples in the form of (mean, standard deviation,
            time-correlation).
        radius : tuple, list, int, float
            Radius of the parachute in meters.
        drag_coefficient : tuple, list, int, float
            Drag coefficient of the inflated canopy shape, used only when
            ``radius`` is not provided.
        height : tuple, list, int, float
            Height of the parachute in meters.
        porosity : tuple, list, int, float
            Porosity of the parachute.
        """
        self.parachute = parachute
        self.cd_s = cd_s
        self.trigger = trigger
        self.sampling_rate = sampling_rate
        self.lag = lag
        self.noise = noise
        self.radius = radius
        self.drag_coefficient = drag_coefficient
        self.height = height
        self.porosity = porosity
        self._seed = None

        self._validate_trigger(trigger)
        self._validate_noise(noise)
        super().__init__(
            parachute,
            cd_s=cd_s,
            trigger=trigger,
            sampling_rate=sampling_rate,
            lag=lag,
            noise=noise,
            name=None,
            radius=radius,
            drag_coefficient=drag_coefficient,
            height=height,
            porosity=porosity,
        )

    def _set_stochastic(self, seed=None):
        """Reseed parameter samplers and remember the seed for pressure noise.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator and the derived parachute
            pressure-noise seed.
        """
        self._seed = seed
        super()._set_stochastic(seed)

    def _validate_trigger(self, trigger):
        """Validates the trigger input. If not None, it must be a non-empty
        list whose members are each a callable, the string "apogee", a height,
        or a ``("time", t_deploy)`` tuple. One of those is chosen per
        simulation.
        """
        if trigger is None:
            return

        valid = (
            isinstance(trigger, list)
            and bool(trigger)
            and all(_is_a_trigger(member) for member in trigger)
        )
        # Raised rather than asserted: `python -O` strips an assert, and this
        # is the only thing standing between a bad trigger and a Parachute
        # that either refuses it much later or reads True as a height of 1.
        if not valid:
            raise AssertionError(
                "`trigger` must be a non-empty list whose members are "
                "callables, the string 'apogee', heights, or ('time', t_deploy)"
            )

    def _validate_noise(self, noise):
        """Validates the noise input. If the noise input argument is not
        None, it must be a list of tuples in the form of
        (mean, standard deviation, time-correlation)
        """
        if noise is not None:
            if not (
                isinstance(noise, list)
                and all(isinstance(member, tuple) for member in noise)
            ):
                raise AssertionError(
                    "`noise` must be a list of tuples in the form of "
                    "(mean, standard deviation, time-correlation)"
                )

    def create_object(self):
        """Creates and returns a Parachute object from the randomly generated
        input arguments.

        Returns
        -------
        parachute : Parachute
            Parachute object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        # Tie pressure noise into the Monte Carlo seed tree when one is set.
        # Key by parachute name so drogue and main on the same rocket do not
        # share one noise stream.
        if self._seed is not None:
            generated_dict["seed"] = _sampler_seed(
                self._seed, ("pressure_noise", generated_dict["name"])
            )
            # Recorded after the seed is in, or the inputs describe a parachute
            # with noise nobody can reproduce.
            self._record_draw(generated_dict)
        return Parachute(**generated_dict)
