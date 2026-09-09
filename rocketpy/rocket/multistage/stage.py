"""One stage of a multistage rocket."""

from rocketpy.rocket.multistage.separable_body import SeparableBody


class Stage(SeparableBody):
    """One stage of a multistage rocket.

    Wraps a fully built single-stage ``Rocket`` describing this stage flying
    by itself: structure, motor, aerodynamic surfaces and parachutes.

    Parameters
    ----------
    name : str
        Stage name, e.g. "booster", "sustainer".
    rocket : Rocket
        This stage flying by itself, motor attached.
    separation : Event, optional
        When this stage (and everything below it) is jettisoned from the
        stages above. Top stage: None.
    separation_delta_v : float, optional
        See SeparableBody.
    ignition : Event, optional
        Event that ignites this stage's motor. Default None.
    ignition_delay : float, optional
        Time between the separation of the stage below and this stage's
        motor ignition, in seconds. Default 0.
    length : float, optional
        Total axial length of this stage, used only when stacking with
        ``interstage_lengths`` and the stage's own aerodynamic surfaces
        don't already mark both ends of its extent (see
        :func:`~rocketpy.rocket.multistage.axial_extent` and
        :meth:`MultiStageRocket._stage_extent`). When the stage has a
        NoseCone but no Tail/Fins, ``length`` extends aft from the nose
        tip; when it has a Tail/Fins but no NoseCone, it extends toward
        the nose from the aft-most surface; only when the stage has no
        surfaces at all does it fall back to treating the stage's own
        coordinate origin (position 0) as its bottom.
    """

    def __init__(
        self,
        name,
        rocket,
        separation=None,
        separation_delta_v=0.0,
        ignition=None,
        ignition_delay=0.0,
        length=None,
    ):
        super().__init__(name=name, separation_delta_v=separation_delta_v)
        self.rocket = rocket
        self.separation = separation
        self.ignition = ignition
        self.ignition_delay = ignition_delay
        self.length = length

    @property
    def burn_out_time(self):
        """Burn out time of this stage's motor in the motor's own time."""
        return self.rocket.motor.burn_out_time

    @property
    def dry_mass(self):
        """Stage dry mass (structure + motor dry mass), in kg."""
        return self.rocket.dry_mass
