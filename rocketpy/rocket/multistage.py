"""Vehicle composition layer for multistage rockets and deployable payloads."""


class SeparableBody:
    """A body that starts attached to the vehicle and becomes a free body
    when its separation event fires. Base class for Stage and Deployable.

    Parameters
    ----------
    name : str
        Unique body name within the vehicle; used to group Mission
        results (e.g. ``mission.flights["booster"]``).
    separation : Event, optional
        Event that releases this body. If None, never separates.
    separation_delta_v : float, optional
        Relative separation speed along the stack's longitudinal axis,
        in m/s, split by momentum conservation. Default 0.
    """

    def __init__(self, name, separation=None, separation_delta_v=0.0):
        self.name = name
        self.separation = separation
        self.separation_delta_v = separation_delta_v


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
    """

    def __init__(
        self,
        name,
        rocket,
        separation=None,
        separation_delta_v=0.0,
        ignition=None,
        ignition_delay=0.0,
    ):
        super().__init__(
            name=name, separation=separation, separation_delta_v=separation_delta_v
        )
        self.rocket = rocket
        self.ignition = ignition
        self.ignition_delay = ignition_delay

    @property
    def burn_out_time(self):
        """Burn out time of this stage's motor in the motor's own time."""
        return self.rocket.motor.burn_out_time

    @property
    def dry_mass(self):
        """Stage dry mass (structure + motor dry mass), in kg."""
        return self.rocket.dry_mass
