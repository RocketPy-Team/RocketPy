"""Vehicle composition layer for multistage rockets and deployable payloads."""


class SeparableBody:
    """A body that starts attached to the vehicle and becomes a free body
    when its separation event fires. Base class for Stage and Deployable.

    Parameters
    ----------
    name : str
        Unique body name within the vehicle; used to group Mission
        results (e.g. ``mission.flights["booster"]``).
    separation_delta_v : float, optional
        Relative separation speed along the stack's longitudinal axis,
        in m/s, split by momentum conservation. Default 0.

    Notes
    -----
    Subclasses each hold their own release-event attribute under its own
    name (``Stage.separation``, ``Deployable.ejection``) rather than a
    shared base attribute, since the two events are conceptually similar
    but not interchangeable.
    """

    def __init__(self, name, separation_delta_v=0.0):
        self.name = name
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
        super().__init__(name=name, separation_delta_v=separation_delta_v)
        self.rocket = rocket
        self.separation = separation
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


class Deployable(SeparableBody):
    """An inert body carried by the vehicle and ejected in flight.

    No aerodynamic identity while attached: contributes only mass and
    inertia at a position inside the carrying stage.

    Free-flight aerodynamics come from one of two sources, mutually
    exclusive:
    - ``free_rocket``: a fully built Rocket or PointMassRocket (full
      control);
    - surfaces added with ``add_surface()``: Mission assembles the
      free-flight Rocket from the deployable's mass, inertia, radius and
      the added surfaces.

    Parameters
    ----------
    name : str
        Unique body name; used to group Mission results.
    mass : float
        Carried mass in kg.
    inertia : tuple of float
        Inertia (I11, I22, I33) about the deployable's own center of
        mass, in kg*m^2.
    position : float
        Position of the deployable's center of mass in the carrying
        stage's rocket coordinate system, in meters.
    radius : float, optional
        The deployable's largest radius in meters. Required only when
        defining its free-flight aerodynamics via add_surface().
    free_rocket : Rocket, PointMassRocket, optional
        Free-flight configuration after ejection. Mutually exclusive
        with add_surface().
    ejection : Event, optional
        When the deployable is released, e.g. Event(trigger="apogee").
    separation_delta_v : float, optional
        See SeparableBody.
    """

    def __init__(
        self,
        name,
        mass,
        inertia,
        position,
        radius=None,
        free_rocket=None,
        ejection=None,
        separation_delta_v=0.0,
    ):
        super().__init__(name=name, separation_delta_v=separation_delta_v)
        self.mass = mass
        self.inertia = inertia
        self.position = position
        self.radius = radius
        self.free_rocket = free_rocket
        self.ejection = ejection
        self.surfaces = []

    def add_surface(self, surface, position):
        """Add an aerodynamic surface to the deployable's free flight.

        Takes effect only after ejection; while attached the deployable
        still contributes only mass and inertia. ``position`` is in the
        deployable's own coordinate system, in meters. Requires
        ``radius`` to be set and is mutually exclusive with
        ``free_rocket``.
        """
        if self.free_rocket is not None:
            raise ValueError(
                "add_surface is mutually exclusive with free_rocket; "
                "a free_rocket was already provided for this deployable."
            )
        if self.radius is None:
            raise ValueError(
                "add_surface requires radius to be set on the deployable."
            )
        self.surfaces.append((surface, position))
