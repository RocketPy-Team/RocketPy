"""An inert body the vehicle carries and ejects in flight."""

from rocketpy.rocket.multistage.separable_body import SeparableBody


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
            raise ValueError("add_surface requires radius to be set on the deployable.")
        self.surfaces.append((surface, position))
