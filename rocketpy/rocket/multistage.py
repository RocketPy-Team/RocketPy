"""Vehicle composition layer for multistage rockets and deployable payloads."""

import numpy as np

from rocketpy.rocket.rocket import Rocket
from rocketpy.tools import parallel_axis_theorem_from_com


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


class MultiStageRocket:
    """A launch vehicle composed of stacked stages and carried deployables.

    Owns the derivation problem: given the stages and deployables, produce
    the single-body ``Rocket`` configuration flown during each part of the
    mission. Mass, inertia and center of mass are composed automatically.

    Parameters
    ----------
    stages : list of Stage or Rocket, optional
        Ordered bottom to top: stages[0] burns first (booster), stages[-1]
        is the final, surviving stage (sustainer). A plain Rocket is
        wrapped in a Stage with defaults (single-stage vehicle).
    stack_power_off_drag, stack_power_on_drag : optional
        Drag overrides for the full stack, accepting the same inputs as
        ``Rocket`` drag (constant, callable, CSV, ...). When None and there
        is a single active stage, that stage's own drag curve is reused.
    interstage_lengths : list of float, optional
        Axial adapter/overlap length between consecutive stages, in
        meters, len(stages) - 1 entries. Default zeros.
    name : str
    """

    def __init__(
        self,
        stages=None,
        stack_power_off_drag=None,
        stack_power_on_drag=None,
        interstage_lengths=None,
        name="MultiStageRocket",
    ):
        self.stages = [
            stage
            if isinstance(stage, Stage)
            else Stage(name=f"stage_{index + 1}", rocket=stage)
            for index, stage in enumerate(stages or [])
        ]
        self.deployables = []
        self.stack_power_off_drag = stack_power_off_drag
        self.stack_power_on_drag = stack_power_on_drag
        self.interstage_lengths = interstage_lengths
        self.name = name

    def add_deployable(
        self,
        name,
        mass,
        inertia,
        position,
        radius=None,
        stage=None,
        free_rocket=None,
        ejection=None,
        separation_delta_v=0.0,
    ):
        """Add a carried body that is ejected during flight. Returns the
        Deployable.

        Parameters
        ----------
        stage : str, Stage, optional
            Which stage carries it. Default: the top stage.
        (remaining parameters match Deployable's constructor)
        """
        deployable = Deployable(
            name=name,
            mass=mass,
            inertia=inertia,
            position=position,
            radius=radius,
            free_rocket=free_rocket,
            ejection=ejection,
            separation_delta_v=separation_delta_v,
        )
        deployable.stage = stage if stage is not None else self.stages[-1]
        self.deployables.append(deployable)
        return deployable

    def flight_rocket(self, active_stages, carried_deployables=()):
        """Build the single-body Rocket for one part of the mission.

        Composes mass, inertia and CoM of the listed stages and
        deployables (parallel axis theorem), attaches the bottom
        (first) active stage's motor, and combines the aerodynamic
        surfaces and drag curves of every active stage.

        The bottom stage contributes its structure-only mass/inertia
        (its motor becomes the composed Rocket's own motor). Every
        other active stage is riding along inert - not yet ignited -
        so it contributes its *full* current mass/inertia/CoM (structure
        + motor + full propellant, evaluated at that stage's own t=0)
        as fixed cargo.

        Every active stage's Rocket coordinate system is assumed to
        already be expressed in a shared/stack frame (positions are
        used as-is); deriving stack positions from interstage_lengths
        and each stage's physical extent is not implemented yet.

        Parameters
        ----------
        active_stages : tuple of Stage
            Stages still attached, bottom to top. active_stages[0] is
            the currently firing stage.
        carried_deployables : tuple of Deployable
            Deployables still aboard.

        Returns
        -------
        Rocket
        """
        bottom_rocket = active_stages[0].rocket
        upper_stages = active_stages[1:]

        total_mass, center_of_mass = self._compose_mass_and_center_of_mass(
            bottom_rocket, upper_stages, carried_deployables
        )
        inertia_11, inertia_22, inertia_33 = self._compose_inertia(
            bottom_rocket, upper_stages, carried_deployables, center_of_mass
        )

        radius = max(stage.rocket.radius for stage in active_stages)
        power_off_drag = (
            self.stack_power_off_drag
            if self.stack_power_off_drag is not None
            else self._derive_stack_drag(active_stages, "power_off_drag", radius)
        )
        power_on_drag = (
            self.stack_power_on_drag
            if self.stack_power_on_drag is not None
            else self._derive_stack_drag(active_stages, "power_on_drag", radius)
        )

        composed_rocket = Rocket(
            radius=radius,
            mass=total_mass,
            inertia=(inertia_11, inertia_22, inertia_33),
            power_off_drag=power_off_drag,
            power_on_drag=power_on_drag,
            center_of_mass_without_motor=center_of_mass,
            coordinate_system_orientation=bottom_rocket.coordinate_system_orientation,
        )
        composed_rocket.add_motor(bottom_rocket.motor, bottom_rocket.motor_position)
        for stage in active_stages:
            for surface, position in stage.rocket.aerodynamic_surfaces:
                composed_rocket.aerodynamic_surfaces.add(surface, position)
        composed_rocket.evaluate_center_of_pressure()
        composed_rocket.evaluate_stability_margin()
        composed_rocket.evaluate_static_margin()

        return composed_rocket

    @staticmethod
    def _compose_mass_and_center_of_mass(bottom_rocket, upper_stages, deployables):
        """Total structural mass and its center, without the bottom
        stage's motor (attached separately by the caller).
        """
        total_mass = bottom_rocket.mass
        weighted_com = bottom_rocket.mass * bottom_rocket.center_of_mass_without_motor
        for stage in upper_stages:
            stage_mass = stage.rocket.total_mass(0)
            total_mass += stage_mass
            weighted_com += stage_mass * stage.rocket.center_of_mass(0)
        for deployable in deployables:
            total_mass += deployable.mass
            weighted_com += deployable.mass * deployable.position
        return total_mass, weighted_com / total_mass

    @staticmethod
    def _compose_inertia(bottom_rocket, upper_stages, deployables, center_of_mass):
        """I_11/I_22/I_33 about ``center_of_mass``, via the parallel axis
        theorem, matching the mass composition in
        ``_compose_mass_and_center_of_mass``.
        """
        bottom_distance = center_of_mass - bottom_rocket.center_of_mass_without_motor
        inertia_11 = parallel_axis_theorem_from_com(
            bottom_rocket.I_11_without_motor, bottom_rocket.mass, bottom_distance
        )
        inertia_22 = parallel_axis_theorem_from_com(
            bottom_rocket.I_22_without_motor, bottom_rocket.mass, bottom_distance
        )
        inertia_33 = bottom_rocket.I_33_without_motor
        for stage in upper_stages:
            stage_mass = stage.rocket.total_mass(0)
            distance = center_of_mass - stage.rocket.center_of_mass(0)
            inertia_11 += parallel_axis_theorem_from_com(
                stage.rocket.I_11(0), stage_mass, distance
            )
            inertia_22 += parallel_axis_theorem_from_com(
                stage.rocket.I_22(0), stage_mass, distance
            )
            inertia_33 += stage.rocket.I_33(0)
        for deployable in deployables:
            distance = center_of_mass - deployable.position
            inertia_11 += parallel_axis_theorem_from_com(
                deployable.inertia[0], deployable.mass, distance
            )
            inertia_22 += parallel_axis_theorem_from_com(
                deployable.inertia[1], deployable.mass, distance
            )
            inertia_33 += deployable.inertia[2]
        return inertia_11, inertia_22, inertia_33

    def _derive_stack_drag(self, active_stages, attr_name, stack_radius):
        """Default stack drag curve: each stage's own curve, rescaled by
        its own reference area and summed, referenced to the stack area.
        Documented approximation: ignores interstage interference.
        """
        stack_area = np.pi * stack_radius**2
        combined = None
        for stage in active_stages:
            stage_rocket = stage.rocket
            scaled = getattr(stage_rocket, attr_name) * (
                stage_rocket.area / stack_area
            )
            combined = scaled if combined is None else combined + scaled
        return combined
