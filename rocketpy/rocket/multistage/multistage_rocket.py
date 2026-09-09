"""The launch vehicle that stacks stages and carries deployables."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.plots.plot_helpers import show_or_save_plot
from rocketpy.plots.rocket_plots import _default_vis_args
from rocketpy.rocket.aero_surface.fins.fins import Fins
from rocketpy.rocket.aero_surface.nose_cone import NoseCone
from rocketpy.rocket.aero_surface.tail import Tail
from rocketpy.rocket.multistage.deployable import Deployable
from rocketpy.rocket.multistage.geometry import axial_extent
from rocketpy.rocket.multistage.stage import Stage
from rocketpy.rocket.rocket import Rocket
from rocketpy.tools import parallel_axis_theorem_from_com


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

    def _index_of_stage(self, stage):
        """Index of ``stage`` within ``self.stages``, matched by name
        when the object itself isn't a literal member.

        Mission's ignition-time shifting builds a fresh ``Stage`` object
        (same name, motor time-shifted) rather than mutating the
        original in place - callers that walk the mission use that
        fresh object, not the one in ``self.stages``, so an identity
        lookup (``list.index``) would raise even though the stage is
        clearly "the sustainer", just a shifted copy of it. Matching by
        name is safe since stage names are already required to be
        unique within one vehicle (they key Mission's own results).
        """
        for index, candidate in enumerate(self.stages):
            if candidate is stage or candidate.name == stage.name:
                return index
        raise ValueError(f"Stage {stage.name!r} is not part of this MultiStageRocket.")

    def _stack_position_of(self, stage):
        """Axial offset to apply to ``stage``'s own coordinate system so
        it sits correctly stacked above the stages below it.

        Returns 0.0 for the bottom stage, and 0.0 for every stage when
        ``interstage_lengths`` is None (the default) - stages are then
        assumed to already share one coordinate frame, exactly as before
        this method existed. Otherwise, each stage's bottom extent is
        placed exactly ``interstage_lengths[i - 1]`` above the previous
        stage's top extent, stacking contiguously bottom to top.
        """
        if self.interstage_lengths is None:
            return 0.0
        index = self._index_of_stage(stage)
        if index == 0:
            return 0.0

        cumulative_top = self._stage_extent(self.stages[0])[1]
        offset = 0.0
        for i in range(1, index + 1):
            stage_bottom = self._stage_extent(self.stages[i])[0]
            offset = cumulative_top + self.interstage_lengths[i - 1] - stage_bottom
            cumulative_top = self._stage_extent(self.stages[i])[1] + offset
        return offset

    @staticmethod
    def _stage_extent(stage):
        """Axial extent (bottom, top) of ``stage``, in its own local
        coordinate system.

        Derived from aerodynamic surfaces via axial_extent() whenever
        they mark both ends: a NoseCone pins the true top (nothing on
        the vehicle is more forward than the nose tip), a Tail or Fins
        pins the true bottom (the aft-most point).

        A stage can have surfaces that mark only ONE of those ends -
        e.g. fins + tail near the aft end but no nose cone (a booster
        under a sustainer that carries its own separate nose), or a
        bare nose cone with no fins/tail marking the aft end. An
        explicit ``length`` then extends from whichever end IS
        trustworthy, not from the stage's own coordinate origin - for a
        stage built by reusing another rocket's own frame (e.g.
        ``deepcopy(calisto)``), that origin is usually nowhere near
        either physical end (calisto's own motor sits at z=-1.255, far
        from 0), so anchoring there would place the derived extent
        somewhere disconnected from where the stage's surfaces are
        actually drawn.

        Only when the stage has no surfaces at all (nothing trustworthy
        to anchor to) does ``length`` fall back to the stage's own
        origin as its bottom, extending toward the nose. Raises a
        clear, stage-named error when neither is available.
        """
        csys = stage.rocket._csys
        surfaces = [surface for surface, *_ in stage.rocket.aerodynamic_surfaces]
        has_nose = any(isinstance(surface, NoseCone) for surface in surfaces)
        has_aft_marker = any(isinstance(surface, (Tail, Fins)) for surface in surfaces)
        fully_marked = has_nose and has_aft_marker

        if surfaces and not fully_marked and stage.length is not None:
            if has_nose or has_aft_marker:
                partial_bottom, partial_top = axial_extent(stage.rocket)
                if has_nose:
                    top = partial_top
                    bottom = top - csys * stage.length
                else:
                    bottom = partial_bottom
                    top = bottom + csys * stage.length
            else:
                bottom, top = 0.0, csys * stage.length
            return (min(bottom, top), max(bottom, top))

        if surfaces:
            return axial_extent(stage.rocket)

        if stage.length is not None:
            top = csys * stage.length
            return (min(0.0, top), max(0.0, top))

        raise ValueError(
            f"Stage {stage.name!r} has no aerodynamic surfaces, so its "
            "axial extent can't be derived, and interstage_lengths was "
            "given. Pass an explicit length=... when constructing this "
            "Stage, or add aerodynamic surfaces to its rocket."
        )

    @staticmethod
    def _undrawn_body_gap(stage):
        """The portion of ``stage``'s declared extent (from
        :meth:`_stage_extent`) that isn't spanned by any of its own
        aerodynamic surfaces, in the stage's own local coordinate
        system - or ``None`` when there's nothing to fill in.

        Arises two ways: a stage with only a nose cone (or only
        fins/tail) and a ``length`` override has a declared extent
        bigger than what its own surfaces alone would draw - e.g. a
        bare nose cone with no body tube of its own stuck on an
        otherwise plain ``Rocket`` - so only the un-marked side is
        undrawn. A stage with NO surfaces at all - e.g. a bare
        interstage adapter, placed purely from its own ``length`` - has
        no drawn body outline anywhere in its declared extent, so the
        whole thing is undrawn. Either way, without filling this in,
        :meth:`draw` would shade and label a span with no (or only
        partial) drawn body outline in it - a real gap (like the small
        interstage gap between stages) looks indistinguishable from a
        stage whose own body just isn't drawn.
        """
        if stage.length is None:
            return None
        surfaces = [surface for surface, *_ in stage.rocket.aerodynamic_surfaces]
        has_nose = any(isinstance(surface, NoseCone) for surface in surfaces)
        has_aft_marker = any(isinstance(surface, (Tail, Fins)) for surface in surfaces)
        if has_nose and has_aft_marker:
            return None  # both ends already marked, nothing to fill

        declared_bottom, declared_top = MultiStageRocket._stage_extent(stage)
        if not (has_nose or has_aft_marker):
            # No nose, no fins/tail - either no surfaces at all, or only
            # point-type ones (RailButtons, GenericSurface, ...) that
            # don't mark either end - the whole declared extent is
            # undrawn.
            return (declared_bottom, declared_top)
        surfaces_bottom, surfaces_top = axial_extent(stage.rocket)
        if has_nose:
            return (declared_bottom, surfaces_bottom)
        return (surfaces_top, declared_top)

    def _deployable_offset(self, deployable):
        """Stack offset for a deployable's carrying stage, or 0.0 if it
        has no ``.stage`` (e.g. constructed directly rather than via
        ``add_deployable``) - matches the pre-stacking default of
        treating positions as already being in a shared frame.
        """
        stage = getattr(deployable, "stage", None)
        if stage is None:
            return 0.0
        return self._stack_position_of(stage)

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
        already be expressed in a shared/stack frame, *unless*
        ``interstage_lengths`` was given, in which case every position
        below is additionally shifted by that stage's stacking offset
        (see :meth:`_stack_position_of`).

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
        bottom_offset = self._stack_position_of(active_stages[0])

        total_mass, center_of_mass = self._compose_mass_and_center_of_mass(
            active_stages, carried_deployables
        )
        inertia_11, inertia_22, inertia_33 = self._compose_inertia(
            active_stages, carried_deployables, center_of_mass
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
        composed_rocket.add_motor(
            bottom_rocket.motor, bottom_rocket.motor_position + bottom_offset
        )
        for stage in active_stages:
            offset = self._stack_position_of(stage)
            for surface, position, *_ in stage.rocket.aerodynamic_surfaces:
                shifted_position = (
                    position
                    if offset == 0.0
                    else Vector([position.x, position.y, position.z + offset])
                )
                composed_rocket.aerodynamic_surfaces.add(surface, shifted_position)
        composed_rocket.evaluate_center_of_pressure()
        composed_rocket.evaluate_stability_margin()
        composed_rocket.evaluate_static_margin()
        # add_motor() already populated surfaces_cp_to_cdm, but from
        # before these surfaces were copied in above (it was empty at
        # that point) - Flight.u_dot_generalized needs this dict to
        # apply aerodynamic forces during a real 6DOF simulation.
        composed_rocket.evaluate_surfaces_cp_to_cdm()

        return composed_rocket

    def _compose_mass_and_center_of_mass(self, active_stages, deployables):
        """Total structural mass and its center, without the bottom
        stage's motor (attached separately by the caller). Every
        position is shifted by its own stack offset (0.0 unless
        interstage_lengths is set - see _stack_position_of).
        """
        bottom_rocket = active_stages[0].rocket
        bottom_offset = self._stack_position_of(active_stages[0])

        total_mass = bottom_rocket.mass
        weighted_com = bottom_rocket.mass * (
            bottom_rocket.center_of_mass_without_motor + bottom_offset
        )
        for stage in active_stages[1:]:
            offset = self._stack_position_of(stage)
            stage_mass = stage.rocket.total_mass(0)
            total_mass += stage_mass
            weighted_com += stage_mass * (stage.rocket.center_of_mass(0) + offset)
        for deployable in deployables:
            offset = self._deployable_offset(deployable)
            total_mass += deployable.mass
            weighted_com += deployable.mass * (deployable.position + offset)
        return total_mass, weighted_com / total_mass

    def _compose_inertia(self, active_stages, deployables, center_of_mass):
        """I_11/I_22/I_33 about ``center_of_mass``, via the parallel axis
        theorem, matching the mass composition in
        ``_compose_mass_and_center_of_mass`` (including stack offsets).
        """
        bottom_rocket = active_stages[0].rocket
        bottom_offset = self._stack_position_of(active_stages[0])

        bottom_distance = center_of_mass - (
            bottom_rocket.center_of_mass_without_motor + bottom_offset
        )
        inertia_11 = parallel_axis_theorem_from_com(
            bottom_rocket.I_11_without_motor, bottom_rocket.mass, bottom_distance
        )
        inertia_22 = parallel_axis_theorem_from_com(
            bottom_rocket.I_22_without_motor, bottom_rocket.mass, bottom_distance
        )
        inertia_33 = bottom_rocket.I_33_without_motor
        for stage in active_stages[1:]:
            offset = self._stack_position_of(stage)
            stage_mass = stage.rocket.total_mass(0)
            distance = center_of_mass - (stage.rocket.center_of_mass(0) + offset)
            inertia_11 += parallel_axis_theorem_from_com(
                stage.rocket.I_11(0), stage_mass, distance
            )
            inertia_22 += parallel_axis_theorem_from_com(
                stage.rocket.I_22(0), stage_mass, distance
            )
            inertia_33 += stage.rocket.I_33(0)
        for deployable in deployables:
            offset = self._deployable_offset(deployable)
            distance = center_of_mass - (deployable.position + offset)
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
            scaled = getattr(stage_rocket, attr_name) * (stage_rocket.area / stack_area)
            combined = scaled if combined is None else combined + scaled
        return combined

    def _stage_and_deployable_positions(self, stages, deployables):
        """(name, bottom, top) per stage and (name, position) per
        deployable, in stack coordinates - i.e. each stage's own extent
        (see :meth:`_stage_extent`) and each deployable's own position,
        both shifted by their stacking offset (0.0 unless
        interstage_lengths is set). Used to annotate :meth:`draw`.

        A stage whose extent can't be derived (no aerodynamic surfaces
        and no explicit length=...) is silently omitted rather than
        raising - this is cosmetic labeling, not the physics composition
        flight_rocket() does (where the same situation does raise, since
        silently guessing there would produce wrong mass/CoM).
        """
        stage_spans = []
        for stage in stages:
            offset = self._stack_position_of(stage)
            try:
                bottom, top = self._stage_extent(stage)
            except ValueError:
                continue
            stage_spans.append((stage.name, bottom + offset, top + offset))
        deployable_positions = [
            (deployable.name, deployable.position + self._deployable_offset(deployable))
            for deployable in deployables
        ]
        return stage_spans, deployable_positions

    def _draw_other_stages_motors(self, ax, vis_args):
        """Draw every stage past the bottom (currently-firing) one's own
        motor onto ``ax``, at its own stacking-offset-adjusted position.

        flight_rocket() can only attach ONE motor to the composed stack
        Rocket - a Rocket carries a single active motor, so
        ``stack.plots.draw()`` only ever renders the bottom stage's own
        motor, even though every other stage is carrying its own real,
        physical motor as inert cargo. Reuses
        :meth:`~rocketpy.plots.rocket_plots._RocketPlots.draw_motor`
        (the same per-motor-type patch generation ``draw()`` itself
        uses) on a shifted copy of each stage's own rocket, then
        refreshes the legend so each motor's repeated labels ("Grains
        Center of Mass", "Nozzle", ...) collapse to one entry instead of
        one row per stage.
        """
        if not self.stages[1:]:
            return
        for stage in self.stages[1:]:
            offset = self._stack_position_of(stage)
            shifted_rocket = deepcopy(stage.rocket)
            shifted_rocket.motor_position += offset
            shifted_rocket.plots.draw_motor(ax, vis_args)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label not in seen and not seen.add(label)
        ]
        ax.legend(*zip(*unique), bbox_to_anchor=(1.05, 1), loc="upper left")

    def _draw_body_gap(self, stage, ax, vis_args):
        """Fill ``stage``'s undrawn body gap (see
        :meth:`_undrawn_body_gap`) with a plain tube outline at the
        stage's own radius, so its shaded span isn't left looking like
        bare empty space next to whatever surface (nose, fins, tail,
        ...) only marks one end of it.
        """
        gap = self._undrawn_body_gap(stage)
        if gap is None:
            return
        offset = self._stack_position_of(stage)
        bottom, top = gap[0] + offset, gap[1] + offset
        radius = stage.rocket.radius
        ax.plot(
            [bottom, top],
            [radius, radius],
            color=vis_args["body"],
            linewidth=vis_args["line_width"],
        )
        ax.plot(
            [bottom, top],
            [-radius, -radius],
            color=vis_args["body"],
            linewidth=vis_args["line_width"],
        )

    def _draw_stage_spans(self, stage_spans, ax, stack_radius, vis_args):
        """Shade, outline and label every stage's own span, filling in
        each stage's undrawn body gap (see :meth:`_draw_body_gap`).
        """
        stages_by_name = {stage.name: stage for stage in self.stages}
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # Data coordinates (not points-offset from a y=0 marker) so the
        # label's vertical position is predictable regardless of the
        # axes' aspect ratio - see _draw_deployable_markers for the
        # matching deployable-label placement below it.
        stage_label_y = stack_radius * 3.2
        for index, (name, bottom, top) in enumerate(stage_spans):
            color = colors[index % len(colors)]
            ax.axvspan(bottom, top, color=color, alpha=0.08)
            ax.axvline(bottom, color=color, linestyle="--", linewidth=1.0)
            ax.axvline(top, color=color, linestyle="--", linewidth=1.0)
            ax.text(
                (bottom + top) / 2,
                stage_label_y,
                name,
                ha="center",
                va="bottom",
                color=color,
                fontsize=8,
            )
            self._draw_body_gap(stages_by_name[name], ax, vis_args)

    @staticmethod
    def _remove_cross_stage_tube_lines(stage_spans, ax):
        """Remove any straight tube segment ``Rocket.plots.draw()``'s
        own ``_draw_tubes`` drew connecting across a stage boundary.

        ``_draw_tubes`` connects whichever surfaces end up adjacent
        once every stage's surfaces are merged into the composed stack
        and sorted by position - it has no notion of "stage". When one
        stage has no nose of its own (relying on the stage above for
        one) or no aft-marker of its own (relying on the stage below),
        it draws one straight tube connecting directly across the
        interstage gap, at one of the two stages' own - mismatched -
        radius, making the two stages look like the same body
        underneath their own shading. Each stage's own undrawn portion
        is still shown correctly, via :meth:`_draw_body_gap`, which
        never crosses a boundary and so is never removed here.
        """
        if len(stage_spans) < 2:
            return
        sorted_spans = sorted(stage_spans, key=lambda span: span[1])
        boundaries = [
            (sorted_spans[i][2] + sorted_spans[i + 1][1]) / 2
            for i in range(len(sorted_spans) - 1)
        ]
        for line in list(ax.lines):
            xdata = line.get_xdata()
            if len(xdata) != 2:
                continue
            x0, x1 = sorted(xdata)
            if any(x0 < boundary < x1 for boundary in boundaries):
                line.remove()

    @staticmethod
    def _draw_deployable_markers(deployable_positions, ax, stack_radius):
        deployable_label_y = -stack_radius * 3.2
        for name, position in deployable_positions:
            ax.scatter([position], [0], marker="^", color="purple", zorder=11)
            ax.text(
                position,
                deployable_label_y,
                name,
                ha="center",
                va="top",
                color="purple",
                fontsize=8,
            )

    def draw(self, vis_args=None, plane="xz", *, filename=None):
        """Draw the stacked vehicle: every stage's aerodynamic surfaces,
        combined exactly as flight_rocket() would compose them for a
        flight with every stage and deployable attached, with each
        stage's own span and each deployable's own position marked and
        labeled on top of the rocket's silhouette. See
        :meth:`Rocket.plots.draw` for the "at least one aerodynamic
        surface" requirement (on at least one stage).
        """
        stack = self.flight_rocket(
            active_stages=tuple(self.stages),
            carried_deployables=tuple(self.deployables),
        )
        ax = stack.plots.draw(vis_args, plane, return_axes=True)
        effective_vis_args = vis_args if vis_args is not None else _default_vis_args()
        self._draw_other_stages_motors(ax, effective_vis_args)

        stage_spans, deployable_positions = self._stage_and_deployable_positions(
            self.stages, self.deployables
        )
        self._remove_cross_stage_tube_lines(stage_spans, ax)
        self._draw_stage_spans(stage_spans, ax, stack.radius, effective_vis_args)
        self._draw_deployable_markers(deployable_positions, ax, stack.radius)

        # The stage/deployable labels above are centered on their own
        # point and can extend past whatever xlim the underlying
        # Rocket.plots.draw() auto-computed from the drawn geometry alone
        # (text extents aren't included in that autoscale) - widen it so
        # labels near either end aren't clipped by the axes edge.
        annotated_x = [
            bound for _, bottom, top in stage_spans for bound in (bottom, top)
        ]
        annotated_x += [position for _, position in deployable_positions]
        if annotated_x:
            xmin, xmax = ax.get_xlim()
            annotated_x += [xmin, xmax]
            xmin, xmax = min(annotated_x), max(annotated_x)
            margin = 0.12 * (xmax - xmin)
            ax.set_xlim(xmin - margin, xmax + margin)

        show_or_save_plot(filename)
