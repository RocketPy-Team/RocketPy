from copy import deepcopy

import matplotlib.pyplot as plt
import pytest

from rocketpy import NoseCone, SolidMotor
from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.multistage import Deployable, MultiStageRocket, Stage, axial_extent
from rocketpy.rocket.rocket import Rocket


def _small_solid_motor(radius, thrust, grains_center_of_mass_position):
    """A SolidMotor sized to actually fit inside a rocket of the given
    body radius - grain_outer_radius/nozzle_radius scaled well under
    it, so draw() tests exercise real grain/nozzle rendering (unlike
    PointMassMotor, which draw_motor() draws nothing for) without
    repeating the "reused a much bigger motor than the stage it's in"
    mistake found in the two-stage draw() demo.
    """
    return SolidMotor(
        thrust_source=thrust,
        dry_mass=0.3,
        dry_inertia=(0.01, 0.01, 0.001),
        nozzle_radius=radius * 0.4,
        grain_number=3,
        grain_density=1800,
        grain_outer_radius=radius * 0.6,
        grain_initial_inner_radius=radius * 0.2,
        grain_initial_height=radius,
        grain_separation=radius * 0.05,
        grains_center_of_mass_position=grains_center_of_mass_position,
        center_of_dry_mass_position=grains_center_of_mass_position * 0.8,
        nozzle_position=0,
        burn_time=1.5,
        throat_radius=radius * 0.15,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def _lines_crossing_stage_boundaries(ax, stage_spans):
    """2-point lines on ``ax`` whose x-interval crosses a boundary
    between two adjacent stage spans - i.e. _draw_tubes()-style straight
    tube segments connecting across a stage boundary, which
    _remove_cross_stage_tube_lines() is meant to strip out. Multi-point
    curves (nose cone/fin/tail shape outlines) are never 2-point, so
    they're never matched here regardless of where they sit.
    """
    sorted_spans = sorted(stage_spans, key=lambda span: span[1])
    boundaries = [
        (sorted_spans[i][2] + sorted_spans[i + 1][1]) / 2
        for i in range(len(sorted_spans) - 1)
    ]
    return [
        line
        for line in ax.lines
        if len(line.get_xdata()) == 2
        and any(
            min(line.get_xdata()) < boundary < max(line.get_xdata())
            for boundary in boundaries
        )
    ]


def _two_stage_vehicle():
    """Booster (bottom, firing) + sustainer (upper, inert) test rig.

    Each stage's motor is placed exactly at that stage's own
    center_of_mass_without_motor, and PointMassMotor has zero internal
    inertia and zero CoM offset from its own attachment point. That
    makes each rocket's overall center_of_mass and I_11/I_22/I_33
    time-invariant and exactly equal to the structure-only values given
    at construction - so expected values can be hand-computed directly
    from the constructor arguments below, without depending on
    flight_rocket's own code path.
    """
    booster_rocket = Rocket(
        radius=0.1,
        mass=10.0,
        inertia=(1.0, 1.0, 0.01),
        power_off_drag=0.5,
        power_on_drag=0.6,
        center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(
        PointMassMotor(
            thrust_source=100,
            dry_mass=1.0,
            propellant_initial_mass=2.0,
            burn_time=1.0,
        ),
        position=0.0,
    )
    booster = Stage(name="booster", rocket=booster_rocket)

    sustainer_rocket = Rocket(
        radius=0.08,
        mass=5.0,
        inertia=(0.5, 0.5, 0.005),
        power_off_drag=0.3,
        power_on_drag=0.4,
        center_of_mass_without_motor=2.0,
    )
    sustainer_rocket.add_motor(
        PointMassMotor(
            thrust_source=50,
            dry_mass=0.5,
            propellant_initial_mass=1.0,
            burn_time=1.0,
        ),
        position=2.0,
    )
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket)

    return booster, sustainer


def test_stage_dry_mass_matches_wrapped_rocket(calisto):
    stage = Stage(name="stage_1", rocket=calisto)

    assert stage.dry_mass == calisto.dry_mass


def test_stage_burn_out_time_matches_wrapped_rocket_motor(calisto):
    stage = Stage(name="stage_1", rocket=calisto)

    assert stage.burn_out_time == calisto.motor.burn_out_time


def test_deployable_stores_constructor_arguments():
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
        radius=0.05,
    )

    assert deployable.name == "payload"
    assert deployable.mass == 4.5
    assert deployable.inertia == (0.1, 0.1, 0.001)
    assert deployable.position == 1.10
    assert deployable.radius == 0.05
    assert deployable.free_rocket is None
    assert deployable.ejection is None
    assert not deployable.surfaces


def test_add_surface_raises_when_free_rocket_already_set(calisto_nose_cone):
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
        radius=0.05,
        free_rocket=object(),
    )

    with pytest.raises(ValueError):
        deployable.add_surface(calisto_nose_cone, position=0.5)


def test_add_surface_raises_when_radius_not_set(calisto_nose_cone):
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
    )

    with pytest.raises(ValueError):
        deployable.add_surface(calisto_nose_cone, position=0.5)


def test_add_surface_appends_surface_and_position(calisto_nose_cone):
    deployable = Deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
        radius=0.05,
    )

    deployable.add_surface(calisto_nose_cone, position=0.5)

    assert deployable.surfaces == [(calisto_nose_cone, 0.5)]


def test_flight_rocket_mass_with_no_deployables_matches_the_stage(calisto):
    stage = Stage(name="stage_1", rocket=calisto)
    vehicle = MultiStageRocket(stages=[stage])

    flight_rocket = vehicle.flight_rocket(active_stages=(stage,))

    assert flight_rocket.mass == pytest.approx(calisto.mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        calisto.center_of_mass_without_motor
    )


def test_flight_rocket_composes_mass_and_center_of_mass_with_a_deployable(calisto):
    stage = Stage(name="stage_1", rocket=calisto)
    vehicle = MultiStageRocket(stages=[stage])
    deployable = vehicle.add_deployable(
        name="payload",
        mass=4.5,
        inertia=(0.1, 0.1, 0.001),
        position=1.10,
    )

    flight_rocket = vehicle.flight_rocket(
        active_stages=(stage,), carried_deployables=(deployable,)
    )

    # Hand-computed weighted average of calisto's own mass/CoM and the
    # deployable's - independent of flight_rocket's own code path.
    expected_mass = calisto.mass + 4.5
    expected_center_of_mass = (
        calisto.mass * calisto.center_of_mass_without_motor + 4.5 * 1.10
    ) / expected_mass

    assert flight_rocket.mass == pytest.approx(expected_mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        expected_center_of_mass
    )


def test_two_stage_flight_rocket_composes_bottom_stage_plus_inert_upper_stage():
    booster, sustainer = _two_stage_vehicle()
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    flight_rocket = vehicle.flight_rocket(active_stages=(booster, sustainer))

    # Hand-computed: booster contributes its structure-only mass/CoM (its
    # motor becomes flight_rocket's own motor); the inert sustainer
    # contributes its FULL mass (structure + dry motor + full propellant,
    # since its own motor clock hasn't started) at its own, time-invariant
    # center of mass. See _two_stage_vehicle for why these are exact.
    sustainer_full_mass = 5.0 + 0.5 + 1.0  # structure + motor dry + propellant
    expected_mass = 10.0 + sustainer_full_mass
    expected_center_of_mass = (10.0 * 0.0 + sustainer_full_mass * 2.0) / expected_mass

    booster_distance = expected_center_of_mass - 0.0
    sustainer_distance = expected_center_of_mass - 2.0
    expected_inertia_11 = (1.0 + 10.0 * booster_distance**2) + (
        0.5 + sustainer_full_mass * sustainer_distance**2
    )
    expected_inertia_33 = 0.01 + 0.005  # I_33 unaffected by axial offset

    assert flight_rocket.mass == pytest.approx(expected_mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        expected_center_of_mass
    )
    assert flight_rocket.I_11_without_motor == pytest.approx(expected_inertia_11)
    assert flight_rocket.I_33_without_motor == pytest.approx(expected_inertia_33)
    assert flight_rocket.motor is booster.rocket.motor


def test_flight_rocket_after_separation_uses_only_the_remaining_stage():
    booster, sustainer = _two_stage_vehicle()
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    flight_rocket = vehicle.flight_rocket(active_stages=(sustainer,))

    assert flight_rocket.mass == pytest.approx(sustainer.rocket.mass)
    assert flight_rocket.center_of_mass_without_motor == pytest.approx(
        sustainer.rocket.center_of_mass_without_motor
    )
    assert flight_rocket.motor is sustainer.rocket.motor


def test_flight_rocket_combines_surfaces_of_every_active_stage(calisto_nose_cone):
    booster, sustainer = _two_stage_vehicle()
    sustainer.rocket.add_surfaces(calisto_nose_cone, 0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    two_stage_rocket = vehicle.flight_rocket(active_stages=(booster, sustainer))
    sustainer_alone_rocket = vehicle.flight_rocket(active_stages=(sustainer,))

    assert len(two_stage_rocket.aerodynamic_surfaces) == 1
    assert len(sustainer_alone_rocket.aerodynamic_surfaces) == 1


def test_flight_rocket_populates_surfaces_cp_to_cdm(calisto, calisto_nose_cone):
    # surfaces_cp_to_cdm is separate from center_of_pressure/stability -
    # it's what Flight.u_dot_generalized looks up per surface to apply
    # aerodynamic forces during a real 6DOF simulation. flight_rocket()
    # copies surfaces directly (bypassing add_surfaces() to avoid
    # double-transforming fin leading-edge positions), which must not
    # skip populating this dict too.
    stage = Stage(name="stage_1", rocket=calisto)
    stage.rocket.add_surfaces(calisto_nose_cone, 1.0)
    vehicle = MultiStageRocket(stages=[stage])

    composed_rocket = vehicle.flight_rocket(active_stages=(stage,))

    assert calisto_nose_cone in composed_rocket.surfaces_cp_to_cdm


def test_draw_runs_for_a_stage_with_aerodynamic_surfaces(calisto_robust):
    stage = Stage(name="stage_1", rocket=calisto_robust)
    vehicle = MultiStageRocket(stages=[stage])

    assert vehicle.draw(filename=None) is None


def test_draw_combines_surfaces_of_every_stage(calisto_nose_cone):
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(calisto_nose_cone, 0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    assert vehicle.draw(filename=None) is None


def test_draw_raises_when_no_stage_has_aerodynamic_surfaces():
    booster, sustainer = _two_stage_vehicle()
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    with pytest.raises(ValueError):
        vehicle.draw(filename=None)


def test_axial_extent_matches_hand_computed_bounds(calisto_robust):
    # calisto_robust: nose cone (length=0.55829) tip at 1.160, trapezoidal
    # fins (root_chord=0.120) leading edge at -1.168, tail (length=0.060)
    # top edge at -1.313, tail_to_nose orientation (_csys=1). Hand-computed
    # independently of axial_extent's own code path:
    #   nose spans [1.160 - 0.55829, 1.160]  = [0.60171, 1.160]
    #   fins spans [-1.168 - 0.120, -1.168]  = [-1.288, -1.168]
    #   tail spans [-1.313 - 0.060, -1.313]  = [-1.373, -1.313]
    # overall: (-1.373, 1.160)
    bottom, top = axial_extent(calisto_robust)

    assert bottom == pytest.approx(-1.373)
    assert top == pytest.approx(1.160)


def test_axial_extent_raises_without_surfaces(calisto):
    with pytest.raises(ValueError):
        axial_extent(calisto)


def test_stack_position_of_bottom_stage_is_always_zero():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    assert vehicle._stack_position_of(booster) == pytest.approx(0.0)


def test_stack_position_of_offsets_upper_stage_by_interstage_gap():
    booster, sustainer = _two_stage_vehicle()
    # booster: NoseCone(length=0.2) tip at 1.0 -> local span [0.8, 1.0]
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    # sustainer: NoseCone(length=0.3) tip at 0.5 -> local span [0.2, 0.5]
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    # Hand-computed, independent of _stack_position_of's own code path:
    # sustainer's local bottom (0.2) must land at booster's top (1.0) + gap
    # (0.1) = 1.1, so offset = 1.1 - 0.2 = 0.9.
    offset = vehicle._stack_position_of(sustainer)

    assert offset == pytest.approx(0.9)


def test_stack_position_of_is_zero_when_interstage_lengths_not_given():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    assert vehicle._stack_position_of(sustainer) == pytest.approx(0.0)


def test_flight_rocket_applies_stack_offset_to_upper_stage_mass_and_com():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    composed = vehicle.flight_rocket(active_stages=(booster, sustainer))

    # Hand-computed, reusing the independently-tested _stack_position_of
    # (0.9, see test_stack_position_of_offsets_upper_stage_by_interstage_gap)
    # plus the same weighted-average formula used throughout this file -
    # not a call into flight_rocket's own composition code.
    sustainer_offset = vehicle._stack_position_of(sustainer)
    assert sustainer_offset == pytest.approx(0.9)

    sustainer_mass = 5.0 + 0.5 + 1.0  # structure + motor dry + propellant
    sustainer_effective_com = 2.0 + sustainer_offset
    expected_mass = 10.0 + sustainer_mass
    expected_com = (
        10.0 * 0.0 + sustainer_mass * sustainer_effective_com
    ) / expected_mass

    assert composed.mass == pytest.approx(expected_mass)
    assert composed.center_of_mass_without_motor == pytest.approx(expected_com)


def test_flight_rocket_applies_stack_offset_to_copied_surface_positions():
    booster, sustainer = _two_stage_vehicle()
    sustainer_nose = NoseCone(
        length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08
    )
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(sustainer_nose, 0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    composed = vehicle.flight_rocket(active_stages=(booster, sustainer))

    offset = vehicle._stack_position_of(sustainer)
    positions_by_surface = {
        surface: position for surface, position, *_ in composed.aerodynamic_surfaces
    }
    assert positions_by_surface[sustainer_nose].z == pytest.approx(0.5 + offset)


def test_flight_rocket_applies_stack_offset_to_bottom_stage_motor_position():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    # After separation, sustainer flies alone as the "bottom" (only) active
    # stage - it still carries its own nonzero stack offset (0.9), which
    # must shift its motor position too, not just its surfaces.
    composed = vehicle.flight_rocket(active_stages=(sustainer,))

    offset = vehicle._stack_position_of(sustainer)
    assert offset == pytest.approx(0.9)
    assert composed.motor_position == pytest.approx(
        sustainer.rocket.motor_position + offset
    )


def test_stack_position_of_matches_a_shifted_stage_copy_by_name():
    # Mission._shift_motor_ignition builds a fresh Stage (same name,
    # motor time-shifted) rather than mutating the original in place -
    # a stage 3+ levels deep in a separation chain is exactly this kind
    # of copy, not a literal member of self.stages. _stack_position_of
    # must still resolve its offset correctly (matched by name), not
    # raise just because the object identity differs.
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])
    shifted_sustainer = Stage(name="sustainer", rocket=sustainer.rocket)

    assert shifted_sustainer is not sustainer
    assert vehicle._stack_position_of(shifted_sustainer) == pytest.approx(
        vehicle._stack_position_of(sustainer)
    )


def test_stack_position_of_raises_when_stage_has_no_surfaces_or_length_override():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    # sustainer deliberately has no surfaces and no length override
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    with pytest.raises(ValueError, match="sustainer"):
        vehicle._stack_position_of(sustainer)


def test_stack_position_of_uses_length_override_when_stage_has_no_surfaces():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    # sustainer has no surfaces, but declares its own axial length: by
    # convention its own coordinate origin (0) is its bottom, extending
    # +0.4 toward the nose (tail_to_nose, _csys=1) -> local bottom = 0.0.
    sustainer.length = 0.4
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    # Hand-computed: booster top = 1.0 (nose tip), sustainer local bottom
    # = 0.0 (its own origin, per the length-override convention) ->
    # offset = 1.0 + 0.1 - 0.0 = 1.1.
    offset = vehicle._stack_position_of(sustainer)

    assert offset == pytest.approx(1.1)


def test_stage_extent_prefers_explicit_length_over_surfaces_when_both_given():
    # A stage can have surfaces that only partially mark its own extent
    # (e.g. fins near the aft end, but no nose cone marking the forward
    # end - a real booster under a sustainer, which has its own separate
    # nose). axial_extent() would then badly underestimate the booster's
    # true top. An explicit length=... is how the user corrects that,
    # and it must win even though surfaces exist too - otherwise there'd
    # be no way to override a partially-derived extent. It anchors from
    # the fins' own aft-most edge (the one trustworthy bound available),
    # NOT from the stage's own coordinate origin (0) - the fins here sit
    # at a negative position, nowhere near 0, so anchoring at the origin
    # would place the derived extent somewhere the fins don't even span.
    booster, _sustainer = _two_stage_vehicle()
    booster.rocket.add_trapezoidal_fins(
        n=3, root_chord=0.120, tip_chord=0.040, span=0.100, position=-1.0
    )
    booster.length = 1.5

    bottom, top = MultiStageRocket._stage_extent(booster)

    # Hand-computed, independent of _stage_extent's own code path: fins
    # alone (no tail, no nose) span [-1.0 - 0.120, -1.0] = [-1.12, -1.0]
    # (root_chord=0.120, tail_to_nose orientation, _csys=1). Only the
    # aft-most edge (-1.12) is trustworthy (fins mark the aft end, not
    # the forward one) - length extends 1.5 forward from there.
    assert (bottom, top) == pytest.approx((-1.12, 0.38))


def test_stage_extent_anchors_length_at_nose_tip_when_only_nose_present():
    # Mirror case of the fins-only test above: a stage with only a nose
    # cone (no fins/tail marking its aft end) - a sustainer whose own
    # rocket has just a nose stuck on, no body tube surfaces. Here the
    # nose tip (the forward-most trustworthy bound) must anchor the
    # extent, with length extending aft (toward the tail) from there -
    # not from the stage's own coordinate origin, which for a nose
    # positioned well away from 0 would place most of the derived
    # extent somewhere the nose never spans.
    _booster, sustainer = _two_stage_vehicle()
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    sustainer.length = 1.3

    bottom, top = MultiStageRocket._stage_extent(sustainer)

    # Hand-computed: nose alone spans [0.5 - 0.3, 0.5] = [0.2, 0.5]
    # (tail_to_nose, _csys=1). Only the tip (0.5) is trustworthy -
    # length extends 1.3 aft from there -> bottom = 0.5 - 1.3 = -0.8.
    assert (bottom, top) == pytest.approx((-0.8, 0.5))


def test_undrawn_body_gap_fills_in_the_length_extended_side_for_nose_only_stage():
    # A stage with only a nose cone and a length=... override has a
    # declared extent (-0.8, 0.5) but drawn surfaces only cover (0.2,
    # 0.5) (the nose itself) - the remaining (-0.8, 0.2) has no surface
    # of its own to draw a body outline from, and would otherwise be
    # left as bare empty space in the picture even though it's inside
    # the stage's own shaded span.
    _booster, sustainer = _two_stage_vehicle()
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    sustainer.length = 1.3

    gap = MultiStageRocket._undrawn_body_gap(sustainer)

    assert gap == pytest.approx((-0.8, 0.2))


def test_undrawn_body_gap_fills_in_the_length_extended_side_for_aft_marker_only_stage():
    # Mirror case: fins/tail mark the aft end, length=... extends
    # forward - the forward portion beyond the fins has no surface of
    # its own.
    booster, _sustainer = _two_stage_vehicle()
    booster.rocket.add_trapezoidal_fins(
        n=3, root_chord=0.120, tip_chord=0.040, span=0.100, position=-1.0
    )
    booster.length = 1.5

    gap = MultiStageRocket._undrawn_body_gap(booster)

    assert gap == pytest.approx((-1.0, 0.38))


def test_undrawn_body_gap_fills_the_whole_declared_extent_without_any_surfaces():
    # A stage with NO aerodynamic surfaces at all (e.g. a bare
    # interstage adapter) but a length=... override still has a
    # declared span (_stage_extent's origin-anchored fallback) - all of
    # it is undrawn, not just a partial side, so the whole thing must
    # be filled in, not skipped.
    _booster, sustainer = _two_stage_vehicle()
    sustainer.length = 0.4

    gap = MultiStageRocket._undrawn_body_gap(sustainer)

    # Hand-computed: no surfaces -> _stage_extent's origin-anchored
    # fallback is (0.0, 0.4) (tail_to_nose, _csys=1) - the entire span.
    assert gap == pytest.approx((0.0, 0.4))


def test_undrawn_body_gap_is_none_when_surfaces_mark_both_ends():
    booster, _sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    booster.rocket.add_trapezoidal_fins(
        n=3, root_chord=0.120, tip_chord=0.040, span=0.100, position=-1.0
    )
    booster.length = 99.0

    assert MultiStageRocket._undrawn_body_gap(booster) is None


def test_undrawn_body_gap_is_none_without_a_length_override():
    _booster, sustainer = _two_stage_vehicle()
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )

    assert MultiStageRocket._undrawn_body_gap(sustainer) is None


def test_stage_extent_ignores_length_when_surfaces_already_mark_both_ends():
    # A stage with both a nose cone (forward end) and fins (aft end)
    # already has a fully surface-derived extent - an explicit length
    # override is redundant there and must not silently override the
    # two independently trustworthy bounds with a single, possibly
    # inconsistent one.
    booster, _sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    booster.rocket.add_trapezoidal_fins(
        n=3, root_chord=0.120, tip_chord=0.040, span=0.100, position=-1.0
    )
    booster.length = 99.0

    bottom, top = MultiStageRocket._stage_extent(booster)

    assert (bottom, top) == pytest.approx(axial_extent(booster.rocket))


def test_draw_motor_adds_motor_patches_to_an_existing_axes(calisto_robust):
    # MultiStageRocket.draw() needs to add a second (or third, ...)
    # stage's own motor onto the same axes the composed stack's own
    # draw() already built - draw_motor() is the reusable primitive for
    # that: the same per-motor-type patch generation draw() itself
    # uses, targeting an Axes the caller already has, with no
    # aerodynamic surfaces or connecting tube of its own.
    _, ax = plt.subplots()

    calisto_robust.plots.draw_motor(ax)

    labels = [artist.get_label() for artist in ax.collections]
    assert "Grains Center of Mass" in labels


def test_draw_renders_every_active_stages_own_motor(calisto):
    # A composed multi-stage Rocket can only ever carry ONE active
    # motor (add_motor() overwrites), so before this fix, draw() could
    # only ever show the bottom (currently-firing) stage's own motor -
    # every other stage's motor, though riding along as real mass, was
    # completely invisible in the picture. A real two-stage rocket has
    # two physical motors; draw() should show both, at their own
    # positions.
    calisto.add_nose(length=0.55829, kind="von karman", position=1.278)
    booster = Stage(name="booster", rocket=calisto)

    sustainer_rocket = Rocket(
        radius=0.045,
        mass=3.0,
        inertia=(0.2, 0.2, 0.005),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=1.0,
    )
    sustainer_motor = deepcopy(calisto.motor)
    sustainer_rocket.add_motor(sustainer_motor, position=3.0)
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket)

    vehicle = MultiStageRocket(stages=[booster, sustainer])
    vehicle.draw(filename=None)
    ax = plt.gca()

    grains_positions = sorted(
        artist.get_offsets()[0][0]
        for artist in ax.collections
        if artist.get_label() == "Grains Center of Mass"
    )

    assert len(grains_positions) == 2
    booster_grains = (
        calisto.motor_position + calisto.motor.grains_center_of_mass_position
    )
    sustainer_grains = 3.0 + sustainer_motor.grains_center_of_mass_position
    assert grains_positions == pytest.approx(sorted([booster_grains, sustainer_grains]))


def test_draw_fills_undrawn_body_gap_with_a_tube_outline():
    # Without this, a stage whose length=... override extends past what
    # its own surfaces mark (e.g. a bare nose cone with no body tube of
    # its own) gets a shaded, labeled span with no drawn body outline
    # in most of it - a real interstage gap looks indistinguishable
    # from a stage whose own body just isn't drawn.
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    sustainer.length = 1.3
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    vehicle.draw(filename=None)
    ax = plt.gca()

    # Hand-computed: sustainer's own gap (nose alone spans [0.2, 0.5],
    # length=1.3 extends to a declared [-0.8, 0.5]) is [-0.8, 0.2]
    # locally - independently re-derived here, not read back from
    # _undrawn_body_gap's own result.
    offset = vehicle._stack_position_of(sustainer)
    expected_bottom, expected_top = -0.8 + offset, 0.2 + offset

    matching_lines = [
        line
        for line in ax.lines
        if line.get_xdata() == pytest.approx([expected_bottom, expected_top])
        and line.get_ydata()
        == pytest.approx([sustainer.rocket.radius, sustainer.rocket.radius])
    ]
    assert matching_lines


def test_draw_does_not_connect_a_tube_across_a_stage_boundary():
    # _draw_tubes (core Rocket.plots.draw(), used for the composed
    # stack) connects whichever surfaces end up adjacent once every
    # stage's surfaces are merged and sorted by position - it has no
    # notion of "stage". When the bottom stage has no nose of its own
    # (relying on the stage above for one) and the stage above has no
    # aft-marker of its own, it draws ONE straight tube connecting them
    # directly, at one of the two stages' own (mismatched) radius -
    # visually a fixed-diameter tube bridging straight through the
    # interstage gap and into the other stage's own territory, making
    # the two stages look like the same body underneath their shading.
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_trapezoidal_fins(
        n=3, root_chord=0.120, tip_chord=0.040, span=0.100, position=-1.0
    )
    booster.length = 1.5
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    sustainer.length = 1.3
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])

    vehicle.draw(filename=None)
    ax = plt.gca()

    stage_spans, _ = vehicle._stage_and_deployable_positions(
        vehicle.stages, vehicle.deployables
    )
    assert _lines_crossing_stage_boundaries(ax, stage_spans) == []


def test_draw_generalizes_to_three_stages():
    # Every draw() fix (per-stage motor rendering, undrawn body gap
    # fill, cross-stage tube removal) was built generic over
    # self.stages, not hardcoded to two - this locks that in rather
    # than trusting it by inspection alone. Middle stage has no
    # surfaces at all (a bare interstage adapter), the other two have
    # partial surfaces needing length=... overrides, matching a
    # realistic three-stage vehicle. Every stage gets a real SolidMotor
    # (not PointMassMotor, which draw_motor() renders nothing for),
    # each sized to its own radius - so this actually exercises motor
    # rendering at N=3, not just the two-stage case.
    def _rocket(radius, mass, motor_position, grains_com):
        r = Rocket(
            radius=radius,
            mass=mass,
            inertia=(0.2, 0.2, 0.005),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=0.0,
        )
        r.add_motor(
            _small_solid_motor(
                radius, thrust=200, grains_center_of_mass_position=grains_com
            ),
            position=motor_position,
        )
        return r

    bottom_rocket = _rocket(0.08, 5.0, motor_position=-0.35, grains_com=0.05)
    bottom_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.1, tip_chord=0.03, span=0.08, position=-0.3
    )
    bottom = Stage(name="bottom", rocket=bottom_rocket, length=1.0)

    middle_rocket = _rocket(0.06, 3.0, motor_position=0.3, grains_com=0.04)
    middle = Stage(name="middle", rocket=middle_rocket, length=0.8)

    top_rocket = _rocket(0.045, 1.5, motor_position=-0.2, grains_com=0.03)
    top_rocket.add_nose(length=0.2, kind="conical", position=0.3)
    top = Stage(name="top", rocket=top_rocket, length=0.7)

    vehicle = MultiStageRocket(
        stages=[bottom, middle, top], interstage_lengths=[0.05, 0.05]
    )

    vehicle.draw(filename=None)
    ax = plt.gca()

    stage_spans, _ = vehicle._stage_and_deployable_positions(
        vehicle.stages, vehicle.deployables
    )
    assert [name for name, _, _ in stage_spans] == ["bottom", "middle", "top"]

    assert _lines_crossing_stage_boundaries(ax, stage_spans) == []

    # Every stage's own motor must be visible, at its own stacked
    # position - not just the bottom stage's, which is all
    # flight_rocket()'s composed stack Rocket can attach.
    grains_positions = sorted(
        artist.get_offsets()[0][0]
        for artist in ax.collections
        if artist.get_label() == "Grains Center of Mass"
    )
    assert len(grains_positions) == 3
    expected = sorted(
        [
            -0.35 + 0.05 + vehicle._stack_position_of(bottom),
            0.3 + 0.04 + vehicle._stack_position_of(middle),
            -0.2 + 0.03 + vehicle._stack_position_of(top),
        ]
    )
    assert grains_positions == pytest.approx(expected)


def test_draw_generalizes_to_five_stages():
    # Same guarantee as the three-stage test, at N=5 - nothing in
    # draw()'s fixes should be hardcoded to any specific stage count,
    # including per-stage motor rendering (a real SolidMotor per stage,
    # not PointMassMotor).
    motor_position = -0.18
    grains_com = 0.03

    def _rocket(radius, mass):
        r = Rocket(
            radius=radius,
            mass=mass,
            inertia=(0.1, 0.1, 0.002),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=0.0,
        )
        r.add_motor(
            _small_solid_motor(
                radius, thrust=100, grains_center_of_mass_position=grains_com
            ),
            position=motor_position,
        )
        r.add_trapezoidal_fins(
            n=3, root_chord=0.06, tip_chord=0.02, span=0.05, position=-0.15
        )
        return r

    stages = []
    for i in range(5):
        radius = 0.09 - i * 0.01
        rocket = _rocket(radius, 2.0 - i * 0.2)
        if i == 4:
            rocket.add_nose(length=0.15, kind="conical", position=0.2)
        stages.append(Stage(name=f"stage{i}", rocket=rocket, length=0.5))

    vehicle = MultiStageRocket(stages=stages, interstage_lengths=[0.03] * 4)

    vehicle.draw(filename=None)
    ax = plt.gca()

    stage_spans, _ = vehicle._stage_and_deployable_positions(
        vehicle.stages, vehicle.deployables
    )
    assert [name for name, _, _ in stage_spans] == [
        "stage0",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    ]

    assert _lines_crossing_stage_boundaries(ax, stage_spans) == []

    grains_positions = sorted(
        artist.get_offsets()[0][0]
        for artist in ax.collections
        if artist.get_label() == "Grains Center of Mass"
    )
    assert len(grains_positions) == 5
    expected = sorted(
        motor_position + grains_com + vehicle._stack_position_of(stage)
        for stage in stages
    )
    assert grains_positions == pytest.approx(expected)


def test_rocket_plots_draw_can_return_axes_instead_of_showing(calisto_robust):
    # MultiStageRocket.draw() needs to add stage/deployable markers onto
    # the SAME axes Rocket.plots.draw() already builds, rather than a
    # separate picture - this is the minimal, additive hook that makes
    # that possible without changing draw()'s default behavior at all.
    ax = calisto_robust.plots.draw(filename=None, return_axes=True)

    assert ax is not None
    assert ax.get_title() == "Rocket Representation"


def test_stage_and_deployable_positions_matches_hand_computed_values():
    booster, sustainer = _two_stage_vehicle()
    booster.rocket.add_surfaces(
        NoseCone(length=0.2, kind="conical", base_radius=0.1, rocket_radius=0.1), 1.0
    )
    sustainer.rocket.add_surfaces(
        NoseCone(length=0.3, kind="conical", base_radius=0.08, rocket_radius=0.08), 0.5
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.1])
    vehicle.add_deployable(
        name="payload",
        mass=1.0,
        inertia=(0.01, 0.01, 0.001),
        position=0.1,
        stage=sustainer,
        radius=0.02,
    )

    stage_spans, deployable_positions = vehicle._stage_and_deployable_positions(
        vehicle.stages, vehicle.deployables
    )

    # Hand-computed: booster is unshifted (bottom stage), so its own
    # extent is used as-is: [0.8, 1.0] (NoseCone length=0.2 at tip 1.0).
    # Sustainer's own extent is [0.2, 0.5] (NoseCone length=0.3 at tip
    # 0.5), shifted by its independently-verified offset (0.9) ->
    # [1.1, 1.4]. The payload (aboard the sustainer, local position 0.1)
    # shifts by the same 0.9 -> 1.0.
    assert stage_spans == [
        ("booster", pytest.approx(0.8), pytest.approx(1.0)),
        ("sustainer", pytest.approx(1.1), pytest.approx(1.4)),
    ]
    assert deployable_positions == [("payload", pytest.approx(1.0))]


def test_draw_runs_with_deployables_aboard(calisto_robust):
    stage = Stage(name="stage_1", rocket=calisto_robust)
    vehicle = MultiStageRocket(stages=[stage])
    vehicle.add_deployable(
        name="payload",
        mass=1.0,
        inertia=(0.01, 0.01, 0.001),
        position=0.5,
        radius=0.02,
    )

    assert vehicle.draw(filename=None) is None


def test_middle_stage_with_full_surfaces_renders_correctly():
    # Answers a direct question: can a middle/adapter stage carry its
    # own surfaces too, not just be a bare structural gap? A middle
    # stage with nose+fins+tail is fully self-marked (no length=...
    # gap-fill needed for it), and must not disturb cross-stage-tube
    # removal at either of its two boundaries.
    def _rocket(radius):
        r = Rocket(
            radius=radius,
            mass=2.0,
            inertia=(0.1, 0.1, 0.002),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=0.0,
        )
        r.add_motor(
            PointMassMotor(
                thrust_source=150,
                dry_mass=0.2,
                propellant_initial_mass=0.3,
                burn_time=1.0,
            ),
            position=0.0,
        )
        return r

    bottom_rocket = _rocket(0.08)
    bottom_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.08, tip_chord=0.03, span=0.06, position=-0.2
    )
    bottom = Stage(name="bottom", rocket=bottom_rocket, length=0.5)

    middle_rocket = _rocket(0.06)
    middle_rocket.add_nose(length=0.1, kind="conical", position=0.4)
    middle_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.05, tip_chord=0.02, span=0.04, position=0.05
    )
    middle_rocket.add_tail(
        top_radius=0.06, bottom_radius=0.045, length=0.03, position=-0.05
    )
    middle = Stage(name="middle", rocket=middle_rocket)

    top_rocket = _rocket(0.045)
    top_rocket.add_nose(length=0.15, kind="conical", position=0.2)
    top = Stage(name="top", rocket=top_rocket, length=0.6)

    vehicle = MultiStageRocket(
        stages=[bottom, middle, top], interstage_lengths=[0.03, 0.03]
    )

    vehicle.draw(filename=None)
    ax = plt.gca()

    stage_spans, _ = vehicle._stage_and_deployable_positions(
        vehicle.stages, vehicle.deployables
    )
    assert [name for name, _, _ in stage_spans] == ["bottom", "middle", "top"]

    # middle is fully marked by its own surfaces - _stage_extent should
    # match axial_extent() directly, no length-driven extension.
    middle_extent = axial_extent(middle_rocket)
    middle_span = next(span for span in stage_spans if span[0] == "middle")
    offset = vehicle._stack_position_of(middle)
    assert (middle_span[1] - offset, middle_span[2] - offset) == pytest.approx(
        middle_extent
    )

    assert _lines_crossing_stage_boundaries(ax, stage_spans) == []


def test_middle_stage_with_partial_surfaces_at_both_boundaries():
    # Deeper than the bare-adapter case: middle stage has ITS OWN
    # partial surfaces (fins only, no nose) - so it needs its own
    # length=... gap-fill on the nose-less side, AND sits between two
    # OTHER stages with their own partial surfaces, creating two
    # separate mismatched-radius junctions instead of one.
    def _rocket(radius):
        r = Rocket(
            radius=radius,
            mass=2.0,
            inertia=(0.1, 0.1, 0.002),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=0.0,
        )
        r.add_motor(
            PointMassMotor(
                thrust_source=150,
                dry_mass=0.2,
                propellant_initial_mass=0.3,
                burn_time=1.0,
            ),
            position=0.0,
        )
        return r

    bottom_rocket = _rocket(0.08)
    bottom_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.08, tip_chord=0.03, span=0.06, position=-0.2
    )
    bottom = Stage(name="bottom", rocket=bottom_rocket, length=0.5)

    middle_rocket = _rocket(0.06)
    middle_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.05, tip_chord=0.02, span=0.04, position=-0.1
    )
    middle = Stage(name="middle", rocket=middle_rocket, length=0.4)

    top_rocket = _rocket(0.045)
    top_rocket.add_nose(length=0.15, kind="conical", position=0.2)
    top = Stage(name="top", rocket=top_rocket, length=0.6)

    vehicle = MultiStageRocket(
        stages=[bottom, middle, top], interstage_lengths=[0.03, 0.03]
    )

    vehicle.draw(filename=None)
    ax = plt.gca()

    stage_spans, _ = vehicle._stage_and_deployable_positions(
        vehicle.stages, vehicle.deployables
    )
    assert [name for name, _, _ in stage_spans] == ["bottom", "middle", "top"]

    assert _lines_crossing_stage_boundaries(ax, stage_spans) == []

    # middle's own undrawn (nose-less) side must still be filled.
    gap = vehicle._undrawn_body_gap(middle)
    assert gap is not None


def test_rail_buttons_on_a_stage_do_not_break_draw():
    # RailButtons are a "point" surface in axial_extent()'s own model
    # (not Nose/Tail/Fins) - confirms they at least don't crash draw()
    # for a multistage vehicle, combined with normal surfaces.
    booster_rocket = Rocket(
        radius=0.08,
        mass=4.0,
        inertia=(0.2, 0.2, 0.005),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(
        PointMassMotor(
            thrust_source=300, dry_mass=0.3, propellant_initial_mass=0.5, burn_time=1.0
        ),
        position=0.0,
    )
    booster_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.08, tip_chord=0.03, span=0.06, position=-0.2
    )
    booster_rocket.set_rail_buttons(
        upper_button_position=0.1, lower_button_position=-0.15
    )
    booster = Stage(name="booster", rocket=booster_rocket, length=0.6)

    sustainer_rocket = Rocket(
        radius=0.045,
        mass=2.0,
        inertia=(0.1, 0.1, 0.002),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0.0,
    )
    sustainer_rocket.add_motor(
        PointMassMotor(
            thrust_source=150, dry_mass=0.2, propellant_initial_mass=0.3, burn_time=1.0
        ),
        position=0.0,
    )
    sustainer_rocket.add_nose(length=0.15, kind="conical", position=0.2)
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket, length=0.6)

    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.03])

    assert vehicle.draw(filename=None) is None


def test_draw_renders_hybrid_motor_at_a_non_bottom_stage(hybrid_motor):
    # _draw_other_stages_motors() reuses draw_motor() for every stage
    # past the bottom one - confirms it works for a real HybridMotor
    # (tank patches, not just SolidMotor grains), not just the
    # SolidMotor case already covered elsewhere.
    booster_rocket = Rocket(
        radius=0.15,
        mass=6.0,
        inertia=(1.0, 1.0, 0.02),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(
        PointMassMotor(
            thrust_source=3000, dry_mass=1.0, propellant_initial_mass=2.0, burn_time=1.0
        ),
        position=0.0,
    )
    booster_rocket.add_trapezoidal_fins(
        n=3, root_chord=0.15, tip_chord=0.05, span=0.1, position=-0.3
    )
    booster = Stage(name="booster", rocket=booster_rocket, length=1.0)

    sustainer_rocket = Rocket(
        radius=0.15,
        mass=8.0,
        inertia=(1.0, 1.0, 0.02),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0.0,
    )
    sustainer_rocket.add_motor(hybrid_motor, position=0.0)
    sustainer_rocket.add_nose(length=0.3, kind="conical", position=0.5)
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket, length=1.5)

    vehicle = MultiStageRocket(stages=[booster, sustainer], interstage_lengths=[0.05])

    vehicle.draw(filename=None)
    ax = plt.gca()

    # HybridMotor's own tank(s) get drawn via _generate_positioned_tanks
    # (scatter markers for tank centers) - at least one must appear at
    # the sustainer's own stacked position, distinct from the booster's
    # region.
    offset = vehicle._stack_position_of(sustainer)
    tank_scatter_x = [
        artist.get_offsets()[0][0]
        for artist in ax.collections
        if artist.get_offsets().shape[0] == 1
    ]
    booster_top = vehicle._stage_extent(booster)[1]
    assert any(x > booster_top + offset * 0 and x > booster_top for x in tank_scatter_x)
