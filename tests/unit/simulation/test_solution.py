"""Tests for the flight Solution container.

These tests build solutions by hand and never run a Flight, so they are fast
and safe to run locally. What a phase's states mean, and how its canonical
state is rebuilt, is tested in ``test_dynamics.py``.
"""

import numpy as np
import pytest

from rocketpy.simulation.helpers.dynamics import (
    CANONICAL_INDEX,
    PARACHUTE_DYNAMICS,
    SIX_DOF_DYNAMICS,
    _PhaseDynamics,
)
from rocketpy.simulation.solution import PhaseSolution, Solution


def canonical_row(t, fill=None):
    """Build a 14-value canonical row ``[t, *state]``."""
    state = [float(t)] * 13 if fill is None else list(fill)
    return [float(t), *state]


def descent_row(t, fill=None):
    """Build a 7-value reduced row ``[t, x, y, z, vx, vy, vz]``."""
    state = [float(t)] * 6 if fill is None else list(fill)
    return [float(t), *state]


def stub_derivative(flight, t, u, post_processing=False):
    """Stand-in equations of motion. These tests store rows, never integrate."""
    return [t] if post_processing else list(u)


# Every phase RocketPy ships integrates the full canonical state, so the
# reduced-width behaviour is exercised against a purpose-built phase rather than
# whichever preset happens to be short at the moment.
DESCENT_DYNAMICS = _PhaseDynamics(
    "reduced_descent",
    stub_derivative,
    ("x", "y", "z", "vx", "vy", "vz"),
    ("ax", "ay", "az"),
)


def build_mixed_solution():
    """A solution with a canonical phase followed by a reduced one."""
    solution = Solution()
    solution.start_phase(
        SIX_DOF_DYNAMICS, t_start=0.0, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(3):
        solution.append(canonical_row(t))
    frozen = solution.canonical_row(-1)[1:]
    solution.start_phase(
        DESCENT_DYNAMICS,
        t_start=2.0,
        start_canonical=tuple(frozen),
        name="descent",
    )
    for t in range(3, 6):
        solution.append(descent_row(t))
    return solution


# ---------------------------------------------------------------------------
# PhaseSolution read access
# ---------------------------------------------------------------------------


def test_phase_describes_itself_without_holding_rows():
    """A phase says what it was flown with and where its rows begin."""
    solution = build_mixed_solution()
    phase = solution.phases[1]  # the reduced descent

    # a phase is not a list and does not own rows: they live on the solution,
    # so an index is never mistaken for a row number in the whole flight
    assert not hasattr(phase, "__len__")
    assert not hasattr(phase, "__getitem__")
    assert not hasattr(phase, "rows")
    assert phase.dynamics.width == 6
    assert phase.start == 3
    assert solution.phase_span(1) == (3, 6)


def test_reading_one_phase_of_the_flight():
    """The solution reports a single phase's rows, times and named states."""
    solution = build_mixed_solution()

    assert solution.phase_rows(1) == [descent_row(3), descent_row(4), descent_row(5)]
    assert solution.phase_time(1).tolist() == [3.0, 4.0, 5.0]
    assert solution.phase_series(1, "vz")[:, 1].tolist() == [3.0, 4.0, 5.0]
    assert solution.phase_canonical_array(1).shape == (3, 14)
    # a phase with no rows yet reports nothing rather than failing
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13))
    assert solution.phase_rows(2) == []
    assert solution.phase_time(2).tolist() == []
    assert solution.phase_canonical_array(2).shape == (0, 14)


def test_phase_results_match_the_whole_solution():
    """Per-phase answers line up with the same rows read from the flight."""
    solution = build_mixed_solution()

    assert np.allclose(
        np.vstack(
            [solution.phase_canonical_array(0), solution.phase_canonical_array(1)]
        ),
        solution.canonical_array,
    )
    assert np.allclose(
        np.concatenate([solution.phase_time(0), solution.phase_time(1)]), solution.time
    )
    assert np.allclose(
        np.vstack([solution.phase_series(0, "vz"), solution.phase_series(1, "vz")]),
        solution["vz"],
    )
    assert solution.at_index(4) == solution.at(4.0)
    assert solution.canonical_row(3)[0] == 3.0


def test_at_index_reads_a_single_row():
    """Reading one row by index agrees with reading it by time."""
    solution = build_mixed_solution()

    assert solution.at_index(-2)["vz"] == 4.0
    assert solution.at_index(0)["z"] == 0.0
    assert solution.at_index(-1) == solution.at_index(5)
    assert solution.at_index(4) == solution.at(4.0)


def test_phase_frozen_variable_reads_as_a_constant():
    """A variable the phase does not integrate reads at its start-of-phase value."""
    solution = build_mixed_solution()
    # e0 is not integrated during the descent, so it holds the value it had
    # when the phase began
    assert solution.phase_series(1, "e0")[:, 1].tolist() == [2.0] * 3


def test_phase_unknown_state_raises():
    solution = build_mixed_solution()
    with pytest.raises(KeyError, match="not defined in this flight phase"):
        solution.phase_series(1, "not_a_state")


def test_phase_without_rows_raises():
    solution = Solution()
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13))
    with pytest.raises(KeyError, match="no stored states yet"):
        solution.phase_series(0, "vz")


def test_phase_canonical_array_follows_an_edit():
    """Nothing per-phase is cached, so an edited row shows up immediately."""
    solution = build_mixed_solution()
    assert solution.phase_canonical_array(1)[-1][1] == 5.0
    solution.replace_last(descent_row(5, fill=[9.0] * 6))
    assert solution.phase_canonical_array(1)[-1][1] == 9.0
    # and again when the row is written by position rather than at the tail
    solution[-1] = descent_row(5, fill=[7.0] * 6)
    assert solution.phase_canonical_array(1)[-1][1] == 7.0


# ---------------------------------------------------------------------------
# Reading a state by name
# ---------------------------------------------------------------------------


def test_state_dict_reads_own_and_frozen_variables():
    frozen = [0.0] * 13
    frozen[CANONICAL_INDEX["e0"]] = 0.9
    phase = PhaseSolution(DESCENT_DYNAMICS, start_canonical=frozen)
    state = phase.state_dict([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # variables this phase integrates come from the row
    assert state["z"] == 3.0
    assert state["vz"] == 6.0
    # variables it does not are held at their value when the phase began
    assert state["e0"] == 0.9


def test_state_dict_missing_name_raises():
    phase = PhaseSolution(DESCENT_DYNAMICS, start_canonical=[0.0] * 13)
    with pytest.raises(KeyError):
        _ = phase.state_dict([0, 0, 0, 0, 0, 0])["not_a_state"]


# ---------------------------------------------------------------------------
# Solution list-like behaviour (kept for backward compatibility)
# ---------------------------------------------------------------------------


def test_reading_as_a_list_always_gives_canonical_rows():
    """The list surface reports 14-value rows, whatever the phase stored."""
    solution = build_mixed_solution()
    assert len(solution) == 6
    rows = list(solution)
    assert len(rows) == 6
    # every row is canonical, including the ones a 6-state phase
    # actually integrated
    assert all(len(row) == 14 for row in rows)
    assert len(solution[-1]) == 14
    assert len(solution[0]) == 14
    assert solution[-1][0] == 5.0
    assert solution[2][0] == 2.0
    assert solution[3][0] == 3.0  # first reduced row
    # the attitude the descent held is filled in rather than left out
    assert solution[-1][CANONICAL_INDEX["e0"] + 1] == 2.0


def test_raw_row_gives_the_phase_states():
    """``raw_row`` reports the row exactly as its phase integrated it."""
    solution = build_mixed_solution()
    assert len(solution.raw_row(-1)) == 7  # reduced row
    assert len(solution.raw_row(0)) == 14  # canonical row
    assert solution.raw_row(-1) == descent_row(5)
    assert solution.raw_row(-2)[0] == 4.0


def test_negative_index_across_boundary():
    solution = build_mixed_solution()
    # solution[-4] is the last row of the canonical phase
    assert solution[-4][0] == 2.0
    assert len(solution.raw_row(-4)) == 14
    assert len(solution.raw_row(-3)) == 7  # first reduced row


def test_slice_returns_canonical_rows():
    solution = build_mixed_solution()
    sliced = solution[1:4]
    assert [row[0] for row in sliced] == [1.0, 2.0, 3.0]
    assert all(len(row) == 14 for row in sliced)


def test_tail_accessors_read_the_most_recent_row():
    solution = build_mixed_solution()
    assert solution.last_time == 5.0
    assert solution.raw_row(-1) == descent_row(5)
    assert solution.last_state == descent_row(5)[1:]
    assert len(solution.last_state) == 6  # raw reduced state


def test_tail_accessors_skip_an_empty_new_phase():
    """A phase that has just opened has no rows yet, so the previous one wins."""
    solution = build_mixed_solution()
    solution.start_phase(
        SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13), name="fresh"
    )
    assert solution.last_time == 5.0
    assert solution.last_state == descent_row(5)[1:]
    # the fresh phase is still the one being flown, but it owns no rows
    assert solution.tail is not solution.last_phase
    assert solution.tail.start == len(solution)


def test_tail_accessors_on_an_empty_solution_raise():
    solution = Solution()
    with pytest.raises(IndexError):
        _ = solution.last_time


def test_replace_last_overwrites_the_final_row():
    solution = build_mixed_solution()
    solution.replace_last(descent_row(9))
    assert len(solution) == 6
    assert solution.last_time == 9.0
    assert solution["vz"][-1, 1] == 9.0


def test_insert_before_last_keeps_the_final_row():
    solution = build_mixed_solution()
    solution.insert_before_last(descent_row(4.5))
    assert len(solution) == 7
    assert solution.time.tolist() == [0, 1, 2, 3, 4, 4.5, 5]
    assert solution.last_time == 5.0


def test_drop_last_removes_and_returns_the_raw_row():
    solution = build_mixed_solution()
    row = solution.drop_last()
    assert row == descent_row(5)
    assert len(solution) == 5
    assert solution.last_time == 4.0


def test_tail_mutation_wrong_width_raises():
    solution = build_mixed_solution()
    with pytest.raises(ValueError):
        solution.replace_last(canonical_row(9))
    with pytest.raises(ValueError):
        solution.insert_before_last(canonical_row(9))


def test_append_wrong_width_raises():
    solution = build_mixed_solution()
    with pytest.raises(ValueError):
        solution.append([6.0] * 14)  # canonical width into a reduced tail


def test_setitem_replaces_tail_row():
    solution = build_mixed_solution()
    solution[-1] = descent_row(9)
    assert solution[-1][0] == 9.0


def test_setitem_wrong_width_raises():
    solution = build_mixed_solution()
    with pytest.raises(ValueError):
        solution[-1] = canonical_row(9)


def test_insert_before_tail_first_row():
    """Exact-time insert whose time precedes the tail phase's first row."""
    solution = build_mixed_solution()
    # pop the tail down to a single reduced row
    solution.pop(-1)
    solution.pop(-1)
    assert solution.phase_span(1) == (3, 4)
    solution.insert(-1, descent_row(2.5))
    # inserted before the tail's only row, still in the reduced phase
    assert solution.phase_span(1) == (3, 5)
    assert solution.phase_rows(1)[0][0] == 2.5
    assert solution.tail.dynamics.width == 6


def test_pop_across_boundary():
    solution = build_mixed_solution()
    solution.pop(-1)
    solution.pop(-1)
    solution.pop(-1)  # empties the reduced phase
    assert solution.phase_rows(1) == []
    assert solution.phase_span(1) == (3, 3)
    # the emptied phase is still the one being flown, but the last row is the
    # canonical one before it
    assert solution.last_phase is solution.phases[0]
    # popping again removes the last canonical row
    row = solution.pop(-1)
    assert len(row) == 14


def test_np_array_homogeneous():
    solution = Solution()
    solution.start_phase(SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13))
    for t in range(4):
        solution.append(canonical_row(t))
    array = np.array(solution)
    assert array.shape == (4, 14)
    assert array[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_np_array_mixed_width_gives_the_canonical_table():
    """A ragged flight still turns into one rectangular array, as it used to."""
    solution = build_mixed_solution()
    array = np.array(solution)
    assert array.shape == (6, 14)
    assert array[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.allclose(array, solution.canonical_array)


# ---------------------------------------------------------------------------
# Named queries
# ---------------------------------------------------------------------------


def test_canonical_series_full_timeline_with_freeze():
    solution = build_mixed_solution()
    # vz is defined in every phase (translational)
    vz = solution["vz"]
    assert vz.shape == (6, 2)
    assert vz[:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    # e0 is frozen during the reduced descent at its value at t=2
    e0 = solution["e0"]
    assert e0[:, 1].tolist() == [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_series_partial_covers_only_defining_phases():
    # A phase that integrates a state no other phase has.
    heading_dynamics = _PhaseDynamics(
        "parafoil", stub_derivative, ("x", "y", "z", "vx", "vy", "vz", "heading")
    )
    solution = Solution()
    solution.start_phase(
        SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(2):
        solution.append(canonical_row(t))
    solution.start_phase(
        heading_dynamics, start_canonical=tuple([0.0] * 13), name="parafoil"
    )
    for t in range(2, 5):
        solution.append([float(t), 0, 0, 0, 0, 0, 0, float(t)])  # heading = t
    # the ascent does not define heading, so the history has a gap and says so
    with pytest.warns(UserWarning, match="not defined during ascent"):
        heading = solution["heading"]
    # only the parafoil phase defines heading
    assert heading[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert heading[:, 1].tolist() == [2.0, 3.0, 4.0]


def test_series_covering_every_phase_does_not_warn(recwarn):
    solution = build_mixed_solution()
    assert solution["vz"].shape == (6, 2)
    assert len(recwarn) == 0


def test_series_unknown_name_raises():
    solution = build_mixed_solution()
    with pytest.raises(KeyError):
        _ = solution["not_a_state"]


def test_solution_bad_index_type_raises():
    solution = build_mixed_solution()
    with pytest.raises(TypeError, match="must be integers, slices, or state names"):
        _ = solution[1.5]


def test_series_cache_invalidated_on_append():
    solution = Solution()
    solution.start_phase(SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13))
    for t in range(3):
        solution.append(canonical_row(t))
    first = solution["z"]
    assert first.shape == (3, 2)
    solution.append(canonical_row(3))
    second = solution["z"]
    assert second.shape == (4, 2)


def test_time_and_canonical_array():
    solution = build_mixed_solution()
    assert solution.time.tolist() == [0, 1, 2, 3, 4, 5]
    canonical = solution.canonical_array
    assert canonical.shape == (6, 14)
    # attitude columns frozen during descent
    assert canonical[3:, CANONICAL_INDEX["e0"] + 1].tolist() == [2.0, 2.0, 2.0]


def test_canonical_row_across_boundary():
    solution = build_mixed_solution()
    row = solution.canonical_row(-2)  # a reduced row
    assert len(row) == 14
    assert row[CANONICAL_INDEX["e0"] + 1] == 2.0  # frozen attitude


def test_at_returns_state_by_name_and_warns():
    solution = build_mixed_solution()
    state = solution.at(3.0)
    assert state["z"] == 3.0
    with pytest.warns(UserWarning):
        solution.at(3.4, atol=1e-3)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip():
    solution = build_mixed_solution()
    data = solution.to_dict()
    assert data["format"] == "rocketpy/solution"
    assert data["version"] == 2
    assert len(data["phases"]) == 2
    # the rows are stored once for the whole flight, and each phase says where
    # its own rows begin
    assert len(data["rows"]) == 6
    assert [phase["start"] for phase in data["phases"]] == [0, 3]
    restored = Solution.from_dict(data)
    assert len(restored) == len(solution)
    assert np.allclose(restored.canonical_array, solution.canonical_array)
    # the reduced phase keeps its states even though its key is not a built-in
    assert restored.phases[1].dynamics.states == DESCENT_DYNAMICS.states


def test_to_dict_recovers_a_built_in_dynamics_by_name():
    """A built-in phase comes back as the very same object, derivative and all."""
    solution = Solution()
    solution.start_phase(PARACHUTE_DYNAMICS, t_start=0.0, start_canonical=None)
    solution.append(canonical_row(0))
    restored = Solution.from_dict(solution.to_dict())
    assert restored.phases[0].dynamics is PARACHUTE_DYNAMICS


def test_from_legacy_list():
    rows = [canonical_row(0), canonical_row(1), canonical_row(2)]
    solution = Solution.from_legacy_list(rows)
    assert len(solution) == 3
    assert solution.phases[0].dynamics is SIX_DOF_DYNAMICS
    assert solution["x"][:, 0].tolist() == [0.0, 1.0, 2.0]


def test_from_dict_unknown_dynamics_name_tolerated():
    """A phase whose name is not recognized still reads back its states."""
    restored = Solution.from_dict(
        {
            "format": "rocketpy/solution",
            "version": 2,
            "phases": [
                {
                    "name": "custom",
                    "dynamics": "not_a_known_name",
                    "state_names": ["x", "y", "z", "vx", "vy", "vz"],
                    "t_start": 0.0,
                    "start_canonical": [0.0] * 13,
                    "start": 0,
                }
            ],
            "rows": [descent_row(0)],
        }
    )
    phase = restored.phases[0]
    assert phase.dynamics.states == ("x", "y", "z", "vx", "vy", "vz")
    assert phase.bound_dynamics is None
    assert len(restored) == 1
    assert restored.phase_series(0, "vz")[:, 1].tolist() == [0.0]


def test_from_dict_reads_the_older_per_phase_layout():
    """A flight saved when each phase carried its own rows still loads."""
    solution = build_mixed_solution()
    nested = {
        "format": "rocketpy/solution",
        "version": 1,
        "phases": [
            {
                "name": phase.name,
                "dynamics": phase.dynamics.name,
                "state_names": list(phase.dynamics.states),
                "t_start": phase.t_start,
                "start_canonical": (
                    list(phase.start_canonical)
                    if phase.start_canonical is not None
                    else None
                ),
                "rows": solution.phase_rows(index),
            }
            for index, phase in enumerate(solution.phases)
        ],
    }
    restored = Solution.from_dict(nested)
    assert len(restored) == len(solution)
    assert [phase.start for phase in restored.phases] == [0, 3]
    assert np.allclose(restored.canonical_array, solution.canonical_array)


def test_phase_starts_that_disagree_with_the_rows_are_rejected():
    phase = PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=0)
    later = PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=9)
    with pytest.raises(ValueError, match="only has 2 rows"):
        Solution([phase, later], [canonical_row(0), canonical_row(1)])
    rows = [canonical_row(0), canonical_row(1), canonical_row(2)]
    out_of_order = [
        PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=0),
        PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=2),
        PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=1),
    ]
    with pytest.raises(ValueError, match="order they were flown"):
        Solution(out_of_order, rows)
    not_from_the_start = [PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=1)]
    with pytest.raises(ValueError, match="must start at the first row"):
        Solution(not_from_the_start, rows)


def test_a_bound_dynamics_is_split_from_its_definition():
    """A live phase keeps both the definition and the flight-bound form."""
    bound = SIX_DOF_DYNAMICS.bind(object())
    phase = PhaseSolution(bound, tuple([0.0] * 13))
    assert phase.dynamics is SIX_DOF_DYNAMICS
    assert phase.bound_dynamics is bound


def test_canonical_derivative_zero_fills_unintegrated_states():
    phase = PhaseSolution(DESCENT_DYNAMICS, tuple([0.0] * 13))
    assert phase.canonical_derivative([1, 2, 3, 4, 5, 6]) == [
        1,
        2,
        3,
        4,
        5,
        6,
        *([0.0] * 7),
    ]


def test_a_reduced_phase_needs_the_state_it_starts_from():
    with pytest.raises(ValueError, match="does not integrate"):
        PhaseSolution(DESCENT_DYNAMICS, None)
    # a phase that integrates everything needs no anchor
    assert PhaseSolution(SIX_DOF_DYNAMICS, None).start_canonical is None


# ---------------------------------------------------------------------------
# Phase boundaries: which phase a row belongs to as rows come and go
# ---------------------------------------------------------------------------


def assert_starts_agree(solution):
    """The phases and the solution must never disagree on where rows begin."""
    assert [phase.start for phase in solution.phases] == solution._starts


def test_phase_starts_follow_inserts_and_drops():
    """Adding or removing a row moves every phase that begins after it."""
    solution = build_mixed_solution()
    assert [phase.start for phase in solution.phases] == [0, 3]

    # inserting inside the first phase pushes the second one along
    solution.insert(1, canonical_row(0.5))
    assert [phase.start for phase in solution.phases] == [0, 4]
    assert_starts_agree(solution)

    # ... and removing it puts things back
    solution.pop(1)
    assert [phase.start for phase in solution.phases] == [0, 3]

    # a row inserted exactly at a phase's start joins the later phase
    solution.insert(3, descent_row(2.5))
    assert [phase.start for phase in solution.phases] == [0, 3]
    assert solution.phase_rows(1)[0][0] == 2.5
    solution.pop(3)

    # changes at the very end never move anything
    solution.insert_before_last(descent_row(4.5))
    assert [phase.start for phase in solution.phases] == [0, 3]
    solution.drop_last()
    assert [phase.start for phase in solution.phases] == [0, 3]
    assert_starts_agree(solution)


def test_a_phase_opened_after_a_drop_still_starts_at_the_end():
    """A phase with no rows sits at the end, and moves when rows are removed."""
    solution = build_mixed_solution()
    fresh = solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="fresh")
    assert fresh.start == 6
    solution.drop_last()
    assert fresh.start == 5
    assert solution.phase_rows(2) == []
    assert_starts_agree(solution)


def test_index_resolution_skips_empty_phases():
    """Every row resolves to a phase that owns rows, empty ones passed over."""
    solution = Solution()
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="first")
    solution.append(canonical_row(0))
    # two phases that never stored a row
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="skipped")
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="also skipped")
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="last")
    solution.append(canonical_row(1))

    assert solution.phase_index_at(0) == 0
    assert solution.phase_index_at(1) == 3
    assert solution.phase_index_at(-1) == 3
    assert solution.phases[solution.phase_index_at(1)].name == "last"
    # the empty phases report themselves as holding nothing
    assert solution.phase_rows(1) == []
    assert solution.phase_rows(2) == []
    # and they are left out of a state's history rather than reported as gaps
    assert solution["vz"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Post-process values recorded beside the rows
# ---------------------------------------------------------------------------


def test_post_values_track_every_row_mutation():
    """Values move with their row, so the two can never drift apart."""
    solution = build_mixed_solution()
    # one entry per row from the start, empty until something is recorded
    assert solution.phase_post(0) == [None, None, None]
    assert solution.phase_post(1) == [None, None, None]

    solution.set_last_post([1.0, 2.0, 3.0])
    assert solution.phase_post(1) == [None, None, [1.0, 2.0, 3.0]]

    # overwriting the row's states makes its recorded values stale
    solution.replace_last(descent_row(5, fill=[9.0] * 6))
    assert solution.phase_post(1) == [None, None, None]

    solution.set_last_post([4.0, 5.0, 6.0])
    # a row inserted just before the last one leaves a gap, and the last row
    # keeps the values that belong to it
    solution.insert_before_last(descent_row(4.5))
    assert solution.phase_post(1) == [None, None, None, [4.0, 5.0, 6.0]]

    solution.drop_last()
    assert solution.phase_post(1) == [None, None, None]


def test_post_values_stay_the_same_length_as_the_rows():
    solution = build_mixed_solution()
    for mutate in (
        lambda: solution.append(descent_row(6)),
        lambda: solution.insert(0, canonical_row(-1)),
        lambda: solution.pop(0),
        lambda: solution.insert_before_last(descent_row(5.5)),
        lambda: solution.drop_last(),
        lambda: solution.__setitem__(-1, descent_row(9)),
    ):
        mutate()
        assert len(solution._post_rows) == len(solution)


def test_recording_values_does_not_disturb_the_cached_states():
    """Values are not part of the flight's states, so nothing is rebuilt."""
    solution = build_mixed_solution()
    before = solution.canonical_array
    solution.set_last_post([1.0, 2.0, 3.0])
    assert solution.canonical_array is before


def test_set_last_post_without_rows_raises():
    solution = Solution()
    solution.start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13))
    with pytest.raises(IndexError):
        solution.set_last_post([1.0])
