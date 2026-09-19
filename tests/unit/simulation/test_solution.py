"""Tests for the flight Solution container.

These tests build solutions by hand and never run a Flight, so they are fast
and safe to run locally. What a phase's states mean, and how its canonical
state is rebuilt, is tested in ``test_dynamics.py``.
"""

import numpy as np
import pytest

from rocketpy.simulation.helpers.dynamics import (
    BUILT_IN_DYNAMICS,
    CANONICAL_INDEX,
    CANONICAL_STATE_NAMES,
    FULL_POST_PROCESS_VARS,
    PARACHUTE_DYNAMICS,
    SIX_DOF_DYNAMICS,
    _PhaseDynamics,
)
from rocketpy.simulation.solution import Solution, _PhaseSolution


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
    solution._start_phase(
        SIX_DOF_DYNAMICS, t_start=0.0, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(3):
        solution._append(canonical_row(t))
    frozen = solution.canonical_row(-1)[1:]
    solution._start_phase(
        DESCENT_DYNAMICS,
        t_start=2.0,
        start_canonical=tuple(frozen),
        name="descent",
    )
    for t in range(3, 6):
        solution._append(descent_row(t))
    return solution


# ---------------------------------------------------------------------------
# _PhaseSolution read access
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


def phase_rows(solution, phase_index):
    """One phase's rows as stored, read through its span."""
    start, stop = solution.phase_span(phase_index)
    return [solution.raw_row(index) for index in range(start, stop)]


def test_reading_one_phase_of_the_flight():
    """A phase's span picks its rows out of the whole flight."""
    solution = build_mixed_solution()

    start, stop = solution.phase_span(1)
    assert phase_rows(solution, 1) == [descent_row(3), descent_row(4), descent_row(5)]
    assert solution.time[start:stop].tolist() == [3.0, 4.0, 5.0]
    assert solution["vz"][start:stop, 1].tolist() == [3.0, 4.0, 5.0]
    assert solution.canonical_array[start:stop].shape == (3, 14)
    assert solution[start:stop] == [solution.canonical_row(i) for i in range(3, 6)]
    # a phase with no rows yet spans nothing rather than failing
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13))
    assert solution.phase_span(2) == (6, 6)
    assert phase_rows(solution, 2) == []
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
    start, stop = solution.phase_span(1)
    assert solution["e0"][start:stop, 1].tolist() == [2.0] * 3
    assert solution.value_at(start, "e0") == 2.0


def test_phase_unknown_state_raises():
    solution = build_mixed_solution()
    with pytest.raises(KeyError, match="not defined in this flight phase"):
        solution.value_at(3, "not_a_state")


def test_canonical_array_follows_an_edit():
    """The cached table is rebuilt after a row changes."""
    solution = build_mixed_solution()
    assert solution.canonical_array[-1][1] == 5.0
    solution._replace_last(descent_row(5, fill=[9.0] * 6))
    assert solution.canonical_array[-1][1] == 9.0
    # and again when the row is written by position rather than at the tail
    solution._set_row(-1, descent_row(5, fill=[7.0] * 6))
    assert solution.canonical_array[-1][1] == 7.0
    assert solution.time[-1] == 5.0


# ---------------------------------------------------------------------------
# Reading a state by name
# ---------------------------------------------------------------------------


def test_state_dict_reads_own_and_frozen_variables():
    frozen = [0.0] * 13
    frozen[CANONICAL_INDEX["e0"]] = 0.9
    phase = _PhaseSolution(DESCENT_DYNAMICS, start_canonical=frozen)
    state = phase.state_dict([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # variables this phase integrates come from the row
    assert state["z"] == 3.0
    assert state["vz"] == 6.0
    # variables it does not are held at their value when the phase began
    assert state["e0"] == 0.9


def test_state_dict_missing_name_raises():
    phase = _PhaseSolution(DESCENT_DYNAMICS, start_canonical=[0.0] * 13)
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
    assert [row[0] for row in solution[-2:]] == [4.0, 5.0]
    assert [row[0] for row in solution[::2]] == [0.0, 2.0, 4.0]
    assert solution[4:1] == []


def test_slice_only_builds_the_rows_it_returns():
    """Slicing must not walk the whole flight, which callbacks do per step."""
    solution = build_mixed_solution()
    built = []
    original = Solution.canonical_row
    try:
        Solution.canonical_row = lambda self, index: (
            built.append(index) or original(self, index)
        )
        assert len(solution[-2:]) == 2
    finally:
        Solution.canonical_row = original
    assert built == [4, 5]


def test_tail_accessors_read_the_most_recent_row():
    solution = build_mixed_solution()
    assert solution.raw_row(-1)[0] == 5.0
    assert solution.raw_row(-1) == descent_row(5)
    assert solution.raw_row(-1)[1:] == descent_row(5)[1:]
    assert len(solution.raw_row(-1)[1:]) == 6  # raw reduced state


def test_the_phase_being_flown_is_not_always_the_one_holding_the_last_row():
    """A phase that has just opened has no rows yet, so the previous one wins."""
    solution = build_mixed_solution()
    solution._start_phase(
        SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13), name="fresh"
    )
    assert solution.raw_row(-1)[0] == 5.0
    assert solution.raw_row(-1)[1:] == descent_row(5)[1:]
    # the fresh phase is still the one being flown, but it owns no rows
    assert solution.phase_index_at(-1) == 1
    assert solution.phases[-1].start == len(solution)


def test_phase_accessors_on_an_empty_solution_raise():
    solution = Solution()
    with pytest.raises(IndexError):
        _ = solution.raw_row(-1)[0]


def test_replace_last_overwrites_the_final_row():
    solution = build_mixed_solution()
    solution._replace_last(descent_row(9))
    assert len(solution) == 6
    assert solution.raw_row(-1)[0] == 9.0
    assert solution["vz"][-1, 1] == 9.0


def test_insert_before_last_keeps_the_final_row():
    solution = build_mixed_solution()
    solution._insert_before_last(descent_row(4.5))
    assert len(solution) == 7
    assert solution.time.tolist() == [0, 1, 2, 3, 4, 4.5, 5]
    assert solution.raw_row(-1)[0] == 5.0


def test_drop_last_removes_and_returns_the_raw_row():
    solution = build_mixed_solution()
    row = solution._drop_last()
    assert row == descent_row(5)
    assert len(solution) == 5
    assert solution.raw_row(-1)[0] == 4.0


def test_setitem_replaces_tail_row():
    solution = build_mixed_solution()
    solution._set_row(-1, descent_row(9))
    assert solution[-1][0] == 9.0


def test_insert_before_tail_first_row():
    """Exact-time insert whose time precedes the tail phase's first row."""
    solution = build_mixed_solution()
    # pop the tail down to a single reduced row
    solution._pop(-1)
    solution._pop(-1)
    assert solution.phase_span(1) == (3, 4)
    solution._insert(-1, descent_row(2.5))
    # inserted before the tail's only row, still in the reduced phase
    assert solution.phase_span(1) == (3, 5)
    assert solution.raw_row(3)[0] == 2.5
    assert solution.phases[-1].dynamics.width == 6


def test_pop_across_boundary():
    solution = build_mixed_solution()
    solution._pop(-1)
    solution._pop(-1)
    solution._pop(-1)  # empties the reduced phase
    assert solution.phase_span(1) == (3, 3)
    # the emptied phase is still the one being flown, but the last row is the
    # canonical one before it
    assert solution.phase_index_at(-1) == 0
    # popping again removes the last canonical row
    row = solution._pop(-1)
    assert len(row) == 14


def test_np_array_homogeneous():
    solution = Solution()
    solution._start_phase(SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13))
    for t in range(4):
        solution._append(canonical_row(t))
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


def test_np_array_hands_back_an_array_the_caller_owns():
    """Editing the array must not rewrite what the flight reports afterwards."""
    solution = build_mixed_solution()
    array = np.array(solution)
    array[0, 1] = 999.0

    assert not np.shares_memory(array, solution.canonical_array)
    assert solution.canonical_array[0, 1] != 999.0
    assert solution[0][1] != 999.0
    # asking for a type change still copies
    assert not np.shares_memory(
        np.array(solution, dtype=np.float32), solution.canonical_array
    )


def test_the_flights_own_table_cannot_be_written_to():
    solution = build_mixed_solution()
    with pytest.raises(ValueError):
        solution.canonical_array[0, 1] = 999.0
    # and a caller that asks for it without a copy gets that same table
    assert np.shares_memory(solution.__array__(copy=False), solution.canonical_array)
    with pytest.raises(ValueError):
        solution.__array__(dtype=np.float32, copy=False)


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
    solution._start_phase(
        SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(2):
        solution._append(canonical_row(t))
    solution._start_phase(
        heading_dynamics, start_canonical=tuple([0.0] * 13), name="parafoil"
    )
    for t in range(2, 5):
        solution._append([float(t), 0, 0, 0, 0, 0, 0, float(t)])  # heading = t
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
    solution._start_phase(SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13))
    for t in range(3):
        solution._append(canonical_row(t))
    first = solution["z"]
    assert first.shape == (3, 2)
    solution._append(canonical_row(3))
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
    assert restored["vz"][:, 1].tolist() == [0.0]


def test_from_dict_finds_built_in_dynamics_by_name():
    solution = build_mixed_solution()
    restored = Solution.from_dict(solution.to_dict())
    assert restored.phases[0].dynamics is SIX_DOF_DYNAMICS


def test_from_dict_ignores_a_built_in_whose_states_changed():
    """A built-in phase saved with other states than it has now is not used."""
    data = build_mixed_solution().to_dict()
    data["phases"][1]["dynamics"] = "six_dof"
    restored = Solution.from_dict(data)
    assert restored.phases[1].dynamics is not SIX_DOF_DYNAMICS
    assert restored.phases[1].dynamics.states == DESCENT_DYNAMICS.states


def test_rebuilt_canonical_states_survive_save_and_load(monkeypatch):
    """A phase with states of its own rebuilds the canonical ones after loading."""

    def to_canonical(values):
        values["z"], values["vz"] = values["h"], values["v"]
        return [values[name] for name in CANONICAL_STATE_NAMES]

    altitude = _PhaseDynamics(
        "altitude", stub_derivative, ("h", "v"), to_canonical=to_canonical
    )
    monkeypatch.setitem(BUILT_IN_DYNAMICS, "altitude", altitude)
    solution = Solution()
    solution._start_phase(altitude, start_canonical=tuple([0.0] * 13))
    solution._append([0.0, 100.0, 50.0])
    solution._append([1.0, 150.0, 40.0])

    restored = Solution.from_dict(solution.to_dict())
    assert restored.phases[0].dynamics is altitude
    assert restored["z"][:, 1].tolist() == [100.0, 150.0]
    assert np.array_equal(restored.canonical_array, solution.canonical_array)


def test_phase_starts_that_disagree_with_the_rows_are_rejected():
    phase = _PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=0)
    later = _PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=9)
    with pytest.raises(ValueError, match="only has 2 rows"):
        Solution([phase, later], [canonical_row(0), canonical_row(1)])
    rows = [canonical_row(0), canonical_row(1), canonical_row(2)]
    out_of_order = [
        _PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=0),
        _PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=2),
        _PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=1),
    ]
    with pytest.raises(ValueError, match="order they were flown"):
        Solution(out_of_order, rows)
    not_from_the_start = [_PhaseSolution(SIX_DOF_DYNAMICS, tuple([0.0] * 13), start=1)]
    with pytest.raises(ValueError, match="must start at the first row"):
        Solution(not_from_the_start, rows)


def test_a_bound_dynamics_is_split_from_its_definition():
    """A live phase keeps both the definition and the flight-bound form."""
    bound = SIX_DOF_DYNAMICS.bind(object())
    phase = _PhaseSolution(bound, tuple([0.0] * 13))
    assert phase.dynamics is SIX_DOF_DYNAMICS
    assert phase.bound_dynamics is bound


def test_canonical_derivative_zero_fills_unintegrated_states():
    phase = _PhaseSolution(DESCENT_DYNAMICS, tuple([0.0] * 13))
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
        _PhaseSolution(DESCENT_DYNAMICS, None)
    # a phase that integrates everything needs no anchor
    assert _PhaseSolution(SIX_DOF_DYNAMICS, None).start_canonical is None


# ---------------------------------------------------------------------------
# Phase boundaries: which phase a row belongs to as rows come and go
# ---------------------------------------------------------------------------


def assert_starts_agree(solution):
    """Every phase's span must begin where the phase says it starts."""
    for index, phase in enumerate(solution.phases):
        assert solution.phase_span(index)[0] == phase.start


def test_phase_starts_follow_inserts_and_drops():
    """Adding or removing a row moves every phase that begins after it."""
    solution = build_mixed_solution()
    assert [phase.start for phase in solution.phases] == [0, 3]

    # inserting inside the first phase pushes the second one along
    solution._insert(1, canonical_row(0.5))
    assert [phase.start for phase in solution.phases] == [0, 4]
    assert_starts_agree(solution)

    # ... and removing it puts things back
    solution._pop(1)
    assert [phase.start for phase in solution.phases] == [0, 3]

    # a row inserted exactly at a phase's start joins the later phase
    solution._insert(3, descent_row(2.5))
    assert [phase.start for phase in solution.phases] == [0, 3]
    assert solution.raw_row(3)[0] == 2.5
    solution._pop(3)

    # changes at the very end never move anything
    solution._insert_before_last(descent_row(4.5))
    assert [phase.start for phase in solution.phases] == [0, 3]
    solution._drop_last()
    assert [phase.start for phase in solution.phases] == [0, 3]
    assert_starts_agree(solution)


def test_a_phase_opened_after_a_drop_still_starts_at_the_end():
    """A phase with no rows sits at the end, and moves when rows are removed."""
    solution = build_mixed_solution()
    fresh = solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="fresh")
    assert fresh.start == 6
    solution._drop_last()
    assert fresh.start == 5
    assert solution.phase_span(2) == (5, 5)
    assert_starts_agree(solution)


def test_index_resolution_skips_empty_phases():
    """Every row resolves to a phase that owns rows, empty ones passed over."""
    solution = Solution()
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="first")
    solution._append(canonical_row(0))
    # two phases that never stored a row
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="skipped")
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="also skipped")
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="last")
    solution._append(canonical_row(1))

    assert solution.phase_index_at(0) == 0
    assert solution.phase_index_at(1) == 3
    assert solution.phase_index_at(-1) == 3
    assert solution.phases[solution.phase_index_at(1)].name == "last"
    # the empty phases report themselves as holding nothing
    assert solution.phase_span(1) == (1, 1)
    assert solution.phase_span(2) == (1, 1)
    # and they are left out of a state's history rather than reported as gaps
    assert solution["vz"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Post-process values recorded beside the rows
# ---------------------------------------------------------------------------


def post_values(solution, phase_index):
    """The values recorded beside one phase's rows, ``None`` where nothing is."""
    start, stop = solution.phase_span(phase_index)
    return solution._post_values[start:stop]


def test_post_values_track_every_row_mutation():
    """Values move with their row, so the two can never drift apart."""
    solution = build_mixed_solution()
    # one entry per row from the start, empty until something is recorded
    assert post_values(solution, 0) == [None, None, None]
    assert post_values(solution, 1) == [None, None, None]

    solution._set_post_values(-1, [1.0, 2.0, 3.0])
    assert post_values(solution, 1) == [None, None, [1.0, 2.0, 3.0]]

    # overwriting the row's states makes its recorded values stale
    solution._replace_last(descent_row(5, fill=[9.0] * 6))
    assert post_values(solution, 1) == [None, None, None]

    solution._set_post_values(-1, [4.0, 5.0, 6.0])
    # a row inserted just before the last one leaves a gap for its own values,
    # and the last row keeps the values that belong to it
    solution._insert_before_last(descent_row(4.5))
    assert post_values(solution, 1) == [None, None, None, [4.0, 5.0, 6.0]]

    solution._set_post_values(-2, [7.0, 8.0, 9.0])
    assert post_values(solution, 1) == [
        None,
        None,
        [7.0, 8.0, 9.0],
        [4.0, 5.0, 6.0],
    ]
    solution._drop_last()
    assert post_values(solution, 1) == [None, None, [7.0, 8.0, 9.0]]


def test_post_values_stay_the_same_length_as_the_rows():
    solution = build_mixed_solution()
    for mutate in (
        lambda: solution._append(descent_row(6)),
        lambda: solution._insert(0, canonical_row(-1)),
        lambda: solution._pop(0),
        lambda: solution._insert_before_last(descent_row(5.5)),
        lambda: solution._drop_last(),
        lambda: solution._set_row(-1, descent_row(9)),
    ):
        mutate()
        assert len(solution._post_values) == len(solution)


def test_recording_values_does_not_disturb_the_cached_states():
    """Values are not part of the flight's states, so nothing is rebuilt."""
    solution = build_mixed_solution()
    before = solution.canonical_array
    solution._set_post_values(-1, [1.0, 2.0, 3.0])
    assert solution.canonical_array is before


def test_set_post_values_outside_the_rows_raises():
    solution = Solution()
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13))
    with pytest.raises(IndexError):
        solution._set_post_values(-1, [1.0])
    solution._append(canonical_row(0))
    with pytest.raises(IndexError):
        solution._set_post_values(1, [1.0])


def test_the_last_row_is_read_through_the_phase_that_owns_it():
    """A phase that took no steps must not be used to read the last row.

    The phase being flown and the phase that owns the last row are not always
    the same, and they can store different states. Reading the row through the
    wrong one would rebuild the canonical state from the wrong layout.
    """
    solution = build_mixed_solution()  # the phase holding the last row stores 6
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13), name="took no steps")

    assert solution.phases[-1].dynamics.width == 13
    assert solution.phases[solution.phase_index_at(-1)].dynamics.width == 6
    # the last row is a reduced one, so only the phase that owns it can read it
    canonical = solution.canonical_row(-1)[1:]
    assert len(canonical) == 13
    assert canonical[CANONICAL_INDEX["e0"]] == 2.0  # held from when it began


def test_value_at_reads_one_state_without_building_the_rest():
    solution = build_mixed_solution()
    # a state the phase integrates comes straight from the row
    assert solution.value_at(-2, "vz") == 4.0
    assert solution.value_at(0, "z") == 0.0
    # one it does not is held at its value when the phase began
    assert solution.value_at(-1, "e0") == 2.0
    # and it agrees with reading the whole state
    for index in range(len(solution)):
        assert solution.value_at(index, "vz") == solution.at_index(index)["vz"]


def test_value_at_unknown_state_raises():
    solution = build_mixed_solution()
    with pytest.raises(KeyError, match="not defined in this flight phase"):
        solution.value_at(-1, "not_a_state")


# ---------------------------------------------------------------------------
# PostProcessSolution: reading the post-process variables
# ---------------------------------------------------------------------------


def replay_derivative(flight, t, u, post_processing=False):
    """Report values that say which row they came from, so replay is visible."""
    return [t, 2 * t, 3 * t] if post_processing else list(u)


REPLAY_DYNAMICS = _PhaseDynamics(
    "replay_descent",
    replay_derivative,
    ("x", "y", "z", "vx", "vy", "vz"),
    ("ax", "ay", "az"),
)


def replay_thrust_derivative(flight, t, u, post_processing=False):
    """Like :func:`replay_derivative`, for a phase that also reports thrust."""
    return [t, 2 * t, 3 * t, 100.0] if post_processing else list(u)


REPLAY_THRUST_DYNAMICS = _PhaseDynamics(
    "replay_ascent",
    replay_thrust_derivative,
    ("x", "y", "z", "vx", "vy", "vz"),
    ("ax", "ay", "az", "net_thrust"),
)


def build_replay_solution(ascent=False):
    """A solution whose phases can work their own values out again.

    With ``ascent``, a phase that also reports thrust comes first, so the two
    phases do not compute the same variables.
    """
    solution = Solution()
    if ascent:
        solution._start_phase(
            REPLAY_THRUST_DYNAMICS.bind(None),
            t_start=0.0,
            start_canonical=tuple([0.0] * 13),
            name="ascent",
        )
        for t in range(2):
            solution._append(descent_row(t))
    solution._start_phase(
        REPLAY_DYNAMICS.bind(None),
        t_start=0.0,
        start_canonical=tuple([0.0] * 13),
        name="descent",
    )
    for t in range(3):
        solution._append(descent_row(t))
    return solution


def test_post_names_gather_every_phase_variable():
    """Phases need not compute the same variables, and all of them are listed."""
    solution = build_mixed_solution()
    assert solution.post.names == FULL_POST_PROCESS_VARS
    assert "net_thrust" in solution.post
    # the descent computes only three of them
    assert solution.phases[1].dynamics.post_process_vars == ("ax", "ay", "az")


def test_values_recorded_during_the_flight_are_read_back_by_name():
    solution = build_replay_solution()
    solution.records_post_values = True
    for index, values in enumerate([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]):
        solution._set_post_values(index, values)
    assert solution.post["ay"].tolist() == [[0.0, 8.0], [1.0, 2.0], [2.0, 5.0]]


def test_a_recording_flight_is_never_worked_out_again():
    """Replaying would read the rocket as it ended, so it must not happen."""
    solution = build_replay_solution()
    solution.records_post_values = True
    for index in range(len(solution)):
        solution._set_post_values(index, [99.0, 99.0, 99.0])
    # the stub dynamics would give 0, 1, 2 if it were replayed
    assert solution.post["ax"][:, 1].tolist() == [99.0, 99.0, 99.0]


def test_a_gap_in_a_recording_flight_is_an_error():
    """A row with nothing recorded is a bug, not a reason to replay."""
    solution = build_replay_solution()
    solution.records_post_values = True
    solution._set_post_values(0, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="row 1 has none"):
        solution.post["ax"]
    with pytest.raises(ValueError, match="row 1 has none"):
        solution.post.at_index(1)


def test_values_not_recorded_are_worked_out_from_the_stored_states():
    """A flight that recorded nothing replays its own states instead."""
    solution = build_replay_solution()
    assert solution.post["ax"].tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert solution.post["az"].tolist() == [[0.0, 0.0], [1.0, 3.0], [2.0, 6.0]]


def test_a_partly_recorded_phase_is_worked_out_again():
    """A flight that does not record is replayed, whatever is left lying in it."""
    solution = build_replay_solution()
    solution._set_post_values(1, [99.0, 99.0, 99.0])
    assert solution.post["ax"].tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_a_phase_that_does_not_compute_a_variable_reports_zero():
    """A parachute descent really has no thrust, so zero is the right answer."""
    solution = build_replay_solution(ascent=True)
    assert solution.post.names == ("ax", "ay", "az", "net_thrust")
    # the ascent is the first two rows, the descent the last three
    assert solution.post["net_thrust"].tolist() == [
        [0.0, 100.0],
        [1.0, 100.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ]


def test_the_replay_is_kept_until_something_changes():
    """The replayed tables are cached; a history is a cheap slice of them."""
    solution = build_replay_solution()
    first = solution.post["ax"]
    tables = solution.post._phase_tables()
    assert solution.post._phase_tables() is tables
    assert solution.post["ax"].tolist() == first.tolist()
    # recording a value replaces what was worked out for that phase
    solution.records_post_values = True
    for index, values in enumerate([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]):
        solution._set_post_values(index, values)
    assert solution.post._phase_tables() is not tables
    assert solution.post["ax"].tolist() == [[0.0, 7.0], [1.0, 1.0], [2.0, 4.0]]
    solution.records_post_values = False
    # so does storing a new row
    solution._append(descent_row(3))
    assert solution.post["ax"].shape == (4, 2)


def test_reading_one_row_does_not_work_out_the_whole_flight():
    solution = build_replay_solution()
    solution._set_post_values(-1, [4.0, 5.0, 6.0])
    assert solution.post.at_index(-1) == {"ax": 4.0, "ay": 5.0, "az": 6.0}
    # a row with nothing recorded is worked out on its own
    assert solution.post.at_index(0) == {"ax": 0.0, "ay": 0.0, "az": 0.0}
    assert solution.post.at(1.0) == {"ax": 1.0, "ay": 2.0, "az": 3.0}


def test_reading_a_row_outside_the_flight_raises():
    solution = build_replay_solution()
    with pytest.raises(IndexError):
        solution.post.at_index(3)


def test_phase_values_gives_one_phase_at_a_time():
    solution = build_replay_solution()
    assert solution.post.phase_values(0).tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    ]


def test_an_unknown_variable_says_what_the_flight_computes():
    solution = build_replay_solution()
    with pytest.raises(KeyError, match="No flight phase computed"):
        solution.post["not_a_variable"]


def test_a_solution_read_back_from_a_file_cannot_report_them():
    """Working the values out again needs equations of motion, which are code."""
    solution = Solution()
    solution._start_phase(
        REPLAY_DYNAMICS, t_start=0.0, start_canonical=tuple([0.0] * 13)
    )
    solution._append(descent_row(0))
    with pytest.raises(KeyError, match="read back from a saved file"):
        solution.post["ax"]
    with pytest.raises(KeyError, match="read back from a saved file"):
        solution.post.at_index(0)


def test_a_flight_with_no_rows_has_nothing_to_report():
    solution = Solution()
    solution._start_phase(REPLAY_DYNAMICS.bind(None), start_canonical=tuple([0.0] * 13))
    assert solution.post.names == ("ax", "ay", "az")
    with pytest.raises(KeyError, match="no stored states yet"):
        _ = solution.post["ax"]


def test_canonical_row_fills_in_what_the_phase_does_not_integrate():
    """The reduced descent supplies six states; the other seven are held."""
    solution = build_mixed_solution()
    assert len(solution.raw_row(-1)[1:]) == 6
    canonical = solution.canonical_row(-1)[1:]
    assert len(canonical) == 13
    # position and velocity come from the row itself
    assert list(canonical[:6]) == [5.0] * 6
    # the rest are held at the value they had when the descent began
    assert list(canonical[6:]) == [2.0] * 7


def test_canonical_row_skips_a_phase_that_stored_no_rows():
    """A phase can begin and end without a row, and must not be read from."""
    solution = build_mixed_solution()
    solution._start_phase(
        SIX_DOF_DYNAMICS, t_start=5.0, start_canonical=tuple([0.0] * 13), name="empty"
    )
    # the newest phase is canonical and the last row is a 6-value descent row,
    # so reading through it would report the row's states as canonical ones
    assert solution.phases[-1].name == "empty"
    assert len(solution.canonical_row(-1)[1:]) == 13
    assert list(solution.canonical_row(-1)[1:][:6]) == [5.0] * 6


def test_canonical_row_without_rows_raises():
    solution = Solution()
    solution._start_phase(SIX_DOF_DYNAMICS, tuple([0.0] * 13))
    with pytest.raises(IndexError):
        _ = solution.canonical_row(-1)[1:]


# ---------------------------------------------------------------------------
# Phases narrower and wider than the canonical state, in one flight
# ---------------------------------------------------------------------------
#
# A phase may integrate fewer than the thirteen canonical states, and it may add
# states of its own on top of them. Both are load-bearing: the first is what lets
# a descent skip an attitude it does not model, the second is what a parafoil
# heading or an air brake position would use. The tests below run a flight that
# does both at once, so the pieces are exercised against each other rather than
# one at a time.

PARAFOIL_STATES = ("x", "y", "z", "vx", "vy", "vz", "heading")

PARAFOIL_DYNAMICS = _PhaseDynamics("parafoil", stub_derivative, PARAFOIL_STATES)

TURNING_PARAFOIL_DYNAMICS = _PhaseDynamics(
    "turning_parafoil",
    stub_derivative,
    PARAFOIL_STATES,
    to_canonical=lambda values: [
        values["heading"] * 2 if name == "e0" else values[name]
        for name in CANONICAL_STATE_NAMES
    ],
)


def parafoil_row(t, heading=None):
    """Build an 8-value row ``[t, x, y, z, vx, vy, vz, heading]``."""
    return [float(t), *[float(t)] * 6, float(t) if heading is None else float(heading)]


def build_three_width_solution(parafoil_dynamics=PARAFOIL_DYNAMICS):
    """A flight of three phases: canonical, then narrower, then wider.

    Rows 0-1 are the canonical ascent, rows 2-3 the six-state descent, rows 4-5
    the seven-state parafoil.
    """
    solution = Solution()
    solution._start_phase(
        SIX_DOF_DYNAMICS, t_start=0.0, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(2):
        solution._append(canonical_row(t))
    solution._start_phase(
        DESCENT_DYNAMICS,
        t_start=2.0,
        start_canonical=tuple(solution.canonical_row(-1)[1:]),
        name="descent",
    )
    for t in range(2, 4):
        solution._append(descent_row(t))
    solution._start_phase(
        parafoil_dynamics,
        t_start=4.0,
        start_canonical=tuple(solution.canonical_row(-1)[1:]),
        name="parafoil",
    )
    for t in range(4, 6):
        solution._append(parafoil_row(t))
    return solution


def test_each_phase_stores_rows_at_its_own_width():
    """The row list holds three different widths and still reads as one flight."""
    solution = build_three_width_solution()
    assert [len(solution.raw_row(index)) for index in range(len(solution))] == [
        14,  # canonical ascent: time plus 13 states
        14,
        7,  # narrower descent: time plus 6 states
        7,
        8,  # wider parafoil: time plus 6 states and a heading
        8,
    ]


def test_a_flight_of_mixed_widths_still_reads_as_canonical_rows():
    """Whatever a phase stored, the flight reads as 14-value canonical rows."""
    solution = build_three_width_solution()
    assert all(len(row) == 14 for row in solution)
    assert np.array(solution).shape == (6, 14)
    assert solution.canonical_array.shape == (6, 14)
    assert [len(solution[index]) for index in range(-6, 0)] == [14] * 6


def test_a_canonical_state_spans_every_width(recwarn):
    """Every phase can report a canonical state, so its history has no gap."""
    solution = build_three_width_solution()
    vz = solution["vz"]
    assert vz[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert vz[:, 1].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(recwarn) == 0


def test_a_state_no_narrow_phase_integrates_is_held_at_its_last_value():
    """w3 stops being integrated at t=2 and holds the value it had then."""
    solution = build_three_width_solution()
    w3 = solution["w3"]
    assert w3[:, 1].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_an_extra_state_reads_by_name_over_the_phases_that_have_it():
    """The heading exists only in the parafoil, and the history says so."""
    solution = build_three_width_solution()
    with pytest.warns(UserWarning, match="not defined during ascent, descent"):
        heading = solution["heading"]
    assert heading[:, 0].tolist() == [4.0, 5.0]
    assert heading[:, 1].tolist() == [4.0, 5.0]


def test_an_extra_state_is_not_part_of_the_canonical_row():
    """A canonical row is always the same 13 states, whatever the phase added."""
    solution = build_three_width_solution()
    assert len(solution.raw_row(-1)) == 8
    assert len(solution.canonical_row(-1)) == 14


def test_a_whole_state_reads_with_the_extra_one_only_where_it_exists():
    solution = build_three_width_solution()
    parafoil_state = solution.at_index(-1)
    assert parafoil_state["heading"] == 5.0
    assert len(parafoil_state) == 14  # the 13 canonical states plus the heading
    ascent_state = solution.at_index(0)
    assert "heading" not in ascent_state
    assert len(ascent_state) == 13
    descent_state = solution.at_index(2)
    assert "heading" not in descent_state
    assert len(descent_state) == 13


def test_reading_one_value_works_for_every_width():
    solution = build_three_width_solution()
    assert solution.value_at(-1, "heading") == 5.0
    assert solution.value_at(-1, "vz") == 5.0
    # held at its value when the narrower descent began
    assert solution.value_at(-1, "w3") == 1.0
    # the heading does not exist in the phases before the parafoil
    with pytest.raises(KeyError, match="not defined in this flight phase"):
        solution.value_at(0, "heading")
    with pytest.raises(KeyError, match="not defined in this flight phase"):
        solution.value_at(2, "heading")


def test_one_phase_reads_on_its_own_at_every_width():
    solution = build_three_width_solution()
    assert solution.phase_span(1) == (2, 4)
    assert solution.phase_span(2) == (4, 6)
    assert [solution.value_at(i, "heading") for i in (4, 5)] == [4.0, 5.0]
    assert solution["vz"][2:4, 1].tolist() == [2.0, 3.0]
    assert solution.canonical_array[2:4].shape == (2, 14)
    assert solution.canonical_array[4:6].shape == (2, 14)


def test_a_loaded_row_of_the_wrong_width_is_rejected():
    """Rows written by the simulation always fit; a file's rows are checked."""
    data = build_three_width_solution().to_dict()
    data["rows"][-1] = canonical_row(5)  # 14 values in the 8-value parafoil
    with pytest.raises(ValueError, match="stored in flight phase 'parafoil'"):
        Solution.from_dict(data)


def test_an_extra_state_can_rebuild_a_canonical_one():
    """A parafoil that turns rebuilds e0 from its heading rather than freezing it."""
    solution = build_three_width_solution(TURNING_PARAFOIL_DYNAMICS)
    # e0 is 2 * heading during the parafoil, and frozen at 1.0 before it
    assert solution["e0"][:, 1].tolist() == [0.0, 1.0, 1.0, 1.0, 8.0, 10.0]
    assert solution.value_at(-1, "e0") == 10.0
    assert solution.at_index(-1)["e0"] == 10.0


def test_saving_a_flight_keeps_every_phase_width():
    """Both the narrower and the wider phase survive a round trip."""
    solution = build_three_width_solution()
    restored = Solution.from_dict(solution.to_dict())

    assert [phase.dynamics.states for phase in restored.phases] == [
        CANONICAL_STATE_NAMES,
        ("x", "y", "z", "vx", "vy", "vz"),
        PARAFOIL_STATES,
    ]
    assert [len(restored.raw_row(index)) for index in range(len(restored))] == [
        14,
        14,
        7,
        7,
        8,
        8,
    ]
    assert restored.canonical_array.tolist() == solution.canonical_array.tolist()
    with pytest.warns(UserWarning, match="not defined during"):
        assert restored["heading"][:, 1].tolist() == [4.0, 5.0]
