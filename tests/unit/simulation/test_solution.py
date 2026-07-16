"""Tests for the flight Solution container and its state schemas.

These tests build solutions by hand and never run a Flight, so they are fast
and safe to run locally.
"""

import numpy as np
import pytest

from rocketpy.simulation.solution import (
    CANONICAL_INDEX,
    CANONICAL_SCHEMA,
    CANONICAL_STATE_NAMES,
    PARACHUTE_3T_SCHEMA,
    Solution,
    SolutionSegment,
    StateSchema,
    StateView,
)

CANONICAL_ATOL = 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3]


def canonical_row(t, fill=None):
    """Build a 14-value canonical row ``[t, *state]``."""
    state = [float(t)] * 13 if fill is None else list(fill)
    return [float(t), *state]


def parachute_row(t, fill=None):
    """Build a 7-value parachute row ``[t, x, y, z, vx, vy, vz]``."""
    state = [float(t)] * 6 if fill is None else list(fill)
    return [float(t), *state]


def build_mixed_solution():
    """A solution with a canonical segment followed by a parachute segment."""
    solution = Solution()
    solution.start_segment(
        CANONICAL_SCHEMA, t_start=0.0, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(3):
        solution.append(canonical_row(t))
    frozen = solution.canonical_row(-1)[1:]
    solution.start_segment(
        PARACHUTE_3T_SCHEMA,
        t_start=2.0,
        start_canonical=tuple(frozen),
        name="descent",
    )
    for t in range(3, 6):
        solution.append(parachute_row(t))
    return solution


# ---------------------------------------------------------------------------
# StateSchema
# ---------------------------------------------------------------------------


def test_schema_index_and_width():
    assert CANONICAL_SCHEMA.width == 13
    assert CANONICAL_SCHEMA.is_canonical
    assert PARACHUTE_3T_SCHEMA.width == 6
    assert not PARACHUTE_3T_SCHEMA.is_canonical
    assert CANONICAL_SCHEMA.index_of("vz") == 5
    assert PARACHUTE_3T_SCHEMA.index_of("vz") == 5


def test_schema_index_of_unknown_raises():
    with pytest.raises(KeyError):
        PARACHUTE_3T_SCHEMA.index_of("e0")


def test_canonicalize_identity_for_canonical_schema():
    state = list(range(13))
    result = CANONICAL_SCHEMA.canonicalize(state, None)
    # Canonical schema returns the same object, no copy.
    assert result is state


def test_canonicalize_freeze_fallback():
    frozen = [0.0] * 13
    frozen[CANONICAL_INDEX["e0"]] = 0.7
    frozen[CANONICAL_INDEX["w1"]] = 1.3
    state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # x, y, z, vx, vy, vz
    result = PARACHUTE_3T_SCHEMA.canonicalize(state, frozen)
    assert len(result) == 13
    assert result[:6] == state  # translational states preserved
    assert result[CANONICAL_INDEX["e0"]] == 0.7  # frozen attitude
    assert result[CANONICAL_INDEX["w1"]] == 1.3


def test_canonicalize_reconstruction_hook():
    # A schema that integrates a "heading" and rebuilds e0 from it.
    schema = StateSchema(
        ("x", "y", "z", "vx", "vy", "vz", "heading"),
        reconstructors={"e0": lambda view: np.cos(view["heading"] / 2)},
    )
    frozen = [0.0] * 13
    state = [0, 0, 0, 0, 0, 0, np.pi]  # heading = pi
    result = schema.canonicalize(state, frozen)
    assert result[CANONICAL_INDEX["e0"]] == pytest.approx(np.cos(np.pi / 2))


def test_canonicalize_derivative_zero_fills_missing():
    state_dot = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]  # vx..az of parachute
    result = PARACHUTE_3T_SCHEMA.canonicalize_derivative(state_dot)
    assert result[:6] == state_dot
    assert result[6:] == [0.0] * 7  # attitude/rate derivatives are zero


def test_subset_from_canonical():
    canonical = list(range(13))
    subset = PARACHUTE_3T_SCHEMA.subset_from_canonical(canonical)
    assert subset == [0, 1, 2, 3, 4, 5]
    assert CANONICAL_SCHEMA.subset_from_canonical(canonical) == canonical


def test_select_atol_scalar_passthrough():
    assert PARACHUTE_3T_SCHEMA.select_atol(1e-4) == 1e-4


def test_select_atol_canonical_vector_reduced():
    result = PARACHUTE_3T_SCHEMA.select_atol(CANONICAL_ATOL)
    assert result == [1e-3] * 6
    assert CANONICAL_SCHEMA.select_atol(CANONICAL_ATOL) == CANONICAL_ATOL


def test_select_atol_bad_length_raises():
    with pytest.raises(ValueError):
        PARACHUTE_3T_SCHEMA.select_atol([1e-3, 1e-3])


def test_select_atol_matching_width_passthrough():
    custom = [1, 2, 3, 4, 5, 6]
    assert PARACHUTE_3T_SCHEMA.select_atol(custom) == custom


def test_schema_from_names_recovers_registered():
    assert StateSchema.from_names(CANONICAL_STATE_NAMES) is CANONICAL_SCHEMA
    assert (
        StateSchema.from_names(("x", "y", "z", "vx", "vy", "vz")) is PARACHUTE_3T_SCHEMA
    )
    fresh = StateSchema.from_names(("a", "b"))
    assert fresh.names == ("a", "b")


# ---------------------------------------------------------------------------
# StateView
# ---------------------------------------------------------------------------


def test_state_view_name_and_attribute_access():
    view = StateView([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], PARACHUTE_3T_SCHEMA)
    assert view["z"] == 3.0
    assert view.vz == 6.0
    assert view[0] == 1.0  # integer indexing returns raw


def test_state_view_frozen_canonical_name():
    frozen = [0.0] * 13
    frozen[CANONICAL_INDEX["e0"]] = 0.9
    view = StateView([0, 0, 0, 0, 0, 0], PARACHUTE_3T_SCHEMA, frozen=frozen)
    assert view["e0"] == 0.9


def test_state_view_missing_name_raises():
    view = StateView([0, 0, 0, 0, 0, 0], PARACHUTE_3T_SCHEMA)
    with pytest.raises(KeyError):
        _ = view["e0"]  # no frozen anchor supplied
    with pytest.raises(AttributeError):
        _ = view.not_a_state


def test_state_view_canonical_cached():
    frozen = [7.0] * 13
    view = StateView([1, 2, 3, 4, 5, 6], PARACHUTE_3T_SCHEMA, frozen=frozen)
    first = view.canonical
    assert view.canonical is first  # cached
    assert first[:6] == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Solution list-like behaviour
# ---------------------------------------------------------------------------


def test_len_iter_and_indexing():
    solution = build_mixed_solution()
    assert len(solution) == 6
    rows = list(solution)
    assert len(rows) == 6
    assert len(solution[-1]) == 7  # parachute row
    assert len(solution[0]) == 14  # canonical row
    assert solution[-1][0] == 5.0
    assert solution[2][0] == 2.0
    assert solution[3][0] == 3.0  # first parachute row


def test_negative_index_across_boundary():
    solution = build_mixed_solution()
    # solution[-4] is the last canonical row (index 2), 14 wide
    assert len(solution[-4]) == 14
    assert solution[-4][0] == 2.0
    assert len(solution[-3]) == 7  # first parachute row


def test_slice_returns_raw_rows():
    solution = build_mixed_solution()
    sliced = solution[1:4]
    assert [row[0] for row in sliced] == [1.0, 2.0, 3.0]


def test_append_wrong_width_raises():
    solution = build_mixed_solution()
    with pytest.raises(ValueError):
        solution.append([6.0] * 14)  # canonical width into parachute tail


def test_iadd_appends_rows():
    solution = Solution()
    solution.start_segment(CANONICAL_SCHEMA, start_canonical=tuple([0.0] * 13))
    solution += [canonical_row(0), canonical_row(1)]
    assert len(solution) == 2
    assert solution[-1][0] == 1.0


def test_setitem_replaces_tail_row():
    solution = build_mixed_solution()
    solution[-1] = parachute_row(9)
    assert solution[-1][0] == 9.0


def test_setitem_wrong_width_raises():
    solution = build_mixed_solution()
    with pytest.raises(ValueError):
        solution[-1] = canonical_row(9)


def test_insert_before_tail_first_row():
    """Exact-time insert whose time precedes the tail segment's first row."""
    solution = build_mixed_solution()
    # pop the tail down to a single parachute row
    solution.pop(-1)
    solution.pop(-1)
    assert len(solution.tail.rows) == 1
    solution.insert(-1, parachute_row(2.5))
    # inserted before the tail's only row, still in the parachute segment
    assert solution.tail.rows[0][0] == 2.5
    assert solution.tail.width == 6


def test_pop_across_boundary():
    solution = build_mixed_solution()
    solution.pop(-1)
    solution.pop(-1)
    solution.pop(-1)  # empties parachute segment
    assert len(solution.tail.rows) == 0
    # popping again removes the last canonical row
    row = solution.pop(-1)
    assert len(row) == 14


def test_np_array_homogeneous():
    solution = Solution()
    solution.start_segment(CANONICAL_SCHEMA, start_canonical=tuple([0.0] * 13))
    for t in range(4):
        solution.append(canonical_row(t))
    array = np.array(solution)
    assert array.shape == (4, 14)
    assert array[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_np_array_mixed_width_raises():
    solution = build_mixed_solution()
    with pytest.raises(TypeError):
        np.array(solution)


# ---------------------------------------------------------------------------
# Named queries
# ---------------------------------------------------------------------------


def test_canonical_series_full_timeline_with_freeze():
    solution = build_mixed_solution()
    # vz is defined in every phase (translational)
    vz = solution["vz"]
    assert vz.shape == (6, 2)
    assert vz[:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    # e0 is frozen during the parachute descent at its value at t=2
    e0 = solution["e0"]
    assert e0[:, 1].tolist() == [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_series_partial_covers_only_defining_segments():
    # A schema with a non-canonical variable defined only in one segment.
    heading_schema = StateSchema(("x", "y", "z", "vx", "vy", "vz", "heading"))
    solution = Solution()
    solution.start_segment(
        CANONICAL_SCHEMA, start_canonical=tuple([0.0] * 13), name="ascent"
    )
    for t in range(2):
        solution.append(canonical_row(t))
    solution.start_segment(
        heading_schema, start_canonical=tuple([0.0] * 13), name="parafoil"
    )
    for t in range(2, 5):
        solution.append([float(t), 0, 0, 0, 0, 0, 0, float(t)])  # heading = t
    heading = solution["heading"]
    # only the parafoil segment defines heading
    assert heading[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert heading[:, 1].tolist() == [2.0, 3.0, 4.0]


def test_series_unknown_name_raises():
    solution = build_mixed_solution()
    with pytest.raises(KeyError):
        _ = solution["not_a_state"]


def test_series_cache_invalidated_on_append():
    solution = Solution()
    solution.start_segment(CANONICAL_SCHEMA, start_canonical=tuple([0.0] * 13))
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
    row = solution.canonical_row(-2)  # a parachute row
    assert len(row) == 14
    assert row[CANONICAL_INDEX["e0"] + 1] == 2.0  # frozen attitude


def test_canonical_states_stop():
    solution = build_mixed_solution()
    states = solution.canonical_states(stop=-1)
    assert len(states) == 5
    assert all(len(state) == 13 for state in states)


def test_at_returns_state_view_and_warns():
    solution = build_mixed_solution()
    view = solution.at(3.0)
    assert view["z"] == 3.0
    with pytest.warns(UserWarning):
        solution.at(3.4, atol=1e-3)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip():
    solution = build_mixed_solution()
    data = solution.to_dict()
    assert data["format"] == "rocketpy/solution"
    assert len(data["segments"]) == 2
    restored = Solution.from_dict(data)
    assert len(restored) == len(solution)
    assert np.allclose(restored.canonical_array, solution.canonical_array)
    # reduced segment schema recovered from the registry
    assert restored.segments[1].schema is PARACHUTE_3T_SCHEMA


def test_from_legacy_list():
    rows = [canonical_row(0), canonical_row(1), canonical_row(2)]
    solution = Solution.from_legacy_list(rows)
    assert len(solution) == 3
    assert solution.segments[0].schema is CANONICAL_SCHEMA
    assert solution["x"][:, 0].tolist() == [0.0, 1.0, 2.0]


def test_segment_from_dict_unknown_dynamics_key_tolerated():
    segment = SolutionSegment.from_dict(
        {
            "name": "custom",
            "dynamics": "not_a_registered_key",
            "state_names": ["x", "y", "z", "vx", "vy", "vz"],
            "t_start": 0.0,
            "start_canonical": [0.0] * 13,
            "rows": [parachute_row(0)],
        }
    )
    assert segment.schema is PARACHUTE_3T_SCHEMA
    assert len(segment.rows) == 1
