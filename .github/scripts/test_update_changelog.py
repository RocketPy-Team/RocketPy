#!/usr/bin/env python3
"""Self-contained tests for update_changelog.py.

Runs without network or the ``google-genai`` package (the LLM import in the
script is lazy). Exercises the deterministic pieces that keep the changelog
safe: section splitting, duplicate detection, prefix handling, the fallback
classifier, and the validator that guards LLM output.

Run locally with either:
    python .github/scripts/test_update_changelog.py
    pytest .github/scripts/test_update_changelog.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import update_changelog as uc  # noqa: E402

SAMPLE = """\
# RocketPy Change Log

## [Unreleased] - yyyy-mm-dd

### Added

### Changed

### Fixed

## [v1.13.0] - 2026-07-21

### Added

- ENH: something old [#100](https://github.com/RocketPy-Team/RocketPy/pull/100)
"""


def test_split_roundtrips():
    before, block, after = uc.split_changelog(SAMPLE)
    assert before + block + after == SAMPLE
    assert block.startswith("## [Unreleased]")
    assert "## [v1.13.0]" not in block
    assert "## [v1.13.0]" in after


def test_already_present():
    _, block, after = uc.split_changelog(SAMPLE)
    assert uc.already_present(after, 100) is True
    assert uc.already_present(block, 100) is False
    assert uc.already_present(block, 999) is False


def test_detect_prefix():
    assert uc.detect_prefix("BUG: fix a thing") == "BUG"
    assert uc.detect_prefix("BUG/MNT: pre-release review fixes") == "BUG/MNT"
    assert uc.detect_prefix("Add a shiny feature") is None


def test_no_double_prefix():
    # The historical bug: a title already carrying a prefix must not gain another.
    entry = uc.build_entry("BUG/MNT: pre-release fixes", 1074, "ENH", "BUG/MNT")
    assert entry.startswith("- BUG/MNT: pre-release fixes ")
    assert "ENH:" not in entry
    assert "[#1074](https://github.com/RocketPy-Team/RocketPy/pull/1074)" in entry


def test_build_entry_adds_prefix_when_missing():
    entry = uc.build_entry("Add a shiny feature", 200, "ENH", None)
    assert entry.startswith("- ENH: Add a shiny feature ")


def test_fallback_routing():
    # Bug label -> Fixed
    section, prefix, _ = uc.fallback_section_and_prefix("Fix crash", "Bug,Flight")
    assert section == "### Fixed" and prefix == "BUG"
    # Refactor label -> Changed
    section, _, _ = uc.fallback_section_and_prefix("Tidy internals", "Refactor")
    assert section == "### Changed"
    # Existing prefix wins the routing even without a matching label
    section, prefix, existing = uc.fallback_section_and_prefix("BUG: oops", "")
    assert section == "### Fixed" and existing == "BUG"
    # Default
    section, prefix, _ = uc.fallback_section_and_prefix("New capability", "Enhancement")
    assert section == "### Added" and prefix == "ENH"


def test_fallback_update_inserts_once_under_right_section():
    _, block, _ = uc.split_changelog(SAMPLE)
    updated = uc.fallback_update(block, "BUG: fix a thing", 321, "Bug")
    assert updated.count("[#321]") == 1
    fixed = updated.split("### Fixed", 1)[1]
    assert "- BUG: fix a thing" in fixed
    # Not misplaced under Added.
    added = updated.split("### Added", 1)[1].split("### Changed", 1)[0]
    assert "[#321]" not in added


def test_ensure_section_inserts_in_canonical_order():
    _, block, _ = uc.split_changelog(SAMPLE)
    # SAMPLE has Added/Changed/Fixed but no Removed subsection.
    lines = uc.ensure_section(block.splitlines(keepends=True), "### Removed")
    joined = "".join(lines)
    # Removed must sit after Changed and before Fixed.
    assert (
        joined.index("### Changed")
        < joined.index("### Removed")
        < joined.index("### Fixed")
    )


def test_validator_accepts_good_output():
    _, block, _ = uc.split_changelog(SAMPLE)
    good = block.replace(
        "### Fixed\n",
        "### Fixed\n\n- BUG: fix it [#500](https://github.com/RocketPy-Team/RocketPy/pull/500)\n",
    )
    assert uc.validate_llm_block(block, good, 500) is None


def test_validator_rejects_dropped_entry():
    old = block_with_entry()
    # New block loses the pre-existing entry.
    new = "## [Unreleased] - yyyy-mm-dd\n\n### Fixed\n\n- BUG: new [#500](https://github.com/RocketPy-Team/RocketPy/pull/500)\n\n"
    assert uc.validate_llm_block(old, new, 500) is not None


def test_validator_rejects_missing_new_ref():
    _, block, _ = uc.split_changelog(SAMPLE)
    assert uc.validate_llm_block(block, block, 500) is not None  # no new ref at all


def test_validator_rejects_leaked_version_header():
    _, block, _ = uc.split_changelog(SAMPLE)
    leaked = (
        "## [Unreleased] - yyyy-mm-dd\n\n### Fixed\n\n"
        "- BUG: x [#500](https://github.com/RocketPy-Team/RocketPy/pull/500)\n\n"
        "## [v1.13.0] - 2026-07-21\n"
    )
    assert uc.validate_llm_block(block, leaked, 500) is not None


def test_validator_rejects_runaway_growth():
    _, block, _ = uc.split_changelog(SAMPLE)
    huge = (
        "## [Unreleased] - yyyy-mm-dd\n\n### Fixed\n\n"
        "- BUG: x [#500](https://github.com/RocketPy-Team/RocketPy/pull/500)\n"
        + ("x" * (uc.MAX_BLOCK_GROWTH + 50))
    )
    assert uc.validate_llm_block(block, huge, 500) is not None


def block_with_entry():
    return (
        "## [Unreleased] - yyyy-mm-dd\n\n### Added\n\n"
        "- ENH: keep me [#200](https://github.com/RocketPy-Team/RocketPy/pull/200)\n\n"
        "### Fixed\n\n"
    )


def _run_all():
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok   {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
