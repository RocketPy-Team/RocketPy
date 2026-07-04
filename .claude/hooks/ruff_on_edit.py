#!/usr/bin/env python3
"""Claude Code ``PostToolUse`` hook: auto-format + lint-fix edited Python files.

Runs after every ``Edit``/``Write``/``MultiEdit``. Reads the hook payload from
stdin, and if the touched file is a ``.py`` file, runs ``ruff format`` followed by
``ruff check --fix`` on *just that one file*. ruff is ~instant per file, so this
keeps Claude's edits continuously formatted and lint-clean without slowing the
session down.

Design notes
------------
* ruff is invoked as ``<current python> -m ruff`` so it always resolves to the
  same environment the hook runs in (matches the ``python -m ruff`` convention
  the repo already uses) instead of relying on a ``ruff`` binary being on PATH.
* Cross-platform: pure Python, no shell built-ins, no OS-specific paths.
* Degrades to a silent no-op when ruff is not installed, so it can never block
  editing for a contributor who has not installed the linter yet.
* Exit code 2 feeds any *unfixable* lint issues back to Claude so it fixes them;
  the common case (format + autofix succeed) exits 0 silently.
"""

import importlib.util
import json
import subprocess
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return 0

    if importlib.util.find_spec("ruff") is None:
        # ruff not installed in this environment: never block the workflow.
        return 0

    ruff = [sys.executable, "-m", "ruff"]
    subprocess.run(ruff + ["format", file_path], capture_output=True, text=True)
    fix = subprocess.run(
        ruff + ["check", "--fix", file_path], capture_output=True, text=True
    )

    if fix.returncode != 0:
        sys.stderr.write(
            f"[ruff] Lint issues ruff could not auto-fix in {file_path}:\n"
            f"{fix.stdout}{fix.stderr}"
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
