#!/usr/bin/env python3
"""Claude Code ``PreToolUse`` hook: block ``git push`` if ruff would fail in CI.

Runs before every ``Bash`` command. If the command is a ``git push``, it runs the
two fast ruff steps from the Linters CI job (``ruff check .`` and
``ruff format --check .``) over the repo. If either would fail, the push is
blocked (exit code 2) and the failure is reported back to Claude so it gets fixed
*before* it turns the CI red.

Only ruff runs here (both steps together are ~0.15s). Pylint is intentionally left
to CI: on this repo pylint costs seconds-per-file of startup, which is too slow for
an interactive guard. ruff is what has actually been breaking the Linters job.

Design notes
------------
* ruff is invoked as ``<current python> -m ruff`` (see ``ruff_on_edit.py``).
* Cross-platform: pure Python, no shell built-ins, no OS-specific paths.
* Silent no-op when the command is not a push, or when ruff is not installed.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys

# Match ``git push`` as a real command (start of line or after a shell
# separator), tolerating global flags like ``git -C . push``. Avoids most
# false positives from the substring "push" appearing elsewhere.
_GIT_PUSH = re.compile(r"(?:^|[\s;&|(`])git(?:\s+-\S+|\s+--\S+)*\s+push\b")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(command, str) or not _GIT_PUSH.search(command):
        return 0

    if importlib.util.find_spec("ruff") is None:
        return 0  # ruff not installed: don't block pushes.

    # Target the project root explicitly. ``CLAUDE_PROJECT_DIR`` is set by Claude
    # Code to the native-format project root; passing it as a path argument (not
    # as the subprocess ``cwd``) keeps this robust across platforms/shells. Falls
    # back to ".", which resolves against the hook's working directory.
    target = os.environ.get("CLAUDE_PROJECT_DIR") or "."
    ruff = [sys.executable, "-m", "ruff"]
    check = subprocess.run(ruff + ["check", target], capture_output=True, text=True)
    fmt = subprocess.run(
        ruff + ["format", "--check", target], capture_output=True, text=True
    )
    if check.returncode == 0 and fmt.returncode == 0:
        return 0

    out = [
        "[ruff] Push blocked — this would fail the Linters CI job. "
        "Fix the issues below, then push again.\n"
    ]
    if check.returncode != 0:
        out.append("--- ruff check ---\n" + (check.stdout or "") + (check.stderr or ""))
    if fmt.returncode != 0:
        out.append(
            "--- ruff format --check . ---\n"
            + (fmt.stdout or "")
            + (fmt.stderr or "")
            + "\nFix formatting with:  python -m ruff format .\n"
        )
    sys.stderr.write("\n".join(out))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
