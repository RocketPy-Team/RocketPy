# Claude Code hooks

Project-level [Claude Code hooks](https://docs.claude.com/en/docs/claude-code/hooks)
that keep Claude's edits lint-clean and stop lint failures from reaching CI. They
are wired up in [`.claude/settings.json`](../settings.json) and run automatically
for anyone using Claude Code in this repo.

Both hooks only run **ruff** (the fast linter/formatter, ~0.15s for the whole
repo). Pylint is intentionally left to CI: on this codebase its per-file startup
cost is seconds, too slow for an interactive guard. ruff is also what has actually
been breaking the `Linters` job.

## Hooks

| Hook | Event | What it does |
| --- | --- | --- |
| [`ruff_on_edit.py`](ruff_on_edit.py) | `PostToolUse` on `Edit`/`Write`/`MultiEdit` | After Claude edits a `.py` file, runs `ruff format` then `ruff check --fix` on that one file. Surfaces any unfixable lint issues back to Claude. |
| [`ruff_guard_push.py`](ruff_guard_push.py) | `PreToolUse` on `Bash` | Before a `git push`, runs `ruff check` + `ruff format --check` over the repo and **blocks the push** if either would fail (i.e. before it turns CI red). |

## Design guarantees

- **Cross-platform** (Windows / macOS / Linux): pure Python, no shell built-ins,
  no OS-specific paths. ruff is invoked as `<current python> -m ruff` so it
  resolves to the same environment the hook runs in.
- **Never gets in the way**: if `ruff` is not installed, both hooks are a silent
  no-op. The push guard only acts on actual `git push` commands.

## Requirements

A working `python` on `PATH` (an activated virtualenv is the normal setup) with
`ruff` installed: `pip install ruff` — already included in `pip install .[tests]`.

## Testing a hook manually

Pipe a sample event payload to a hook on stdin:

```bash
printf '{"tool_input":{"file_path":"some_file.py"}}' | python .claude/hooks/ruff_on_edit.py
printf '{"tool_input":{"command":"git push"}}'      | python .claude/hooks/ruff_guard_push.py
```
