---
name: rocketpy-release
description: >-
  Cut a new RocketPy release: prepare the version-bump branch, update the
  changelog and all version references, and guide the PR-to-master + PyPI flow.
  Use when the user wants to release a new version (e.g. "release v1.13",
  "fazer um release", "bump version and cut a release", "REL PR").
---

# RocketPy Release

Prepare and guide a RocketPy version release following the team's established
`REL:` pattern. This skill prepares the branch and the version-bump commit; the
human opens the PR and publishes the GitHub Release themselves.

## Branching model

RocketPy uses **`develop`** (integration) and **`master`** (released). All
feature/fix work lands on `develop`. A release promotes `develop` → `master`.

Since **v1.12.0** the team uses a **single streamlined PR** (see PR #935):
one `rel/vX.Y.Z` branch — created off `develop`, with the version already
bumped — opened **straight to `master`**. (Older releases used three PRs; do
NOT revive that.)

PyPI is published automatically by `.github/workflows/publish-to-pypi.yml` when
a **GitHub Release is published** (`on: release: [published]`). Creating the
release/tag in the GitHub UI is the final manual trigger.

## Before you start — gather state

Run these and confirm with the user:

```bash
git fetch origin
grep -m1 '^version' pyproject.toml                 # current version
git show origin/master:pyproject.toml | grep -m1 '^version'   # released version
git rev-list --left-right --count origin/master...origin/develop  # ahead/behind
git tag --sort=-creatordate | head -5              # last tags
```

Confirm the **target version** (semver: MAJOR.MINOR.PATCH) with the user before
touching files. Never guess the number.

## Step 1 — Create the release branch

Branch off the latest `develop`:

```bash
git checkout develop && git pull origin develop
git checkout -b rel/vX.Y.Z
```

## Step 2 — The version-bump commit

Update **every** version reference. These have been missed before (e.g.
`installation.rst` sat stale at 1.11.0 through 1.12.x) — check all of them:

| File | What to change |
|------|----------------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `docs/conf.py` | `release = "X.Y.Z"` and `copyright = "<year>, RocketPy Team"` |
| `docs/user/installation.rst` | `pip install rocketpy==X.Y.Z` |
| `CHANGELOG.md` | see below |

Sanity-sweep for any other stragglers (ignore `.venv`):

```bash
grep -rn "<OLD_VERSION>" --include=*.py --include=*.toml --include=*.rst . \
  | grep -v '.venv'
```

### CHANGELOG.md

The changelog `[Unreleased]` section is auto-populated on every merge to
`develop` by `.github/workflows/changelog.yml`, so entries should already be
present. To release:

1. Rename the current `## [Unreleased] - yyyy-mm-dd` header to
   `## [vX.Y.Z] - YYYY-MM-DD` (today's date).
2. Insert a **fresh empty** `[Unreleased]` block above it, preserving the
   template comment and empty subsections:

   ```markdown
   ## [Unreleased] - yyyy-mm-dd

   <!-- These are the changes that were not released yet, please add them correctly.
   Attention: The newest changes should be on top -->

   ### Added

   ### Changed

   ### Fixed
   ```
3. Add a `Changed` entry to the released section for the bump itself:
   `- REL: bumps up rocketpy version to X.Y.Z [#PR](https://github.com/RocketPy-Team/RocketPy/pull/PR)`
   (fill the PR number after the PR is opened, or leave a placeholder to edit).
4. Prune obvious noise if the maintainer wants (tests, CI, merge commits are
   not meant to be in the changelog per the file's own header comment).

### Commit

Match the historical message style (`REL: bump version to X.Y.Z`):

```bash
git add pyproject.toml docs/conf.py docs/user/installation.rst CHANGELOG.md
git commit -m "REL: bumps up rocketpy version to X.Y.Z"
```

Keep it to **one** bump commit. Do not include unrelated changes.

## Step 3 — Push and hand off the PR

```bash
git push -u origin rel/vX.Y.Z
```

**Stop here and let the human open the PR** (team preference — they open it
themselves). Give them the ready-to-use details:

- **Base:** `master`  ←  **Compare:** `rel/vX.Y.Z`
- **Title:** `REL: vX.Y.Z` (or `Rel/vX.Y.Z`)
- **Body:** summarize the highlights; note the single-PR-to-master approach.

## Step 4 — After the PR merges (guide the human)

1. **Sync `master` back into `develop`** so develop carries the new version and
   the merge history. Historically a PR titled like
   `MNT: update develop to release vX.Y.Z`, base `develop` ← compare `master`.
2. **Create the GitHub Release** in the UI: new tag `vX.Y.Z` targeting
   `master`, title `vX.Y.Z`, paste the changelog section as notes, Publish.
   → This triggers `publish-to-pypi.yml` and ships to PyPI.
3. Verify the PyPI publish workflow succeeded and the new version is live.

## Guardrails

- Only prepare the branch + bump commit. **Do not open the PR or publish the
  release** unless the user explicitly asks — they do those steps themselves.
- Confirm the target version with the user before editing.
- Base the PR on **`master`**, not `develop`.
- One clean `REL:` commit; no unrelated edits.
