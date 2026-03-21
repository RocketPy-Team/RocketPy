# RocketPy Workspace Instructions

## Code Style
- Use snake_case for variables, functions, methods, and modules. Use descriptive names.
- Use PascalCase for classes and UPPER_SNAKE_CASE for constants.
- Keep lines at 88 characters and follow PEP 8 unless existing code in the target file differs.
- Run Ruff as the source of truth for formatting/import organization:
  - `make format`
  - `make lint`
- Use NumPy-style docstrings for public classes, methods, and functions, including units.
- In case of tooling drift between docs and config, prefer current repository tooling in `Makefile`
  and `pyproject.toml`.

## Architecture
- RocketPy is a modular Python library; keep feature logic in the correct package boundary:
  - `rocketpy/simulation`: flight simulation and Monte Carlo orchestration.
  - `rocketpy/rocket`, `rocketpy/motors`, `rocketpy/environment`: domain models.
  - `rocketpy/mathutils`: numerical primitives and interpolation utilities.
  - `rocketpy/plots`, `rocketpy/prints`: output and visualization layers.
- Prefer extending existing classes/patterns over introducing new top-level abstractions.
- Preserve public API stability in `rocketpy/__init__.py` exports.

## Build and Test
- Use Makefile targets for OS-agnostic workflows:
  - `make install`
  - `make pytest`
  - `make pytest-slow`
  - `make coverage`
  - `make coverage-report`
  - `make build-docs`
- Before finishing code changes, run focused tests first, then broader relevant suites.
- When running Python directly in this workspace, prefer `.venv/Scripts/python.exe`.
- Slow tests are explicitly marked with `@pytest.mark.slow` and are run with `make pytest-slow`.
- For docs changes, check `make build-docs` output and resolve warnings/errors when practical.

## Development Workflow
- Target pull requests to `develop` by default; `master` is the stable branch.
- Use branch names in `type/description` format, such as:
  - `bug/<description>`
  - `doc/<description>`
  - `enh/<description>`
  - `mnt/<description>`
  - `tst/<description>`
- Prefer rebasing feature branches on top of `develop` to keep history linear.
- Keep commit and PR titles explicit and prefixed with project acronyms when possible:
  - `BUG`, `DOC`, `ENH`, `MNT`, `TST`, `BLD`, `REL`, `REV`, `STY`, `DEV`.

## Conventions
- SI units are the default. Document units and coordinate-system references explicitly.
- Position/reference-frame arguments are critical in this codebase. Be explicit about orientation
  (for example, `tail_to_nose`, `nozzle_to_combustion_chamber`).
- Include unit tests for new behavior. Follow AAA structure and clear test names.
- Use fixtures from `tests/fixtures`; if adding a new fixture module, update `tests/conftest.py`.
- Use `pytest.approx` for floating-point checks where appropriate.
- Use `@cached_property` for expensive computations when helpful, and be careful with stale-cache
  behavior when underlying mutable state changes.
- Keep behavior backward compatible across the public API exported via `rocketpy/__init__.py`.
- Prefer extending existing module patterns over creating new top-level package structure.

## Testing Taxonomy
- Unit tests are mandatory for new behavior.
- Unit tests in RocketPy can be sociable (real collaborators allowed) but should still be fast and
  method-focused.
- Treat tests as integration tests when they are strongly I/O-oriented or broad across many methods,
  including `all_info()` convention cases.
- Acceptance tests represent realistic user/flight scenarios and may compare simulation thresholds to
  known flight data.

## Documentation Links
- Contributor workflow and setup: `docs/development/setting_up.rst`
- Style and naming details: `docs/development/style_guide.rst`
- Testing philosophy and structure: `docs/development/testing.rst`
- API reference conventions: `docs/reference/index.rst`
- Domain/physics background: `docs/technical/index.rst`

## Scoped Customizations
- Simulation-specific rules: `.github/instructions/simulation-safety.instructions.md`
- Test-authoring rules: `.github/instructions/tests-python.instructions.md`
- RST/Sphinx documentation rules: `.github/instructions/sphinx-docs.instructions.md`
- Specialized review persona: `.github/agents/rocketpy-reviewer.agent.md`
