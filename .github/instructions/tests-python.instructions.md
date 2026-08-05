---
description: "Use when creating or editing pytest files in tests/. Enforces AAA structure, naming conventions, fixture usage, parameterization, slow-test marking, and numerical assertion practices for RocketPy."
name: "RocketPy Pytest Standards"
applyTo: "tests/**/*.py"
---
# RocketPy Test Authoring Guidelines

- Unit tests are mandatory for new behavior.
- Follow AAA structure in each test: Arrange, Act, Assert.
- Use descriptive test names matching project convention:
  - `test_methodname`
  - `test_methodname_stateundertest`
  - `test_methodname_expectedbehaviour`
- Include docstrings that clearly state expected behavior and context.
- Prefer parameterization for scenario matrices instead of duplicated tests.
- Classify tests correctly:
  - `tests/unit`: fast, method-focused tests (sociable unit tests are acceptable in RocketPy).
  - `tests/integration`: broad multi-method/component interactions and strongly I/O-oriented cases.
  - `tests/acceptance`: realistic end-user/flight scenarios with threshold-based expectations.
- By RocketPy convention, tests centered on `all_info()` behavior are integration tests.
- Reuse fixtures from `tests/fixtures` whenever possible.
- Keep fixture organization aligned with existing categories under `tests/fixtures`
  (environment, flight, motor, rockets, surfaces, units, etc.).
- If you add a new fixture module, update `tests/conftest.py` so fixtures are discoverable.
- Keep tests deterministic: set seeds when randomness is involved and avoid unstable external
  dependencies unless integration behavior explicitly requires them.
- Use `pytest.approx` for floating-point comparisons with realistic tolerances.
- Mark expensive tests with `@pytest.mark.slow` and ensure they can run under the project slow-test
  workflow.
- Include at least one negative or edge-case assertion for new behaviors.
- When adding a bug fix, include a regression test that fails before the fix and passes after it.

See:
- `docs/development/testing.rst`
- `docs/development/style_guide.rst`
- `docs/development/setting_up.rst`
