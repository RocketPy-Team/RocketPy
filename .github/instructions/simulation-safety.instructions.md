---
description: "Use when editing rocketpy/simulation code, including Flight state updates, Monte Carlo orchestration, post-processing, or cached computations. Covers simulation state safety, unit/reference-frame clarity, and regression checks."
name: "Simulation Safety"
applyTo: "rocketpy/simulation/**/*.py"
---
# Simulation Safety Guidelines

- Keep simulation logic inside `rocketpy/simulation` and avoid leaking domain behavior that belongs in
  `rocketpy/rocket`, `rocketpy/motors`, or `rocketpy/environment`.
- Preserve public API behavior and exported names used by `rocketpy/__init__.py`.
- Prefer extending existing simulation components before creating new abstractions:
  - `flight.py`: simulation state, integration flow, and post-processing.
  - `monte_carlo.py`: orchestration and statistical execution workflows.
  - `flight_data_exporter.py` and `flight_data_importer.py`: persistence and interchange.
  - `flight_comparator.py`: comparative analysis outputs.
- Be explicit with physical units and reference frames in new parameters, attributes, and docstrings.
- For position/orientation-sensitive behavior, use explicit conventions (for example
  `tail_to_nose`, `nozzle_to_combustion_chamber`) and avoid implicit assumptions.
- Treat state mutation carefully when cached values exist.
- If changes can invalidate `@cached_property` values, either avoid post-computation mutation or
  explicitly invalidate affected caches in a controlled, documented way.
- Keep numerical behavior deterministic unless stochastic behavior is intentional and documented.
- For Monte Carlo and stochastic code paths, make randomness controllable and reproducible when tests
  rely on it.
- Prefer vectorized NumPy operations for hot paths and avoid introducing Python loops in
  performance-critical sections without justification.
- Guard against numerical edge cases (zero/near-zero denominators, interpolation limits, and boundary
  conditions).
- Do not change default numerical tolerances or integration behavior without documenting motivation and
  validating regression impact.
- Add focused regression tests for changed behavior, including edge cases and orientation-dependent
  behavior.
- For floating-point expectations, use `pytest.approx` with meaningful tolerances.
- Run focused tests first, then broader relevant tests (`make pytest` and `make pytest-slow` when
  applicable).

See:
- `docs/development/testing.rst`
- `docs/development/style_guide.rst`
- `docs/development/setting_up.rst`
- `docs/technical/index.rst`
