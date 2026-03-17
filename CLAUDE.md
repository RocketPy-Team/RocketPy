# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
make install

# Run tests
make pytest              # standard test suite
make pytest-slow         # slow/marked tests with verbose output
make coverage            # tests with coverage

# Lint and format
make format              # ruff format + isort
make lint                # ruff + pylint
make ruff-lint           # ruff only
make pylint              # pylint only

# Docs
make build-docs
```

**Run a single test file:**
```bash
pytest tests/unit/test_environment.py -v
```

**Run a single test by name:**
```bash
pytest tests/unit/test_environment.py::test_methodname_expectedbehaviour -v
```

## Architecture

RocketPy simulates 6-DOF rocket trajectories. The core workflow is linear:

```
Environment → Motor → Rocket → Flight
```

**`rocketpy/environment/`** — Atmospheric models, weather data fetching (NOAA, Wyoming soundings, GFS forecasts). The `Environment` class (~116KB) is the entry point for atmospheric conditions.

**`rocketpy/motors/`** — `Motor` is the base class. `SolidMotor`, `HybridMotor`, and `LiquidMotor` extend it. `Tank`, `TankGeometry`, and `Fluid` support liquid/hybrid propellant modeling.

**`rocketpy/rocket/`** — `Rocket` aggregates a motor and aerodynamic surfaces. `aero_surface/` contains fins, nose cone, and tail implementations. `Parachute` uses trigger functions for deployment.

**`rocketpy/simulation/`** — `Flight` (~162KB) is the simulation engine, integrating equations of motion with scipy's LSODA solver. `MonteCarlo` orchestrates many `Flight` runs for dispersion analysis.

**`rocketpy/stochastic/`** — Wraps any component (Environment, Rocket, Motor, Flight) with uncertainty distributions for Monte Carlo input generation.

**`rocketpy/mathutils/`** — `Function` class wraps callable data (arrays, lambdas, files) with interpolation and mathematical operations. Heavily used throughout for aerodynamic curves, thrust profiles, etc.

**`rocketpy/plots/` and `rocketpy/prints/`** — Visualization and text output, each mirroring the module structure of the core classes.

**`rocketpy/sensors/`** — Simulated sensors (accelerometer, gyroscope, barometer, GNSS) that can be attached to a `Rocket`.

**`rocketpy/sensitivity/`** — Global sensitivity analysis via `SensitivityModel`.

## Coding Standards

- **Docstrings:** NumPy style with `Parameters`, `Returns`, and `Examples` sections. Always include units for physical quantities (e.g., "in meters", "in radians").
- **No type hints in function signatures** — put types in the docstring `Parameters` section instead.
- **SI units by default** throughout the codebase (meters, kilograms, seconds, radians).
- **No magic numbers** — name constants with `UPPER_SNAKE_CASE` and comment their physical meaning.
- **Performance:** Use vectorized numpy operations. Cache expensive computations with `cached_property`.
- **Test names:** `test_methodname_expectedbehaviour` pattern. Use `pytest.approx` for float comparisons.
- **Tests follow AAA** (Arrange, Act, Assert) with fixtures from `tests/fixtures/`.
- **Backward compatibility:** Use deprecation warnings before removing public API features; document changes in CHANGELOG.
