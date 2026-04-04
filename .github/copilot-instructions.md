# GitHub Copilot Instructions for RocketPy

This file provides instructions for GitHub Copilot when working on the RocketPy codebase.
These guidelines help ensure consistency with the project's coding standards and development practices.

## Project Overview

RocketPy is a Python library for 6-DOF rocket trajectory simulation.
It's designed for high-power rocketry applications with focus on accuracy, performance, and ease of use.

## Coding Standards

### Naming Conventions
- **Use `snake_case` for all new code** - variables, functions, methods, and modules
- **Use descriptive names** - prefer `angle_of_attack` over `a` or `alpha`
- **Class names use PascalCase** - e.g., `SolidMotor`, `Environment`, `Flight`
- **Constants use UPPER_SNAKE_CASE** - e.g., `DEFAULT_GRAVITY`, `EARTH_RADIUS`

### Code Style
- Follow **PEP 8** guidelines
- Line length: **88 characters** (Black's default)
- Organize imports with **isort**
- Our official formatter is the **ruff frmat**

### Documentation
- **All public classes, methods, and functions must have docstrings**
- Use **NumPy style docstrings**
- Include **Parameters**, **Returns**, and **Examples** sections
- Document **units** for physical quantities (e.g., "in meters", "in radians")

### Testing
- Write **unit tests** for all new features using pytest
- Follow **AAA pattern** (Arrange, Act, Assert)
- Use descriptive test names following: `test_methodname_expectedbehaviour`
- Include test docstrings explaining expected behavior
- Use **parameterization** for testing multiple scenarios
- Create pytest fixtures to avoid code repetition

## Domain-Specific Guidelines

### Physical Units and Conventions
- **SI units by default** - meters, kilograms, seconds, radians
- **Document coordinate systems** clearly (e.g., "tail_to_nose", "nozzle_to_combustion_chamber")
- **Position parameters** are critical - always document reference points
- Use **descriptive variable names** for physical quantities

### Rocket Components
- **Motors**: SolidMotor, HybridMotor and LiquidMotor classes are children classes of the Motor class
- **Aerodynamic Surfaces**: They have Drag curves and lift coefficients
- **Parachutes**: Trigger functions, deployment conditions
- **Environment**: Atmospheric models, weather data, wind profiles

### Mathematical Operations
- Use **numpy arrays** for vectorized operations (this improves performance)
- Prefer **scipy functions** for numerical integration and optimization
- **Handle edge cases** in calculations (division by zero, sqrt of negative numbers)
- **Validate input ranges** for physical parameters
- Monte Carlo simulations: sample from `numpy.random` for random number generation and creates several iterations to assess uncertainty in simulations.

## File Structure and Organization

### Source Code Organization

Reminds that `rocketpy` is a Python package served as a library, and its source code is organized into several modules to facilitate maintainability and clarity. The following structure is recommended:

```
rocketpy/
├── core/           # Core simulation classes
├── motors/         # Motor implementations
├── environment/    # Atmospheric and environmental models
├── plots/          # Plotting and visualization
├── tools/          # Utility functions
└── mathutils/      # Mathematical utilities
```

Please refer to popular Python packages like `scipy`,  `numpy`, and `matplotlib` for inspiration on module organization.

### Test Organization
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── acceptance/     # Acceptance tests
└── fixtures/       # Test fixtures organized by component
```

### Documentation Structure
```
docs/
├── user/           # User guides and tutorials
├── development/    # Development documentation
├── reference/      # API reference
├── examples/       # Flight examples and notebooks
└── technical/      # Technical documentation
```

## Common Patterns and Practices

### Error Handling
- Use **descriptive error messages** with context
- **Validate inputs** at class initialization and method entry
- Raise **appropriate exception types** (ValueError, TypeError, etc.)
- Include **suggestions for fixes** in error messages

### Performance Considerations
- Use **vectorized operations** where possible
- **Cache expensive computations** when appropriate (we frequently use `cached_property`)
- Keep in mind that RocketPy must be fast!

### Backward Compatibility
- **Avoid breaking changes** in public APIs
- Use **deprecation warnings** before removing features
- **Document code changes** in docstrings and CHANGELOG

## AI Assistant Guidelines

### Code Generation
- **Always include docstrings** for new functions and classes
- **Follow existing patterns** in the codebase
- **Consider edge cases** and error conditions

### Code Review and Suggestions
- **Check for consistency** with existing code style
- **Verify physical units** and coordinate systems
- **Ensure proper error handling** and input validation
- **Suggest performance improvements** when applicable
- **Recommend additional tests** for new functionality

### Documentation Assistance
- **Use NumPy docstring format** consistently
- **Include practical examples** in docstrings
- **Document physical meanings** of parameters
- **Cross-reference related functions** and classes

## Testing Guidelines

### Unit Tests
- **Test individual methods** in isolation
- **Use fixtures** from the appropriate test fixture modules
- **Mock external dependencies** when necessary
- **Test both happy path and error conditions**

### Integration Tests
- **Test interactions** between components
- **Verify end-to-end workflows** (Environment → Motor → Rocket → Flight)

### Test Data
- **Use realistic parameters** for rocket simulations
- **Include edge cases** (very small/large rockets, extreme conditions)
- **Test with different coordinate systems** and orientations

## Project-Specific Considerations

### User Experience
- **Provide helpful error messages** with context and suggestions
- **Include examples** in docstrings and documentation
- **Support common use cases** with reasonable defaults

## Examples of Good Practices

### Function Definition
```python
def calculate_drag_force(
    velocity,
    air_density,
    drag_coefficient,
    reference_area
):
    """Calculate drag force using the standard drag equation.

    Parameters
    ----------
    velocity : float
        Velocity magnitude in m/s.
    air_density : float
        Air density in kg/m³.
    drag_coefficient : float
        Dimensionless drag coefficient.
    reference_area : float
        Reference area in m².

    Returns
    -------
    float
        Drag force in N.

    Examples
    --------
    >>> drag_force = calculate_drag_force(100, 1.225, 0.5, 0.01)
    >>> print(f"Drag force: {drag_force:.2f} N")
    """
    if velocity < 0:
        raise ValueError("Velocity must be non-negative")
    if air_density <= 0:
        raise ValueError("Air density must be positive")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")

    return 0.5 * air_density * velocity**2 * drag_coefficient * reference_area
```

### Test Example
```python
def test_calculate_drag_force_returns_correct_value():
    """Test drag force calculation with known inputs."""
    # Arrange
    velocity = 100.0  # m/s
    air_density = 1.225  # kg/m³
    drag_coefficient = 0.5
    reference_area = 0.01  # m²
    expected_force = 30.625  # N

    # Act
    result = calculate_drag_force(velocity, air_density, drag_coefficient, reference_area)

    # Assert
    assert abs(result - expected_force) < 1e-6
```


Remember: RocketPy prioritizes accuracy, performance, and usability. Always consider the physical meaning of calculations and provide clear, well-documented interfaces for users.
