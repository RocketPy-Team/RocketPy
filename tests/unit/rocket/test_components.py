"""Unit tests for rocketpy.rocket.components.Components."""

import pytest

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.components import Components


def test_components_add_default_ref_factor():
    """Components.add stores ref_factor=1.0 when omitted."""
    components = Components()
    component = object()
    position = Vector([0, 0, 1.0])

    components.add(component, position)

    assert len(components) == 1
    assert components[0].component is component
    assert components[0].position == position
    assert components[0].ref_factor == pytest.approx(1.0)


def test_components_add_custom_ref_factor():
    """Components.add stores an explicit ref_factor on the component tuple."""
    components = Components()
    component = object()
    position = Vector([0, 0, -0.5])
    ref_factor = 0.25

    components.add(component, position, ref_factor=ref_factor)

    stored = components[0]
    assert stored.component is component
    assert stored.position == position
    assert stored.ref_factor == pytest.approx(ref_factor)


def test_components_to_dict_from_dict_preserves_ref_factor():
    """Serialization round-trip keeps ref_factor, defaulting when absent."""
    components = Components()
    component = {"name": "dummy"}
    position = Vector([0, 0, 0.1])
    components.add(component, position, ref_factor=4.0)

    restored = Components.from_dict(components.to_dict())
    assert restored[0].ref_factor == pytest.approx(4.0)

    legacy = Components.from_dict(
        {"components": [{"component": component, "position": position}]}
    )
    assert legacy[0].ref_factor == pytest.approx(1.0)


def test_rocket_add_surfaces_stores_computed_ref_factor(calisto):
    """Rocket aero-surface add path stores (surface.rocket_radius / rocket.radius)**2."""
    from rocketpy import NoseCone

    surface_radius = calisto.radius / 2
    expected_ref_factor = (surface_radius / calisto.radius) ** 2
    nose = NoseCone(
        length=0.55829,
        kind="vonkarman",
        base_radius=surface_radius,
        rocket_radius=surface_radius,
        name="Half-radius nose",
    )

    calisto.add_surfaces(nose, 1.16)
    stored = next(
        entry for entry in calisto.aerodynamic_surfaces if entry.component is nose
    )

    assert stored.ref_factor == pytest.approx(expected_ref_factor)
    assert stored.ref_factor == pytest.approx(0.25)


def test_rocket_add_surfaces_matching_radius_stores_unit_ref_factor(calisto):
    """Matching surface and rocket radii store ref_factor of 1.0."""
    from rocketpy import NoseCone

    nose = NoseCone(
        length=0.55829,
        kind="vonkarman",
        base_radius=calisto.radius,
        rocket_radius=calisto.radius,
        name="Matching nose",
    )

    calisto.add_surfaces(nose, 1.16)
    stored = next(
        entry for entry in calisto.aerodynamic_surfaces if entry.component is nose
    )

    assert stored.ref_factor == pytest.approx(1.0)
