"""Unit tests for the rocket drawing helpers in ``rocketpy.plots.rocket_plots``."""

from unittest.mock import patch

import pytest

from rocketpy import LinearGenericSurface
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.motors.ring_cluster_motor import RingClusterMotor

REFERENCE_AREA = 0.005
REFERENCE_LENGTH = 0.08


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("plane", ["xz", "yz"])
@pytest.mark.parametrize(
    "fin_fixture",
    ["calisto_trapezoidal_fin", "calisto_elliptical_fin", "calisto_free_form_fin"],
)
def test_draw_a_rocket_carrying_one_fin(  # pylint: disable=unused-argument
    mock_show, request, calisto_robust, plane, fin_fixture
):
    """A single ``Fin`` takes ``_draw_fin``, not the ``_draw_fins`` set branch.

    That method rotates one fin out of its own coordinate system into the body
    frame, and it projects differently in each plane, so both are drawn here.
    """
    fin = request.getfixturevalue(fin_fixture)
    calisto_robust.add_surfaces(fin, Vector([0, 0, -1.04956]))

    assert calisto_robust.draw(plane=plane, filename=None) is None


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_draw_a_rocket_carrying_a_generic_surface(  # pylint: disable=unused-argument
    mock_show, calisto_robust, plane
):
    """A ``GenericSurface`` has no outline to trace, so it gets a scatter point.

    ``_draw_generic_surface`` is the only branch that reads the surface position
    by index rather than by attribute, and it picks a different index per plane.
    """
    surface = LinearGenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={},
        name="Canard",
    )
    calisto_robust.add_surfaces(surface, Vector([0, 0, -0.5]))

    assert calisto_robust.draw(plane=plane, filename=None) is None


@patch("matplotlib.pyplot.show")
def test_draw_a_rocket_with_a_hybrid_motor(  # pylint: disable=unused-argument
    mock_show, calisto_hybrid_modded, calisto_nose_cone
):
    """The hybrid branch of ``_generate_motor_patches`` draws grains and tanks."""
    calisto_hybrid_modded.add_surfaces(calisto_nose_cone, 1.160)

    assert calisto_hybrid_modded.draw(filename=None) is None


@patch("matplotlib.pyplot.show")
def test_draw_a_rocket_with_a_liquid_motor(  # pylint: disable=unused-argument
    mock_show, calisto_liquid_modded, calisto_nose_cone
):
    """The liquid branch draws positioned tanks and no combustion chamber."""
    calisto_liquid_modded.add_surfaces(calisto_nose_cone, 1.160)

    assert calisto_liquid_modded.draw(filename=None) is None


@patch("matplotlib.pyplot.show")
def test_draw_a_rocket_with_a_motor_cluster(  # pylint: disable=unused-argument
    mock_show, calisto_motorless, cesaroni_m1670, calisto_nose_cone
):
    """A cluster repeats the grain patches around the ring.

    Only the first offset keeps its legend entry, so the loop that relabels the
    rest runs solely when there is more than one motor.
    """
    calisto_motorless.add_motor(
        RingClusterMotor(motor=cesaroni_m1670, number=3, radius=0.05),
        position=-1.373,
    )
    calisto_motorless.add_surfaces(calisto_nose_cone, 1.160)

    assert calisto_motorless.draw(filename=None) is None


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize(
    "rocket_fixture,nose_position",
    [("calisto", 1.160), ("calisto_nose_to_tail", -1.160)],
)
def test_draw_a_rocket_whose_nozzle_sits_behind_its_last_surface(  # pylint: disable=unused-argument
    mock_show, request, rocket_fixture, nose_position, calisto_nose_cone
):
    """``_draw_nozzle_tube`` only draws when the nozzle is past the last surface.

    A rocket carrying nothing but a nose cone leaves that gap open, and the
    comparison flips with the coordinate system, so both orientations are drawn.
    """
    rocket = request.getfixturevalue(rocket_fixture)
    rocket.add_surfaces(calisto_nose_cone, nose_position)

    assert rocket.draw(filename=None) is None


def test_draw_refuses_a_rocket_with_no_aerodynamic_surfaces(calisto_motorless):
    """There is nothing to draw the body around without at least one surface."""
    with pytest.raises(ValueError, match="at least one aerodynamic surface"):
        calisto_motorless.draw(filename=None)


def test_draw_refuses_a_plane_it_cannot_project_onto(calisto_robust):
    """Only the two longitudinal planes are defined."""
    with pytest.raises(ValueError, match="must be 'xz' or 'yz'"):
        calisto_robust.draw(plane="xy", filename=None)
