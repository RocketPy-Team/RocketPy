import numpy as np
import pytest

from rocketpy.exceptions import InvalidParameterError
from rocketpy.mathutils import Matrix
from rocketpy.plots.plate_plots import _PlatePlots
from rocketpy.rocket.plate import Plate


@pytest.mark.parametrize(
    "shape, dimensions, material, thickness, absolute_magnetic_permeability, grid_spacing, z_points, angular_points, relative_magnetic_permeability, name",
    [
        (
            0,
            0.05,
            "personalized",
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong shape: not str
        (
            "hi",
            0.05,
            "personalized",
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong shape: not allowed str
        (
            "rectangular",
            "0.05",
            "personalized",
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong dimensions rectangular: string
        (
            "rectangular",
            [0, 9, 10],
            "personalized",
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong dimensions rectangular: list of not 2 elements
        (
            "circular",
            "0.05",
            "personalized",
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong dimensions: string
        (
            "circular",
            0.05,
            0,
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong material: not a string
        (
            "circular",
            0.05,
            "0",
            0.002,
            1e-3,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong material: not allowed str
        (
            "circular",
            0.05,
            "personalized",
            0.002,
            None,
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong magnetic inputs: neither relative nor absolute defined
        (
            "circular",
            0.05,
            "personalized",
            0.002,
            "None",
            0.0002,
            40,
            40,
            None,
            "plate_test",
        ),  # wrong abs magnetic permeability: str
        (
            "circular",
            0.05,
            "personalized",
            0.002,
            None,
            0.0002,
            40,
            40,
            "None",
            "plate_test",
        ),  # wrong relative magnetic permeability: str
    ],
)
def test_validate_parameters(
    shape,
    dimensions,
    material,
    thickness,
    absolute_magnetic_permeability,
    grid_spacing,
    z_points,
    angular_points,
    relative_magnetic_permeability,
    name,
):
    """Tests that InvalidParameterError are raised with incorrect entry parameters."""
    with pytest.raises(InvalidParameterError):
        Plate(
            shape,
            dimensions,
            material,
            thickness,
            absolute_magnetic_permeability,
            relative_magnetic_permeability,
            grid_spacing,
            z_points,
            angular_points,
            name,
        )


@pytest.mark.parametrize(
    "vertices",
    [
        [
            [10, 3, 4],
            [0.03, 0.04, 0.4],
            [0.03, 0.004, 0],
        ],  # wrong vertex: z out of bonds
        [
            [0.002, 3, 0.4],
            [0.03, 0.04, -0.04],
            [0.03, 0.004, 0],
        ],  # wrong vertex: radius out of bonds
        [[0.001, 0.003, 0.04], [0.003, -0.004, 0.4]],  # wrong size: at least 3
    ],
)
def test_generate_personalized_points_bounds_out_of_bonds(vertices, calisto_robust):
    """Tests the raise of InvalidParameterError when the vertices are out of the rocket."""
    test_plate = Plate(
        shape="personalized",
        dimensions=vertices,
        material="carbon_steel",
        thickness=0.001,
        grid_spacing=0.002,
        name="test_personalized_plate",
    )
    with pytest.raises(InvalidParameterError):
        calisto_robust.add_plate(test_plate)


@pytest.mark.parametrize(
    "vertices",
    [
        [
            [0, 0, 0],
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
        ],  # wrong vertices: 1D (line)
        [
            [0, 0, 0],
            [-0.001, -0.001, -0.001],
            [0.001, 0.001, 0.001],
        ],  # wrong vertices: 1D (line)
        [
            [0, 0, 0],
            [-0.001, -0.001, -0.001],
            [0.001, 0.001, 0.001],
            [-0.002, -0.002, -0.002],
            [0.002, 0.002, 0.002],
        ],  # wrong vertices: 1D (line)
    ],
)
def test_generate_personalized_points_bounds_collinear(vertices, calisto_robust):
    """Tests the raise of InvalidParameterError when the vertices don't form a 2D surface."""
    test_plate = Plate(
        shape="personalized",
        dimensions=vertices,
        material="carbon_steel",
        thickness=0.001,
        grid_spacing=0.002,
        name="test_personalized_plate",
    )
    with pytest.raises(
        InvalidParameterError,
        match="All vertices of the personalized plate are collinear.",
    ):
        calisto_robust.add_plate(test_plate)


@pytest.mark.parametrize(
    "angle, expected_x",  # considering that calisto's rocket frame is tail to nose, the same as bacs
    [
        (0, 0.0),  # Angle = 0°   -> x = 0
        (90, -0.0635),  # Angle = 90°  -> x = -R
        (180, 0.0),  # Angle = 180° -> x = 0
        (270, +0.0635),  # Angle = 270° -> x = +R
    ],
)
def test_generate_angular_plate_position(angle, expected_x, calisto_robust):
    """Tests proper position vector generation in the BACS frame
    based on the angular position.
    """
    small_plate = Plate(
        shape="circular",
        dimensions=0.004,
        material="carbon_steel",
        thickness=0.0002,
        z_points=40,
        angular_points=70,
        name="small_circular_plate",
    )
    calisto_robust.add_plate(small_plate, position=angle, height=0.2)
    one_x_true = any(
        pytest.approx(expected_x, abs=1e-3) == point[0] for point in small_plate.points
    )
    assert one_x_true


@pytest.mark.parametrize(
    "plate",
    [
        "test_circular_plate",
        "test_squared_plate",
        "test_rectangular_plate",
    ],
)
def test_plate_dimensions(request, plate, calisto_robust):
    """Tests proper position vector generation in the BACS frame
    based on dimensions.
    """
    plate_object = request.getfixturevalue(plate)
    height = 0.0
    position_deg = 0.0

    calisto_robust.add_plate(plate_object, position=position_deg, height=height)

    radius = calisto_robust.general_radius(height, frame="ucs")

    # Center position in BACS
    z_bacs_center = (
        height - calisto_robust.center_of_dry_mass_position
    ) * calisto_robust._csys

    if plate_object.shape == "circular":
        half_height = plate_object.dimensions
        theta = plate_object.dimensions / radius
    else:
        half_height = plate_object.dimensions[1] / 2.0
        theta = (plate_object.dimensions[0] / 2.0) / radius

    max_expected_x = radius * np.sin(theta)
    max_expected_y = radius
    expected_max_z = z_bacs_center + half_height
    expected_min_z = z_bacs_center - half_height

    points = plate_object.points

    assert any(pytest.approx(max_expected_x, abs=1e-5) == p[0] for p in points)
    assert any(pytest.approx(max_expected_y, abs=1e-5) == p[1] for p in points)
    assert any(pytest.approx(expected_max_z, abs=1e-5) == p[2] for p in points)
    assert any(pytest.approx(expected_min_z, abs=1e-5) == p[2] for p in points)


@pytest.mark.parametrize(
    "height_ucs",
    [
        1.15,  # almost out of the rocket(upper bound)
        -1.2,  # almost out of rocket(lower bound)
    ],
)
def test_less_points_plate(height_ucs, calisto_robust):
    """Tests that when the dimensions, or the point generated are out of the rocket
    no error is raised and some points have been dismissed.
    """
    z_points = 50
    angular_points = 70
    test_plate = Plate(
        shape="circular",
        material="carbon_steel",
        thickness=0.001,
        dimensions=0.4,
        z_points=z_points,
        angular_points=angular_points,
        name="less_points_than_default_plate",
    )
    calisto_robust.add_plate(test_plate, position=30, height=height_ucs)
    assert 0 < len(test_plate.points) < z_points * angular_points


def test_several_positions_soft_iron_distortion_matrix(
    test_squared_plate, calisto_robust
):
    """Ensures proper handling of the soft iron distoriton matrices
    as a dictionary.
    """
    calisto_robust.add_plate(test_squared_plate, position=45, height=0.2)
    test_squared_plate.calculate_soft_iron_distortion_matrix((0, 0, 0), frame="bacs")
    test_squared_plate.calculate_soft_iron_distortion_matrix(
        (0.004, 0.003, 0.5), frame="bacs"
    )
    assert len(test_squared_plate._magnetic_distortion_matrices) == 2

    list_keys = list(test_squared_plate._magnetic_distortion_matrices)
    assert list_keys[0] == (0, 0, 0)
    assert list_keys[1] == (0.004, 0.003, 0.5)


def test_no_soft_iron_distortion_matrix(calisto_robust):
    """Tests that when the relative magnetic permeability is 1, no distortion is added."""
    test_plate = Plate(
        shape="circular",
        dimensions=0.04,
        material="personalized",
        thickness=0.001,
        absolute_magnetic_permeability=4 * np.pi * 1e-7,
        name="identity_soft_iron",
    )
    calisto_robust.add_plate(test_plate, position=30, height=0.3)
    test_plate.calculate_soft_iron_distortion_matrix([0, 0, 0], frame="bacs")
    assert test_plate._magnetic_distortion_matrices[(0, 0, 0)] == Matrix.zeros()


@pytest.mark.parametrize(
    "dimensions_1, material_1, absolute_magnetic_permeability_1, dimensions_2, material_2, absolute_magnetic_permeability_2",
    [
        (
            0.7,
            "carbon_steel",
            None,
            0.007,
            "carbon_steel",
            None,
        ),  # bigger plate -> higher distortion
        (
            0.07,
            "iron",
            None,
            0.07,
            "carbon_steel",
            None,
        ),  # bigger absolute magnetic permeability -> higher distortion
        (
            0.07,
            "personalized",
            1.25e-7,
            0.07,
            "personalized",
            1.25e-6,
        ),  # lower absolute magnetic permeability (diamagnetic materials) than vacuum -> higher distortion
    ],
)
def test_compare_soft_iron_distortion_matrix(
    dimensions_1,
    material_1,
    absolute_magnetic_permeability_1,
    dimensions_2,
    material_2,
    absolute_magnetic_permeability_2,
    calisto_robust,
):
    """
    Tests that
    - The plate class is able to handle wrapped plates, which is also
      reflected in the magnitude of the distortion of the soft iron
      distortion matrix.
    - Bigger magnetic permeability causes bigger magnetic distortion.
    """
    z_points = 150
    angular_points = 200
    shape = "circular"
    thickness = 0.003
    position = 30
    height = 0.6

    # definition of plates
    big_plate = Plate(
        shape=shape,
        dimensions=dimensions_1,
        material=material_1,
        absolute_magnetic_permeability=absolute_magnetic_permeability_1,
        thickness=thickness,
        z_points=z_points,
        angular_points=angular_points,
    )
    small_plate = Plate(
        shape=shape,
        dimensions=dimensions_2,
        material=material_2,
        absolute_magnetic_permeability=absolute_magnetic_permeability_2,
        thickness=thickness,
        z_points=z_points,
        angular_points=angular_points,
    )

    # big_plate
    calisto_robust.add_plate(big_plate, position=position, height=height)
    big_plate.calculate_soft_iron_distortion_matrix((0, 0, 0), frame="bacs")
    big_plate_matrix = big_plate._magnetic_distortion_matrices[(0, 0, 0)]

    # small_plate
    calisto_robust.add_plate(small_plate, position=position, height=height)
    small_plate.calculate_soft_iron_distortion_matrix((0, 0, 0), frame="bacs")
    small_plate_matrix = small_plate._magnetic_distortion_matrices[(0, 0, 0)]

    for row in range(3):
        for column in range(3):
            assert abs(big_plate_matrix[row, column]) > abs(
                small_plate_matrix[row, column]
            )


def test_rocket_belonging(test_circular_plate, calisto_robust):
    """Tests if the belonging attribute works as intended"""
    calisto_robust.add_plate(test_circular_plate, position=-0.3, height=0.3)
    assert isinstance(test_circular_plate.plots, _PlatePlots)
    assert not isinstance(test_circular_plate._csys, type(None))
    assert not isinstance(test_circular_plate.center_of_dry_mass_position, type(None))


@pytest.mark.parametrize(
    "plate, position, height",
    [
        ("test_personalized_plate", None, None),
        ("test_circular_plate", 45, 0.3),
        ("test_squared_plate", 45, 0.4),
    ],
)
def test_plate_prints_and_plots(request, plate, position, height, calisto_robust):
    """Test the print methods of the Plate class. Checks if all attributes are
    printed and plotted correctly.
    """
    plate_obj = request.getfixturevalue(plate)
    with pytest.raises(InvalidParameterError):
        plate_obj.plots.draw()
    with pytest.raises(InvalidParameterError):
        plate_obj.plots.draw_3d()
    calisto_robust.add_plate(plate_obj, position, height)
    plate_obj.prints.all()
    plate_obj.plots.all()
    assert True


def test_from_dict():
    """Tests if from_dict returns a Plate object."""
    plate_dict = {
        "shape": "rectangular",
        "dimensions": 0.003,
        "material": "personalized",
        "thickness": 0.001,
        "absolute_magnetic_permeability": 1e-5,
        "relative_magnetic_permeability": None,
        "z_points": 50,
        "angular_points": 50,
        "name": "from_dict_plate",
    }
    plate_from_dict = Plate.from_dict(plate_dict)
    assert isinstance(plate_from_dict, Plate)
    assert plate_from_dict.dimensions == (0.003, 0.003)
    assert plate_from_dict.material == "personalized"
    assert plate_from_dict.thickness == pytest.approx(0.001)
    assert plate_from_dict.z_points == pytest.approx(50)
    assert plate_from_dict.angular_points == pytest.approx(50)
