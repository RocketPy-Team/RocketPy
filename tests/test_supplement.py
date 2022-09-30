from itertools import cycle
import numpy as np
import pytest

from rocketpy import supplement


def test_disk_geometry():
    Disk0 = supplement.Disk(0)
    Disk1 = supplement.Disk(1)
    Disk2 = supplement.Disk(3)

    assert Disk0.radius == 0
    assert Disk1.radius == 1
    assert Disk2.radius == 3

    assert Disk0.area == 0
    assert Disk1.area == np.pi
    assert Disk2.area == np.pi * 3**2

    assert Disk0.centroid == 0
    assert Disk1.centroid == 0
    assert Disk2.centroid == 0

    assert Disk0.volume == 0
    assert Disk1.volume == 0
    assert Disk2.volume == 0

    assert Disk0.volume_to_height() == 0
    assert Disk1.volume_to_height() == 0
    assert Disk2.volume_to_height() == 0


def test_cylinder_geometry():
    Cylinder0 = supplement.Cylinder(0, 0)
    Cylinder1 = supplement.Cylinder(3, 10)
    Cylinder2 = supplement.Cylinder(1, 2 / np.pi, 1)

    assert Cylinder0.radius == 0
    assert Cylinder1.radius == 3
    assert Cylinder2.radius == 1

    assert Cylinder0.sectional_area == 0.0
    assert Cylinder1.sectional_area == np.pi * 3**2
    assert Cylinder2.sectional_area == np.pi

    assert Cylinder0.centroid == 0.0
    assert Cylinder1.centroid == 5.0
    assert Cylinder2.centroid == 1 / np.pi

    assert Cylinder0.volume == 0.0
    assert Cylinder1.volume == np.pi * 3**2 * 10
    assert Cylinder2.volume == 2.0

    assert Cylinder1.volume_to_height(np.pi * 3**2 * 5) == 5.0
    assert Cylinder2.volume_to_height(1) == 1 / np.pi

    assert Cylinder0.filled_height == 0
    assert Cylinder1.filled_height == 0
    assert Cylinder2.filled_height == 1 / np.pi


def test_cylinder_filling():
    Cylinder1 = supplement.Cylinder(3, 10)
    Cylinder2 = supplement.Cylinder(1, 2 / np.pi, 1)

    # Change filling
    Cylinder1.filled_volume = np.pi * 3**2 * 5
    Cylinder2.filled_volume = 2

    assert Cylinder1.filled_volume == np.pi * 3**2 * 5
    assert Cylinder2.filled_volume == 2

    assert Cylinder1.filled_height == 5.0
    assert Cylinder2.filled_height == 2 / np.pi

    assert Cylinder1.filled_centroid == 2.5
    assert Cylinder2.filled_centroid == 1 / np.pi


def test_hemisphere_geometry():
    Hemisphere0 = supplement.Hemisphere(0)
    Hemisphere1 = supplement.Hemisphere(3)

    assert Hemisphere0.radius == 0
    assert Hemisphere1.radius == 3

    assert Hemisphere0.centroid == 0
    assert Hemisphere1.centroid == 0

    assert Hemisphere0.volume == 0.0
    assert Hemisphere1.volume == 2 / 3 * np.pi * 3**3

    assert np.isclose(
        Hemisphere1.volume_to_height(1 / 3 * np.pi * 3**3),
        3 * (1 - 2 * np.cos(4 * np.pi / 9)),
        atol=1e-10,
        rtol=1e-12,
    )

    assert Hemisphere0.filled_height == 0
    assert Hemisphere1.filled_height == 0


def test_hemisphere_filling():
    Hemisphere1 = supplement.Hemisphere(3)

    # Change filling
    Hemisphere1.filled_volume = 1 / 3 * np.pi * 3**3

    assert Hemisphere1.filled_volume == 1 / 3 * np.pi * 3**3

    assert np.isclose(
        Hemisphere1.filled_height,
        3 * (1 - 2 * np.cos(4 * np.pi / 9)),
        atol=1e-10,
        rtol=1e-12,
    )

    assert np.isclose(
        Hemisphere1.filled_centroid, 0.698077340960644, atol=1e-10, rtol=1e-12
    )
