import numpy as np
import unittest

from rocketpy.motors import TankGeometry


class TestGeometry(unittest.TestCase):
    def test_disk_geometry(self):
        self.assertRaises(ValueError, TankGeometry.Disk, 0)
        Disk1 = TankGeometry.Disk(3)

        self.assertEqual(Disk1.radius, 3)
        self.assertEqual(Disk1.area, np.pi * 3**2)
        self.assertEqual(Disk1.centroid, 0)
        self.assertEqual(Disk1.empty_centroid, 0)
        self.assertEqual(Disk1.volume, 0)
        self.assertEqual(Disk1.empty_volume, 0)
        self.assertEqual(Disk1.volume_to_height(), 0)

    def test_cylinder_geometry(self):
        self.assertRaises(ValueError, TankGeometry.Cylinder, 0, 0)
        Cylinder1 = TankGeometry.Cylinder(3, 10)
        Cylinder2 = TankGeometry.Cylinder(1, 2 / np.pi, 1)

        self.assertEqual(Cylinder1.radius, 3)
        self.assertEqual(Cylinder2.radius, 1)

        self.assertEqual(Cylinder1.sectional_area, np.pi * 3**2)
        self.assertEqual(Cylinder2.sectional_area, np.pi)

        self.assertEqual(Cylinder1.centroid, 5.0)
        self.assertEqual(Cylinder2.centroid, 1 / np.pi)

        self.assertEqual(Cylinder1.volume, np.pi * 3**2 * 10)
        self.assertEqual(Cylinder2.volume, 2.0)

        self.assertEqual(Cylinder1.volume_to_height(np.pi * 3**2 * 5), 5.0)
        self.assertEqual(Cylinder2.volume_to_height(1), 1 / np.pi)

        self.assertEqual(Cylinder1.filled_height, 0)
        self.assertEqual(Cylinder2.filled_height, 1 / np.pi)

        self.assertEqual(Cylinder1.empty_centroid, 5.0)
        self.assertEqual(Cylinder2.empty_centroid, 3 / (2 * np.pi))

    def test_cylinder_filling(self):
        Cylinder1 = TankGeometry.Cylinder(3, 10)
        Cylinder2 = TankGeometry.Cylinder(1, 2 / np.pi, 1)

        # Change filling
        Cylinder1.filled_volume = np.pi * 3**2 * 5
        Cylinder2.filled_volume = 2

        self.assertEqual(Cylinder1.filled_volume, np.pi * 3**2 * 5)
        self.assertEqual(Cylinder2.filled_volume, 2)

        self.assertEqual(Cylinder1.empty_volume, np.pi * 3**2 * 5)
        self.assertEqual(Cylinder2.empty_volume, 0)

        self.assertEqual(Cylinder1.filled_height, 5.0)
        self.assertEqual(Cylinder2.filled_height, 2 / np.pi)

        self.assertEqual(Cylinder1.filled_centroid, 2.5)
        self.assertEqual(Cylinder2.filled_centroid, 1 / np.pi)

    def test_hemisphere_geometry(self):
        self.assertRaises(ValueError, TankGeometry.Hemisphere, 0)
        Hemisphere1 = TankGeometry.Hemisphere(3)

        self.assertEqual(Hemisphere1.radius, 3)
        self.assertEqual(Hemisphere1.centroid, 0)
        self.assertEqual(Hemisphere1.volume, 2 / 3 * np.pi * 3**3)
        self.assertAlmostEqual(
            Hemisphere1.volume_to_height(1 / 3 * np.pi * 3**3),
            3 * (1 - 2 * np.cos(4 * np.pi / 9)),
        )
        self.assertEqual(Hemisphere1.filled_height, 0)
        self.assertEqual(Hemisphere1.filled_centroid, 0)

    def test_hemisphere_filling(self):
        Hemisphere1 = TankGeometry.Hemisphere(3)
        Hemisphere2 = TankGeometry.Hemisphere(3, fill_direction="downwards")

        # Change filling
        Hemisphere1.filled_volume = 1 / 3 * np.pi * 3**3
        Hemisphere2.filled_volume = 1 / 3 * np.pi * 3**3

        self.assertEqual(Hemisphere1.filled_volume, 1 / 3 * np.pi * 3**3)
        self.assertEqual(Hemisphere2.filled_volume, 1 / 3 * np.pi * 3**3)

        self.assertAlmostEqual(
            Hemisphere1.filled_height, 3 * (1 - 2 * np.cos(4 * np.pi / 9))
        )
        self.assertAlmostEqual(Hemisphere2.filled_height, 3 * 2 * np.cos(4 * np.pi / 9))

        self.assertAlmostEqual(Hemisphere1.filled_centroid, 1.260033593037774)
        self.assertAlmostEqual(Hemisphere2.filled_centroid, 0.510033593037774)
