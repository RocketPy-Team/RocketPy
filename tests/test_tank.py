import numpy as np
import unittest

from rocketpy.motors import Fluid, TankGeometry, UllageBasedTank


class TestTank(unittest.TestCase):
    def setUp(self):
        self.LiquidN2O = Fluid(name="Liquid Nitrous Oxide", density=855, quality=1)
        self.VapourN2O = Fluid(name="Vapour Nitrous Oxide", density=101, quality=0)

    def test_flat_cap_tank(self):
        self.n2o_ullage_tank = UllageBasedTank(
            name="Oxidizer Tank",
            diameter=0.2,
            height=1.6,
            bottomCap="flat",
            upperCap="flat",
            ullage=np.pi * 0.1**2 * 0.8,
            gas=self.VapourN2O,
            liquid=self.LiquidN2O,
        )

        self.assertAlmostEqual(self.n2o_ullage_tank.centerOfMass(0), 0.48452, 5)
        self.assertAlmostEqual(self.n2o_ullage_tank.inertiaTensor(0)[0], 2.794, 3)
        self.assertEqual(self.n2o_ullage_tank.inertiaTensor(0)[1], 0)

    def test_spheric_cap_tank(self):
        self.n2o_ullage_tank = UllageBasedTank(
            name="Oxidizer Tank",
            diameter=0.2,
            height=1.6,
            bottomCap="spherical",
            upperCap="spherical",
            ullage=np.pi * 0.1**2 * 0.8 + 2 / 3 * np.pi * 0.1**3,
            gas=self.VapourN2O,
            liquid=self.LiquidN2O,
        )

        self.assertAlmostEqual(self.n2o_ullage_tank.centerOfMass(0), 0.558, 3)
        self.assertAlmostEqual(self.n2o_ullage_tank.inertiaTensor(0)[0], 3.550, 3)
        self.assertEqual(self.n2o_ullage_tank.inertiaTensor(0)[1], 0)

    def test_spherical_tank(self):
        self.n2o_ullage_tank = UllageBasedTank(
            name="Oxidizer Tank",
            diameter=2,
            height=0,
            bottomCap="spherical",
            upperCap="spherical",
            ullage=4 / 3 * np.pi * 1**3,
            gas=self.VapourN2O,
        )

        self.assertAlmostEqual(self.n2o_ullage_tank.centerOfMass(0), 1)
        self.assertAlmostEqual(self.n2o_ullage_tank.inertiaTensor(0)[0], 169.227, 3)
        self.assertEqual(self.n2o_ullage_tank.inertiaTensor(0)[1], 0)

        self.n2o_ullage_tank = UllageBasedTank(
            name="Oxidizer Tank",
            diameter=2,
            height=0,
            bottomCap="spherical",
            upperCap="spherical",
            ullage=0,
            liquid=self.LiquidN2O,
        )

        self.assertAlmostEqual(self.n2o_ullage_tank.centerOfMass(0), 1)
        self.assertAlmostEqual(self.n2o_ullage_tank.inertiaTensor(0)[0], 1432.566, 3)
        self.assertEqual(self.n2o_ullage_tank.inertiaTensor(0)[1], 0)
