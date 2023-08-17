import os
from math import isclose

import numpy as np
import pytest

from rocketpy import Fluid
from rocketpy.Function import Function
from rocketpy.motors import TankGeometry
from rocketpy.motors.Fluid import Fluid
from rocketpy.motors.Tank import LevelBasedTank, MassBasedTank, MassFlowRateBasedTank


# @PBales1
def test_mass_based_motor():
    lox = Fluid(name="LOx", density=1141.7, quality=1.0)  # Placeholder quality value
    propane = Fluid(
        name="Propane", density=493, quality=1.0
    )  # Placeholder quality value
    n2 = Fluid(
        name="Nitrogen Gas", density=51.75, quality=1.0
    )  # Placeholder quality value; density value may be estimate

    top_endcap = lambda y: np.sqrt(
        0.0775**2 - (y - 0.692300000000001) ** 2
    )  # Hemisphere equation creating top endcap
    bottom_endcap = lambda y: np.sqrt(
        0.0775**2 - (0.0775 - y) ** 2
    )  # Hemisphere equation creating bottom endcap

    # Generate tank geometry {radius: height, ...}
    real_geometry = TankGeometry(
        {
            (0, 0.0559): bottom_endcap,
            (0.0559, 0.7139): lambda y: 0.0744,
            (0.7139, 0.7698): top_endcap,
        }
    )

    # Import liquid mass data
    lox_masses = "./data/berkeley/Test135LoxMass.csv"
    example_liquid_masses = "./data/berkeley/ExampleTankLiquidMassData.csv"

    # Import gas mass data
    gas_masses = "./data/berkeley/Test135GasMass.csv"
    example_gas_masses = "./data/berkeley/ExampleTankGasMassData.csv"

    # Generate tanks based on Berkeley SEB team's real tank geometries
    real_tank_lox = MassBasedTank(
        name="Real Tank",
        geometry=real_geometry,
        flux_time=(0, 10),
        liquid_mass=lox_masses,
        gas_mass=gas_masses,
        liquid=lox,
        gas=n2,
        discretize=None,
    )

    # Generate tank geometry {radius: height, ...}
    example_geometry = TankGeometry({(0, 5): 1})

    # Generate tanks based on simplified tank geometry
    example_tank_lox = MassBasedTank(
        name="Example Tank",
        geometry=example_geometry,
        flux_time=(0, 10),
        liquid_mass=example_liquid_masses,
        gas_mass=example_gas_masses,
        liquid=lox,
        gas=n2,
        discretize=None,
    )

    initial_liquid_mass = 5
    initial_gas_mass = 0
    liquid_mass_flow_rate_in = 0.1
    gas_mass_flow_rate_in = 0.1
    liquid_mass_flow_rate_out = 0.2
    gas_mass_flow_rate_out = 0.05

    def test(calculated, expected, t, real=False):
        """Iterate over time range and test that calculated value is close to actual value"""
        j = 0
        for i in np.arange(0, t, 0.1):
            try:
                print(calculated.get_value(i), expected(i))
                assert isclose(calculated.get_value(i), expected(i), rel_tol=5e-2)
            except IndexError:
                break

            if real:
                j += 4
            else:
                j += 1

    def test_mass():
        """Test mass function of MassBasedTank subclass of Tank"""
        example_expected = (
            lambda t: initial_liquid_mass
            + t * (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out)
            + initial_gas_mass
            + t * (gas_mass_flow_rate_in - gas_mass_flow_rate_out)
        )
        example_calculated = example_tank_lox.fluid_mass

        lox_vals = Function(lox_masses).y_array

        real_expected = lambda t: lox_vals[t]
        real_calculated = real_tank_lox.fluid_mass

        test(example_calculated, example_expected, 5)
        test(real_calculated, real_expected, 15.5, real=True)

    def test_net_mfr():
        """Test net_mass_flow_rate function of MassBasedTank subclass of Tank"""
        example_expected = (
            lambda t: liquid_mass_flow_rate_in
            - liquid_mass_flow_rate_out
            + gas_mass_flow_rate_in
            - gas_mass_flow_rate_out
        )
        example_calculated = example_tank_lox.net_mass_flow_rate

        liquid_mfrs = Function(example_liquid_masses).y_array

        gas_mfrs = Function(example_gas_masses).y_array

        real_expected = lambda t: (liquid_mfrs[t] + gas_mfrs[t]) / t
        real_calculated = real_tank_lox.net_mass_flow_rate

        test(example_calculated, example_expected, 10)
        test(real_calculated, real_expected, 15.5, real=True)

    test_mass()
    test_net_mfr()


# @curtisjhu
def test_ullage_based_motor():
    lox = Fluid(name="LOx", density=1141.7, quality=1.0)
    n2 = Fluid(name="Nitrogen Gas", density=51.75, quality=1.0)

    test_dir = "./data/berkeley/"

    top_endcap = lambda y: np.sqrt(0.0775**2 - (y - 0.692300000000001) ** 2)
    bottom_endcap = lambda y: np.sqrt(0.0775**2 - (0.0775 - y) ** 2)
    tank_geometry = TankGeometry(
        {
            (0, 0.0559): bottom_endcap,
            (0.0559, 0.7139): lambda y: 0.0744,
            (0.7139, 0.7698): top_endcap,
        }
    )

    ullage_data = Function(os.path.abspath(test_dir + "loxUllage.csv")).get_source()
    levelTank = LevelBasedTank(
        name="LevelTank",
        geometry=tank_geometry,
        flux_time=(0, 10),
        gas=n2,
        liquid=lox,
        liquid_height=ullage_data,
        discretize=None,
    )

    mass_data = Function(test_dir + "loxMass.csv").get_source()
    mass_flow_rate_data = Function(test_dir + "loxMFR.csv").get_source()

    def align_time_series(small_source, large_source):
        assert isinstance(small_source, np.ndarray) and isinstance(
            large_source, np.ndarray
        ), "Must be np.ndarrays"
        if small_source.shape[0] > large_source.shape[0]:
            small_source, large_source = large_source, small_source

        result_larger_source = np.ndarray(small_source.shape)
        result_smaller_source = np.ndarray(small_source.shape)
        tolerance = 0.1
        curr_ind = 0
        for val in small_source:
            time = val[0]
            delta_time_vector = abs(time - large_source[:, 0])
            largeIndex = np.argmin(delta_time_vector)
            delta_time = abs(time - large_source[largeIndex][0])

            if delta_time < tolerance:
                result_larger_source[curr_ind] = large_source[largeIndex]
                result_smaller_source[curr_ind] = val
                curr_ind += 1
        return result_larger_source, result_smaller_source

    assert np.allclose(levelTank.liquid_height, ullage_data)

    calculated_mass = levelTank.liquid_mass.set_discrete(
        mass_data[0][0], mass_data[0][-1], len(mass_data[0])
    )
    calculated_mass, mass_data = align_time_series(
        calculated_mass.get_source(), mass_data
    )
    assert np.allclose(calculated_mass, mass_data, rtol=1, atol=2)
    # Function(calculated_mass).plot1D()
    # Function(mass_data).plot1D()

    calculated_mfr = levelTank.net_mass_flow_rate.set_discrete(
        mass_flow_rate_data[0][0],
        mass_flow_rate_data[0][-1],
        len(mass_flow_rate_data[0]),
    )
    calculated_mfr, test_mfr = align_time_series(
        calculated_mfr.get_source(), mass_flow_rate_data
    )
    # assert np.allclose(calculated_mfr, test_mfr)
    # Function(calculated_mfr).plot1D()
    # Function(test_mfr).plot1D()


# @gautamsaiy
def test_mfr_tank_basic():
    def test(t, a, tol=1e-4):
        for i in np.arange(0, 10, 1):
            print(t.get_value(i), a(i))
            assert isclose(t.get_value(i), a(i), abs_tol=tol)

    def test_nmfr():
        nmfr = (
            lambda x: liquid_mass_flow_rate_in
            + gas_mass_flow_rate_in
            - liquid_mass_flow_rate_out
            - gas_mass_flow_rate_out
        )
        test(t.net_mass_flow_rate, nmfr)

    def test_mass():
        m = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        ) + (initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x)
        lm = t.fluid_mass
        test(lm, m)

    def test_liquid_height():
        alv = (
            lambda x: (
                initial_liquid_mass
                + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
            )
            / lox.density
        )
        alh = lambda x: alv(x) / (np.pi)
        tlh = t.liquid_height
        test(tlh, alh)

    def test_com():
        liquid_mass = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        )  # liquid mass
        liquid_volume = lambda x: liquid_mass(x) / lox.density  # liquid volume
        liquid_height = lambda x: liquid_volume(x) / (np.pi)  # liquid height
        gas_mass = lambda x: (
            initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x
        )  # gas mass
        gas_volume = lambda x: gas_mass(x) / n2.density
        gas_height = lambda x: gas_volume(x) / np.pi + liquid_height(x)

        liquid_com = lambda x: liquid_height(x) / 2  # liquid com
        gas_com = lambda x: (gas_height(x) - liquid_height(x)) / 2 + liquid_height(
            x
        )  # gas com
        acom = lambda x: (liquid_mass(x) * liquid_com(x) + gas_mass(x) * gas_com(x)) / (
            liquid_mass(x) + gas_mass(x)
        )

        tcom = t.center_of_mass
        test(tcom, acom)

    def test_inertia():
        liquid_mass = lambda x: (
            initial_liquid_mass
            + (liquid_mass_flow_rate_in - liquid_mass_flow_rate_out) * x
        )  # liquid mass
        liquid_volume = lambda x: liquid_mass(x) / lox.density  # liquid volume
        liquid_height = lambda x: liquid_volume(x) / (np.pi)  # liquid height
        gas_mass = lambda x: (
            initial_gas_mass + (gas_mass_flow_rate_in - gas_mass_flow_rate_out) * x
        )  # gas mass
        gas_volume = lambda x: gas_mass(x) / n2.density
        gas_height = lambda x: gas_volume(x) / np.pi + liquid_height(x)

        liquid_com = lambda x: liquid_height(x) / 2  # liquid com
        gas_com = lambda x: (gas_height(x) - liquid_height(x)) / 2 + liquid_height(
            x
        )  # gas com
        acom = lambda x: (liquid_mass(x) * liquid_com(x) + gas_mass(x) * gas_com(x)) / (
            liquid_mass(x) + gas_mass(x)
        )

        r = 1
        ixy_gas = (
            lambda x: 1 / 4 * gas_mass(x) * r**2
            + 1 / 12 * gas_mass(x) * (gas_height(x) - liquid_height(x)) ** 2
            + gas_mass(x) * (gas_com(x) - acom(x)) ** 2
        )
        ixy_liq = (
            lambda x: 1 / 4 * liquid_mass(x) * r**2
            + 1 / 12 * liquid_mass(x) * (liquid_height(x) - t.geometry.bottom) ** 2
            + liquid_mass(x) * (liquid_com(x) - acom(x)) ** 2
        )
        ixy = lambda x: ixy_gas(x) + ixy_liq(x)
        test(t.gas_inertia, ixy_gas, tol=1e-3)
        test(t.liquid_inertia, ixy_liq, tol=1e-3)
        test(t.inertia, ixy, tol=1e-3)

    tank_radius_function = TankGeometry({(0, 5): 1})
    lox = Fluid(name="LOx", density=1141, quality=1.0)  # Placeholder quality value
    n2 = Fluid(
        name="Nitrogen Gas", density=51.75, quality=1.0
    )  # Placeholder quality value; density value may be estimate
    initial_liquid_mass = 5
    initial_gas_mass = 0.1
    liquid_mass_flow_rate_in = 0.1
    gas_mass_flow_rate_in = 0.01
    liquid_mass_flow_rate_out = 0.2
    gas_mass_flow_rate_out = 0.02

    t = MassFlowRateBasedTank(
        name="Test Tank",
        geometry=tank_radius_function,
        flux_time=(0, 10),
        initial_liquid_mass=initial_liquid_mass,
        initial_gas_mass=initial_gas_mass,
        liquid_mass_flow_rate_in=Function(0.1).set_discrete(0, 10, 1000),
        gas_mass_flow_rate_in=Function(0.01).set_discrete(0, 10, 1000),
        liquid_mass_flow_rate_out=Function(0.2).set_discrete(0, 10, 1000),
        gas_mass_flow_rate_out=Function(0.02).set_discrete(0, 10, 1000),
        liquid=lox,
        gas=n2,
        discretize=None,
    )

    test_nmfr()
    test_mass()
    test_liquid_height()
    test_com()
    test_inertia()


pressurant_tank_radius = 0.135 / 2
fuel_tank_radius = 0.0744
oxidizer_tank_radius = 0.0744
pressurant_tank_height = 0.846
fuel_tank_height = 0.658
oxidizer_tank_height = 0.658


def test_tank_bounds(pressurant_tank, fuel_tank, oxidizer_tank):
    expected_pressurant_tank_height = (
        pressurant_tank_height + 2 * pressurant_tank_radius
    )
    expected_fuel_tank_height = fuel_tank_height + 2 * fuel_tank_radius
    expected_oxidizer_tank_height = oxidizer_tank_height + 2 * oxidizer_tank_radius

    assert pressurant_tank.geometry.total_height == pytest.approx(
        expected_pressurant_tank_height, 1e-6
    )
    assert fuel_tank.geometry.total_height == pytest.approx(
        expected_fuel_tank_height, 1e-6
    )
    assert oxidizer_tank.geometry.total_height == pytest.approx(
        expected_oxidizer_tank_height, 1e-6
    )


def test_tank_total_volume(pressurant_tank, fuel_tank, oxidizer_tank):
    expected_pressurant_tank_volume = (
        np.pi * pressurant_tank_radius**2 * pressurant_tank_height
        + 4 / 3 * np.pi * pressurant_tank_radius**3
    )
    expected_fuel_tank_volume = (
        np.pi * fuel_tank_radius**2 * fuel_tank_height
        + 4 / 3 * np.pi * fuel_tank_radius**3
    )
    expected_oxidizer_tank_volume = (
        np.pi * oxidizer_tank_radius**2 * oxidizer_tank_height
        + 4 / 3 * np.pi * oxidizer_tank_radius**3
    )

    assert pressurant_tank.geometry.total_volume == pytest.approx(
        expected_pressurant_tank_volume, abs=1e-6
    )
    assert fuel_tank.geometry.total_volume == pytest.approx(
        expected_fuel_tank_volume, abs=1e-6
    )
    assert oxidizer_tank.geometry.total_volume == pytest.approx(
        expected_oxidizer_tank_volume, abs=1e-6
    )


def test_tank_volume(pressurant_tank, fuel_tank, oxidizer_tank):
    for tank in [pressurant_tank, fuel_tank, oxidizer_tank]:
        tank_volume = tank_volume_function(
            tank.geometry.radius(0), tank.geometry.total_height, tank.geometry.bottom
        )
        for h in np.linspace(tank.geometry.bottom, tank.geometry.top, 101):
            assert tank.geometry.volume(h) == pytest.approx(tank_volume(h), abs=1e-6)


def test_tank_centroid(pressurant_tank, fuel_tank, oxidizer_tank):
    for tank in [pressurant_tank, fuel_tank, oxidizer_tank]:
        tank_centroid = tank_centroid_function(
            tank.geometry.radius(0), tank.geometry.total_height, tank.geometry.bottom
        )
        for h, liquid_com in zip(
            tank.liquid_height.y_array, tank.liquid_center_of_mass.y_array
        ):
            assert liquid_com == pytest.approx(tank_centroid(h), abs=1e-3)


def test_tank_inertia(pressurant_tank, fuel_tank, oxidizer_tank):
    for tank in [pressurant_tank, fuel_tank, oxidizer_tank]:
        tank_inertia = tank_inertia_function(
            tank.geometry.radius(0), tank.geometry.total_height, tank.geometry.bottom
        )
        for h in tank.liquid_height.y_array:
            print(
                h,
                tank.geometry.Ix_volume(tank.geometry.bottom, h)(h),
                tank_inertia(h)[0],
                tank.name,
            )
            assert tank.geometry.Ix_volume(tank.geometry.bottom, h)(h) == pytest.approx(
                tank_inertia(h)[0], abs=1e-4
            )


"""Auxiliary testing functions"""


def cylinder_volume(radius, height):
    """Returns the volume of a cylinder with the given radius and height.

    Parameters
    ----------
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder.

    Returns
    -------
    float
        The volume of the cylinder.
    """
    return np.pi * radius**2 * height


def lower_spherical_cap_volume(radius, height=None):
    """Returns the volume of a spherical cap with the given radius and filled
    height that is filled from its convex side.

    Parameters
    ----------
    radius : float
        The radius of the spherical cap.
    height : float, optional
        The height of the spherical cap. If not given, the radius is used.

    Returns
    -------
    float
        The volume of the spherical cap.
    """
    if height is None:
        height = radius
    return np.pi / 3 * height**2 * (3 * radius - height)


def upper_spherical_cap_volume(radius, height=None):
    """Returns the volume of a spherical cap with the given radius and filled
    height that is filled from its concave side.

    Parameters
    ----------
    radius : float
        The radius of the spherical cap.
    height : float, optional
        The height of the spherical cap. If not given, the radius is used.

    Returns
    -------
    float
        The volume of the spherical cap.
    """
    if height is None:
        height = radius
    return np.pi / 3 * height * (3 * radius**2 - height**2)


def tank_volume_function(tank_radius, tank_height, zero_height=0):
    """Returns a function that calculates the volume of a cylindrical tank
    with spherical caps.

    Parameters
    ----------
    tank_radius : float
        The radius of the cylindrical part of the tank.
    tank_height : float
        The height of the tank including caps.
    zero_height : float, optional
        The coordinate of the bottom of the tank. Defaults to 0.

    Returns
    -------
    function
        A function that calculates the volume of the tank for a given height.
    """

    def tank_volume(h):
        h = h - zero_height
        if h < tank_radius:
            return lower_spherical_cap_volume(tank_radius, h)
        elif tank_radius <= h < tank_height - tank_radius:
            return lower_spherical_cap_volume(tank_radius) + cylinder_volume(
                tank_radius, h - tank_radius
            )
        else:
            return (
                lower_spherical_cap_volume(tank_radius)
                + cylinder_volume(tank_radius, tank_height - 2 * tank_radius)
                + upper_spherical_cap_volume(
                    tank_radius, h - (tank_height - tank_radius)
                )
            )

    return tank_volume


def cylinder_centroid(height):
    """Returns the centroid of a cylinder with the given height.

    Parameters
    ----------
    height : float
        The height of the cylinder.

    Returns
    -------
    float
        The centroid of the cylinder.
    """
    return height / 2


def lower_spherical_cap_centroid(radius, height=None):
    """Returns the centroid of a spherical cap with the given radius and filled
    height that is filled from its convex side.

    Parameters
    ----------
    radius : float
        The radius of the spherical cap.
    height : float, optional
        The height of the spherical cap. If not given, the radius is used.

    Returns
    -------
    float
        The centroid of the spherical cap.
    """
    if height is None:
        height = radius
    return radius - (0.75 * (2 * radius - height) ** 2 / (3 * radius - height))


def upper_spherical_cap_centroid(radius, height=None):
    """Returns the centroid of a spherical cap with the given radius and filled
    height that is filled from its concave side.

    Parameters
    ----------
    radius : float
        The radius of the spherical cap.
    height : float, optional
        The height of the spherical cap. If not given, the radius is used.

    Returns
    -------
    float
        The centroid of the spherical cap.
    """
    if height is None:
        height = radius
    return (
        0.75
        * (height**3 - 2 * height * radius**2)
        / (height**2 - 3 * radius**2)
    )


def tank_centroid_function(tank_radius, tank_height, zero_height=0):
    """Returns a function that calculates the centroid of a cylindrical tank
    with spherical caps.

    Parameters
    ----------
    tank_radius : float
        The radius of the cylindrical part of the tank.
    tank_height : float
        The height of the tank including caps.
    zero_height : float, optional
        The coordinate of the bottom of the tank. Defaults to 0.

    Returns
    -------
    function
        A function that calculates the centroid of the tank for a given height.
    """

    def tank_centroid(h):
        h = h - zero_height
        if h < tank_radius:
            centroid = lower_spherical_cap_centroid(tank_radius, h)
        elif tank_radius <= h < tank_height - tank_radius:
            base = tank_radius
            balance = lower_spherical_cap_volume(
                tank_radius
            ) * lower_spherical_cap_centroid(tank_radius) + cylinder_volume(
                tank_radius, h - base
            ) * (
                cylinder_centroid(h - base) + tank_radius
            )
            volume = lower_spherical_cap_volume(tank_radius) + cylinder_volume(
                tank_radius, h - base
            )
            centroid = balance / volume
        else:
            base = tank_height - tank_radius
            balance = (
                lower_spherical_cap_volume(tank_radius)
                * lower_spherical_cap_centroid(tank_radius)
                + cylinder_volume(tank_radius, tank_height - 2 * tank_radius)
                * (cylinder_centroid(tank_height - 2 * tank_radius) + tank_radius)
                + upper_spherical_cap_volume(tank_radius, h - base)
                * (upper_spherical_cap_centroid(tank_radius, h - base) + base)
            )
            volume = (
                lower_spherical_cap_volume(tank_radius)
                + cylinder_volume(tank_radius, tank_height - 2 * tank_radius)
                + upper_spherical_cap_volume(tank_radius, h - base)
            )
            centroid = balance / volume
        return centroid + zero_height

    return tank_centroid
