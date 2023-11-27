import os
from math import isclose

import numpy as np
import pytest

from rocketpy import (
    Fluid,
    Function,
    LevelBasedTank,
    MassBasedTank,
    MassFlowRateBasedTank,
    TankGeometry,
)

pressurant_params = (0.135 / 2, 0.981)
fuel_params = (0.0744, 0.8068)
oxidizer_params = (0.0744, 0.8068)

parametrize_fixtures = pytest.mark.parametrize(
    "params",
    [
        ("pressurant_tank", pressurant_params),
        ("fuel_tank", fuel_params),
        ("oxidizer_tank", oxidizer_params),
    ],
)


@parametrize_fixtures
def test_tank_bounds(params, request):
    """Test basic geometric properties of the tanks."""
    tank, (expected_radius, expected_height) = params
    tank = request.getfixturevalue(tank)

    expected_total_height = expected_height

    assert tank.geometry.radius(0) == pytest.approx(expected_radius, abs=1e-6)
    assert tank.geometry.total_height == pytest.approx(expected_total_height, abs=1e-6)


@parametrize_fixtures
def test_tank_coordinates(params, request):
    """Test basic coordinate values of the tanks."""
    tank, (radius, height) = params
    tank = request.getfixturevalue(tank)

    expected_bottom = -height / 2
    expected_top = height / 2

    assert tank.geometry.bottom == pytest.approx(expected_bottom, abs=1e-6)
    assert tank.geometry.top == pytest.approx(expected_top, abs=1e-6)


@parametrize_fixtures
def test_tank_total_volume(params, request):
    """Test the total volume of the tanks comparing to the analytically
    calculated values.
    """
    tank, (radius, height) = params
    tank = request.getfixturevalue(tank)

    expected_total_volume = (
        np.pi * radius**2 * (height - 2 * radius) + 4 / 3 * np.pi * radius**3
    )

    assert tank.geometry.total_volume == pytest.approx(expected_total_volume, abs=1e-6)


@parametrize_fixtures
def test_tank_volume(params, request):
    """Test the volume of the tanks at different heights comparing to the
    analytically calculated values.
    """
    tank, (radius, height) = params
    tank = request.getfixturevalue(tank)

    total_height = height
    bottom = -height / 2
    top = height / 2

    expected_volume = tank_volume_function(radius, total_height, bottom)

    for h in np.linspace(bottom, top, 101):
        assert tank.geometry.volume(h) == pytest.approx(expected_volume(h), abs=1e-6)


@parametrize_fixtures
def test_tank_centroid(params, request):
    """Test the centroid of the tanks at different heights comparing to the
    analytically calculated values.
    """
    tank, (radius, height) = params
    tank = request.getfixturevalue(tank)

    total_height = height
    bottom = -height / 2

    expected_centroid = tank_centroid_function(radius, total_height, bottom)

    for h, liquid_com in zip(
        tank.liquid_height.y_array, tank.liquid_center_of_mass.y_array
    ):
        # Loss of accuracy to 1e-3 when liquid height is close to zero
        assert liquid_com == pytest.approx(expected_centroid(h), abs=1e-3)


@parametrize_fixtures
def test_tank_inertia(params, request):
    """Test the inertia of the tanks at different heights comparing to the
    analytically calculated values.
    """
    tank, (radius, height) = params
    tank = request.getfixturevalue(tank)

    total_height = height
    bottom = -height / 2

    expected_inertia = tank_inertia_function(radius, total_height, bottom)

    for h in tank.liquid_height.y_array:
        assert tank.geometry.Ix_volume(tank.geometry.bottom, h)(h) == pytest.approx(
            expected_inertia(h)[0], abs=1e-5
        )


def test_mass_based_tank():
    """Tests the MassBasedTank subclass of Tank regarding its mass and
    net mass flow rate properties. The test is performed on both a real
    tank and a simplified tank.
    """
    lox = Fluid(name="LOx", density=1141.7)
    propane = Fluid(
        name="Propane",
        density=493,
    )
    n2 = Fluid(
        name="Nitrogen Gas",
        density=51.75,
    )  # density value may be estimate

    top_endcap = lambda y: np.sqrt(
        0.0775**2 - (y - 0.7924) ** 2
    )  # Hemisphere equation creating top endcap
    bottom_endcap = lambda y: np.sqrt(
        0.0775**2 - (0.0775 - y) ** 2
    )  # Hemisphere equation creating bottom endcap

    # Generate tank geometry {radius: height, ...}
    real_geometry = TankGeometry(
        {
            (0, 0.0559): bottom_endcap,
            (0.0559, 0.8039): lambda y: 0.0744,
            (0.8039, 0.8698): top_endcap,
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

    # Assert volume bounds
    assert (real_tank_lox.gas_height <= real_tank_lox.geometry.top).all
    assert (real_tank_lox.fluid_volume <= real_tank_lox.geometry.total_volume).all
    assert (example_tank_lox.gas_height <= example_tank_lox.geometry.top).all
    assert (example_tank_lox.fluid_volume <= example_tank_lox.geometry.total_volume).all

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


def test_level_based_tank():
    """Test LevelBasedTank subclass of Tank class using Berkeley SEB team's
    tank data of fluid level.
    """
    lox = Fluid(name="LOx", density=1141.7)
    n2 = Fluid(name="Nitrogen Gas", density=51.75)

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

    calculated_mfr = levelTank.net_mass_flow_rate.set_discrete(
        mass_flow_rate_data[0][0],
        mass_flow_rate_data[0][-1],
        len(mass_flow_rate_data[0]),
    )
    calculated_mfr, test_mfr = align_time_series(
        calculated_mfr.get_source(), mass_flow_rate_data
    )


def test_mfr_tank_basic():
    """Test MassFlowRateTank subclass of Tank class regarding its properties,
    such as net_mass_flow_rate, fluid_mass, center_of_mass and inertia.
    """

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
    lox = Fluid(
        name="LOx",
        density=1141,
    )
    n2 = Fluid(
        name="Nitrogen Gas",
        density=51.75,
    )  # density value may be estimate
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
        # Coordinate shift to the bottom of the tank
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
        # Coordinate shift to the bottom of the tank
        h = h - zero_height
        cylinder_height = tank_height - 2 * tank_radius

        if h < tank_radius:
            centroid = lower_spherical_cap_centroid(tank_radius, h)

        elif tank_radius <= h < tank_height - tank_radius:
            # Fluid height from cylinder base
            base = tank_radius
            height = h - base

            balance = lower_spherical_cap_volume(
                tank_radius
            ) * lower_spherical_cap_centroid(tank_radius) + cylinder_volume(
                tank_radius, height
            ) * (
                cylinder_centroid(height) + tank_radius
            )
            volume = lower_spherical_cap_volume(tank_radius) + cylinder_volume(
                tank_radius, height
            )
            centroid = balance / volume

        else:
            # Fluid height from upper cap base
            base = tank_height - tank_radius
            height = h - base

            balance = (
                lower_spherical_cap_volume(tank_radius)
                * lower_spherical_cap_centroid(tank_radius)
                + cylinder_volume(tank_radius, cylinder_height)
                * (cylinder_centroid(cylinder_height) + tank_radius)
                + upper_spherical_cap_volume(tank_radius, height)
                * (upper_spherical_cap_centroid(tank_radius, height) + base)
            )
            volume = (
                lower_spherical_cap_volume(tank_radius)
                + cylinder_volume(tank_radius, cylinder_height)
                + upper_spherical_cap_volume(tank_radius, height)
            )
            centroid = balance / volume

        return centroid + zero_height

    return tank_centroid


def cylinder_inertia(radius, height, reference=0):
    """Returns the inertia of a cylinder with the given radius and height.

    Parameters
    ----------
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder.
    reference : float, optional
        The coordinate of the axis of rotation.

    Returns
    -------
    numpy.ndarray
        The inertia of the cylinder in the x, y, and z directions.
    """
    # Evaluate inertia and perform coordinate shift to the reference point
    inertia_x = cylinder_volume(radius, height) * (
        radius**2 / 4 + height**2 / 12 + (height / 2 - reference) ** 2
    )
    inertia_y = inertia_x
    inertia_z = cylinder_volume(radius, height) * radius**2 / 2

    return np.array([inertia_x, inertia_y, inertia_z])


def lower_spherical_cap_inertia(radius, height=None, reference=0):
    """Returns the inertia of a spherical cap with the given radius and filled
    height that is filled from its convex side.

    Parameters
    ----------
    radius : float
        The radius of the spherical cap.
    height : float
        The height of the spherical cap. If not given, the radius is used.
    reference : float, optional
        The coordinate of the axis of rotation.

    Returns
    -------
    numpy.ndarray
        The inertia of the spherical cap in the x, y, and z directions.
    """
    if height is None:
        height = radius

    centroid = lower_spherical_cap_centroid(radius, height)

    # Evaluate inertia and perform coordinate shift to the reference point
    inertia_x = lower_spherical_cap_volume(radius, height) * (
        (
            np.pi
            * height**2
            * (
                -9 * height**3
                + 45 * height**2 * radius
                - 80 * height * radius**2
                + 60 * radius**3
            )
            / 60
        )
        - (radius - centroid) ** 2
        + (centroid - reference) ** 2
    )
    inertia_y = inertia_x
    inertia_z = lower_spherical_cap_volume(radius, height) * (
        np.pi
        * height**3
        * (3 * height**2 - 15 * height * radius + 20 * radius**2)
        / 30
    )
    return np.array([inertia_x, inertia_y, inertia_z])


def upper_spherical_cap_inertia(radius, height=None, reference=0):
    """Returns the inertia of a spherical cap with the given radius and filled
    height that is filled from its concave side.

    Parameters
    ----------
    radius : float
        The radius of the spherical cap.
    height : float
        The height of the spherical cap. If not given, the radius is used.
    reference : float, optional
        The coordinate of the axis of rotation.

    Returns
    -------
    numpy.ndarray
        The inertia of the spherical cap in the x, y, and z directions.
    """
    if height is None:
        height = radius

    centroid = upper_spherical_cap_centroid(radius, height)

    # Evaluate inertia and perform coordinate shift to the reference point
    inertia_x = upper_spherical_cap_volume(radius, height) * (
        (
            (
                np.pi
                * height
                * (-9 * height**4 + 10 * height**2 * radius**2 + 15 * radius**4)
            )
            / 60
        )
        - centroid**2
        + (centroid - reference) ** 2
    )
    inertia_y = inertia_x
    inertia_z = upper_spherical_cap_volume(radius, height) * (
        np.pi
        * height
        * (3 * height**4 - 10 * height**2 * radius**2 + 15 * radius**4)
        / 30
    )
    return np.array([inertia_x, inertia_y, inertia_z])


def tank_inertia_function(tank_radius, tank_height, zero_height=0):
    """Returns a function that calculates the inertia of a cylindrical tank
    with spherical caps. The reference point is the tank centroid.

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
        A function that calculates the inertia of the tank for a given height.
    """

    def tank_inertia(h):
        # Coordinate shift to the bottom of the tank
        h = h - zero_height
        center = tank_height / 2
        cylinder_height = tank_height - 2 * tank_radius

        if h < tank_radius:
            inertia = lower_spherical_cap_inertia(tank_radius, h, center)

        elif tank_radius <= h < tank_height - tank_radius:
            # Fluid height from cylinder base
            base = tank_radius
            height = h - base

            lower_centroid = lower_spherical_cap_centroid(tank_radius)
            cyl_centroid = cylinder_centroid(height) + base
            lower_inertia = lower_spherical_cap_inertia(tank_radius, reference=center)
            cyl_inertia = cylinder_inertia(tank_radius, height, reference=center - base)

            inertia = lower_inertia + cyl_inertia

        else:
            # Fluid height from upper cap base
            base = tank_height - tank_radius
            height = h - base

            lower_centroid = lower_spherical_cap_centroid(tank_radius)
            cyl_centroid = cylinder_centroid(cylinder_height) + tank_radius
            upper_centroid = upper_spherical_cap_centroid(tank_radius, height) + base

            lower_inertia = lower_spherical_cap_inertia(
                tank_radius, reference=lower_centroid - center
            )
            cyl_inertia = cylinder_inertia(
                tank_radius, cylinder_height, cyl_centroid - tank_radius
            )
            upper_inertia = upper_spherical_cap_inertia(
                tank_radius, height, upper_centroid - base
            )

            inertia = lower_inertia + cyl_inertia + upper_inertia

        return inertia

    return tank_inertia
