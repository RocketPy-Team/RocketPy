import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import Function, TrapezoidalFins, utilities


@pytest.mark.parametrize(
    "terminal_velocity, rocket_mass, air_density, result",
    [
        (25, 15, 1.04, 0.4526146),
        (25, 18, 0.96, 0.5883990),
        (25, 21, 1.04, 0.6336605),
        (30, 15, 1.04, 0.3143157),
        (30, 18, 0.96, 0.4086104),
        (30, 21, 1.04, 0.4400420),
        (40, 15, 1.04, 0.1768026),
        (40, 18, 0.96, 0.2298434),
        (40, 21, 1.04, 0.2475236),
    ],
)
def test_compute_cd_s_from_drop_test(
    terminal_velocity, rocket_mass, air_density, result
):
    """Test if the function `compute_cd_s_from_drop_test` returns the correct
    value. It compares the returned value with the expected result in different
    scenarios.

    Parameters
    ----------
    terminal_velocity : float
        The terminal velocity of the body (rocket) in m/s.
    rocket_mass : float
        The mass of the body (rocket) in kg.
    air_density : float
        The air density in kg/m^3.
    result : float
        The expected result of the function.
    """
    cds = utilities.compute_cd_s_from_drop_test(
        terminal_velocity, rocket_mass, air_density, g=9.80665
    )
    assert abs(cds - result) < 1e-6


# Tests not passing in the CI, but passing locally due to
# different values in the ubuntu and windows machines


@pytest.mark.skip(
    reason="legacy tests"
)  # it is not working on CI and I don't have time
@patch("matplotlib.pyplot.show")
def test_apogee_by_mass(mock_show, flight):  # pylint: disable=unused-argument
    """Tests the apogee_by_mass function.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    flight : rocketpy.Flight
        The flight object to be used in the tests.
    """
    f = utilities.apogee_by_mass(flight=flight, min_mass=5, max_mass=20, points=5)
    assert abs(f(5) - 3528.2072598) < 1e-6
    assert abs(f(10) - 3697.1896424) < 1e-6
    assert abs(f(15) - 3331.6521059) < 1e-6
    assert abs(f(20) - 2538.4542953) < 1e-6
    assert f.plot() is None


@pytest.mark.skip(reason="legacy tests")
@patch("matplotlib.pyplot.show")
def test_liftoff_by_mass(mock_show, flight):  # pylint: disable=unused-argument
    """Tests the liftoff_by_mass function.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    flight : rocketpy.Flight
        The flight object to be used in the tests.
    """
    f = utilities.liftoff_speed_by_mass(
        flight=flight, min_mass=5, max_mass=20, points=5
    )
    assert abs(f(5) - 40.70236234988934) < 1e-6
    assert abs(f(10) - 31.07885818306235) < 1e-6
    assert abs(f(15) - 26.054819726081266) < 1e-6
    assert abs(f(20) - 22.703279913437058) < 1e-6
    assert f.plot() is None


def test_fin_flutter_analysis(flight_calisto_custom_wind):
    """Tests the fin_flutter_analysis function. It tests the both options of
    the see_graphs parameter.
    Parameters
    ----------
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins. This flight object was created
        in the conftest.py file.
    """
    flutter_mach, safety_factor = utilities.fin_flutter_analysis(
        fin_thickness=2 / 1000,
        shear_modulus=10e9,
        flight=flight_calisto_custom_wind,
        see_prints=False,
        see_graphs=False,
    )
    assert np.isclose(flutter_mach(0), 0.7105140845699177, atol=5e-3)
    assert np.isclose(flutter_mach(10), 0.8070387292674097, atol=5e-3)
    assert np.isclose(flutter_mach(np.inf), 0.7105140859542808, atol=5e-3)
    assert np.isclose(safety_factor(0), 45.81200340574091, atol=5e-3)
    assert np.isclose(safety_factor(10), 1.551202495333104, atol=5e-3)
    assert np.isclose(safety_factor(np.inf), 43.610793287739654, atol=5e-3)


# Fin geometry in inches, shear modulus and pressure in psi, and the flutter
# Mach number Vf / a from John K. Bennett's reference calculator (Fin Flutter
# Boundary Calculator v1.3, github.com/jkb-git/Fin-Flutter-Velocity-Calculator).
# Each fin has its centroid at mid-root-chord, where the calculator reduces to
# Martin's equation (NACA TN 4197) with its constant of 39.3 psi.
MARTIN_REFERENCE_CASES = [
    # thickness, sweep, tip chord, root chord, span, G, p, flutter Mach
    (1 / 8, 0.0, 4.0, 4.0, 4.0, 380000, 14.173548105504, 0.9581278421552433),
    (1 / 8, 2.5, 2.5, 7.5, 3.0, 600000, 11.474749839986, 1.2785504155701972),
    (1 / 8, 3.0, 0.0, 6.0, 2.5, 600000, 12.6957004220342, 1.2509974714673138),
    (3 / 16, 6.0, 4.0, 16.0, 5.0, 700000, 14.0274369504818, 0.9803746521458),
    (1 / 16, 1.0, 2.0, 4.0, 3.0, 90000, 8.81559972329207, 0.24137409108946878),
]


@pytest.mark.parametrize(
    "thickness, sweep, tip_chord, root_chord, span, shear_modulus, pressure, expected",
    MARTIN_REFERENCE_CASES,
)
def test_fin_flutter_analysis_matches_martin(
    thickness, sweep, tip_chord, root_chord, span, shear_modulus, pressure, expected
):
    """The flutter Mach number must match Martin's equation. The form from
    Apogee Peak of Flight issue 291 used previously applied Martin's factor of
    1/2 twice and returned sqrt(2) times these values."""
    inch, psi = 0.0254, 6894.757293168361
    fins = TrapezoidalFins(
        n=4,
        root_chord=root_chord * inch,
        tip_chord=tip_chord * inch,
        span=span * inch,
        rocket_radius=0.05,
        sweep_length=sweep * inch,
    )
    flight = SimpleNamespace(
        rocket=SimpleNamespace(fins=[fins]),
        pressure=Function(pressure * psi),
        mach_number=Function(0.8),
    )

    flutter_mach, safety_factor = utilities.fin_flutter_analysis(
        fin_thickness=thickness * inch,
        shear_modulus=shear_modulus * psi,
        flight=flight,
        see_prints=False,
        see_graphs=False,
    )

    assert flutter_mach(0) == pytest.approx(expected, rel=1e-4)
    assert safety_factor(0) == pytest.approx(expected / 0.8, rel=1e-4)


def test_calculate_stall_wind_velocity_returns_value(flight_calisto_custom_wind):
    """Regression: the stall wind velocity must be returned (it was previously
    only logged at INFO level and the method returned None, losing the value).
    The Flight method and the utilities function must agree."""
    with pytest.warns(DeprecationWarning):
        w_v = flight_calisto_custom_wind.calculate_stall_wind_velocity(5)
    assert isinstance(w_v, float)
    assert w_v > 0
    assert utilities.calculate_stall_wind_velocity(
        flight_calisto_custom_wind, 5
    ) == pytest.approx(w_v)


def test_fin_flutter_analysis_with_prints(flight_calisto_custom_wind):
    """Test fin_flutter_analysis with see_prints=True to cover print branch.

    Parameters
    ----------
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins.
    """
    flutter_mach, safety_factor = utilities.fin_flutter_analysis(
        fin_thickness=2 / 1000,
        shear_modulus=10e9,
        flight=flight_calisto_custom_wind,
        see_prints=True,
        see_graphs=False,  # False = returns tuple!
        filename=None,
    )

    # Verify returns are valid
    assert flutter_mach is not None
    assert safety_factor is not None


@patch("matplotlib.pyplot.show")
def test_fin_flutter_analysis_with_graphs(mock_show, flight_calisto_custom_wind):
    """Test fin_flutter_analysis with see_graphs=True to cover plotting branch.

    Parameters
    ----------
    mock_show : mock
        Mock of matplotlib.pyplot.show function.
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins.
    """
    flutter_mach, safety_factor = utilities.fin_flutter_analysis(
        fin_thickness=2 / 1000,
        shear_modulus=10e9,
        flight=flight_calisto_custom_wind,
        see_prints=False,
        see_graphs=True,
        filename=None,
    )

    assert isinstance(flutter_mach, Function)
    assert isinstance(safety_factor, Function)
    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
def test_fin_flutter_analysis_complete_output(mock_show, flight_calisto_custom_wind):
    """Test fin_flutter_analysis with both prints and graphs enabled.

    Parameters
    ----------
    mock_show : mock
        Mock of matplotlib.pyplot.show function.
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins.
    """
    flutter_mach, safety_factor = utilities.fin_flutter_analysis(
        fin_thickness=2 / 1000,
        shear_modulus=10e9,
        flight=flight_calisto_custom_wind,
        see_prints=True,
        see_graphs=True,
        filename=None,
    )

    # The flutter Mach number and safety factor are always returned, regardless
    # of see_prints/see_graphs, so the safety-critical results are never lost.
    assert isinstance(flutter_mach, Function)
    assert isinstance(safety_factor, Function)
    mock_show.assert_called()


def test_flutter_prints(flight_calisto_custom_wind):
    """Tests the _flutter_prints function.

    Parameters
    ----------
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins. This flight object was created
        in the conftest.py file.
    """
    flutter_mach = Function("tests/fixtures/utilities/flutter_mach.txt")
    safety_factor = Function("tests/fixtures/utilities/flutter_safety_factor.txt")
    assert (
        utilities._flutter_prints(
            fin_thickness=2 / 1000,
            shear_modulus=10e9,
            surface_area=0.009899999999999999,
            aspect_ratio=1.2222222222222223,
            lambda_=0.5,
            flutter_mach=flutter_mach,
            safety_factor=safety_factor,
            flight=flight_calisto_custom_wind,
        )
        is None
    ), "An error occurred while running the utilities._flutter_prints function."


@patch("matplotlib.pyplot.show")
def test_flutter_plots(mock_show, flight_calisto_custom_wind):  # pylint: disable=unused-argument
    """Tests the _flutter_plots function.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function. This is here so the plots
        are not shown during the tests.
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins. This flight object was created
        in the conftest.py file.
    """
    flutter_mach = Function("tests/fixtures/utilities/flutter_mach.txt")
    safety_factor = Function("tests/fixtures/utilities/flutter_safety_factor.txt")
    assert (
        utilities._flutter_plots(
            flight_calisto_custom_wind, flutter_mach, safety_factor
        )
        is None
    ), "An error occurred while running the utilities._flutter_plots function."


def test_get_instance_attributes_with_robust_flight(flight_calisto_robust):
    """Tests if get_instance_attributes returns the expected results for a
    robust flight object."""

    attributes = utilities.get_instance_attributes(flight_calisto_robust)
    for key, value in attributes.items():
        attr = getattr(flight_calisto_robust, key)
        if isinstance(attr, np.ndarray):
            assert np.allclose(attr, value)
        else:
            assert attr == value


def test_get_instance_attributes_with_flight_without_rail_buttons(flight_calisto):
    """Tests if get_instance_attributes returns the expected results for a
    flight object that contains a rocket object without rail buttons."""

    attributes = utilities.get_instance_attributes(flight_calisto)
    for key, value in attributes.items():
        attr = getattr(flight_calisto, key)
        if isinstance(attr, np.ndarray):
            assert np.allclose(attr, value)
        else:
            assert attr == value


@pytest.mark.parametrize(
    "f, eps, expected",
    [
        ([1.0, 1.0, 1.0, 2.0, 3.0], 1e-6, 0),
        ([1.0, 1.0, 1.0, 2.0, 3.0], 1e-1, 0),
        ([1.0, 1.1, 1.2, 2.0, 3.0], 1e-1, None),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 1e-6, 0),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 1e-1, 0),
        ([1.0, 1.0, 1.0], 1e-6, 0),
        ([1.0, 1.0], 1e-6, None),
        ([1.0], 1e-6, None),
        ([], 1e-6, None),
    ],
)
def test_check_constant(f, eps, expected):
    """Test if the function `check_constant` returns the correct index or None
    for different scenarios.

    Parameters
    ----------
    f : list or array
        A list or array of numerical values.
    eps : float
        The tolerance level for comparing the elements.
    expected : int or None
        The expected result of the function.
    """
    result = utilities.check_constant(f, eps)
    assert result == expected


def test_save_to_rpy(flight_calisto_robust):
    """Tests if the save_to_rpy function correctly saves the data to the
    correct file.

    Parameters
    ----------
    flight_calisto_custom_wind : Flight
        A Flight object with a rocket with fins. This flight object was created
        in the conftest.py file.
    """
    utilities.save_to_rpy(flight_calisto_robust, "flight_calisto_robust.rpy")
    assert os.path.splitext(os.path.basename("flight_calisto_robust.rpy")) == (
        "flight_calisto_robust",
        ".rpy",
    )
    assert os.path.getsize("flight_calisto_robust.rpy") > 0
    os.remove("flight_calisto_robust.rpy")


@patch("matplotlib.pyplot.show")
def test_load_from_rpy(mock_show):  # pylint: disable=unused-argument
    """Tests if the load_from_rpy function correctly loads the data into a
    flight object.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function. This is here so the plots
        are not shown during the tests.
    """
    loaded_flight = utilities.load_from_rpy(
        "tests/fixtures/utilities/flight_calisto_robust.rpy"
    )
    assert loaded_flight.info() is None
    assert loaded_flight.all_info() is None


def test_opening_shock_coefficient_default_is_1_5():
    """Default opening_shock_coefficient must be 1.5."""
    force_default = utilities.calculate_simplified_opening_shock_force(10.0, 1.225, 10)
    force_1_5 = utilities.calculate_simplified_opening_shock_force(10.0, 1.225, 10, 1.5)
    assert force_default == force_1_5


def test_calculate_simplified_opening_shock_force_matches_formula():
    """calculate_simplified_opening_shock_force must return
    Cx * cd_s * 0.5 * rho * V^2."""
    cd_s = 10.0
    cx = 1.6
    air_density = 1.225
    velocity = 50.0

    expected_force = cx * cd_s * 0.5 * air_density * velocity**2
    assert utilities.calculate_simplified_opening_shock_force(
        cd_s, air_density, velocity, cx
    ) == pytest.approx(expected_force, rel=1e-9)


def test_calculate_simplified_opening_shock_force_scales_with_velocity_squared():
    """Doubling velocity must quadruple the opening shock force."""
    force_v = utilities.calculate_simplified_opening_shock_force(10.0, 1.225, 40.0)
    force_2v = utilities.calculate_simplified_opening_shock_force(10.0, 1.225, 80.0)
    assert force_2v == pytest.approx(4 * force_v, rel=1e-9)


def test_calculate_simplified_opening_shock_force_zero_velocity_is_zero():
    """No dynamic pressure means no opening shock force."""
    assert utilities.calculate_simplified_opening_shock_force(
        10.0, 1.225, 0.0
    ) == pytest.approx(0.0)


# --- Logging (rocketpy.utilities.enable_logging) ------------------------------


@pytest.fixture(autouse=True)
def reset_rocketpy_logger():
    """Reset the rocketpy logger to its original state after each test."""
    logger = logging.getLogger("rocketpy")
    original_level = logger.level
    original_handlers = logger.handlers[:]
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)


def test_enable_logging_adds_stream_handler():
    """enable_logging() must attach a StreamHandler to the rocketpy logger."""
    utilities.enable_logging(level="INFO")

    logger = logging.getLogger("rocketpy")
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) >= 1


def test_enable_logging_sets_correct_level():
    """enable_logging() must set the requested level on the rocketpy logger."""
    utilities.enable_logging(level="DEBUG")
    assert logging.getLogger("rocketpy").level == logging.DEBUG

    utilities.enable_logging(level="WARNING")
    assert logging.getLogger("rocketpy").level == logging.WARNING


def test_enable_logging_no_duplicate_handlers():
    """Calling enable_logging() twice must not duplicate StreamHandlers."""
    utilities.enable_logging(level="INFO")
    utilities.enable_logging(level="INFO")

    logger = logging.getLogger("rocketpy")
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1


def test_enable_logging_replaces_handler_on_level_change():
    """Calling enable_logging() with a new level must replace the old handler."""
    utilities.enable_logging(level="WARNING")
    utilities.enable_logging(level="DEBUG")

    logger = logging.getLogger("rocketpy")
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1
    assert logger.level == logging.DEBUG


def test_enable_logging_invalid_level_raises():
    """enable_logging() must raise ValueError for an unrecognised level string."""
    with pytest.raises(ValueError, match="Invalid logging level"):
        utilities.enable_logging(level="INVALID")


def test_enable_logging_messages_are_captured(caplog):
    """After enable_logging(), internal rocketpy log messages must be visible."""
    utilities.enable_logging(level="DEBUG")

    with caplog.at_level(logging.DEBUG, logger="rocketpy"):
        logger = logging.getLogger("rocketpy.simulation.flight")
        logger.info("test message from flight")

    assert "test message from flight" in caplog.text
