import copy
import json
import warnings
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import (
    Function,
    GenericSurface,
    LinearGenericSurface,
    NoseCone,
    Rocket,
    SolidMotor,
    TrapezoidalFin,
)
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.motors.empty_motor import EmptyMotor
from rocketpy.motors.motor import Motor


@patch("matplotlib.pyplot.show")
def test_elliptical_fins(mock_show, calisto_robust, calisto_trapezoidal_fins):  # pylint: disable=unused-argument
    test_rocket = calisto_robust
    calisto_robust.aerodynamic_surfaces.remove(calisto_trapezoidal_fins)
    test_rocket.add_elliptical_fins(4, span=0.100, root_chord=0.120, position=-1.168)
    static_margin = test_rocket.static_margin(0)
    assert test_rocket.all_info() is None or not abs(static_margin - 2.30) < 0.01


def test_evaluate_static_margin_assert_cp_equals_cm(dimensionless_calisto):
    rocket = dimensionless_calisto
    rocket.evaluate_center_of_pressure()
    rocket.evaluate_static_margin()

    burn_time = rocket.motor.burn_time

    assert pytest.approx(
        rocket.center_of_mass(0) / (2 * rocket.radius), 1e-8
    ) == pytest.approx(rocket.static_margin(0), 1e-8)
    assert pytest.approx(
        rocket.center_of_mass(burn_time[1]) / (2 * rocket.radius), 1e-8
    ) == pytest.approx(rocket.static_margin(burn_time[1]), 1e-8)
    assert pytest.approx(rocket.total_lift_coeff_der(0), 1e-8) == pytest.approx(0, 1e-8)
    assert pytest.approx(rocket.aerodynamic_center(0), 1e-8) == pytest.approx(0, 1e-8)


def test_static_margin_lazy_until_accessed(calisto_motorless):
    """Static margin must not be discretized until first access."""
    rocket = calisto_motorless

    with patch.object(
        rocket._static_margin,
        "set_discrete",
        wraps=rocket._static_margin.set_discrete,
    ) as mock_set_discrete:
        rocket.add_nose(length=0.55829, kind="ogive", position=1.160)
        mock_set_discrete.assert_not_called()

        static_margin = rocket.static_margin
        assert mock_set_discrete.call_count == 1
        assert isinstance(static_margin, Function)

        # Second access must reuse the cached Function.
        _ = rocket.static_margin(0)
        assert mock_set_discrete.call_count == 1


def test_static_margin_rebuilds_after_adding_surface(calisto):
    """Adding an aero surface invalidates SM; access rebuilds it once."""
    rocket = calisto
    margin_before = rocket.static_margin(0)

    with patch.object(
        rocket._static_margin,
        "set_discrete",
        wraps=rocket._static_margin.set_discrete,
    ) as mock_set_discrete:
        rocket.add_nose(length=0.55829, kind="ogive", position=1.160)
        mock_set_discrete.assert_not_called()

        margin_after = rocket.static_margin(0)
        assert mock_set_discrete.call_count == 1

        _ = rocket.static_margin(0)
        assert mock_set_discrete.call_count == 1

    assert margin_after != pytest.approx(margin_before, abs=1e-6)


def test_aerodynamic_center_lazy_until_accessed(calisto):
    """The center of pressure is only rebuilt when read, and only once."""
    rocket = calisto
    _ = rocket.aerodynamic_center(0)

    with patch.object(
        rocket,
        "evaluate_center_of_pressure",
        wraps=rocket.evaluate_center_of_pressure,
    ) as mock_evaluate:
        rocket.add_nose(length=0.55829, kind="ogive", position=1.160)
        mock_evaluate.assert_not_called()

        _ = rocket.aerodynamic_center(0)
        _ = rocket.aerodynamic_center(0)
        assert mock_evaluate.call_count == 1


def test_add_motor_rebuilds_only_the_margins(calisto_motorless, cesaroni_m1670):
    """A motor moves the center of mass, not the center of pressure."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="ogive", position=1.160)
    margin_before = rocket.static_margin(0)

    with patch.object(
        rocket,
        "evaluate_center_of_pressure",
        wraps=rocket.evaluate_center_of_pressure,
    ) as mock_evaluate:
        rocket.add_motor(cesaroni_m1670, position=-1.373)
        margin_after = rocket.static_margin(0)
        mock_evaluate.assert_not_called()

    assert margin_after != pytest.approx(margin_before, abs=1e-6)


def test_a_kept_static_margin_follows_a_new_surface(calisto):
    """The margin is rebuilt in place, so a kept reference stays current."""
    static_margin = calisto.static_margin
    static_margin_yaw = calisto.static_margin_yaw
    margin_before = static_margin(0)

    calisto.add_nose(length=0.55829, kind="ogive", position=1.160)

    assert calisto.static_margin is static_margin
    assert calisto.static_margin_yaw is static_margin_yaw
    assert static_margin(0) != pytest.approx(margin_before, abs=1e-6)


def test_a_failed_margin_rebuild_is_tried_again(calisto):
    """A rebuild that raises leaves the margins outdated, not half-built."""
    margin_before = calisto.static_margin(0)
    calisto.add_nose(length=0.55829, kind="ogive", position=1.160)
    with patch.object(
        calisto, "evaluate_static_margin", side_effect=RuntimeError("failed")
    ):
        with pytest.raises(RuntimeError):
            _ = calisto.static_margin
    # The next read builds them again, this time with the new surface
    assert calisto.static_margin(0) != pytest.approx(margin_before, abs=1e-6)


def test_evaluate_center_of_pressure_updates_the_margins(
    calisto_robust, calisto_trapezoidal_fins
):
    """Re-evaluating the center of pressure brings the margins up to date."""
    margin_before = calisto_robust.static_margin(0)
    calisto_trapezoidal_fins.tip_chord = 0.080  # changes the fin set in place
    calisto_robust.evaluate_center_of_pressure()
    assert calisto_robust.static_margin(0) != pytest.approx(margin_before, abs=1e-6)


def test_rocket_follows_a_surface_changed_in_place(
    calisto_robust, calisto_trapezoidal_fins
):
    """No evaluate call is needed after changing a surface already added."""
    rocket, fins = calisto_robust, calisto_trapezoidal_fins
    center_before = rocket.aerodynamic_center(0)
    margin_before = rocket.static_margin(0)
    cp_to_cdm_before = [*rocket.surfaces_cp_to_cdm[fins]]

    fins.tip_chord = 0.080

    assert rocket.aerodynamic_center(0) != pytest.approx(center_before)
    assert rocket.static_margin(0) != pytest.approx(margin_before)
    assert [*rocket.surfaces_cp_to_cdm[fins]] != pytest.approx(cp_to_cdm_before)


def test_a_shared_surface_updates_every_rocket(
    calisto, calisto_nose_to_tail, calisto_nose_cone
):
    """A surface added to two rockets updates both when it changes."""
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto_nose_to_tail.add_surfaces(calisto_nose_cone, -1.160)
    centers_before = [
        calisto.aerodynamic_center(0),
        calisto_nose_to_tail.aerodynamic_center(0),
    ]

    calisto_nose_cone.length = 0.8

    centers_after = [
        calisto.aerodynamic_center(0),
        calisto_nose_to_tail.aerodynamic_center(0),
    ]
    assert centers_after[0] != pytest.approx(centers_before[0])
    assert centers_after[1] != pytest.approx(centers_before[1])


def test_cm_eccentricity_moves_the_surfaces_already_added(calisto_robust):
    """Setting the center of mass eccentricity after the surfaces were added must
    give the same surface lever arms as setting it before."""
    rocket = calisto_robust
    before = {
        s: [*rocket.surfaces_cp_to_cdm[s]] for s, _ in rocket.aerodynamic_surfaces
    }

    rocket.add_cm_eccentricity(0.01, -0.02)

    for surface, _ in rocket.aerodynamic_surfaces:
        x, y, z = rocket.surfaces_cp_to_cdm[surface]
        assert x == pytest.approx(before[surface][0] - 0.01 * rocket._csys)
        assert y == pytest.approx(before[surface][1] + 0.02)
        assert z == pytest.approx(before[surface][2])


def test_asymmetry_warning_is_shown_once_per_configuration(calisto):
    """A rocket that is not axisymmetric warns when its aerodynamic center is
    built, and not again until its surfaces change."""
    surface = GenericSurface(
        calisto.area, 2 * calisto.radius, {"cN": lambda alpha: 2 * alpha}
    )
    calisto.add_surfaces(surface, -1.0)
    with pytest.warns(UserWarning, match="not axisymmetric"):
        _ = calisto.aerodynamic_center(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = calisto.aerodynamic_center(0)
        _ = calisto.static_margin(0)


def test_a_canted_fin_moves_its_leading_edge(calisto):
    """Canting a fin already on the rocket places it as if it were added canted:
    the position the user gave is kept, the lever arm follows the cant."""

    geometry = {"root_chord": 0.12, "tip_chord": 0.04, "span": 0.1}
    fin = TrapezoidalFin(0, rocket_radius=0.0635, **geometry)
    canted = TrapezoidalFin(0, rocket_radius=0.0635, cant_angle=5, **geometry)
    reference = copy.deepcopy(calisto)
    reference.add_surfaces(canted, -1.0)
    calisto.add_surfaces(fin, -1.0)
    cp_to_cdm_before = [*calisto.surfaces_cp_to_cdm[fin]]

    fin.cant_angle = 5
    _ = calisto.aerodynamic_center(0)

    position = next(p for s, p in calisto.aerodynamic_surfaces if s is fin)
    assert [*position] == pytest.approx([0.0, 0.0, -1.0])
    assert [*calisto.surfaces_cp_to_cdm[fin]] != pytest.approx(cp_to_cdm_before)
    assert [*calisto.surfaces_cp_to_cdm[fin]] == pytest.approx(
        [*reference.surfaces_cp_to_cdm[canted]]
    )


def test_a_canted_fin_keeps_its_position_through_save_and_load(calisto):
    fin = TrapezoidalFin(
        0, root_chord=0.12, tip_chord=0.04, span=0.1, rocket_radius=0.0635, cant_angle=5
    )
    calisto.add_surfaces(fin, -1.0)
    loaded = json.loads(json.dumps(calisto, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert [*loaded.aerodynamic_surfaces[-1].position] == pytest.approx(
        [*calisto.aerodynamic_surfaces[-1].position]
    )


@pytest.mark.parametrize(
    "k, type_",
    ([2 / 3, "conical"], [0.46469957130675876, "ogive"], [0.563, "lvhaack"]),
)
def test_add_nose_assert_cp_cm_plus_nose(k, type_, calisto, dimensionless_calisto, m):
    calisto.add_nose(length=0.55829, kind=type_, position=1.160)
    cpz = (1.160) - k * 0.55829  # Relative to the center of dry mass
    clalpha = 2

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)

    assert clalpha == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert calisto.aerodynamic_center(0) == pytest.approx(cpz, 1e-8)

    dimensionless_calisto.add_nose(length=0.55829 * m, kind=type_, position=(1.160) * m)
    assert pytest.approx(dimensionless_calisto.static_margin(0), 1e-8) == pytest.approx(
        calisto.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_calisto.static_margin(np.inf), 1e-8
    ) == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.total_lift_coeff_der(0), 1e-8
    ) == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.aerodynamic_center(0) / m, 1e-8
    ) == pytest.approx(calisto.aerodynamic_center(0), 1e-8)


def test_add_tail_assert_cp_cm_plus_tail(calisto, dimensionless_calisto, m):
    calisto.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.313,
    )

    clalpha = -2 * (1 - (0.0635 / 0.0435) ** (-2)) * (0.0635 / (calisto.radius)) ** 2
    cpz = (-1.313) - (0.06 / 3) * (
        1 + (1 - (0.0635 / 0.0435)) / (1 - (0.0635 / 0.0435) ** 2)
    )

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert np.abs(clalpha) == pytest.approx(
        np.abs(calisto.total_lift_coeff_der(0)), 1e-8
    )
    assert calisto.aerodynamic_center(0) == cpz

    dimensionless_calisto.add_tail(
        top_radius=0.0635 * m,
        bottom_radius=0.0435 * m,
        length=0.060 * m,
        position=(-1.313) * m,
    )
    assert pytest.approx(dimensionless_calisto.static_margin(0), 1e-8) == pytest.approx(
        calisto.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_calisto.static_margin(np.inf), 1e-8
    ) == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.total_lift_coeff_der(0), 1e-8
    ) == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.aerodynamic_center(0) / m, 1e-8
    ) == pytest.approx(calisto.aerodynamic_center(0), 1e-8)


@pytest.mark.parametrize(
    "sweep_angle, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [(39.8, 2.51, 3.16, 1.50), (-10, 2.47, 3.21, 1.49), (29.1, 2.50, 3.28, 1.52)],
)
def test_add_trapezoidal_fins_sweep_angle(
    calisto,
    sweep_angle,
    expected_fin_cpz,
    expected_clalpha,
    expected_cpz_cm,
    calisto_nose_cone,
):
    # Reference values from OpenRocket
    calisto.add_surfaces(calisto_nose_cone, Vector([0, 0, 1.160]))
    fin_set = calisto.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_angle=sweep_angle,
        position=-1.064,
    )

    # Check center of pressure
    translate = 1.160
    cpz = -1.300 - fin_set.cpz
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = fin_set.clalpha(0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - calisto.aerodynamic_center(0) == pytest.approx(
        expected_cpz_cm, 0.01
    )


@pytest.mark.parametrize(
    "sweep_length, expected_fin_cpz, expected_clalpha, expected_cpz_cm",
    [
        (0.075, 2.28, 3.16, 1.502),
        (-0.0159, 2.24, 3.21, 1.485),
        (0.05, 2.27, 3.28, 1.513),
    ],
)
def test_add_trapezoidal_fins_sweep_length(
    calisto,
    sweep_length,
    expected_fin_cpz,
    expected_clalpha,
    expected_cpz_cm,
    calisto_nose_cone,
):
    # Reference values from OpenRocket
    calisto.add_surfaces(calisto_nose_cone, Vector([0, 0, 1.160]))
    fin_set = calisto.add_trapezoidal_fins(
        n=3,
        span=0.090,
        root_chord=0.100,
        tip_chord=0.050,
        sweep_length=sweep_length,
        position=-1.064,
    )

    # Check center of pressure
    translate = 1.160
    cpz = -fin_set.cp[2] - 1.064
    assert translate - cpz == pytest.approx(expected_fin_cpz, 0.01)

    # Check lift coefficient derivative
    cl_alpha = fin_set.clalpha(0.0)
    assert cl_alpha == pytest.approx(expected_clalpha, 0.01)

    # Check rocket's center of pressure (just double checking)
    assert translate - calisto.aerodynamic_center(0) == pytest.approx(
        expected_cpz_cm, 0.01
    )

    assert isinstance(calisto.aerodynamic_surfaces[0].component, NoseCone)


def test_add_fins_assert_cp_cm_plus_fins(calisto, dimensionless_calisto, m):
    calisto.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
    )

    cpz = (-1.168) - (
        ((0.120 - 0.040) / 3) * ((0.120 + 2 * 0.040) / (0.120 + 0.040))
        + (1 / 6) * (0.120 + 0.040 - 0.120 * 0.040 / (0.120 + 0.040))
    )

    clalpha = (4 * 4 * (0.1 / (2 * calisto.radius)) ** 2) / (
        1
        + np.sqrt(
            1
            + (2 * np.sqrt((0.12 / 2 - 0.04 / 2) ** 2 + 0.1**2) / (0.120 + 0.040)) ** 2
        )
    )
    clalpha *= 1 + calisto.radius / (0.1 + calisto.radius)

    static_margin_initial = (calisto.center_of_mass(0) - cpz) / (2 * calisto.radius)
    assert static_margin_initial == pytest.approx(calisto.static_margin(0), 1e-8)

    static_margin_final = (calisto.center_of_mass(np.inf) - cpz) / (2 * calisto.radius)
    assert static_margin_final == pytest.approx(calisto.static_margin(np.inf), 1e-8)

    assert np.abs(clalpha) == pytest.approx(
        np.abs(calisto.total_lift_coeff_der(0)), 1e-8
    )
    assert calisto.aerodynamic_center(0) == pytest.approx(cpz, 1e-8)

    dimensionless_calisto.add_trapezoidal_fins(
        4,
        span=0.100 * m,
        root_chord=0.120 * m,
        tip_chord=0.040 * m,
        position=(-1.168) * m,
    )
    assert pytest.approx(dimensionless_calisto.static_margin(0), 1e-8) == pytest.approx(
        calisto.static_margin(0), 1e-8
    )
    assert pytest.approx(
        dimensionless_calisto.static_margin(np.inf), 1e-8
    ) == pytest.approx(calisto.static_margin(np.inf), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.total_lift_coeff_der(0), 1e-8
    ) == pytest.approx(calisto.total_lift_coeff_der(0), 1e-8)
    assert pytest.approx(
        dimensionless_calisto.aerodynamic_center(0) / m, 1e-8
    ) == pytest.approx(calisto.aerodynamic_center(0), 1e-8)


@pytest.mark.parametrize(
    """cdm_position, grain_cm_position, nozzle_position, coord_direction,
    motor_position, expected_motor_cdm, expected_motor_cpp""",
    [
        (0.317, 0.397, 0, "nozzle_to_combustion_chamber", -1.373, -1.056, -0.976),
        (0, 0.08, -0.317, "nozzle_to_combustion_chamber", -1, -1, -0.92),
        (-0.317, -0.397, 0, "combustion_chamber_to_nozzle", -1.373, -1.056, -0.976),
        (0, -0.08, 0.317, "combustion_chamber_to_nozzle", -1, -1, -0.92),
        (1.317, 1.397, 1, "nozzle_to_combustion_chamber", -2.373, -1.056, -0.976),
    ],
)
def test_add_motor_coordinates(
    calisto_motorless,
    cdm_position,
    grain_cm_position,
    nozzle_position,
    coord_direction,
    motor_position,
    expected_motor_cdm,
    expected_motor_cpp,
):
    """Test the method add_motor and related position properties in a Rocket
    instance.

    This test checks the correctness of the `add_motor` method and the computed
    `motor_center_of_dry_mass_position` and `center_of_propellant_position`
    properties in the `Rocket` class using various parameters related to the
    motor's position, nozzle's position, and other related coordinates.
    Different scenarios are tested using parameterization, checking scenarios
    moving from the nozzle to the combustion chamber and vice versa, and with
    various specific physical and geometrical characteristics of the motor.

    Parameters
    ----------
    calisto_motorless : Rocket instance
        A predefined instance of a Rocket without a motor, used as a base for testing.
    cdm_position : float
        Position of the center of dry mass of the motor.
    grain_cm_position : float
        Position of the grains' center of mass.
    nozzle_position : float
        Position of the nozzle.
    coord_direction : str
        Direction for coordinate system orientation;
        it can be "nozzle_to_combustion_chamber" or "combustion_chamber_to_nozzle".
    motor_position : float
        Position where the motor should be added to the rocket.
    expected_motor_cdm : float
        Expected position of the motor's center of dry mass after addition.
    expected_motor_cpp : float
        Expected position of the center of propellant after addition.
    """
    example_motor = SolidMotor(
        thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=0,
        dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=cdm_position,
        nozzle_position=nozzle_position,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=grain_cm_position,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation=coord_direction,
    )
    calisto = calisto_motorless
    calisto.add_motor(example_motor, position=motor_position)

    calculated_motor_cdm = calisto.motor_center_of_dry_mass_position
    calculated_motor_cpp = calisto.center_of_propellant_position

    assert pytest.approx(expected_motor_cdm) == calculated_motor_cdm
    assert pytest.approx(expected_motor_cpp) == calculated_motor_cpp(0)


def test_add_cm_eccentricity_assert_properties_set(calisto):
    calisto.add_cm_eccentricity(x=4, y=5)

    assert calisto.cp_eccentricity_x == -4
    assert calisto.cp_eccentricity_y == -5

    assert calisto.thrust_eccentricity_x == -4
    assert calisto.thrust_eccentricity_y == -5


def test_add_thrust_eccentricity_assert_properties_set(calisto):
    calisto.add_thrust_eccentricity(x=4, y=5)

    assert calisto.thrust_eccentricity_x == 4
    assert calisto.thrust_eccentricity_y == 5


def test_add_cp_eccentricity_assert_properties_set(calisto):
    calisto.add_cp_eccentricity(x=4, y=5)

    assert calisto.cp_eccentricity_x == 4
    assert calisto.cp_eccentricity_y == 5


def test_add_motor(calisto_motorless, cesaroni_m1670):
    """Tests the add_motor method of the Rocket class.
    Both with respect to return instances and expected behaviour.
    Parameters
    ----------
    calisto_motorless : Rocket instance
        A predefined instance of a Rocket without a motor, used as a base for testing.
    cesaroni_m1670 : rocketpy.SolidMotor
        Cesaroni M1670 motor
    """

    assert isinstance(calisto_motorless.motor, EmptyMotor)
    center_of_mass_motorless = calisto_motorless.center_of_mass
    calisto_motorless.add_motor(cesaroni_m1670, 0)

    assert isinstance(calisto_motorless.motor, Motor)
    center_of_mass_with_motor = calisto_motorless.center_of_mass

    assert center_of_mass_motorless is not center_of_mass_with_motor


def test_check_missing_all_components(calisto_motorless):
    """Tests the _check_missing_components method for a Rocket with no components."""
    with pytest.warns(UserWarning) as record:
        calisto_motorless._check_missing_components()

    assert len(record) == 1
    msg = str(record[0].message)
    assert "motor" in msg
    assert "aerodynamic surfaces" in msg


def test_check_missing_some_components(calisto):
    """Tests the _check_missing_components method for a Rocket missing some components."""
    calisto.aerodynamic_surfaces = []

    with pytest.warns(UserWarning) as record:
        calisto._check_missing_components()

    assert len(record) == 1
    msg = str(record[0].message)
    assert "aerodynamic surfaces" in msg


def test_check_missing_no_components_missing(calisto_robust):
    """Tests the _check_missing_components method for a complete Rocket."""
    # Catch all warnings that occur inside this 'with' block.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calisto_robust._check_missing_components()
        # For a complete rocket, this method should NOT issue any warnings.
    assert len(w) == 0


def test_set_rail_button(calisto):
    rail_buttons = calisto.set_rail_buttons(0.2, -0.5, 30)
    # assert buttons_distance
    assert (
        rail_buttons.buttons_distance
        == calisto.rail_buttons[0].component.buttons_distance
        == pytest.approx(0.7, 1e-12)
    )
    # assert buttons position on rocket
    assert calisto.rail_buttons[0].position.z == -0.5
    # assert angular position
    assert (
        rail_buttons.angular_position
        == calisto.rail_buttons[0].component.angular_position
        == 30
    )
    # assert upper button position
    assert calisto.rail_buttons[0].component.buttons_distance + calisto.rail_buttons[
        0
    ].position.z == pytest.approx(0.2, 1e-12)


def test_evaluate_total_mass(calisto_motorless):
    """Tests the evaluate_total_mass method of the Rocket class.
    Both with respect to return instances and expected behaviour.

    Parameters
    ----------
    calisto_motorless : Rocket instance
        A predefined instance of a Rocket without a motor, used as a base for testing.
    """
    assert isinstance(calisto_motorless.evaluate_total_mass(), Function)


def test_evaluate_center_of_mass(calisto):
    """Tests the evaluate_center_of_mass method of the Rocket class.
    Both with respect to return instances and expected behaviour.
    Parameters
    ----------
    calisto : Rocket instance
        A predefined instance of the calisto Rocket with a motor, used as a base for testing.
    """
    assert isinstance(calisto.evaluate_center_of_mass(), Function)


def test_evaluate_nozzle_to_cdm(calisto):
    expected_distance = 1.255
    atol = 1e-3  # Equivalent to 1mm
    assert pytest.approx(expected_distance, atol) == calisto.nozzle_to_cdm
    # Test if calling the function returns the same result
    res = calisto.evaluate_nozzle_to_cdm()
    assert pytest.approx(expected_distance, atol) == res


def test_evaluate_nozzle_gyration_tensor(calisto):
    expected_gyration_tensor = np.array(
        [[1.5752660, 0, 0], [0, 1.5752660, 0], [0, 0, 0.0005445]]
    )
    atol = 1e-3 * 1e-2 * 1e-2  # Equivalent to 1g * 1cm^2
    assert np.allclose(
        expected_gyration_tensor, np.array(calisto.nozzle_gyration_tensor), atol=atol
    )
    # Test if calling the function returns the same result
    res = calisto.evaluate_nozzle_gyration_tensor()
    assert np.allclose(expected_gyration_tensor, np.array(res), atol=atol)


def test_evaluate_com_to_cdm_function(calisto):
    atol = 1e-3  # Equivalent to 1mm
    assert np.allclose(
        (calisto.center_of_dry_mass_position - calisto.center_of_mass).source,
        calisto.com_to_cdm_function.source,
        atol=atol,
    )


def test_get_inertia_tensor_at_time(calisto):
    # Expected values (for t = 0)
    # TODO: compute these values by hand or using CAD.
    I_11 = 10.516647727227216
    I_22 = 10.516647727227216
    I_33 = 0.0379420341586346

    # Set tolerance threshold
    atol = 1e-5

    # Get inertia tensor at t = 0
    inertia_tensor = calisto.get_inertia_tensor_at_time(0)

    # Check if the values are close to the expected ones
    assert pytest.approx(I_11, atol) == inertia_tensor.x[0]
    assert pytest.approx(I_22, atol) == inertia_tensor.y[1]
    assert pytest.approx(I_33, atol) == inertia_tensor.z[2]
    # Check if products of inertia are zero
    assert pytest.approx(0, atol) == inertia_tensor.x[1]
    assert pytest.approx(0, atol) == inertia_tensor.x[2]
    assert pytest.approx(0, atol) == inertia_tensor.y[0]
    assert pytest.approx(0, atol) == inertia_tensor.y[2]
    assert pytest.approx(0, atol) == inertia_tensor.z[0]
    assert pytest.approx(0, atol) == inertia_tensor.z[1]


def test_get_inertia_tensor_derivative_at_time(calisto):
    # Expected values (for t = 2s)
    # TODO: compute these values by hand or using CAD.
    I_11_dot = -0.7164327431607691
    I_22_dot = -0.7164327431607691
    I_33_dot = -0.0006714936623050

    # Set tolerance threshold
    atol = 1e-3

    # Get inertia tensor at t = 2s
    inertia_tensor = calisto.get_inertia_tensor_derivative_at_time(2)

    # Check if the values are close to the expected ones
    assert pytest.approx(I_11_dot, atol) == inertia_tensor.x[0]
    assert pytest.approx(I_22_dot, atol) == inertia_tensor.y[1]
    assert pytest.approx(I_33_dot, atol) == inertia_tensor.z[2]
    # Check if products of inertia are zero
    assert pytest.approx(0, atol) == inertia_tensor.x[1]
    assert pytest.approx(0, atol) == inertia_tensor.x[2]
    assert pytest.approx(0, atol) == inertia_tensor.y[0]
    assert pytest.approx(0, atol) == inertia_tensor.y[2]
    assert pytest.approx(0, atol) == inertia_tensor.z[0]
    assert pytest.approx(0, atol) == inertia_tensor.z[1]


def test_add_thrust_eccentricity(calisto):
    """Test add_thrust_eccentricity method of the Rocket class."""
    calisto.add_thrust_eccentricity(0.1, 0.1)
    assert calisto.thrust_eccentricity_x == 0.1
    assert calisto.thrust_eccentricity_y == 0.1


def test_add_cm_eccentricity(calisto):
    """Test add_cm_eccentricity method of the Rocket class."""
    calisto.add_cm_eccentricity(-0.1, -0.1)
    assert calisto.cp_eccentricity_x == 0.1
    assert calisto.cp_eccentricity_y == 0.1
    assert calisto.thrust_eccentricity_x == 0.1
    assert calisto.thrust_eccentricity_y == 0.1


class TestAddSurfaces:
    """Test the add_surfaces method with different nose cone configurations.
    More specifically, this will check the static margin of the rocket with
    different nose cone configurations."""

    @pytest.fixture(autouse=True)
    def setup(self, calisto):
        self.calisto = calisto
        self.length = 0.55829
        self.kind = "vonkarman"
        self.position = 1.16
        self.bluffness = 0
        self.base_radius = 0.0635
        self.rocket_radius = 0.0635

    def test_add_surfaces_base_equals_rocket_radius(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=self.base_radius,
            bluffness=self.bluffness,
            rocket_radius=self.rocket_radius,
            name="Nose Cone 1",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(1, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)

    def test_add_surfaces_base_half_rocket_radius(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=self.base_radius / 2,
            bluffness=self.bluffness,
            rocket_radius=self.rocket_radius,
            name="Nose Cone 2",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(0.5, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)

    def test_add_surfaces_base_radius_none(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=None,
            bluffness=self.bluffness,
            rocket_radius=self.rocket_radius * 2,
            name="Nose Cone 3",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(1, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)

    def test_add_surfaces_rocket_radius_none(self):
        nose = NoseCone(
            self.length,
            self.kind,
            base_radius=self.base_radius,
            bluffness=self.bluffness,
            rocket_radius=None,
            name="Nose Cone 4",
        )
        self.calisto.add_surfaces(nose, self.position)
        assert nose.radius_ratio == pytest.approx(1, 1e-8)
        assert self.calisto.static_margin(0) == pytest.approx(-8.9053, 0.01)


def test_coordinate_system_orientation(
    calisto_nose_cone, cesaroni_m1670, calisto_trapezoidal_fins
):
    """Test if the coordinate system orientation is working properly. This test
    basically checks if the static margin is the same for the same rocket with
    different coordinate system orientation.

    Parameters
    ----------
    calisto_nose_cone : rocketpy.NoseCone
        Nose cone of the rocket
    cesaroni_m1670 : rocketpy.SolidMotor
        Cesaroni M1670 motor
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        Trapezoidal fins of the rocket
    """
    motor_nozzle_to_combustion_chamber = cesaroni_m1670

    motor_combustion_chamber_to_nozzle = SolidMotor(
        thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=-0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=-0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="combustion_chamber_to_nozzle",
    )

    rocket_tail_to_nose = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket_tail_to_nose.add_motor(motor_nozzle_to_combustion_chamber, position=-1.373)

    rocket_tail_to_nose.add_surfaces(calisto_nose_cone, 1.160)
    rocket_tail_to_nose.add_surfaces(calisto_trapezoidal_fins, -1.168)

    static_margin_tail_to_nose = rocket_tail_to_nose.static_margin

    rocket_nose_to_tail = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )

    rocket_nose_to_tail.add_motor(motor_combustion_chamber_to_nozzle, position=1.373)

    rocket_nose_to_tail.add_surfaces(calisto_nose_cone, -1.160)
    rocket_nose_to_tail.add_surfaces(calisto_trapezoidal_fins, 1.168)

    static_margin_nose_to_tail = rocket_nose_to_tail.static_margin

    assert np.array_equal(static_margin_tail_to_nose, static_margin_nose_to_tail)


def test_drag_csv_header_order_independent_for_multivariable_input(tmp_path):
    """Ensure drag CSV independent-variable columns are interpreted by name.

    This test checks that swapping the order of header-defined variables
    (mach and reynolds) yields equivalent drag interpolation results.
    """

    ordered_csv = tmp_path / "drag_mach_reynolds.csv"
    ordered_csv.write_text(
        "mach,reynolds,cd\n0.5,0.1,0.6\n1.0,0.1,1.1\n0.5,0.2,0.7\n1.0,0.2,1.2\n",
        encoding="utf-8",
    )

    swapped_csv = tmp_path / "drag_reynolds_mach.csv"
    swapped_csv.write_text(
        "reynolds,mach,cd\n0.1,0.5,0.6\n0.1,1.0,1.1\n0.2,0.5,0.7\n0.2,1.0,1.2\n",
        encoding="utf-8",
    )

    rocket_ordered = Rocket(
        radius=0.05,
        mass=1.0,
        inertia=(1.0, 1.0, 1.0),
        power_off_drag=str(ordered_csv),
        power_on_drag=str(ordered_csv),
        center_of_mass_without_motor=0.0,
    )

    rocket_swapped = Rocket(
        radius=0.05,
        mass=1.0,
        inertia=(1.0, 1.0, 1.0),
        power_off_drag=str(swapped_csv),
        power_on_drag=str(swapped_csv),
        center_of_mass_without_motor=0.0,
    )

    drag_ordered = rocket_ordered.power_off_drag_7d(0, 0, 0.8, 0.15, 0, 0, 0)
    drag_swapped = rocket_swapped.power_off_drag_7d(0, 0, 0.8, 0.15, 0, 0, 0)

    # The coefficient is stored at minimal dimension over the present columns,
    # keyed by name, so column order in the header does not matter.
    ordered_csv_function = rocket_ordered.power_off_drag_7d.function
    swapped_csv_function = rocket_swapped.power_off_drag_7d.function

    assert drag_ordered == pytest.approx(0.95)
    assert drag_swapped == pytest.approx(0.95)
    assert drag_swapped == pytest.approx(drag_ordered)
    assert set(rocket_ordered.power_off_drag_7d.depends_on) == {"mach", "reynolds"}
    assert set(rocket_swapped.power_off_drag_7d.depends_on) == {"mach", "reynolds"}
    assert ordered_csv_function.is_regular_grid
    assert swapped_csv_function.is_regular_grid


def test_drag_input_types_supported_for_power_on_and_power_off(tmp_path):
    """Ensure drag input processing accepts all supported input types.

    This test validates that both ``power_off_drag`` and ``power_on_drag``
    accept and correctly evaluate all supported input categories.
    """
    query = (1.0, 0.0, 0.8, 0.15, 0.0, 0.0, 0.0)

    csv_drag = tmp_path / "drag_mach_reynolds.csv"
    csv_drag.write_text(
        "mach,reynolds,cd\n0.5,0.1,0.6\n1.0,0.1,1.1\n0.5,0.2,0.7\n1.0,0.2,1.2\n",
        encoding="utf-8",
    )

    txt_drag = tmp_path / "drag_curve.txt"
    txt_drag.write_text("0.0,0.2\n1.0,0.4\n", encoding="utf-8")

    function_1d = Function(
        lambda mach: 0.2 + mach,
        inputs=["mach"],
        outputs=["cd"],
        interpolation="linear",
    )
    function_7d = Function(
        lambda alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate: (
            mach + reynolds
        ),
        inputs=[
            "alpha",
            "beta",
            "mach",
            "reynolds",
            "pitch_rate",
            "yaw_rate",
            "roll_rate",
        ],
        outputs=["cd"],
        interpolation="linear",
    )

    drag_7d_table = [
        (*coords, float(sum(coords))) for coords in product((0.0, 1.0), repeat=7)
    ]
    drag_7d_query = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    test_cases = [
        ("int", 1, query, 1.0),
        ("float", 0.37, query, 0.37),
        ("csv_path", str(csv_drag), query, 0.95),
        ("txt_path", str(txt_drag), query, 0.36),
        ("function_1d", function_1d, query, 1.0),
        ("function_7d", function_7d, query, 0.95),
        (
            "callable_1d",
            lambda mach: 0.2 + mach,
            query,
            1.0,
        ),
        (
            "callable_7d",
            lambda alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate: (
                mach + reynolds
            ),
            query,
            0.95,
        ),
        ("list_pairs", [[0.0, 0.2], [1.0, 0.4]], query, 0.36),
        ("tuple_pairs", ((0.0, 0.2), (1.0, 0.4)), query, 0.36),
        ("list_7d_entries", drag_7d_table, drag_7d_query, 4.0),
    ]

    for _, drag_input, query_point, expected in test_cases:
        rocket = Rocket(
            radius=0.05,
            mass=1.0,
            inertia=(1.0, 1.0, 1.0),
            power_off_drag=drag_input,
            power_on_drag=drag_input,
            center_of_mass_without_motor=0.0,
        )

        assert rocket.power_off_drag_7d(*query_point) == pytest.approx(expected)
        assert rocket.power_on_drag_7d(*query_point) == pytest.approx(expected)


# Review of 2026-09-26: drag inputs, full-body helpers, positions, length


def _bare_rocket(**kwargs):
    kwargs.setdefault("power_off_drag", 0.5)
    kwargs.setdefault("power_on_drag", 0.5)
    return Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        center_of_mass_without_motor=0,
        **kwargs,
    )


def test_a_one_argument_drag_function_is_read_by_its_name():
    rocket = _bare_rocket(power_off_drag=lambda alpha: 0.4 + alpha**2)
    assert rocket.power_off_drag_7d.depends_on == ("alpha",)
    assert rocket.power_off_drag_7d(0.5, 0, 0.3, 0, 0, 0, 0) == pytest.approx(0.65)
    assert _bare_rocket(
        power_off_drag=lambda m: 0.4 + m
    ).power_off_drag_7d.depends_on == ("mach",)


def test_drag_can_be_assigned_after_construction():
    rocket = _bare_rocket()
    rocket.power_off_drag = lambda mach: 0.9 + mach
    assert rocket.power_off_drag_7d(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(1.4)
    assert rocket.power_off_drag(0.5) == pytest.approx(1.4)


def test_length_can_be_given(calisto_robust):
    rocket = _bare_rocket(length=3.0)
    assert rocket.length == 3.0
    loaded = json.loads(json.dumps(rocket, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert loaded.length == 3.0
    assert calisto_robust.length == pytest.approx(2.533, abs=1e-3)


def test_prints_list_the_lift_slope_of_every_surface(calisto_robust, capsys):
    calisto_robust.prints.all()
    out = capsys.readouterr().out
    section = out[out.index("Lift Coefficient Derivatives") :][:400]
    for surface, _ in calisto_robust.aerodynamic_surfaces:
        assert f"{surface.name} Lift Coefficient Derivative" in section


def test_plots_survive_a_rocket_without_a_length():
    """No surfaces yet, or a point-like full-body model: the percent-of-length
    axis is skipped instead of crashing."""
    _bare_rocket().plots.static_margin()
    rocket = _bare_rocket()
    surface = LinearGenericSurface(
        rocket.area, 2 * rocket.radius, {"cN_alpha": 2, "cY_beta": -2}
    )
    rocket.add_full_body_aerodynamics(surface, position=0.0)
    rocket.plots.static_margin()


def test_the_same_fin_added_twice_keeps_both_positions():
    rocket = _bare_rocket()
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    fin = TrapezoidalFin(0, 0.12, 0.04, 0.1, 0.0635)
    rocket.add_surfaces([fin, fin], [-1.168, -0.5])
    rocket._refresh_aerodynamics()
    assert [p.z for s, p in rocket.aerodynamic_surfaces if s is fin] == [-1.168, -0.5]


def test_set_position_accepts_a_number(calisto_robust):
    surface = calisto_robust.aerodynamic_surfaces[0][0]
    calisto_robust.aerodynamic_surfaces.set_position(surface, 1.0)
    calisto_robust.aerodynamic_center(0.3)
    assert calisto_robust.aerodynamic_surfaces.get_positions()[0].z == 1.0


def test_axisymmetric_linear_surface_fills_in_the_yaw_plane():
    """Pitch-only derivatives with ``axisymmetric=True`` give the same rocket
    in yaw, also from wind-frame derivatives and after saving and loading;
    without it a warning says the rocket has no side force."""
    rocket = _bare_rocket()
    area, diameter = rocket.area, 2 * rocket.radius
    pitch_only = {"cN_alpha": 3.0, "cm_alpha": -1.0, "cN_q": 4.0, "cm_q": -50.0}
    linear = LinearGenericSurface(area, diameter, pitch_only, axisymmetric=True)
    by_hand = LinearGenericSurface(
        area,
        diameter,
        {**pitch_only, "cY_beta": -3.0, "cn_beta": 1.0, "cY_r": 4.0, "cn_r": -50.0},
    )
    state = (0.07, -0.04, 0.5, 0, 0.01, 0.02, 0.003)
    for name in ("cN", "cY", "cm", "cn"):
        assert getattr(linear, name)(*state) == getattr(by_hand, name)(*state)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rocket.add_full_body_aerodynamics(linear, position=0.0)
    assert rocket.aerodynamic_center(0.3) == pytest.approx(
        rocket.aerodynamic_center_yaw(0.3)
    )
    assert rocket.is_axisymmetric

    wind = LinearGenericSurface(
        area, diameter, {"cL_alpha": 2.5, "cD_0": 0.5}, axisymmetric=True
    )
    at_zero = (0.0, 0.0, 0.5, 0, 0, 0, 0)
    assert wind.cN_alpha(*at_zero) == pytest.approx(3.0)
    assert wind.cY_beta(*at_zero) == pytest.approx(-3.0)

    loaded = json.loads(json.dumps(linear, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert sorted(loaded.to_dict()["coefficients"]) == sorted(pitch_only)
    assert loaded.cY(*state) == linear.cY(*state)

    with pytest.warns(UserWarning, match="side force"):
        _bare_rocket().add_full_body_aerodynamics(
            LinearGenericSurface(area, diameter, {"cN_alpha": 3.0}), position=0.0
        )


def test_axisymmetric_linear_surface_refuses_what_singles_out_a_plane():
    """With ``axisymmetric=True`` the yaw plane comes from the pitch plane, so
    a yaw derivative, or a pitch derivative that reads the angle of one plane,
    is refused; the total angle of attack is the same in every plane."""
    with pytest.raises(ValueError, match="cY_beta cannot be given"):
        LinearGenericSurface(
            1.0, 1.0, {"cN_alpha": 2.0, "cY_beta": -2.0}, axisymmetric=True
        )
    with pytest.raises(ValueError, match="cQ_beta cannot be given"):
        LinearGenericSurface(
            1.0, 1.0, {"cL_alpha": 2.0, "cQ_beta": -2.0}, axisymmetric=True
        )
    for name in ("cN_0", "cm_0", "cN_p", "cm_p"):
        with pytest.raises(ValueError, match=f"{name} cannot be given"):
            LinearGenericSurface(
                1.0, 1.0, {"cN_alpha": 2.0, name: 0.1}, axisymmetric=True
            )
    for angle, slope in (
        ("alpha", lambda alpha: 2.0 - alpha**2),
        ("beta", lambda beta: 2.0 - beta**2),
    ):
        with pytest.raises(ValueError, match=f"cN_alpha depends on {angle}"):
            LinearGenericSurface(1.0, 1.0, {"cN_alpha": slope}, axisymmetric=True)

    surface = LinearGenericSurface(
        1.0,
        1.0,
        {"cN_alpha": lambda alpha_total: 9 - 40 * alpha_total**2},
        axisymmetric=True,
    )
    assert surface.cN(0.2, 0, 0, 0, 0, 0, 0) == pytest.approx(1.48)
    assert surface.cY(0, 0.2, 0, 0, 0, 0, 0) == pytest.approx(-1.48)
