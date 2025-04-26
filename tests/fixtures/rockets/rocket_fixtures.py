import numpy as np
import pytest

from rocketpy import Rocket


@pytest.fixture
def calisto_motorless():
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same rocket that has been used in the getting started guide for years
    but without a motor.

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    return calisto


@pytest.fixture
def calisto(calisto_motorless, cesaroni_m1670):  # old name: rocket
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same rocket that has been used in the getting started guide for
    years. The Calisto rocket is the Projeto Jupiter's project launched at the
    2019 Spaceport America Cup.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    cesaroni_m1670 : rocketpy.SolidMotor
        An object of the SolidMotor class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(cesaroni_m1670, position=-1.373)
    return calisto


@pytest.fixture
def calisto_nose_to_tail(cesaroni_m1670):
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same as the calisto fixture, but with the coordinate system
    orientation set to "nose_to_tail" instead of "tail_to_nose". This allows to
    check if the coordinate system orientation is being handled correctly in
    the code.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        An object of the SolidMotor class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        The Calisto rocket with the coordinate system orientation set to
        "nose_to_tail". Rail buttons are already set, as well as the motor.
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )
    calisto.add_motor(cesaroni_m1670, position=1.373)
    calisto.set_rail_buttons(
        upper_button_position=-0.082,
        lower_button_position=0.618,
        angular_position=0,
    )
    return calisto


@pytest.fixture
def calisto_liquid_modded(calisto_motorless, liquid_motor):
    """Create a simple object of the Rocket class to be used in the tests. This
    is an example of the Calisto rocket with a liquid motor.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    liquid_motor : rocketpy.LiquidMotor

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(liquid_motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_hybrid_modded(calisto_motorless, hybrid_motor):
    """Create a simple object of the Rocket class to be used in the tests. This
    is an example of the Calisto rocket with a hybrid motor.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    hybrid_motor : rocketpy.HybridMotor

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(hybrid_motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_robust(
    calisto,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_rail_buttons,  # pylint: disable=unused-argument
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto fixture, but with
    all the aerodynamic surfaces and parachutes added. This avoids repeating the
    same code in all tests.

    Parameters
    ----------
    calisto : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    calisto_nose_cone : rocketpy.NoseCone
        The nose cone of the Calisto rocket. This is a pytest fixture too.
    calisto_tail : rocketpy.Tail
        The boat tail of the Calisto rocket. This is a pytest fixture too.
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        The trapezoidal fins of the Calisto rocket. This is a pytest fixture
    calisto_rail_buttons : rocketpy.RailButtons
        The rail buttons of the Calisto rocket. This is a pytest fixture too.
    calisto_main_chute : rocketpy.Parachute
        The main parachute of the Calisto rocket. This is a pytest fixture too.
    calisto_drogue_chute : rocketpy.Parachute
        The drogue parachute of the Calisto rocket. This is a pytest fixture

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    # we follow this format: calisto.add_surfaces(surface, position)
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.add_surfaces(calisto_trapezoidal_fins, -1.168)
    # calisto.add_surfaces(calisto_rail_buttons, -1.168)
    # TODO: if I use the line above, the calisto won't have rail buttons attribute
    #       we need to apply a check in the add_surfaces method to set the rail buttons
    calisto.set_rail_buttons(
        upper_button_position=0.082,
        lower_button_position=-0.618,
        angular_position=0,
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)
    return calisto


@pytest.fixture
def calisto_nose_to_tail_robust(
    calisto_nose_to_tail,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Calisto with nose to tail coordinate system orientation. This is the same
    as calisto_robust, but with the coordinate system orientation set to
    "nose_to_tail"."""
    csys = -1
    # we follow this format: calisto.add_surfaces(surface, position)
    calisto_nose_to_tail.add_surfaces(calisto_nose_cone, 1.160 * csys)
    calisto_nose_to_tail.add_surfaces(calisto_tail, -1.313 * csys)
    calisto_nose_to_tail.add_surfaces(calisto_trapezoidal_fins, -1.168 * csys)
    calisto_nose_to_tail.set_rail_buttons(
        upper_button_position=0.082 * csys,
        lower_button_position=-0.618 * csys,
        angular_position=360 - 0,
    )
    calisto_nose_to_tail.parachutes.append(calisto_main_chute)
    calisto_nose_to_tail.parachutes.append(calisto_drogue_chute)
    return calisto_nose_to_tail


@pytest.fixture
def calisto_air_brakes_clamp_on(calisto_robust, controller_function):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto_robust fixture,
    but with air brakes added, with clamping.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture.
    controller_function : function
        A function that controls the air brakes. This is a pytest fixture.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    calisto = calisto_robust
    # remove parachutes
    calisto.parachutes = []
    calisto.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        clamp=True,
    )
    return calisto


@pytest.fixture
def calisto_air_brakes_clamp_off(calisto_robust, controller_function):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto_robust fixture,
    but with air brakes added, without clamping.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture.
    controller_function : function
        A function that controls the air brakes. This is a pytest fixture.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    calisto = calisto_robust
    # remove parachutes
    calisto.parachutes = []
    calisto.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        clamp=False,
    )
    return calisto


@pytest.fixture
def calisto_with_sensors(
    calisto,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    ideal_accelerometer,
    ideal_gyroscope,
    ideal_barometer,
    ideal_gnss,
):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto fixture, but with
    a set of ideal sensors added at the center of dry mass, meaning the readings
    will be the same as the values saved on a Flight object.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.add_surfaces(calisto_trapezoidal_fins, -1.168)
    # double sensors to test using same instance twice
    calisto.add_sensor(ideal_accelerometer, -0.1180124376577797)
    calisto.add_sensor(ideal_accelerometer, -0.1180124376577797)
    calisto.add_sensor(ideal_gyroscope, -0.1180124376577797)
    calisto.add_sensor(ideal_barometer, -0.1180124376577797)
    calisto.add_sensor(ideal_gnss, -0.1180124376577797)
    return calisto


@pytest.fixture  # old name: dimensionless_rocket
def dimensionless_calisto(kg, m, dimensionless_cesaroni_m1670):
    """The dimensionless version of the Calisto rocket. This is the same rocket
    as defined in the calisto fixture, but with all the parameters converted to
    dimensionless values. This allows to check if the dimensions are being
    handled correctly in the code.

    Parameters
    ----------
    kg : numericalunits.kg
        An object of the numericalunits.kg class. This is a pytest fixture too.
    m : numericalunits.m
        An object of the numericalunits.m class. This is a pytest fixture too.
    dimensionless_cesaroni_m1670 : rocketpy.SolidMotor
        The dimensionless version of the Cesaroni M1670 motor. This is a pytest
        fixture too.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    example_rocket = Rocket(
        radius=0.0635 * m,
        mass=14.426 * kg,
        inertia=(6.321 * (kg * m**2), 6.321 * (kg * m**2), 0.034 * (kg * m**2)),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0 * m,
        coordinate_system_orientation="tail_to_nose",
    )
    example_rocket.add_motor(dimensionless_cesaroni_m1670, position=(-1.373) * m)
    return example_rocket


@pytest.fixture
def prometheus_rocket(generic_motor_cesaroni_M1520):
    """Create a simple object of the Rocket class to be used in the tests. This
    is the Prometheus rocket, a rocket documented in the Flight Examples section
    of the RocketPy documentation.

    Parameters
    ----------
    generic_motor_cesaroni_M1520 : GenericMotor
        An object of the GenericMotor class. This is a pytest fixture too.
    """

    def prometheus_cd_at_ma(mach):
        """Gives the drag coefficient of the rocket at a given mach number."""
        if mach <= 0.15:
            return 0.422
        elif mach <= 0.45:
            return 0.422 + (mach - 0.15) * (0.38 - 0.422) / (0.45 - 0.15)
        elif mach <= 0.77:
            return 0.38 + (mach - 0.45) * (0.32 - 0.38) / (0.77 - 0.45)
        elif mach <= 0.82:
            return 0.32 + (mach - 0.77) * (0.3 - 0.32) / (0.82 - 0.77)
        elif mach <= 0.88:
            return 0.3 + (mach - 0.82) * (0.3 - 0.3) / (0.88 - 0.82)
        elif mach <= 0.94:
            return 0.3 + (mach - 0.88) * (0.32 - 0.3) / (0.94 - 0.88)
        elif mach <= 0.99:
            return 0.32 + (mach - 0.94) * (0.37 - 0.32) / (0.99 - 0.94)
        elif mach <= 1.04:
            return 0.37 + (mach - 0.99) * (0.44 - 0.37) / (1.04 - 0.99)
        elif mach <= 1.24:
            return 0.44 + (mach - 1.04) * (0.43 - 0.44) / (1.24 - 1.04)
        elif mach <= 1.33:
            return 0.43 + (mach - 1.24) * (0.42 - 0.43) / (1.33 - 1.24)
        elif mach <= 1.49:
            return 0.42 + (mach - 1.33) * (0.39 - 0.42) / (1.49 - 1.33)
        else:
            return 0.39

    prometheus = Rocket(
        radius=0.06985,  # 5.5" diameter circle
        mass=13.93,
        inertia=(
            4.87,
            4.87,
            0.05,
        ),
        power_off_drag=prometheus_cd_at_ma,
        power_on_drag=lambda x: prometheus_cd_at_ma(x) * 1.02,  # 5% increase in drag
        center_of_mass_without_motor=0.9549,
        coordinate_system_orientation="tail_to_nose",
    )

    prometheus.set_rail_buttons(0.69, 0.21, 60)

    prometheus.add_motor(motor=generic_motor_cesaroni_M1520, position=0)
    prometheus.add_nose(length=0.742, kind="Von Karman", position=2.229)
    prometheus.add_trapezoidal_fins(
        n=3,
        span=0.13,
        root_chord=0.268,
        tip_chord=0.136,
        position=0.273,
        sweep_length=0.066,
    )
    prometheus.add_parachute(
        "Drogue",
        cd_s=1.6 * np.pi * 0.3048**2,  # Cd = 1.6, D_chute = 24 in
        trigger="apogee",
    )
    prometheus.add_parachute(
        "Main",
        cd_s=2.2 * np.pi * 0.9144**2,  # Cd = 2.2, D_chute = 72 in
        trigger=457.2,  # 1500 ft
    )
    return prometheus
