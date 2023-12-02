import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from .environment.environment import Environment
from .mathutils.function import Function
from .rocket.aero_surface import TrapezoidalFins
from .simulation.flight import Flight


# TODO: Needs tests
def compute_cd_s_from_drop_test(
    terminal_velocity, rocket_mass, air_density=1.225, g=9.80665
):
    """Returns the parachute's cd_s calculated through its final speed, air
    density in the landing point, the rocket's mass and the force of gravity
    in the landing point.

    Parameters
    ----------
    terminal_velocity : float
        Rocket's speed in m/s when landing.
    rocket_mass : float
        Rocket's dry mass in kg.
    air_density : float, optional
        Air density, in kg/m^3, right before the rocket lands. Default value is
        1.225.
    g : float, optional
        Gravitational acceleration experienced by the rocket and parachute
        during descent in m/s^2. Default value is the standard gravity, 9.80665.

    Returns
    -------
    cd_s : float
        Number equal to drag coefficient times reference area for parachute.

    """

    return 2 * rocket_mass * g / ((terminal_velocity**2) * air_density)


# TODO: Needs tests
def calculate_equilibrium_altitude(
    rocket_mass,
    cd_s,
    z0,
    v0=0,
    env=None,
    eps=1e-3,
    max_step=0.1,
    see_graphs=True,
    g=9.80665,
    estimated_final_time=10,
):
    """Returns a dictionary containing the time, altitude and velocity of the
    system rocket-parachute in which the terminal velocity is reached.


    Parameters
    ----------
    rocket_mass : float
        Rocket's mass in kg.
    cd_s : float
        Number equal to drag coefficient times reference area for parachute.
    z0 : float
        Initial altitude of the rocket in meters.
    v0 : float, optional
        Rocket's initial speed in m/s. Must be negative
    env : Environment, optional
        Environmental conditions at the time of the launch.
    eps : float, optional
        acceptable error in meters.
    max_step: float, optional
        maximum allowed time step size to solve the integration
    see_graphs : boolean, optional
        True if you want to see time vs altitude and time vs speed graphs,
        False otherwise.
    g : float, optional
        Gravitational acceleration experienced by the rocket and parachute
        during descent in m/s^2. Default value is the standard gravity, 9.80665.
    estimated_final_time: float, optional
        Estimative of how much time (in seconds) will spend until vertical
        terminal velocity is reached. Must be positive. Default is 10. It can
        affect the final result if the value is not high enough. Increase the
        estimative in case the final solution is not founded.


    Returns
    -------
    altitude_function: Function
        Altitude as a function of time. Always a Function object.
    velocity_function:
        Vertical velocity as a function of time. Always a Function object.
    final_sol : dictionary
        Dictionary containing the values for time, altitude and speed of
        the rocket when it reaches terminal velocity.
    """
    final_sol = {}

    if not v0 < 0:
        print("Please set a valid negative value for v0")
        return None

    # TODO: Improve docs
    def check_constant(f, eps):
        """_summary_

        Parameters
        ----------
        f : array, list

            _description_
        eps : float
            _description_

        Returns
        -------
        int, None
            _description_
        """
        for i in range(len(f) - 2):
            if abs(f[i + 2] - f[i + 1]) < eps and abs(f[i + 1] - f[i]) < eps:
                return i
        return None

    if env == None:
        environment = Environment(
            latitude=0,
            longitude=0,
            elevation=1000,
            date=(2020, 3, 4, 12),
        )
    else:
        environment = env

    # TODO: Improve docs
    def du(z, u):
        """_summary_

        Parameters
        ----------
        z : float
            _description_
        u : float
            velocity, in m/s, at a given z altitude

        Returns
        -------
        float
            _description_
        """
        return (
            u[1],
            -g + environment.density(z) * ((u[1]) ** 2) * cd_s / (2 * rocket_mass),
        )

    u0 = [z0, v0]

    us = solve_ivp(
        fun=du,
        t_span=(0, estimated_final_time),
        y0=u0,
        vectorized=True,
        method="LSODA",
        max_step=max_step,
    )

    constant_index = check_constant(us.y[1], eps)

    # TODO: Improve docs by explaining what is happening below with constant_index
    if constant_index is not None:
        final_sol = {
            "time": us.t[constant_index],
            "altitude": us.y[0][constant_index],
            "velocity": us.y[1][constant_index],
        }

    altitude_function = Function(
        source=np.array(list(zip(us.t, us.y[0])), dtype=np.float64),
        inputs="Time (s)",
        outputs="Altitude (m)",
        interpolation="linear",
    )

    velocity_function = Function(
        source=np.array(list(zip(us.t, us.y[1])), dtype=np.float64),
        inputs="Time (s)",
        outputs="Vertical Velocity (m/s)",
        interpolation="linear",
    )

    if see_graphs:
        altitude_function()
        velocity_function()

    return altitude_function, velocity_function, final_sol


def fin_flutter_analysis(
    fin_thickness, shear_modulus, flight, see_prints=True, see_graphs=True
):
    """Calculate and plot the Fin Flutter velocity using the pressure profile
    provided by the selected atmospheric model. It considers the Flutter
    Boundary Equation that published in NACA Technical Paper 4197.
    These results are only estimates of a real problem and may not be useful for
    fins made from non-isotropic materials.
    Currently, this function works if only a single set of fins is added,
    otherwise it will use the last set of fins added to the rocket.

    Parameters
    ----------
    fin_thickness : float
        The fin thickness, in meters
    shear_modulus : float
        Shear Modulus of fins' material, must be given in Pascal
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data
    see_prints : boolean, optional
        True if you want to see the prints, False otherwise.
    see_graphs : boolean, optional
        True if you want to see the graphs, False otherwise. If False, the
        function will return the vectors containing the data for the graphs.

    Return
    ------
    None
    """

    # First, we need identify if there is at least a fin set in the rocket
    for aero_surface in flight.rocket.aerodynamic_surfaces:
        if isinstance(aero_surface, TrapezoidalFins):
            # s: surface area; ar: aspect ratio; la: lambda
            root_chord = aero_surface.root_chord
            s = (aero_surface.tip_chord + root_chord) * aero_surface.span / 2
            ar = aero_surface.span * aero_surface.span / s
            la = aero_surface.tip_chord / root_chord

    # This ensures that a fin set was found in the rocket, if not, break
    try:
        s = s
    except NameError:
        print("There is no fin set in the rocket, can't run a Flutter Analysis.")
        return None

    # Calculate the Fin Flutter Mach Number
    flutter_mach = (
        (shear_modulus * 2 * (ar + 2) * (fin_thickness / root_chord) ** 3)
        / (1.337 * (ar**3) * (la + 1) * flight.pressure)
    ) ** 0.5

    safety_factor = _flutter_safety_factor(flight, flutter_mach)

    # Prints everything
    if see_prints:
        _flutter_prints(
            fin_thickness,
            shear_modulus,
            s,
            ar,
            la,
            flutter_mach,
            safety_factor,
            flight,
        )

    # Plots everything
    if see_graphs:
        _flutter_plots(flight, flutter_mach, safety_factor)
        return None
    else:
        return flutter_mach, safety_factor


def _flutter_safety_factor(flight, flutter_mach):
    """Calculates the safety factor for the fin flutter analysis.

    Parameters
    ----------
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data
    flutter_mach : rocketpy.Function
        Mach Number at which the fin flutter occurs. See the
        `fin_flutter_analysis` function for more details.

    Returns
    -------
    rocketpy.Function
        The safety factor for the fin flutter analysis.
    """
    safety_factor = [[t, 0] for t in flutter_mach[:, 0]]
    for i in range(len(flutter_mach)):
        try:
            safety_factor[i][1] = flutter_mach[i][1] / flight.mach_number[i][1]
        except ZeroDivisionError:
            safety_factor[i][1] = np.nan

    # Function needs to remove NaN and Inf values from the source
    safety_factor = np.array(safety_factor)
    safety_factor = safety_factor[~np.isnan(safety_factor).any(axis=1)]
    safety_factor = safety_factor[~np.isinf(safety_factor).any(axis=1)]

    safety_factor = Function(
        source=safety_factor,
        inputs="Time (s)",
        outputs="Fin Flutter Safety Factor",
        interpolation="linear",
    )

    return safety_factor


def _flutter_plots(flight, flutter_mach, safety_factor):
    """Plot the Fin Flutter Mach Number and the Safety Factor for the flutter.

    Parameters
    ----------
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data
    flutter_mach : rocketpy.Function
        Function containing the Fin Flutter Mach Number,
        see fin_flutter_analysis for more details.
    safety_factor : rocketpy.Function
        Function containing the Safety Factor for the fin flutter.
        See fin_flutter_analysis for more details.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(211)
    ax1.plot(
        flutter_mach[:, 0],
        flutter_mach[:, 1],
        label="Fin flutter Mach Number",
    )
    ax1.plot(
        flight.mach_number[:, 0],
        flight.mach_number[:, 1],
        label="Rocket Freestream Speed",
    )
    ax1.set_xlim(0, flight.apogee_time if flight.apogee_time != 0.0 else flight.tFinal)
    ax1.set_title("Fin Flutter Mach Number x Time(s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mach")
    ax1.legend()
    ax1.grid()

    ax2 = plt.subplot(212)
    ax2.plot(safety_factor[:, 0], safety_factor[:, 1])
    ax2.set_xlim(flight.out_of_rail_time, flight.apogee_time)
    ax2.set_ylim(0, 6)
    ax2.set_title("Fin Flutter Safety Factor")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Safety Factor")
    ax2.grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return None


def _flutter_prints(
    fin_thickness,
    shear_modulus,
    s,
    ar,
    la,
    flutter_mach,
    safety_factor,
    flight,
):
    """Prints out the fin flutter analysis results. See fin_flutter_analysis for
    more details.

    Parameters
    ----------
    fin_thickness : float
        The fin thickness, in meters
    shear_modulus : float
        Shear Modulus of fins' material, must be given in Pascal
    s : float
        Fin surface area, in squared meters
    ar : float
        Fin aspect ratio
    la : float
        Fin lambda, defined as the tip_chord / root_chord ratio
    flutter_mach : rocketpy.Function
        The Mach Number at which the fin flutter occurs, considering the
        variation of the speed of sound with altitude. See fin_flutter_analysis
        for more details.
    safety_factor : rocketpy.Function
        The Safety Factor for the fin flutter. Defined as the Fin Flutter Mach
        Number divided by the Freestream Mach Number.
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data

    Returns
    -------
    None
    """
    time_index = np.argmin(flutter_mach[:, 1])
    time_min_mach = flutter_mach[time_index, 0]
    min_mach = flutter_mach[time_index, 1]
    min_vel = min_mach * flight.speed_of_sound(time_min_mach)

    time_index = np.argmin(safety_factor[:, 1])
    time_min_sf = safety_factor[time_index, 0]
    min_sf = safety_factor[time_index, 1]
    altitude_min_sf = flight.z(time_min_sf) - flight.env.elevation

    print("\nFin's parameters")
    print(f"Surface area (S): {s:.4f} m2")
    print(f"Aspect ratio (AR): {ar:.3f}")
    print(f"tip_chord/root_chord ratio = \u03BB = {la:.3f}")
    print(f"Fin Thickness: {fin_thickness:.5f} m")
    print(f"Shear Modulus (G): {shear_modulus:.3e} Pa")

    print("\nFin Flutter Analysis")
    print(f"Minimum Fin Flutter Velocity: {min_vel:.3f} m/s at {time_min_mach:.2f} s")
    print(f"Minimum Fin Flutter Mach Number: {min_mach:.3f} ")
    print(f"Minimum Safety Factor: {min_sf:.3f} at {time_min_sf:.2f} s")
    print(f"Altitude of minimum Safety Factor: {altitude_min_sf:.3f} m (AGL)\n")

    return None


def create_dispersion_dictionary(filename):
    """Creates a dictionary with the rocket data provided by a .csv file.
    File should be organized in four columns: attribute_class, parameter_name,
    mean_value, standard_deviation. The first row should be the header.
    It is advised to use ";" as separator, but "," should work on most of cases.
    The "," separator might cause problems if the data set contains lists where
    the items are separated by commas.

    Parameters
    ----------
    filename : string
        String with the path to the .csv file. The file should follow the
        following structure:

        .. code-block::

            attribute_class; parameter_name; mean_value; standard_deviation;

            environment; ensemble_member; [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];;

            motor; impulse; 1415.15; 35.3;

            motor; burn_time; 5.274; 1;

            motor; nozzle_radius; 0.021642; 0.0005;

            motor; throat_radius; 0.008; 0.0005;

            motor; grain_separation; 0.006; 0.001;

            motor; grain_density; 1707; 50;

    Returns
    -------
    dictionary
        Dictionary with all rocket data to be used in dispersion analysis. The
        dictionary will follow the following structure:

        .. code-block:: python

            analysis_parameters = {
                'environment': {
                    'ensemble_member': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                },
                'motor': {
                    'impulse': (1415.15, 35.3),
                    'burn_time': (5.274, 1),
                    'nozzle_radius': (0.021642, 0.0005),
                    'throat_radius': (0.008, 0.0005),
                    'grain_separation': (0.006, 0.001),
                    'grain_density': (1707, 50),
                    }
            }
    """
    try:
        file = np.genfromtxt(
            filename, usecols=(1, 2, 3), skip_header=1, delimiter=";", dtype=str
        )
    except ValueError:
        warnings.warn(
            f"Error caught: the recommended delimiter is ';'. If using ',' "
            + "instead, be aware that some resources might not work as "
            + "expected if your data set contains lists where the items are "
            + "separated by commas. Please consider changing the delimiter to "
            + "';' if that is the case."
        )
        warnings.warn(traceback.format_exc())
        file = np.genfromtxt(
            filename, usecols=(1, 2, 3), skip_header=1, delimiter=",", dtype=str
        )
    analysis_parameters = dict()
    for row in file:
        if row[0] != "":
            if row[2] == "":
                try:
                    analysis_parameters[row[0].strip()] = float(row[1])
                except ValueError:
                    analysis_parameters[row[0].strip()] = eval(row[1])
            else:
                try:
                    analysis_parameters[row[0].strip()] = (float(row[1]), float(row[2]))
                except ValueError:
                    analysis_parameters[row[0].strip()] = ""
    return analysis_parameters


def apogee_by_mass(flight, min_mass, max_mass, points=10, plot=True):
    """Returns a Function object that estimates the apogee of a rocket given
    its mass (no motor). The function will use the rocket's mass as the
    independent variable and the estimated apogee as the dependent variable.
    The function will use the rocket's environment and inclination to estimate
    the apogee. This is useful when you want to adjust the rocket's mass to
    reach a specific apogee.

    Parameters
    ----------
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data
    min_mass : float
        The minimum value for the rocket's mass to calculate the apogee, given
        in kilograms (kg). This value should be the minimum rocket's mass,
        therefore, a positive value is expected. See the Rocket.mass attribute
        for more details.
    max_mass : float
        The maximum value for the rocket's mass to calculate the apogee, given
        in kilograms (kg). This value should be the maximum rocket's mass,
        therefore, a positive value is expected and it should be higher than the
        min_mass attribute. See the Rocket.mass attribute for more details.
    points : int, optional
        The number of points to calculate the apogee between the mass
        boundaries, by default 10. Increasing this value will refine the
        results, but will also increase the computational time.
    plot : bool, optional
        If True, the function will plot the results, by default True.

    Returns
    -------
    rocketpy.Function
        Function object containing the estimated apogee as a function of the
        rocket's mass (without motor nor propellant).
    """
    rocket = flight.rocket

    def apogee(mass):
        # First we need to modify the rocket's mass and update values
        rocket.mass = float(mass)
        rocket.evaluate_total_mass()
        rocket.evaluate_center_of_mass()
        rocket.evaluate_reduced_mass()
        rocket.evaluate_thrust_to_weight()
        rocket.evaluate_center_of_pressure()
        rocket.evaluate_static_margin()
        # Then we can run the flight simulation
        test_flight = Flight(
            rocket=rocket,
            environment=flight.env,
            rail_length=flight.rail_length,
            inclination=flight.inclination,
            heading=flight.heading,
            terminate_on_apogee=True,
        )
        return test_flight.apogee - flight.env.elevation

    x = np.linspace(min_mass, max_mass, points)
    y = np.array([apogee(m) for m in x])
    source = np.array(list(zip(x, y)), dtype=np.float64)

    retfunc = Function(
        source, inputs="Rocket Mass without motor (kg)", outputs="Apogee AGL (m)"
    )
    if plot:
        retfunc.plot(min_mass, max_mass, points)
    return retfunc


def liftoff_speed_by_mass(flight, min_mass, max_mass, points=10, plot=True):
    """Returns a Function object that estimates the liftoff speed of a rocket
    given its mass (without motor). The function will use the rocket's mass as
    the independent variable and the estimated liftoff speed as the dependent
    variable. The function will use the rocket's environment and inclination
    to estimate the liftoff speed. This is useful when you want to adjust the
    rocket's mass to reach a specific liftoff speed.

    Parameters
    ----------
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data
    min_mass : float
        The minimum value for the rocket's mass to calculate the out of rail
        speed, given in kilograms (kg). This value should be the minimum
        rocket's mass, therefore, a positive value is expected. See the
        Rocket.mass attribute for more details.
    max_mass : float
        The maximum value for the rocket's mass to calculate the out of rail
        speed, given in kilograms (kg). This value should be the maximum
        rocket's mass, therefore, a positive value is expected and it should be
        higher than the min_mass attribute. See the Rocket.mass attribute for
        more details.
    points : int, optional
        The number of points to calculate the liftoff speed between the mass
        boundaries, by default 10. Increasing this value will refine the
        results, but will also increase the computational time.
    plot : bool, optional
        If True, the function will plot the results, by default True.

    Returns
    -------
    rocketpy.Function
        Function object containing the estimated liftoff speed as a function of
        the rocket's mass (without motor nor propellant).
    """
    rocket = flight.rocket

    def liftoff_speed(mass):
        # First we need to modify the rocket's mass and update values
        rocket.mass = float(mass)
        rocket.evaluate_total_mass()
        rocket.evaluate_center_of_mass()
        rocket.evaluate_reduced_mass()
        rocket.evaluate_thrust_to_weight()
        rocket.evaluate_center_of_pressure()
        rocket.evaluate_static_margin()
        # Then we can run the flight simulation
        test_flight = Flight(
            rocket=rocket,
            environment=flight.env,
            rail_length=flight.rail_length,
            inclination=flight.inclination,
            heading=flight.heading,
            terminate_on_apogee=True,
        )
        return test_flight.out_of_rail_velocity

    x = np.linspace(min_mass, max_mass, points)
    y = np.array([liftoff_speed(m) for m in x])
    source = np.array(list(zip(x, y)), dtype=np.float64)

    retfunc = Function(
        source,
        inputs="Rocket Mass without motor (kg)",
        outputs="Out of Rail Speed (m/s)",
    )
    if plot:
        retfunc.plot(min_mass, max_mass, points)
    return retfunc
