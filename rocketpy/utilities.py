import inspect
import json
import os
import warnings
from datetime import date
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from packaging import version as packaging_version
from scipy.integrate import solve_ivp

from ._encoders import RocketPyDecoder, RocketPyEncoder
from .environment.environment import Environment
from .mathutils.function import Function
from .plots.plot_helpers import show_or_save_plot
from .rocket.aero_surface import TrapezoidalFins
from .simulation.flight import Flight


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


def check_constant(f, eps):
    """
    Check for three consecutive elements in the list that are approximately
    equal within a tolerance.

    Parameters
    ----------
    f : list or array
        A list or array of numerical values.
    eps : float
        The tolerance level for comparing the elements.

    Returns
    -------
    int or None
        The index of the first element in the first sequence of three
        consecutive elements that are approximately equal within the tolerance.
        Returns None if no such sequence is found.
    """
    for i in range(len(f) - 2):
        if abs(f[i + 2] - f[i + 1]) < eps and abs(f[i + 1] - f[i]) < eps:
            return i


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

    if v0 >= 0:  # pragma: no cover
        raise ValueError("Please set a valid negative value for v0")

    if env is None:
        environment = Environment(
            latitude=0,
            longitude=0,
            elevation=1000,
            date=(2020, 3, 4, 12),
        )
    else:
        environment = env

    def du(z, u):
        """Returns the derivative of the velocity at a given altitude.

        Parameters
        ----------
        z : float
            altitude, in meters, at a given time
        u : float
            velocity, in m/s, at a given z altitude

        Returns
        -------
        float
            velocity at a given altitude
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


# pylint: disable=too-many-statements
def fin_flutter_analysis(
    fin_thickness,
    shear_modulus,
    flight,
    see_prints=True,
    see_graphs=True,
    *,
    filename=None,
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
    flight : Flight
        Flight object containing the rocket's flight data
    see_prints : boolean, optional
        True if you want to see the prints, False otherwise.
    see_graphs : boolean, optional
        True if you want to see the graphs, False otherwise. If False, the
        function will return the vectors containing the data for the graphs.
    filename : str | None, optional
        The path the plot should be saved to. By default None, in which case the
        plot will be shown instead of saved. Supported file endings are: eps,
        jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp
        (these are the formats supported by matplotlib).

    Return
    ------
    None
    """
    found_fin = False
    surface_area = None
    aspect_ratio = None
    lambda_ = None

    # First, we need identify if there is at least one fin set in the rocket
    for aero_surface in flight.rocket.fins:
        if isinstance(aero_surface, TrapezoidalFins):
            root_chord = aero_surface.root_chord
            surface_area = (aero_surface.tip_chord + root_chord) * aero_surface.span / 2
            aspect_ratio = aero_surface.span * aero_surface.span / surface_area
            lambda_ = aero_surface.tip_chord / root_chord
            if not found_fin:
                found_fin = True
            else:
                warnings.warn("More than one fin set found. The last one will be used.")
    if not found_fin:  # pragma: no cover
        raise AttributeError(
            "There is no TrapezoidalFins in the rocket, can't run Flutter Analysis."
        )

    # Calculate variables
    flutter_mach = _flutter_mach_number(
        fin_thickness, shear_modulus, flight, root_chord, aspect_ratio, lambda_
    )
    safety_factor = _flutter_safety_factor(flight, flutter_mach)

    # Prints and plots
    if see_prints:
        _flutter_prints(
            fin_thickness,
            shear_modulus,
            surface_area,
            aspect_ratio,
            lambda_,
            flutter_mach,
            safety_factor,
            flight,
        )
    if see_graphs:
        _flutter_plots(flight, flutter_mach, safety_factor, filename)
    else:
        return flutter_mach, safety_factor


def _flutter_mach_number(
    fin_thickness, shear_modulus, flight, root_chord, aspect_ratio, lambda_
):
    flutter_mach = (
        (shear_modulus * 2 * (aspect_ratio + 2) * (fin_thickness / root_chord) ** 3)
        / (1.337 * (aspect_ratio**3) * (lambda_ + 1) * flight.pressure)
    ) ** 0.5
    flutter_mach.set_title("Fin Flutter Mach Number")
    flutter_mach.set_outputs("Mach")
    return flutter_mach


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
    safety_factor = flutter_mach / flight.mach_number
    safety_factor.set_title("Fin Flutter Safety Factor")
    safety_factor.set_outputs("Safety Factor")
    return safety_factor


def _flutter_plots(flight, flutter_mach, safety_factor, *, filename=None):
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
    filename : str | None, optional
        The path the plot should be saved to. By default None, in which case the
        plot will be shown instead of saved. Supported file endings are: eps,
        jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp
        (these are the formats supported by matplotlib).

    Returns
    -------
    None
    """
    # TODO: move to rocketpy.plots submodule
    _ = plt.figure(figsize=(6, 6))
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
    show_or_save_plot(filename)


def _flutter_prints(
    fin_thickness,
    shear_modulus,
    surface_area,
    aspect_ratio,
    lambda_,
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
    surface_area : float
        Fin surface area, in squared meters
    aspect_ratio : float
        Fin aspect ratio
    lambda_ : float
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
    # TODO: move to rocketpy.prints submodule
    time_index = np.argmin(flutter_mach[:, 1])
    time_min_mach = flutter_mach[time_index, 0]
    min_mach = flutter_mach[time_index, 1]
    min_vel = min_mach * flight.speed_of_sound(time_min_mach)

    time_index = np.argmin(safety_factor[:, 1])
    time_min_sf = safety_factor[time_index, 0]
    min_sf = safety_factor[time_index, 1]
    altitude_min_sf = flight.z(time_min_sf) - flight.env.elevation

    print("\nFin's parameters")
    print(f"Surface area (S): {surface_area:.4f} m2")
    print(f"Aspect ratio (AR): {aspect_ratio:.3f}")
    print(f"tip_chord/root_chord ratio = \u03bb = {lambda_:.3f}")
    print(f"Fin Thickness: {fin_thickness:.5f} m")
    print(f"Shear Modulus (G): {shear_modulus:.3e} Pa")

    print("\nFin Flutter Analysis")
    print(f"Minimum Fin Flutter Velocity: {min_vel:.3f} m/s at {time_min_mach:.2f} s")
    print(f"Minimum Fin Flutter Mach Number: {min_mach:.3f} ")
    print(f"Minimum Safety Factor: {min_sf:.3f} at {time_min_sf:.2f} s")
    print(f"Altitude of minimum Safety Factor: {altitude_min_sf:.3f} m (AGL)\n")


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


def get_instance_attributes(instance):
    """Returns a dictionary with all attributes of a given instance.

    Parameters
    ----------
    instance : object
        Instance of a class.

    Returns
    -------
    dictionary
        Dictionary with all attributes of the given instance.
    """
    attributes_dict = {}
    members = inspect.getmembers(instance)
    for member in members:
        # Filter out methods and protected attributes
        if not inspect.ismethod(member[1]) and not member[0].startswith("__"):
            attributes_dict[member[0]] = member[1]
    return attributes_dict


def save_to_rpy(flight: Flight, filename: str, include_outputs=False):
    """Saves a .rpy file into the given path, containing key simulation
    informations to reproduce the results.

    Parameters
    ----------
    flight : rocketpy.Flight
        Flight object containing the rocket's flight data
    filename : str
        Path where the file will be saved in
    include_outputs : bool, optional
        If True, the function will include extra outputs into the file,
        by default False

    Returns
    -------
    None
    """
    file = Path(filename).with_suffix(".rpy")

    with open(file, "w") as f:
        data = {"date": str(date.today()), "version": version("rocketpy")}
        data["simulation"] = flight
        json.dump(
            data,
            f,
            cls=RocketPyEncoder,
            indent=2,
            include_outputs=include_outputs,
        )


def load_from_rpy(filename: str, resimulate=False):
    """Loads the saved data from a .rpy file into a Flight object.

    Parameters
    ----------
    filename : str
        Path where the file to be loaded is
    resimulate : bool, optional
        If True, the function will resimulate the Flight object,
        by default False

    Returns
    -------
    rocketpy.Flight
        Flight object containing simulation information from the .rpy file
    """
    ext = os.path.splitext(os.path.basename(filename))[1]
    if ext != ".rpy":  # pragma: no cover
        raise ValueError(f"Invalid file extension: {ext}. Allowed: .rpy")

    with open(filename, "r") as f:
        data = json.load(f)
        if packaging_version.parse(data["version"]) > packaging_version.parse(
            version("rocketpy")
        ):
            warnings.warn(
                "The file was saved in an updated version of",
                f"RocketPy (v{data['version']}), the current",
                f"imported module is v{version('rocketpy')}",
            )
        simulation = json.dumps(data["simulation"])
        flight = json.loads(simulation, cls=RocketPyDecoder, resimulate=resimulate)
    return flight
