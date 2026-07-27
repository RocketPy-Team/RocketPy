import warnings
from bisect import bisect_left, bisect_right
from copy import deepcopy
from functools import cached_property

import numpy as np
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, OdeSolver, Radau

from rocketpy.simulation.flight_data_exporter import FlightDataExporter

from .._logging import enable_logging, logger
from ..environment.environment import Environment
from ..environment.models import Earth, Space
from ..mathutils.epoch import Epoch
from ..mathutils.flight_state import FlightState
from ..mathutils.function import Function, funcify_method
from ..mathutils.orbital_elements import OrbitalElements
from ..mathutils.reference_frame import (
    FlatEarthDatum,
    ReferenceFrame,
    quaternion_from_matrix,
)
from ..mathutils.vector_function import VectorFunction
from ..mathutils.vector_matrix import Matrix
from ..motors.solid_motor import SolidMotor
from ..plots.flight_plots import _FlightPlots
from ..prints.flight_prints import _FlightPrints
from ..tools import (
    deprecated,
    euler313_to_quaternions,
    find_closest,
    inverted_haversine_array,
    quaternions_to_nutation,
    quaternions_to_precession,
    quaternions_to_spin,
)
from .events.event import Event
from .events.event_builders import build_core_events
from .helpers.event_calling import (
    build_event_kwargs,
    call_events,
    compute_needs_union,
    infer_step_size,
    process_overshootable_event,
    update_overshootable_event_kwargs,
)
from .helpers.flight_derivatives import (
    u_dot,
    u_dot_earth_centered,
    u_dot_generalized,
    u_dot_generalized_3dof,
    u_dot_parachute,
    udot_rail1,
    udot_rail2,
    udot_rail_earth_centered,
)
from .helpers.flight_phase import _FlightPhases, _TimeNodes
from .orbit import FlightOrbit

ODE_SOLVER_MAP = {
    "RK23": RK23,
    "RK45": RK45,
    "DOP853": DOP853,
    "Radau": Radau,
    "BDF": BDF,
    "LSODA": LSODA,
}


class Flight:  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """Keeps all flight information and has a method to simulate flight.

    Attributes
    ----------
    Flight.env : Environment or Earth
        Local weather/site for classic flight, or the Earth domain for
        Earth-centered propagation.
    Flight.rocket : Rocket
        Rocket class describing rocket. See Rocket class for more
        details.
    Flight.parachutes : Parachute
        Direct link to parachutes of the Rocket. See Rocket class
        for more details.
    Flight.frontal_surface_wind : float
        Surface wind speed in m/s aligned with the launch rail.
    Flight.lateral_surface_wind : float
        Surface wind speed in m/s perpendicular to launch rail.
    Flight.rail_length : float, int
        Launch rail length in meters.
    Flight.effective_1rl : float
        Original rail length minus the distance measured from nozzle exit
        to the upper rail button. It assumes the nozzle to be aligned with
        the beginning of the rail.
    Flight.effective_2rl : float
        Original rail length minus the distance measured from nozzle exit
        to the lower rail button. It assumes the nozzle to be aligned with
        the beginning of the rail.
    Flight.name: str
        Name of the flight.
    Flight._controllers : list
        List of controllers to be used during simulation.
    Flight.max_time : int, float
        Maximum simulation time allowed. Refers to physical time
        being simulated, not time taken to run simulation.
    Flight.max_time_step : int, float
        Maximum time step to use during numerical integration in seconds.
    Flight.min_time_step : int, float
        Minimum time step to use during numerical integration in seconds.
    Flight.rtol : int, float
        Maximum relative error tolerance to be tolerated in the
        numerical integration scheme.
    Flight.atol : int, float
        Maximum absolute error tolerance to be tolerated in the
        integration scheme.
    Flight.time_overshoot : bool, optional
        If True, decouples ODE time step from parachute and controller trigger
        functions sampling rate. The time steps can overshoot the necessary
        trigger function evaluation points and then interpolation is used to
        calculate them and feed the triggers. Can greatly improve run
        time in some cases.
    Flight.terminate_on_apogee : bool
        Whether to terminate simulation when rocket reaches apogee.
    Flight.solver : scipy.integrate.LSODA
        Scipy LSODA integration scheme.
    Flight.x : Function
        Rocket's X coordinate (positive east) as a function of time.
    Flight.y : Function
        Rocket's Y coordinate (positive north) as a function of time.
    Flight.z : Function
        Rocket's z coordinate (positive up) as a function of time.
    Flight.vx : Function
        Velocity of the rocket's center of dry mass in the X (East) direction of
        the inertial frame as a function of time.
    Flight.vy : Function
        Velocity of the rocket's center of dry mass in the Y (North) direction of
        the inertial frame as a function of time.
    Flight.vz : Function
        Velocity of the rocket's center of dry mass in the Z (Up) direction of
        the inertial frame as a function of time.
    Flight.e0 : Function
        Rocket's Euler parameter 0 as a function of time.
    Flight.e1 : Function
        Rocket's Euler parameter 1 as a function of time.
    Flight.e2 : Function
        Rocket's Euler parameter 2 as a function of time.
    Flight.e3 : Function
        Rocket's Euler parameter 3 as a function of time.
    Flight.w1 : Function
        Angular velocity of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        pitch rate (q).
    Flight.w2 : Function
        Angular velocity of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw rate (r).
    Flight.w3 : Function
        Angular velocity of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll rate (p).
    Flight.latitude: Function
        Rocket's latitude coordinates (positive North) as a function of time.
        The Equator has a latitude equal to 0, by convention.
    Flight.longitude: Function
        Rocket's longitude coordinates (positive East) as a function of time.
        Greenwich meridian has a longitude equal to 0, by convention.
    Flight.inclination : int, float
        Launch rail inclination angle relative to ground, given in degrees.
    Flight.heading : int, float
        Launch heading angle relative to north given in degrees.
    Flight.initial_solution : list
        List defines initial condition - [t_initial, x_init,
        y_init, z_init, vx_init, vy_init, vz_init, e0_init, e1_init,
        e2_init, e3_init, w1_init, w2_init, w3_init]
    Flight.t_initial : int, float
        Initial simulation time in seconds. Usually 0.
    Flight.solution : list
        Solution array which keeps results from each numerical
        integration.
    Flight.t : float
        Current integration time.
    Flight.y : list
        Current integration state vector u.
    Flight.out_of_rail_time : int, float
        Time, in seconds, in which the rocket completely leaves the
        rail.
    Flight.out_of_rail_state : list
        State vector u corresponding to state when the rocket
        completely leaves the rail.
    Flight.out_of_rail_velocity : int, float
        Velocity, in m/s, with which the rocket completely leaves the
        rail.
    Flight.apogee_state : array
        State vector u corresponding to state when the rocket's
        vertical velocity is zero in the apogee.
    Flight.apogee_time : int, float
        Time, in seconds, in which the rocket's vertical velocity
        reaches zero in the apogee.
    Flight.apogee_x : int, float
        X coordinate (positive east) of the center of mass of the
        rocket when it reaches apogee.
    Flight.apogee_y : int, float
        Y coordinate (positive north) of the center of mass of the
        rocket when it reaches apogee.
    Flight.apogee : int, float
        Z coordinate, or altitude, of the center of mass of the
        rocket when it reaches apogee.
    Flight.x_impact : int, float
        X coordinate (positive east) of the center of mass of the
        rocket when it impacts ground.
    Flight.y_impact : int, float
        Y coordinate (positive east) of the center of mass of the
        rocket when it impacts ground.
    Flight.z_impact : int, float
        Z coordinate (positive up) of the center of mass of the
        rocket when it impacts ground.
    Flight.impact_velocity : int, float
        Velocity magnitude of the center of mass of the rocket when
        it impacts ground.
    Flight.impact_state : array
        State vector u corresponding to state when the rocket
        impacts the ground.
    Flight.parachute_events : array
        List that stores parachute events triggered during flight.
    Flight.function_evaluations : array
        List that stores number of derivative function evaluations
        during numerical integration in cumulative manner.
    Flight.function_evaluations_per_time_step : list
        List that stores number of derivative function evaluations
        per time step during numerical integration.
    Flight.time_steps : array
        List of time steps taking during numerical integration in
        seconds.
    Flight.wind_velocity_x : Function
        Wind velocity X (East) experienced by the rocket as a
        function of time.
    Flight.wind_velocity_y : Function
        Wind velocity Y (North) experienced by the rocket as a
        function of time.
    Flight.density : Function
        Air density experienced by the rocket as a function of
        time.
    Flight.pressure : Function
        Air pressure experienced by the rocket as a function of
        time.
    Flight.dynamic_viscosity : Function
        Air dynamic viscosity experienced by the rocket as a function of
        time.
    Flight.speed_of_sound : Function
        Speed of Sound in air experienced by the rocket as a
        function of time.
    Flight.ax : Function
        Acceleration of the rocket's center of dry mass along the X (East)
        axis in the inertial frame as a function of time.
    Flight.ay : Function
        Acceleration of the rocket's center of dry mass along the Y (North)
        axis in the inertial frame as a function of time.
    Flight.az : Function
        Acceleration of the rocket's center of dry mass along the Z (Up)
        axis in the inertial frame as a function of time.
    Flight.alpha1 : Function
        Angular acceleration of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw acceleration.
    Flight.alpha2 : Function
        Angular acceleration of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw acceleration.
    Flight.alpha3 : Function
        Angular acceleration of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll acceleration.
    Flight.speed : Function
        Rocket velocity magnitude in m/s relative to ground as a
        function of time.
    Flight.max_speed : float
        Maximum velocity magnitude in m/s reached by the rocket
        relative to ground during flight.
    Flight.max_speed_time : float
        Time in seconds at which rocket reaches maximum velocity
        magnitude relative to ground.
    Flight.horizontal_speed : Function
        Rocket's velocity magnitude in the horizontal (North-East)
        plane in m/s as a function of time.
    Flight.acceleration : Function
        Rocket acceleration magnitude in m/s² relative to ground as a
        function of time.
    Flight.max_acceleration : float
        Maximum acceleration magnitude in m/s² reached by the rocket
        relative to ground during flight.
    Flight.max_acceleration_time : float
        Time in seconds at which rocket reaches maximum acceleration
        magnitude relative to ground.
    Flight.path_angle : Function
        Rocket's flight path angle, or the angle that the
        rocket's velocity makes with the horizontal (North-East)
        plane. Measured in degrees and expressed as a function
        of time.
    Flight.attitude_vector_x : Function
        Rocket's attitude vector, or the vector that points
        in the rocket's axis of symmetry, component in the X
        direction (East) as a function of time.
    Flight.attitude_vector_y : Function
        Rocket's attitude vector, or the vector that points
        in the rocket's axis of symmetry, component in the Y
        direction (East) as a function of time.
    Flight.attitude_vector_z : Function
        Rocket's attitude vector, or the vector that points
        in the rocket's axis of symmetry, component in the Z
        direction (East) as a function of time.
    Flight.attitude_angle : Function
        Rocket's attitude angle, or the angle that the
        rocket's axis of symmetry makes with the horizontal (North-East)
        plane. Measured in degrees and expressed as a function
        of time.
    Flight.lateral_attitude_angle : Function
        Rocket's lateral attitude angle, or the angle that the
        rocket's axis of symmetry makes with plane defined by
        the launch rail direction and the Z (up) axis.
        Measured in degrees and expressed as a function
        of time.
    Flight.phi : Function
        Rocket's Spin Euler Angle, φ, according to the 3-2-3 rotation
        system nomenclature (NASA Standard Aerospace). Measured in degrees and
        expressed as a function of time.
    Flight.theta : Function
        Rocket's Nutation Euler Angle, θ, according to the 3-2-3 rotation
        system nomenclature (NASA Standard Aerospace). Measured in degrees and
        expressed as a function of time.
    Flight.psi : Function
        Rocket's Precession Euler Angle, ψ, according to the 3-2-3 rotation
        system nomenclature (NASA Standard Aerospace). Measured in degrees and
        expressed as a function of time.
    Flight.R1 : Function
        Aerodynamic force acting along the x-axis of the rocket's body frame
        as a function of time. Expressed in Newtons (N).
    Flight.R2 : Function
        Aerodynamic force acting along the y-axis of the rocket's body frame
        as a function of time. Expressed in Newtons (N).
    Flight.R3 : Function
        Aerodynamic force acting along the z-axis of the rocket's body frame
        as a function of time. Expressed in Newtons (N).
    Flight.M1 : Function
        Aerodynamic moment acting along the x-axis of the rocket's body
        frame as a function of time. Expressed in Newtons (N).
    Flight.M2 : Function
        Aerodynamic moment acting along the y-axis of the rocket's body
        frame as a function of time. Expressed in Newtons (N).
    Flight.M3 : Function
        Aerodynamic moment acting along the z-axis of the rocket's body
        frame as a function of time. Expressed in Newtons (N).
    Flight.net_thrust : Function
        Rocket's engine net thrust as a function of time in Newton.
        This is the actual thrust force experienced by the rocket.
        It may be corrected with the atmospheric pressure if a reference
        pressure is defined.
    Flight.aerodynamic_normal_force : Function
        Resultant aerodynamic force perpendicular to the rocket's longitudinal
        axis in the body frame, as a function of time. Equal to sqrt(R1² + R2²).
        Units in N. Can be called or accessed as array.
    Flight.aerodynamic_axial_force : Function
        Aerodynamic force along the rocket's longitudinal axis in the body
        frame, as a function of time. Equal to -R3. Units in N. Can be called
        or accessed as array.
    Flight.aerodynamic_lift : Function
        Aerodynamic lift force in the aerodynamic frame (perpendicular to
        freestream), as a function of time. Computed as N·cos(α) - A·sin(α).
        Units in N.
    Flight.aerodynamic_drag : Function
        Aerodynamic drag force in the aerodynamic frame (opposing freestream),
        as a function of time. Computed as N·sin(α) + A·cos(α). Units in N.
    Flight.aerodynamic_bending_moment : Function
        Resultant moment perpendicular to rocket's axis due to
        aerodynamic effects as a function of time. Units in N m.
        Expressed as a function of time. Can be called or accessed
        as array.
    Flight.aerodynamic_spin_moment : Function
        Resultant moment aligned with the rockets axis due to
        aerodynamic effects as a function of time. Units in N m.
        Expressed as a function of time. Can be called or accessed
        as array.
    Flight.rail_button1_normal_force : Function
        Upper rail button normal force in N as a function
        of time.
    Flight.max_rail_button1_normal_force : float
        Maximum upper rail button normal force experienced
        during rail flight phase in N.
    Flight.rail_button1_shear_force : Function
        Upper rail button shear force in N as a function
        of time.
    Flight.max_rail_button1_shear_force : float
        Maximum upper rail button shear force experienced
        during rail flight phase in N.
    Flight.rail_button2_normal_force : Function
        Lower rail button normal force in N as a function
        of time.
    Flight.max_rail_button2_normal_force : float
        Maximum lower rail button normal force experienced
        during rail flight phase in N.
    Flight.rail_button2_shear_force : Function
        Lower rail button shear force in N as a function
        of time.
    Flight.max_rail_button2_shear_force : float
        Maximum lower rail button shear force experienced
        during rail flight phase in N.
    Flight.rotational_energy : Function
        Rocket's rotational kinetic energy as a function of time.
        Units in J.
    Flight.translational_energy : Function
        Rocket's translational kinetic energy as a function of time.
        Units in J.
    Flight.kinetic_energy : Function
        Rocket's total kinetic energy as a function of time.
        Units in J.
    Flight.potential_energy : Function
        Rocket's gravitational potential energy as a function of
        time. Units in J.
    Flight.total_energy : Function
        Rocket's total mechanical energy as a function of time.
        Units in J.
    Flight.thrust_power : Function
        Rocket's engine thrust power output as a function
        of time in Watts.
    Flight.drag_power : Function
        Aerodynamic drag power output as a function
        of time in Watts.
    Flight.attitude_frequency_response : Function
        Fourier Frequency Analysis of the rocket's attitude angle.
        Expressed as the absolute value of the magnitude as a function
        of frequency in Hz.
    Flight.omega1_frequency_response : Function
        Fourier Frequency Analysis of the rocket's angular velocity omega 1.
        Expressed as the absolute value of the magnitude as a function
        of frequency in Hz.
    Flight.omega2_frequency_response : Function
        Fourier Frequency Analysis of the rocket's angular velocity omega 2.
        Expressed as the absolute value of the magnitude as a function
        of frequency in Hz.
    Flight.omega3_frequency_response : Function
        Fourier Frequency Analysis of the rocket's angular velocity omega 3.
        Expressed as the absolute value of the magnitude as a function
        of frequency in Hz.
    Flight.static_margin : Function
        Rocket's static margin during flight in calibers.
    Flight.stability_margin : Function
            Rocket's stability margin during flight, in calibers.
    Flight.initial_stability_margin : float
        Rocket's initial stability margin in calibers.
    Flight.out_of_rail_stability_margin : float
        Rocket's stability margin in calibers when it leaves the rail.
    Flight.stream_velocity_x : Function
        Freestream velocity x (East) component, in m/s, as a function of
        time.
    Flight.stream_velocity_y : Function
        Freestream velocity y (North) component, in m/s, as a function of
        time.
    Flight.stream_velocity_z : Function
        Freestream velocity z (up) component, in m/s, as a function of
        time.
    Flight.free_stream_speed : Function
        Freestream velocity magnitude, in m/s, as a function of time.
    Flight.apogee_freestream_speed : float
        Freestream speed of the rocket at apogee in m/s.
    Flight.mach_number : Function
        Rocket's Mach number defined as its freestream speed
        divided by the speed of sound at its altitude. Expressed
        as a function of time.
    Flight.max_mach_number : float
        Rocket's maximum Mach number experienced during flight.
    Flight.max_mach_number_time : float
        Time at which the rocket experiences the maximum Mach number.
    Flight.reynolds_number : Function
        Rocket's Reynolds number, using its diameter as reference
        length and free_stream_speed as reference velocity. Expressed
        as a function of time.
    Flight.max_reynolds_number : float
        Rocket's maximum Reynolds number experienced during flight.
    Flight.max_reynolds_number_time : float
        Time at which the rocket experiences the maximum Reynolds number.
    Flight.dynamic_pressure : Function
        Dynamic pressure experienced by the rocket in Pa as a function
        of time, defined by 0.5*rho*V^2, where rho is air density and V
        is the freestream speed.
    Flight.max_dynamic_pressure : float
        Maximum dynamic pressure, in Pa, experienced by the rocket.
    Flight.max_dynamic_pressure_time : float
        Time at which the rocket experiences maximum dynamic pressure.
    Flight.total_pressure : Function
        Total pressure experienced by the rocket in Pa as a function
        of time.
    Flight.max_total_pressure : float
        Maximum total pressure, in Pa, experienced by the rocket.
    Flight.max_total_pressure_time : float
        Time at which the rocket experiences maximum total pressure.
    Flight.angle_of_attack : Function
        Rocket's angle of attack in degrees as a function of time.
        Defined as the minimum angle between the attitude vector and
        the freestream velocity vector. Can be called or accessed as
        array.
    Flight.simulation_mode : str
        Simulation mode for the flight. Can be "6DOF" or "3DOF".
    Flight.rail_button1_bending_moment : Function
        Internal bending moment at upper rail button attachment point in N·m
        as a function of time. Calculated using beam theory during rail phase.
    Flight.max_rail_button1_bending_moment : float
        Maximum internal bending moment experienced at upper rail button
        attachment point during rail flight phase in N·m.
    Flight.rail_button2_bending_moment : Function
        Internal bending moment at lower rail button attachment point in N·m
        as a function of time. Calculated using beam theory during rail phase.
    Flight.max_rail_button2_bending_moment : float
        Maximum internal bending moment experienced at lower rail button
        attachment point during rail flight phase in N·m.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        rocket,
        environment: Environment | Earth,
        rail_length,
        inclination=80.0,
        heading=90.0,
        initial_solution=None,
        terminate_on_apogee=False,
        max_time=600,
        max_time_step=np.inf,
        min_time_step=0,
        rtol=1e-6,
        atol=None,
        time_overshoot=True,
        verbose=False,
        name="Flight",
        equations_of_motion="standard",
        ode_solver="LSODA",
        simulation_mode="6DOF",
        custom_events=None,
        space: Space | None = None,
        forces=None,
    ):
        """Run one datum-inertial Flight from a rail or a typed state.

        A standalone :class:`Environment` selects ``FlatEarthDatum`` and keeps
        the established low_altitude workflow. An :class:`Earth` selects its
        rotating geocentric datum and permits the same 6DOF integration to
        continue from rail launch through orbital flight. ``initial_solution``
        may be a :class:`FlightState`; :meth:`from_state` and
        :meth:`from_orbit` provide clearer intent-oriented spellings.
        """
        typed_state = (
            initial_solution if isinstance(initial_solution, FlightState) else None
        )
        self._initialize(
            rocket=rocket,
            environment=environment,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
            initial_solution=None if typed_state is not None else initial_solution,
            terminate_on_apogee=terminate_on_apogee,
            max_time=max_time,
            max_time_step=max_time_step,
            min_time_step=min_time_step,
            rtol=rtol,
            atol=atol,
            time_overshoot=time_overshoot,
            verbose=verbose,
            name=name,
            equations_of_motion=equations_of_motion,
            ode_solver=ode_solver,
            simulation_mode=simulation_mode,
            custom_events=custom_events,
            space=space,
            _forces=forces,
            _initial_state=typed_state,
        )

    def _initialize(  # pylint: disable=too-many-arguments,too-many-statements
        self,
        rocket,
        environment,
        rail_length,
        inclination=80.0,
        heading=90.0,
        initial_solution=None,
        terminate_on_apogee=False,
        max_time=600,
        max_time_step=np.inf,
        min_time_step=0,
        rtol=1e-6,
        atol=None,
        time_overshoot=True,
        verbose=False,
        name="Flight",
        equations_of_motion="standard",
        ode_solver="LSODA",
        simulation_mode="6DOF",
        custom_events=None,
        space=None,
        _initial_state=None,
        _forces=None,
    ):
        """Run a trajectory simulation.

        Parameters
        ----------
        rocket : Rocket
            Rocket to simulate.
        environment : Environment or Earth
            Local Environment for a traditional simulation, or Earth for an
            Earth-centered launch, continuation, or orbital simulation.
        rail_length : int, float
            Length in which the rocket will be attached to the rail, only
            moving along a fixed direction, that is, the line parallel to the
            rail. If an initial_solution is passed, the rail length is used to
            decide whether the rocket is still on the rail or has already left
            it.
        inclination : int, float, optional
            Rail inclination angle relative to ground, given in degrees.
            Default is 80.
        heading : int, float, optional
            Heading angle relative to north given in degrees.
            Default is 90, which points in the x (east) direction.
        initial_solution : array, Flight, optional
            Initial solution array to be used. Format is:

            .. code-block:: python

                initial_solution = [
                    self.t_initial,
                    x_init, y_init, z_init,
                    vx_init, vy_init, vz_init,
                    e0_init, e1_init, e2_init, e3_init,
                    w1_init, w2_init, w3_init
                ]

            If a Flight object is used, the last state vector will be
            used as initial solution. If None, the initial solution will start
            with all null values, except for the euler parameters which will be
            calculated based on given values of inclination and heading.
            Default is None.
        terminate_on_apogee : boolean, optional
            Whether to terminate simulation when rocket reaches apogee.
            Default is False.
        max_time : int, float, optional
            Maximum time in which to simulate trajectory in seconds.
            Using this without setting a max_time_step may cause unexpected
            errors. Default is 600.
        max_time_step : int, float, optional
            Maximum time step to use during integration in seconds.
            Default is 0.01.
        min_time_step : int, float, optional
            Minimum time step to use during integration in seconds.
            Default is 0.01.
        rtol : float, array, optional
            Maximum relative error tolerance to be tolerated in the
            integration scheme. Can be given as array for each
            state space variable. Default is 1e-6.
        atol : float, optional
            Maximum absolute error tolerance to be tolerated in the
            integration scheme. Can be given as array for each
            state space variable. Default is 6*[1e-3] + 4*[1e-6] + 3*[1e-3].
        time_overshoot : bool, optional
            If True, decouples ODE time step from parachute and controller
            trigger functions sampling rate. The time steps can overshoot the
            necessary trigger function evaluation points and then interpolation
            is used to calculate them and feed the triggers. Can greatly improve
            run time in some cases. Default is True.
        verbose : bool or str, optional
            If truthy, shows a live readout of the current simulation time and
            enables console logging for milestones such as phase transitions and
            completion. ``True`` uses the ``INFO`` level; pass a level name
            instead (e.g. ``"debug"``, ``"warning"``) to control how much detail
            is shown. This is a convenience wrapper around
            :func:`rocketpy.enable_logging`; for finer control configure the
            ``"rocketpy"`` logger directly (e.g. via
            :func:`rocketpy.set_log_level`). Default is False.
        name : str, optional
            Name of the flight. Default is "Flight".
        equations_of_motion : str, optional
            Type of equations of motion to use. Can be "standard" or
            "solid_propulsion". Default is "standard". Solid propulsion is a
            more restricted set of equations of motion that only works for
            solid propulsion rockets. Solid motors are automatically mapped to
            "solid_propulsion".
        ode_solver : str, ``scipy.integrate.OdeSolver``, optional
            Integration method to use to solve the equations of motion ODE.
            Available options are: 'RK23', 'RK45', 'DOP853', 'Radau', 'BDF',
            'LSODA' from ``scipy.integrate.solve_ivp``.
            Default is 'LSODA', which is recommended for most flights.
            A custom ``scipy.integrate.OdeSolver`` can be passed as well.
            For more information on the integration methods, see the scipy
            documentation [1]_.
        simulation_mode : str, optional
            Equations-of-motion mode. Accepted spellings are ``"3DOF"``,
            ``"3 DOF"``, ``"6DOF"`` and ``"6 DOF"``. Default is ``"6DOF"``.
        custom_events : Event or list[Event], optional
            Event or list of Events to be monitored during flight. See Event
            class for more details. Default is None.
        space : Space, optional
            Space domain containing third bodies and space perturbations.
        forces : iterable, optional
            Additional acceleration models scoped to this Earth-centered
            Flight. Each model is callable or exposes ``acceleration(epoch,
            state, vehicle, earth)`` and returns a GCRF vector in m/s².
        Returns
        -------
        None

        References
        ----------
        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        """
        # Save arguments
        if not isinstance(environment, (Environment, Earth)):
            raise TypeError("environment must be an Environment or Earth.")
        if space is not None and not isinstance(space, Space):
            raise TypeError("space must be a Space or None.")
        if isinstance(environment, Environment) and space is not None:
            raise ValueError("Space is only valid when environment is Earth.")
        if isinstance(environment, Environment) and _forces:
            raise ValueError("Per-leg forces are only supported with Earth.")
        if (
            isinstance(environment, Earth)
            and initial_solution is not None
            and _initial_state is None
        ):
            raise ValueError(
                "Earth-centered continuation requires a typed FlightState; "
                "use Flight.from_state()."
            )
        if _initial_state is not None and not isinstance(_initial_state, FlightState):
            raise TypeError("_initial_state must be a FlightState instance.")
        if isinstance(environment, Environment) and _initial_state is not None:
            if _initial_state.frame != ReferenceFrame.FLAT_EARTH:
                raise ValueError(
                    "A GCRF state requires Earth, not a local Environment."
                )

        self.env = environment
        self.initial_environment = environment
        self.earth = environment if isinstance(environment, Earth) else None
        self.space = space
        self.rocket = rocket
        self.vehicle = rocket
        if _initial_state is not None and initial_solution is not None:
            raise ValueError("Use either initial_state or initial_solution, not both.")
        self.datum = (
            environment.datum
            if isinstance(environment, Earth)
            else FlatEarthDatum(
                name=f"Flat {environment.earth_datum.name}",
                semi_major_axis=environment.earth_datum.semi_major_axis,
                flattening=environment.earth_datum.flattening,
                angular_velocity=environment.earth_datum.angular_velocity,
                gravitational_parameter=environment.earth_datum.gravitational_parameter,
            )
        )
        self.reference_frame = self.datum.integration_frame
        self._launch_with_earth = isinstance(environment, Earth) and (
            _initial_state is None
        )
        if self._launch_with_earth:
            _initial_state = self._earth_launch_state(inclination, heading)
        elif (
            _initial_state is not None and _initial_state.frame != self.reference_frame
        ):
            _initial_state = self._convert_initial_state(_initial_state)
        self.initial_state = _initial_state
        if (
            isinstance(environment, Environment)
            and self.initial_state is not None
            and self.initial_state.frame != ReferenceFrame.FLAT_EARTH
        ):
            raise ValueError(
                "Standalone Environment uses FlatEarthDatum and flat-Earth states."
            )
        self.start_epoch = (
            self.initial_state.epoch
            if self.initial_state is not None
            else getattr(environment, "epoch", Epoch.relative_origin())
        )
        self.mission_time_offset = (
            self.initial_state.elapsed_time if self.initial_state is not None else 0.0
        )
        self.forces = self._validate_forces(_forces or ())
        self._state_type = FlightState
        self.rail_length = rail_length
        if self._launch_with_earth and self.rail_length is None:
            raise ValueError("An Earth launch requires a rail_length.")
        if self.rail_length is None and _initial_state is None:
            raise ValueError("rail_length may be None only when initial_state is used.")
        if self.rail_length is not None and self.rail_length <= 0:
            raise ValueError("Rail length must be a positive value.")
        self.parachutes = self.rocket.parachutes[:]
        self.inclination = inclination
        self.heading = heading
        self.max_time = max_time
        if self.earth is not None:
            preparation_end = (
                self.start_epoch + float(self.max_time)
                if np.isfinite(self.max_time)
                else self.start_epoch
            )
            self.earth.prepare(
                self.start_epoch,
                preparation_end,
            )
        else:
            self.datum.prepare_epoch(self.start_epoch)
        self.max_time_step = max_time_step
        self.min_time_step = min_time_step
        self.rtol = rtol
        self.atol = atol or 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3]
        self.initial_solution = initial_solution
        self.time_overshoot = time_overshoot
        self.terminate_on_apogee = terminate_on_apogee
        self.name = name
        self.equations_of_motion = equations_of_motion
        self.simulation_mode = self._normalize_simulation_mode(simulation_mode)
        self.ode_solver = ode_solver
        self.verbose = verbose
        if verbose:
            # Convenience: route progress/diagnostics to the console. ``verbose``
            # may also be a logging level name (e.g. "debug") to control how much
            # detail is shown; True maps to "INFO". Users who manage logging
            # themselves can leave verbose=False and configure the "rocketpy"
            # logger directly.
            enable_logging("INFO" if verbose is True else verbose)
        self.custom_events = [] if custom_events is None else custom_events

        # Flight initialization
        self.__init_events()
        self.__init_solution_monitors()
        self.__init_equations_of_motion()
        self.__init_solver_monitors()

        # Simulate flight
        self.__simulate()

        # Initialize data exporter object
        self.exports = FlightDataExporter(self)

        # Initialize prints and plots objects
        self.prints = _FlightPrints(self)
        self.plots = _FlightPlots(self)

    def __repr__(self):
        return (
            f"<Flight(rocket= {self.rocket}, "
            f"environment= {self.env}, "
            f"rail_length= {self.rail_length}, "
            f"inclination= {self.inclination}, "
            f"heading = {self.heading},"
            f"name= {self.name})>"
        )

    def _earth_launch_state(self, inclination, heading):
        """Construct a rail-attached state in the datum inertial frame."""
        environment = self.earth.local_environment
        if environment is None:
            raise ValueError(
                "Launching with Earth requires an Environment as an atmosphere layer."
            )
        epoch = environment.epoch
        epoch.require_absolute("A launch with a rotating Earth datum")
        self.datum.prepare_epoch(epoch)
        latitude = np.radians(environment.latitude)
        longitude = np.radians(environment.longitude)

        roll = 0.0
        try:
            button = self.rocket.rail_buttons[0].component
            roll += (
                button.angular_position_rad
                if self.rocket._csys == 1
                else 2.0 * np.pi - button.angular_position_rad
            )
        except (AttributeError, IndexError):
            pass
        local_quaternion = euler313_to_quaternions(
            roll,
            np.radians(float(inclination) - 90.0),
            np.radians(-float(heading)),
        )
        body_to_local = np.asarray(
            Matrix.transformation(local_quaternion).components, dtype=float
        )
        fixed_to_enu = self.datum.topocentric_matrix(latitude, longitude)
        body_to_fixed = fixed_to_enu.T @ body_to_local
        self._launch_fixed_to_enu = fixed_to_enu
        self._launch_latitude_rad = latitude
        self._launch_longitude_rad = longitude
        self._launch_site_fixed = self.datum.geodetic_to_itrs(
            latitude, longitude, environment.elevation
        )
        self._launch_elevation = float(environment.elevation)
        self._launch_direction_fixed = body_to_fixed[:, 2].copy()
        inertial_to_fixed = self.datum.orientation_matrix(epoch)
        body_to_inertial = inertial_to_fixed.T @ body_to_fixed
        quaternion = quaternion_from_matrix(body_to_inertial)
        angular_velocity = body_to_inertial.T @ (
            self.datum.inertial_angular_velocity(epoch)
        )
        return FlightState.geodetic(
            epoch=epoch,
            latitude=environment.latitude,
            longitude=environment.longitude,
            altitude=environment.elevation,
            velocity_enu=(0.0, 0.0, 0.0),
            frame=self.reference_frame,
            datum=self.datum,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
        )

    def earth_launch_rail_coordinates(self, time, state):
        """Return distance and relative speed along the rotating launch rail."""
        epoch = self.current_epoch(time)
        position_fixed, velocity_fixed, _ = self.datum._to_ecef_coordinates(
            epoch,
            np.asarray(state[:3], dtype=float),
            np.asarray(state[3:6], dtype=float),
        )
        relative_position = position_fixed - self._launch_site_fixed
        return (
            float(np.dot(relative_position, self._launch_direction_fixed)),
            float(np.dot(velocity_fixed, self._launch_direction_fixed)),
        )

    def _convert_initial_state(self, state):
        """Convert a fixed-frame state to this datum's inertial frame."""
        if state.frame not in (self.datum.fixed_frame, self.datum.integration_frame):
            raise ValueError(
                f"State frame {state.frame.value!r} is not owned by "
                f"{type(self.datum).__name__}."
            )
        position, velocity, _ = self.datum.transform_kinematics(
            state.epoch,
            state.position,
            state.velocity,
            source=state.frame,
            target=self.reference_frame,
        )
        body_to_source = np.asarray(
            Matrix.transformation(state.quaternion).components, dtype=float
        )
        if state.frame == self.datum.fixed_frame:
            source_to_target = self.datum.orientation_matrix(state.epoch).T
            body_to_target = source_to_target @ body_to_source
            angular_velocity = state.angular_velocity + body_to_target.T @ (
                self.datum.inertial_angular_velocity(state.epoch)
            )
        else:
            body_to_target = body_to_source
            angular_velocity = state.angular_velocity
        return FlightState.cartesian(
            epoch=state.epoch,
            position=position,
            velocity=velocity,
            quaternion=quaternion_from_matrix(body_to_target),
            angular_velocity=angular_velocity,
            frame=self.reference_frame,
            elapsed_time=state.elapsed_time,
        )

    @staticmethod
    def _normalize_simulation_mode(value):
        """Return the established compact spelling for a Flight mode."""
        if not isinstance(value, str):
            raise TypeError("simulation_mode must be a string.")
        normalized = value.upper().replace(" ", "").replace("-", "")
        aliases = {"3DOF": "3DOF", "3D": "3DOF", "6DOF": "6DOF", "6D": "6DOF"}
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Invalid simulation_mode: {value!r}. Use '3DOF' or '6DOF'."
            ) from exc

    @classmethod
    def from_launch(
        cls,
        vehicle,
        environment: Environment | Earth,
        rail_length,
        **kwargs,
    ):
        """Run a rail launch in the inertial frame owned by the environment."""
        return cls(
            rocket=vehicle,
            environment=environment,
            rail_length=rail_length,
            **kwargs,
        )

    @classmethod
    def from_state(
        cls,
        vehicle,
        environment: Environment | Earth,
        state: FlightState,
        *,
        space: Space | None = None,
        duration=None,
        forces=None,
        **kwargs,
    ):
        """Continue a vehicle from a typed state.

        ``environment`` is a local Environment for flat-Earth state or Earth
        for GCRF state. Frame selection is inferred from ``state``.
        """
        if duration is not None:
            if "max_time" in kwargs:
                raise ValueError("Use either duration or max_time, not both.")
            kwargs["max_time"] = float(duration)
        flight = cls.__new__(cls)
        flight._initialize(
            rocket=vehicle,
            environment=environment,
            rail_length=None,
            space=space,
            _initial_state=state,
            _forces=forces,
            **kwargs,
        )
        return flight

    @classmethod
    def from_orbit(
        cls,
        vehicle,
        earth: Earth,
        state: FlightState,
        *,
        space: Space | None = None,
        **kwargs,
    ):
        """Run an Earth-centered phase; an explicit GCRF state is required."""
        if not isinstance(earth, Earth):
            raise TypeError("Flight.from_orbit requires an Earth.")
        if ReferenceFrame.coerce(state.frame) != ReferenceFrame.GCRF:
            raise ValueError("Flight.from_orbit requires a GCRF FlightState.")
        return cls.from_state(vehicle, earth, state, space=space, **kwargs)

    @staticmethod
    def _validate_forces(forces):
        """Validate and preserve explicitly supplied per-Flight forces."""
        validated = []
        for model in forces:
            if not callable(model) and not hasattr(model, "acceleration"):
                raise TypeError(
                    "Flight force models must be callables or expose acceleration()."
                )
            validated.append(model)
        return validated

    def current_epoch(self, time):
        """Return the simulation epoch at elapsed ``time`` in seconds."""
        return self.start_epoch + (float(time) - self.t_initial)

    def __init_events(self):
        """Initialize events and event triggers. The order of the list is the
        order that the events will be called. Core events should be first, then
        sensors, then parachutes, then controllers, then user defined events."""
        self.__init_eventful_objects()

        user_events = (
            self.custom_events
            if isinstance(self.custom_events, list)
            else [self.custom_events]
        )
        self.events = [*build_core_events()]

        if self._uses_automatic_fall_environment_event():
            self.events.append(self._build_automatic_fall_environment_event())

        # Sensor events (position-specific, created when sensors added to rocket)
        if hasattr(self.rocket, "_sensor_events"):
            self.events.extend(self.rocket._sensor_events)

        # Parachute events
        for parachute in self.parachutes:
            parachute._reset_signals()  # reset parachute pressure signals
            self.events.append(parachute.event)

        # Controller events
        for controller in self._controllers:
            self.events.append(controller.event)

        # User-defined events are appended last
        self.events.extend(user_events)

        self._overshootable_events = []
        self._non_overshootable_events = []
        self._has_change_dynamics_events = False
        for event in self.events:
            # Separate time overshootable and non overshootable events
            if (
                self.time_overshoot
                and event.sampling_rate is not None
                and event.time_overshootable
            ):
                self._overshootable_events.append(event)
            else:
                self._non_overshootable_events.append(event)

            self._has_change_dynamics_events |= event.changes_dynamics

            event.reset()  # Reset event state (commands/logs/enabled flag)

        # Sort by canonical order (lower priority runs first)
        self._overshootable_events.sort(key=lambda x: x.priority)
        self._non_overshootable_events.sort(key=lambda x: x.priority)

    def _uses_automatic_fall_environment_event(self):
        """Return whether this Flight starts above the automatic handoff."""
        if (
            self.earth is None
            or not self.earth.atmosphere.creates_automatic_fall_environment
        ):
            return False
        if self._launch_with_earth:
            # The trigger itself requires descending motion, so a ground launch
            # can safely register the event before it first climbs through the
            # handoff altitude.
            return True
        position_itrf, _, _ = self.earth.transform_kinematics(
            self.start_epoch,
            self.initial_state.position,
            source=self.initial_state.frame,
            target=ReferenceFrame.ITRF,
        )
        altitude = self.earth.datum.itrs_to_geodetic(position_itrf)[2]
        # A continuation beginning at the handoff (for example a Mission leg)
        # must not create the same regional Environment a second time.
        return (
            altitude > self.earth.atmosphere.AUTOMATIC_FALL_ENVIRONMENT_ALTITUDE + 1.0
        )

    def _build_automatic_fall_environment_event(self):
        """Create the Atmosphere-owned one-shot 20 km descent event."""

        atmosphere = self.earth.atmosphere
        trigger_altitude = atmosphere.AUTOMATIC_FALL_ENVIRONMENT_ALTITUDE
        armed = {"value": not self._launch_with_earth}

        def altitude_and_descent(flight, state, time):
            epoch = flight.current_epoch(time)
            position_itrf, velocity_itrf, _ = flight.datum.transform_kinematics(
                epoch,
                state[:3],
                state[3:6],
                source=flight.reference_frame,
                target=ReferenceFrame.ITRF,
            )
            latitude, longitude, altitude = flight.datum.itrs_to_geodetic(position_itrf)
            descending = np.dot(position_itrf, velocity_itrf) < 0
            return epoch, latitude, longitude, altitude, descending

        def trigger(**event_kwargs):
            flight = event_kwargs["flight"]
            if flight.reference_frame != ReferenceFrame.GCRF:
                return False
            *_, altitude, descending = altitude_and_descent(
                flight, event_kwargs["state"], event_kwargs["time"]
            )
            if altitude > trigger_altitude + 1.0:
                armed["value"] = True
                return False
            return armed["value"] and altitude <= trigger_altitude and descending

        def callback(**event_kwargs):
            flight = event_kwargs["flight"]
            earth = flight.earth
            epoch, latitude, longitude, altitude, _ = altitude_and_descent(
                flight, event_kwargs["state"], event_kwargs["time"]
            )
            regional = atmosphere.create_fall_environment(
                epoch=epoch,
                latitude=np.degrees(latitude),
                longitude=np.degrees(longitude),
            )
            return {
                "latitude": np.degrees(latitude),
                "longitude": np.degrees(longitude),
                "altitude": altitude,
                "earth": earth,
                "regional_environment": regional,
            }

        def exact_altitude(state, **event_kwargs):
            flight = event_kwargs["flight"]
            position_itrf, _, _ = flight.datum.transform_kinematics(
                flight.current_epoch(event_kwargs["time"]),
                state[:3],
                source=flight.reference_frame,
                target=ReferenceFrame.ITRF,
            )
            altitude = flight.datum.itrs_to_geodetic(position_itrf)[2]
            return altitude - trigger_altitude

        return Event(
            callback=callback,
            trigger=trigger,
            exact_time_function=exact_altitude,
            trigger_only_once=True,
            changes_dynamics=True,
            name="Automatic Fall Environment",
            priority=0,
        )

    def __init_eventful_objects(self):
        """Initialize controllers and sensors"""
        self._controllers = self.rocket._controllers[:]
        self.sensors = self.rocket.sensors.get_components()
        self.sensors_by_name = self.rocket.sensors_by_name

        # reset controllable object to initial state (only airbrakes for now)
        for air_brakes in self.rocket.air_brakes:
            air_brakes._reset()

        self.sensor_data = {}
        for sensor in self.sensors:
            sensor._reset(self.rocket)  # resets noise and measurement list
            self.sensor_data[sensor] = []

    def __init_solution_monitors(self):
        # Initialize solution monitors
        self.out_of_rail_time = 0
        self.out_of_rail_time_index = 0
        self.out_of_rail_state = np.array([0])
        self.apogee_state = np.array([0])
        self.apogee_x = 0
        self.apogee_y = 0
        self.apogee = 0
        self.apogee_time = 0
        self.x_impact = 0
        self.y_impact = 0
        self.z_impact = 0
        self.impact_velocity = 0
        self.impact_state = np.array([0])
        self.parachute_events = []
        self._active_parachute = None
        self._parachute_inflation_time = np.inf
        self.__post_processed_variables = []
        self._orbital_force_history = []

    def __init_equations_of_motion(self):
        """Initialize equations of motion."""
        # set all derivative functions
        self.u_dot = lambda t, u, post_processing=False: u_dot(
            self, t, u, post_processing
        )
        self.u_dot_generalized = lambda t, u, post_processing=False: u_dot_generalized(
            self, t, u, post_processing
        )
        self.u_dot_generalized_3dof = lambda t, u, post_processing=False: (
            u_dot_generalized_3dof(self, t, u, post_processing)
        )
        self.u_dot_parachute = lambda t, u, post_processing=False: u_dot_parachute(
            self, t, u, post_processing
        )
        self.udot_rail1 = lambda t, u, post_processing=False: udot_rail1(
            self, t, u, post_processing
        )
        self.udot_rail2 = lambda t, u, post_processing=False: udot_rail2(
            self, t, u, post_processing
        )
        self.udot_rail_earth_centered = lambda t, u, post_processing=False: (
            udot_rail_earth_centered(self, t, u, post_processing)
        )
        self.u_dot_earth_centered = lambda t, u, post_processing=False: (
            u_dot_earth_centered(self, t, u, post_processing)
        )

        if self.reference_frame == ReferenceFrame.GCRF:
            self.u_dot_generalized = self.u_dot_earth_centered
            return

        # Determine if a point-mass model is used.
        is_point_mass = self.rocket._is_point_mass or (
            hasattr(self.rocket, "motor") and self.rocket.motor._is_point_mass
        )
        is_solid_motor = hasattr(self.rocket, "motor") and isinstance(
            self.rocket.motor, SolidMotor
        )
        if is_point_mass and self.simulation_mode == "6DOF":
            inertia = getattr(self.rocket, "_point_mass_inertia", ())
            if not inertia or not all(value > 0 for value in inertia[:3]):
                raise ValueError(
                    "PointMassRocket can use 6DOF, but it needs a positive "
                    "three-axis inertia. Pass inertia=(I11, I22, I33) or select 3DOF."
                )

        if is_solid_motor and is_point_mass:
            if self.equations_of_motion != "solid_propulsion":
                warnings.warn(
                    "A SolidMotor was detected. Simulation will use "
                    "'solid_propulsion'.",
                    UserWarning,
                )
            self.equations_of_motion = "solid_propulsion"

        # Set the equations of motion based on the final simulation mode.
        if self.simulation_mode in ("3 DOF", "3DOF"):
            self.u_dot_generalized = self.u_dot_generalized_3dof
        elif self.simulation_mode in ("6 DOF", "6DOF"):
            self.u_dot_generalized = (
                self.u_dot
                if self.equations_of_motion == "solid_propulsion"
                else self.u_dot_generalized
            )
        else:
            raise ValueError(
                f"Invalid simulation_mode: {self.simulation_mode}. "
                "Must be '3DOF' or '6DOF'."
            )

    def __init_solver_monitors(self):
        # Initialize solver monitors
        self.function_evaluations = []
        # Initialize solution state
        self.solution = []
        self.__init_flight_state()

        self.t_initial = self.initial_solution[0]
        self.solution.append(self.initial_solution)
        self.t = self.solution[-1][0]
        self.y_sol = self.solution[-1][1:]

        self.__set_ode_solver(self.ode_solver)

    def __init_flight_state(self):
        """Initialize flight state variables."""
        if self.initial_state is not None:
            self.t_initial = 0.0
            self.initial_solution = [self.t_initial, *self.initial_state.to_array()]
            if self._launch_with_earth:
                self.out_of_rail_state = np.array([0])
                self.initial_derivative = self.udot_rail_earth_centered
            else:
                self.out_of_rail_state = self.initial_state.to_array()
                self.out_of_rail_time = self.t_initial
                self.out_of_rail_time_index = 0
                self.initial_derivative = self.u_dot_generalized
        elif self.initial_solution is None:
            # Initialize time and state variables
            self.t_initial = 0
            x_init, y_init, z_init = 0, 0, self.env.elevation
            vx_init, vy_init, vz_init = 0, 0, 0
            w1_init, w2_init, w3_init = 0, 0, 0
            # Initialize attitude
            # Precession / Heading Angle
            self.psi_init = np.radians(-self.heading)
            # Nutation / Attitude Angle
            self.theta_init = np.radians(self.inclination - 90)
            # Spin / Bank Angle
            self.phi_init = 0

            # Consider Rail Buttons position, if there is rail buttons
            try:
                self.phi_init += (
                    self.rocket.rail_buttons[0].component.angular_position_rad
                    if self.rocket._csys == 1
                    else 2 * np.pi
                    - self.rocket.rail_buttons[0].component.angular_position_rad
                )
            except IndexError:
                pass

            # 3-1-3 Euler Angles to Euler Parameters
            e0_init, e1_init, e2_init, e3_init = euler313_to_quaternions(
                self.phi_init, self.theta_init, self.psi_init
            )
            # Store initial conditions
            self.initial_solution = [
                self.t_initial,
                x_init,
                y_init,
                z_init,
                vx_init,
                vy_init,
                vz_init,
                e0_init,
                e1_init,
                e2_init,
                e3_init,
                w1_init,
                w2_init,
                w3_init,
            ]
            # Set initial derivative for rail phase
            self.initial_derivative = self.udot_rail1
        elif isinstance(self.initial_solution, Flight):
            # Initialize time and state variables based on last solution of
            # previous flight
            self.t_initial = self.initial_solution.solution[-1][0]
            self.initial_solution = self.initial_solution.solution[-1]
            # Set unused monitors
            self.out_of_rail_state = self.initial_solution[1:]
            self.out_of_rail_time = self.initial_solution[0]
            self.out_of_rail_time_index = 0
            # Set initial derivative for 6-DOF flight phase
            self.initial_derivative = self.u_dot_generalized
        else:
            self.t_initial = self.initial_solution[0]
            is_initial_state_out_of_rail = (
                self.initial_solution[1] ** 2
                + self.initial_solution[2] ** 2
                + (self.initial_solution[3] - self.env.elevation) ** 2
                >= self.effective_1rl**2
            )
            if is_initial_state_out_of_rail:
                self.out_of_rail_state = self.initial_solution[1:]
                self.out_of_rail_time = self.initial_solution[0]
                self.out_of_rail_time_index = 0
                self.initial_derivative = self.u_dot_generalized
            else:
                self.initial_derivative = self.udot_rail1
        # Get initial state derivative
        self.initial_derivative(
            self.t_initial, self.initial_solution[1:], post_processing=True
        )

    def __set_ode_solver(self, solver):
        """Sets the ODE solver to be used in the simulation.

        Parameters
        ----------
        solver : str, ``scipy.integrate.OdeSolver``
            Integration method to use to solve the equations of motion ODE,
            or a custom ``scipy.integrate.OdeSolver``.
        """
        if isinstance(solver, OdeSolver):
            self._solver = solver
        else:
            try:
                self._solver = ODE_SOLVER_MAP[solver]
            except KeyError as e:
                raise ValueError(
                    f"Invalid ``ode_solver`` input: {solver}. "
                    f"Available options are: {', '.join(ODE_SOLVER_MAP.keys())}"
                ) from e

        self.__is_lsoda = issubclass(self._solver, LSODA)

    def __simulate(self):
        """Simulate the flight trajectory."""
        self.__initialize_flight_phases()

        logger.info("Starting flight simulation of '%s'.", self.name)
        for phase_index, phase in self.flight_phases:
            if phase.name:
                logger.info(
                    "Entering flight phase '%s' at t=%.3f s.",
                    phase.name,
                    phase.t,
                )
            self.__simulate_phase(phase, phase_index)

        self.__finalize_simulation()

    def __initialize_flight_phases(self):
        """Initialize phase container for simulation execution."""
        self.flight_phases = _FlightPhases(
            t_initial=self.t_initial,
            initial_derivative=self.initial_derivative,
            max_time=self.max_time,
            verbose=self.verbose,
        )

    def __simulate_phase(self, phase, phase_index):
        """Run one Flight phase across all time nodes."""
        phase.solver = self._solver(
            phase.derivative,
            t0=phase.t,
            y0=self.solution[-1][1:],
            t_bound=phase.time_bound,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_time_step,
            min_step=self.min_time_step,
        )
        # Store baseline function evaluations
        self.function_evaluations.append(0)

        # Initialize phase time nodes
        self.__setup_phase_time_nodes(phase)
        self.__simulate_phase_nodes(phase, phase_index)

    def __setup_phase_time_nodes(self, phase):
        """Set up time nodes for the current phase.

        Parameters
        ----------
        phase : _FlightPhase
            The current flight phase.
        """
        phase.time_nodes = _TimeNodes()

        # Add first time node
        phase.time_nodes.add_node(phase.t, [])

        # Add last time node
        phase.time_nodes.add_node(phase.time_bound, [])

        # Add non-overshootable events as time nodes
        phase.time_nodes.add_event_list(
            self._non_overshootable_events, phase.t, phase.time_bound
        )

        # Organize time nodes
        phase.time_nodes.sort()
        phase.time_nodes.merge()

        # Clear first node events if phase is configured to do so
        if phase.clear:
            phase.time_nodes[0].events = []

    def __simulate_phase_nodes(self, phase, phase_index):
        """Run all nodes for a phase until completion."""
        for node_index, node in phase.time_nodes:
            self.__prepare_node_solver(phase, node, node_index)
            if self.__evaluate_node_events(phase, phase_index, node, node_index):
                self.__sync_node_solver_bound(phase, node, node_index)
            self.__run_node_solver_loop(phase, phase_index, node_index)

    def __prepare_node_solver(self, phase, node, node_index):
        """Update solver bounds/state before processing a node."""
        next_node = phase.time_nodes[node_index + 1]
        # Determine time bound for this time node
        node.time_bound = next_node.t
        # Update solver time bound and status to run until next node
        phase.solver.t_bound = node.time_bound
        if self.__is_lsoda:
            phase.solver._lsoda_solver._integrator.rwork[0] = phase.solver.t_bound
            phase.solver._lsoda_solver._integrator.call_args[4] = (
                phase.solver._lsoda_solver._integrator.rwork
            )
        phase.solver.status = "running"

    def __evaluate_node_events(self, phase, phase_index, node, node_index):
        """Node boundary event evaluation is handled by the scheduler loop."""
        return call_events(
            flight=self,
            events=node.events,
            phase=phase,
            phase_index=phase_index,
            node_index=node_index,
            time=node.t,
            state=self.solution[-1][1:],
            step_size=infer_step_size(self, node.t),
            needs=compute_needs_union(node.events),
        )

    def __sync_node_solver_bound(self, phase, node, node_index):
        """Re-synchronize solver t_bound after node events modified the schedule."""
        next_node = phase.time_nodes[node_index + 1]
        node.time_bound = next_node.t
        phase.solver.t_bound = node.time_bound
        if self.__is_lsoda:
            phase.solver._lsoda_solver._integrator.rwork[0] = phase.solver.t_bound
            phase.solver._lsoda_solver._integrator.call_args[4] = (
                phase.solver._lsoda_solver._integrator.rwork
            )

    def __run_node_solver_loop(self, phase, phase_index, node_index):
        """Advance solver for node interval while it remains running."""
        while phase.solver.status == "running":
            self.__execute_solver_step(phase)
            self.__process_events(phase, phase_index, node_index)
            self.__post_process_step(phase)

    def __execute_solver_step(self, phase):
        """Execute one solver step and update simulation history."""
        # Execute solver step, log solution and function evaluations
        phase.solver.step()
        if phase.solver.status == "failed":
            message = getattr(phase.solver, "message", "unknown integration error")
            raise RuntimeError(
                f"{type(phase.solver).__name__} failed at t={phase.solver.t:.9g} s: "
                f"{message}"
            )
        self.solution += [[phase.solver.t, *phase.solver.y]]
        self.function_evaluations.append(phase.solver.nfev)

        # Update time and state
        self.t = phase.solver.t
        self.y_sol = phase.solver.y
        logger.debug("Current simulation time: %.4f s", self.t)
        # Live single-line progress readout (overwritten in place via carriage
        # return). This is a console UX nicety, kept separate from logging.
        if self.verbose:
            print(f"Current Simulation Time: {self.t:3.4f} s", end="\r")

    def __process_events(self, phase, phase_index, node_index):
        # 1. Determine if we must rollback
        time, state, events_to_evaluate, rolled_back = (
            self.__process_overshootable_nodes(phase, phase_index, node_index)
        )

        # 2. Add continuous events to the pool and order based on priority
        events_to_evaluate.extend(phase.time_nodes.continuous_events)
        events_to_evaluate.sort(key=lambda event: event.priority)

        # 3. Evaluate combined events in sequence ordered by priority
        call_events(
            flight=self,
            events=events_to_evaluate,
            phase=phase,
            phase_index=phase_index,
            node_index=node_index,
            time=time,
            state=state,
            step_size=infer_step_size(self, time),
            needs=compute_needs_union(events_to_evaluate),
        )

        # 4. If overshoot processing rolled back the state but no command
        # changed phase/derivative, restart solver from the rolled back state
        if rolled_back and phase.solver.status == "running":
            self.__restart_phase_from_rollback(phase, phase_index, node_index)

    def __process_overshootable_nodes(self, phase, phase_index, node_index):
        """Checks for overshootable node triggers without executing state
        changes, rolls back if needed."""
        # Construct nodes considering all events discrete call times
        overshootable_nodes = self.__build_overshootable_nodes()

        if len(overshootable_nodes) < 2:
            return self.t, self.y_sol, [], False  # Early exit

        # Evaluate common kwargs parameters for event triggers.
        # Use needs from all currently-enabled overshootable events.
        step_needs = compute_needs_union(self._overshootable_events)
        event_kwargs = build_event_kwargs(
            self,
            self.t,
            self.y_sol,
            phase.solver.step_size,
            phase,
            rollback=True,
            needs=step_needs,
        )

        # Feed overshootable time nodes trigger
        events_to_evaluate = []
        interpolator = phase.solver.dense_output()
        for _, overshootable_node in overshootable_nodes:
            interpolated_state = interpolator(overshootable_node.t)
            interpolated_time = overshootable_node.t

            # Per-node needs: only enabled events scheduled at this node.
            node_needs = compute_needs_union(overshootable_node.events)
            event_kwargs = update_overshootable_event_kwargs(
                flight=self,
                phase=phase,
                event_kwargs=event_kwargs,
                interpolated_time=interpolated_time,
                interpolated_state=interpolated_state,
                needs=node_needs,
            )

            # Check events for current node
            rolled_back = False
            for event in overshootable_node.events:
                rolled_back, should_recall = process_overshootable_event(
                    flight=self,
                    event=event,
                    event_kwargs=event_kwargs,
                    phase=phase,
                    phase_index=phase_index,
                    node_index=node_index,
                    rolled_back=rolled_back,
                )
                if should_recall:
                    events_to_evaluate.append(event)

            if rolled_back:
                return interpolated_time, interpolated_state, events_to_evaluate, True

        return self.t, self.y_sol, events_to_evaluate, False

    def __restart_phase_from_rollback(self, phase, _phase_index, node_index):
        """Restart integration from rollback state.

        If the rollback did not change the flight phase or derivative, prefer a
        lightweight in-place solver reinitialization instead of creating a new
        phase. This keeps `dense_output()` and solution history coherent while
        avoiding the overhead of creating a separate phase object.
        """

        # Trim future nodes and insert a node at the rollback time.
        phase.time_nodes.flush_after(node_index)
        phase.time_nodes.add_node(self.t, [])

        # Reinitialize the solver in-place so integration continues from the
        # rolled-back time/state using the same derivative and solver settings.
        phase.solver = self._solver(
            phase.derivative,
            t0=self.t,
            y0=self.y_sol,
            t_bound=phase.time_bound,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_time_step,
            min_step=self.min_time_step,
        )

        # Align function-evaluation tracking with a fresh solver lifecycle.
        self.function_evaluations.append(0)

        # Ensure the solver is marked running so the node loop continues.
        phase.solver.status = "running"

    def __build_overshootable_nodes(self):
        """Build the overshootable node list for the current solver step."""
        overshootable_nodes = _TimeNodes()
        overshootable_nodes.add_event_list(
            self._overshootable_events,
            self.solution[-2][0],
            self.t,
        )

        # Add last time node (always skipped).
        overshootable_nodes.add_node(self.t, [])

        overshootable_nodes.sort()
        overshootable_nodes.merge()

        # Remove first node if it is the same as current time.
        if overshootable_nodes and overshootable_nodes[0].t == self.solution[-2][0]:
            del overshootable_nodes.list[0]

        return overshootable_nodes

    def __post_process_step(self, phase):
        """Run derivative post-processing to capture parameter changes made by controllers.

        This method recalculates derivatives with post_processing=True flag to capture
        any aerodynamic or other parameter changes that resulted from controller actions.
        The post_processing flag in the derivative ensures that forces, moments, and
        other derived quantities are properly recorded for later analysis.

        Notes
        -----
        This is critical for accurate post-processing variable tracking when controllers
        modify rocket parameters (e.g., cant angle, air brake deflection) that affect
        aerodynamic forces and moments.
        """
        if self._has_change_dynamics_events:
            phase.derivative(self.t, self.y_sol, post_processing=True)

    def __finalize_simulation(self):
        """Finalize and cache simulation outputs."""
        self.t_final = self.t
        if self._has_change_dynamics_events:
            # cache post process variables
            self.__evaluate_post_process = np.array(self.__post_processed_variables)
        if self.sensors:
            self.__cache_sensor_data()
        logger.info("Simulation completed at time: %.4f s", self.t)

    def __cache_sensor_data(self):
        """Cache sensor data for simulations with sensors."""
        sensor_data = {}
        sensors = []
        for sensor in self.sensors:
            # skip sensors that are used more then once in the rocket
            if sensor not in sensors:
                sensors.append(sensor)
                sensor_data[sensor] = sensor.measured_data[:]
        self.sensor_data = sensor_data

    def get_solution_at_time(self, t, atol=1e-3):
        """Returns the solution state vector at a given time. If the time is
        not found in the solution, the closest time is used and a warning is
        raised.

        Parameters
        ----------
        t : float
            Time in seconds.
        atol : float, optional
            Absolute tolerance for time comparison. Default is 1e-3. If the
            difference between the time and the closest time in the solution
            is greater than this value, a warning is raised.

        Returns
        -------
        solution : np.array
            Solution state at time t. The array follows the format of the
            solution array, with the first column being time like this:
            [t, x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3].

        """
        time_index = find_closest(self.time, t)
        if abs(self.time[time_index] - t) > atol:
            warnings.warn(
                f"Time {t} not found in solution. Closest time is "
                f"{self.time[time_index]}. Using closest time.",
                UserWarning,
            )
        return self.solution_array[time_index, :]

    @cached_property
    def effective_1rl(self):
        """Original rail length minus the distance measured from nozzle exit
        to the upper rail button. It assumes the nozzle to be aligned with
        the beginning of the rail."""
        nozzle = self.rocket.nozzle_position
        try:
            rail_buttons = self.rocket.rail_buttons[0]
            upper_r_button = (
                rail_buttons.component.buttons_distance * self.rocket._csys
                + rail_buttons.position.z
            )
        except IndexError:  # No rail buttons defined
            upper_r_button = nozzle
        effective_1rl = self.rail_length - abs(nozzle - upper_r_button)
        return effective_1rl

    @cached_property
    def effective_2rl(self):
        """Original rail length minus the distance measured from nozzle exit
        to the lower rail button. It assumes the nozzle to be aligned with
        the beginning of the rail."""
        nozzle = self.rocket.nozzle_position
        try:
            rail_buttons = self.rocket.rail_buttons[0]
            lower_r_button = rail_buttons.position.z
        except IndexError:  # No rail buttons defined
            lower_r_button = nozzle
        effective_2rl = self.rail_length - abs(nozzle - lower_r_button)
        return effective_2rl

    @cached_property
    def frontal_surface_wind(self):
        """Frontal wind velocity at the surface level. The frontal wind is
        defined as the wind blowing in the direction of the rocket's heading.

        Returns
        -------
        float
            Wind velocity in the frontal direction at the surface level.
        """
        environment = (
            self.earth.local_environment if self.earth is not None else self.env
        )
        wind_u = environment.wind_velocity_x.get_value_opt(environment.elevation)
        wind_v = environment.wind_velocity_y.get_value_opt(environment.elevation)
        heading_rad = self.heading * np.pi / 180
        return wind_u * np.sin(heading_rad) + wind_v * np.cos(heading_rad)

    @cached_property
    def lateral_surface_wind(self):
        """Lateral wind velocity at the surface level. The lateral wind is
        defined as the wind blowing perpendicular to the rocket's heading.

        Returns
        -------
        float
            Wind velocity in the lateral direction at the surface level.
        """
        environment = (
            self.earth.local_environment if self.earth is not None else self.env
        )
        wind_u = environment.wind_velocity_x.get_value_opt(environment.elevation)
        wind_v = environment.wind_velocity_y.get_value_opt(environment.elevation)
        heading_rad = self.heading * np.pi / 180

        return -wind_u * np.cos(heading_rad) + wind_v * np.sin(heading_rad)

    @cached_property
    def solution_array(self):
        """Returns solution array of the rocket flight."""
        return np.array(self.solution)

    @property
    def function_evaluations_per_time_step(self):
        """Get the number of function evaluations per time step. This method
        calculates the difference between consecutive function evaluations
        during numerical integration and returns it as a list.

        Returns
        -------
        list
            The list of differences in function evaluations per time step.
        """
        return np.diff(self.function_evaluations).tolist()

    @cached_property
    def time(self):
        """Returns time array from solution."""
        return self.solution_array[:, 0]

    @cached_property
    def time_steps(self):
        """Returns time step array."""
        return np.diff(self.time)

    @cached_property
    def _maximum_geodetic_altitude(self):
        """Highest geodetic altitude used to select the ``all_info`` view."""
        return float(np.nanmax(self.altitude[:, 1]))

    def _first_geodetic_altitude_crossing(self, altitude):
        """Return the first interpolated upward crossing of ``altitude``.

        This private presentation helper deliberately uses geodetic altitude
        for Earth-centred Flights and the established altitude output for
        flat-Earth Flights. ``None`` is returned when the threshold is not
        reached.
        """
        samples = np.asarray(self.altitude[:, :], dtype=float)
        values = samples[:, 1]
        reached = np.flatnonzero(values >= float(altitude))
        if len(reached) == 0:
            return None
        upper_index = int(reached[0])
        if upper_index == 0:
            return float(samples[0, 0])
        t0, h0 = samples[upper_index - 1]
        t1, h1 = samples[upper_index]
        if h1 == h0:
            return float(t1)
        # Refine against the Function interpolation used everywhere else in
        # the presentation, so the boundary evaluates to the requested height
        # rather than merely crossing it in the accepted-state polyline.
        lower_time, upper_time = float(t0), float(t1)
        threshold = float(altitude)
        for _ in range(48):
            midpoint = 0.5 * (lower_time + upper_time)
            if self.altitude(midpoint) < threshold:
                lower_time = midpoint
            else:
                upper_time = midpoint
        return 0.5 * (lower_time + upper_time)

    def position(self, frame=None):
        """Return the position history in a requested reference frame.

        Parameters
        ----------
        frame : ReferenceFrame or str, optional
            Output frame. Defaults to the Flight integration frame.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n, 3)`` in meters.
        """
        target = self.reference_frame if frame is None else ReferenceFrame.coerce(frame)
        positions = self.solution_array[:, 1:4]
        if target == self.reference_frame:
            return positions.copy()
        transformed = []
        for time, position in zip(self.time, positions):
            output, _, _ = self.datum.transform_kinematics(
                self.current_epoch(time),
                position,
                source=self.reference_frame,
                target=target,
            )
            transformed.append(output)
        return np.asarray(transformed)

    def velocity(self, frame=None):
        """Return the velocity history in a requested reference frame, in m/s."""
        target = self.reference_frame if frame is None else ReferenceFrame.coerce(frame)
        positions = self.solution_array[:, 1:4]
        velocities = self.solution_array[:, 4:7]
        if target == self.reference_frame:
            return velocities.copy()
        transformed = []
        for time, position, velocity in zip(self.time, positions, velocities):
            _, output, _ = self.datum.transform_kinematics(
                self.current_epoch(time),
                position,
                velocity,
                source=self.reference_frame,
                target=target,
            )
            transformed.append(output)
        return np.asarray(transformed)

    def position_local(self):
        """Return launch-centred east/north/up position history in meters.

        This is the familiar low_altitude-simulation view while the underlying
        Earth launch remains integrated in its datum's inertial frame. It is
        available for rail-launched Flights, not arbitrary orbital initial
        states.
        """
        if self.reference_frame == ReferenceFrame.FLAT_EARTH:
            local = self.position().copy()
            local[:, 2] -= self.env.elevation
            return local
        if not hasattr(self, "_launch_site_fixed"):
            raise AttributeError(
                "Launch-local coordinates require a rail-launched Flight."
            )
        local = []
        for time, position in zip(self.time, self.solution_array[:, 1:4]):
            position_fixed, _, _ = self.datum._to_ecef_coordinates(
                self.current_epoch(time), position
            )
            local.append(
                self.datum.to_topocentric(
                    position_fixed,
                    latitude_rad=self._launch_latitude_rad,
                    longitude_rad=self._launch_longitude_rad,
                    altitude=self._launch_elevation,
                )[0]
            )
        return np.asarray(local)

    def velocity_local(self):
        """Return launch-local east/north/up velocity history in m/s."""
        if self.reference_frame == ReferenceFrame.FLAT_EARTH:
            return self.velocity()
        if not hasattr(self, "_launch_site_fixed"):
            raise AttributeError(
                "Launch-local coordinates require a rail-launched Flight."
            )
        local = []
        for time, row in zip(self.time, self.solution_array):
            _, velocity_fixed, _ = self.datum._to_ecef_coordinates(
                self.current_epoch(time), row[1:4], row[4:7]
            )
            local.append(
                self.datum.to_topocentric(
                    self._launch_site_fixed,
                    velocity_fixed,
                    latitude_rad=self._launch_latitude_rad,
                    longitude_rad=self._launch_longitude_rad,
                    altitude=self._launch_elevation,
                )[1]
            )
        return np.asarray(local)

    def acceleration_local(self):
        """Return launch-local east/north/up acceleration history in m/s².

        For Earth-centred Flights this includes the rotating-frame kinematic
        terms supplied by the Flight datum, so it is the acceleration seen in
        the Earth-fixed low_altitude view rather than the raw GCRF derivative.
        """
        if self.reference_frame == ReferenceFrame.FLAT_EARTH:
            return np.column_stack((self.ax[:, 1], self.ay[:, 1], self.az[:, 1]))
        if not hasattr(self, "_launch_site_fixed"):
            raise AttributeError(
                "Launch-local coordinates require a rail-launched Flight."
            )
        local = []
        accelerations = np.column_stack((self.ax[:, 1], self.ay[:, 1], self.az[:, 1]))
        for time, row, acceleration in zip(
            self.time, self.solution_array, accelerations
        ):
            _, _, acceleration_fixed = self.datum._to_ecef_coordinates(
                self.current_epoch(time),
                row[1:4],
                row[4:7],
                acceleration,
            )
            local.append(
                self.datum.to_topocentric(
                    self._launch_site_fixed,
                    acceleration_itrs=acceleration_fixed,
                    latitude_rad=self._launch_latitude_rad,
                    longitude_rad=self._launch_longitude_rad,
                    altitude=self._launch_elevation,
                )[2]
            )
        return np.asarray(local)

    def attitude_local(self):
        """Return body-to-launch-ENU rotation matrices over the Flight."""
        # ``direction_cosine_matrixes`` is integration-to-body for use in the
        # force equations; transpose it here to obtain body-to-integration.
        body_to_integration = np.transpose(self.direction_cosine_matrixes, (0, 2, 1))
        if self.reference_frame == ReferenceFrame.FLAT_EARTH:
            return body_to_integration.copy()
        if not hasattr(self, "_launch_site_fixed"):
            raise AttributeError(
                "Launch-local attitude requires a rail-launched Flight."
            )
        matrices = []
        for time, body_to_inertial in zip(self.time, body_to_integration):
            inertial_to_fixed = self.datum.orientation_matrix(self.current_epoch(time))
            matrices.append(
                self._launch_fixed_to_enu @ inertial_to_fixed @ body_to_inertial
            )
        return np.asarray(matrices)

    def attitude_local_quaternions(self):
        """Return scalar-first body-to-launch-ENU quaternions."""
        quaternions = []
        for matrix in self.attitude_local():
            quaternion = np.asarray(quaternion_from_matrix(matrix), dtype=float)
            if quaternions and np.dot(quaternion, quaternions[-1]) < 0:
                quaternion = -quaternion
            quaternions.append(quaternion)
        return np.asarray(quaternions)

    def attitude_local_euler_angles(self):
        """Return local precession, nutation and spin angles in degrees."""
        quaternions = self.attitude_local_quaternions()
        q0, q1, q2, q3 = quaternions.T
        return np.column_stack(
            (
                quaternions_to_precession(q0, q1, q2, q3),
                quaternions_to_nutation(q1, q2),
                quaternions_to_spin(q0, q1, q2, q3),
            )
        )

    def state_at_time(self, time):
        """Return the interpolated typed state in the integration frame."""
        return FlightState.cartesian(
            epoch=self.current_epoch(time),
            position=[self.x(time), self.y(time), self.z(time)],
            velocity=[self.vx(time), self.vy(time), self.vz(time)],
            quaternion=[self.e0(time), self.e1(time), self.e2(time), self.e3(time)],
            angular_velocity=[self.w1(time), self.w2(time), self.w3(time)],
            frame=self.reference_frame,
            elapsed_time=self.mission_time_offset + float(time) - self.t_initial,
        )

    @cached_property
    def epochs(self):
        """Tuple of Epoch values corresponding to accepted solver states."""
        return tuple(self.current_epoch(time) for time in self.time)

    @cached_property
    def orbit(self):
        """Osculating orbital output view for a GCRF Flight."""
        if self.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError(
                "Orbital elements are available only for GCRF Flight simulations."
            )
        return FlightOrbit(self)

    @cached_property
    def orbital_accelerations(self):
        """Mapping of orbital acceleration contributions to vector Functions.

        Each Function returns the GCRF x, y and z acceleration components in
        m/s². The mapping is available only for Earth-centered simulations.
        """
        if self.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError(
                "Orbital acceleration outputs require a GCRF Flight simulation."
            )
        self.__evaluate_post_process
        names = []
        for record in self._orbital_force_history:
            for name in record:
                if name != "time" and name not in names:
                    names.append(name)
        outputs = {}
        for name in names:
            rows = [
                [record["time"], *record.get(name, np.zeros(3))]
                for record in self._orbital_force_history
            ]
            outputs[name] = VectorFunction(
                rows,
                inputs="Time (s)",
                outputs=[
                    f"{name} X (m/s²)",
                    f"{name} Y (m/s²)",
                    f"{name} Z (m/s²)",
                ],
            )
        return outputs

    @cached_property
    def orbital_accelerations_rtn(self):
        """Orbital acceleration contributions resolved in instantaneous RTN."""
        if self.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError(
                "RTN acceleration outputs require a GCRF Flight simulation."
            )
        outputs = {}
        positions = self.position(ReferenceFrame.GCRF)
        velocities = self.velocity(ReferenceFrame.GCRF)
        for name, function in self.orbital_accelerations.items():
            rows = []
            for index, (time, acceleration) in enumerate(
                zip(function[:, 0], function[:, 1:4])
            ):
                rotation = self.datum.rtn_matrix(positions[index], velocities[index])
                rows.append([time, *(rotation @ acceleration)])
            outputs[name] = VectorFunction(
                rows,
                inputs="Time (s)",
                outputs=[
                    f"{name} Radial (m/s²)",
                    f"{name} Transverse (m/s²)",
                    f"{name} Normal (m/s²)",
                ],
            )
        return outputs

    @cached_property
    def specific_orbital_energy(self):
        """Specific mechanical orbital energy in J/kg versus time."""
        if self.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError("Orbital energy requires a GCRF Flight simulation.")
        position = self.position()
        velocity = self.velocity()
        mu = self.datum.gravitational_parameter
        values = 0.5 * np.sum(velocity**2, axis=1) - mu / np.linalg.norm(
            position, axis=1
        )
        return Function(
            np.column_stack((self.time, values)),
            inputs="Time (s)",
            outputs="Specific Orbital Energy (J/kg)",
            interpolation="linear",
            extrapolation="constant",
        )

    @cached_property
    def specific_angular_momentum(self):
        """Specific angular-momentum vector in GCRF, in m²/s."""
        if self.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError(
                "Orbital angular momentum requires a GCRF Flight simulation."
            )
        values = np.cross(self.position(), self.velocity())
        return VectorFunction(
            np.column_stack((self.time, values)),
            inputs="Time (s)",
            outputs=("Angular Momentum X", "Angular Momentum Y", "Angular Momentum Z"),
        )

    def ground_station_observation(
        self,
        latitude,
        longitude,
        *,
        altitude=0.0,
        minimum_elevation=0.0,
    ):
        """Return range, elevation, azimuth, and visibility from a station.

        Latitude, longitude, and minimum elevation are expressed in degrees;
        station altitude and returned range are meters.
        """
        if self.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError(
                "Ground-station observations require a GCRF Flight simulation."
            )
        topocentric = self.datum.to_topocentric(
            self.position(ReferenceFrame.ITRF),
            latitude_rad=np.radians(latitude),
            longitude_rad=np.radians(longitude),
            altitude=altitude,
        )[0]
        ranges = np.linalg.norm(topocentric, axis=1)
        elevation = np.degrees(
            np.arcsin(
                np.divide(
                    topocentric[:, 2],
                    ranges,
                    out=np.zeros_like(ranges),
                    where=ranges > 0.0,
                )
            )
        )
        azimuth = np.degrees(np.arctan2(topocentric[:, 0], topocentric[:, 1])) % 360
        visible = elevation >= float(minimum_elevation)
        common = dict(
            inputs="Time (s)", interpolation="linear", extrapolation="constant"
        )
        return {
            "range": Function(
                np.column_stack((self.time, ranges)), outputs="Range (m)", **common
            ),
            "elevation": Function(
                np.column_stack((self.time, elevation)),
                outputs="Elevation (°)",
                **common,
            ),
            "azimuth": Function(
                np.column_stack((self.time, azimuth)), outputs="Azimuth (°)", **common
            ),
            "visible": Function(
                np.column_stack((self.time, visible.astype(float))),
                outputs="Visible",
                **common,
            ),
        }

    # Process first type of outputs - state vector
    # Transform solution array into Functions
    @funcify_method("Time (s)", "X (m)", "spline", "constant")
    def x(self):
        """Rocket x position relative to the launch pad as a Function of
        time."""
        return self.solution_array[:, [0, 1]]

    @funcify_method("Time (s)", "Y (m)", "spline", "constant")
    def y(self):
        """Rocket y position relative to the launch pad as a Function of
        time."""
        return self.solution_array[:, [0, 2]]

    @funcify_method("Time (s)", "Z (m)", "spline", "constant")
    def z(self):
        """Rocket z position relative to the launch pad as a Function of
        time."""
        return self.solution_array[:, [0, 3]]

    @funcify_method("Time (s)", "Altitude AGL (m)", "spline", "constant")
    def altitude(self):
        """Rocket altitude above ground level as a Function of time. Ground
        level is defined by the environment elevation."""
        if self.reference_frame == ReferenceFrame.GCRF:
            altitude = [
                self.datum.itrs_to_geodetic(position_itrf)[2]
                for position_itrf in self.position(ReferenceFrame.ITRF)
            ]
            if hasattr(self, "_launch_site_fixed"):
                altitude = np.asarray(altitude) - self._launch_elevation
            return np.column_stack((self.time, altitude))
        return self.z - self.env.elevation

    @funcify_method("Time (s)", "Vx (m/s)", "spline", "zero")
    def vx(self):
        """Velocity of the rocket's center of dry mass in the X (East) direction
        of the inertial frame as a function of time."""
        return self.solution_array[:, [0, 4]]

    @funcify_method("Time (s)", "Vy (m/s)", "spline", "zero")
    def vy(self):
        """Velocity of the rocket's center of dry mass in the Y (North)
        direction of the inertial frame as a function of time."""
        return self.solution_array[:, [0, 5]]

    @funcify_method("Time (s)", "Vz (m/s)", "spline", "zero")
    def vz(self):
        """Velocity of the rocket's center of dry mass in the Z (Up) direction of
        the inertial frame as a function of time."""
        return self.solution_array[:, [0, 6]]

    @funcify_method("Time (s)", "e0", "spline", "constant")
    def e0(self):
        """Rocket quaternion e0 as a Function of time."""
        return self.solution_array[:, [0, 7]]

    @funcify_method("Time (s)", "e1", "spline", "constant")
    def e1(self):
        """Rocket quaternion e1 as a Function of time."""
        return self.solution_array[:, [0, 8]]

    @funcify_method("Time (s)", "e2", "spline", "constant")
    def e2(self):
        """Rocket quaternion e2 as a Function of time."""
        return self.solution_array[:, [0, 9]]

    @funcify_method("Time (s)", "e3", "spline", "constant")
    def e3(self):
        """Rocket quaternion e3 as a Function of time."""
        return self.solution_array[:, [0, 10]]

    @funcify_method("Time (s)", "ω1 (rad/s)", "spline", "zero")
    def w1(self):
        """Angular velocity of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        pitch rate (q)."""
        return self.solution_array[:, [0, 11]]

    @funcify_method("Time (s)", "ω2 (rad/s)", "spline", "zero")
    def w2(self):
        """Angular velocity of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw rate (r)."""
        return self.solution_array[:, [0, 12]]

    @funcify_method("Time (s)", "ω3 (rad/s)", "spline", "zero")
    def w3(self):
        """Angular velocity of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll rate (p)."""
        return self.solution_array[:, [0, 13]]

    # Process second type of outputs - accelerations components
    @cached_property
    def __evaluate_post_process(self):
        """Evaluate all post-processing variables by running the simulation
        again but with the post-processing flag set to True.

        Returns
        -------
        np.array
            An array containing all post-processed variables evaluated at each
            time step. Each element of the array is a list containing:
            [t, ax, ay, az, alpha1, alpha2, alpha3, R1, R2, R3, M1, M2, M3, net_thrust]
        """
        self.__post_processed_variables = []
        if self.reference_frame == ReferenceFrame.GCRF:
            self._orbital_force_history = []
        step_times = [step[0] for step in self.solution]
        for phase_index, phase in self.flight_phases:
            init_time = phase.t
            final_time = self.flight_phases[phase_index + 1].t
            current_derivative = phase.derivative
            self._active_parachute = phase.parachute
            # Select the steps with init_time < t <= final_time. The first
            # phase also includes the step at exactly t_initial, which the
            # strict lower bound would otherwise drop.
            if init_time == self.t_initial:
                start = bisect_left(step_times, init_time)
            else:
                start = bisect_right(step_times, init_time)
            end = bisect_right(step_times, final_time)
            for step in self.solution[start:end]:
                current_derivative(step[0], step[1:], post_processing=True)

        return np.array(self.__post_processed_variables)

    @funcify_method("Time (s)", "Ax (m/s²)", "spline", "zero")
    def ax(self):
        """Acceleration of the rocket's center of dry mass along the X (East)
        axis in the inertial frame as a function of time."""
        return self.__evaluate_post_process[:, [0, 1]]

    @funcify_method("Time (s)", "Ay (m/s²)", "spline", "zero")
    def ay(self):
        """Acceleration of the rocket's center of dry mass along the Y (North)
        axis in the inertial frame as a function of time."""
        return self.__evaluate_post_process[:, [0, 2]]

    @funcify_method("Time (s)", "Az (m/s²)", "spline", "zero")
    def az(self):
        """Acceleration of the rocket's center of dry mass along the Z (Up)
        axis in the inertial frame as a function of time."""
        return self.__evaluate_post_process[:, [0, 3]]

    @funcify_method("Time (s)", "α1 (rad/s²)", "spline", "zero")
    def alpha1(self):
        """Angular acceleration of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        pitch acceleration."""
        return self.__evaluate_post_process[:, [0, 4]]

    @funcify_method("Time (s)", "α2 (rad/s²)", "spline", "zero")
    def alpha2(self):
        """Angular acceleration of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw acceleration."""
        return self.__evaluate_post_process[:, [0, 5]]

    @funcify_method("Time (s)", "α3 (rad/s²)", "spline", "zero")
    def alpha3(self):
        """Angular acceleration of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll acceleration."""
        return self.__evaluate_post_process[:, [0, 6]]

    # Process third type of outputs - Temporary values
    @funcify_method("Time (s)", "R1 (N)", "spline", "zero")
    def R1(self):
        """Aerodynamic force acting along the x-axis of the rocket's body frame
        as a function of time. Expressed in Newtons (N)."""
        return self.__evaluate_post_process[:, [0, 7]]

    @funcify_method("Time (s)", "R2 (N)", "spline", "zero")
    def R2(self):
        """Aerodynamic force acting along the y-axis of the rocket's body frame
        as a function of time. Expressed in Newtons (N)."""
        return self.__evaluate_post_process[:, [0, 8]]

    @funcify_method("Time (s)", "R3 (N)", "spline", "zero")
    def R3(self):
        """Aerodynamic force acting along the z-axis of the rocket's body frame
        as a function of time. Expressed in Newtons (N)."""
        return self.__evaluate_post_process[:, [0, 9]]

    @funcify_method("Time (s)", "M1 (Nm)", "linear", "zero")
    def M1(self):
        """Aerodynamic moment acting along the x-axis of the rocket's body
        frame as a function of time. Expressed in Newtons (N)."""
        return self.__evaluate_post_process[:, [0, 10]]

    @funcify_method("Time (s)", "M2 (Nm)", "linear", "zero")
    def M2(self):
        """Aerodynamic moment acting along the y-axis of the rocket's body
        frame as a function of time. Expressed in Newtons (N)."""
        return self.__evaluate_post_process[:, [0, 11]]

    @funcify_method("Time (s)", "M3 (Nm)", "linear", "zero")
    def M3(self):
        """Aerodynamic moment acting along the z-axis of the rocket's body
        frame as a function of time. Expressed in Newtons (N)."""
        return self.__evaluate_post_process[:, [0, 12]]

    @funcify_method("Time (s)", "Net Thrust (N)", "linear", "zero")
    def net_thrust(self):
        """Net thrust of the rocket as a Function of time. This is the
        actual thrust force experienced by the rocket. It may be corrected
        with the atmospheric pressure if a reference pressure is defined."""
        return self.__evaluate_post_process[:, [0, 13]]

    @funcify_method("Time (s)", "Pressure (Pa)", "spline", "constant")
    def pressure(self):
        """Air pressure felt by the rocket as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            return [
                (time, sample[0].pressure)
                for time, sample in zip(self.time, self._orbital_atmosphere_samples)
            ]
        return [(t, self.env.pressure.get_value_opt(z)) for t, z in self.z]

    @funcify_method("Time (s)", "Density (kg/m³)", "spline", "constant")
    def density(self):
        """Air density felt by the rocket as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            return [
                (time, sample[0].density)
                for time, sample in zip(self.time, self._orbital_atmosphere_samples)
            ]
        return [(t, self.env.density.get_value_opt(z)) for t, z in self.z]

    @funcify_method("Time (s)", "Dynamic Viscosity (Pa s)", "spline", "constant")
    def dynamic_viscosity(self):
        """Air dynamic viscosity felt by the rocket as a Function of
        time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            return [
                (time, sample[0].dynamic_viscosity)
                for time, sample in zip(self.time, self._orbital_atmosphere_samples)
            ]
        return [(t, self.env.dynamic_viscosity.get_value_opt(z)) for t, z in self.z]

    @funcify_method("Time (s)", "Speed of Sound (m/s)", "linear", "constant")
    def speed_of_sound(self):
        """Speed of sound in the air felt by the rocket as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            return [
                (time, sample[0].speed_of_sound)
                for time, sample in zip(self.time, self._orbital_atmosphere_samples)
            ]
        return [(t, self.env.speed_of_sound.get_value_opt(z)) for t, z in self.z]

    @funcify_method("Time (s)", "Wind Velocity X (East) (m/s)", "spline", "constant")
    def wind_velocity_x(self):
        """Wind velocity in the X direction (east) as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            return np.column_stack(
                (
                    self.time,
                    [sample[1][0] for sample in self._orbital_atmosphere_samples],
                )
            )
        return [(t, self.env.wind_velocity_x.get_value_opt(z)) for t, z in self.z]

    @funcify_method("Time (s)", "Wind Velocity Y (North) (m/s)", "spline", "constant")
    def wind_velocity_y(self):
        """Wind velocity in the Y direction (north) as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            return np.column_stack(
                (
                    self.time,
                    [sample[1][1] for sample in self._orbital_atmosphere_samples],
                )
            )
        return [(t, self.env.wind_velocity_y.get_value_opt(z)) for t, z in self.z]

    @cached_property
    def _orbital_atmosphere_samples(self):
        """Atmospheric states and inertial atmospheric velocities."""
        if self.reference_frame != ReferenceFrame.GCRF:
            return ()
        samples = []
        for solution in self.solution_array:
            time = solution[0]
            epoch = self.current_epoch(time)
            position_itrf, velocity_itrf, _ = self.datum.transform_kinematics(
                epoch,
                solution[1:4],
                solution[4:7],
                source=ReferenceFrame.GCRF,
                target=ReferenceFrame.ITRF,
            )
            state_itrf = FlightState.cartesian(
                epoch=epoch,
                position=position_itrf,
                velocity=velocity_itrf,
                quaternion=solution[7:11],
                angular_velocity=solution[11:14],
                frame=ReferenceFrame.ITRF,
                elapsed_time=time,
            )
            atmosphere = self.env.evaluate_atmosphere(epoch, state_itrf)
            _, atmospheric_velocity_gcrf, _ = self.datum.transform_kinematics(
                epoch,
                position_itrf,
                atmosphere.wind_velocity,
                source=ReferenceFrame.ITRF,
                target=ReferenceFrame.GCRF,
            )
            samples.append((atmosphere, atmospheric_velocity_gcrf))
        return tuple(samples)

    # Process fourth type of output - values calculated from previous outputs
    # Frame conversions
    @cached_property
    def _stacked_velocity_body_frame(self):
        """Stacked velocity array at the center of dry mass in the body frame
        at each time step."""
        Kt = self.direction_cosine_matrixes
        v = np.array(
            [
                self.vx.y_array,
                self.vy.y_array,
                self.vz.y_array,
            ]
        ).transpose()
        stacked_velocity_body_frame = np.squeeze(np.matmul(Kt, v[:, :, np.newaxis]))
        return stacked_velocity_body_frame

    @funcify_method("Time (s)", "Vx Body Frame (m/s)", "spline", "zero")
    def vx_body_frame(self):
        """Velocity of the rocket's center of dry mass along the x axis of the
        body frame as a function of time."""
        return np.array(
            [self.time, self._stacked_velocity_body_frame[:, 0]]
        ).transpose()

    @funcify_method("Time (s)", "Vy Body Frame (m/s)", "spline", "zero")
    def vy_body_frame(self):
        """Velocity of the rocket's center of dry mass along the y axis of the
        body frame as a function of time."""
        return np.array(
            [self.time, self._stacked_velocity_body_frame[:, 1]]
        ).transpose()

    @funcify_method("Time (s)", "Vz Body Frame (m/s)", "spline", "zero")
    def vz_body_frame(self):
        """Velocity of the rocket's center of dry mass along the z axis of the
        body frame as a function of time."""
        return np.array(
            [self.time, self._stacked_velocity_body_frame[:, 2]]
        ).transpose()

    @cached_property
    def _stacked_acceleration_body_frame(self):
        """Stacked acceleration array at the center of dry mass in the body
        frame at each time step."""
        Kt = self.direction_cosine_matrixes
        a = np.array(
            [
                self.ax.y_array,
                self.ay.y_array,
                self.az.y_array,
            ]
        ).transpose()
        stacked_acceleration_body_frame = np.squeeze(np.matmul(Kt, a[:, :, np.newaxis]))
        return stacked_acceleration_body_frame

    @funcify_method("Time (s)", "Ax Body Frame (m/s²)", "spline", "zero")
    def ax_body_frame(self):
        """Acceleration of the rocket's center of dry mass along the x axis of the
        body frame as a function of time."""
        return np.array(
            [self.time, self._stacked_acceleration_body_frame[:, 0]]
        ).transpose()

    @funcify_method("Time (s)", "Ay Body Frame (m/s²)", "spline", "zero")
    def ay_body_frame(self):
        """Acceleration of the rocket's center of dry mass along the y axis of the
        body frame as a function of time."""
        return np.array(
            [self.time, self._stacked_acceleration_body_frame[:, 1]]
        ).transpose()

    @funcify_method("Time (s)", "Az Body Frame (m/s²)", "spline", "zero")
    def az_body_frame(self):
        """Acceleration of the rocket's center of dry mass along the z axis of the
        body frame as a function of time."""
        return np.array(
            [self.time, self._stacked_acceleration_body_frame[:, 2]]
        ).transpose()

    # Kinematics functions and values
    # Velocity Magnitude
    @funcify_method("Time (s)", "Speed - Velocity Magnitude (m/s)")
    def speed(self):
        """Earth-fixed speed for launches, or integration-frame speed otherwise."""
        if hasattr(self, "_launch_site_fixed"):
            return np.column_stack(
                (self.time, np.linalg.norm(self.velocity_local(), axis=1))
            )
        return (self.vx**2 + self.vy**2 + self.vz**2) ** 0.5

    @property
    def out_of_rail_velocity(self):
        """Velocity at which the rocket leaves the launch rail."""
        return self.speed.get_value_opt(self.out_of_rail_time)

    @cached_property
    def max_speed_time(self):
        """Time at which the rocket reaches its maximum speed."""
        max_speed_time_index = np.argmax(self.speed.get_source()[:, 1])
        return self.speed[max_speed_time_index, 0]

    @cached_property
    def max_speed(self):
        """Maximum speed reached by the rocket."""
        return self.speed.get_value_opt(self.max_speed_time)

    # Accelerations
    @funcify_method("Time (s)", "acceleration Magnitude (m/s²)")
    def acceleration(self):
        """Earth-fixed acceleration for launches, or frame magnitude otherwise."""
        if hasattr(self, "_launch_site_fixed"):
            return np.column_stack(
                (self.time, np.linalg.norm(self.acceleration_local(), axis=1))
            )
        return (self.ax**2 + self.ay**2 + self.az**2) ** 0.5

    @funcify_method("Time (s)", "Axial Acceleration (m/s²)", "spline", "zero")
    def axial_acceleration(self):
        """Axial acceleration magnitude as a Function of time."""
        return (
            self.ax * self.attitude_vector_x
            + self.ay * self.attitude_vector_y
            + self.az * self.attitude_vector_z
        )

    @cached_property
    def max_acceleration_power_on_time(self):
        """Time at which the rocket reaches its maximum acceleration during
        motor burn."""
        burn_out_time_index = find_closest(
            self.acceleration.source[:, 0], self.rocket.motor.burn_out_time
        )
        if burn_out_time_index == 0:
            return 0  # the burn out time is before the first time step

        max_acceleration_time_index = np.argmax(
            self.acceleration[:burn_out_time_index, 1]
        )
        return self.acceleration[max_acceleration_time_index, 0]

    @cached_property
    def max_acceleration_power_on(self):
        """Maximum acceleration reached by the rocket during motor burn."""
        return self.acceleration.get_value_opt(self.max_acceleration_power_on_time)

    @cached_property
    def max_acceleration_power_off_time(self):
        """Time at which the rocket reaches its maximum acceleration after
        motor burn."""
        burn_out_time_index = find_closest(
            self.acceleration.source[:, 0], self.rocket.motor.burn_out_time
        )
        max_acceleration_time_index = np.argmax(
            self.acceleration[burn_out_time_index:, 1]
        )
        return self.acceleration[burn_out_time_index + max_acceleration_time_index, 0]

    @cached_property
    def max_acceleration_power_off(self):
        """Maximum acceleration reached by the rocket after motor burn."""
        return self.acceleration.get_value_opt(self.max_acceleration_power_off_time)

    @cached_property
    def max_acceleration_time(self):
        """Time at which the rocket reaches its maximum acceleration."""
        max_acceleration_time_index = np.argmax(self.acceleration[:, 1])
        return self.acceleration[max_acceleration_time_index, 0]

    @cached_property
    def max_acceleration(self):
        """Maximum acceleration reached by the rocket."""
        max_acceleration_time_index = np.argmax(self.acceleration[:, 1])
        return self.acceleration[max_acceleration_time_index, 1]

    @funcify_method("Time (s)", "Horizontal Speed (m/s)")
    def horizontal_speed(self):
        """Rocket horizontal speed as a Function of time."""
        if hasattr(self, "_launch_site_fixed"):
            velocity = self.velocity_local()
            return np.column_stack((self.time, np.linalg.norm(velocity[:, :2], axis=1)))
        return (self.vx**2 + self.vy**2) ** 0.5

    # Path Angle
    @funcify_method("Time (s)", "Path Angle (°)", "spline", "constant")
    def path_angle(self):
        """Rocket path angle as a Function of time."""
        if hasattr(self, "_launch_site_fixed"):
            velocity = self.velocity_local()
            path_angle = np.degrees(
                np.arctan2(
                    velocity[:, 2],
                    np.linalg.norm(velocity[:, :2], axis=1),
                )
            )
            return np.column_stack((self.time, path_angle))
        path_angle = (180 / np.pi) * np.arctan2(
            self.vz[:, 1], self.horizontal_speed[:, 1]
        )
        return np.column_stack([self.time, path_angle])

    # Attitude Angle
    @funcify_method("Time (s)", "Attitude Vector X Component")
    def attitude_vector_x(self):
        """Rocket attitude vector X component as a Function of time.
        Same as row 1, column 3 of the rotation matrix that defines
        the conversion from the body frame to the inertial frame
        at each time step."""
        return 2 * (self.e1 * self.e3 + self.e0 * self.e2)  # a13

    @funcify_method("Time (s)", "Attitude Vector Y Component")
    def attitude_vector_y(self):
        """Rocket attitude vector Y component as a Function of time.
        Same as row 2, column 3 of the rotation matrix that defines
        the conversion from the body frame to the inertial frame
        at each time step."""
        return 2 * (self.e2 * self.e3 - self.e0 * self.e1)  # a23

    @funcify_method("Time (s)", "Attitude Vector Z Component")
    def attitude_vector_z(self):
        """Rocket attitude vector Z component as a Function of time.
        Same as row 3, column 3 of the rotation matrix that defines
        the conversion from the body frame to the inertial frame
        at each time step."""
        return 1 - 2 * (self.e1**2 + self.e2**2)  # a33

    @funcify_method("Time (s)", "Attitude Angle (°)")
    def attitude_angle(self):
        """Rocket attitude angle as a Function of time."""
        horizontal_attitude_proj = (
            self.attitude_vector_x**2 + self.attitude_vector_y**2
        ) ** 0.5
        attitude_angle = (180 / np.pi) * np.arctan2(
            self.attitude_vector_z[:, 1], horizontal_attitude_proj[:, 1]
        )
        return np.column_stack([self.time, attitude_angle])

    # Lateral Attitude Angle
    @funcify_method("Time (s)", "Lateral Attitude Angle (°)")
    def lateral_attitude_angle(self):
        """Rocket lateral attitude angle as a Function of time."""
        lateral_vector_angle = (np.pi / 180) * (self.heading - 90)
        lateral_vector_x = np.sin(lateral_vector_angle)
        lateral_vector_y = np.cos(lateral_vector_angle)
        attitude_lateral_proj = (
            lateral_vector_x * self.attitude_vector_x[:, 1]
            + lateral_vector_y * self.attitude_vector_y[:, 1]
        )
        attitude_lateral_proj_x = attitude_lateral_proj * lateral_vector_x
        attitude_lateral_proj_y = attitude_lateral_proj * lateral_vector_y
        attitude_lateral_plane_proj_x = (
            self.attitude_vector_x[:, 1] - attitude_lateral_proj_x
        )
        attitude_lateral_plane_proj_y = (
            self.attitude_vector_y[:, 1] - attitude_lateral_proj_y
        )
        attitude_lateral_plane_proj_z = self.attitude_vector_z[:, 1]
        attitude_lateral_plane_proj = (
            attitude_lateral_plane_proj_x**2
            + attitude_lateral_plane_proj_y**2
            + attitude_lateral_plane_proj_z**2
        ) ** 0.5
        lateral_attitude_angle = (180 / np.pi) * np.arctan2(
            attitude_lateral_proj, attitude_lateral_plane_proj
        )
        lateral_attitude_angle = np.column_stack([self.time, lateral_attitude_angle])
        return lateral_attitude_angle

    # Euler Angles
    @funcify_method("Time (s)", "Precession Angle - ψ (°)", "spline", "constant")
    def psi(self):
        """Precession angle as a Function of time."""
        psi = quaternions_to_precession(
            self.e0.y_array, self.e1.y_array, self.e2.y_array, self.e3.y_array
        )
        return np.column_stack([self.time, psi])

    @funcify_method("Time (s)", "Spin Angle - φ (°)", "spline", "constant")
    def phi(self):
        """Spin angle as a Function of time."""
        phi = quaternions_to_spin(
            self.e0.y_array, self.e1.y_array, self.e2.y_array, self.e3.y_array
        )
        return np.column_stack([self.time, phi])

    @funcify_method("Time (s)", "Nutation Angle - θ (°)", "spline", "constant")
    def theta(self):
        """Nutation angle as a Function of time."""
        theta = quaternions_to_nutation(self.e1.y_array, self.e2.y_array)
        return np.column_stack([self.time, theta])

    # Fluid Mechanics variables
    # Freestream Velocity
    @funcify_method("Time (s)", "Freestream Velocity X (m/s)", "spline", "constant")
    def stream_velocity_x(self):
        """Freestream velocity X component as a Function of time."""
        return np.column_stack((self.time, self.wind_velocity_x[:, 1] - self.vx[:, 1]))

    @funcify_method("Time (s)", "Freestream Velocity Y (m/s)", "spline", "constant")
    def stream_velocity_y(self):
        """Freestream velocity Y component as a Function of time."""
        return np.column_stack((self.time, self.wind_velocity_y[:, 1] - self.vy[:, 1]))

    @funcify_method("Time (s)", "Freestream Velocity Z (m/s)", "spline", "constant")
    def stream_velocity_z(self):
        """Freestream velocity Z component as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            atmospheric_z = np.array(
                [sample[1][2] for sample in self._orbital_atmosphere_samples]
            )
            return np.column_stack((self.time, atmospheric_z - self.vz[:, 1]))
        return np.column_stack((self.time, -self.vz[:, 1]))

    @funcify_method("Time (s)", "Freestream Speed (m/s)", "spline", "constant")
    def free_stream_speed(self):
        """Freestream speed as a Function of time."""
        free_stream_speed = (
            self.stream_velocity_x**2
            + self.stream_velocity_y**2
            + self.stream_velocity_z**2
        ) ** 0.5
        return free_stream_speed.get_source()

    # Apogee Freestream speed
    @cached_property
    def apogee_freestream_speed(self):
        """Freestream speed at apogee in m/s."""
        return self.free_stream_speed.get_value_opt(self.apogee_time)

    # Mach Number
    @funcify_method("Time (s)", "Mach Number", "linear", "zero")
    def mach_number(self):
        """Mach number as a Function of time.

        Vacuum-like atmosphere layers report an infinite speed of sound. They
        therefore produce Mach zero instead of being passed through a spline,
        which cannot represent infinite samples.
        """
        speed = self.free_stream_speed[:, 1]
        sound = self.speed_of_sound[:, 1]
        mach = np.divide(
            speed,
            sound,
            out=np.zeros_like(speed),
            where=np.isfinite(sound) & (sound > np.finfo(float).eps),
        )
        return np.column_stack((self.time, mach))

    @cached_property
    def max_mach_number_time(self):
        """Time of maximum Mach number."""
        max_mach_number_time_index = np.argmax(self.mach_number[:, 1])
        return self.mach_number[max_mach_number_time_index, 0]

    @cached_property
    def max_mach_number(self):
        """Maximum Mach number."""
        return self.mach_number.get_value_opt(self.max_mach_number_time)

    # Stability Margin
    @cached_property
    def max_stability_margin_time(self):
        """Time of maximum stability margin."""
        max_stability_margin_time_index = np.argmax(self.stability_margin[:, 1])
        return self.stability_margin[max_stability_margin_time_index, 0]

    @cached_property
    def max_stability_margin(self):
        """Maximum stability margin."""
        return self.stability_margin.get_value_opt(self.max_stability_margin_time)

    @cached_property
    def min_stability_margin_time(self):
        """Time of minimum stability margin."""
        min_stability_margin_time_index = np.argmin(self.stability_margin[:, 1])
        return self.stability_margin[min_stability_margin_time_index, 0]

    @cached_property
    def min_stability_margin(self):
        """Minimum stability margin."""
        return self.stability_margin.get_value_opt(self.min_stability_margin_time)

    @property
    def initial_stability_margin(self):
        """Stability margin at time 0.

        Returns
        -------
        float
        """
        return self.stability_margin.get_value_opt(self.time[0])

    @property
    def out_of_rail_stability_margin(self):
        """Stability margin at the time the rocket leaves the rail.

        Returns
        -------
        float
        """
        return self.stability_margin.get_value_opt(self.out_of_rail_time)

    # Reynolds Number
    @funcify_method("Time (s)", "Reynolds Number", "linear", "zero")
    def reynolds_number(self):
        """Reynolds number, defined as zero where viscosity is unavailable."""
        viscosity = self.dynamic_viscosity[:, 1]
        numerator = (
            self.density[:, 1] * self.free_stream_speed[:, 1] * (2 * self.rocket.radius)
        )
        reynolds = np.divide(
            numerator,
            viscosity,
            out=np.zeros_like(numerator),
            where=np.isfinite(viscosity) & (viscosity > np.finfo(float).eps),
        )
        return np.column_stack((self.time, reynolds))

    @cached_property
    def max_reynolds_number_time(self):
        """Time of maximum Reynolds number."""
        max_reynolds_number_time_index = np.argmax(self.reynolds_number[:, 1])
        return self.reynolds_number[max_reynolds_number_time_index, 0]

    @cached_property
    def max_reynolds_number(self):
        """Maximum Reynolds number."""
        return self.reynolds_number.get_value_opt(self.max_reynolds_number_time)

    # Dynamic Pressure
    @funcify_method("Time (s)", "Dynamic Pressure (Pa)", "spline", "zero")
    def dynamic_pressure(self):
        """Dynamic pressure as a Function of time."""
        return 0.5 * self.density * self.free_stream_speed**2

    @cached_property
    def max_dynamic_pressure_time(self):
        """Time of maximum dynamic pressure."""
        max_dynamic_pressure_time_index = np.argmax(self.dynamic_pressure[:, 1])
        return self.dynamic_pressure[max_dynamic_pressure_time_index, 0]

    @cached_property
    def max_dynamic_pressure(self):
        """Maximum dynamic pressure."""
        return self.dynamic_pressure.get_value_opt(self.max_dynamic_pressure_time)

    # Total Pressure
    @funcify_method("Time (s)", "Total Pressure (Pa)", "linear", "zero")
    def total_pressure(self):
        """Total pressure as a Function of time."""
        values = self.pressure[:, 1] * (1 + 0.2 * self.mach_number[:, 1] ** 2) ** 3.5
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.column_stack((self.time, values))

    @cached_property
    def max_total_pressure_time(self):
        """Time of maximum total pressure."""
        max_total_pressure_time_index = np.argmax(self.total_pressure[:, 1])
        return self.total_pressure[max_total_pressure_time_index, 0]

    @cached_property
    def max_total_pressure(self):
        """Maximum total pressure."""
        return self.total_pressure.get_value_opt(self.max_total_pressure_time)

    # Dynamics functions and variables

    #  Aerodynamic Normal and Axial Forces (body frame)
    @funcify_method("Time (s)", "Aerodynamic Normal Force (N)", "spline", "zero")
    def aerodynamic_normal_force(self):
        """Aerodynamic normal force in the body frame as a Function of time.

        Equal to the resultant of the transverse forces: sqrt(R1² + R2²).
        """
        return (self.R1**2 + self.R2**2) ** 0.5

    @funcify_method("Time (s)", "Aerodynamic Axial Force (N)", "spline", "zero")
    def aerodynamic_axial_force(self):
        """Aerodynamic axial force in the body frame as a Function of time.

        Equal to -R3, the drag component along the rocket's longitudinal axis.
        """
        return -1 * self.R3

    #  Aerodynamic Lift and Drag (aerodynamic frame)
    @funcify_method("Time (s)", "Aerodynamic Lift Force (N)", "spline", "zero")
    def aerodynamic_lift(self):
        """Aerodynamic lift force in the aerodynamic frame as a Function of time.

        Lift is the aerodynamic force perpendicular to the freestream velocity.
        Computed from the body-frame normal (N) and axial (A) forces and the
        angle of attack (α): L = N·cos(α) − A·sin(α).
        """
        alpha_rad = np.deg2rad(self.angle_of_attack.y_array)
        N = self.aerodynamic_normal_force.y_array
        A = self.aerodynamic_axial_force.y_array
        return np.column_stack(
            (self.time, N * np.cos(alpha_rad) - A * np.sin(alpha_rad))
        )

    @funcify_method("Time (s)", "Aerodynamic Drag Force (N)", "spline", "zero")
    def aerodynamic_drag(self):
        """Aerodynamic drag force in the aerodynamic frame as a Function of time.

        Drag is the aerodynamic force opposing the freestream velocity.
        Computed from the body-frame normal (N) and axial (A) forces and the
        angle of attack (α): D = N·sin(α) + A·cos(α).
        """
        alpha_rad = np.deg2rad(self.angle_of_attack.y_array)
        N = self.aerodynamic_normal_force.y_array
        A = self.aerodynamic_axial_force.y_array
        return np.column_stack(
            (self.time, N * np.sin(alpha_rad) + A * np.cos(alpha_rad))
        )

    @funcify_method("Time (s)", "Aerodynamic Bending Moment (Nm)", "spline", "zero")
    def aerodynamic_bending_moment(self):
        """Aerodynamic bending moment as a Function of time."""
        return (self.M1**2 + self.M2**2) ** 0.5

    @funcify_method("Time (s)", "Aerodynamic Spin Moment (Nm)", "spline", "zero")
    def aerodynamic_spin_moment(self):
        """Aerodynamic spin moment as a Function of time."""
        return self.M3

    # Energy
    # Kinetic Energy
    @funcify_method("Time (s)", "Rotational Kinetic Energy (J)")
    def rotational_energy(self):
        """Rotational kinetic energy as a Function of time."""
        rotational_energy = 0.5 * (
            self.rocket.I_11 * self.w1**2
            + self.rocket.I_22 * self.w2**2
            + self.rocket.I_33 * self.w3**2
        )
        rotational_energy.set_discrete_based_on_model(self.w1)
        return rotational_energy

    @funcify_method("Time (s)", "Translational Kinetic Energy (J)", "spline", "zero")
    def translational_energy(self):
        """Translational kinetic energy as a Function of time."""
        # Redefine total_mass time grid to allow for efficient Function algebra
        total_mass = deepcopy(self.rocket.total_mass)
        total_mass.set_discrete_based_on_model(self.vz)
        translational_energy = 0.5 * total_mass * (self.speed**2)
        return translational_energy

    @funcify_method("Time (s)", "Kinetic Energy (J)", "spline", "zero")
    def kinetic_energy(self):
        """Total kinetic energy as a Function of time."""
        return self.rotational_energy + self.translational_energy

    # Potential Energy
    @funcify_method("Time (s)", "Potential Energy (J)", "spline", "constant")
    def potential_energy(self):
        """Potential energy as a Function of time in relation to sea
        level."""
        standard_gravitational_parameter = self.datum.gravitational_parameter
        reference_radius = self.datum.semi_major_axis
        height = (
            self.altitude if self.reference_frame == ReferenceFrame.GCRF else self.z
        )
        # Redefine total_mass time grid to allow for efficient Function algebra
        total_mass = deepcopy(self.rocket.total_mass)
        total_mass.set_discrete_based_on_model(height)
        return (
            standard_gravitational_parameter
            * total_mass
            * (1 / (height + reference_radius) - 1 / reference_radius)
        )

    # Total Mechanical Energy
    @funcify_method("Time (s)", "Mechanical Energy (J)", "spline", "constant")
    def total_energy(self):
        """Total mechanical energy as a Function of time."""
        return self.kinetic_energy + self.potential_energy

    # thrust Power
    @funcify_method("Time (s)", "Thrust Power (W)", "spline", "zero")
    def thrust_power(self):
        """Thrust power as a Function of time."""
        return self.net_thrust * self.speed

    # Drag Power
    @funcify_method("Time (s)", "Drag Power (W)", "spline", "zero")
    def drag_power(self):
        """Drag power as a Function of time."""
        drag_power = self.R3 * self.speed
        drag_power.set_outputs("Drag Power (W)")
        return drag_power

    # Angle of Attack
    @cached_property
    def direction_cosine_matrixes(self):
        """Direction cosine matrix representing the attitude of the body frame,
        relative to the inertial frame, at each time step."""
        # Stack the y_arrays from e0, e1, e2, and e3 along a new axis
        stacked_arrays = np.stack(
            [self.e0.y_array, self.e1.y_array, self.e2.y_array, self.e3.y_array],
            axis=-1,
        )

        # Apply the transformation to the stacked arrays along the last axis
        Kt = np.array([Matrix.transformation(row).transpose for row in stacked_arrays])

        return Kt

    @cached_property
    def stream_velocity_body_frame(self):
        """Stream velocity array at the center of dry mass in the body frame at
        each time step."""
        Kt = self.direction_cosine_matrixes
        stream_velocity = np.array(
            [
                self.stream_velocity_x.y_array,
                self.stream_velocity_y.y_array,
                self.stream_velocity_z.y_array,
            ]
        ).transpose()
        stream_velocity_body = np.squeeze(
            np.matmul(Kt, stream_velocity[:, :, np.newaxis])
        )
        return stream_velocity_body

    @funcify_method("Time (s)", "Angle of Attack (°)", "spline", "constant")
    def angle_of_attack(self):
        """Angle of attack of the rocket with respect to the freestream
        velocity vector. Sometimes called total angle of attack. Defined as the
        angle between the freestream velocity vector and the rocket's z-axis.
        All in the Body Axes Coordinate System."""
        # Define stream velocity z component in body frame
        stream_vz_body = (
            -self.attitude_vector_x.y_array * self.stream_velocity_x.y_array
            - self.attitude_vector_y.y_array * self.stream_velocity_y.y_array
            - self.attitude_vector_z.y_array * self.stream_velocity_z.y_array
        )
        # Define freestream speed list
        free_stream_speed = self.free_stream_speed.y_array

        stream_vz_body_normalized = np.divide(
            stream_vz_body,
            free_stream_speed,
            out=np.zeros_like(stream_vz_body),
            where=free_stream_speed > 1e-6,
        )
        stream_vz_body_normalized = np.clip(stream_vz_body_normalized, -1, 1)

        # Calculate angle of attack and convert to degrees
        angle_of_attack = np.rad2deg(np.arccos(stream_vz_body_normalized))
        angle_of_attack = np.nan_to_num(angle_of_attack)

        return np.column_stack([self.time, angle_of_attack])

    @funcify_method("Time (s)", "Partial Angle of Attack (°)", "spline", "constant")
    def partial_angle_of_attack(self):
        """Partial angle of attack of the rocket with respect to the stream
        velocity vector. By partial angle of attack, it is meant the angle
        between the stream velocity vector in the y-z plane and the rocket's
        z-axis. All in the Body Axes Coordinate System."""
        # Stream velocity in standard aerodynamic frame
        stream_velocity = -self.stream_velocity_body_frame
        alpha = np.arctan2(
            stream_velocity[:, 1],
            stream_velocity[:, 2],
        )  # y-z plane
        return np.column_stack([self.time, np.rad2deg(alpha)])

    @funcify_method("Time (s)", "Beta (°)", "spline", "constant")
    def angle_of_sideslip(self):
        """Angle of sideslip of the rocket with respect to the stream
        velocity vector. Defined as the angle between the stream velocity
        vector in the x-z plane and the rocket's z-axis. All in the Body
        Axes Coordinate System."""
        # Stream velocity in standard aerodynamic frame
        stream_velocity = -self.stream_velocity_body_frame
        beta = np.arctan2(
            stream_velocity[:, 0],
            stream_velocity[:, 2],
        )  # x-z plane
        return np.column_stack([self.time, np.rad2deg(beta)])

    # Frequency response and stability variables
    @funcify_method("Frequency (Hz)", "ω1 Fourier Amplitude", "spline", "zero")
    def omega1_frequency_response(self):
        """Angular velocity 1 frequency response as a Function of
        frequency, as the rocket leaves the launch rail for 5 seconds of flight.
        """
        return self.w1.to_frequency_domain(
            self.out_of_rail_time, self.out_of_rail_time + 5, 100
        )

    @funcify_method("Frequency (Hz)", "ω2 Fourier Amplitude", "spline", "zero")
    def omega2_frequency_response(self):
        """Angular velocity 2 frequency response as a Function of
        frequency, as the rocket leaves the launch rail for 5 seconds of flight.
        """
        return self.w2.to_frequency_domain(
            self.out_of_rail_time, self.out_of_rail_time + 5, 100
        )

    @funcify_method("Frequency (Hz)", "ω3 Fourier Amplitude", "spline", "zero")
    def omega3_frequency_response(self):
        """Angular velocity 3 frequency response as a Function of
        frequency, as the rocket leaves the launch rail for 5 seconds of flight.
        """
        return self.w3.to_frequency_domain(
            self.out_of_rail_time, self.out_of_rail_time + 5, 100
        )

    @funcify_method(
        "Frequency (Hz)", "Attitude Angle Fourier Amplitude", "spline", "zero"
    )
    def attitude_frequency_response(self):
        """Attitude frequency response as a Function of frequency, as
        the rocket leaves the launch rail for 5 seconds of flight."""
        return self.attitude_angle.to_frequency_domain(
            lower=self.out_of_rail_time,
            upper=self.out_of_rail_time + 5,
            sampling_frequency=100,
        )

    @cached_property
    def static_margin(self):
        """Static margin of the rocket."""
        return self.rocket.static_margin

    @funcify_method("Time (s)", "Stability Margin (c)", "linear", "zero")
    def stability_margin(self):
        """Linear stability margin along the flight, in calibers.

        This is the classical margin: it evaluates the rocket's linearized
        stability margin (:meth:`Rocket.stability_margin`) at the realized
        flight Mach and time at each instant, capturing the Mach variation of
        the aerodynamic center together with the center-of-mass shift as
        propellant burns.

        Returns
        -------
        stability : rocketpy.Function
            Stability margin in calibers as a function of time.
        """
        return [(t, self.rocket.stability_margin(m, t)) for t, m in self.mach_number]

    @funcify_method("Time (s)", "Stability Margin - Yaw (c)", "linear", "zero")
    def stability_margin_yaw(self):
        """Linear yaw-plane stability margin along the flight, in calibers.

        Yaw-plane counterpart of :meth:`stability_margin`, using the rocket's
        yaw-plane aerodynamic center (:meth:`Rocket.stability_margin_yaw`).
        Equals :meth:`stability_margin` for an axisymmetric rocket; for a
        non-axisymmetric rocket (e.g. single-plane canards) it differs, since
        the pitch and yaw aerodynamic centers no longer coincide.

        Returns
        -------
        stability : rocketpy.Function
            Yaw-plane stability margin in calibers as a function of time.
        """
        return [
            (t, self.rocket.stability_margin_yaw(m, t)) for t, m in self.mach_number
        ]

    # Dynamic stability
    # TODO: review note: the two methods below this comment needs to have its
    # equations documented in a .rst file
    def _lateral_inertia(self, dry_lateral_inertia, motor_lateral_inertia):
        """Lateral moment of inertia about the instantaneous center of mass, as
        an array over ``self.time``. Uses the reduced-mass formulation of the
        equations of motion: ``I_L = I_dry + I_motor(t) + mu(t) b^2`` with
        ``mu`` the dry/propellant reduced mass and ``b`` the (initial)
        dry-mass-to-propellant distance."""
        dry_mass = self.rocket.dry_mass
        b = (
            -(
                self.rocket.center_of_propellant_position.get_value_opt(0)
                - self.rocket.center_of_dry_mass_position
            )
            * self.rocket._csys
        )
        inertia = np.empty(len(self.time))
        for i, t in enumerate(self.time):
            propellant_mass = self.rocket.motor.propellant_mass.get_value_opt(t)
            total = propellant_mass + dry_mass
            mu = (propellant_mass * dry_mass / total) if total > 0 else 0.0
            inertia[i] = (
                dry_lateral_inertia + motor_lateral_inertia.get_value_opt(t) + mu * b**2
            )
        return inertia

    def _dynamic_stability(self, lift_slope, stability_margin, lateral_inertia):
        """Linearized oscillator coefficients for one plane, as arrays over
        ``self.time``: corrective moment coefficient ``C1`` (restoring moment
        per radian), damping moment coefficient ``C2`` (aerodynamic + jet),
        undamped natural frequency ``omega_n`` and damping ratio ``zeta``.

        ``lift_slope`` is the rocket's total normal-force-curve slope for the
        plane (``total_lift_coeff_der`` for pitch, ``total_side_coeff_der`` for
        yaw); ``stability_margin`` is the matching linear margin
        ``Function(mach, time)``; ``lateral_inertia`` is the array from
        :meth:`_lateral_inertia`.
        """
        area = self.rocket.area
        diameter = 2 * self.rocket.radius
        csys = self.rocket._csys
        nozzle_position = self.rocket.nozzle_position
        mass_flow_rate = self.rocket.motor.total_mass_flow_rate

        corrective = np.empty(len(self.time))
        damping = np.empty(len(self.time))
        for i, t in enumerate(self.time):
            mach = self.mach_number.get_value_opt(t)
            dynamic_pressure = self.dynamic_pressure.get_value_opt(t)
            speed = self.speed.get_value_opt(t)
            density = self.density.get_value_opt(t)
            center_of_mass = self.rocket.center_of_mass.get_value_opt(t)

            # Corrective moment per radian: q A C_Nalpha (x_cm - x_ac).
            margin = stability_margin.get_value_opt(mach, t)  # calibers
            corrective[i] = (
                dynamic_pressure
                * area
                * lift_slope.get_value_opt(mach)
                * margin
                * diameter
            )

            # Aerodynamic damping: 0.5 rho V A sum_i (A_i/A) C_Nalpha_i arm_i^2.
            damping_aero = 0.0
            for surface, position in self.rocket.aerodynamic_surfaces:
                slope = surface.cN_alpha.get_value_opt(
                    0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0
                )
                cp_position = (
                    position.z - csys * surface.center_of_pressure_z.get_value_opt(mach)
                )
                arm = cp_position - center_of_mass
                ref_factor = surface.reference_area / area
                damping_aero += ref_factor * slope * arm**2
            damping_aero *= 0.5 * density * speed * area

            # Jet (propulsive) damping: mdot (x_nozzle - x_cm)^2.
            damping_jet = (
                abs(mass_flow_rate.get_value_opt(t))
                * (nozzle_position - center_of_mass) ** 2
            )
            damping[i] = damping_aero + damping_jet

        positive_corrective = np.clip(corrective, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            natural_frequency = np.sqrt(positive_corrective / lateral_inertia)
            denominator = 2.0 * np.sqrt(positive_corrective * lateral_inertia)
            damping_ratio = np.divide(
                damping,
                denominator,
                out=np.zeros_like(damping),
                where=denominator > 0,
            )
        return corrective, damping, natural_frequency, damping_ratio

    @funcify_method("Time (s)", "Corrective Moment Coefficient (N m/rad)", "linear")
    def corrective_moment_coefficient(self):
        """Pitch-plane corrective (restoring) moment coefficient ``C1`` as a
        function of time -- the aerodynamic restoring moment per radian of angle
        of attack. Positive for a statically stable rocket."""
        inertia = self._lateral_inertia(self.rocket.dry_I_11, self.rocket.motor.I_11)
        corrective, _, _, _ = self._dynamic_stability(
            self.rocket.total_lift_coeff_der, self.rocket.stability_margin, inertia
        )
        return np.column_stack((self.time, corrective))

    @funcify_method("Time (s)", "Damping Moment Coefficient (N m s/rad)", "linear")
    def damping_moment_coefficient(self):
        """Pitch-plane damping moment coefficient ``C2`` as a function of time --
        the moment opposing the pitch rate, summing aerodynamic damping (from
        every surface) and propulsive (jet) damping."""
        inertia = self._lateral_inertia(self.rocket.dry_I_11, self.rocket.motor.I_11)
        _, damping, _, _ = self._dynamic_stability(
            self.rocket.total_lift_coeff_der, self.rocket.stability_margin, inertia
        )
        return np.column_stack((self.time, damping))

    @funcify_method("Time (s)", "Pitch Natural Frequency (rad/s)", "linear")
    def pitch_natural_frequency(self):
        """Undamped natural frequency of the pitch oscillation,
        ``omega_n = sqrt(C1 / I_L)``, as a function of time (rad/s)."""
        inertia = self._lateral_inertia(self.rocket.dry_I_11, self.rocket.motor.I_11)
        _, _, natural_frequency, _ = self._dynamic_stability(
            self.rocket.total_lift_coeff_der, self.rocket.stability_margin, inertia
        )
        return np.column_stack((self.time, natural_frequency))

    @funcify_method("Time (s)", "Pitch Damping Ratio", "linear")
    def pitch_damping_ratio(self):
        """Damping ratio of the pitch oscillation,
        ``zeta = C2 / (2 sqrt(C1 I_L))``, as a function of time. ``zeta < 1`` is
        underdamped (oscillatory), ``zeta > 1`` overdamped."""
        inertia = self._lateral_inertia(self.rocket.dry_I_11, self.rocket.motor.I_11)
        _, _, _, damping_ratio = self._dynamic_stability(
            self.rocket.total_lift_coeff_der, self.rocket.stability_margin, inertia
        )
        return np.column_stack((self.time, damping_ratio))

    @funcify_method("Time (s)", "Yaw Natural Frequency (rad/s)", "linear")
    def yaw_natural_frequency(self):
        """Undamped natural frequency of the yaw oscillation as a function of
        time (rad/s). Equals :meth:`pitch_natural_frequency` for an axisymmetric
        rocket."""
        inertia = self._lateral_inertia(self.rocket.dry_I_22, self.rocket.motor.I_22)
        _, _, natural_frequency, _ = self._dynamic_stability(
            self.rocket.total_side_coeff_der,
            self.rocket.stability_margin_yaw,
            inertia,
        )
        return np.column_stack((self.time, natural_frequency))

    @funcify_method("Time (s)", "Yaw Damping Ratio", "linear")
    def yaw_damping_ratio(self):
        """Damping ratio of the yaw oscillation as a function of time. Equals
        :meth:`pitch_damping_ratio` for an axisymmetric rocket."""
        inertia = self._lateral_inertia(self.rocket.dry_I_22, self.rocket.motor.I_22)
        _, _, _, damping_ratio = self._dynamic_stability(
            self.rocket.total_side_coeff_der,
            self.rocket.stability_margin_yaw,
            inertia,
        )
        return np.column_stack((self.time, damping_ratio))

    # Rail Button Forces

    @cached_property
    def __calculate_rail_button_forces(self):
        """Calculate the forces applied to the rail buttons while rocket is
        still on the launch rail. It will return 0 if no rail buttons are
        defined.

        Returns
        -------
        F11: Function
            Rail Button 1 force in the 1 direction
        F12: Function
            Rail Button 1 force in the 2 direction
        F21: Function
            Rail Button 2 force in the 1 direction
        F22: Function
            Rail Button 2 force in the 2 direction
        """
        # First check for no rail phase or rail buttons
        null_force = Function(0)
        if self.out_of_rail_time_index == 0:  # No rail phase, no rail button forces
            warnings.warn(
                "Trying to calculate rail button forces without a rail phase defined. "
                + "The rail button forces will be set to zero.",
                UserWarning,
            )
            return null_force, null_force, null_force, null_force
        if len(self.rocket.rail_buttons) == 0:
            warnings.warn(
                "Trying to calculate rail button forces without rail buttons defined. "
                + "The rail button forces will be set to zero.",
                UserWarning,
            )
            return null_force, null_force, null_force, null_force

        # Distance from Rail Button 1 (upper) to CM
        rail_buttons_tuple = self.rocket.rail_buttons[0]
        upper_button_position = (
            rail_buttons_tuple.component.buttons_distance
            + rail_buttons_tuple.position.z
        )
        lower_button_position = rail_buttons_tuple.position.z
        angular_position_rad = rail_buttons_tuple.component.angular_position_rad
        D1 = (
            upper_button_position - self.rocket.center_of_dry_mass_position
        ) * self.rocket._csys
        # Distance from Rail Button 2 (lower) to CM
        D2 = (
            lower_button_position - self.rocket.center_of_dry_mass_position
        ) * self.rocket._csys
        F11 = (self.R1 * D2 - self.M2) / (D1 + D2)
        F11.set_outputs("Upper button force direction 1 (m)")
        F12 = (self.R2 * D2 + self.M1) / (D1 + D2)
        F12.set_outputs("Upper button force direction 2 (m)")
        F21 = (self.R1 * D1 + self.M2) / (D1 + D2)
        F21.set_outputs("Lower button force direction 1 (m)")
        F22 = (self.R2 * D1 - self.M1) / (D1 + D2)
        F22.set_outputs("Lower button force direction 2 (m)")

        model = Function(
            F11.get_source()[: self.out_of_rail_time_index + 1, :],
            interpolation=F11.__interpolation__,
        )

        # Limit force calculation to when rocket is in rail
        F11.set_discrete_based_on_model(model)
        F12.set_discrete_based_on_model(model)
        F21.set_discrete_based_on_model(model)
        F22.set_discrete_based_on_model(model)

        rail_button1_normal_force = F11 * np.cos(angular_position_rad) + F12 * np.sin(
            angular_position_rad
        )
        rail_button1_shear_force = F11 * -np.sin(angular_position_rad) + F12 * np.cos(
            angular_position_rad
        )
        rail_button2_normal_force = F21 * np.cos(angular_position_rad) + F22 * np.sin(
            angular_position_rad
        )
        rail_button2_shear_force = F21 * -np.sin(angular_position_rad) + F22 * np.cos(
            angular_position_rad
        )

        return (
            rail_button1_normal_force,
            rail_button1_shear_force,
            rail_button2_normal_force,
            rail_button2_shear_force,
        )

    @funcify_method("Time (s)", "Upper Rail Button Normal Force (N)", "spline", "zero")
    def rail_button1_normal_force(self):
        """Upper rail button normal force as a Function of time. If
        there's no rail button defined, the function returns a null Function."""
        return self.__calculate_rail_button_forces[0]

    @funcify_method("Time (s)", "Upper Rail Button Shear Force (N)", "spline", "zero")
    def rail_button1_shear_force(self):
        """Upper rail button shear force as a Function of time. If
        there's no rail button defined, the function returns a null Function."""
        return self.__calculate_rail_button_forces[1]

    @funcify_method("Time (s)", "Lower Rail Button Normal Force (N)", "spline", "zero")
    def rail_button2_normal_force(self):
        """Lower rail button normal force as a Function of time. If
        there's no rail button defined, the function returns a null Function."""
        return self.__calculate_rail_button_forces[2]

    @funcify_method("Time (s)", "Lower Rail Button Shear Force (N)", "spline", "zero")
    def rail_button2_shear_force(self):
        """Lower rail button shear force as a Function of time. If
        there's no rail button defined, the function returns a null Function."""
        return self.__calculate_rail_button_forces[3]

    @property
    def max_rail_button1_normal_force(self):
        """Maximum upper rail button normal force, in Newtons."""
        return np.abs(self.rail_button1_normal_force.y_array).max()

    @property
    def max_rail_button1_shear_force(self):
        """Maximum upper rail button shear force, in Newtons."""
        return np.abs(self.rail_button1_shear_force.y_array).max()

    @property
    def max_rail_button2_normal_force(self):
        """Maximum lower rail button normal force, in Newtons."""
        return np.abs(self.rail_button2_normal_force.y_array).max()

    @property
    def max_rail_button2_shear_force(self):
        """Maximum lower rail button shear force, in Newtons."""
        return np.abs(self.rail_button2_shear_force.y_array).max()

    @cached_property
    def calculate_rail_button_bending_moments(self):
        """Calculate internal bending moments at rail button attachment points.

        Uses beam theory to determine the internal structural moments for
        stress analysis of the rail button attachments (fasteners and airframe).

        The bending moment at each button attachment consists of:

        1. Normal force moment: $M = N \\times d$, where $N$ is the normal
           reaction force and $d$ is the distance from button to center of
           dry mass.
        2. Shear force cantilever moment: $M = S \\times h$, where $S$ is the
           shear (tangential) force and $h$ is the button standoff height.

        Returns
        -------
        tuple
            rail_button1_bending_moment : Function
                Bending moment at upper rail button as a function of time (N·m).
            max_rail_button1_bending_moment : float
                Maximum upper rail button bending moment (N·m).
            rail_button2_bending_moment : Function
                Bending moment at lower rail button as a function of time (N·m).
            max_rail_button2_bending_moment : float
                Maximum lower rail button bending moment (N·m).

        Notes
        -----
        - Calculated only during the rail phase of flight
        - Maximum values use absolute values for worst-case stress analysis
        - The bending moments represent internal stresses in the rocket
          airframe at the rail button attachment points

        **Assumptions:**

        - Rail buttons act as simple supports: provide reaction forces (normal
          and shear) but no moment reaction at the rail contact point
        - The rocket acts as a beam supported at two points (rail buttons)
        - Bending moments arise from the lever arm effect of reaction forces
          and the cantilever moment from button standoff height
        """
        # Check if rail buttons exist
        null_moment = Function(0)
        if len(self.rocket.rail_buttons) == 0:
            warnings.warn(
                "Trying to calculate rail button bending moments without "
                "rail buttons defined. Setting moments to zero.",
                UserWarning,
            )
            return (null_moment, 0.0, null_moment, 0.0)

        # Get rail button geometry
        rail_buttons_tuple = self.rocket.rail_buttons[0]
        # Rail button standoff height
        h_button = rail_buttons_tuple.component.button_height
        if h_button is None:
            warnings.warn(
                "Rail button height not defined. Bending moments cannot be "
                "calculated. Setting moments to zero.",
                UserWarning,
            )
            return (null_moment, 0.0, null_moment, 0.0)
        upper_button_position = (
            rail_buttons_tuple.component.buttons_distance
            + rail_buttons_tuple.position.z
        )
        lower_button_position = rail_buttons_tuple.position.z

        # Get center of dry mass (handle both callable and property)
        if callable(self.rocket.center_of_dry_mass_position):
            cdm = self.rocket.center_of_dry_mass_position(self.rocket._csys)
        else:
            cdm = self.rocket.center_of_dry_mass_position

        # Distances from buttons to center of dry mass
        d1 = abs(upper_button_position - cdm)
        d2 = abs(lower_button_position - cdm)

        # forces
        N1 = self.rail_button1_normal_force
        N2 = self.rail_button2_normal_force
        S1 = self.rail_button1_shear_force
        S2 = self.rail_button2_shear_force
        t = N1.source[:, 0]

        # Calculate bending moments at attachment points
        # Primary contribution from shear force acting at button height
        # Secondary contribution from normal force creating moment about attachment
        m1_values = N2.source[:, 1] * d2 + S1.source[:, 1] * h_button
        m2_values = N1.source[:, 1] * d1 + S2.source[:, 1] * h_button

        rail_button1_bending_moment = Function(
            np.column_stack([t, m1_values]),
            inputs="Time (s)",
            outputs="Bending Moment (N·m)",
            interpolation="linear",
        )
        rail_button2_bending_moment = Function(
            np.column_stack([t, m2_values]),
            inputs="Time (s)",
            outputs="Bending Moment (N·m)",
            interpolation="linear",
        )

        # Maximum bending moments (absolute value for stress calculations)
        max_rail_button1_bending_moment = float(np.max(np.abs(m1_values)))
        max_rail_button2_bending_moment = float(np.max(np.abs(m2_values)))

        return (
            rail_button1_bending_moment,
            max_rail_button1_bending_moment,
            rail_button2_bending_moment,
            max_rail_button2_bending_moment,
        )

    @property
    def rail_button1_bending_moment(self):
        """Upper rail button bending moment as a Function of time."""
        return self.calculate_rail_button_bending_moments[0]

    @property
    def max_rail_button1_bending_moment(self):
        """Maximum upper rail button bending moment, in N·m."""
        return self.calculate_rail_button_bending_moments[1]

    @property
    def rail_button2_bending_moment(self):
        """Lower rail button bending moment as a Function of time."""
        return self.calculate_rail_button_bending_moments[2]

    @property
    def max_rail_button2_bending_moment(self):
        """Maximum lower rail button bending moment, in N·m."""
        return self.calculate_rail_button_bending_moments[3]

    @funcify_method(
        "Time (s)", "Horizontal Distance to Launch Point (m)", "spline", "constant"
    )
    def drift(self):
        """Rocket horizontal distance to the launch point, in meters, as a
        Function of time."""
        if hasattr(self, "_launch_site_fixed"):
            local = self.position_local()
            distance = np.hypot(local[:, 0], local[:, 1])
        else:
            distance = np.hypot(self.x[:, 1], self.y[:, 1])
        return np.column_stack((self.time, distance))

    @funcify_method("Time (s)", "Bearing (°)", "spline", "constant")
    def bearing(self):
        """Rocket bearing compass, in degrees, as a Function of time."""
        if hasattr(self, "_launch_site_fixed"):
            local = self.position_local()
            x, y = local[:, 0], local[:, 1]
        else:
            x, y = self.x[:, 1], self.y[:, 1]
        bearing = (2 * np.pi - np.arctan2(-x, y)) * (180 / np.pi)
        return np.column_stack((self.time, bearing))

    @funcify_method("Time (s)", "Latitude (°)", "linear", "constant")
    def latitude(self):
        """Rocket latitude coordinate, in degrees, as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            coordinates = [
                self.datum.itrs_to_geodetic(position)
                for position in self.position(ReferenceFrame.ITRF)
            ]
            return np.column_stack(
                (self.time, np.degrees([coordinate[0] for coordinate in coordinates]))
            )
        lat, _ = inverted_haversine_array(
            self.env.latitude,
            self.env.longitude,
            self.drift[:, 1],
            self.bearing[:, 1],
            earth_radius=self.env.earth_radius,
        )
        return np.column_stack((self.time, lat))

    @funcify_method("Time (s)", "Longitude (°)", "linear", "constant")
    def longitude(self):
        """Rocket longitude coordinate, in degrees, as a Function of time."""
        if self.reference_frame == ReferenceFrame.GCRF:
            coordinates = [
                self.datum.itrs_to_geodetic(position)
                for position in self.position(ReferenceFrame.ITRF)
            ]
            return np.column_stack(
                (self.time, np.degrees([coordinate[1] for coordinate in coordinates]))
            )
        _, lon = inverted_haversine_array(
            self.env.latitude,
            self.env.longitude,
            self.drift[:, 1],
            self.bearing[:, 1],
            earth_radius=self.env.earth_radius,
        )
        return np.column_stack((self.time, lon))

    def info(self):
        """Prints out a summary of the data available about the Flight."""
        self.prints.all()

    def all_info(self):
        """Prints out all data and graphs available about the Flight."""
        self.info()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = {
            "rocket": self.rocket,
            "env": self.env,
            "rail_length": self.rail_length,
            "inclination": self.inclination,
            "heading": self.heading,
            "initial_solution": self.initial_solution,
            "terminate_on_apogee": self.terminate_on_apogee,
            "max_time": self.max_time,
            "max_time_step": self.max_time_step,
            "min_time_step": self.min_time_step,
            "rtol": self.rtol,
            "atol": self.atol,
            "time_overshoot": self.time_overshoot,
            "name": self.name,
            "equations_of_motion": self.equations_of_motion,
            "simulation_mode": self.simulation_mode,
            "ode_solver": self.ode_solver,
            "initial_state": self.initial_state,
            "space": self.space,
            "forces": self.forces,
            # The following outputs are essential to run all_info method
            "solution": self.solution,
            "out_of_rail_time": self.out_of_rail_time,
            "out_of_rail_time_index": self.out_of_rail_time_index,
            "apogee_time": self.apogee_time,
            "apogee": self.apogee,
            "parachute_events": self.parachute_events,
            "impact_state": self.impact_state,
            "impact_velocity": self.impact_velocity,
            "x_impact": self.x_impact,
            "y_impact": self.y_impact,
            "t_final": self.t_final,
            "function_evaluations": self.function_evaluations,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "alpha3": self.alpha3,
            "R1": self.R1,
            "R2": self.R2,
            "R3": self.R3,
            "M1": self.M1,
            "M2": self.M2,
            "M3": self.M3,
            "net_thrust": self.net_thrust,
        }

        if kwargs.get("include_outputs", False):
            data.update(
                {
                    "time": self.time,
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                    "vx": self.vx,
                    "vy": self.vy,
                    "vz": self.vz,
                    "e0": self.e0,
                    "e1": self.e1,
                    "e2": self.e2,
                    "e3": self.e3,
                    "w1": self.w1,
                    "w2": self.w2,
                    "w3": self.w3,
                    "altitude": self.altitude,
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "epochs": self.epochs,
                }
            )
            if self.reference_frame == ReferenceFrame.GCRF:
                data["orbit"] = self.orbit
                data["orbital_accelerations"] = self.orbital_accelerations
                data["orbital_accelerations_rtn"] = self.orbital_accelerations_rtn
                data["specific_orbital_energy"] = self.specific_orbital_energy
                data["specific_angular_momentum"] = self.specific_angular_momentum
            else:
                data.update(
                    {
                        "out_of_rail_velocity": self.out_of_rail_velocity,
                        "out_of_rail_state": self.out_of_rail_state,
                        "apogee_x": self.apogee_x,
                        "apogee_y": self.apogee_y,
                        "apogee_state": self.apogee_state,
                        "mach_number": self.mach_number,
                        "stream_velocity_x": self.stream_velocity_x,
                        "stream_velocity_y": self.stream_velocity_y,
                        "stream_velocity_z": self.stream_velocity_z,
                        "free_stream_speed": self.free_stream_speed,
                        "angle_of_attack": self.angle_of_attack,
                        "static_margin": self.static_margin,
                        "stability_margin": self.stability_margin,
                    }
                )

        return data

    @classmethod
    def from_dict(cls, data):
        common = dict(
            rocket=data["rocket"],
            environment=data["env"],
            inclination=data["inclination"],
            heading=data["heading"],
            terminate_on_apogee=data["terminate_on_apogee"],
            max_time=data["max_time"],
            max_time_step=data["max_time_step"],
            min_time_step=data["min_time_step"],
            rtol=data["rtol"],
            atol=data["atol"],
            time_overshoot=data["time_overshoot"],
            name=data["name"],
            equations_of_motion=data["equations_of_motion"],
            simulation_mode=data.get("simulation_mode", "6 DOF"),
            ode_solver=data.get("ode_solver", "LSODA"),
        )
        if data.get("initial_state") is not None:
            return cls.from_state(
                data["rocket"],
                data["env"],
                data["initial_state"],
                space=data.get("space"),
                forces=data.get("forces"),
                **{
                    key: value
                    for key, value in common.items()
                    if key not in ("rocket", "environment")
                },
            )
        return cls(
            rail_length=data["rail_length"],
            initial_solution=None,
            space=data.get("space"),
            **common,
        )

    # These should be deprecated on v1.13
    @deprecated(
        reason="Controller observed variables are no longer supported.",
        version="v1.13.0",
        alternative="Access the desired variables via controller.log",
    )
    def get_controller_observed_variables(self):
        """Retrieve the observed variables from each controller.

        If there is only one controller, its log is returned directly. If
        there are multiple controllers, a list of logs is returned.

        Returns
        -------
        list
            Controller log(s) containing the return values of each
            controller function call.
        """
        observed_variables = [controller.log for controller in self._controllers]
        return (
            observed_variables[0]
            if len(observed_variables) == 1
            else observed_variables
        )

    @deprecated(
        reason="Moved to FlightDataExporter.export_pressures()",
        version="v1.12.0",
        alternative="rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_pressures",
    )
    def export_pressures(self, file_name, time_step):
        """
        .. deprecated:: 1.11
           Use :class:`rocketpy.simulation.flight_data_exporter.FlightDataExporter`
           and call ``.export_pressures(...)``.
        """
        return self.exports.export_pressures(file_name, time_step)

    @deprecated(
        reason="Moved to FlightDataExporter.export_data()",
        version="v1.12.0",
        alternative="rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_data",
    )
    def export_data(self, file_name, *variables, time_step=None):
        """
        .. deprecated:: 1.11
           Use :class:`rocketpy.simulation.flight_data_exporter.FlightDataExporter`
           and call ``.export_data(...)``.
        """
        return self.exports.export_data(file_name, *variables, time_step=time_step)

    @deprecated(
        reason="Moved to FlightDataExporter.export_sensor_data()",
        version="v1.12.0",
        alternative="rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_sensor_data",
    )
    def export_sensor_data(self, file_name, sensor=None):
        """
        .. deprecated:: 1.11
           Use :class:`rocketpy.simulation.flight_data_exporter.FlightDataExporter`
           and call ``.export_sensor_data(...)``.
        """
        return self.exports.export_sensor_data(file_name, sensor=sensor)

    @deprecated(
        reason="Moved to FlightDataExporter.export_kml()",
        version="v1.12.0",
        alternative="rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_kml",
    )
    def export_kml(
        self,
        file_name="trajectory.kml",
        time_step=None,
        extrude=True,
        color="641400F0",
        altitude_mode="absolute",
    ):
        """
        .. deprecated:: 1.11
           Use :class:`rocketpy.simulation.flight_data_exporter.FlightDataExporter`
           and call ``.export_kml(...)``.
        """
        return self.exports.export_kml(
            file_name=file_name,
            time_step=time_step,
            extrude=extrude,
            color=color,
            altitude_mode=altitude_mode,
        )

    @deprecated(
        reason="Moved to rocketpy.utilities.calculate_stall_wind_velocity",
        version="v1.13.0",
        alternative="rocketpy.utilities.calculate_stall_wind_velocity",
    )
    def calculate_stall_wind_velocity(self, stall_angle):
        """Deprecated. See
        :func:`rocketpy.utilities.calculate_stall_wind_velocity`."""
        # Local import avoids a circular import (utilities imports flight).
        from ..utilities import (  # pylint: disable=import-outside-toplevel
            calculate_stall_wind_velocity,
        )

        return calculate_stall_wind_velocity(self, stall_angle)

    @deprecated(
        reason="Prefer direct pair iteration (for example zip(seq, seq[1:]))",
        version="v1.13.0",
        alternative="Use enumerate(zip(node_list, node_list[1:])) directly",
    )
    def time_iterator(self, node_list):
        """Deprecated helper to iterate over all but the last node."""
        i = 0
        while i < len(node_list) - 1:
            yield i, node_list[i]
            i += 1

    @staticmethod
    def FlightPhases(*_args, **_kwargs):  # pylint: disable=invalid-name
        warnings.warn(
            "FlightPhases is deprecated and will be removed in v1.13. "
            "Use _FlightPhases class directly.",
            DeprecationWarning,
        )

    @staticmethod
    def FlightPhase(*_args, **_kwargs):  # pylint: disable=invalid-name
        warnings.warn(
            "FlightPhase is deprecated and will be removed in v1.13. "
            "Use the _FlightPhase class directly.",
            DeprecationWarning,
        )

    @staticmethod
    def TimeNodes(*_args, **_kwargs):  # pylint: disable=invalid-name
        warnings.warn(
            "TimeNodes is deprecated and will be removed in v1.13. "
            "Use _TimeNodes class directly.",
            DeprecationWarning,
        )

    @staticmethod
    def TimeNode(*_args, **_kwargs):  # pylint: disable=invalid-name
        warnings.warn(
            "TimeNode is deprecated and will be removed in v1.13. "
            "Use the _TimeNode class directly.",
            DeprecationWarning,
        )
