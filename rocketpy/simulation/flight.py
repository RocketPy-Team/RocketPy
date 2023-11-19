import math
import warnings
from copy import deepcopy
from functools import cached_property

import numpy as np
import simplekml
from scipy import integrate

from ..mathutils.function import Function, funcify_method
from ..mathutils.vector_matrix import Matrix, Vector
from ..plots.flight_plots import _FlightPlots
from ..prints.flight_prints import _FlightPrints
from ..tools import find_closest


class Flight:
    """Keeps all flight information and has a method to simulate flight.

    Attributes
    ----------
    Flight.env : Environment
        Environment object describing rail length, elevation, gravity and
        weather condition. See Environment class for more details.
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
    Flight.FlightPhases : class
        Helper class to organize and manage different flight phases.
    Flight.TimeNodes : class
        Helper class to manage time discretization during simulation.
    Flight.time_iterator : function
        Helper iterator function to generate time discretization points.
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
        If True, decouples ODE time step from parachute trigger functions
        sampling rate. The time steps can overshoot the necessary trigger
        function evaluation points and then interpolation is used to
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
        Rocket's X velocity as a function of time.
    Flight.vy : Function
        Rocket's Y velocity as a function of time.
    Flight.vz : Function
        Rocket's Z velocity as a function of time.
    Flight.e0 : Function
        Rocket's Euler parameter 0 as a function of time.
    Flight.e1 : Function
        Rocket's Euler parameter 1 as a function of time.
    Flight.e2 : Function
        Rocket's Euler parameter 2 as a function of time.
    Flight.e3 : Function
        Rocket's Euler parameter 3 as a function of time.
    Flight.w1 : Function
        Rocket's angular velocity Omega 1 as a function of time.
        Direction 1 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry.
    Flight.w2 : Function
        Rocket's angular velocity Omega 2 as a function of time.
        Direction 2 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry and direction 1.
    Flight.w3 : Function
        Rocket's angular velocity Omega 3 as a function of time.
        Direction 3 is in the rocket's body axis and points in the
        direction of cylindrical symmetry.
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
        List defines initial condition - [tInit, x_init,
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
    Flight.post_processed : bool
        Defines if solution data has been post processed.
    Flight.initial_solution : list
        List defines initial condition - [tInit, x_init,
        y_init, z_init, vx_init, vy_init, vz_init, e0_init, e1_init,
        e2_init, e3_init, w1_init, w2_init, w3_init]
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
    Flight.function_evaluations_per_time_step : array
        List that stores number of derivative function evaluations
        per time step during numerical integration.
    Flight.time_steps : array
        List of time steps taking during numerical integration in
        seconds.
    Flight.FlightPhases : Flight.FlightPhases
        Stores and manages flight phases.
    Flight.wind_velocity_x : Function
        Wind velocity X (East) experienced by the rocket as a
        function of time. Can be called or accessed as array.
    Flight.wind_velocity_y : Function
        Wind velocity Y (North) experienced by the rocket as a
        function of time. Can be called or accessed as array.
    Flight.density : Function
        Air density experienced by the rocket as a function of
        time. Can be called or accessed as array.
    Flight.pressure : Function
        Air pressure experienced by the rocket as a function of
        time. Can be called or accessed as array.
    Flight.dynamic_viscosity : Function
        Air dynamic viscosity experienced by the rocket as a function of
        time. Can be called or accessed as array.
    Flight.speed_of_sound : Function
        Speed of Sound in air experienced by the rocket as a
        function of time. Can be called or accessed as array.
    Flight.ax : Function
        Rocket's X (East) acceleration as a function of time, in m/s².
        Can be called or accessed as array.
    Flight.ay : Function
        Rocket's Y (North) acceleration as a function of time, in m/s².
        Can be called or accessed as array.
    Flight.az : Function
        Rocket's Z (Up) acceleration as a function of time, in m/s².
        Can be called or accessed as array.
    Flight.alpha1 : Function
        Rocket's angular acceleration Alpha 1 as a function of time.
        Direction 1 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry.
        Units of rad/s². Can be called or accessed as array.
    Flight.alpha2 : Function
        Rocket's angular acceleration Alpha 2 as a function of time.
        Direction 2 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry and direction 1.
        Units of rad/s². Can be called or accessed as array.
    Flight.alpha3 : Function
        Rocket's angular acceleration Alpha 3 as a function of time.
        Direction 3 is in the rocket's body axis and points in the
        direction of cylindrical symmetry.
        Units of rad/s². Can be called or accessed as array.
    Flight.speed : Function
        Rocket velocity magnitude in m/s relative to ground as a
        function of time. Can be called or accessed as array.
    Flight.max_speed : float
        Maximum velocity magnitude in m/s reached by the rocket
        relative to ground during flight.
    Flight.max_speed_time : float
        Time in seconds at which rocket reaches maximum velocity
        magnitude relative to ground.
    Flight.horizontal_speed : Function
        Rocket's velocity magnitude in the horizontal (North-East)
        plane in m/s as a function of time. Can be called or
        accessed as array.
    Flight.acceleration : Function
        Rocket acceleration magnitude in m/s² relative to ground as a
        function of time. Can be called or accessed as array.
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
        of time. Can be called or accessed as array.
    Flight.attitude_vector_x : Function
        Rocket's attitude vector, or the vector that points
        in the rocket's axis of symmetry, component in the X
        direction (East) as a function of time.
        Can be called or accessed as array.
    Flight.attitude_vector_y : Function
        Rocket's attitude vector, or the vector that points
        in the rocket's axis of symmetry, component in the Y
        direction (East) as a function of time.
        Can be called or accessed as array.
    Flight.attitude_vector_z : Function
        Rocket's attitude vector, or the vector that points
        in the rocket's axis of symmetry, component in the Z
        direction (East) as a function of time.
        Can be called or accessed as array.
    Flight.attitude_angle : Function
        Rocket's attitude angle, or the angle that the
        rocket's axis of symmetry makes with the horizontal (North-East)
        plane. Measured in degrees and expressed as a function
        of time. Can be called or accessed as array.
    Flight.lateral_attitude_angle : Function
        Rocket's lateral attitude angle, or the angle that the
        rocket's axis of symmetry makes with plane defined by
        the launch rail direction and the Z (up) axis.
        Measured in degrees and expressed as a function
        of time. Can be called or accessed as array.
    Flight.phi : Function
        Rocket's Spin Euler Angle, φ, according to the 3-2-3 rotation
        system (NASA Standard Aerospace). Measured in degrees and
        expressed as a function of time. Can be called or accessed as array.
    Flight.theta : Function
        Rocket's Nutation Euler Angle, θ, according to the 3-2-3 rotation
        system (NASA Standard Aerospace). Measured in degrees and
        expressed as a function of time. Can be called or accessed as array.
    Flight.psi : Function
        Rocket's Precession Euler Angle, ψ, according to the 3-2-3 rotation
        system (NASA Standard Aerospace). Measured in degrees and
        expressed as a function of time. Can be called or accessed as array.
    Flight.R1 : Function
        Resultant force perpendicular to rockets axis due to
        aerodynamic forces as a function of time. Units in N.
        Expressed as a function of time. Can be called or accessed
        as array.
        Direction 1 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry.
    Flight.R2 : Function
        Resultant force perpendicular to rockets axis due to
        aerodynamic forces as a function of time. Units in N.
        Expressed as a function of time. Can be called or accessed
        as array.
        Direction 2 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry and direction 1.
    Flight.R3 : Function
        Resultant force in rockets axis due to aerodynamic forces
        as a function of time. Units in N. Usually just drag.
        Expressed as a function of time. Can be called or accessed
        as array.
        Direction 3 is in the rocket's body axis and points in the
        direction of cylindrical symmetry.
    Flight.M1 : Function
        Resultant moment (torque) perpendicular to rockets axis due to
        aerodynamic forces and eccentricity as a function of time.
        Units in N*m.
        Expressed as a function of time. Can be called or accessed
        as array.
        Direction 1 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry.
    Flight.M2 : Function
        Resultant moment (torque) perpendicular to rockets axis due to
        aerodynamic forces and eccentricity as a function of time.
        Units in N*m.
        Expressed as a function of time. Can be called or accessed
        as array.
        Direction 2 is in the rocket's body axis and points perpendicular
        to the rocket's axis of cylindrical symmetry and direction 1.
    Flight.M3 : Function
        Resultant moment (torque) in rockets axis due to aerodynamic
        forces and eccentricity as a function of time. Units in N*m.
        Expressed as a function of time. Can be called or accessed
        as array.
        Direction 3 is in the rocket's body axis and points in the
        direction of cylindrical symmetry.
    Flight.aerodynamic_lift : Function
        Resultant force perpendicular to rockets axis due to
        aerodynamic effects as a function of time. Units in N.
        Expressed as a function of time. Can be called or accessed
        as array.
    Flight.aerodynamic_drag : Function
        Resultant force aligned with the rockets axis due to
        aerodynamic effects as a function of time. Units in N.
        Expressed as a function of time. Can be called or accessed
        as array.
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
        of time. Can be called or accessed as array.
    Flight.max_rail_button1_normal_force : float
        Maximum upper rail button normal force experienced
        during rail flight phase in N.
    Flight.rail_button1_shear_force : Function
        Upper rail button shear force in N as a function
        of time. Can be called or accessed as array.
    Flight.max_rail_button1_shear_force : float
        Maximum upper rail button shear force experienced
        during rail flight phase in N.
    Flight.rail_button2_normal_force : Function
        Lower rail button normal force in N as a function
        of time. Can be called or accessed as array.
    Flight.max_rail_button2_normal_force : float
        Maximum lower rail button normal force experienced
        during rail flight phase in N.
    Flight.rail_button2_shear_force : Function
        Lower rail button shear force in N as a function
        of time. Can be called or accessed as array.
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
        of time in Watts. Can be called or accessed as array.
    Flight.drag_power : Function
        Aerodynamic drag power output as a function
        of time in Watts. Can be called or accessed as array.
    Flight.attitude_frequency_response : Function
        Fourier Frequency Analysis of the rocket's attitude angle.
        Expressed as the absolute vale of the magnitude as a function
        of frequency in Hz. Can be called or accessed as array.
    Flight.omega1_frequency_response : Function
        Fourier Frequency Analysis of the rocket's angular velocity omega 1.
        Expressed as the absolute vale of the magnitude as a function
        of frequency in Hz. Can be called or accessed as array.
    Flight.omega2_frequency_response : Function
        Fourier Frequency Analysis of the rocket's angular velocity omega 2.
        Expressed as the absolute vale of the magnitude as a function
        of frequency in Hz. Can be called or accessed as array.
    Flight.omega3_frequency_response : Function
        Fourier Frequency Analysis of the rocket's angular velocity omega 3.
        Expressed as the absolute vale of the magnitude as a function
        of frequency in Hz. Can be called or accessed as array.
    Flight.static_margin : Function
        Rocket's static margin during flight in calibers.
    Flight.stability_margin : Function
            Rocket's stability margin during flight, in calibers.
    Flight.stream_velocity_x : Function
        Freestream velocity x (East) component, in m/s, as a function of
        time. Can be called or accessed as array.
    Flight.stream_velocity_y : Function
        Freestream velocity y (North) component, in m/s, as a function of
        time. Can be called or accessed as array.
    Flight.stream_velocity_z : Function
        Freestream velocity z (up) component, in m/s, as a function of
        time. Can be called or accessed as array.
    Flight.free_stream_speed : Function
        Freestream velocity magnitude, in m/s, as a function of time.
        Can be called or accessed as array.
    Flight.apogee_freestream_speed : float
        Freestream speed of the rocket at apogee in m/s.
    Flight.mach_number : Function
        Rocket's Mach number defined as its freestream speed
        divided by the speed of sound at its altitude. Expressed
        as a function of time. Can be called or accessed as array.
    Flight.max_mach_number : float
        Rocket's maximum Mach number experienced during flight.
    Flight.max_mach_number_time : float
        Time at which the rocket experiences the maximum Mach number.
    Flight.reynolds_number : Function
        Rocket's Reynolds number, using its diameter as reference
        length and free_stream_speed as reference velocity. Expressed
        as a function of time. Can be called or accessed as array.
    Flight.max_reynolds_number : float
        Rocket's maximum Reynolds number experienced during flight.
    Flight.max_reynolds_number_time : float
        Time at which the rocket experiences the maximum Reynolds number.
    Flight.dynamic_pressure : Function
        Dynamic pressure experienced by the rocket in Pa as a function
        of time, defined by 0.5*rho*V^2, where rho is air density and V
        is the freestream speed. Can be called or accessed as array.
    Flight.max_dynamic_pressure : float
        Maximum dynamic pressure, in Pa, experienced by the rocket.
    Flight.max_dynamic_pressure_time : float
        Time at which the rocket experiences maximum dynamic pressure.
    Flight.total_pressure : Function
        Total pressure experienced by the rocket in Pa as a function
        of time. Can be called or accessed as array.
    Flight.max_total_pressure : float
        Maximum total pressure, in Pa, experienced by the rocket.
    Flight.max_total_pressure_time : float
        Time at which the rocket experiences maximum total pressure.
    Flight.angle_of_attack : Function
        Rocket's angle of attack in degrees as a function of time.
        Defined as the minimum angle between the attitude vector and
        the freestream velocity vector. Can be called or accessed as
        array.
    """

    def __init__(
        self,
        rocket,
        environment,
        rail_length,
        inclination=80,
        heading=90,
        initial_solution=None,
        terminate_on_apogee=False,
        max_time=600,
        max_time_step=np.inf,
        min_time_step=0,
        rtol=1e-6,
        atol=6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3],
        time_overshoot=True,
        verbose=False,
        name="Flight",
        equations_of_motion="standard",
    ):
        """Run a trajectory simulation.

        Parameters
        ----------
        rocket : Rocket
            Rocket to simulate.
        environment : Environment
            Environment to run simulation on.
        rail_length : int, float
            Length in which the rocket will be attached to the rail, only
            moving along a fixed direction, that is, the line parallel to the
            rail. Currently, if the an initial_solution is passed, the rail
            length is not used.
        inclination : int, float, optional
            Rail inclination angle relative to ground, given in degrees.
            Default is 80.
        heading : int, float, optional
            Heading angle relative to north given in degrees.
            Default is 90, which points in the x direction.
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
            state space variable. Default is 1e-3.
        atol : float, optional
            Maximum absolute error tolerance to be tolerated in the
            integration scheme. Can be given as array for each
            state space variable. Default is 6*[1e-3] + 4*[1e-6] + 3*[1e-3].
        time_overshoot : bool, optional
            If True, decouples ODE time step from parachute trigger functions
            sampling rate. The time steps can overshoot the necessary trigger
            function evaluation points and then interpolation is used to
            calculate them and feed the triggers. Can greatly improve run
            time in some cases. Default is True.
        verbose : bool, optional
            If true, verbose mode is activated. Default is False.
        name : str, optional
            Name of the flight. Default is "Flight".
        equations_of_motion : str, optional
            Type of equations of motion to use. Can be "standard" or
            "solid_propulsion". Default is "standard". Solid propulsion is a
            more restricted set of equations of motion that only works for
            solid propulsion rockets. Such equations were used in RocketPy v0
            and are kept here for backwards compatibility.

        Returns
        -------
        None
        """
        # Fetch helper classes and functions
        FlightPhases = self.FlightPhases
        TimeNodes = self.TimeNodes
        time_iterator = self.time_iterator

        # Save rocket, parachutes, environment, maximum simulation time
        # and termination events
        self.env = environment
        self.rocket = rocket
        self.rail_length = rail_length
        if self.rail_length <= 0:
            raise ValueError("Rail length must be a positive value.")
        self.parachutes = self.rocket.parachutes[:]
        self.inclination = inclination
        self.heading = heading
        self.max_time = max_time
        self.max_time_step = max_time_step
        self.min_time_step = min_time_step
        self.rtol = rtol
        self.atol = atol
        self.initial_solution = initial_solution
        self.time_overshoot = time_overshoot
        self.terminate_on_apogee = terminate_on_apogee
        self.name = name
        self.equations_of_motion = equations_of_motion

        # Flight initialization
        self.__init_post_process_variables()
        self.__init_solution_monitors()
        self.__init_equations_of_motion()

        # Initialize prints and plots objects
        self.prints = _FlightPrints(self)
        self.plots = _FlightPlots(self)

        # Initialize solver monitors
        self.__init_solver_monitors()

        # Create known flight phases
        self.FlightPhases = FlightPhases()
        self.FlightPhases.add_phase(
            self.t_initial, self.initial_derivative, clear=False
        )
        self.FlightPhases.add_phase(self.max_time)

        # Simulate flight
        for phase_index, phase in time_iterator(self.FlightPhases):
            # print('\nCurrent Flight Phase List')
            # print(self.FlightPhases)
            # print('\n\tCurrent Flight Phase')
            # print('\tIndex: ', phase_index, ' | Phase: ', phase)
            # Determine maximum time for this flight phase
            phase.time_bound = self.FlightPhases[phase_index + 1].t

            # Evaluate callbacks
            for callback in phase.callbacks:
                callback(self)

            # Create solver for this flight phase
            self.function_evaluations.append(0)
            phase.solver = integrate.LSODA(
                phase.derivative,
                t0=phase.t,
                y0=self.y_sol,
                t_bound=phase.time_bound,
                min_step=self.min_time_step,
                max_step=self.max_time_step,
                rtol=self.rtol,
                atol=self.atol,
            )
            # print('\n\tSolver Initialization Details')
            # print('\t_initial Time: ', phase.t)
            # print('\t_initial State: ', self.y_sol)
            # print('\tTime Bound: ', phase.time_bound)
            # print('\tMin Step: ', self.min_time_step)
            # print('\tMax Step: ', self.max_time_step)
            # print('\tTolerances: ', self.rtol, self.atol)

            # Initialize phase time nodes
            phase.TimeNodes = TimeNodes()
            # Add first time node to permanent list
            phase.TimeNodes.add_node(phase.t, [], [])
            # Add non-overshootable parachute time nodes
            if self.time_overshoot is False:
                phase.TimeNodes.add_parachutes(
                    self.parachutes, phase.t, phase.time_bound
                )
            # Add lst time node to permanent list
            phase.TimeNodes.add_node(phase.time_bound, [], [])
            # Sort time nodes
            phase.TimeNodes.sort()
            # Merge equal time nodes
            phase.TimeNodes.merge()
            # Clear triggers from first time node if necessary
            if phase.clear:
                phase.TimeNodes[0].parachutes = []
                phase.TimeNodes[0].callbacks = []

            # print('\n\tPhase Time Nodes')
            # print('\tTime Nodes Length: ', str(len(phase.TimeNodes)), ' | Time Nodes Preview: ', phase.TimeNodes[0:3])

            # Iterate through time nodes
            for node_index, node in time_iterator(phase.TimeNodes):
                # print('\n\t\tCurrent Time Node')
                # print('\t\tIndex: ', node_index, ' | Time Node: ', node)
                # Determine time bound for this time node
                node.time_bound = phase.TimeNodes[node_index + 1].t
                phase.solver.t_bound = node.time_bound
                phase.solver._lsoda_solver._integrator.rwork[0] = phase.solver.t_bound
                phase.solver._lsoda_solver._integrator.call_args[
                    4
                ] = phase.solver._lsoda_solver._integrator.rwork
                phase.solver.status = "running"

                # Feed required parachute and discrete controller triggers
                for callback in node.callbacks:
                    callback(self)

                for parachute in node.parachutes:
                    # Calculate and save pressure signal
                    pressure = self.env.pressure.get_value_opt(self.y_sol[2])
                    parachute.clean_pressure_signal.append([node.t, pressure])
                    # Calculate and save noise
                    noise = parachute.noise_function()
                    parachute.noise_signal.append([node.t, noise])
                    parachute.noisy_pressure_signal.append([node.t, pressure + noise])
                    # Gets height above ground level considering noise
                    hAGL = (
                        self.env.pressure.find_input(
                            pressure + noise,
                            self.y_sol[2],
                        )
                        - self.env.elevation
                    )
                    if parachute.triggerfunc(pressure + noise, hAGL, self.y_sol):
                        # print('\nEVENT DETECTED')
                        # print('Parachute Triggered')
                        # print('Name: ', parachute.name, ' | Lag: ', parachute.lag)
                        # Remove parachute from flight parachutes
                        self.parachutes.remove(parachute)
                        # Create flight phase for time after detection and before inflation
                        # Must only be created if parachute has any lag
                        i = 1
                        if parachute.lag != 0:
                            self.FlightPhases.add_phase(
                                node.t,
                                phase.derivative,
                                clear=True,
                                index=phase_index + i,
                            )
                            i += 1
                        # Create flight phase for time after inflation
                        callbacks = [
                            lambda self, parachute_cd_s=parachute.cd_s: setattr(
                                self, "parachute_cd_s", parachute_cd_s
                            )
                        ]
                        self.FlightPhases.add_phase(
                            node.t + parachute.lag,
                            self.u_dot_parachute,
                            callbacks,
                            clear=False,
                            index=phase_index + i,
                        )
                        # Prepare to leave loops and start new flight phase
                        phase.TimeNodes.flush_after(node_index)
                        phase.TimeNodes.add_node(self.t, [], [])
                        phase.solver.status = "finished"
                        # Save parachute event
                        self.parachute_events.append([self.t, parachute])

                # Step through simulation
                while phase.solver.status == "running":
                    # Step
                    phase.solver.step()
                    # Save step result
                    self.solution += [[phase.solver.t, *phase.solver.y]]
                    # Step step metrics
                    self.function_evaluations_per_time_step.append(
                        phase.solver.nfev - self.function_evaluations[-1]
                    )
                    self.function_evaluations.append(phase.solver.nfev)
                    self.time_steps.append(phase.solver.step_size)
                    # Update time and state
                    self.t = phase.solver.t
                    self.y_sol = phase.solver.y
                    if verbose:
                        print(
                            "Current Simulation Time: {:3.4f} s".format(self.t),
                            end="\r",
                        )
                    # print('\n\t\t\tCurrent Step Details')
                    # print('\t\t\tIState: ', phase.solver._lsoda_solver._integrator.istate)
                    # print('\t\t\tTime: ', phase.solver.t)
                    # print('\t\t\tAltitude: ', phase.solver.y[2])
                    # print('\t\t\tEvals: ', self.function_evaluations_per_time_step[-1])

                    # Check for first out of rail event
                    if len(self.out_of_rail_state) == 1 and (
                        self.y_sol[0] ** 2
                        + self.y_sol[1] ** 2
                        + (self.y_sol[2] - self.env.elevation) ** 2
                        >= self.effective_1rl**2
                    ):
                        # Rocket is out of rail
                        # Check exactly when it went out using root finding
                        # States before and after
                        # t0 -> 0
                        # print('\nEVENT DETECTED')
                        # print('Rocket is Out of Rail!')
                        # Disconsider elevation
                        self.solution[-2][3] -= self.env.elevation
                        self.solution[-1][3] -= self.env.elevation
                        # Get points
                        y0 = (
                            sum([self.solution[-2][i] ** 2 for i in [1, 2, 3]])
                            - self.effective_1rl**2
                        )
                        yp0 = 2 * sum(
                            [
                                self.solution[-2][i] * self.solution[-2][i + 3]
                                for i in [1, 2, 3]
                            ]
                        )
                        t1 = self.solution[-1][0] - self.solution[-2][0]
                        y1 = (
                            sum([self.solution[-1][i] ** 2 for i in [1, 2, 3]])
                            - self.effective_1rl**2
                        )
                        yp1 = 2 * sum(
                            [
                                self.solution[-1][i] * self.solution[-1][i + 3]
                                for i in [1, 2, 3]
                            ]
                        )
                        # Put elevation back
                        self.solution[-2][3] += self.env.elevation
                        self.solution[-1][3] += self.env.elevation
                        # Cubic Hermite interpolation (ax**3 + bx**2 + cx + d)
                        D = float(phase.solver.step_size)
                        d = float(y0)
                        c = float(yp0)
                        b = float((3 * y1 - yp1 * D - 2 * c * D - 3 * d) / (D**2))
                        a = float(-(2 * y1 - yp1 * D - c * D - 2 * d) / (D**3)) + 1e-5
                        # Find roots
                        d0 = b**2 - 3 * a * c
                        d1 = 2 * b**3 - 9 * a * b * c + 27 * d * a**2
                        c1 = ((d1 + (d1**2 - 4 * d0**3) ** (0.5)) / 2) ** (1 / 3)
                        t_roots = []
                        for k in [0, 1, 2]:
                            c2 = c1 * (-1 / 2 + 1j * (3**0.5) / 2) ** k
                            t_roots.append(-(1 / (3 * a)) * (b + c2 + d0 / c2))
                        # Find correct root
                        valid_t_root = []
                        for t_root in t_roots:
                            if 0 < t_root.real < t1 and abs(t_root.imag) < 0.001:
                                valid_t_root.append(t_root.real)
                        if len(valid_t_root) > 1:
                            raise ValueError(
                                "Multiple roots found when solving for rail exit time."
                            )
                        elif len(valid_t_root) == 0:
                            raise ValueError(
                                "No valid roots found when solving for rail exit time."
                            )
                        # Determine final state when upper button is going out of rail
                        self.t = valid_t_root[0] + self.solution[-2][0]
                        interpolator = phase.solver.dense_output()
                        self.y_sol = interpolator(self.t)
                        self.solution[-1] = [self.t, *self.y_sol]
                        self.out_of_rail_time = self.t
                        self.out_of_rail_time_index = len(self.solution) - 1
                        self.out_of_rail_state = self.y_sol
                        self.out_of_rail_velocity = (
                            self.y_sol[3] ** 2 + self.y_sol[4] ** 2 + self.y_sol[5] ** 2
                        ) ** (0.5)
                        # Create new flight phase
                        self.FlightPhases.add_phase(
                            self.t,
                            self.u_dot_generalized,
                            index=phase_index + 1,
                        )
                        # Prepare to leave loops and start new flight phase
                        phase.TimeNodes.flush_after(node_index)
                        phase.TimeNodes.add_node(self.t, [], [])
                        phase.solver.status = "finished"

                    # Check for apogee event
                    if len(self.apogee_state) == 1 and self.y_sol[5] < 0:
                        # print('\nPASSIVE EVENT DETECTED')
                        # print('Rocket Has Reached Apogee!')
                        # Apogee reported
                        # Assume linear vz(t) to detect when vz = 0
                        vz0 = self.solution[-2][6]
                        t0 = self.solution[-2][0]
                        vz1 = self.solution[-1][6]
                        t1 = self.solution[-1][0]
                        t_root = -(t1 - t0) * vz0 / (vz1 - vz0) + t0
                        # Fetch state at t_root
                        interpolator = phase.solver.dense_output()
                        self.apogee_state = interpolator(t_root)
                        # Store apogee data
                        self.apogee_time = t_root
                        self.apogee_x = self.apogee_state[0]
                        self.apogee_y = self.apogee_state[1]
                        self.apogee = self.apogee_state[2]
                        if self.terminate_on_apogee:
                            # print('Terminate on Apogee Activated!')
                            self.t = t_root
                            self.t_final = self.t
                            self.state = self.apogee_state
                            # Roll back solution
                            self.solution[-1] = [self.t, *self.state]
                            # Set last flight phase
                            self.FlightPhases.flush_after(phase_index)
                            self.FlightPhases.add_phase(self.t)
                            # Prepare to leave loops and start new flight phase
                            phase.TimeNodes.flush_after(node_index)
                            phase.TimeNodes.add_node(self.t, [], [])
                            phase.solver.status = "finished"
                    # Check for impact event
                    if self.y_sol[2] < self.env.elevation:
                        # print('\nPASSIVE EVENT DETECTED')
                        # print('Rocket Has Reached Ground!')
                        # Impact reported
                        # Check exactly when it went out using root finding
                        # States before and after
                        # t0 -> 0
                        # Disconsider elevation
                        self.solution[-2][3] -= self.env.elevation
                        self.solution[-1][3] -= self.env.elevation
                        # Get points
                        y0 = self.solution[-2][3]
                        yp0 = self.solution[-2][6]
                        t1 = self.solution[-1][0] - self.solution[-2][0]
                        y1 = self.solution[-1][3]
                        yp1 = self.solution[-1][6]
                        # Put elevation back
                        self.solution[-2][3] += self.env.elevation
                        self.solution[-1][3] += self.env.elevation
                        # Cubic Hermite interpolation (ax**3 + bx**2 + cx + d)
                        D = float(phase.solver.step_size)
                        d = float(y0)
                        c = float(yp0)
                        b = float((3 * y1 - yp1 * D - 2 * c * D - 3 * d) / (D**2))
                        a = float(-(2 * y1 - yp1 * D - c * D - 2 * d) / (D**3))
                        # Find roots
                        d0 = b**2 - 3 * a * c
                        d1 = 2 * b**3 - 9 * a * b * c + 27 * d * a**2
                        c1 = ((d1 + (d1**2 - 4 * d0**3) ** (0.5)) / 2) ** (1 / 3)
                        t_roots = []
                        for k in [0, 1, 2]:
                            c2 = c1 * (-1 / 2 + 1j * (3**0.5) / 2) ** k
                            t_roots.append(-(1 / (3 * a)) * (b + c2 + d0 / c2))
                        # Find correct root
                        valid_t_root = []
                        for t_root in t_roots:
                            if 0 < t_root.real < t1 and abs(t_root.imag) < 0.001:
                                valid_t_root.append(t_root.real)
                        if len(valid_t_root) > 1:
                            raise ValueError(
                                "Multiple roots found when solving for impact time."
                            )
                        # Determine impact state at t_root
                        self.t = valid_t_root[0] + self.solution[-2][0]
                        interpolator = phase.solver.dense_output()
                        self.y_sol = interpolator(self.t)
                        # Roll back solution
                        self.solution[-1] = [self.t, *self.y_sol]
                        # Save impact state
                        self.impact_state = self.y_sol
                        self.x_impact = self.impact_state[0]
                        self.y_impact = self.impact_state[1]
                        self.z_impact = self.impact_state[2]
                        self.impact_velocity = self.impact_state[5]
                        self.t_final = self.t
                        # Set last flight phase
                        self.FlightPhases.flush_after(phase_index)
                        self.FlightPhases.add_phase(self.t)
                        # Prepare to leave loops and start new flight phase
                        phase.TimeNodes.flush_after(node_index)
                        phase.TimeNodes.add_node(self.t, [], [])
                        phase.solver.status = "finished"

                    # List and feed overshootable time nodes
                    if self.time_overshoot:
                        # Initialize phase overshootable time nodes
                        overshootable_nodes = TimeNodes()
                        # Add overshootable parachute time nodes
                        overshootable_nodes.add_parachutes(
                            self.parachutes, self.solution[-2][0], self.t
                        )
                        # Add last time node (always skipped)
                        overshootable_nodes.add_node(self.t, [], [])
                        if len(overshootable_nodes) > 1:
                            # Sort overshootable time nodes
                            overshootable_nodes.sort()
                            # Merge equal overshootable time nodes
                            overshootable_nodes.merge()
                            # Clear if necessary
                            if overshootable_nodes[0].t == phase.t and phase.clear:
                                overshootable_nodes[0].parachutes = []
                                overshootable_nodes[0].callbacks = []
                            # print('\n\t\t\t\tOvershootable Time Nodes')
                            # print('\t\t\t\tInterval: ', self.solution[-2][0], self.t)
                            # print('\t\t\t\tOvershootable Nodes Length: ', str(len(overshootable_nodes)), ' | Overshootable Nodes: ', overshootable_nodes)
                            # Feed overshootable time nodes trigger
                            interpolator = phase.solver.dense_output()
                            for (
                                overshootable_index,
                                overshootable_node,
                            ) in time_iterator(overshootable_nodes):
                                # print('\n\t\t\t\tCurrent Overshootable Node')
                                # print('\t\t\t\tIndex: ', overshootable_index, ' | Overshootable Node: ', overshootable_node)
                                # Calculate state at node time
                                overshootable_node.y = interpolator(
                                    overshootable_node.t
                                )
                                # Calculate and save pressure signal
                                pressure = self.env.pressure.get_value_opt(
                                    overshootable_node.y[2]
                                )
                                for parachute in overshootable_node.parachutes:
                                    # Save pressure signal
                                    parachute.clean_pressure_signal.append(
                                        [overshootable_node.t, pressure]
                                    )
                                    # Calculate and save noise
                                    noise = parachute.noise_function()
                                    parachute.noise_signal.append(
                                        [overshootable_node.t, noise]
                                    )
                                    parachute.noisy_pressure_signal.append(
                                        [overshootable_node.t, pressure + noise]
                                    )
                                    # Gets height above ground level considering noise
                                    hAGL = (
                                        self.env.pressure.find_input(
                                            pressure + noise,
                                            overshootable_node.y[2],
                                        )
                                        - self.env.elevation
                                    )

                                    if parachute.triggerfunc(
                                        pressure + noise, hAGL, overshootable_node.y
                                    ):
                                        # print('\nEVENT DETECTED')
                                        # print('Parachute Triggered')
                                        # print('Name: ', parachute.name, ' | Lag: ', parachute.lag)
                                        # Remove parachute from flight parachutes
                                        self.parachutes.remove(parachute)
                                        # Create flight phase for time after detection and before inflation
                                        # Must only be created if parachute has any lag
                                        i = 1
                                        if parachute.lag != 0:
                                            self.FlightPhases.add_phase(
                                                overshootable_node.t,
                                                phase.derivative,
                                                clear=True,
                                                index=phase_index + i,
                                            )
                                            i += 1
                                        # Create flight phase for time after inflation
                                        callbacks = [
                                            lambda self, parachute_cd_s=parachute.cd_s: setattr(
                                                self, "parachute_cd_s", parachute_cd_s
                                            )
                                        ]
                                        self.FlightPhases.add_phase(
                                            overshootable_node.t + parachute.lag,
                                            self.u_dot_parachute,
                                            callbacks,
                                            clear=False,
                                            index=phase_index + i,
                                        )
                                        # Rollback history
                                        self.t = overshootable_node.t
                                        self.y_sol = overshootable_node.y
                                        self.solution[-1] = [
                                            overshootable_node.t,
                                            *overshootable_node.y,
                                        ]
                                        # Prepare to leave loops and start new flight phase
                                        overshootable_nodes.flush_after(
                                            overshootable_index
                                        )
                                        phase.TimeNodes.flush_after(node_index)
                                        phase.TimeNodes.add_node(self.t, [], [])
                                        phase.solver.status = "finished"
                                        # Save parachute event
                                        self.parachute_events.append(
                                            [self.t, parachute]
                                        )

        self.t_final = self.t
        if verbose:
            print("Simulation Completed at Time: {:3.4f} s".format(self.t))

    def __init_post_process_variables(self):
        """Initialize post-process variables."""
        # Initialize all variables calculated after initialization.
        # Important to do so that MATLAB® can access them
        self._drift = Function(0)
        self._bearing = Function(0)
        self._latitude = Function(0)
        self._longitude = Function(0)

    def __init_solution_monitors(self):
        # Initialize solution monitors
        self.out_of_rail_time = 0
        self.out_of_rail_time_index = 0
        self.out_of_rail_state = np.array([0])
        self.out_of_rail_velocity = 0
        self.apogee_state = np.array([0])
        self.apogee_time = 0
        self.apogee_x = 0
        self.apogee_y = 0
        self.apogee = 0
        self.x_impact = 0
        self.y_impact = 0
        self.impact_velocity = 0
        self.impact_state = np.array([0])
        self.parachute_events = []
        self.post_processed = False

        return None

    def __init_flight_state(self):
        """Initialize flight state variables."""
        if self.initial_solution is None:
            # Initialize time and state variables
            self.t_initial = 0
            x_init, y_init, z_init = 0, 0, self.env.elevation
            vx_init, vy_init, vz_init = 0, 0, 0
            w1_init, w2_init, w3_init = 0, 0, 0
            # Initialize attitude
            psi_init = -self.heading * (np.pi / 180)  # Precession / Heading Angle
            theta_init = (self.inclination - 90) * (np.pi / 180)  # Nutation Angle
            e0_init = np.cos(psi_init / 2) * np.cos(theta_init / 2)
            e1_init = np.cos(psi_init / 2) * np.sin(theta_init / 2)
            e2_init = np.sin(psi_init / 2) * np.sin(theta_init / 2)
            e3_init = np.sin(psi_init / 2) * np.cos(theta_init / 2)
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
            self.initial_solution = self.initial_solution.solution[-1]
            # Set unused monitors
            self.out_of_rail_state = self.initial_solution[1:]
            self.out_of_rail_time = self.initial_solution[0]
            self.out_of_rail_time_index = 0
            # Set initial derivative for 6-DOF flight phase
            self.initial_derivative = self.u_dot_generalized
        else:
            # Initial solution given, ignore rail phase
            # TODO: Check if rocket is actually out of rail. Otherwise, start at rail
            self.out_of_rail_state = self.initial_solution[1:]
            self.out_of_rail_time = self.initial_solution[0]
            self.out_of_rail_time_index = 0
            self.initial_derivative = self.u_dot_generalized

    def __init_solver_monitors(self):
        # Initialize solver monitors
        self.function_evaluations = []
        self.function_evaluations_per_time_step = []
        self.time_steps = []
        # Initialize solution state
        self.solution = []
        self.__init_flight_state()

        self.t_initial = self.initial_solution[0]
        self.solution.append(self.initial_solution)
        self.t = self.solution[-1][0]
        self.y_sol = self.solution[-1][1:]

    def __init_equations_of_motion(self):
        """Initialize equations of motion."""
        if self.equations_of_motion == "solid_propulsion":
            self.u_dot_generalized = self.u_dot

    def __init_equations_of_motion(self):
        """Initialize equations of motion."""
        if self.equations_of_motion == "solid_propulsion":
            self.u_dot_generalized = self.u_dot

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
                + rail_buttons.position
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
            lower_r_button = rail_buttons.position
        except IndexError:  # No rail buttons defined
            lower_r_button = nozzle
        effective_2rl = self.rail_length - abs(nozzle - lower_r_button)
        return effective_2rl

    @cached_property
    def frontal_surface_wind(self):
        # Surface wind magnitude in the frontal direction at the rail's elevation
        wind_u = self.env.wind_velocity_x(self.env.elevation)
        wind_v = self.env.wind_velocity_y(self.env.elevation)
        heading_rad = self.heading * np.pi / 180
        frontal_surface_wind = wind_u * np.sin(heading_rad) + wind_v * np.cos(
            heading_rad
        )
        return frontal_surface_wind

    @cached_property
    def lateral_surface_wind(self):
        # Surface wind magnitude in the lateral direction at the rail's elevation
        wind_u = self.env.wind_velocity_x(self.env.elevation)
        wind_v = self.env.wind_velocity_y(self.env.elevation)
        heading_rad = self.heading * np.pi / 180
        lateral_surface_wind = -wind_u * np.cos(heading_rad) + wind_v * np.sin(
            heading_rad
        )
        return lateral_surface_wind

    def udot_rail1(self, t, u, post_processing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying in 1 DOF motion in the rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack. Default is False.

        Return
        ------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].

        """
        # Check if post processing mode is on
        if post_processing:
            # Use u_dot post processing code
            return self.u_dot_generalized(t, u, True)

        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Retrieve important quantities
        # Mass
        M = self.rocket.total_mass.get_value_opt(t)

        # Get freestream speed
        free_stream_speed = (
            (self.env.wind_velocity_x.get_value_opt(z) - vx) ** 2
            + (self.env.wind_velocity_y.get_value_opt(z) - vy) ** 2
            + (vz) ** 2
        ) ** 0.5
        free_stream_mach = free_stream_speed / self.env.speed_of_sound.get_value_opt(z)
        drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)

        # Calculate Forces
        thrust = self.rocket.motor.thrust.get_value_opt(t)
        rho = self.env.density.get_value_opt(z)
        R3 = -0.5 * rho * (free_stream_speed**2) * self.rocket.area * (drag_coeff)

        # Calculate Linear acceleration
        a3 = (R3 + thrust) / M - (
            e0**2 - e1**2 - e2**2 + e3**2
        ) * self.env.gravity(z)
        if a3 > 0:
            ax = 2 * (e1 * e3 + e0 * e2) * a3
            ay = 2 * (e2 * e3 - e0 * e1) * a3
            az = (1 - 2 * (e1**2 + e2**2)) * a3
        else:
            ax, ay, az = 0, 0, 0

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def udot_rail2(self, t, u, post_processing=False):
        """[Still not implemented] Calculates derivative of u state vector with
        respect to time when rocket is flying in 3 DOF motion in the rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack, by default False

        Returns
        -------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].
        """
        # Hey! We will finish this function later, now we just can use u_dot
        return self.u_dot_generalized(t, u, post_processing=post_processing)

    def u_dot(self, t, u, post_processing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying in 6 DOF motion during ascent out of rail
        and descent without parachute.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack, by default False

        Returns
        -------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].
        """

        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        # Determine lift force and moment
        R1, R2 = 0, 0
        M1, M2, M3 = 0, 0, 0
        # Determine current behavior
        if t < self.rocket.motor.burn_out_time:
            # Motor burning
            # Retrieve important motor quantities
            # Inertias
            Tz = self.rocket.motor.I_33.get_value_opt(t)
            Ti = self.rocket.motor.I_11.get_value_opt(t)
            Tzdot = self.rocket.motor.I_33.differentiate(t, dx=1e-6)
            Tidot = self.rocket.motor.I_11.differentiate(t, dx=1e-6)
            # Mass
            Mtdot = self.rocket.motor.mass_flow_rate.get_value_opt(t)
            Mt = self.rocket.motor.propellant_mass.get_value_opt(t)
            # Thrust
            thrust = self.rocket.motor.thrust.get_value_opt(t)
            # Off center moment
            M1 += self.rocket.thrust_eccentricity_x * thrust
            M2 -= self.rocket.thrust_eccentricity_y * thrust
        else:
            # Motor stopped
            # Retrieve important motor quantities
            # Inertias
            Tz, Ti, Tzdot, Tidot = 0, 0, 0, 0
            # Mass
            Mtdot, Mt = 0, 0
            # thrust
            thrust = 0

        # Retrieve important quantities
        # Inertias
        Rz = self.rocket.dry_I_33
        Ri = self.rocket.dry_I_11
        # Mass
        Mr = self.rocket.dry_mass
        M = Mt + Mr
        mu = (Mt * Mr) / (Mt + Mr)
        # Geometry
        # b = -self.rocket.distance_rocket_propellant
        b = (
            -(
                self.rocket.center_of_propellant_position(0)
                - self.rocket.center_of_dry_mass_position
            )
            * self.rocket._csys
        )
        # c = -self.rocket.distance_rocket_nozzle
        c = (
            -(self.rocket.nozzle_position - self.rocket.center_of_dry_mass_position)
            * self.rocket._csys
        )
        a = b * Mt / M
        rN = self.rocket.motor.nozzle_radius
        # Prepare transformation matrix
        a11 = 1 - 2 * (e2**2 + e3**2)
        a12 = 2 * (e1 * e2 - e0 * e3)
        a13 = 2 * (e1 * e3 + e0 * e2)
        a21 = 2 * (e1 * e2 + e0 * e3)
        a22 = 1 - 2 * (e1**2 + e3**2)
        a23 = 2 * (e2 * e3 - e0 * e1)
        a31 = 2 * (e1 * e3 - e0 * e2)
        a32 = 2 * (e2 * e3 + e0 * e1)
        a33 = 1 - 2 * (e1**2 + e2**2)
        # Transformation matrix: (123) -> (XYZ)
        K = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
        # Transformation matrix: (XYZ) -> (123) or K transpose
        Kt = [[a11, a21, a31], [a12, a22, a32], [a13, a23, a33]]

        # Calculate Forces and Moments
        # Get freestream speed
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)
        free_stream_speed = (
            (wind_velocity_x - vx) ** 2 + (wind_velocity_y - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        free_stream_mach = free_stream_speed / self.env.speed_of_sound.get_value_opt(z)

        # Determine aerodynamics forces
        # Determine Drag Force
        if t < self.rocket.motor.burn_out_time:
            drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)
        else:
            drag_coeff = self.rocket.power_off_drag.get_value_opt(free_stream_mach)
        rho = self.env.density.get_value_opt(z)
        R3 = -0.5 * rho * (free_stream_speed**2) * self.rocket.area * (drag_coeff)
        # R3 += self.__computeDragForce(z, Vector(vx, vy, vz))
        # Off center moment
        M1 += self.rocket.cp_eccentricity_y * R3
        M2 -= self.rocket.cp_eccentricity_x * R3
        # Get rocket velocity in body frame
        vx_b = a11 * vx + a21 * vy + a31 * vz
        vy_b = a12 * vx + a22 * vy + a32 * vz
        vz_b = a13 * vx + a23 * vy + a33 * vz
        # Calculate lift and moment for each component of the rocket
        for aero_surface, position in self.rocket.aerodynamic_surfaces:
            comp_cp = (
                position - self.rocket.center_of_dry_mass_position
            ) * self.rocket._csys - aero_surface.cpz
            surface_radius = aero_surface.rocket_radius
            reference_area = np.pi * surface_radius**2
            # Component absolute velocity in body frame
            comp_vx_b = vx_b + comp_cp * omega2
            comp_vy_b = vy_b - comp_cp * omega1
            comp_vz_b = vz_b
            # Wind velocity at component
            comp_z = z + comp_cp
            comp_wind_vx = self.env.wind_velocity_x.get_value_opt(comp_z)
            comp_wind_vy = self.env.wind_velocity_y.get_value_opt(comp_z)
            # Component freestream velocity in body frame
            comp_wind_vx_b = a11 * comp_wind_vx + a21 * comp_wind_vy
            comp_wind_vy_b = a12 * comp_wind_vx + a22 * comp_wind_vy
            comp_wind_vz_b = a13 * comp_wind_vx + a23 * comp_wind_vy
            comp_stream_vx_b = comp_wind_vx_b - comp_vx_b
            comp_stream_vy_b = comp_wind_vy_b - comp_vy_b
            comp_stream_vz_b = comp_wind_vz_b - comp_vz_b
            comp_stream_speed = (
                comp_stream_vx_b**2 + comp_stream_vy_b**2 + comp_stream_vz_b**2
            ) ** 0.5
            # Component attack angle and lift force
            comp_attack_angle = 0
            comp_lift, comp_lift_xb, comp_lift_yb = 0, 0, 0
            if comp_stream_vx_b**2 + comp_stream_vy_b**2 != 0:
                # normalize component stream velocity in body frame
                comp_stream_vz_bn = comp_stream_vz_b / comp_stream_speed
                if -1 * comp_stream_vz_bn < 1:
                    comp_attack_angle = np.arccos(-comp_stream_vz_bn)
                    c_lift = aero_surface.cl(comp_attack_angle, free_stream_mach)
                    # component lift force magnitude
                    comp_lift = (
                        0.5 * rho * (comp_stream_speed**2) * reference_area * c_lift
                    )
                    # component lift force components
                    lift_dir_norm = (
                        comp_stream_vx_b**2 + comp_stream_vy_b**2
                    ) ** 0.5
                    comp_lift_xb = comp_lift * (comp_stream_vx_b / lift_dir_norm)
                    comp_lift_yb = comp_lift * (comp_stream_vy_b / lift_dir_norm)
                    # add to total lift force
                    R1 += comp_lift_xb
                    R2 += comp_lift_yb
                    # Add to total moment
                    M1 -= (comp_cp + a) * comp_lift_yb
                    M2 += (comp_cp + a) * comp_lift_xb
            # Calculates Roll Moment
            try:
                clf_delta, cld_omega, cant_angle_rad = aero_surface.roll_parameters
                M3f = (
                    (1 / 2 * rho * free_stream_speed**2)
                    * reference_area
                    * 2
                    * surface_radius
                    * clf_delta(free_stream_mach)
                    * cant_angle_rad
                )
                M3d = (
                    (1 / 2 * rho * free_stream_speed)
                    * reference_area
                    * (2 * surface_radius) ** 2
                    * cld_omega(free_stream_mach)
                    * omega3
                    / 2
                )
                M3 += M3f - M3d
            except AttributeError:
                pass
        # Calculate derivatives
        # Angular acceleration
        alpha1 = (
            M1
            - (
                omega2 * omega3 * (Rz + Tz - Ri - Ti - mu * b**2)
                + omega1
                * (
                    (Tidot + Mtdot * (Mr - 1) * (b / M) ** 2)
                    - Mtdot * ((rN / 2) ** 2 + (c - b * mu / Mr) ** 2)
                )
            )
        ) / (Ri + Ti + mu * b**2)
        alpha2 = (
            M2
            - (
                omega1 * omega3 * (Ri + Ti + mu * b**2 - Rz - Tz)
                + omega2
                * (
                    (Tidot + Mtdot * (Mr - 1) * (b / M) ** 2)
                    - Mtdot * ((rN / 2) ** 2 + (c - b * mu / Mr) ** 2)
                )
            )
        ) / (Ri + Ti + mu * b**2)
        alpha3 = (M3 - omega3 * (Tzdot - Mtdot * (rN**2) / 2)) / (Rz + Tz)
        # Euler parameters derivative
        e0dot = 0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3)
        e1dot = 0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3)
        e2dot = 0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3)
        e3dot = 0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2)

        # Linear acceleration
        L = [
            (R1 - b * Mt * (omega2**2 + omega3**2) - 2 * c * Mtdot * omega2) / M,
            (R2 + b * Mt * (alpha3 + omega1 * omega2) + 2 * c * Mtdot * omega1) / M,
            (R3 - b * Mt * (alpha2 - omega1 * omega3) + thrust) / M,
        ]
        ax, ay, az = np.dot(K, L)
        az -= self.env.gravity(z)  # Include gravity

        # Create u_dot
        u_dot = [
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            e0dot,
            e1dot,
            e2dot,
            e3dot,
            alpha1,
            alpha2,
            alpha3,
        ]

        if post_processing:
            # Dynamics variables
            self.R1_list.append([t, R1])
            self.R2_list.append([t, R2])
            self.R3_list.append([t, R3])
            self.M1_list.append([t, M1])
            self.M2_list.append([t, M2])
            self.M3_list.append([t, M3])
            # Atmospheric Conditions
            self.wind_velocity_x_list.append(
                [t, self.env.wind_velocity_x.get_value_opt(z)]
            )
            self.wind_velocity_y_list.append(
                [t, self.env.wind_velocity_y.get_value_opt(z)]
            )
            self.density_list.append([t, self.env.density.get_value_opt(z)])
            self.dynamic_viscosity_list.append(
                [t, self.env.dynamic_viscosity.get_value_opt(z)]
            )
            self.pressure_list.append([t, self.env.pressure.get_value_opt(z)])
            self.speed_of_sound_list.append(
                [t, self.env.speed_of_sound.get_value_opt(z)]
            )

        return u_dot

    def u_dot_generalized(self, t, u, post_processing=False):
        """Calculates derivative of u state vector with respect to time when the
        rocket is flying in 6 DOF motion in space and significant mass variation
        effects exist. Typical flight phases include powered ascent after launch
        rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, q0, q1,
            q2, q3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self variables
            such as self.angle_of_attack, by default False.

        Returns
        -------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0_dot, e1_dot, e2_dot, e3_dot, alpha1, alpha2, alpha3].
        """
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Create necessary vectors
        # r = Vector([x, y, z])               # CDM position vector
        v = Vector([vx, vy, vz])  # CDM velocity vector
        e = [e0, e1, e2, e3]  # Euler parameters/quaternions
        w = Vector([omega1, omega2, omega3])  # Angular velocity vector

        # Retrieve necessary quantities
        rho = self.env.density.get_value_opt(z)
        total_mass = self.rocket.total_mass.get_value_opt(t)
        total_mass_dot = self.rocket.total_mass.differentiate(t)
        total_mass_ddot = self.rocket.total_mass.differentiate(t, order=2)
        ## CM position vector and time derivatives relative to CDM in body frame
        r_CM_z = (
            -1
            * (
                (
                    self.rocket.center_of_propellant_position
                    - self.rocket.center_of_dry_mass_position
                )
                * self.rocket._csys
            )
            * self.rocket.motor.propellant_mass
            / total_mass
        )
        r_CM = Vector([0, 0, r_CM_z.get_value_opt(t)])
        r_CM_dot = Vector([0, 0, r_CM_z.differentiate(t)])
        r_CM_ddot = Vector([0, 0, r_CM_z.differentiate(t, order=2)])
        ## Nozzle gyration tensor
        r_NOZ = (
            -(self.rocket.nozzle_position - self.rocket.center_of_dry_mass_position)
            * self.rocket._csys
        )
        S_noz_33 = 0.5 * self.rocket.motor.nozzle_radius**2
        S_noz_11 = 0.5 * S_noz_33 + 0.25 * r_NOZ**2
        S_noz_22 = S_noz_11
        S_noz_12 = 0
        S_noz_13 = 0
        S_noz_23 = 0
        S_nozzle = Matrix(
            [
                [S_noz_11, S_noz_12, S_noz_13],
                [S_noz_12, S_noz_22, S_noz_23],
                [S_noz_13, S_noz_23, S_noz_33],
            ]
        )
        ## Inertia tensor
        I_11 = self.rocket.I_11.get_value_opt(t)
        I_12 = self.rocket.I_12.get_value_opt(t)
        I_13 = self.rocket.I_13.get_value_opt(t)
        I_22 = self.rocket.I_22.get_value_opt(t)
        I_23 = self.rocket.I_23.get_value_opt(t)
        I_33 = self.rocket.I_33.get_value_opt(t)
        I = Matrix(
            [
                [I_11, I_12, I_13],
                [I_12, I_22, I_23],
                [I_13, I_23, I_33],
            ]
        )
        ## Inertia tensor time derivative in the body frame
        I_11_dot = self.rocket.I_11.differentiate(t)
        I_12_dot = self.rocket.I_12.differentiate(t)
        I_13_dot = self.rocket.I_13.differentiate(t)
        I_22_dot = self.rocket.I_22.differentiate(t)
        I_23_dot = self.rocket.I_23.differentiate(t)
        I_33_dot = self.rocket.I_33.differentiate(t)
        I_dot = Matrix(
            [
                [I_11_dot, I_12_dot, I_13_dot],
                [I_12_dot, I_22_dot, I_23_dot],
                [I_13_dot, I_23_dot, I_33_dot],
            ]
        )
        ## Inertia tensor relative to CM
        H = (r_CM.cross_matrix @ -r_CM.cross_matrix) * total_mass
        I_CM = I - H

        # Prepare transformation matrices
        K = Matrix.transformation(e)
        Kt = K.transpose

        # Compute aerodynamic forces and moments
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

        ## Drag force
        rho = self.env.density.get_value_opt(z)
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)
        wind_velocity = Vector([wind_velocity_x, wind_velocity_y, 0])
        free_stream_speed = abs((wind_velocity - Vector(v)))
        free_stream_mach = free_stream_speed / self.env.speed_of_sound.get_value_opt(z)
        if t < self.rocket.motor.burn_out_time:
            drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)
        else:
            drag_coeff = self.rocket.power_off_drag.get_value_opt(free_stream_mach)
        R3 += -0.5 * rho * (free_stream_speed**2) * self.rocket.area * (drag_coeff)

        ## Off center moment
        M1 += self.rocket.cp_eccentricity_y * R3
        M2 -= self.rocket.cp_eccentricity_x * R3

        # Get rocket velocity in body frame
        vB = Kt @ v
        # Calculate lift and moment for each component of the rocket
        for aero_surface, position in self.rocket.aerodynamic_surfaces:
            comp_cpz = (
                position - self.rocket.center_of_dry_mass_position
            ) * self.rocket._csys - aero_surface.cpz
            comp_cp = Vector([0, 0, comp_cpz])
            surface_radius = aero_surface.rocket_radius
            reference_area = np.pi * surface_radius**2
            # Component absolute velocity in body frame
            comp_vb = vB + (w ^ comp_cp)
            # Wind velocity at component altitude
            comp_z = z + (K @ comp_cp).z
            comp_wind_vx = self.env.wind_velocity_x.get_value_opt(comp_z)
            comp_wind_vy = self.env.wind_velocity_y.get_value_opt(comp_z)
            # Component freestream velocity in body frame
            comp_wind_vb = Kt @ Vector([comp_wind_vx, comp_wind_vy, 0])
            comp_stream_velocity = comp_wind_vb - comp_vb
            comp_stream_vx_b, comp_stream_vy_b, comp_stream_vz_b = comp_stream_velocity
            comp_stream_speed = abs(comp_stream_velocity)
            comp_stream_mach = (
                comp_stream_speed / self.env.speed_of_sound.get_value_opt(z)
            )
            # Component attack angle and lift force
            comp_attack_angle = 0
            comp_lift, comp_lift_xb, comp_lift_yb = 0, 0, 0
            if comp_stream_vx_b**2 + comp_stream_vy_b**2 != 0:
                # Normalize component stream velocity in body frame
                comp_stream_vz_bn = comp_stream_vz_b / comp_stream_speed
                if -1 * comp_stream_vz_bn < 1:
                    comp_attack_angle = np.arccos(-comp_stream_vz_bn)
                    c_lift = aero_surface.cl(comp_attack_angle, comp_stream_mach)
                    # Component lift force magnitude
                    comp_lift = (
                        0.5 * rho * (comp_stream_speed**2) * reference_area * c_lift
                    )
                    # Component lift force components
                    lift_dir_norm = (
                        comp_stream_vx_b**2 + comp_stream_vy_b**2
                    ) ** 0.5
                    comp_lift_xb = comp_lift * (comp_stream_vx_b / lift_dir_norm)
                    comp_lift_yb = comp_lift * (comp_stream_vy_b / lift_dir_norm)
                    # Add to total lift force
                    R1 += comp_lift_xb
                    R2 += comp_lift_yb
                    # Add to total moment
                    M1 -= (comp_cpz + r_CM_z.get_value_opt(t)) * comp_lift_yb
                    M2 += (comp_cpz + r_CM_z.get_value_opt(t)) * comp_lift_xb
            # Calculates Roll Moment
            try:
                clf_delta, cld_omega, cant_angle_rad = aero_surface.roll_parameters
                M3f = (
                    (1 / 2 * rho * comp_stream_speed**2)
                    * reference_area
                    * 2
                    * surface_radius
                    * clf_delta(comp_stream_mach)
                    * cant_angle_rad
                )
                M3d = (
                    (1 / 2 * rho * comp_stream_speed)
                    * reference_area
                    * (2 * surface_radius) ** 2
                    * cld_omega(comp_stream_mach)
                    * omega3
                    / 2
                )
                M3 += M3f - M3d
            except AttributeError:
                pass
        weightB = Kt @ Vector([0, 0, -total_mass * self.env.gravity(z)])
        T00 = total_mass * r_CM
        T03 = (
            2 * total_mass_dot * (Vector([0, 0, r_NOZ]) - r_CM)
            - 2 * total_mass * r_CM_dot
        )
        T04 = (
            self.rocket.motor.thrust(t) * Vector([0, 0, 1])
            - total_mass * r_CM_ddot
            - 2 * total_mass_dot * r_CM_dot
            + total_mass_ddot * (Vector([0, 0, r_NOZ]) - r_CM)
        )
        T05 = total_mass_dot * S_nozzle - I_dot

        T20 = ((w ^ T00) ^ w) + (w ^ T03) + T04 + weightB + Vector([R1, R2, R3])

        T21 = ((I @ w) ^ w) + T05 @ w - (weightB ^ r_CM) + Vector([M1, M2, M3])

        # Angular velocity derivative
        w_dot = I_CM.inverse @ (T21 + (T20 ^ r_CM))

        # Velocity vector derivative
        v_dot = K @ (T20 / total_mass - (r_CM ^ w_dot))

        # Euler parameters derivative
        e_dot = [
            0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3),
            0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3),
            0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3),
            0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2),
        ]

        # Position vector derivative
        r_dot = [vx, vy, vz]

        # Create u_dot
        u_dot = [*r_dot, *v_dot, *e_dot, *w_dot]

        if post_processing:
            # Dynamics variables
            self.R1_list.append([t, R1])
            self.R2_list.append([t, R2])
            self.R3_list.append([t, R3])
            self.M1_list.append([t, M1])
            self.M2_list.append([t, M2])
            self.M3_list.append([t, M3])
            # Atmospheric Conditions
            self.wind_velocity_x_list.append(
                [t, self.env.wind_velocity_x.get_value_opt(z)]
            )
            self.wind_velocity_y_list.append(
                [t, self.env.wind_velocity_y.get_value_opt(z)]
            )
            self.density_list.append([t, self.env.density.get_value_opt(z)])
            self.dynamic_viscosity_list.append(
                [t, self.env.dynamic_viscosity.get_value_opt(z)]
            )
            self.pressure_list.append([t, self.env.pressure.get_value_opt(z)])
            self.speed_of_sound_list.append(
                [t, self.env.speed_of_sound.get_value_opt(z)]
            )

        return u_dot

    def u_dot_parachute(self, t, u, post_processing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying under parachute. A 3 DOF approximation is
        used.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack. Default is False.

        Return
        ------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].

        """
        # Parachute data
        cd_s = self.parachute_cd_s
        ka = 1
        R = 1.5
        rho = self.env.density.get_value_opt(u[2])
        to = 1.2
        ma = ka * rho * (4 / 3) * np.pi * R**3
        mp = self.rocket.mass
        eta = 1
        Rdot = (6 * R * (1 - eta) / (1.2**6)) * (
            (1 - eta) * t**5 + eta * (to**3) * (t**2)
        )
        Rdot = 0
        # Get relevant state data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        # Get wind data
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)
        free_stream_speed = (
            (wind_velocity_x - vx) ** 2 + (wind_velocity_y - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        freestream_x = vx - wind_velocity_x
        freestream_y = vy - wind_velocity_y
        freestream_z = vz
        # Determine drag force
        pseudoD = (
            -0.5 * rho * cd_s * free_stream_speed
            - ka * rho * 4 * np.pi * (R**2) * Rdot
        )
        Dx = pseudoD * freestream_x
        Dy = pseudoD * freestream_y
        Dz = pseudoD * freestream_z
        ax = Dx / (mp + ma)
        ay = Dy / (mp + ma)
        az = (Dz - 9.8 * mp) / (mp + ma)

        if post_processing:
            # Dynamics variables
            self.R1_list.append([t, Dx])
            self.R2_list.append([t, Dy])
            self.R3_list.append([t, Dz])
            self.M1_list.append([t, 0])
            self.M2_list.append([t, 0])
            self.M3_list.append([t, 0])
            # Atmospheric Conditions
            self.wind_velocity_x_list.append([t, self.env.wind_velocity_x(z)])
            self.wind_velocity_y_list.append([t, self.env.wind_velocity_y(z)])
            self.density_list.append([t, self.env.density(z)])
            self.dynamic_viscosity_list.append([t, self.env.dynamic_viscosity(z)])
            self.pressure_list.append([t, self.env.pressure(z)])
            self.speed_of_sound_list.append([t, self.env.speed_of_sound(z)])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    @cached_property
    def solution_array(self):
        """Returns solution array of the rocket flight."""
        return np.array(self.solution)

    @cached_property
    def time(self):
        """Returns time array from solution."""
        return self.solution_array[:, 0]

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

    # Process first type of outputs - state vector
    # Transform solution array into Functions
    @funcify_method("Time (s)", "X (m)", "spline", "constant")
    def x(self):
        """Rocket x position as a Function of time."""
        return self.solution_array[:, [0, 1]]

    @funcify_method("Time (s)", "Y (m)", "spline", "constant")
    def y(self):
        """Rocket y position as a Function of time."""
        return self.solution_array[:, [0, 2]]

    @funcify_method("Time (s)", "Z (m)", "spline", "constant")
    def z(self):
        """Rocket z position as a Function of time."""
        return self.solution_array[:, [0, 3]]

    @funcify_method("Time (s)", "Vx (m/s)", "spline", "zero")
    def vx(self):
        """Rocket x velocity as a Function of time."""
        return self.solution_array[:, [0, 4]]

    @funcify_method("Time (s)", "Vy (m/s)", "spline", "zero")
    def vy(self):
        """Rocket y velocity as a Function of time."""
        return self.solution_array[:, [0, 5]]

    @funcify_method("Time (s)", "Vz (m/s)", "spline", "zero")
    def vz(self):
        """Rocket z velocity as a Function of time."""
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
        """Rocket angular velocity ω1 as a Function of time."""
        return self.solution_array[:, [0, 11]]

    @funcify_method("Time (s)", "ω2 (rad/s)", "spline", "zero")
    def w2(self):
        """Rocket angular velocity ω2 as a Function of time."""
        return self.solution_array[:, [0, 12]]

    @funcify_method("Time (s)", "ω3 (rad/s)", "spline", "zero")
    def w3(self):
        """Rocket angular velocity ω3 as a Function of time."""
        return self.solution_array[:, [0, 13]]

    # Process second type of outputs - accelerations components
    @funcify_method("Time (s)", "Ax (m/s²)", "spline", "zero")
    def ax(self):
        """Rocket x acceleration as a Function of time."""
        return self.retrieve_acceleration_arrays[0]

    @funcify_method("Time (s)", "Ay (m/s²)", "spline", "zero")
    def ay(self):
        """Rocket y acceleration as a Function of time."""
        return self.retrieve_acceleration_arrays[1]

    @funcify_method("Time (s)", "Az (m/s²)", "spline", "zero")
    def az(self):
        """Rocket z acceleration as a Function of time."""
        return self.retrieve_acceleration_arrays[2]

    @funcify_method("Time (s)", "α1 (rad/s²)", "spline", "zero")
    def alpha1(self):
        """Rocket angular acceleration α1 as a Function of time."""
        return self.retrieve_acceleration_arrays[3]

    @funcify_method("Time (s)", "α2 (rad/s²)", "spline", "zero")
    def alpha2(self):
        """Rocket angular acceleration α2 as a Function of time."""
        return self.retrieve_acceleration_arrays[4]

    @funcify_method("Time (s)", "α3 (rad/s²)", "spline", "zero")
    def alpha3(self):
        """Rocket angular acceleration α3 as a Function of time."""
        return self.retrieve_acceleration_arrays[5]

    # Process third type of outputs - Temporary values
    @funcify_method("Time (s)", "R1 (N)", "spline", "zero")
    def R1(self):
        """Aerodynamic force along the first axis that is perpendicular to the
        rocket's axis of symmetry as a Function of time."""
        return self.retrieve_temporary_values_arrays[0]

    @funcify_method("Time (s)", "R2 (N)", "spline", "zero")
    def R2(self):
        """Aerodynamic force along the second axis that is perpendicular to the
        rocket's axis of symmetry as a Function of time."""
        return self.retrieve_temporary_values_arrays[1]

    @funcify_method("Time (s)", "R3 (N)", "spline", "zero")
    def R3(self):
        """Aerodynamic force along the rocket's axis of symmetry as a
        Function of time."""
        return self.retrieve_temporary_values_arrays[2]

    @funcify_method("Time (s)", "M1 (Nm)", "spline", "zero")
    def M1(self):
        """Aerodynamic bending moment in the same direction as the axis that is
        perpendicular to the rocket's axis of symmetry as a Function of
        time.
        """
        return self.retrieve_temporary_values_arrays[3]

    @funcify_method("Time (s)", "M2 (Nm)", "spline", "zero")
    def M2(self):
        """Aerodynamic bending moment in the same direction as the axis that is
        perpendicular to the rocket's axis of symmetry as a Function
        of time."""
        return self.retrieve_temporary_values_arrays[4]

    @funcify_method("Time (s)", "M3 (Nm)", "spline", "zero")
    def M3(self):
        """Aerodynamic bending moment in the same direction as the rocket's
        axis of symmetry as a Function of time."""
        return self.retrieve_temporary_values_arrays[5]

    @funcify_method("Time (s)", "Pressure (Pa)", "spline", "constant")
    def pressure(self):
        """Air pressure felt by the rocket as a Function of time."""
        return self.retrieve_temporary_values_arrays[6]

    @funcify_method("Time (s)", "Density (kg/m³)", "spline", "constant")
    def density(self):
        """Air density felt by the rocket as a Function of time."""
        return self.retrieve_temporary_values_arrays[7]

    @funcify_method("Time (s)", "Dynamic Viscosity (Pa s)", "spline", "constant")
    def dynamic_viscosity(self):
        """Air dynamic viscosity felt by the rocket as a Function of
        time."""
        return self.retrieve_temporary_values_arrays[8]

    @funcify_method("Time (s)", "Speed of Sound (m/s)", "spline", "constant")
    def speed_of_sound(self):
        """Speed of sound in the air felt by the rocket as a Function
        of time."""
        return self.retrieve_temporary_values_arrays[9]

    @funcify_method("Time (s)", "Wind Velocity X (East) (m/s)", "spline", "constant")
    def wind_velocity_x(self):
        """Wind velocity in the X direction (east) as a Function of
        time."""
        return self.retrieve_temporary_values_arrays[10]

    @funcify_method("Time (s)", "Wind Velocity Y (North) (m/s)", "spline", "constant")
    def wind_velocity_y(self):
        """Wind velocity in the y direction (north) as a Function of
        time."""
        return self.retrieve_temporary_values_arrays[11]

    # Process fourth type of output - values calculated from previous outputs

    # Kinematics functions and values
    # Velocity Magnitude
    @funcify_method("Time (s)", "Speed - Velocity Magnitude (m/s)")
    def speed(self):
        """Rocket speed, or velocity magnitude, as a Function of time."""
        return (self.vx**2 + self.vy**2 + self.vz**2) ** 0.5

    @cached_property
    def max_speed_time(self):
        """Time at which the rocket reaches its maximum speed."""
        max_speed_time_index = np.argmax(self.speed.get_source()[:, 1])
        return self.speed[max_speed_time_index, 0]

    @cached_property
    def max_speed(self):
        """Maximum speed reached by the rocket."""
        return self.speed(self.max_speed_time)

    # Accelerations
    @funcify_method("Time (s)", "acceleration Magnitude (m/s²)")
    def acceleration(self):
        """Rocket acceleration magnitude as a Function of time."""
        return (self.ax**2 + self.ay**2 + self.az**2) ** 0.5

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
        return self.acceleration(self.max_acceleration_power_on_time)

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
        return self.acceleration[max_acceleration_time_index, 0]

    @cached_property
    def max_acceleration_power_off(self):
        """Maximum acceleration reached by the rocket after motor burn."""
        return self.acceleration(self.max_acceleration_power_off_time)

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
        return (self.vx**2 + self.vy**2) ** 0.5

    # Path Angle
    @funcify_method("Time (s)", "Path Angle (°)", "spline", "constant")
    def path_angle(self):
        """Rocket path angle as a Function of time."""
        path_angle = (180 / np.pi) * np.arctan2(
            self.vz[:, 1], self.horizontal_speed[:, 1]
        )
        return np.column_stack([self.time, path_angle])

    # Attitude Angle
    @funcify_method("Time (s)", "Attitude Vector X Component")
    def attitude_vector_x(self):
        """Rocket attitude vector X component as a Function of time."""
        return 2 * (self.e1 * self.e3 + self.e0 * self.e2)  # a13

    @funcify_method("Time (s)", "Attitude Vector Y Component")
    def attitude_vector_y(self):
        """Rocket attitude vector Y component as a Function of time."""
        return 2 * (self.e2 * self.e3 - self.e0 * self.e1)  # a23

    @funcify_method("Time (s)", "Attitude Vector Z Component")
    def attitude_vector_z(self):
        """Rocket attitude vector Z component as a Function of time."""
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
        attitude_angle = np.column_stack([self.time, attitude_angle])
        return attitude_angle

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
        psi = (180 / np.pi) * (
            np.arctan2(self.e3[:, 1], self.e0[:, 1])
            + np.arctan2(-self.e2[:, 1], -self.e1[:, 1])
        )  # Precession angle
        psi = np.column_stack([self.time, psi])  # Precession angle
        return psi

    @funcify_method("Time (s)", "Spin Angle - φ (°)", "spline", "constant")
    def phi(self):
        """Spin angle as a Function of time."""
        phi = (180 / np.pi) * (
            np.arctan2(self.e3[:, 1], self.e0[:, 1])
            - np.arctan2(-self.e2[:, 1], -self.e1[:, 1])
        )  # Spin angle
        phi = np.column_stack([self.time, phi])  # Spin angle
        return phi

    @funcify_method("Time (s)", "Nutation Angle - θ (°)", "spline", "constant")
    def theta(self):
        """Nutation angle as a Function of time."""
        theta = (
            (180 / np.pi)
            * 2
            * np.arcsin(-((self.e1[:, 1] ** 2 + self.e2[:, 1] ** 2) ** 0.5))
        )  # Nutation angle
        theta = np.column_stack([self.time, theta])  # Nutation angle
        return theta

    # Fluid Mechanics variables
    # Freestream Velocity
    @funcify_method("Time (s)", "Freestream Velocity X (m/s)", "spline", "constant")
    def stream_velocity_x(self):
        """Freestream velocity X component as a Function of time."""
        stream_velocity_x = np.column_stack(
            (self.time, self.wind_velocity_x[:, 1] - self.vx[:, 1])
        )
        return stream_velocity_x

    @funcify_method("Time (s)", "Freestream Velocity Y (m/s)", "spline", "constant")
    def stream_velocity_y(self):
        """Freestream velocity Y component as a Function of time."""
        stream_velocity_y = np.column_stack(
            (self.time, self.wind_velocity_y[:, 1] - self.vy[:, 1])
        )
        return stream_velocity_y

    @funcify_method("Time (s)", "Freestream Velocity Z (m/s)", "spline", "constant")
    def stream_velocity_z(self):
        """Freestream velocity Z component as a Function of time."""
        stream_velocity_z = np.column_stack((self.time, -self.vz[:, 1]))
        return stream_velocity_z

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
        return self.free_stream_speed(self.apogee_time)

    # Mach Number
    @funcify_method("Time (s)", "Mach Number", "spline", "zero")
    def mach_number(self):
        """Mach number as a Function of time."""
        return self.free_stream_speed / self.speed_of_sound

    @cached_property
    def max_mach_number_time(self):
        """Time of maximum Mach number."""
        max_mach_number_time_index = np.argmax(self.mach_number[:, 1])
        return self.mach_number[max_mach_number_time_index, 0]

    @cached_property
    def max_mach_number(self):
        """Maximum Mach number."""
        return self.mach_number(self.max_mach_number_time)

    # Stability Margin
    @cached_property
    def max_stability_margin_time(self):
        """Time of maximum stability margin."""
        max_stability_margin_time_index = np.argmax(self.stability_margin[:, 1])
        return self.stability_margin[max_stability_margin_time_index, 0]

    @cached_property
    def max_stability_margin(self):
        """Maximum stability margin."""
        return self.stability_margin(self.max_stability_margin_time)

    @cached_property
    def min_stability_margin_time(self):
        """Time of minimum stability margin."""
        min_stability_margin_time_index = np.argmin(self.stability_margin[:, 1])
        return self.stability_margin[min_stability_margin_time_index, 0]

    @cached_property
    def min_stability_margin(self):
        """Minimum stability margin."""
        return self.stability_margin(self.min_stability_margin_time)

    # Reynolds Number
    @funcify_method("Time (s)", "Reynolds Number", "spline", "zero")
    def reynolds_number(self):
        """Reynolds number as a Function of time."""
        return (self.density * self.free_stream_speed / self.dynamic_viscosity) * (
            2 * self.rocket.radius
        )

    @cached_property
    def max_reynolds_number_time(self):
        """Time of maximum Reynolds number."""
        max_reynolds_number_time_index = np.argmax(self.reynolds_number[:, 1])
        return self.reynolds_number[max_reynolds_number_time_index, 0]

    @cached_property
    def max_reynolds_number(self):
        """Maximum Reynolds number."""
        return self.reynolds_number(self.max_reynolds_number_time)

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
        return self.dynamic_pressure(self.max_dynamic_pressure_time)

    # Total Pressure
    @funcify_method("Time (s)", "Total Pressure (Pa)", "spline", "zero")
    def total_pressure(self):
        return self.pressure * (1 + 0.2 * self.mach_number**2) ** (3.5)

    @cached_property
    def max_total_pressure_time(self):
        """Time of maximum total pressure."""
        max_total_pressure_time_index = np.argmax(self.total_pressure[:, 1])
        return self.total_pressure[max_total_pressure_time_index, 0]

    @cached_property
    def max_total_pressure(self):
        """Maximum total pressure."""
        return self.total_pressure(self.max_total_pressure_time)

    # Dynamics functions and variables

    #  Aerodynamic Lift and Drag
    @funcify_method("Time (s)", "Aerodynamic Lift Force (N)", "spline", "zero")
    def aerodynamic_lift(self):
        """Aerodynamic lift force as a Function of time."""
        return (self.R1**2 + self.R2**2) ** 0.5

    @funcify_method("Time (s)", "Aerodynamic Drag Force (N)", "spline", "zero")
    def aerodynamic_drag(self):
        """Aerodynamic drag force as a Function of time."""
        return -1 * self.R3

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
        translational_energy = (
            0.5 * total_mass * (self.vx**2 + self.vy**2 + self.vz**2)
        )
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
        # Constants
        GM = 3.986004418e14
        # Redefine total_mass time grid to allow for efficient Function algebra
        total_mass = deepcopy(self.rocket.total_mass)
        total_mass.set_discrete_based_on_model(self.z)
        potential_energy = (
            GM
            * total_mass
            * (1 / (self.z + self.env.earth_radius) - 1 / self.env.earth_radius)
        )
        return potential_energy

    # Total Mechanical Energy
    @funcify_method("Time (s)", "Mechanical Energy (J)", "spline", "constant")
    def total_energy(self):
        """Total mechanical energy as a Function of time."""
        return self.kinetic_energy + self.potential_energy

    # thrust Power
    @funcify_method("Time (s)", "thrust Power (W)", "spline", "zero")
    def thrust_power(self):
        """thrust power as a Function of time."""
        thrust = deepcopy(self.rocket.motor.thrust)
        thrust = thrust.set_discrete_based_on_model(self.speed)
        thrust_power = thrust * self.speed
        return thrust_power

    # Drag Power
    @funcify_method("Time (s)", "Drag Power (W)", "spline", "zero")
    def drag_power(self):
        """Drag power as a Function of time."""
        drag_power = self.R3 * self.speed
        drag_power.set_outputs("Drag Power (W)")
        return drag_power

    # Angle of Attack
    @funcify_method("Time (s)", "Angle of Attack (°)", "spline", "constant")
    def angle_of_attack(self):
        """Angle of attack of the rocket with respect to the freestream
        velocity vector."""
        dot_product = [
            -self.attitude_vector_x.get_value_opt(i)
            * self.stream_velocity_x.get_value_opt(i)
            - self.attitude_vector_y.get_value_opt(i)
            * self.stream_velocity_y.get_value_opt(i)
            - self.attitude_vector_z.get_value_opt(i)
            * self.stream_velocity_z.get_value_opt(i)
            for i in self.time
        ]
        # Define freestream speed list
        free_stream_speed = [self.free_stream_speed(i) for i in self.time]
        free_stream_speed = np.nan_to_num(free_stream_speed)

        # Normalize dot product
        dot_product_normalized = [
            i / j if j > 1e-6 else 0 for i, j in zip(dot_product, free_stream_speed)
        ]
        dot_product_normalized = np.nan_to_num(dot_product_normalized)
        dot_product_normalized = np.clip(dot_product_normalized, -1, 1)

        # Calculate angle of attack and convert to degrees
        angle_of_attack = np.rad2deg(np.arccos(dot_product_normalized))
        angle_of_attack = np.column_stack([self.time, angle_of_attack])

        return angle_of_attack

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
        """Stability margin of the rocket along the flight, it considers the
        variation of the center of pressure position according to the mach
        number, as well as the variation of the center of gravity position
        according to the propellant mass evolution.

        Parameters
        ----------
        None

        Returns
        -------
        stability : rocketpy.Function
            Stability margin as a rocketpy.Function of time. The stability margin
            is defined as the distance between the center of pressure and the
            center of gravity, divided by the rocket diameter.
        """
        return [(t, self.rocket.stability_margin(m, t)) for t, m in self.mach_number]

    # Rail Button Forces

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

    @funcify_method(
        "Time (s)", "Horizontal Distance to Launch Point (m)", "spline", "constant"
    )
    def drift(self):
        """Rocket horizontal distance to tha launch point, in meters, as a
        Function of time."""
        return np.column_stack(
            (self.time, (self.x[:, 1] ** 2 + self.y[:, 1] ** 2) ** 0.5)
        )

    @funcify_method("Time (s)", "Bearing (°)", "spline", "constant")
    def bearing(self):
        """Rocket bearing compass, in degrees, as a Function of time."""
        x, y = self.x[:, 1], self.y[:, 1]
        bearing = (2 * np.pi - np.arctan2(-x, y)) * (180 / np.pi)
        return np.column_stack((self.time, bearing))

    @funcify_method("Time (s)", "Latitude (°)", "linear", "constant")
    def latitude(self):
        """Rocket latitude coordinate, in degrees, as a Function of
        time.
        """
        lat1 = np.deg2rad(self.env.latitude)  # Launch lat point converted to radians

        # Applies the haversine equation to find final lat/lon coordinates
        latitude = np.rad2deg(
            np.arcsin(
                np.sin(lat1) * np.cos(self.drift[:, 1] / self.env.earth_radius)
                + np.cos(lat1)
                * np.sin(self.drift[:, 1] / self.env.earth_radius)
                * np.cos(np.deg2rad(self.bearing[:, 1]))
            )
        )
        return np.column_stack((self.time, latitude))

    @funcify_method("Time (s)", "Longitude (°)", "linear", "constant")
    def longitude(self):
        """Rocket longitude coordinate, in degrees, as a Function of
        time.
        """
        lat1 = np.deg2rad(self.env.latitude)  # Launch lat point converted to radians
        lon1 = np.deg2rad(self.env.longitude)  # Launch lon point converted to radians

        # Applies the haversine equation to find final lat/lon coordinates
        longitude = np.rad2deg(
            lon1
            + np.arctan2(
                np.sin(np.deg2rad(self.bearing[:, 1]))
                * np.sin(self.drift[:, 1] / self.env.earth_radius)
                * np.cos(lat1),
                np.cos(self.drift[:, 1] / self.env.earth_radius)
                - np.sin(lat1) * np.sin(np.deg2rad(self.latitude[:, 1])),
            )
        )

        return np.column_stack((self.time, longitude))

    @cached_property
    def retrieve_acceleration_arrays(self):
        """Retrieve acceleration arrays from the integration scheme

        Parameters
        ----------

        Returns
        -------
        ax: list
            acceleration in x direction
        ay: list
            acceleration in y direction
        az: list
            acceleration in z direction
        alpha1: list
            angular acceleration in x direction
        alpha2: list
            angular acceleration in y direction
        alpha3: list
            angular acceleration in z direction
        """
        # Initialize acceleration arrays
        ax, ay, az = [[0, 0]], [[0, 0]], [[0, 0]]
        alpha1, alpha2, alpha3 = [[0, 0]], [[0, 0]], [[0, 0]]
        # Go through each time step and calculate accelerations
        # Get flight phases
        for phase_index, phase in self.time_iterator(self.FlightPhases):
            init_time = phase.t
            final_time = self.FlightPhases[phase_index + 1].t
            current_derivative = phase.derivative
            # Call callback functions
            for callback in phase.callbacks:
                callback(self)
            # Loop through time steps in flight phase
            for step in self.solution:  # Can be optimized
                if init_time < step[0] <= final_time:
                    # Get derivatives
                    u_dot = current_derivative(step[0], step[1:])
                    # Get accelerations
                    ax_value, ay_value, az_value = u_dot[3:6]
                    alpha1_value, alpha2_value, alpha3_value = u_dot[10:]
                    # Save accelerations
                    ax.append([step[0], ax_value])
                    ay.append([step[0], ay_value])
                    az.append([step[0], az_value])
                    alpha1.append([step[0], alpha1_value])
                    alpha2.append([step[0], alpha2_value])
                    alpha3.append([step[0], alpha3_value])

        return ax, ay, az, alpha1, alpha2, alpha3

    @cached_property
    def retrieve_temporary_values_arrays(self):
        """Retrieve temporary values arrays from the integration scheme.
        Currently, the following temporary values are retrieved: ``R1`` , ``R2``
        ``R3`` , ``M1`` , ``M2`` , ``M3`` , ``pressure`` , ``density`` ,
        ``dynamic_viscosity`` , ``speed_of_sound`` .

        Returns
        -------
        self.R1_list: list
            R1 values.
        self.R2_list: list
            R2 values.
        self.R3_list: list
            R3 values are the aerodynamic force values in the rocket's axis
            direction.
        self.M1_list: list
            M1 values.
        self.M2_list: list
            Aerodynamic bending moment in e2 direction at each time step.
        self.M3_list: list
            Aerodynamic bending moment in e3 direction at each time step.
        self.pressure_list: list
            Air pressure at each time step.
        self.density_list: list
            Air density at each time step.
        self.dynamic_viscosity_list: list
            Dynamic viscosity at each time step.
        self.speed_of_sound_list: list
            Speed of sound at each time step.
        self.wind_velocity_x_list: list
            Wind velocity in x direction at each time step.
        self.wind_velocity_y_list: list
            Wind velocity in y direction at each time step.
        """

        # Initialize force and atmospheric arrays
        self.R1_list = []
        self.R2_list = []
        self.R3_list = []
        self.M1_list = []
        self.M2_list = []
        self.M3_list = []
        self.pressure_list = []
        self.density_list = []
        self.dynamic_viscosity_list = []
        self.speed_of_sound_list = []
        self.wind_velocity_x_list = []
        self.wind_velocity_y_list = []

        # Go through each time step and calculate forces and atmospheric values
        # Get flight phases
        for phase_index, phase in self.time_iterator(self.FlightPhases):
            init_time = phase.t
            final_time = self.FlightPhases[phase_index + 1].t
            current_derivative = phase.derivative
            # Call callback functions
            for callback in phase.callbacks:
                callback(self)
            # Loop through time steps in flight phase
            for step in self.solution:  # Can be optimized
                if init_time < step[0] <= final_time or (
                    init_time == self.t_initial and step[0] == self.t_initial
                ):
                    # Call derivatives in post processing mode
                    u_dot = current_derivative(step[0], step[1:], post_processing=True)

        temporary_values = [
            self.R1_list,
            self.R2_list,
            self.R3_list,
            self.M1_list,
            self.M2_list,
            self.M3_list,
            self.pressure_list,
            self.density_list,
            self.dynamic_viscosity_list,
            self.speed_of_sound_list,
            self.wind_velocity_x_list,
            self.wind_velocity_y_list,
        ]

        return temporary_values

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
        null_force = []
        if self.out_of_rail_time_index == 0:  # No rail phase, no rail button forces
            warnings.warn(
                "Trying to calculate rail button forces without a rail phase defined."
                + "The rail button forces will be set to zero."
            )
            return null_force, null_force, null_force, null_force
        if len(self.rocket.rail_buttons) == 0:
            warnings.warn(
                "Trying to calculate rail button forces without rail buttons defined."
                + "The rail button forces will be set to zero."
            )
            return null_force, null_force, null_force, null_force

        # Distance from Rail Button 1 (upper) to CM
        rail_buttons_tuple = self.rocket.rail_buttons[0]
        upper_button_position = (
            rail_buttons_tuple.component.buttons_distance + rail_buttons_tuple.position
        )
        lower_button_position = rail_buttons_tuple.position
        angular_position_rad = (
            rail_buttons_tuple.component.angular_position * np.pi / 180
        )
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

    def _calculate_pressure_signal(self):
        """Calculate the pressure signal from the pressure sensor.
        It creates a signal_function attribute in the parachute object.
        Parachute works as a subclass of Rocket class.

        Returns
        -------
        None
        """
        # Transform parachute sensor feed into functions
        for parachute in self.rocket.parachutes:
            parachute.clean_pressure_signal_function = Function(
                parachute.clean_pressure_signal,
                "Time (s)",
                "Pressure - Without Noise (Pa)",
                "linear",
            )
            parachute.noisy_pressure_signal_function = Function(
                parachute.noisy_pressure_signal,
                "Time (s)",
                "Pressure - With Noise (Pa)",
                "linear",
            )
            parachute.noise_signal_function = Function(
                parachute.noise_signal, "Time (s)", "Pressure Noise (Pa)", "linear"
            )

        return None

    def post_process(self, interpolation="spline", extrapolation="natural"):
        """This method is **deprecated** and is only kept here for backwards
        compatibility. All attributes that need to be post processed are
        computed just in time.

        Post-process all Flight information produced during
        simulation. Includes the calculation of maximum values,
        calculation of secondary values such as energy and conversion
        of lists to Function objects to facilitate plotting.

        Returns
        -------
        None
        """
        # Register post processing
        self.post_processed = True

        return None

    def calculate_stall_wind_velocity(self, stall_angle):
        """Function to calculate the maximum wind velocity before the angle of
        attack exceeds a desired angle, at the instant of departing rail launch.
        Can be helpful if you know the exact stall angle of all aerodynamics
        surfaces.

        Parameters
        ----------
        stall_angle : float
            Angle, in degrees, for which you would like to know the maximum wind
            speed before the angle of attack exceeds it
        Return
        ------
        None
        """
        v_f = self.out_of_rail_velocity

        # Convert angle to radians
        theta = self.inclination * 3.14159265359 / 180
        stall_angle = stall_angle * 3.14159265359 / 180

        c = (math.cos(stall_angle) ** 2 - math.cos(theta) ** 2) / math.sin(
            stall_angle
        ) ** 2
        w_v = (
            2 * v_f * math.cos(theta) / c
            + (
                4 * v_f * v_f * math.cos(theta) * math.cos(theta) / (c**2)
                + 4 * 1 * v_f * v_f / c
            )
            ** 0.5
        ) / 2

        # Convert stall_angle to degrees
        stall_angle = stall_angle * 180 / np.pi
        print(
            "Maximum wind velocity at Rail Departure time before angle"
            + f" of attack exceeds {stall_angle:.3f}°: {w_v:.3f} m/s"
        )

        return None

    def export_pressures(self, file_name, time_step):
        """Exports the pressure experienced by the rocket during the flight to
        an external file, the '.csv' format is recommended, as the columns will
        be separated by commas. It can handle flights with or without
        parachutes, although it is not possible to get a noisy pressure signal
        if no parachute is added.

        If a parachute is added, the file will contain 3 columns: time in
        seconds, clean pressure in Pascals and noisy pressure in Pascals.
        For flights without parachutes, the third column will be discarded

        This function was created especially for the 'Projeto Jupiter'
        Electronics Subsystems team and aims to help in configuring
        micro-controllers.

        Parameters
        ----------
        file_name : string
            The final file name,
        time_step : float
            Time step desired for the final file

        Return
        ------
        None
        """
        time_points = np.arange(0, self.t_final, time_step)
        # pylint: disable=W1514, E1121
        with open(file_name, "w") as file:
            if len(self.rocket.parachutes) == 0:
                print("No parachutes in the rocket, saving static pressure.")
                for t in time_points:
                    file.write(f"{t:f}, {self.pressure(t):.5f}\n")
            else:
                for parachute in self.rocket.parachutes:
                    for t in time_points:
                        p_cl = parachute.clean_pressure_signal(t)
                        p_ns = parachute.noisy_pressure_signal(t)
                        file.write(f"{t:f}, {p_cl:.5f}, {p_ns:.5f}\n")
                    # We need to save only 1 parachute data
                    break

    def export_data(self, file_name, *variables, time_step=None):
        """Exports flight data to a comma separated value file (.csv).

        Data is exported in columns, with the first column representing time
        steps. The first line of the file is a header line, specifying the
        meaning of each column and its units.

        Parameters
        ----------
        file_name : string
            The file name or path of the exported file. Example: flight_data.csv
            Do not use forbidden characters, such as / in Linux/Unix and
            `<, >, :, ", /, \\, | ?, *` in Windows.
        variables : strings, optional
            Names of the data variables which shall be exported. Must be Flight
            class attributes which are instances of the Function class. Usage
            example: test_flight.export_data('test.csv', 'z', 'angle_of_attack',
            'mach_number').
        time_step : float, optional
            Time step desired for the data. If None, all integration time steps
            will be exported. Otherwise, linear interpolation is carried out to
            calculate values at the desired time steps. Example: 0.001.
        """

        # Fast evaluation for the most basic scenario
        if time_step is None and len(variables) == 0:
            np.savetxt(
                file_name,
                self.solution,
                fmt="%.6f",
                delimiter=",",
                header=""
                "Time (s),"
                "X (m),"
                "Y (m),"
                "Z (m),"
                "E0,"
                "E1,"
                "E2,"
                "E3,"
                "W1 (rad/s),"
                "W2 (rad/s),"
                "W3 (rad/s)",
            )
            return

        # Not so fast evaluation for general case
        if variables is None:
            variables = [
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "e0",
                "e1",
                "e2",
                "e3",
                "w1",
                "w2",
                "w3",
            ]

        if time_step is None:
            time_points = self.time
        else:
            time_points = np.arange(self.t_initial, self.t_final, time_step)

        exported_matrix = [time_points]
        exported_header = "Time (s)"

        # Loop through variables, get points and names (for the header)
        for variable in variables:
            if variable in self.__dict__.keys():
                variable_function = self.__dict__[variable]
            # Deal with decorated Flight methods
            else:
                try:
                    obj = getattr(self.__class__, variable)
                    variable_function = obj.__get__(self, self.__class__)
                except AttributeError:
                    raise AttributeError(
                        "Variable '{}' not found in Flight class".format(variable)
                    )
            variable_points = variable_function(time_points)
            exported_matrix += [variable_points]
            exported_header += ", " + variable_function.__outputs__[0]

        exported_matrix = np.array(exported_matrix).T  # Fix matrix orientation

        np.savetxt(
            file_name,
            exported_matrix,
            fmt="%.6f",
            delimiter=",",
            header=exported_header,
            encoding="utf-8",
        )

        return

    def export_kml(
        self,
        file_name="trajectory.kml",
        time_step=None,
        extrude=True,
        color="641400F0",
        altitude_mode="absolute",
    ):
        """Exports flight data to a .kml file, which can be opened with Google
        Earth to display the rocket's trajectory.

        Parameters
        ----------
        file_name : string
            The file name or path of the exported file. Example: flight_data.csv
            Do not use forbidden characters, such as '/' in Linux/Unix and
            '<, >, :, ", /, \\, | ?, *' in Windows.
        time_step : float, optional
            Time step desired for the data. If None, all integration time steps
            will be exported. Otherwise, linear interpolation is carried out to
            calculate values at the desired time steps. Example: 0.001.
        extrude: bool, optional
            To be used if you want to project the path over ground by using an
            extruded polygon. In case False only the linestring containing the
            flight path will be created. Default is True.
        color : str, optional
            Color of your trajectory path, need to be used in specific kml
            format. Refer to http://www.zonums.com/gmaps/kml_color/ for more
            info.
        altitude_mode: str
            Select elevation values format to be used on the kml file. Use
            'relativetoground' if you want use Above Ground Level elevation, or
            'absolute' if you want to parse elevation using Above Sea Level.
            Default is 'relativetoground'. Only works properly if the ground
            level is flat. Change to 'absolute' if the terrain is to irregular
            or contains mountains.
        Returns
        -------
        None
        """
        # Define time points vector
        if time_step is None:
            time_points = self.time
        else:
            time_points = np.arange(self.t_initial, self.t_final + time_step, time_step)
        # Open kml file with simplekml library
        kml = simplekml.Kml(open=1)
        trajectory = kml.newlinestring(name="Rocket Trajectory - Powered by RocketPy")
        coords = []
        if altitude_mode == "relativetoground":
            # In this mode the elevation data will be the Above Ground Level
            # elevation. Only works properly if the ground level is similar to
            # a plane, i.e. it might not work well if the terrain has mountains
            for t in time_points:
                coords.append(
                    (
                        self.longitude(t),
                        self.latitude(t),
                        self.z(t) - self.env.elevation,
                    )
                )
            trajectory.coords = coords
            trajectory.altitudemode = simplekml.AltitudeMode.relativetoground
        else:  # altitude_mode == 'absolute'
            # In this case the elevation data will be the Above Sea Level elevation
            # Ensure you use the correct value on self.env.elevation, otherwise
            # the trajectory path can be offset from ground
            for t in time_points:
                coords.append((self.longitude(t), self.latitude(t), self.z(t)))
            trajectory.coords = coords
            trajectory.altitudemode = simplekml.AltitudeMode.absolute
        # Modify style of trajectory linestring
        trajectory.style.linestyle.color = color
        trajectory.style.polystyle.color = color
        if extrude:
            trajectory.extrude = 1
        # Save the KML
        kml.save(file_name)
        print("File ", file_name, " saved with success!")

        return None

    def info(self):
        """Prints out a summary of the data available about the Flight.

        Returns
        -------
        None
        """
        self.prints.all()
        return None

    def all_info(self):
        """Prints out all data and graphs available about the Flight.

        Returns
        -------
        None
        """

        # Print a summary of data about the flight
        self.info()

        self.plots.all()

        return None

    def time_iterator(self, node_list):
        i = 0
        while i < len(node_list) - 1:
            yield i, node_list[i]
            i += 1

    class FlightPhases:
        """Class to handle flight phases. It is used to store the derivatives
        and callbacks for each flight phase. It is also used to handle the
        insertion of flight phases in the correct order, according to their
        initial time.

        Attributes
        ----------
        list : list
            A list of FlightPhase objects. See FlightPhase subclass.
        """

        def __init__(self, init_list=None):
            init_list = [] if init_list is None else init_list
            self.list = init_list[:]

        def __getitem__(self, index):
            return self.list[index]

        def __len__(self):
            return len(self.list)

        def __repr__(self):
            return str(self.list)

        def display_warning(self, *messages):
            """A simple function to print a warning message."""
            print("WARNING:", *messages)

        def add(self, flight_phase, index=None):
            """Add a flight phase to the list. It will be inserted in the
            correct position, according to its initial time. If no index is
            provided, it will be appended to the end of the list. If by any
            reason the flight phase cannot be inserted in the correct position,
            a warning will be displayed and the flight phase will be inserted
            in the closest position possible.

            Parameters
            ----------
            flight_phase : FlightPhase
                The flight phase object to be added. See FlightPhase class.
            index : int, optional
                The index of the flight phase to be added. If no index is
                provided, the flight phase will be appended to the end of the
                list. Default is None.

            Returns
            -------
            None
            """
            # Handle first phase
            if len(self.list) == 0:
                self.list.append(flight_phase)
                return None

            # Handle appending to last position
            if index is None:
                previous_phase = self.list[-1]
                if flight_phase.t > previous_phase.t:
                    self.list.append(flight_phase)
                    return None
                warning_msg = (
                    (
                        "Trying to add flight phase starting together with the one preceding it. ",
                        "This may be caused by multiple parachutes being triggered simultaneously.",
                    )
                    if flight_phase.t == previous_phase.t
                    else (
                        "Trying to add flight phase starting *before* the one *preceding* it. ",
                        "This may be caused by multiple parachutes being triggered simultaneously",
                        " or by having a negative parachute lag.",
                    )
                )
                self.display_warning(*warning_msg)
                flight_phase.t += 1e-7 if flight_phase.t == previous_phase.t else 0
                self.add(
                    flight_phase, -2 if flight_phase.t < previous_phase.t else None
                )
                return None

            # Handle inserting into intermediary position.
            # Check if new flight phase respects time
            next_phase = self.list[index]
            previous_phase = self.list[index - 1]
            if previous_phase.t < flight_phase.t < next_phase.t:
                self.list.insert(index, flight_phase)
                return None
            warning_msg = (
                (
                    "Trying to add flight phase starting *together* with the one *preceding* it. ",
                    "This may be caused by multiple parachutes being triggered simultaneously.",
                )
                if flight_phase.t == previous_phase.t
                else (
                    "Trying to add flight phase starting *together* with the one *proceeding* it. ",
                    "This may be caused by multiple parachutes being triggered simultaneously.",
                )
                if flight_phase.t == next_phase.t
                else (
                    "Trying to add flight phase starting *before* the one *preceding* it. ",
                    "This may be caused by multiple parachutes being triggered simultaneously",
                    " or by having a negative parachute lag.",
                )
                if flight_phase.t < previous_phase.t
                else (
                    "Trying to add flight phase starting *after* the one *proceeding* it.",
                    "This may be caused by multiple parachutes being triggered simultaneously.",
                )
            )
            self.display_warning(*warning_msg)
            adjust = 1e-7 if flight_phase.t in {previous_phase.t, next_phase.t} else 0
            new_index = (
                index - 1
                if flight_phase.t < previous_phase.t
                else index + 1
                if flight_phase.t > next_phase.t
                else index
            )
            flight_phase.t += adjust
            self.add(flight_phase, new_index)

        def add_phase(self, t, derivatives=None, callback=None, clear=True, index=None):
            """Add a new flight phase to the list, with the specified
            characteristics. This method creates a new FlightPhase instance and
            adds it to the flight phases list, either at the specified index
            position or appended to the end. See FlightPhases.add() for more
            information.

            Parameters
            ----------
            t : float
                The initial time of the new flight phase.
            derivatives : function, optional
                A function representing the derivatives of the flight phase.
                Default is None.
            callback : list of functions, optional
                A list of callback functions to be executed during the flight
                phase. Default is None. You can also pass an empty list.
            clear : bool, optional
                A flag indicating whether to clear the solution after the phase.
                Default is True.
            index : int, optional
                The index at which the new flight phase should be inserted.
                If not provided, the flight phase will be appended
                to the end of the list. Default is None.

            Returns
            -------
            None
            """
            self.add(self.FlightPhase(t, derivatives, callback, clear), index)

        def flush_after(self, index):
            """This function deletes all flight phases after a given index.

            Parameters
            ----------
            index : int
                The index of the last flight phase to be kept.

            Returns
            -------
            None
            """
            del self.list[index + 1 :]

        class FlightPhase:
            """Class to store a flight phase. It stores the initial time, the
            derivative function, the callback functions and a flag to clear
            the solution after the phase.

            Attributes
            ----------
            t : float
                The initial time of the flight phase.
            derivative : function
                A function representing the derivatives of the flight phase.
            callbacks : list of functions
                A list of callback functions to be executed during the flight
                phase.
            clear : bool
                A flag indicating whether to clear the solution after the phase.
            """

            def __init__(self, t, derivative=None, callbacks=None, clear=True):
                self.t = t
                self.derivative = derivative
                self.callbacks = callbacks[:] if callbacks is not None else []
                self.clear = clear

            def __repr__(self):
                if self.derivative is None:
                    return "{Initial Time: " + str(self.t) + " | Derivative: None}"
                return (
                    "{Initial Time: "
                    + str(self.t)
                    + " | Derivative: "
                    + self.derivative.__name__
                    + "}"
                )

    class TimeNodes:
        def __init__(self, init_list=[]):
            self.list = init_list[:]

        def __getitem__(self, index):
            return self.list[index]

        def __len__(self):
            return len(self.list)

        def __repr__(self):
            return str(self.list)

        def add(self, time_node):
            self.list.append(time_node)

        def add_node(self, t, parachutes, callbacks):
            self.list.append(self.TimeNode(t, parachutes, callbacks))

        def add_parachutes(self, parachutes, t_init, t_end):
            # Iterate over parachutes
            for parachute in parachutes:
                # Calculate start of sampling time nodes
                pcDt = 1 / parachute.sampling_rate
                parachute_node_list = [
                    self.TimeNode(i * pcDt, [parachute], [])
                    for i in range(
                        math.ceil(t_init / pcDt), math.floor(t_end / pcDt) + 1
                    )
                ]
                self.list += parachute_node_list

        def sort(self):
            self.list.sort(key=(lambda node: node.t))

        def merge(self):
            # Initialize temporary list
            self.tmp_list = [self.list[0]]
            self.copy_list = self.list[1:]
            # Iterate through all other time nodes
            for node in self.copy_list:
                # If there is already another node with similar time: merge
                if abs(node.t - self.tmp_list[-1].t) < 1e-7:
                    self.tmp_list[-1].parachutes += node.parachutes
                    self.tmp_list[-1].callbacks += node.callbacks
                # Add new node to tmp list if there is none with the same time
                else:
                    self.tmp_list.append(node)
            # Save tmp list to permanent
            self.list = self.tmp_list

        def flush_after(self, index):
            del self.list[index + 1 :]

        class TimeNode:
            def __init__(self, t, parachutes, callbacks):
                self.t = t
                self.parachutes = parachutes
                self.callbacks = callbacks

            def __repr__(self):
                return (
                    "{Initial Time: "
                    + str(self.t)
                    + " | Parachutes: "
                    + str(len(self.parachutes))
                    + "}"
                )
