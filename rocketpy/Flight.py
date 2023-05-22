# -*- coding: utf-8 -*-

__author__ = (
    "Giovani Hidalgo Ceotto, Guilherme Fernandes Alves, João Lemes Gribel Soares"
)
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import math
import time
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import simplekml
from scipy import integrate

from .Function import Function, funcify_method
from .plots.flight_plots import _FlightPlots
from .prints.flight_prints import _FlightPrints
from .tools import Vector, Matrix

try:
    from functools import cached_property
except ImportError:
    from .tools import cached_property

try:
    from functools import cached_property
except ImportError:
    from .tools import cached_property


class Flight:
    """Keeps all flight information and has a method to simulate flight.

    Attributes
    ----------
        Other classes:
        Flight.env : Environment
            Environment object describing rail length, elevation, gravity and
            weather condition. See Environment class for more details.
        Flight.rocket : Rocket
            Rocket class describing rocket. See Rocket class for more
            details.
        Flight.parachutes : Parachutes
            Direct link to parachutes of the Rocket. See Rocket class
            for more details.
        Flight.frontalSurfaceWind : float
            Surface wind speed in m/s aligned with the launch rail.
        Flight.lateralSurfaceWind : float
            Surface wind speed in m/s perpendicular to launch rail.

        Helper classes:
        Flight.flightPhases : class
            Helper class to organize and manage different flight phases.
        Flight.timeNodes : class
            Helper class to manage time discretization during simulation.

        Helper functions:
        Flight.timeIterator : function
            Helper iterator function to generate time discretization points.

        Helper parameters:
        Flight.effective1RL : float
            Original rail length minus the distance measured from nozzle exit
            to the upper rail button. It assumes the nozzle to be aligned with
            the beginning of the rail.
        Flight.effective2RL : float
            Original rail length minus the distance measured from nozzle exit
            to the lower rail button. It assumes the nozzle to be aligned with
            the beginning of the rail.
        Flight.name: str
            Name of the flight.


        Numerical Integration settings:
        Flight.maxTime : int, float
            Maximum simulation time allowed. Refers to physical time
            being simulated, not time taken to run simulation.
        Flight.maxTimeStep : int, float
            Maximum time step to use during numerical integration in seconds.
        Flight.minTimeStep : int, float
            Minimum time step to use during numerical integration in seconds.
        Flight.rtol : int, float
            Maximum relative error tolerance to be tolerated in the
            numerical integration scheme.
        Flight.atol : int, float
            Maximum absolute error tolerance to be tolerated in the
            integration scheme.
        Flight.timeOvershoot : bool, optional
            If True, decouples ODE time step from parachute trigger functions
            sampling rate. The time steps can overshoot the necessary trigger
            function evaluation points and then interpolation is used to
            calculate them and feed the triggers. Can greatly improve run
            time in some cases.
        Flight.terminateOnApogee : bool
            Whether to terminate simulation when rocket reaches apogee.
        Flight.solver : scipy.integrate.LSODA
            Scipy LSODA integration scheme.

        State Space Vector Definition:
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

        Solution attributes:
        Flight.inclination : int, float
            Launch rail inclination angle relative to ground, given in degrees.
        Flight.heading : int, float
            Launch heading angle relative to north given in degrees.
        Flight.initialSolution : list
            List defines initial condition - [tInit, xInit,
            yInit, zInit, vxInit, vyInit, vzInit, e0Init, e1Init,
            e2Init, e3Init, w1Init, w2Init, w3Init]
        Flight.tInitial : int, float
            Initial simulation time in seconds. Usually 0.
        Flight.solution : list
            Solution array which keeps results from each numerical
            integration.
        Flight.t : float
            Current integration time.
        Flight.y : list
            Current integration state vector u.
        Flight.postProcessed : bool
            Defines if solution data has been post processed.

        Solution monitor attributes:
        Flight.initialSolution : list
            List defines initial condition - [tInit, xInit,
            yInit, zInit, vxInit, vyInit, vzInit, e0Init, e1Init,
            e2Init, e3Init, w1Init, w2Init, w3Init]
        Flight.outOfRailTime : int, float
            Time, in seconds, in which the rocket completely leaves the
            rail.
        Flight.outOfRailState : list
            State vector u corresponding to state when the rocket
            completely leaves the rail.
        Flight.outOfRailVelocity : int, float
            Velocity, in m/s, with which the rocket completely leaves the
            rail.
        Flight.apogeeState : array
            State vector u corresponding to state when the rocket's
            vertical velocity is zero in the apogee.
        Flight.apogeeTime : int, float
            Time, in seconds, in which the rocket's vertical velocity
            reaches zero in the apogee.
        Flight.apogeeX : int, float
            X coordinate (positive east) of the center of mass of the
            rocket when it reaches apogee.
        Flight.apogeeY : int, float
            Y coordinate (positive north) of the center of mass of the
            rocket when it reaches apogee.
        Flight.apogee : int, float
            Z coordinate, or altitude, of the center of mass of the
            rocket when it reaches apogee.
        Flight.xImpact : int, float
            X coordinate (positive east) of the center of mass of the
            rocket when it impacts ground.
        Flight.yImpact : int, float
            Y coordinate (positive east) of the center of mass of the
            rocket when it impacts ground.
        Flight.impactVelocity : int, float
            Velocity magnitude of the center of mass of the rocket when
            it impacts ground.
        Flight.impactState : array
            State vector u corresponding to state when the rocket
            impacts the ground.
        Flight.parachuteEvents : array
            List that stores parachute events triggered during flight.
        Flight.functionEvaluations : array
            List that stores number of derivative function evaluations
            during numerical integration in cumulative manner.
        Flight.functionEvaluationsPerTimeStep : array
            List that stores number of derivative function evaluations
            per time step during numerical integration.
        Flight.timeSteps : array
            List of time steps taking during numerical integration in
            seconds.
        Flight.flightPhases : Flight.FlightPhases
            Stores and manages flight phases.

        Solution Secondary Attributes:
        Atmospheric:
        Flight.windVelocityX : Function
            Wind velocity X (East) experienced by the rocket as a
            function of time. Can be called or accessed as array.
        Flight.windVelocityY : Function
            Wind velocity Y (North) experienced by the rocket as a
            function of time. Can be called or accessed as array.
        Flight.density : Function
            Air density experienced by the rocket as a function of
            time. Can be called or accessed as array.
        Flight.pressure : Function
            Air pressure experienced by the rocket as a function of
            time. Can be called or accessed as array.
        Flight.dynamicViscosity : Function
            Air dynamic viscosity experienced by the rocket as a function of
            time. Can be called or accessed as array.
        Flight.speedOfSound : Function
            Speed of Sound in air experienced by the rocket as a
            function of time. Can be called or accessed as array.

        Kinematics:
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
        Flight.maxSpeed : float
            Maximum velocity magnitude in m/s reached by the rocket
            relative to ground during flight.
        Flight.maxSpeedTime : float
            Time in seconds at which rocket reaches maximum velocity
            magnitude relative to ground.
        Flight.horizontalSpeed : Function
            Rocket's velocity magnitude in the horizontal (North-East)
            plane in m/s as a function of time. Can be called or
            accessed as array.
        Flight.Acceleration : Function
            Rocket acceleration magnitude in m/s² relative to ground as a
            function of time. Can be called or accessed as array.
        Flight.maxAcceleration : float
            Maximum acceleration magnitude in m/s² reached by the rocket
            relative to ground during flight.
        Flight.maxAccelerationTime : float
            Time in seconds at which rocket reaches maximum acceleration
            magnitude relative to ground.
        Flight.pathAngle : Function
            Rocket's flight path angle, or the angle that the
            rocket's velocity makes with the horizontal (North-East)
            plane. Measured in degrees and expressed as a function
            of time. Can be called or accessed as array.
        Flight.attitudeVectorX : Function
            Rocket's attitude vector, or the vector that points
            in the rocket's axis of symmetry, component in the X
            direction (East) as a function of time.
            Can be called or accessed as array.
        Flight.attitudeVectorY : Function
            Rocket's attitude vector, or the vector that points
            in the rocket's axis of symmetry, component in the Y
            direction (East) as a function of time.
            Can be called or accessed as array.
        Flight.attitudeVectorZ : Function
            Rocket's attitude vector, or the vector that points
            in the rocket's axis of symmetry, component in the Z
            direction (East) as a function of time.
            Can be called or accessed as array.
        Flight.attitudeAngle : Function
            Rocket's attitude angle, or the angle that the
            rocket's axis of symmetry makes with the horizontal (North-East)
            plane. Measured in degrees and expressed as a function
            of time. Can be called or accessed as array.
        Flight.lateralAttitudeAngle : Function
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

        Dynamics:
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
        Flight.aerodynamicLift : Function
            Resultant force perpendicular to rockets axis due to
            aerodynamic effects as a function of time. Units in N.
            Expressed as a function of time. Can be called or accessed
            as array.
        Flight.aerodynamicDrag : Function
            Resultant force aligned with the rockets axis due to
            aerodynamic effects as a function of time. Units in N.
            Expressed as a function of time. Can be called or accessed
            as array.
        Flight.aerodynamicBendingMoment : Function
            Resultant moment perpendicular to rocket's axis due to
            aerodynamic effects as a function of time. Units in N m.
            Expressed as a function of time. Can be called or accessed
            as array.
        Flight.aerodynamicSpinMoment : Function
            Resultant moment aligned with the rockets axis due to
            aerodynamic effects as a function of time. Units in N m.
            Expressed as a function of time. Can be called or accessed
            as array.
        Flight.railButton1NormalForce : Function
            Upper rail button normal force in N as a function
            of time. Can be called or accessed as array.
        Flight.maxRailButton1NormalForce : float
            Maximum upper rail button normal force experienced
            during rail flight phase in N.
        Flight.railButton1ShearForce : Function
            Upper rail button shear force in N as a function
            of time. Can be called or accessed as array.
        Flight.maxRailButton1ShearForce : float
            Maximum upper rail button shear force experienced
            during rail flight phase in N.
        Flight.railButton2NormalForce : Function
            Lower rail button normal force in N as a function
            of time. Can be called or accessed as array.
        Flight.maxRailButton2NormalForce : float
            Maximum lower rail button normal force experienced
            during rail flight phase in N.
        Flight.railButton2ShearForce : Function
            Lower rail button shear force in N as a function
            of time. Can be called or accessed as array.
        Flight.maxRailButton2ShearForce : float
            Maximum lower rail button shear force experienced
            during rail flight phase in N.
        Flight.rotationalEnergy : Function
            Rocket's rotational kinetic energy as a function of time.
            Units in J.
        Flight.translationalEnergy : Function
            Rocket's translational kinetic energy as a function of time.
            Units in J.
        Flight.kineticEnergy : Function
            Rocket's total kinetic energy as a function of time.
            Units in J.
        Flight.potentialEnergy : Function
            Rocket's gravitational potential energy as a function of
            time. Units in J.
        Flight.totalEnergy : Function
            Rocket's total mechanical energy as a function of time.
            Units in J.
        Flight.thrustPower : Function
            Rocket's engine thrust power output as a function
            of time in Watts. Can be called or accessed as array.
        Flight.dragPower : Function
            Aerodynamic drag power output as a function
            of time in Watts. Can be called or accessed as array.

        Stability and Control:
        Flight.attitudeFrequencyResponse : Function
            Fourier Frequency Analysis of the rocket's attitude angle.
            Expressed as the absolute vale of the magnitude as a function
            of frequency in Hz. Can be called or accessed as array.
        Flight.omega1FrequencyResponse : Function
            Fourier Frequency Analysis of the rocket's angular velocity omega 1.
            Expressed as the absolute vale of the magnitude as a function
            of frequency in Hz. Can be called or accessed as array.
        Flight.omega2FrequencyResponse : Function
            Fourier Frequency Analysis of the rocket's angular velocity omega 2.
            Expressed as the absolute vale of the magnitude as a function
            of frequency in Hz. Can be called or accessed as array.
        Flight.omega3FrequencyResponse : Function
            Fourier Frequency Analysis of the rocket's angular velocity omega 3.
            Expressed as the absolute vale of the magnitude as a function
            of frequency in Hz. Can be called or accessed as array.

        Flight.staticMargin : Function
            Rocket's static margin during flight in calibers.

        Fluid Mechanics:
        Flight.streamVelocityX : Function
            Freestream velocity x (East) component, in m/s, as a function of
            time. Can be called or accessed as array.
        Flight.streamVelocityY : Function
            Freestream velocity y (North) component, in m/s, as a function of
            time. Can be called or accessed as array.
        Flight.streamVelocityZ : Function
            Freestream velocity z (up) component, in m/s, as a function of
            time. Can be called or accessed as array.
        Flight.freestreamSpeed : Function
            Freestream velocity magnitude, in m/s, as a function of time.
            Can be called or accessed as array.
        Flight.apogeeFreestreamSpeed : float
            Freestream speed of the rocket at apogee in m/s.
        Flight.MachNumber : Function
            Rocket's Mach number defined as its freestream speed
            divided by the speed of sound at its altitude. Expressed
            as a function of time. Can be called or accessed as array.
        Flight.maxMachNumber : float
            Rocket's maximum Mach number experienced during flight.
        Flight.maxMachNumberTime : float
            Time at which the rocket experiences the maximum Mach number.
        Flight.ReynoldsNumber : Function
            Rocket's Reynolds number, using its diameter as reference
            length and freestreamSpeed as reference velocity. Expressed
            as a function of time. Can be called or accessed as array.
        Flight.maxReynoldsNumber : float
            Rocket's maximum Reynolds number experienced during flight.
        Flight.maxReynoldsNumberTime : float
            Time at which the rocket experiences the maximum Reynolds number.
        Flight.dynamicPressure : Function
            Dynamic pressure experienced by the rocket in Pa as a function
            of time, defined by 0.5*rho*V^2, where rho is air density and V
            is the freestream speed. Can be called or accessed as array.
        Flight.maxDynamicPressure : float
            Maximum dynamic pressure, in Pa, experienced by the rocket.
        Flight.maxDynamicPressureTime : float
            Time at which the rocket experiences maximum dynamic pressure.
        Flight.totalPressure : Function
            Total pressure experienced by the rocket in Pa as a function
            of time. Can be called or accessed as array.
        Flight.maxTotalPressure : float
            Maximum total pressure, in Pa, experienced by the rocket.
        Flight.maxTotalPressureTime : float
            Time at which the rocket experiences maximum total pressure.
        Flight.angleOfAttack : Function
            Rocket's angle of attack in degrees as a function of time.
            Defined as the minimum angle between the attitude vector and
            the freestream velocity vector. Can be called or accessed as
            array.
    """

    def __init__(
        self,
        rocket,
        environment,
        inclination=80,
        heading=90,
        initialSolution=None,
        terminateOnApogee=False,
        maxTime=600,
        maxTimeStep=np.inf,
        minTimeStep=0,
        rtol=1e-6,
        atol=6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3],
        timeOvershoot=True,
        verbose=False,
        name="Flight",
        uDot="standard",
    ):
        """Run a trajectory simulation.

        Parameters
        ----------
        rocket : Rocket
            Rocket to simulate. See help(Rocket) for more information.
        environment : Environment
            Environment to run simulation on. See help(Environment) for
            more information.
        inclination : int, float, optional
            Rail inclination angle relative to ground, given in degrees.
            Default is 80.
        heading : int, float, optional
            Heading angle relative to north given in degrees.
            Default is 90, which points in the x direction.
        initialSolution : array, Flight, optional
            Initial solution array to be used. Format is
            initialSolution = []
                self.tInitial,
                xInit, yInit, zInit,
                vxInit, vyInit, vzInit,
                e0Init, e1Init, e2Init, e3Init,
                w1Init, w2Init, w3Init
            ].
            If a Flight object is used, the last state vector will be used as
            initial solution. If None, the initial solution will start with
            all null values, except for the euler parameters which will be calculated based
            on given values of inclination and heading. Default is None.
        terminateOnApogee : boolean, optional
            Whether to terminate simulation when rocket reaches apogee.
            Default is False.
        maxTime : int, float, optional
            Maximum time in which to simulate trajectory in seconds.
            Using this without setting a maxTimeStep may cause unexpected
            errors. Default is 600.
        maxTimeStep : int, float, optional
            Maximum time step to use during integration in seconds.
            Default is 0.01.
        minTimeStep : int, float, optional
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
        timeOvershoot : bool, optional
            If True, decouples ODE time step from parachute trigger functions
            sampling rate. The time steps can overshoot the necessary trigger
            function evaluation points and then interpolation is used to
            calculate them and feed the triggers. Can greatly improve run
            time in some cases. Default is True.
        verbose : bool, optional
            If true, verbose mode is activated. Default is False.
        name : str, optional
            Name of the flight. Default is "Flight".

        Returns
        -------
        None
        """
        # Fetch helper classes and functions
        FlightPhases = self.FlightPhases
        TimeNodes = self.TimeNodes
        timeIterator = self.timeIterator

        # Save rocket, parachutes, environment, maximum simulation time
        # and termination events
        self.env = environment
        self.rocket = rocket
        self.parachutes = self.rocket.parachutes[:]
        self.inclination = inclination
        self.heading = heading
        self.maxTime = maxTime
        self.maxTimeStep = maxTimeStep
        self.minTimeStep = minTimeStep
        self.rtol = rtol
        self.atol = atol
        self.initialSolution = initialSolution
        self.timeOvershoot = timeOvershoot
        self.terminateOnApogee = terminateOnApogee
        self.name = name

        # Modifying Rail Length for a better out of rail condition
        upperButtonPosition, lowerButtonPosition = self.rocket.railButtons.position
        nozzlePosition = self.rocket.motorPosition
        self.effective1RL = self.env.rL - abs(nozzlePosition - upperButtonPosition)
        self.effective2RL = self.env.rL - abs(nozzlePosition - lowerButtonPosition)

        # Flight initialization
        self.__init_post_process_variables()
        self.__init_solution_monitors()

        # Initialize prints and plots objects
        self.prints = _FlightPrints(self)
        self.plots = _FlightPlots(self)

        # Initialize solver monitors
        self.__init_solver_monitors()

        # Create known flight phases
        self.flightPhases = FlightPhases()
        self.flightPhases.addPhase(self.tInitial, self.initialDerivative, clear=False)
        self.flightPhases.addPhase(self.maxTime)

        # Simulate flight
        for phase_index, phase in timeIterator(self.flightPhases):
            # print('\nCurrent Flight Phase List')
            # print(self.flightPhases)
            # print('\n\tCurrent Flight Phase')
            # print('\tIndex: ', phase_index, ' | Phase: ', phase)
            # Determine maximum time for this flight phase
            phase.timeBound = self.flightPhases[phase_index + 1].t

            # Evaluate callbacks
            for callback in phase.callbacks:
                callback(self)

            # Create solver for this flight phase
            self.functionEvaluations.append(0)
            phase.solver = integrate.LSODA(
                phase.derivative,
                t0=phase.t,
                y0=self.ySol,
                t_bound=phase.timeBound,
                min_step=self.minTimeStep,
                max_step=self.maxTimeStep,
                rtol=self.rtol,
                atol=self.atol,
            )
            # print('\n\tSolver Initialization Details')
            # print('\tInitial Time: ', phase.t)
            # print('\tInitial State: ', self.ySol)
            # print('\tTime Bound: ', phase.timeBound)
            # print('\tMin Step: ', self.minTimeStep)
            # print('\tMax Step: ', self.maxTimeStep)
            # print('\tTolerances: ', self.rtol, self.atol)

            # Initialize phase time nodes
            phase.timeNodes = TimeNodes()
            # Add first time node to permanent list
            phase.timeNodes.addNode(phase.t, [], [])
            # Add non-overshootable parachute time nodes
            if self.timeOvershoot is False:
                phase.timeNodes.addParachutes(self.parachutes, phase.t, phase.timeBound)
            # Add lst time node to permanent list
            phase.timeNodes.addNode(phase.timeBound, [], [])
            # Sort time nodes
            phase.timeNodes.sort()
            # Merge equal time nodes
            phase.timeNodes.merge()
            # Clear triggers from first time node if necessary
            if phase.clear:
                phase.timeNodes[0].parachutes = []
                phase.timeNodes[0].callbacks = []

            # print('\n\tPhase Time Nodes')
            # print('\tTime Nodes Length: ', str(len(phase.timeNodes)), ' | Time Nodes Preview: ', phase.timeNodes[0:3])

            # Iterate through time nodes
            for node_index, node in timeIterator(phase.timeNodes):
                # print('\n\t\tCurrent Time Node')
                # print('\t\tIndex: ', node_index, ' | Time Node: ', node)
                # Determine time bound for this time node
                node.timeBound = phase.timeNodes[node_index + 1].t
                phase.solver.t_bound = node.timeBound
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
                    pressure = self.env.pressure.getValueOpt(self.ySol[2])
                    parachute.cleanPressureSignal.append([node.t, pressure])
                    # Calculate and save noise
                    noise = parachute.noiseFunction()
                    parachute.noiseSignal.append([node.t, noise])
                    parachute.noisyPressureSignal.append([node.t, pressure + noise])
                    if parachute.trigger(pressure + noise, self.ySol):
                        # print('\nEVENT DETECTED')
                        # print('Parachute Triggered')
                        # print('Name: ', parachute.name, ' | Lag: ', parachute.lag)
                        # Remove parachute from flight parachutes
                        self.parachutes.remove(parachute)
                        # Create flight phase for time after detection and before inflation
                        # Must only be created if parachute has any lag
                        if parachute.lag != 0:
                            self.flightPhases.addPhase(
                                node.t,
                                phase.derivative,
                                clear=True,
                                index=phase_index + 1,
                            )
                        # Create flight phase for time after inflation
                        callbacks = [
                            lambda self, parachuteCdS=parachute.CdS: setattr(
                                self, "parachuteCdS", parachuteCdS
                            )
                        ]
                        self.flightPhases.addPhase(
                            node.t + parachute.lag,
                            self.uDotParachute,
                            callbacks,
                            clear=False,
                            index=phase_index + 2,
                        )
                        # Prepare to leave loops and start new flight phase
                        phase.timeNodes.flushAfter(node_index)
                        phase.timeNodes.addNode(self.t, [], [])
                        phase.solver.status = "finished"
                        # Save parachute event
                        self.parachuteEvents.append([self.t, parachute])

                # Step through simulation
                while phase.solver.status == "running":
                    # Step
                    phase.solver.step()
                    # Save step result
                    self.solution += [[phase.solver.t, *phase.solver.y]]
                    # Step step metrics
                    self.functionEvaluationsPerTimeStep.append(
                        phase.solver.nfev - self.functionEvaluations[-1]
                    )
                    self.functionEvaluations.append(phase.solver.nfev)
                    self.timeSteps.append(phase.solver.step_size)
                    # Update time and state
                    self.t = phase.solver.t
                    self.ySol = phase.solver.y
                    if verbose:
                        print(
                            "Current Simulation Time: {:3.4f} s".format(self.t),
                            end="\r",
                        )
                    # print('\n\t\t\tCurrent Step Details')
                    # print('\t\t\tIState: ', phase.solver._lsoda_solver._integrator.istate)
                    # print('\t\t\tTime: ', phase.solver.t)
                    # print('\t\t\tAltitude: ', phase.solver.y[2])
                    # print('\t\t\tEvals: ', self.functionEvaluationsPerTimeStep[-1])

                    # Check for first out of rail event
                    if len(self.outOfRailState) == 1 and (
                        self.ySol[0] ** 2
                        + self.ySol[1] ** 2
                        + (self.ySol[2] - self.env.elevation) ** 2
                        >= self.effective1RL**2
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
                            - self.effective1RL**2
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
                            - self.effective1RL**2
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
                        self.ySol = interpolator(self.t)
                        self.solution[-1] = [self.t, *self.ySol]
                        self.outOfRailTime = self.t
                        self.outOfRailTimeIndex = len(self.solution) - 1
                        self.outOfRailState = self.ySol
                        self.outOfRailVelocity = (
                            self.ySol[3] ** 2 + self.ySol[4] ** 2 + self.ySol[5] ** 2
                        ) ** (0.5)
                        # Create new flight phase
                        if uDot == "standard":
                            self.flightPhases.addPhase(
                                self.t, self.uDot, index=phase_index + 1
                            )
                        else:
                            self.flightPhases.addPhase(
                                self.t, self.uDotVariableMassNumpyBased, index=phase_index + 1
                            )
                        # Prepare to leave loops and start new flight phase
                        phase.timeNodes.flushAfter(node_index)
                        phase.timeNodes.addNode(self.t, [], [])
                        phase.solver.status = "finished"

                    # Check for apogee event
                    if len(self.apogeeState) == 1 and self.ySol[5] < 0:
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
                        self.apogeeState = interpolator(t_root)
                        # Store apogee data
                        self.apogeeTime = t_root
                        self.apogeeX = self.apogeeState[0]
                        self.apogeeY = self.apogeeState[1]
                        self.apogee = self.apogeeState[2]
                        if self.terminateOnApogee:
                            # print('Terminate on Apogee Activated!')
                            self.t = t_root
                            self.tFinal = self.t
                            self.state = self.apogeeState
                            # Roll back solution
                            self.solution[-1] = [self.t, *self.state]
                            # Set last flight phase
                            self.flightPhases.flushAfter(phase_index)
                            self.flightPhases.addPhase(self.t)
                            # Prepare to leave loops and start new flight phase
                            phase.timeNodes.flushAfter(node_index)
                            phase.timeNodes.addNode(self.t, [], [])
                            phase.solver.status = "finished"
                    # Check for impact event
                    if self.ySol[2] < self.env.elevation:
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
                        self.ySol = interpolator(self.t)
                        # Roll back solution
                        self.solution[-1] = [self.t, *self.ySol]
                        # Save impact state
                        self.impactState = self.ySol
                        self.xImpact = self.impactState[0]
                        self.yImpact = self.impactState[1]
                        self.zImpact = self.impactState[2]
                        self.impactVelocity = self.impactState[5]
                        self.tFinal = self.t
                        # Set last flight phase
                        self.flightPhases.flushAfter(phase_index)
                        self.flightPhases.addPhase(self.t)
                        # Prepare to leave loops and start new flight phase
                        phase.timeNodes.flushAfter(node_index)
                        phase.timeNodes.addNode(self.t, [], [])
                        phase.solver.status = "finished"

                    # List and feed overshootable time nodes
                    if self.timeOvershoot:
                        # Initialize phase overshootable time nodes
                        overshootableNodes = TimeNodes()
                        # Add overshootable parachute time nodes
                        overshootableNodes.addParachutes(
                            self.parachutes, self.solution[-2][0], self.t
                        )
                        # Add last time node (always skipped)
                        overshootableNodes.addNode(self.t, [], [])
                        if len(overshootableNodes) > 1:
                            # Sort overshootable time nodes
                            overshootableNodes.sort()
                            # Merge equal overshootable time nodes
                            overshootableNodes.merge()
                            # Clear if necessary
                            if overshootableNodes[0].t == phase.t and phase.clear:
                                overshootableNodes[0].parachutes = []
                                overshootableNodes[0].callbacks = []
                            # print('\n\t\t\t\tOvershootable Time Nodes')
                            # print('\t\t\t\tInterval: ', self.solution[-2][0], self.t)
                            # print('\t\t\t\tOvershootable Nodes Length: ', str(len(overshootableNodes)), ' | Overshootable Nodes: ', overshootableNodes)
                            # Feed overshootable time nodes trigger
                            interpolator = phase.solver.dense_output()
                            for overshootable_index, overshootableNode in timeIterator(
                                overshootableNodes
                            ):
                                # print('\n\t\t\t\tCurrent Overshootable Node')
                                # print('\t\t\t\tIndex: ', overshootable_index, ' | Overshootable Node: ', overshootableNode)
                                # Calculate state at node time
                                overshootableNode.y = interpolator(overshootableNode.t)
                                # Calculate and save pressure signal
                                pressure = self.env.pressure.getValueOpt(
                                    overshootableNode.y[2]
                                )
                                for parachute in overshootableNode.parachutes:
                                    # Save pressure signal
                                    parachute.cleanPressureSignal.append(
                                        [overshootableNode.t, pressure]
                                    )
                                    # Calculate and save noise
                                    noise = parachute.noiseFunction()
                                    parachute.noiseSignal.append(
                                        [overshootableNode.t, noise]
                                    )
                                    parachute.noisyPressureSignal.append(
                                        [overshootableNode.t, pressure + noise]
                                    )
                                    if parachute.trigger(
                                        pressure + noise, overshootableNode.y
                                    ):
                                        # print('\nEVENT DETECTED')
                                        # print('Parachute Triggered')
                                        # print('Name: ', parachute.name, ' | Lag: ', parachute.lag)
                                        # Remove parachute from flight parachutes
                                        self.parachutes.remove(parachute)
                                        # Create flight phase for time after detection and before inflation
                                        self.flightPhases.addPhase(
                                            overshootableNode.t,
                                            phase.derivative,
                                            clear=True,
                                            index=phase_index + 1,
                                        )
                                        # Create flight phase for time after inflation
                                        callbacks = [
                                            lambda self, parachuteCdS=parachute.CdS: setattr(
                                                self, "parachuteCdS", parachuteCdS
                                            )
                                        ]
                                        self.flightPhases.addPhase(
                                            overshootableNode.t + parachute.lag,
                                            self.uDotParachute,
                                            callbacks,
                                            clear=False,
                                            index=phase_index + 2,
                                        )
                                        # Rollback history
                                        self.t = overshootableNode.t
                                        self.ySol = overshootableNode.y
                                        self.solution[-1] = [
                                            overshootableNode.t,
                                            *overshootableNode.y,
                                        ]
                                        # Prepare to leave loops and start new flight phase
                                        overshootableNodes.flushAfter(
                                            overshootable_index
                                        )
                                        phase.timeNodes.flushAfter(node_index)
                                        phase.timeNodes.addNode(self.t, [], [])
                                        phase.solver.status = "finished"
                                        # Save parachute event
                                        self.parachuteEvents.append([self.t, parachute])

        self.tFinal = self.t
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
        self.outOfRailTime = 0
        self.outOfRailTimeIndex = 0
        self.outOfRailState = np.array([0])
        self.outOfRailVelocity = 0
        self.apogeeState = np.array([0])
        self.apogeeTime = 0
        self.apogeeX = 0
        self.apogeeY = 0
        self.apogee = 0
        self.xImpact = 0
        self.yImpact = 0
        self.impactVelocity = 0
        self.impactState = np.array([0])
        self.parachuteEvents = []
        self.postProcessed = False

        return None

    def __init_flight_state(self):
        """Initialize flight state variables."""
        if self.initialSolution is None:
            # Initialize time and state variables
            self.tInitial = 0
            xInit, yInit, zInit = 0, 0, self.env.elevation
            vxInit, vyInit, vzInit = 0, 0, 0
            w1Init, w2Init, w3Init = 0, 0, 0
            # Initialize attitude
            psiInit = -self.heading * (np.pi / 180)  # Precession / Heading Angle
            thetaInit = (self.inclination - 90) * (np.pi / 180)  # Nutation Angle
            e0Init = np.cos(psiInit / 2) * np.cos(thetaInit / 2)
            e1Init = np.cos(psiInit / 2) * np.sin(thetaInit / 2)
            e2Init = np.sin(psiInit / 2) * np.sin(thetaInit / 2)
            e3Init = np.sin(psiInit / 2) * np.cos(thetaInit / 2)
            # Store initial conditions
            self.initialSolution = [
                self.tInitial,
                xInit,
                yInit,
                zInit,
                vxInit,
                vyInit,
                vzInit,
                e0Init,
                e1Init,
                e2Init,
                e3Init,
                w1Init,
                w2Init,
                w3Init,
            ]
            # Set initial derivative for rail phase
            self.initialDerivative = self.uDotRail1
        elif isinstance(self.initialSolution, Flight):
            # Initialize time and state variables based on last solution of
            # previous flight
            self.initialSolution = self.initialSolution.solution[-1]
            # Set unused monitors
            self.outOfRailState = self.initialSolution[1:]
            self.outOfRailTime = self.initialSolution[0]
            # Set initial derivative for 6-DOF flight phase
            self.initialDerivative = self.uDot
        else:
            # Initial solution given, ignore rail phase
            # TODO: Check if rocket is actually out of rail. Otherwise, start at rail
            self.outOfRailState = self.initialSolution[1:]
            self.outOfRailTime = self.initialSolution[0]
            self.outOfRailTimeIndex = 0
            self.initialDerivative = self.uDot

    def __init_solver_monitors(self):
        # Initialize solver monitors
        self.functionEvaluations = []
        self.functionEvaluationsPerTimeStep = []
        self.timeSteps = []
        # Initialize solution state
        self.solution = []
        self.__init_flight_state()

        self.tInitial = self.initialSolution[0]
        self.solution.append(self.initialSolution)
        self.t = self.solution[-1][0]
        self.ySol = self.solution[-1][1:]

    @cached_property
    def effective1RL(self):
        # Modifying Rail Length for a better out of rail condition
        nozzle = (
            self.rocket.motorPosition - self.rocket.centerOfDryMassPosition
        ) * self.rocket._csys  # Kinda works for single nozzle
        try:
            upperRButton = max(self.rocket.railButtons[0])
        except AttributeError:  # If there is no rail button
            upperRButton = nozzle
        effective1RL = self.env.rL - abs(nozzle - upperRButton)

        return effective1RL

    @cached_property
    def effective2RL(self):
        # Modifying Rail Length for a better out of rail condition
        nozzle = (
            self.rocket.motorPosition - self.rocket.centerOfDryMassPosition
        ) * self.rocket._csys
        try:
            lowerRButton = min(self.rocket.railButtons[0])
        except AttributeError:
            lowerRButton = nozzle
        effective2RL = self.env.rL - abs(nozzle - lowerRButton)
        return effective2RL

    @cached_property
    def frontalSurfaceWind(self):
        # Surface wind magnitude in the frontal direction at the rail's elevation
        windU = self.env.windVelocityX(self.env.elevation)
        windV = self.env.windVelocityY(self.env.elevation)
        headingRad = self.heading * np.pi / 180
        frontalSurfaceWind = windU * np.sin(headingRad) + windV * np.cos(headingRad)
        return frontalSurfaceWind

    @cached_property
    def lateralSurfaceWind(self):
        # Surface wind magnitude in the lateral direction at the rail's elevation
        windU = self.env.windVelocityX(self.env.elevation)
        windV = self.env.windVelocityY(self.env.elevation)
        headingRad = self.heading * np.pi / 180
        lateralSurfaceWind = -windU * np.cos(headingRad) + windV * np.sin(headingRad)
        return lateralSurfaceWind

    def uDotRail1(self, t, u, postProcessing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying in 1 DOF motion in the rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        postProcessing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.attackAngle. Default is False.

        Return
        ------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].

        """
        # Check if post processing mode is on
        if postProcessing:
            # Use uDot post processing code
            return self.uDot(t, u, True)

        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Retrieve important quantities
        # Mass
        M = self.rocket.totalMass.getValueOpt(t)

        # Get freestream speed
        freestreamSpeed = (
            (self.env.windVelocityX.getValueOpt(z) - vx) ** 2
            + (self.env.windVelocityY.getValueOpt(z) - vy) ** 2
            + (vz) ** 2
        ) ** 0.5
        freestreamMach = freestreamSpeed / self.env.speedOfSound.getValueOpt(z)
        dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)

        # Calculate Forces
        Thrust = self.rocket.motor.thrust.getValueOpt(t)
        rho = self.env.density.getValueOpt(z)
        R3 = -0.5 * rho * (freestreamSpeed**2) * self.rocket.area * (dragCoeff)

        # Calculate Linear acceleration
        a3 = (R3 + Thrust) / M - (e0**2 - e1**2 - e2**2 + e3**2) * self.env.g
        if a3 > 0:
            ax = 2 * (e1 * e3 + e0 * e2) * a3
            ay = 2 * (e2 * e3 - e0 * e1) * a3
            az = (1 - 2 * (e1**2 + e2**2)) * a3
        else:
            ax, ay, az = 0, 0, 0

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def uDotRail2(self, t, u, postProcessing=False):
        """[Still not implemented] Calculates derivative of u state vector with
        respect to time when rocket is flying in 3 DOF motion in the rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        postProcessing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.attackAngle, by default False

        Returns
        -------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].
        """
        # Hey! We will finish this function later, now we just can use uDot
        return self.uDot(t, u, postProcessing=postProcessing)

    def uDot(self, t, u, postProcessing=False):
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
        postProcessing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.attackAngle, by default False

        Returns
        -------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].
        """

        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        # Determine lift force and moment
        R1, R2 = 0, 0
        M1, M2, M3 = 0, 0, 0
        # Determine current behavior
        if t < self.rocket.motor.burnOutTime:
            # Motor burning
            # Retrieve important motor quantities
            # Inertias
            Tz = self.rocket.motor.I_33.getValueOpt(t)
            Ti = self.rocket.motor.I_11.getValueOpt(t)
            TzDot = self.rocket.motor.I_33.differentiate(t, dx=1e-6)
            TiDot = self.rocket.motor.I_11.differentiate(t, dx=1e-6)
            # Mass
            MtDot = self.rocket.motor.massDot.getValueOpt(t)
            Mt = self.rocket.motor.mass.getValueOpt(t)
            # Thrust
            Thrust = self.rocket.motor.thrust.getValueOpt(t)
            # Off center moment
            M1 += self.rocket.thrustEccentricityX * Thrust
            M2 -= self.rocket.thrustEccentricityY * Thrust
        else:
            # Motor stopped
            # Retrieve important motor quantities
            # Inertias
            Tz, Ti, TzDot, TiDot = 0, 0, 0, 0
            # Mass
            MtDot, Mt = 0, 0
            # Thrust
            Thrust = 0

        # Retrieve important quantities
        # Inertias
        Rz = self.rocket.dry_I_33
        Ri = self.rocket.dry_I_11
        # Mass
        Mr = self.rocket.mass
        M = Mt + Mr
        mu = (Mt * Mr) / (Mt + Mr)
        # Geometry
        # b = -self.rocket.distanceRocketPropellant
        b = (
            -(
                self.rocket.centerOfPropellantPosition(0)
                - self.rocket.centerOfDryMassPosition
            )
            * self.rocket._csys
        )
        # c = -self.rocket.distanceRocketNozzle
        c = (
            -(self.rocket.motorPosition - self.rocket.centerOfDryMassPosition)
            * self.rocket._csys
        )
        a = b * Mt / M
        rN = self.rocket.motor.nozzleRadius
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
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = (
            (windVelocityX - vx) ** 2 + (windVelocityY - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        freestreamMach = freestreamSpeed / self.env.speedOfSound.getValueOpt(z)

        # Determine aerodynamics forces
        # Determine Drag Force
        if t < self.rocket.motor.burnOutTime:
            dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)
        else:
            dragCoeff = self.rocket.powerOffDrag.getValueOpt(freestreamMach)
        rho = self.env.density.getValueOpt(z)
        R3 = -0.5 * rho * (freestreamSpeed**2) * self.rocket.area * (dragCoeff)
        # Off center moment
        M1 += self.rocket.cpEccentricityY * R3
        M2 -= self.rocket.cpEccentricityX * R3
        # Get rocket velocity in body frame
        vxB = a11 * vx + a21 * vy + a31 * vz
        vyB = a12 * vx + a22 * vy + a32 * vz
        vzB = a13 * vx + a23 * vy + a33 * vz
        # Calculate lift and moment for each component of the rocket
        for aeroSurface, position in self.rocket.aerodynamicSurfaces:
            compCp = (
                position - self.rocket.centerOfDryMassPosition
            ) * self.rocket._csys - aeroSurface.cpz
            surfaceRadius = aeroSurface.rocketRadius
            referenceArea = np.pi * surfaceRadius**2
            # Component absolute velocity in body frame
            compVxB = vxB + compCp * omega2
            compVyB = vyB - compCp * omega1
            compVzB = vzB
            # Wind velocity at component
            compZ = z + compCp
            compWindVx = self.env.windVelocityX.getValueOpt(compZ)
            compWindVy = self.env.windVelocityY.getValueOpt(compZ)
            # Component freestream velocity in body frame
            compWindVxB = a11 * compWindVx + a21 * compWindVy
            compWindVyB = a12 * compWindVx + a22 * compWindVy
            compWindVzB = a13 * compWindVx + a23 * compWindVy
            compStreamVxB = compWindVxB - compVxB
            compStreamVyB = compWindVyB - compVyB
            compStreamVzB = compWindVzB - compVzB
            compStreamSpeed = (
                compStreamVxB**2 + compStreamVyB**2 + compStreamVzB**2
            ) ** 0.5
            # Component attack angle and lift force
            compAttackAngle = 0
            compLift, compLiftXB, compLiftYB = 0, 0, 0
            if compStreamVxB**2 + compStreamVyB**2 != 0:
                # Normalize component stream velocity in body frame
                compStreamVzBn = compStreamVzB / compStreamSpeed
                if -1 * compStreamVzBn < 1:
                    compAttackAngle = np.arccos(-compStreamVzBn)
                    cLift = aeroSurface.cl(compAttackAngle, freestreamMach)
                    # Component lift force magnitude
                    compLift = (
                        0.5 * rho * (compStreamSpeed**2) * referenceArea * cLift
                    )
                    # Component lift force components
                    liftDirNorm = (compStreamVxB**2 + compStreamVyB**2) ** 0.5
                    compLiftXB = compLift * (compStreamVxB / liftDirNorm)
                    compLiftYB = compLift * (compStreamVyB / liftDirNorm)
                    # Add to total lift force
                    R1 += compLiftXB
                    R2 += compLiftYB
                    # Add to total moment
                    M1 -= (compCp + a) * compLiftYB
                    M2 += (compCp + a) * compLiftXB
            # Calculates Roll Moment
            try:
                Clfdelta, Cldomega, cantAngleRad = aeroSurface.rollParameters
                M3f = (
                    (1 / 2 * rho * freestreamSpeed**2)
                    * referenceArea
                    * 2
                    * surfaceRadius
                    * Clfdelta(freestreamMach)
                    * cantAngleRad
                )
                M3d = (
                    (1 / 2 * rho * freestreamSpeed)
                    * referenceArea
                    * (2 * surfaceRadius) ** 2
                    * Cldomega(freestreamMach)
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
                    (TiDot + MtDot * (Mr - 1) * (b / M) ** 2)
                    - MtDot * ((rN / 2) ** 2 + (c - b * mu / Mr) ** 2)
                )
            )
        ) / (Ri + Ti + mu * b**2)
        alpha2 = (
            M2
            - (
                omega1 * omega3 * (Ri + Ti + mu * b**2 - Rz - Tz)
                + omega2
                * (
                    (TiDot + MtDot * (Mr - 1) * (b / M) ** 2)
                    - MtDot * ((rN / 2) ** 2 + (c - b * mu / Mr) ** 2)
                )
            )
        ) / (Ri + Ti + mu * b**2)
        alpha3 = (M3 - omega3 * (TzDot - MtDot * (rN**2) / 2)) / (Rz + Tz)
        # Euler parameters derivative
        e0Dot = 0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3)
        e1Dot = 0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3)
        e2Dot = 0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3)
        e3Dot = 0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2)

        # Linear acceleration
        L = [
            (R1 - b * Mt * (omega2**2 + omega3**2) - 2 * c * MtDot * omega2) / M,
            (R2 + b * Mt * (alpha3 + omega1 * omega2) + 2 * c * MtDot * omega1) / M,
            (R3 - b * Mt * (alpha2 - omega1 * omega3) + Thrust) / M,
        ]
        ax, ay, az = np.dot(K, L)
        az -= self.env.g  # Include gravity

        # Create uDot
        uDot = [
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            e0Dot,
            e1Dot,
            e2Dot,
            e3Dot,
            alpha1,
            alpha2,
            alpha3,
        ]

        if postProcessing:
            # Dynamics variables
            self.R1_list.append([t, R1])
            self.R2_list.append([t, R2])
            self.R3_list.append([t, R3])
            self.M1_list.append([t, M1])
            self.M2_list.append([t, M2])
            self.M3_list.append([t, M3])
            # Atmospheric Conditions
            self.windVelocityX_list.append([t, self.env.windVelocityX.getValueOpt(z)])
            self.windVelocityY_list.append([t, self.env.windVelocityY.getValueOpt(z)])
            self.density_list.append([t, self.env.density.getValueOpt(z)])
            self.dynamicViscosity_list.append(
                [t, self.env.dynamicViscosity.getValueOpt(z)]
            )
            self.pressure_list.append([t, self.env.pressure.getValueOpt(z)])
            self.speedOfSound_list.append([t, self.env.speedOfSound.getValueOpt(z)])

        return uDot

    def __transformationMatrices(self, e):
        """Returns the transformation matrices to/from body frame from/to 
        inertial frame, based on Euler parameters/quaternions.
        
        Parameters
        ----------
        e : array-like
            Array of 4 Euler parameters/quaternions [e0, e1, e2, e3].

        Returns
        -------
        k : 3x3 np.array
            Transformation matrix from body frame to inertial frame.
        kt : 3x3 np.array
            Transformation matrix from inertial frame to body frame.
        """
        e0, e1, e2, e3 = e
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
        K = Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        # Transformation matrix: (XYZ) -> (123) or K transpose
        Kt = Matrix([[a11, a21, a31], [a12, a22, a32], [a13, a23, a33]])

        return K, Kt

    def __computeDragForce(self, z, v):
        """Returns the drag force acting on the rocket given an altitude z and
        the rocket velocity vector v.
        
        Parameters
        ----------
        z : float
            Rocket altitude above sea level in meters.
        v : np.array
            Velocity vector vx, vy, vz in the inertial reference frame.

            
        Returns
        -------
        float
            Drag force in N, which acts in the direction of the rocket's
            axis of symmetry.
        """
        rho = self.env.density.getValueOpt(z)
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        wind_velocity = Vector([windVelocityX, windVelocityY, 0])
        freestreamSpeed = abs((wind_velocity - v))
        freestreamMach = freestreamSpeed / self.env.speedOfSound.getValueOpt(z)
        dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)
        return -0.5 * rho * (freestreamSpeed**2) * self.rocket.area * (dragCoeff)

    def __skew_symmetric_matrix(self, u):
        """Returns the skew symmetric matrix representation of the cross
        product of u with any vector.
        
        Parameters
        ----------
        u : 3x1 array
            Array to be transformed into screw symmetric matrix
        
        Returns
        -------
        3x3 np.array
            Screw symmetric matrix of u
        """
        return np.array([
            [    0, -u[2],  u[1]],
            [ u[2],     0, -u[0]],
            [-u[1],  u[0],     0]
        ])

    def uDotVariableMassNumpyBased(self, t, u, postProcessing=False):
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
        postProcessing : bool, optional
            If True, adds flight data information directly to self variables
            such as self.attackAngle, by default False.

        Returns
        -------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].
        """
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        
        # Create necessary vectors
        r = np.array([x, y, z]) # CDM position vector
        v = np.array([vx, vy, vz]) # CDM velocity vector
        e = np.array([e0, e1, e2, e3]) # Euler parameters/quaternions
        w = np.array([omega1, omega2, omega3]) # Angular velocity vector

        # Retrieve necessary quantities
        rho = self.env.density.getValueOpt(z)
        total_mass = self.rocket.totalMass.getValueOpt(t)
        total_mass_dot = self.rocket.totalMass.differentiate(t)
        total_mass_ddot = self.rocket.totalMass.differentiate(t, order=2)
        ## CM position vector and time derivatives relative to CDM in body frame
        r_CM_z = -1 * (
            (
                self.rocket.centerOfPropellantPosition
                - self.rocket.centerOfDryMassPosition
            )
            * self.rocket._csys
        ) * self.rocket.motor.mass / total_mass
        r_CM = np.array([0, 0, r_CM_z.getValueOpt(t)])
        r_CM_dot = np.array([0, 0, r_CM_z.differentiate(t)])
        r_CM_ddot = np.array([0, 0, r_CM_z.differentiate(t, order=2)])
        ## Nozzle gyration tensor
        r_NOZ = (
            -(self.rocket.motorPosition - self.rocket.centerOfDryMassPosition)
            * self.rocket._csys
        )
        S_noz_33 = 0.5 * self.rocket.motor.nozzleRadius**2
        S_noz_11 = 0.5 * S_noz_33 + 0.25 * r_NOZ**2
        S_noz_22 = S_noz_11
        S_noz_12 = 0
        S_noz_13 = 0
        S_noz_23 = 0
        S_nozzle = np.array([
            [S_noz_11, S_noz_12, S_noz_13],
            [S_noz_12, S_noz_22, S_noz_23],
            [S_noz_13, S_noz_23, S_noz_33],
        ])
        ## Inertia tensor
        I_11 = self.rocket.I_11.getValueOpt(t)
        I_12 = self.rocket.I_12.getValueOpt(t)
        I_13 = self.rocket.I_13.getValueOpt(t)
        I_22 = self.rocket.I_22.getValueOpt(t)
        I_23 = self.rocket.I_23.getValueOpt(t)
        I_33 = self.rocket.I_33.getValueOpt(t)
        I = np.array([
            [I_11, I_12, I_13],
            [I_12, I_22, I_23],
            [I_13, I_23, I_33],
        ])
        ## Inertia tensor time derivative in the body frame
        I_11_dot = self.rocket.I_11.differentiate(t)
        I_12_dot = self.rocket.I_12.differentiate(t)
        I_13_dot = self.rocket.I_13.differentiate(t)
        I_22_dot = self.rocket.I_22.differentiate(t)
        I_23_dot = self.rocket.I_23.differentiate(t)
        I_33_dot = self.rocket.I_33.differentiate(t)
        I_dot = np.array([
            [I_11_dot, I_12_dot, I_13_dot],
            [I_12_dot, I_22_dot, I_23_dot],
            [I_13_dot, I_23_dot, I_33_dot],
        ])
        ## Inertia tensor relative to CM
        r_CM_X = self.__skew_symmetric_matrix(r_CM)
        H = total_mass * np.dot(r_CM_X, -r_CM_X)
        I_CM = I - H

        # Prepare transformation matrices
        K, Kt = self.__transformationMatrices(e)

        # Compute aerodynamic forces and moments
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

        ## Drag force
        R3 += self.__computeDragForce(z, v)
        
        ## Off center moment
        M1 += self.rocket.cpEccentricityY * R3
        M2 -= self.rocket.cpEccentricityX * R3

        # Get rocket velocity in body frame
        vB = np.dot(Kt, v)
        # Calculate lift and moment for each component of the rocket
        for aeroSurface, position in self.rocket.aerodynamicSurfaces:
            compCpz = (
                position - self.rocket.centerOfDryMassPosition
            ) * self.rocket._csys - aeroSurface.cpz
            compCp = np.array([0, 0, compCpz])
            surfaceRadius = aeroSurface.rocketRadius
            referenceArea = np.pi * surfaceRadius**2
            # Component absolute velocity in body frame
            compVB = vB + np.cross(w, compCp)
            # Wind velocity at component altitude
            compZ = z + np.dot(K, compCp)[2]
            compWindVx = self.env.windVelocityX.getValueOpt(compZ)
            compWindVy = self.env.windVelocityY.getValueOpt(compZ)
            # Component freestream velocity in body frame
            compWindVB = np.dot(Kt, np.array([compWindVx, compWindVy, 0]))
            compStreamVelocity = compWindVB - compVB
            compStreamVxB, compStreamVyB, compStreamVzB = compStreamVelocity
            compStreamSpeed = np.linalg.norm(compStreamVelocity)
            compStreamMach = compStreamSpeed/self.env.speedOfSound.getValueOpt(z)
            # Component attack angle and lift force
            compAttackAngle = 0
            compLift, compLiftXB, compLiftYB = 0, 0, 0
            if compStreamVxB**2 + compStreamVyB**2 != 0:
                # Normalize component stream velocity in body frame
                compStreamVzBn = compStreamVzB / compStreamSpeed
                if -1 * compStreamVzBn < 1:
                    compAttackAngle = np.arccos(-compStreamVzBn)
                    cLift = aeroSurface.cl(compAttackAngle, compStreamMach)
                    # Component lift force magnitude
                    compLift = (
                        0.5 * rho * (compStreamSpeed**2) * referenceArea * cLift
                    )
                    # Component lift force components
                    liftDirNorm = (compStreamVxB**2 + compStreamVyB**2) ** 0.5
                    compLiftXB = compLift * (compStreamVxB / liftDirNorm)
                    compLiftYB = compLift * (compStreamVyB / liftDirNorm)
                    # Add to total lift force
                    R1 += compLiftXB
                    R2 += compLiftYB
                    # Add to total moment
                    M1 -= (compCpz + r_CM_z.getValueOpt(t)) * compLiftYB
                    M2 += (compCpz + r_CM_z.getValueOpt(t)) * compLiftXB
            # Calculates Roll Moment
            try:
                Clfdelta, Cldomega, cantAngleRad = aeroSurface.rollParameters
                M3f = (
                    (1 / 2 * rho * compStreamSpeed**2)
                    * referenceArea
                    * 2
                    * surfaceRadius
                    * Clfdelta(compStreamMach)
                    * cantAngleRad
                )
                M3d = (
                    (1 / 2 * rho * compStreamSpeed)
                    * referenceArea
                    * (2 * surfaceRadius) ** 2
                    * Cldomega(compStreamMach)
                    * omega3
                    / 2
                )
                M3 += M3f - M3d
            except AttributeError:
                pass

        weightB = np.dot(Kt, np.array([0, 0, -total_mass * self.env.g]))
        T00 = total_mass * r_CM
        T03 = 2*total_mass_dot*(np.array([0, 0, r_NOZ]) - r_CM) - 2*total_mass*r_CM_dot
        T04 = (
            self.rocket.motor.thrust(t) * np.array([0, 0, 1])
            - total_mass * r_CM_ddot
            - 2 * total_mass_dot * r_CM_dot
            + total_mass_ddot * (np.array([0, 0, r_NOZ]) - r_CM)
        )
        T05 = total_mass_dot * S_nozzle - I_dot

        T20 = (
            np.cross(np.cross(w, T00), w) +
            np.cross(w, T03) +
            T04 +
            weightB +
            np.array([R1, R2, R3])
        )

        T21 = (
            np.cross(np.dot(I, w), w) +
            np.dot(T05, w) +
            np.cross(weightB, r_CM) +
            np.array([M1, M2, M3])
        )

        # Angular velocity derivative
        w_dot = np.dot(np.linalg.inv(I_CM), (T21 + np.cross(T20, r_CM)).T).flatten()

        # Velocity vector derivative
        v_dot = np.dot(K, T20/total_mass - np.cross(r_CM, w_dot))

        # Euler parameters derivative
        e_dot = [
            0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3),
            0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3),
            0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3),
            0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2),
        ]

        # Position vector derivative
        r_dot = [vx, vy, vz]

        # Create uDot
        uDot = [*r_dot, *v_dot, *e_dot, *w_dot]

        if postProcessing:
            # Dynamics variables
            self.R1_list.append([t, R1])
            self.R2_list.append([t, R2])
            self.R3_list.append([t, R3])
            self.M1_list.append([t, M1])
            self.M2_list.append([t, M2])
            self.M3_list.append([t, M3])
            # Atmospheric Conditions
            self.windVelocityX_list.append([t, self.env.windVelocityX.getValueOpt(z)])
            self.windVelocityY_list.append([t, self.env.windVelocityY.getValueOpt(z)])
            self.density_list.append([t, self.env.density.getValueOpt(z)])
            self.dynamicViscosity_list.append(
                [t, self.env.dynamicViscosity.getValueOpt(z)]
            )
            self.pressure_list.append([t, self.env.pressure.getValueOpt(z)])
            self.speedOfSound_list.append([t, self.env.speedOfSound.getValueOpt(z)])

        return uDot

    def uDotVariableMassNumpyFree(self, t, u, postProcessing=False):
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
        postProcessing : bool, optional
            If True, adds flight data information directly to self variables
            such as self.attackAngle, by default False.

        Returns
        -------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].
        """
        # Retrieve integration data
        x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        
        # Create necessary vectors
        r = Vector([x, y, z]) # CDM position vector
        v = Vector([vx, vy, vz]) # CDM velocity vector
        e = Vector([e0, e1, e2, e3]) # Euler parameters/quaternions
        w = Vector([omega1, omega2, omega3]) # Angular velocity vector

        # Retrieve necessary quantities
        rho = self.env.density.getValueOpt(z)
        total_mass = self.rocket.totalMass.getValueOpt(t)
        total_mass_dot = self.rocket.totalMass.differentiate(t)
        total_mass_ddot = self.rocket.totalMass.differentiate(t, order=2)
        ## CM position vector and time derivatives relative to CDM in body frame
        r_CM_z = -1 * (
            (
                self.rocket.centerOfPropellantPosition
                - self.rocket.centerOfDryMassPosition
            )
            * self.rocket._csys
        ) * self.rocket.motor.mass / total_mass
        r_CM = Vector([0, 0, r_CM_z.getValueOpt(t)])
        r_CM_dot = Vector([0, 0, r_CM_z.differentiate(t)])
        r_CM_ddot = Vector([0, 0, r_CM_z.differentiate(t, order=2)])
        ## Nozzle gyration tensor
        r_NOZ = (
            -(self.rocket.motorPosition - self.rocket.centerOfDryMassPosition)
            * self.rocket._csys
        )
        S_noz_33 = 0.5 * self.rocket.motor.nozzleRadius**2
        S_noz_11 = 0.5 * S_noz_33 + 0.25 * r_NOZ**2
        S_noz_22 = S_noz_11
        S_noz_12 = 0
        S_noz_13 = 0
        S_noz_23 = 0
        S_nozzle = Matrix([
            [S_noz_11, S_noz_12, S_noz_13],
            [S_noz_12, S_noz_22, S_noz_23],
            [S_noz_13, S_noz_23, S_noz_33],
        ])
        ## Inertia tensor
        I_11 = self.rocket.I_11.getValueOpt(t)
        I_12 = self.rocket.I_12.getValueOpt(t)
        I_13 = self.rocket.I_13.getValueOpt(t)
        I_22 = self.rocket.I_22.getValueOpt(t)
        I_23 = self.rocket.I_23.getValueOpt(t)
        I_33 = self.rocket.I_33.getValueOpt(t)
        I = Matrix([
            [I_11, I_12, I_13],
            [I_12, I_22, I_23],
            [I_13, I_23, I_33],
        ])
        ## Inertia tensor time derivative in the body frame
        I_11_dot = self.rocket.I_11.differentiate(t)
        I_12_dot = self.rocket.I_12.differentiate(t)
        I_13_dot = self.rocket.I_13.differentiate(t)
        I_22_dot = self.rocket.I_22.differentiate(t)
        I_23_dot = self.rocket.I_23.differentiate(t)
        I_33_dot = self.rocket.I_33.differentiate(t)
        I_dot = Matrix([
            [I_11_dot, I_12_dot, I_13_dot],
            [I_12_dot, I_22_dot, I_23_dot],
            [I_13_dot, I_23_dot, I_33_dot],
        ])
        ## Inertia tensor relative to CM
        r_CM_X = self.cross_matrix(r_CM)
        H = total_mass * (r_CM_X @ -r_CM_X)
        I_CM = I - H

        # Prepare transformation matrices
        K, Kt = self.__transformationMatrices(e)

        # Compute aerodynamic forces and moments
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

        ## Drag force
        R3 += self.__computeDragForce(z, v)
        
        ## Off center moment
        M1 += self.rocket.cpEccentricityY * R3
        M2 -= self.rocket.cpEccentricityX * R3

        # Get rocket velocity in body frame
        vB = Kt @ v
        # Calculate lift and moment for each component of the rocket
        for aeroSurface, position in self.rocket.aerodynamicSurfaces:
            compCpz = (
                position - self.rocket.centerOfDryMassPosition
            ) * self.rocket._csys - aeroSurface.cpz
            compCp = Vector([0, 0, compCpz])
            surfaceRadius = aeroSurface.rocketRadius
            referenceArea = np.pi * surfaceRadius**2
            # Component absolute velocity in body frame
            compVB = vB + w * compCp
            # Wind velocity at component altitude
            compZ = z + (K @ compCp).z
            compWindVx = self.env.windVelocityX.getValueOpt(compZ)
            compWindVy = self.env.windVelocityY.getValueOpt(compZ)
            # Component freestream velocity in body frame
            compWindVB = Kt @ [compWindVx, compWindVy, 0]
            compStreamVelocity = compWindVB - compVB
            compStreamVxB, compStreamVyB, compStreamVzB = compStreamVelocity
            compStreamSpeed = abs(compStreamVelocity)
            compStreamMach = compStreamSpeed/self.env.speedOfSound.getValueOpt(z)
            # Component attack angle and lift force
            compAttackAngle = 0
            compLift, compLiftXB, compLiftYB = 0, 0, 0
            if compStreamVxB**2 + compStreamVyB**2 != 0:
                # Normalize component stream velocity in body frame
                compStreamVzBn = compStreamVzB / compStreamSpeed
                if -1 * compStreamVzBn < 1:
                    compAttackAngle = np.arccos(-compStreamVzBn)
                    cLift = aeroSurface.cl(compAttackAngle, compStreamMach)
                    # Component lift force magnitude
                    compLift = (
                        0.5 * rho * (compStreamSpeed**2) * referenceArea * cLift
                    )
                    # Component lift force components
                    liftDirNorm = (compStreamVxB**2 + compStreamVyB**2) ** 0.5
                    compLiftXB = compLift * (compStreamVxB / liftDirNorm)
                    compLiftYB = compLift * (compStreamVyB / liftDirNorm)
                    # Add to total lift force
                    R1 += compLiftXB
                    R2 += compLiftYB
                    # Add to total moment
                    M1 -= (compCpz + r_CM_z.getValueOpt(t)) * compLiftYB
                    M2 += (compCpz + r_CM_z.getValueOpt(t)) * compLiftXB
            # Calculates Roll Moment
            try:
                Clfdelta, Cldomega, cantAngleRad = aeroSurface.rollParameters
                M3f = (
                    (1 / 2 * rho * compStreamSpeed**2)
                    * referenceArea
                    * 2
                    * surfaceRadius
                    * Clfdelta(compStreamMach)
                    * cantAngleRad
                )
                M3d = (
                    (1 / 2 * rho * compStreamSpeed)
                    * referenceArea
                    * (2 * surfaceRadius) ** 2
                    * Cldomega(compStreamMach)
                    * omega3
                    / 2
                )
                M3 += M3f - M3d
            except AttributeError:
                pass

        weightB = Kt @ [0, 0, -total_mass * self.env.g]
        T00 = total_mass * r_CM
        T03 = 2*total_mass_dot*(Vector([0, 0, r_NOZ]) - r_CM) - 2*total_mass*r_CM_dot
        T04 = (
            self.rocket.motor.thrust(t) * Vector([0, 0, 1])
            - total_mass * r_CM_ddot
            - 2 * total_mass_dot * r_CM_dot
            + total_mass_ddot * (Vector([0, 0, r_NOZ]) - r_CM)
        )
        T05 = total_mass_dot * S_nozzle - I_dot

        T20 = (
            (w * T00) * w +
            w * T03 +
            T04 +
            weightB +
            Vector([R1, R2, R3])
        )

        T21 = (
            I @ w * w +
            T05 @ w +
            weightB * r_CM +
            Vector([M1, M2, M3])
        )

        # Angular velocity derivative
        w_dot = I_CM.inverse @ ((T21 + T20) * r_CM)

        # Velocity vector derivative
        v_dot = K * (T20/total_mass - (r_CM * w_dot))

        # Euler parameters derivative
        e_dot = [
            0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3),
            0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3),
            0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3),
            0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2),
        ]

        # Position vector derivative
        r_dot = [vx, vy, vz]

        # Create uDot
        uDot = [*r_dot, *v_dot, *e_dot, *w_dot]

        if postProcessing:
            # Dynamics variables
            self.R1_list.append([t, R1])
            self.R2_list.append([t, R2])
            self.R3_list.append([t, R3])
            self.M1_list.append([t, M1])
            self.M2_list.append([t, M2])
            self.M3_list.append([t, M3])
            # Atmospheric Conditions
            self.windVelocityX_list.append([t, self.env.windVelocityX.getValueOpt(z)])
            self.windVelocityY_list.append([t, self.env.windVelocityY.getValueOpt(z)])
            self.density_list.append([t, self.env.density.getValueOpt(z)])
            self.dynamicViscosity_list.append(
                [t, self.env.dynamicViscosity.getValueOpt(z)]
            )
            self.pressure_list.append([t, self.env.pressure.getValueOpt(z)])
            self.speedOfSound_list.append([t, self.env.speedOfSound.getValueOpt(z)])

        return uDot

    def uDotGeneralized(self, t, u, postProcessing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying in 6 DOF motion, during ascent out of rail
        and descent without parachute.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, q0, q1,
            q2, q3, omega1, omega2, omega3].
        postProcessing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.attackAngle. Default is False.

        Returns
        -------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].
        """
        # Unpack state vector
        x, y, z, vx, vy, vz, q0, q1, q2, q3, omega1, omega2, omega3 = u

        # Compute quaternion derivatives
        q_0_d = 0.5 * (-omega1 * q1 - omega2 * q2 - omega3 * q3)
        q_1_d = 0.5 * (omega1 * q0 + omega3 * q2 - omega2 * q3)
        q_2_d = 0.5 * (omega2 * q0 - omega3 * q1 + omega1 * q3)
        q_3_d = 0.5 * (omega3 * q0 + omega2 * q1 - omega1 * q2)

        # Load mass and inertia properties
        dry_mass = self.rocket.mass
        propellant_mass = self.rocket.motor.mass.getValueOpt(t)
        total_mass = dry_mass + propellant_mass
        reduced_mass = (dry_mass * propellant_mass) / (dry_mass + propellant_mass)
        m_d = self.rocket.motor.massDot.getValueOpt(t)
        m_dd = self.rocket.motor.massDot.differentiate(t, dx=1e-6)
        I_11 = self.rocket.I_11.getValueOpt(t)
        I_22 = self.rocket.I_22.getValueOpt(t)
        I_33 = self.rocket.I_33.getValueOpt(t)
        I_12 = self.rocket.I_12.getValueOpt(t)
        I_13 = self.rocket.I_13.getValueOpt(t)
        I_23 = self.rocket.I_23.getValueOpt(t)
        I_11_d = self.rocket.I_11.differentiate(t, dx=1e-6)
        I_22_d = self.rocket.I_22.differentiate(t, dx=1e-6)
        I_33_d = self.rocket.I_33.differentiate(t, dx=1e-6)
        I_12_d = self.rocket.I_12.differentiate(t, dx=1e-6)
        I_13_d = self.rocket.I_13.differentiate(t, dx=1e-6)
        I_23_d = self.rocket.I_23.differentiate(t, dx=1e-6)

        # Geometry
        # b = -self.rocket.distanceRocketPropellant
        b = (
            -(
                self.rocket.centerOfPropellantPosition(0)
                - self.rocket.centerOfDryMassPosition
            )
            * self.rocket._csys
        )
        # c = -self.rocket.distanceRocketNozzle
        c = (
            -(self.rocket.motorPosition - self.rocket.centerOfDryMassPosition)
            * self.rocket._csys
        )
        a = b * propellant_mass / total_mass
        # Load nozzle data
        r_noz_scalar = (
            self.rocket.distanceRocketNozzle
        )  # TODO: Make sure value is negative
        S_noz_zz = 0.5 * self.rocket.motor.nozzleRadius**2
        S_noz_xx = 0.5 * S_noz_zz + 0.25 * c**2
        S_noz_yy = S_noz_xx
        S_noz_xy = 0
        S_noz_yz = 0
        S_noz_zx = 0

        # Load center of mass data
        r_cm_scalar = self.rocket.centerOfMass.getValueOpt(
            t
        )  # TODO: Make sure value is negative
        r_cm_d = self.rocket.centerOfMass.differentiate(t, dx=1e-6)
        r_cm_dd = self.rocket.centerOfMass.differentiate(t, order=2, dx=1e-6)

        # Compute forces and moments
        g = self.env.g
        Thrust = self.rocket.motor.thrust.getValueOpt(t)
        R1, R2, R3 = 0, 0, 0
        M1, M2, M3 = 0, 0, 0
        ## Compute aerodynamics forces and moments
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = (
            (windVelocityX - vx) ** 2 + (windVelocityY - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        freestreamMach = freestreamSpeed / self.env.speedOfSound.getValueOpt(z)
        ### Drag Force
        if t < self.rocket.motor.burnOutTime:
            dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)
        else:
            dragCoeff = self.rocket.powerOffDrag.getValueOpt(freestreamMach)
        rho = self.env.density.getValueOpt(z)
        R3 = -0.5 * rho * (freestreamSpeed**2) * self.rocket.area * (dragCoeff)
        ### Off center moments
        M1 += self.rocket.cpEccentricityY * R3
        M2 -= self.rocket.cpEccentricityX * R3
        M1 += self.rocket.thrustEccentricityX * Thrust
        M2 -= self.rocket.thrustEccentricityY * Thrust
        ### Get rocket velocity in body frame
        a11 = 1 - 2 * (q2**2 + q3**2)
        a12 = 2 * (q1 * q2 - q0 * q3)
        a13 = 2 * (q1 * q3 + q0 * q2)
        a21 = 2 * (q1 * q2 + q0 * q3)
        a22 = 1 - 2 * (q1**2 + q3**2)
        a23 = 2 * (q2 * q3 - q0 * q1)
        a31 = 2 * (q1 * q3 - q0 * q2)
        a32 = 2 * (q2 * q3 + q0 * q1)
        a33 = 1 - 2 * (q1**2 + q2**2)
        vxB = a11 * vx + a21 * vy + a31 * vz
        vyB = a12 * vx + a22 * vy + a32 * vz
        vzB = a13 * vx + a23 * vy + a33 * vz
        ### Calculate lift and moment for each component of the rocket
        for aerodynamicSurface in self.rocket.aerodynamicSurfaces:
            compCp = aerodynamicSurface["cp"][2]
            # Component absolute velocity in body frame
            compVxB = vxB + compCp * omega2
            compVyB = vyB - compCp * omega1
            compVzB = vzB
            # Wind velocity at component
            compZ = z + compCp
            compWindVx = self.env.windVelocityX.getValueOpt(compZ)
            compWindVy = self.env.windVelocityY.getValueOpt(compZ)
            # Component freestream velocity in body frame
            compWindVxB = a11 * compWindVx + a21 * compWindVy
            compWindVyB = a12 * compWindVx + a22 * compWindVy
            compWindVzB = a13 * compWindVx + a23 * compWindVy
            compStreamVxB = compWindVxB - compVxB
            compStreamVyB = compWindVyB - compVyB
            compStreamVzB = compWindVzB - compVzB
            compStreamSpeed = (
                compStreamVxB**2 + compStreamVyB**2 + compStreamVzB**2
            ) ** 0.5
            # Component attack angle and lift force
            compAttackAngle = 0
            compLift, compLiftXB, compLiftYB = 0, 0, 0
            if compStreamVxB**2 + compStreamVyB**2 != 0:
                # Normalize component stream velocity in body frame
                compStreamVzBn = compStreamVzB / compStreamSpeed
                if -1 * compStreamVzBn < 1:
                    compAttackAngle = np.arccos(-compStreamVzBn)
                    cLift = aerodynamicSurface["cl"](compAttackAngle, freestreamMach)
                    # Component lift force magnitude
                    compLift = (
                        0.5 * rho * (compStreamSpeed**2) * self.rocket.area * cLift
                    )
                    # Component lift force components
                    liftDirNorm = (compStreamVxB**2 + compStreamVyB**2) ** 0.5
                    compLiftXB = compLift * (compStreamVxB / liftDirNorm)
                    compLiftYB = compLift * (compStreamVyB / liftDirNorm)
                    # Add to total lift force
                    R1 += compLiftXB
                    R2 += compLiftYB
                    # Add to total moment
                    M1 -= (compCp + a) * compLiftYB
                    M2 += (compCp + a) * compLiftXB
            # Calculates Roll Moment
            if aerodynamicSurface["name"] == "Fins":
                Clfdelta, Cldomega, cantAngleRad = aerodynamicSurface["roll parameters"]
                M3f = (
                    (1 / 2 * rho * freestreamSpeed**2)
                    * self.rocket.area
                    * 2
                    * self.rocket.radius
                    * Clfdelta(freestreamMach)
                    * cantAngleRad
                )
                M3d = (
                    (1 / 2 * rho * freestreamSpeed)
                    * self.rocket.area
                    * (2 * self.rocket.radius) ** 2
                    * Cldomega(freestreamMach)
                    * omega3
                    / 2
                )
                M3 += M3f - M3d

        # Compute matrices
        A_matrix = AA.autofunc_c(
            total_mass, r_cm_scalar, q0, q1, q2, q3, I_11, I_22, I_33, I_12, I_23, I_13
        )
        b_vector = bb.autofunc_c(
            q0,
            q1,
            q2,
            q3,
            omega1,
            omega2,
            omega3,
            q_0_d,
            q_1_d,
            q_2_d,
            q_3_d,
            total_mass,
            m_d,
            m_dd,
            r_cm_scalar,
            r_cm_d,
            r_cm_dd,
            r_noz_scalar,
            R3,
            R1,
            R2,
            Thrust,
            I_11,
            I_22,
            I_33,
            I_12,
            I_23,
            I_13,
            I_11_d,
            I_22_d,
            I_33_d,
            I_12_d,
            I_23_d,
            I_13_d,
            M1,
            M2,
            M3,
            g,
            S_noz_xx,
            S_noz_yy,
            S_noz_zz,
            S_noz_xy,
            S_noz_yz,
            S_noz_zx,
        )

        # Solve linear system to compute state vector derivatives
        ax, ay, az, alpha1, alpha2, alpha3 = np.linalg.solve(A_matrix, b_vector)

        # Return state vector derivatives
        uDot = [
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            q_0_d,
            q_1_d,
            q_2_d,
            q_3_d,
            alpha1,
            alpha2,
            alpha3,
        ]

        if postProcessing:
            # Dynamics variables
            self.R1.append([t, R1])
            self.R2.append([t, R2])
            self.R3.append([t, R3])
            self.M1.append([t, M1])
            self.M2.append([t, M2])
            self.M3.append([t, M3])
            # Atmospheric Conditions
            self.windVelocityX.append([t, self.env.windVelocityX(z)])
            self.windVelocityY.append([t, self.env.windVelocityY(z)])
            self.density.append([t, self.env.density(z)])
            self.dynamicViscosity.append([t, self.env.dynamicViscosity(z)])
            self.pressure.append([t, self.env.pressure(z)])
            self.speedOfSound.append([t, self.env.speedOfSound(z)])

        return uDot

    def uDotParachute(self, t, u, postProcessing=False):
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
        postProcessing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.attackAngle. Default is False.

        Return
        ------
        uDot : list
            State vector defined by uDot = [vx, vy, vz, ax, ay, az,
            e0Dot, e1Dot, e2Dot, e3Dot, alpha1, alpha2, alpha3].

        """
        # Parachute data
        CdS = self.parachuteCdS
        ka = 1
        R = 1.5
        rho = self.env.density.getValueOpt(u[2])
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
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = (
            (windVelocityX - vx) ** 2 + (windVelocityY - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        freestreamX = vx - windVelocityX
        freestreamY = vy - windVelocityY
        freestreamZ = vz
        # Determine drag force
        pseudoD = (
            -0.5 * rho * CdS * freestreamSpeed - ka * rho * 4 * np.pi * (R**2) * Rdot
        )
        Dx = pseudoD * freestreamX
        Dy = pseudoD * freestreamY
        Dz = pseudoD * freestreamZ
        ax = Dx / (mp + ma)
        ay = Dy / (mp + ma)
        az = (Dz - 9.8 * mp) / (mp + ma)

        if postProcessing:
            # Dynamics variables
            self.R1_list.append([t, Dx])
            self.R2_list.append([t, Dy])
            self.R3_list.append([t, Dz])
            self.M1_list.append([t, 0])
            self.M2_list.append([t, 0])
            self.M3_list.append([t, 0])
            # Atmospheric Conditions
            self.windVelocityX_list.append([t, self.env.windVelocityX(z)])
            self.windVelocityY_list.append([t, self.env.windVelocityY(z)])
            self.density_list.append([t, self.env.density(z)])
            self.dynamicViscosity_list.append([t, self.env.dynamicViscosity(z)])
            self.pressure_list.append([t, self.env.pressure(z)])
            self.speedOfSound_list.append([t, self.env.speedOfSound(z)])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    @cached_property
    def solutionArray(self):
        """Returns solution array of the rocket flight."""
        return np.array(self.solution)

    @cached_property
    def time(self):
        """Returns time array from solution."""
        return self.solutionArray[:, 0]

    # Process first type of outputs - state vector
    # Transform solution array into Functions
    @funcify_method("Time (s)", "X (m)", "spline", "constant")
    def x(self):
        """Rocket x position as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 1]]

    @funcify_method("Time (s)", "Y (m)", "spline", "constant")
    def y(self):
        """Rocket y position as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 2]]

    @funcify_method("Time (s)", "Z (m)", "spline", "constant")
    def z(self):
        """Rocket z position as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 3]]

    @funcify_method("Time (s)", "Vx (m/s)", "spline", "zero")
    def vx(self):
        """Rocket x velocity as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 4]]

    @funcify_method("Time (s)", "Vy (m/s)", "spline", "zero")
    def vy(self):
        """Rocket y velocity as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 5]]

    @funcify_method("Time (s)", "Vz (m/s)", "spline", "zero")
    def vz(self):
        """Rocket z velocity as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 6]]

    @funcify_method("Time (s)", "e0", "spline", "constant")
    def e0(self):
        """Rocket quaternion e0 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 7]]

    @funcify_method("Time (s)", "e1", "spline", "constant")
    def e1(self):
        """Rocket quaternion e1 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 8]]

    @funcify_method("Time (s)", "e2", "spline", "constant")
    def e2(self):
        """Rocket quaternion e2 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 9]]

    @funcify_method("Time (s)", "e3", "spline", "constant")
    def e3(self):
        """Rocket quaternion e3 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 10]]

    @funcify_method("Time (s)", "ω1 (rad/s)", "spline", "zero")
    def w1(self):
        """Rocket angular velocity ω1 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 11]]

    @funcify_method("Time (s)", "ω2 (rad/s)", "spline", "zero")
    def w2(self):
        """Rocket angular velocity ω2 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 12]]

    @funcify_method("Time (s)", "ω3 (rad/s)", "spline", "zero")
    def w3(self):
        """Rocket angular velocity ω3 as a rocketpy.Function of time."""
        return self.solutionArray[:, [0, 13]]

    # Process second type of outputs - accelerations components
    @funcify_method("Time (s)", "Ax (m/s²)", "spline", "zero")
    def ax(self):
        """Rocket x acceleration as a rocketpy.Function of time."""
        return self.retrieve_acceleration_arrays[0]

    @funcify_method("Time (s)", "Ay (m/s²)", "spline", "zero")
    def ay(self):
        """Rocket y acceleration as a rocketpy.Function of time."""
        return self.retrieve_acceleration_arrays[1]

    @funcify_method("Time (s)", "Az (m/s²)", "spline", "zero")
    def az(self):
        """Rocket z acceleration as a rocketpy.Function of time."""
        return self.retrieve_acceleration_arrays[2]

    @funcify_method("Time (s)", "α1 (rad/s²)", "spline", "zero")
    def alpha1(self):
        """Rocket angular acceleration α1 as a rocketpy.Function of time."""
        return self.retrieve_acceleration_arrays[3]

    @funcify_method("Time (s)", "α2 (rad/s²)", "spline", "zero")
    def alpha2(self):
        """Rocket angular acceleration α2 as a rocketpy.Function of time."""
        return self.retrieve_acceleration_arrays[4]

    @funcify_method("Time (s)", "α3 (rad/s²)", "spline", "zero")
    def alpha3(self):
        """Rocket angular acceleration α3 as a rocketpy.Function of time."""
        return self.retrieve_acceleration_arrays[5]

    # Process third type of outputs - Temporary values
    @funcify_method("Time (s)", "R1 (N)", "spline", "zero")
    def R1(self):
        """Aerodynamic force along the first axis that is perpendicular to the
        rocket's axis of symmetry as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[0]

    @funcify_method("Time (s)", "R2 (N)", "spline", "zero")
    def R2(self):
        """Aerodynamic force along the second axis that is perpendicular to the
        rocket's axis of symmetry as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[1]

    @funcify_method("Time (s)", "R3 (N)", "spline", "zero")
    def R3(self):
        """Aerodynamic force along the rocket's axis of symmetry as a rocketpy.Function
        of time."""
        return self.retrieve_temporary_values_arrays[2]

    @funcify_method("Time (s)", "M1 (Nm)", "spline", "zero")
    def M1(self):
        """Aerodynamic bending moment in the same direction as the axis that is
        perpendicular to the rocket's axis of symmetry as a rocketpy.Function of time.
        """
        return self.retrieve_temporary_values_arrays[3]

    @funcify_method("Time (s)", "M2 (Nm)", "spline", "zero")
    def M2(self):
        """Aerodynamic bending moment in the same direction as the axis that is
        perpendicular to the rocket's axis of symmetry as a rocketpy.Function of time.
        """
        return self.retrieve_temporary_values_arrays[4]

    @funcify_method("Time (s)", "M3 (Nm)", "spline", "zero")
    def M3(self):
        """Aerodynamic bending moment in the same direction as the rocket's axis of
        symmetry as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[5]

    @funcify_method("Time (s)", "Pressure (Pa)", "spline", "constant")
    def pressure(self):
        """Air pressure felt by the rocket as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[6]

    @funcify_method("Time (s)", "Density (kg/m³)", "spline", "constant")
    def density(self):
        """Air density felt by the rocket as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[7]

    @funcify_method("Time (s)", "Dynamic Viscosity (Pa s)", "spline", "constant")
    def dynamicViscosity(self):
        """Air dynamic viscosity felt by the rocket as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[8]

    @funcify_method("Time (s)", "Speed of Sound (m/s)", "spline", "constant")
    def speedOfSound(self):
        """Speed of sound in the air felt by the rocket as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[9]

    @funcify_method("Time (s)", "Wind Velocity X (East) (m/s)", "spline", "constant")
    def windVelocityX(self):
        """Wind velocity in the X direction (east) as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[10]

    @funcify_method("Time (s)", "Wind Velocity Y (North) (m/s)", "spline", "constant")
    def windVelocityY(self):
        """Wind velocity in the y direction (north) as a rocketpy.Function of time."""
        return self.retrieve_temporary_values_arrays[11]

    # Process fourth type of output - values calculated from previous outputs

    # Kinematics functions and values
    # Velocity Magnitude
    @funcify_method("Time (s)", "Speed - Velocity Magnitude (m/s)")
    def speed(self):
        """Rocket speed, or velocity magnitude, as a rocketpy.Function of time."""
        return (self.vx**2 + self.vy**2 + self.vz**2) ** 0.5

    @cached_property
    def maxSpeedTime(self):
        """Time at which the rocket reaches its maximum speed."""
        maxSpeedTimeIndex = np.argmax(self.speed[:, 1])
        return self.speed[maxSpeedTimeIndex, 0]

    @cached_property
    def maxSpeed(self):
        """Maximum speed reached by the rocket."""
        return self.speed(self.maxSpeedTime)

    # Accelerations
    @funcify_method("Time (s)", "Acceleration Magnitude (m/s²)")
    def acceleration(self):
        """Rocket acceleration magnitude as a rocketpy.Function of time."""
        return (self.ax**2 + self.ay**2 + self.az**2) ** 0.5

    @cached_property
    def maxAcceleration(self):
        """Maximum acceleration reached by the rocket."""
        maxAccelerationTimeIndex = np.argmax(self.acceleration[:, 1])
        return self.acceleration[maxAccelerationTimeIndex, 1]

    @cached_property
    def maxAccelerationTime(self):
        """Time at which the rocket reaches its maximum acceleration."""
        maxAccelerationTimeIndex = np.argmax(self.acceleration[:, 1])
        return self.acceleration[maxAccelerationTimeIndex, 0]

    @funcify_method("Time (s)", "Horizontal Speed (m/s)")
    def horizontalSpeed(self):
        """Rocket horizontal speed as a rocketpy.Function of time."""
        return (self.vx**2 + self.vy**2) ** 0.5

    # Path Angle
    @funcify_method("Time (s)", "Path Angle (°)", "spline", "constant")
    def pathAngle(self):
        """Rocket path angle as a rocketpy.Function of time."""
        pathAngle = (180 / np.pi) * np.arctan2(
            self.vz[:, 1], self.horizontalSpeed[:, 1]
        )
        return np.column_stack([self.time, pathAngle])

    # Attitude Angle
    @funcify_method("Time (s)", "Attitude Vector X Component")
    def attitudeVectorX(self):
        """Rocket attitude vector X component as a rocketpy.Function of time."""
        return 2 * (self.e1 * self.e3 + self.e0 * self.e2)  # a13

    @funcify_method("Time (s)", "Attitude Vector Y Component")
    def attitudeVectorY(self):
        """Rocket attitude vector Y component as a rocketpy.Function of time."""
        return 2 * (self.e2 * self.e3 - self.e0 * self.e1)  # a23

    @funcify_method("Time (s)", "Attitude Vector Z Component")
    def attitudeVectorZ(self):
        """Rocket attitude vector Z component as a rocketpy.Function of time."""
        return 1 - 2 * (self.e1**2 + self.e2**2)  # a33

    @funcify_method("Time (s)", "Attitude Angle (°)")
    def attitudeAngle(self):
        """Rocket attitude angle as a rocketpy.Function of time."""
        horizontalAttitudeProj = (
            self.attitudeVectorX**2 + self.attitudeVectorY**2
        ) ** 0.5
        attitudeAngle = (180 / np.pi) * np.arctan2(
            self.attitudeVectorZ[:, 1], horizontalAttitudeProj[:, 1]
        )
        attitudeAngle = np.column_stack([self.time, attitudeAngle])
        return attitudeAngle

    # Lateral Attitude Angle
    @funcify_method("Time (s)", "Lateral Attitude Angle (°)")
    def lateralAttitudeAngle(self):
        """Rocket lateral attitude angle as a rocketpy.Function of time."""
        lateralVectorAngle = (np.pi / 180) * (self.heading - 90)
        lateralVectorX = np.sin(lateralVectorAngle)
        lateralVectorY = np.cos(lateralVectorAngle)
        attitudeLateralProj = (
            lateralVectorX * self.attitudeVectorX[:, 1]
            + lateralVectorY * self.attitudeVectorY[:, 1]
        )
        attitudeLateralProjX = attitudeLateralProj * lateralVectorX
        attitudeLateralProjY = attitudeLateralProj * lateralVectorY
        attitudeLateralPlaneProjX = self.attitudeVectorX[:, 1] - attitudeLateralProjX
        attitudeLateralPlaneProjY = self.attitudeVectorY[:, 1] - attitudeLateralProjY
        attitudeLateralPlaneProjZ = self.attitudeVectorZ[:, 1]
        attitudeLateralPlaneProj = (
            attitudeLateralPlaneProjX**2
            + attitudeLateralPlaneProjY**2
            + attitudeLateralPlaneProjZ**2
        ) ** 0.5
        lateralAttitudeAngle = (180 / np.pi) * np.arctan2(
            attitudeLateralProj, attitudeLateralPlaneProj
        )
        lateralAttitudeAngle = np.column_stack([self.time, lateralAttitudeAngle])
        return lateralAttitudeAngle

    # Euler Angles
    @funcify_method("Time (s)", "Precession Angle - ψ (°)", "spline", "constant")
    def psi(self):
        """Precession angle as a rocketpy.Function of time."""
        psi = (180 / np.pi) * (
            np.arctan2(self.e3[:, 1], self.e0[:, 1])
            + np.arctan2(-self.e2[:, 1], -self.e1[:, 1])
        )  # Precession angle
        psi = np.column_stack([self.time, psi])  # Precession angle
        return psi

    @funcify_method("Time (s)", "Spin Angle - φ (°)", "spline", "constant")
    def phi(self):
        """Spin angle as a rocketpy.Function of time."""
        phi = (180 / np.pi) * (
            np.arctan2(self.e3[:, 1], self.e0[:, 1])
            - np.arctan2(-self.e2[:, 1], -self.e1[:, 1])
        )  # Spin angle
        phi = np.column_stack([self.time, phi])  # Spin angle
        return phi

    @funcify_method("Time (s)", "Nutation Angle - θ (°)", "spline", "constant")
    def theta(self):
        """Nutation angle as a rocketpy.Function of time."""
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
    def streamVelocityX(self):
        """Freestream velocity X component as a rocketpy.Function of time."""
        streamVelocityX = np.column_stack(
            (self.time, self.windVelocityX[:, 1] - self.vx[:, 1])
        )
        return streamVelocityX

    @funcify_method("Time (s)", "Freestream Velocity Y (m/s)", "spline", "constant")
    def streamVelocityY(self):
        """Freestream velocity Y component as a rocketpy.Function of time."""
        streamVelocityY = np.column_stack(
            (self.time, self.windVelocityY[:, 1] - self.vy[:, 1])
        )
        return streamVelocityY

    @funcify_method("Time (s)", "Freestream Velocity Z (m/s)", "spline", "constant")
    def streamVelocityZ(self, interpolation="spline", extrapolation="natural"):
        """Freestream velocity Z component as a rocketpy.Function of time."""
        streamVelocityZ = np.column_stack((self.time, -self.vz[:, 1]))
        return streamVelocityZ

    @funcify_method("Time (s)", "Freestream Speed (m/s)", "spline", "constant")
    def freestreamSpeed(self):
        """Freestream speed as a rocketpy.Function of time."""
        freestreamSpeed = (
            self.streamVelocityX**2
            + self.streamVelocityY**2
            + self.streamVelocityZ**2
        ) ** 0.5
        return freestreamSpeed.source

    # Apogee Freestream speed
    @cached_property
    def apogeeFreestreamSpeed(self):
        """Freestream speed at apogee in m/s."""
        return self.freestreamSpeed(self.apogeeTime)

    # Mach Number
    @funcify_method("Time (s)", "Mach Number", "spline", "zero")
    def MachNumber(self):
        """Mach number as a rocketpy.Function of time."""
        return self.freestreamSpeed / self.speedOfSound

    @cached_property
    def maxMachNumberTime(self):
        """Time of maximum Mach number."""
        maxMachNumberTimeIndex = np.argmax(self.MachNumber[:, 1])
        return self.MachNumber[maxMachNumberTimeIndex, 0]

    @cached_property
    def maxMachNumber(self):
        """Maximum Mach number."""
        return self.MachNumber(self.maxMachNumberTime)

    # Reynolds Number
    @funcify_method("Time (s)", "Reynolds Number", "spline", "zero")
    def ReynoldsNumber(self):
        """Reynolds number as a rocketpy.Function of time."""
        return (self.density * self.freestreamSpeed / self.dynamicViscosity) * (
            2 * self.rocket.radius
        )

    @cached_property
    def maxReynoldsNumberTime(self):
        """Time of maximum Reynolds number."""
        maxReynoldsNumberTimeIndex = np.argmax(self.ReynoldsNumber[:, 1])
        return self.ReynoldsNumber[maxReynoldsNumberTimeIndex, 0]

    @cached_property
    def maxReynoldsNumber(self):
        """Maximum Reynolds number."""
        return self.ReynoldsNumber(self.maxReynoldsNumberTime)

    # Dynamic Pressure
    @funcify_method("Time (s)", "Dynamic Pressure (Pa)", "spline", "zero")
    def dynamicPressure(self):
        """Dynamic pressure as a rocketpy.Function of time."""
        return 0.5 * self.density * self.freestreamSpeed**2

    @cached_property
    def maxDynamicPressureTime(self):
        """Time of maximum dynamic pressure."""
        maxDynamicPressureTimeIndex = np.argmax(self.dynamicPressure[:, 1])
        return self.dynamicPressure[maxDynamicPressureTimeIndex, 0]

    @cached_property
    def maxDynamicPressure(self):
        """Maximum dynamic pressure."""
        return self.dynamicPressure(self.maxDynamicPressureTime)

    # Total Pressure
    @funcify_method("Time (s)", "Total Pressure (Pa)", "spline", "zero")
    def totalPressure(self):
        return self.pressure * (1 + 0.2 * self.MachNumber**2) ** (3.5)

    @cached_property
    def maxTotalPressureTime(self):
        """Time of maximum total pressure."""
        maxTotalPressureTimeIndex = np.argmax(self.totalPressure[:, 1])
        return self.totalPressure[maxTotalPressureTimeIndex, 0]

    @cached_property
    def maxTotalPressure(self):
        """Maximum total pressure."""
        return self.totalPressure(self.maxTotalPressureTime)

    # Dynamics functions and variables

    #  Aerodynamic Lift and Drag
    @funcify_method("Time (s)", "Aerodynamic Lift Force (N)", "spline", "zero")
    def aerodynamicLift(self):
        """Aerodynamic lift force as a rocketpy.Function of time."""
        return (self.R1**2 + self.R2**2) ** 0.5

    @funcify_method("Time (s)", "Aerodynamic Drag Force (N)", "spline", "zero")
    def aerodynamicDrag(self):
        """Aerodynamic drag force as a rocketpy.Function of time."""
        return -1 * self.R3

    @funcify_method("Time (s)", "Aerodynamic Bending Moment (Nm)", "spline", "zero")
    def aerodynamicBendingMoment(self):
        """Aerodynamic bending moment as a rocketpy.Function of time."""
        return (self.M1**2 + self.M2**2) ** 0.5

    @funcify_method("Time (s)", "Aerodynamic Spin Moment (Nm)", "spline", "zero")
    def aerodynamicSpinMoment(self):
        """Aerodynamic spin moment as a rocketpy.Function of time."""
        return self.M3

    # Energy
    # Kinetic Energy
    @funcify_method("Time (s)", "Rotational Kinetic Energy (J)")
    def rotationalEnergy(self):
        # b = -self.rocket.distanceRocketPropellant
        b = (
            -(self.rocket.motorPosition - self.rocket.centerOfDryMassPosition)
            * self.rocket._csys
        )
        mu = self.rocket.reducedMass
        Rz = self.rocket.dry_I_33
        Ri = self.rocket.dry_I_11
        Tz = self.rocket.motor.I_33
        Ti = self.rocket.motor.I_11
        I1, I2, I3 = (Ri + Ti + mu * b**2), (Ri + Ti + mu * b**2), (Rz + Tz)
        # Redefine I1, I2 and I3 time grid to allow for efficient Function algebra
        I1.setDiscreteBasedOnModel(self.w1)
        I2.setDiscreteBasedOnModel(self.w1)
        I3.setDiscreteBasedOnModel(self.w1)
        rotationalEnergy = 0.5 * (
            I1 * self.w1**2 + I2 * self.w2**2 + I3 * self.w3**2
        )
        rotationalEnergy.setDiscreteBasedOnModel(self.w1)
        return rotationalEnergy

    @funcify_method("Time (s)", "Translational Kinetic Energy (J)", "spline", "zero")
    def translationalEnergy(self):
        """Translational kinetic energy as a rocketpy.Function of time."""
        # Redefine totalMass time grid to allow for efficient Function algebra
        totalMass = deepcopy(self.rocket.totalMass)
        totalMass.setDiscreteBasedOnModel(self.vz)
        translationalEnergy = (
            0.5 * totalMass * (self.vx**2 + self.vy**2 + self.vz**2)
        )
        return translationalEnergy

    @funcify_method("Time (s)", "Kinetic Energy (J)", "spline", "zero")
    def kineticEnergy(self):
        """Total kinetic energy as a rocketpy.Function of time."""
        return self.rotationalEnergy + self.translationalEnergy

    # Potential Energy
    @funcify_method("Time (s)", "Potential Energy (J)", "spline", "constant")
    def potentialEnergy(self):
        """Potential energy as a rocketpy.Function of time."""
        # Redefine totalMass time grid to allow for efficient Function algebra
        totalMass = deepcopy(self.rocket.totalMass)
        totalMass.setDiscreteBasedOnModel(self.z)
        # TODO: change calculation method to account for variable gravity
        potentialEnergy = totalMass * self.env.g * self.z
        return potentialEnergy

    # Total Mechanical Energy
    @funcify_method("Time (s)", "Mechanical Energy (J)", "spline", "constant")
    def totalEnergy(self):
        """Total mechanical energy as a rocketpy.Function of time."""
        return self.kineticEnergy + self.potentialEnergy

    # Thrust Power
    @funcify_method("Time (s)", "Thrust Power (W)", "spline", "zero")
    def thrustPower(self):
        """Thrust power as a rocketpy.Function of time."""
        thrust = deepcopy(self.rocket.motor.thrust)
        thrust = thrust.setDiscreteBasedOnModel(self.speed)
        thrustPower = thrust * self.speed
        return thrustPower

    # Drag Power
    @funcify_method("Time (s)", "Drag Power (W)", "spline", "zero")
    def dragPower(self):
        """Drag power as a rocketpy.Function of time."""
        dragPower = self.R3 * self.speed
        dragPower.setOutputs("Drag Power (W)")
        return dragPower

    # Angle of Attack
    @funcify_method("Time (s)", "Angle of Attack (°)", "spline", "constant")
    def angleOfAttack(self):
        """Angle of attack of the rocket with respect to the freestream
        velocity vector."""
        dotProduct = [
            -self.attitudeVectorX.getValueOpt(i) * self.streamVelocityX.getValueOpt(i)
            - self.attitudeVectorY.getValueOpt(i) * self.streamVelocityY.getValueOpt(i)
            - self.attitudeVectorZ.getValueOpt(i) * self.streamVelocityZ.getValueOpt(i)
            for i in self.time
        ]
        # Define freestream speed list
        freestreamSpeed = [self.freestreamSpeed(i) for i in self.time]
        freestreamSpeed = np.nan_to_num(freestreamSpeed)

        # Normalize dot product
        dotProductNormalized = [
            i / j if j > 1e-6 else 0 for i, j in zip(dotProduct, freestreamSpeed)
        ]
        dotProductNormalized = np.nan_to_num(dotProductNormalized)
        dotProductNormalized = np.clip(dotProductNormalized, -1, 1)

        # Calculate angle of attack and convert to degrees
        angleOfAttack = np.rad2deg(np.arccos(dotProductNormalized))
        angleOfAttack = np.column_stack([self.time, angleOfAttack])

        return angleOfAttack

    # Frequency response and stability variables
    @funcify_method("Frequency (Hz)", "ω1 Fourier Amplitude", "spline", "zero")
    def omega1FrequencyResponse(self):
        """Angular velocity 1 frequency response as a rocketpy.Function of frequency,
        as the rocket leaves the launch rail for 5 seconds of flight."""
        return self.w1.toFrequencyDomain(
            self.outOfRailTime, self.outOfRailTime + 5, 100
        )

    @funcify_method("Frequency (Hz)", "ω2 Fourier Amplitude", "spline", "zero")
    def omega2FrequencyResponse(self):
        """Angular velocity 2 frequency response as a rocketpy.Function of frequency,
        as the rocket leaves the launch rail for 5 seconds of flight."""
        return self.w2.toFrequencyDomain(
            self.outOfRailTime, self.outOfRailTime + 5, 100
        )

    @funcify_method("Frequency (Hz)", "ω3 Fourier Amplitude", "spline", "zero")
    def omega3FrequencyResponse(self):
        """Angular velocity 3 frequency response as a rocketpy.Function of frequency,
        as the rocket leaves the launch rail for 5 seconds of flight."""
        return self.w3.toFrequencyDomain(
            self.outOfRailTime, self.outOfRailTime + 5, 100
        )

    @funcify_method(
        "Frequency (Hz)", "Attitude Angle Fourier Amplitude", "spline", "zero"
    )
    def attitudeFrequencyResponse(self):
        """Attitude frequency response as a rocketpy.Function of frequency, as the
        rocket leaves the launch rail for 5 seconds of flight."""
        return self.attitudeAngle.toFrequencyDomain(
            lower=self.outOfRailTime,
            upper=self.outOfRailTime + 5,
            samplingFrequency=100,
        )

    @cached_property
    def staticMargin(self):
        """Static margin of the rocket."""
        return self.rocket.staticMargin

    # Rail Button Forces
    @funcify_method("Time (s)", "Upper Rail Button Normal Force (N)", "spline", "zero")
    def railButton1NormalForce(self):
        """Upper rail button normal force as a rocketpy.Function of time."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F11, F12 = self.calculate_rail_button_forces[0:2]
        else:
            F11, F12 = self.calculate_rail_button_forces()[0:2]
        alpha = self.rocket.railButtons.angularPosition * (np.pi / 180)
        return F11 * np.cos(alpha) + F12 * np.sin(alpha)

    @funcify_method("Time (s)", "Upper Rail Button Shear Force (N)", "spline", "zero")
    def railButton1ShearForce(self):
        """Upper rail button shear force as a rocketpy.Function of time."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F11, F12 = self.calculate_rail_button_forces[0:2]
        else:
            F11, F12 = self.calculate_rail_button_forces()[0:2]
        alpha = self.rocket.railButtons.angularPosition * (
            np.pi / 180
        )  # Rail buttons angular position
        return F11 * -np.sin(alpha) + F12 * np.cos(alpha)

    @funcify_method("Time (s)", "Lower Rail Button Normal Force (N)", "spline", "zero")
    def railButton2NormalForce(self):
        """Lower rail button normal force as a rocketpy.Function of time."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F21, F22 = self.calculate_rail_button_forces[2:4]
        else:
            F21, F22 = self.calculate_rail_button_forces()[2:4]
        alpha = self.rocket.railButtons.angularPosition * (np.pi / 180)
        return F21 * np.cos(alpha) + F22 * np.sin(alpha)

    @funcify_method("Time (s)", "Lower Rail Button Shear Force (N)", "spline", "zero")
    def railButton2ShearForce(self):
        """Lower rail button shear force as a rocketpy.Function of time."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F21, F22 = self.calculate_rail_button_forces[2:4]
        else:
            F21, F22 = self.calculate_rail_button_forces()[2:4]
        alpha = self.rocket.railButtons.angularPosition * (np.pi / 180)
        return F21 * -np.sin(alpha) + F22 * np.cos(alpha)

    @cached_property
    def maxRailButton1NormalForce(self):
        """Maximum upper rail button normal force, in Newtons."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F11 = self.calculate_rail_button_forces[0]
        else:
            F11 = self.calculate_rail_button_forces()[0]
        if self.outOfRailTimeIndex == 0:
            return 0
        else:
            return np.max(self.railButton1NormalForce)

    @cached_property
    def maxRailButton1ShearForce(self):
        """Maximum upper rail button shear force, in Newtons."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F11 = self.calculate_rail_button_forces[0]
        else:
            F11 = self.calculate_rail_button_forces()[0]
        if self.outOfRailTimeIndex == 0:
            return 0
        else:
            return np.max(self.railButton1ShearForce)

    @cached_property
    def maxRailButton2NormalForce(self):
        """Maximum lower rail button normal force, in Newtons."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F11 = self.calculate_rail_button_forces[0]
        else:
            F11 = self.calculate_rail_button_forces()[0]
        if self.outOfRailTimeIndex == 0:
            return 0
        else:
            return np.max(self.railButton2NormalForce)

    @cached_property
    def maxRailButton2ShearForce(self):
        """Maximum lower rail button shear force, in Newtons."""
        if isinstance(self.calculate_rail_button_forces, tuple):
            F11 = self.calculate_rail_button_forces[0]
        else:
            F11 = self.calculate_rail_button_forces()[0]
        if self.outOfRailTimeIndex == 0:
            return 0
        else:
            return np.max(self.railButton2ShearForce)

    @funcify_method(
        "Time (s)", "Horizontal Distance to Launch Point (m)", "spline", "constant"
    )
    def drift(self):
        """Rocket horizontal distance to tha launch point, in meters, as a
        rocketpy.Function of time."""
        return np.column_stack(
            (self.time, (self.x[:, 1] ** 2 + self.y[:, 1] ** 2) ** 0.5)
        )

        return drift

    @funcify_method("Time (s)", "Bearing (°)", "spline", "constant")
    def bearing(self):
        """Rocket bearing compass, in degrees, as a rocketpy.Function of time."""
        x, y = self.x[:, 1], self.y[:, 1]
        bearing = (2 * np.pi - np.arctan2(-x, y)) * (180 / np.pi)
        return np.column_stack((self.time, bearing))

    @funcify_method("Time (s)", "Latitude (°)", "linear", "constant")
    def latitude(self):
        """Rocket latitude coordinate, in degrees, as a rocketpy.Function of time."""
        lat1 = np.deg2rad(self.env.lat)  # Launch lat point converted to radians

        # Applies the haversine equation to find final lat/lon coordinates
        latitude = np.rad2deg(
            np.arcsin(
                np.sin(lat1) * np.cos(self.drift[:, 1] / self.env.earthRadius)
                + np.cos(lat1)
                * np.sin(self.drift[:, 1] / self.env.earthRadius)
                * np.cos(np.deg2rad(self.bearing[:, 1]))
            )
        )
        return np.column_stack((self.time, latitude))

    @funcify_method("Time (s)", "Longitude (°)", "linear", "constant")
    def longitude(self):
        """Rocket longitude coordinate, in degrees, as a rocketpy.Function of time."""
        lat1 = np.deg2rad(self.env.lat)  # Launch lat point converted to radians
        lon1 = np.deg2rad(self.env.lon)  # Launch lon point converted to radians

        # Applies the haversine equation to find final lat/lon coordinates
        longitude = np.rad2deg(
            lon1
            + np.arctan2(
                np.sin(np.deg2rad(self.bearing[:, 1]))
                * np.sin(self.drift[:, 1] / self.env.earthRadius)
                * np.cos(lat1),
                np.cos(self.drift[:, 1] / self.env.earthRadius)
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
        for phase_index, phase in self.timeIterator(self.flightPhases):
            initTime = phase.t
            finalTime = self.flightPhases[phase_index + 1].t
            currentDerivative = phase.derivative
            # Call callback functions
            for callback in phase.callbacks:
                callback(self)
            # Loop through time steps in flight phase
            for step in self.solution:  # Can be optimized
                if initTime < step[0] <= finalTime:
                    # Get derivatives
                    uDot = currentDerivative(step[0], step[1:])
                    # Get accelerations
                    ax_value, ay_value, az_value = uDot[3:6]
                    alpha1_value, alpha2_value, alpha3_value = uDot[10:]
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
        Currently, the following temporary values are retrieved:
            - R1
            - R2
            - R3
            - M1
            - M2
            - M3
            - pressure
            - density
            - dynamicViscosity
            - speedOfSound

        Parameters
        ----------
        None

        Returns
        -------
        self.R1_list: list
            R1 values
        self.R2_list: list
            R2 values
        self.R3_list: list
            R3 values are the aerodynamic force values in the rocket's axis direction
        self.M1_list: list
            M1 values
        self.M2_list: list
            Aerodynamic bending moment in ? direction at each time step
        self.M3_list: list
            Aerodynamic bending moment in ? direction at each time step
        self.pressure_list: list
            Air pressure at each time step
        self.density_list: list
            Air density at each time step
        self.dynamicViscosity_list: list
            Dynamic viscosity at each time step
        elf_list._speedOfSound: list
            Speed of sound at each time step
        self.windVelocityX_list: list
            Wind velocity in x direction at each time step
        self.windVelocityY_list: list
            Wind velocity in y direction at each time step
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
        self.dynamicViscosity_list = []
        self.speedOfSound_list = []
        self.windVelocityX_list = []
        self.windVelocityY_list = []

        # Go through each time step and calculate forces and atmospheric values
        # Get flight phases
        for phase_index, phase in self.timeIterator(self.flightPhases):
            initTime = phase.t
            finalTime = self.flightPhases[phase_index + 1].t
            currentDerivative = phase.derivative
            # Call callback functions
            for callback in phase.callbacks:
                callback(self)
            # Loop through time steps in flight phase
            for step in self.solution:  # Can be optimized
                if initTime < step[0] <= finalTime or (
                    initTime == self.tInitial and step[0] == self.tInitial
                ):
                    # Call derivatives in post processing mode
                    uDot = currentDerivative(step[0], step[1:], postProcessing=True)

        temporary_values = [
            self.R1_list,
            self.R2_list,
            self.R3_list,
            self.M1_list,
            self.M2_list,
            self.M3_list,
            self.pressure_list,
            self.density_list,
            self.dynamicViscosity_list,
            self.speedOfSound_list,
            self.windVelocityX_list,
            self.windVelocityY_list,
        ]

        return temporary_values

    @cached_property
    def calculate_rail_button_forces(self):
        """Calculate the forces applied to the rail buttons while rocket is still
        on the launch rail. It will return 0 if no rail buttons are defined.

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
        if self.rocket.railButtons is None:
            warnings.warn(
                "Trying to calculate rail button forces without rail buttons defined."
            )
            return 0, 0, 0, 0
        if self.outOfRailTimeIndex == 0:
            # No rail phase, no rail button forces
            nullForce = 0 * self.R1
            return nullForce, nullForce, nullForce, nullForce

        # Distance from Rail Button 1 (upper) to CM
        D1 = (
            self.rocket.railButtons.position[0] - self.rocket.centerOfDryMassPosition
        ) * self.rocket._csys
        # Distance from Rail Button 2 (lower) to CM
        D2 = (
            self.rocket.railButtons.position[1] - self.rocket.centerOfDryMassPosition
        ) * self.rocket._csys
        F11 = (self.R1 * D2 - self.M2) / (D1 + D2)
        F11.setOutputs("Upper button force direction 1 (m)")
        F12 = (self.R2 * D2 + self.M1) / (D1 + D2)
        F12.setOutputs("Upper button force direction 2 (m)")
        F21 = (self.R1 * D1 + self.M2) / (D1 + D2)
        F21.setOutputs("Lower button force direction 1 (m)")
        F22 = (self.R2 * D1 - self.M1) / (D1 + D2)
        F22.setOutputs("Lower button force direction 2 (m)")

        model = Function(
            F11.source[: self.outOfRailTimeIndex + 1, :],
            interpolation=F11.__interpolation__,
        )

        # Limit force calculation to when rocket is in rail
        F11.setDiscreteBasedOnModel(model)
        F12.setDiscreteBasedOnModel(model)
        F21.setDiscreteBasedOnModel(model)
        F22.setDiscreteBasedOnModel(model)

        return F11, F12, F21, F22

    def _calculate_pressure_signal(self):
        """Calculate the pressure signal from the pressure sensor.
        It creates a SignalFunction attribute in the parachute object.
        Parachute works as a subclass of Rocket class.

        Returns
        -------
        None
        """
        # Transform parachute sensor feed into functions
        for parachute in self.rocket.parachutes:
            parachute.cleanPressureSignalFunction = Function(
                parachute.cleanPressureSignal,
                "Time (s)",
                "Pressure - Without Noise (Pa)",
                "linear",
            )
            parachute.noisyPressureSignalFunction = Function(
                parachute.noisyPressureSignal,
                "Time (s)",
                "Pressure - With Noise (Pa)",
                "linear",
            )
            parachute.noiseSignalFunction = Function(
                parachute.noiseSignal, "Time (s)", "Pressure Noise (Pa)", "linear"
            )

        return None

    def postProcess(self, interpolation="spline", extrapolation="natural"):
        """Post-process all Flight information produced during
        simulation. Includes the calculation of maximum values,
        calculation of secondary values such as energy and conversion
        of lists to Function objects to facilitate plotting.

        * This method is deprecated and is only kept here for backwards compatibility.
        * All attributes that need to be post processed are computed just in time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Register post processing
        self.postProcessed = True

        return None

    def info(self):
        """Prints out a summary of the data available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        self.prints.all()
        return None

    def calculateStallWindVelocity(self, stallAngle):
        """Function to calculate the maximum wind velocity before the angle of
        attack exceeds a desired angle, at the instant of departing rail launch.
        Can be helpful if you know the exact stall angle of all aerodynamics
        surfaces.

        Parameters
        ----------
        stallAngle : float
            Angle, in degrees, for which you would like to know the maximum wind
            speed before the angle of attack exceeds it
        Return
        ------
        None
        """
        vF = self.outOfRailVelocity

        # Convert angle to radians
        theta = self.inclination * 3.14159265359 / 180
        stallAngle = stallAngle * 3.14159265359 / 180

        c = (math.cos(stallAngle) ** 2 - math.cos(theta) ** 2) / math.sin(
            stallAngle
        ) ** 2
        wV = (
            2 * vF * math.cos(theta) / c
            + (
                4 * vF * vF * math.cos(theta) * math.cos(theta) / (c**2)
                + 4 * 1 * vF * vF / c
            )
            ** 0.5
        ) / 2

        # Convert stallAngle to degrees
        stallAngle = stallAngle * 180 / np.pi
        print(
            "Maximum wind velocity at Rail Departure time before angle of attack exceeds {:.3f}°: {:.3f} m/s".format(
                stallAngle, wV
            )
        )

        return None

    def exportPressures(self, fileName, timeStep):
        """Exports the pressure experienced by the rocket during the flight to
        an external file, the '.csv' format is recommended, as the columns will
        be separated by commas. It can handle flights with or without parachutes,
        although it is not possible to get a noisy pressure signal if no
        parachute is added.

        If a parachute is added, the file will contain 3 columns: time in seconds,
        clean pressure in Pascals and noisy pressure in Pascals. For flights without
        parachutes, the third column will be discarded

        This function was created especially for the 'Projeto Jupiter' Electronics
        Subsystems team and aims to help in configuring micro-controllers.

        Parameters
        ----------
        fileName : string
            The final file name,
        timeStep : float
            Time step desired for the final file

        Return
        ------
        None
        """

        timePoints = np.arange(0, self.tFinal, timeStep)

        # Create the file
        file = open(fileName, "w")

        if len(self.rocket.parachutes) == 0:
            pressure = self.env.pressure(self.z(timePoints))
            for i in range(0, timePoints.size, 1):
                file.write("{:f}, {:.5f}\n".format(timePoints[i], pressure[i]))

        else:
            for parachute in self.rocket.parachutes:
                for i in range(0, timePoints.size, 1):
                    pCl = Function(
                        parachute.cleanPressureSignal,
                        "Time (s)",
                        "Pressure - Without Noise (Pa)",
                        "linear",
                    )(timePoints[i])
                    pNs = Function(
                        parachute.noisyPressureSignal,
                        "Time (s)",
                        "Pressure - With Noise (Pa)",
                        "linear",
                    )(timePoints[i])
                    file.write("{:f}, {:.5f}, {:.5f}\n".format(timePoints[i], pCl, pNs))
                # We need to save only 1 parachute data
                pass

        file.close()

        return None

    def exportData(self, fileName, *variables, timeStep=None):
        """Exports flight data to a comma separated value file (.csv).

        Data is exported in columns, with the first column representing time
        steps. The first line of the file is a header line, specifying the
        meaning of each column and its units.

        Parameters
        ----------
        fileName : string
            The file name or path of the exported file. Example: flight_data.csv.
            Do not use forbidden characters, such as '/' in Linux/Unix and
            '<, >, :, ", /, \\, | ?, *' in Windows.
        variables : strings, optional
            Names of the data variables which shall be exported. Must be Flight
            class attributes which are instances of the Function class. Usage
            example: TestFlight.exportData('test.csv', 'z', 'angleOfAttack',
            'machNumber').
        timeStep : float, optional
            Time step desired for the data. If None, all integration time steps
            will be exported. Otherwise, linear interpolation is carried out to
            calculate values at the desired time steps. Example: 0.001.
        """

        # Fast evaluation for the most basic scenario
        if timeStep is None and len(variables) == 0:
            np.savetxt(
                fileName,
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

        if timeStep is None:
            timePoints = self.time
        else:
            timePoints = np.arange(self.tInitial, self.tFinal, timeStep)

        exportedMatrix = [timePoints]
        exportedHeader = "Time (s)"

        # Loop through variables, get points and names (for the header)
        for variable in variables:
            if variable in self.__dict__.keys():
                variableFunction = self.__dict__[variable]
            # Deal with decorated Flight methods
            else:
                try:
                    obj = getattr(self.__class__, variable)
                    variableFunction = obj.__get__(self, self.__class__)
                except AttributeError:
                    raise AttributeError(
                        "Variable '{}' not found in Flight class".format(variable)
                    )
            variablePoints = variableFunction(timePoints)
            exportedMatrix += [variablePoints]
            exportedHeader += ", " + variableFunction.__outputs__[0]

        exportedMatrix = np.array(exportedMatrix).T  # Fix matrix orientation

        np.savetxt(
            fileName,
            exportedMatrix,
            fmt="%.6f",
            delimiter=",",
            header=exportedHeader,
            encoding="utf-8",
        )

        return

    def exportKML(
        self,
        fileName="trajectory.kml",
        timeStep=None,
        extrude=True,
        color="641400F0",
        altitudeMode="absolute",
    ):
        """Exports flight data to a .kml file, which can be opened with Google Earth to display the rocket's trajectory.

        Parameters
        ----------
        fileName : string
            The file name or path of the exported file. Example: flight_data.csv.
            Do not use forbidden characters, such as '/' in Linux/Unix and
            '<, >, :, ", /, \\, | ?, *' in Windows.
        timeStep : float, optional
            Time step desired for the data. If None, all integration time steps
            will be exported. Otherwise, linear interpolation is carried out to
            calculate values at the desired time steps. Example: 0.001.
        extrude: bool, optional
            To be used if you want to project the path over ground by using an
            extruded polygon. In case False only the linestring containing the
            flight path will be created. Default is True.
        color : str, optional
            Color of your trajectory path, need to be used in specific kml format.
            Refer to http://www.zonums.com/gmaps/kml_color/ for more info.
        altitudeMode: str
            Select elevation values format to be used on the kml file. Use
            'relativetoground' if you want use Above Ground Level elevation, or
            'absolute' if you want to parse elevation using Above Sea Level.
            Default is 'relativetoground'. Only works properly if the ground level is flat.
            Change to 'absolute' if the terrain is to irregular or contains mountains.
        Returns
        -------
        None
        """
        # Define time points vector
        if timeStep is None:
            timePoints = self.time
        else:
            timePoints = np.arange(self.tInitial, self.tFinal + timeStep, timeStep)
        # Open kml file with simplekml library
        kml = simplekml.Kml(open=1)
        trajectory = kml.newlinestring(name="Rocket Trajectory - Powered by RocketPy")
        coords = []
        if altitudeMode == "relativetoground":
            # In this mode the elevation data will be the Above Ground Level
            # elevation. Only works properly if the ground level is similar to
            # a plane, i.e. it might not work well if the terrain has mountains
            for t in timePoints:
                coords.append(
                    (
                        self.longitude(t),
                        self.latitude(t),
                        self.z(t) - self.env.elevation,
                    )
                )
            trajectory.coords = coords
            trajectory.altitudemode = simplekml.AltitudeMode.relativetoground
        else:  # altitudeMode == 'absolute'
            # In this case the elevation data will be the Above Sea Level elevation
            # Ensure you use the correct value on self.env.elevation, otherwise
            # the trajectory path can be offset from ground
            for t in timePoints:
                coords.append((self.longitude(t), self.latitude(t), self.z(t)))
            trajectory.coords = coords
            trajectory.altitudemode = simplekml.AltitudeMode.absolute
        # Modify style of trajectory linestring
        trajectory.style.linestyle.color = color
        trajectory.style.polystyle.color = color
        if extrude:
            trajectory.extrude = 1
        # Save the KML
        kml.save(fileName)
        print("File ", fileName, " saved with success!")

        return None

    def allInfo(self):
        """Prints out all data and graphs available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Print a summary of data about the flight
        self.info()

        self.plots.all()

        return None

    def animate(self, start=0, stop=None, fps=12, speed=4, elev=None, azim=None):
        """Plays an animation of the flight. Not implemented yet. Only
        kinda works outside notebook.
        """
        # Set up stopping time
        stop = self.tFinal if stop is None else stop
        # Speed = 4 makes it almost real time - matplotlib is way to slow
        # Set up graph
        fig = plt.figure(figsize=(18, 15))
        axes = fig.gca(projection="3d")
        # Initialize time
        timeRange = np.linspace(start, stop, fps * (stop - start))
        # Initialize first frame
        axes.set_title("Trajectory and Velocity Animation")
        axes.set_xlabel("X (m)")
        axes.set_ylabel("Y (m)")
        axes.set_zlabel("Z (m)")
        axes.view_init(elev, azim)
        R = axes.quiver(0, 0, 0, 0, 0, 0, color="r", label="Rocket")
        V = axes.quiver(0, 0, 0, 0, 0, 0, color="g", label="Velocity")
        W = axes.quiver(0, 0, 0, 0, 0, 0, color="b", label="Wind")
        S = axes.quiver(0, 0, 0, 0, 0, 0, color="black", label="Freestream")
        axes.legend()
        # Animate
        for t in timeRange:
            R.remove()
            V.remove()
            W.remove()
            S.remove()
            # Calculate rocket position
            Rx, Ry, Rz = self.x(t), self.y(t), self.z(t)
            Ru = 1 * (2 * (self.e1(t) * self.e3(t) + self.e0(t) * self.e2(t)))
            Rv = 1 * (2 * (self.e2(t) * self.e3(t) - self.e0(t) * self.e1(t)))
            Rw = 1 * (1 - 2 * (self.e1(t) ** 2 + self.e2(t) ** 2))
            # Calculate rocket Mach number
            Vx = self.vx(t) / 340.40
            Vy = self.vy(t) / 340.40
            Vz = self.vz(t) / 340.40
            # Calculate wind Mach Number
            z = self.z(t)
            Wx = self.env.windVelocityX(z) / 20
            Wy = self.env.windVelocityY(z) / 20
            # Calculate freestream Mach Number
            Sx = self.streamVelocityX(t) / 340.40
            Sy = self.streamVelocityY(t) / 340.40
            Sz = self.streamVelocityZ(t) / 340.40
            # Plot Quivers
            R = axes.quiver(Rx, Ry, Rz, Ru, Rv, Rw, color="r")
            V = axes.quiver(Rx, Ry, Rz, -Vx, -Vy, -Vz, color="g")
            W = axes.quiver(Rx - Vx, Ry - Vy, Rz - Vz, Wx, Wy, 0, color="b")
            S = axes.quiver(Rx, Ry, Rz, Sx, Sy, Sz, color="black")
            # Adjust axis
            axes.set_xlim(Rx - 1, Rx + 1)
            axes.set_ylim(Ry - 1, Ry + 1)
            axes.set_zlim(Rz - 1, Rz + 1)
            # plt.pause(1/(fps*speed))
            try:
                plt.pause(1 / (fps * speed))
            except:
                time.sleep(1 / (fps * speed))

    def timeIterator(self, nodeList):
        i = 0
        while i < len(nodeList) - 1:
            yield i, nodeList[i]
            i += 1

    class FlightPhases:
        def __init__(self, init_list=[]):
            self.list = init_list[:]

        def __getitem__(self, index):
            return self.list[index]

        def __len__(self):
            return len(self.list)

        def __repr__(self):
            return str(self.list)

        def add(self, flightPhase, index=None):
            # Handle first phase
            if len(self.list) == 0:
                self.list.append(flightPhase)
            # Handle appending to last position
            elif index is None:
                # Check if new flight phase respects time
                previousPhase = self.list[-1]
                if flightPhase.t > previousPhase.t:
                    # All good! Add phase.
                    self.list.append(flightPhase)
                elif flightPhase.t == previousPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting together with the one preceding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    flightPhase.t += 1e-7
                    self.add(flightPhase)
                elif flightPhase.t < previousPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting before the one preceding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    print("Or by having a negative parachute lag.")
                    self.add(flightPhase, -2)
            # Handle inserting into intermediary position
            else:
                # Check if new flight phase respects time
                nextPhase = self.list[index]
                previousPhase = self.list[index - 1]
                if previousPhase.t < flightPhase.t < nextPhase.t:
                    # All good! Add phase.
                    self.list.insert(index, flightPhase)
                elif flightPhase.t < previousPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting before the one preceding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    print("Or by having a negative parachute lag.")
                    self.add(flightPhase, index - 1)
                elif flightPhase.t == previousPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting together with the one preceding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    flightPhase.t += 1e-7
                    self.add(flightPhase, index)
                elif flightPhase.t == nextPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting together with the one proceeding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    flightPhase.t += 1e-7
                    self.add(flightPhase, index + 1)
                elif flightPhase.t > nextPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting after the one proceeding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    self.add(flightPhase, index + 1)

        def addPhase(self, t, derivatives=None, callback=[], clear=True, index=None):
            self.add(self.FlightPhase(t, derivatives, callback, clear), index)

        def flushAfter(self, index):
            del self.list[index + 1 :]

        class FlightPhase:
            def __init__(self, t, derivative=None, callbacks=[], clear=True):
                self.t = t
                self.derivative = derivative
                self.callbacks = callbacks[:]
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

        def add(self, timeNode):
            self.list.append(timeNode)

        def addNode(self, t, parachutes, callbacks):
            self.list.append(self.TimeNode(t, parachutes, callbacks))

        def addParachutes(self, parachutes, t_init, t_end):
            # Iterate over parachutes
            for parachute in parachutes:
                # Calculate start of sampling time nodes
                pcDt = 1 / parachute.samplingRate
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

        def flushAfter(self, index):
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
