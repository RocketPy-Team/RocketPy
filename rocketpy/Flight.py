# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"

import re
import math
import bisect
import warnings
import time
from datetime import datetime, timedelta
from inspect import signature, getsourcelines
from collections import namedtuple

import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from .Function import Function

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
            Wheater to terminate simulation when rocket reaches apogee.
        Flight.solver : scipy.integrate.LSODA
            Scipy LSODA integration scheme.

        State Space Vector Definition:
        (Only available after Flight.postProcess has been called.)
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
            Z coordinate, or altitute, of the center of mass of the
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
        (Only available after Flight.postProcess has been called.)
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
            Rocket's attiude vector, or the vector that points
            in the rocket's axis of symmetry, component in the X
            direction (East) as a function of time.
            Can be called or accessed as array.
        Flight.attitudeVectorY : Function
            Rocket's attiude vector, or the vector that points
            in the rocket's axis of symmetry, component in the Y
            direction (East) as a function of time.
            Can be called or accessed as array.
        Flight.attitudeVectorZ : Function
            Rocket's attiude vector, or the vector that points
            in the rocket's axis of symmetry, component in the Z
            direction (East) as a function of time.
            Can be called or accessed as array.
        Flight.attitudeAngle : Function
            Rocket's attiude angle, or the angle that the
            rocket's axis of symmetry makes with the horizontal (North-East)
            plane. Measured in degrees and expressed as a function
            of time. Can be called or accessed as array.
        Flight.lateralAttitudeAngle : Function
            Rocket's lateral attiude angle, or the angle that the
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
            Resultant momentum perpendicular to rockets axis due to
            aerodynamic forces and excentricity as a function of time.
            Units in N*m. 
            Expressed as a function of time. Can be called or accessed
            as array.
            Direction 1 is in the rocket's body axis and points perpendicular
            to the rocket's axis of cylindrical symmetry.
        Flight.M2 : Function
            Resultant momentum perpendicular to rockets axis due to
            aerodynamic forces and excentricity as a function of time.
            Units in N*m.
            Expressed as a function of time. Can be called or accessed
            as array.     
            Direction 2 is in the rocket's body axis and points perpendicular
            to the rocket's axis of cylindrical symmetry and direction 1.           
        Flight.M3 : Function
            Resultant momentum in rockets axis due to aerodynamic
            forces and excentricity as a function of time. Units in N*m.
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
            Rocket's Mach number defined as it's freestream speed
            devided by the speed of sound at its altitude. Expressed
            as a function of time. Can be called or accessed as array.
        Flight.maxMachNumber : float
            Rocket's maximum Mach number experienced during flight.
        Flight.maxMachNumberTime : float
            Time at which the rocket experiences the maximum Mach number.
        Flight.ReynoldsNumber : Function
            Rocket's Reynolds number, using it's diameter as reference
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
        
        Fin Flutter Analysis:
        Flight.flutterMachNumber: Function
            The freestream velocity at which begins flutter phenomenon in
            rocket's fins. It's expressed as a function of the air pressure
            experienced  for the rocket. Can be called or accessed as array.
        Flight.difference: Function
            Difference between flutterMachNumber and machNumber, as a function of time.
        Flight.safetyFactor: Function
            Ratio between the flutterMachNumber and machNumber, as a function of time.     
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
        initialSolution : array, optional
            Initial solution array to be used. Format is
            initialSolution = [self.tInitial,
                                xInit, yInit, zInit,
                                vxInit, vyInit, vzInit,
                                e0Init, e1Init, e2Init, e3Init,
                                w1Init, w2Init, w3Init].
            If None, the initial solution will start with all null values,
            except for the euler parameters which will be calculated based
            on given values of inclination and heading. Default is None.
        terminateOnApogee : boolean, optioanal
            Wheater to terminate simulation when rocket reaches apogee.
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

        # Modifying Rail Length for a better out of rail condition
        upperRButton = max(self.rocket.railButtons[0])
        lowerRButton = min(self.rocket.railButtons[0])
        nozzle = self.rocket.distanceRocketNozzle
        self.effective1RL = self.env.rL - abs(nozzle - upperRButton)
        self.effective2RL = self.env.rL - abs(nozzle - lowerRButton)

        # Flight initialization
        # Initialize solution monitors
        self.outOfRailTime = 0
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
        # Intialize solver monitors
        self.functionEvaluations = []
        self.functionEvaluationsPerTimeStep = []
        self.timeSteps = []
        # Initialize solution state
        self.solution = []
        if self.initialSolution is None:
            # Initialize time and state variables
            self.tInitial = 0
            xInit, yInit, zInit = 0, 0, self.env.elevation
            vxInit, vyInit, vzInit = 0, 0, 0
            w1Init, w2Init, w3Init = 0, 0, 0
            # Initialize attitude
            psiInit = -heading * (np.pi / 180)  # Precession / Heading Angle
            thetaInit = (inclination - 90) * (np.pi / 180)  # Nutation Angle
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
        self.tInitial = self.initialSolution[0]
        self.solution.append(self.initialSolution)
        self.t = self.solution[-1][0]
        self.y = self.solution[-1][1:]

        # Calculate normal and lateral surface wind
        windU = self.env.windVelocityX(self.env.elevation)
        windV = self.env.windVelocityY(self.env.elevation)
        headingRad = heading * np.pi / 180
        self.frontalSurfaceWind = windU * np.sin(headingRad) + windV * np.cos(
            headingRad
        )
        self.lateralSurfaceWind = -windU * np.cos(headingRad) + windV * np.sin(
            headingRad
        )

        # Create knonw flight phases
        self.flightPhases = FlightPhases()
        self.flightPhases.addPhase(self.tInitial, self.uDotRail1, clear=False)
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
                y0=self.y,
                t_bound=phase.timeBound,
                min_step=self.minTimeStep,
                max_step=self.maxTimeStep,
                rtol=self.rtol,
                atol=self.atol,
            )
            # print('\n\tSolver Initialization Details')
            # print('\tInitial Time: ', phase.t)
            # print('\tInitial State: ', self.y)
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
                    pressure = self.env.pressure.getValueOpt(self.y[2])
                    parachute.cleanPressureSignal.append([node.t, pressure])
                    # Calculate and save noise
                    noise = parachute.noiseFunction()
                    parachute.noiseSignal.append([node.t, noise])
                    parachute.noisyPressureSignal.append([node.t, pressure + noise])
                    if parachute.trigger(pressure + noise, self.y):
                        # print('\nEVENT DETECTED')
                        # print('Parachute Triggered')
                        # print('Name: ', parachute.name, ' | Lag: ', parachute.lag)
                        # Remove parachute from flight parachutes
                        self.parachutes.remove(parachute)
                        # Create flight phase for time after detection and before inflation
                        self.flightPhases.addPhase(
                            node.t, phase.derivative, clear=True, index=phase_index + 1
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
                    self.y = phase.solver.y
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
                        self.y[0] ** 2
                        + self.y[1] ** 2
                        + (self.y[2] - self.env.elevation) ** 2
                        >= self.effective1RL ** 2
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
                            - self.effective1RL ** 2
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
                            - self.effective1RL ** 2
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
                        b = float((3 * y1 - yp1 * D - 2 * c * D - 3 * d) / (D ** 2))
                        a = float(-(2 * y1 - yp1 * D - c * D - 2 * d) / (D ** 3))
                        # Find roots
                        d0 = b ** 2 - 3 * a * c
                        d1 = 2 * b ** 3 - 9 * a * b * c + 27 * d * a ** 2
                        c1 = ((d1 + (d1 ** 2 - 4 * d0 ** 3) ** (0.5)) / 2) ** (1 / 3)
                        t_roots = []
                        for k in [0, 1, 2]:
                            c2 = c1 * (-1 / 2 + 1j * (3 ** 0.5) / 2) ** k
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
                        # Determine final state when upper button is going out of rail
                        self.t = valid_t_root[0] + self.solution[-2][0]
                        interpolator = phase.solver.dense_output()
                        self.y = interpolator(self.t)
                        self.solution[-1] = [self.t, *self.y]
                        self.outOfRailTime = self.t
                        self.outOfRailState = self.y
                        self.outOfRailVelocity = (
                            self.y[3] ** 2 + self.y[4] ** 2 + self.y[5] ** 2
                        ) ** (0.5)
                        # Create new flight phase
                        self.flightPhases.addPhase(
                            self.t, self.uDot, index=phase_index + 1
                        )
                        # Prepare to leave loops and start new flight phase
                        phase.timeNodes.flushAfter(node_index)
                        phase.timeNodes.addNode(self.t, [], [])
                        phase.solver.status = "finished"
                    
                    # Check for apogee event
                    if len(self.apogeeState) == 1 and self.y[5] < 0:
                        # print('\nPASSIVE EVENT DETECTED')
                        # print('Rocket Has Reached Apogee!')
                        # Apogee reported
                        # Assume linear vz(t) to detect when vz = 0
                        vz0 = self.solution[-2][6]
                        t0 = self.solution[-2][0]
                        vz1 = self.solution[-1][6]
                        t1 = self.solution[-1][0]
                        t_root = -(t1 - t0) * vz0 / (vz1 - vz0) + t0
                        # Fecth state at t_root
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
                    if self.y[2] < self.env.elevation:
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
                        b = float((3 * y1 - yp1 * D - 2 * c * D - 3 * d) / (D ** 2))
                        a = float(-(2 * y1 - yp1 * D - c * D - 2 * d) / (D ** 3))
                        # Find roots
                        d0 = b ** 2 - 3 * a * c
                        d1 = 2 * b ** 3 - 9 * a * b * c + 27 * d * a ** 2
                        c1 = ((d1 + (d1 ** 2 - 4 * d0 ** 3) ** (0.5)) / 2) ** (1 / 3)
                        t_roots = []
                        for k in [0, 1, 2]:
                            c2 = c1 * (-1 / 2 + 1j * (3 ** 0.5) / 2) ** k
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
                        self.y = interpolator(self.t)
                        # Roll back solution
                        self.solution[-1] = [self.t, *self.y]
                        # Save impact state
                        self.impactState = self.y
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
                                        self.y = overshootableNode.y
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

        # Get freestrean speed
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
        R3 = -0.5 * rho * (freestreamSpeed ** 2) * self.rocket.area * (dragCoeff)

        # Calculate Linear acceleration
        a3 = (R3 + Thrust) / M - (e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2) * self.env.g
        if a3 > 0:
            ax = 2 * (e1 * e3 + e0 * e2) * a3
            ay = 2 * (e2 * e3 - e0 * e1) * a3
            az = (1 - 2 * (e1 ** 2 + e2 ** 2)) * a3
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
        return self.uDot(t, u, postProcessing= postProcessing)

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
        # Determine current behaviour
        if t < self.rocket.motor.burnOutTime:
            # Motor burning
            # Retrieve important motor quantities
            # Inertias
            Tz = self.rocket.motor.inertiaZ.getValueOpt(t)
            Ti = self.rocket.motor.inertiaI.getValueOpt(t)
            TzDot = self.rocket.motor.inertiaZDot.getValueOpt(t)
            TiDot = self.rocket.motor.inertiaIDot.getValueOpt(t)
            # Mass
            MtDot = self.rocket.motor.massDot.getValueOpt(t)
            Mt = self.rocket.motor.mass.getValueOpt(t)
            # Thrust
            Thrust = self.rocket.motor.thrust.getValueOpt(t)
            # Off center moment
            M1 += self.rocket.thrustExcentricityX * Thrust
            M2 -= self.rocket.thrustExcentricityY * Thrust
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
        Rz = self.rocket.inertiaZ
        Ri = self.rocket.inertiaI
        # Mass
        Mr = self.rocket.mass
        M = Mt + Mr
        mu = (Mt * Mr) / (Mt + Mr)
        # Geometry
        b = -self.rocket.distanceRocketPropellant
        c = -self.rocket.distanceRocketNozzle
        a = b * Mt / M
        rN = self.rocket.motor.nozzleRadius
        # Prepare transformation matrix
        a11 = 1 - 2 * (e2 ** 2 + e3 ** 2)
        a12 = 2 * (e1 * e2 - e0 * e3)
        a13 = 2 * (e1 * e3 + e0 * e2)
        a21 = 2 * (e1 * e2 + e0 * e3)
        a22 = 1 - 2 * (e1 ** 2 + e3 ** 2)
        a23 = 2 * (e2 * e3 - e0 * e1)
        a31 = 2 * (e1 * e3 - e0 * e2)
        a32 = 2 * (e2 * e3 + e0 * e1)
        a33 = 1 - 2 * (e1 ** 2 + e2 ** 2)
        # Transformation matrix: (123) -> (XYZ)
        K = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
        # Transformation matrix: (XYZ) -> (123) or K transpose
        Kt = [[a11, a21, a31], [a12, a22, a32], [a13, a23, a33]]

        # Calculate Forces and Moments
        # Get freestrean speed
        windVelocityX = self.env.windVelocityX.getValueOpt(z)
        windVelocityY = self.env.windVelocityY.getValueOpt(z)
        freestreamSpeed = (
            (windVelocityX - vx) ** 2 + (windVelocityY - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        freestreamMach = freestreamSpeed / self.env.speedOfSound.getValueOpt(z)

        # Determine aerodynamics forces
        # Determine Drag Force
        if t > self.rocket.motor.burnOutTime:
            dragCoeff = self.rocket.powerOnDrag.getValueOpt(freestreamMach)
        else:
            dragCoeff = self.rocket.powerOffDrag.getValueOpt(freestreamMach)
        rho = self.env.density.getValueOpt(z)
        R3 = -0.5 * rho * (freestreamSpeed ** 2) * self.rocket.area * (dragCoeff)
        # Off center moment
        M1 += self.rocket.cpExcentricityY * R3
        M2 -= self.rocket.cpExcentricityX * R3
        # Get rocket velocity in body frame
        vxB = a11 * vx + a21 * vy + a31 * vz
        vyB = a12 * vx + a22 * vy + a32 * vz
        vzB = a13 * vx + a23 * vy + a33 * vz
        # Calculate lift and moment for each component of the rocket
        for aerodynamicSurface in self.rocket.aerodynamicSurfaces:
            compCp = aerodynamicSurface[0][2]
            clalpha = aerodynamicSurface[1]
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
                compStreamVxB ** 2 + compStreamVyB ** 2 + compStreamVzB ** 2
            ) ** 0.5
            # Component attack angle and lift force
            compAttackAngle = 0
            compLift, compLiftXB, compLiftYB = 0, 0, 0
            if compStreamVxB ** 2 + compStreamVyB ** 2 != 0:
                # Normalize component stream velocity in body frame
                compStreamVzBn = compStreamVzB / compStreamSpeed
                if -1 * compStreamVzBn < 1:
                    compAttackAngle = np.arccos(-compStreamVzBn)
                    # Component lift force magnitude
                    compLift = (
                        0.5
                        * rho
                        * (compStreamSpeed ** 2)
                        * self.rocket.area
                        * clalpha
                        * compAttackAngle
                    )
                    # Component lift force components
                    liftDirNorm = (compStreamVxB ** 2 + compStreamVyB ** 2) ** 0.5
                    compLiftXB = compLift * (compStreamVxB / liftDirNorm)
                    compLiftYB = compLift * (compStreamVyB / liftDirNorm)
                    # Add to total lift force
                    R1 += compLiftXB
                    R2 += compLiftYB
                    # Add to total moment
                    M1 -= (compCp + a) * compLiftYB
                    M2 += (compCp + a) * compLiftXB
        # Calculate derivatives
        # Angular acceleration
        alpha1 = (
            M1
            - (
                omega2 * omega3 * (Rz + Tz - Ri - Ti - mu * b ** 2)
                + omega1
                * (
                    (TiDot + MtDot * (Mr - 1) * (b / M) ** 2)
                    - MtDot * ((rN / 2) ** 2 + (c - b * mu / Mr) ** 2)
                )
            )
        ) / (Ri + Ti + mu * b ** 2)
        alpha2 = (
            M2
            - (
                omega1 * omega3 * (Ri + Ti + mu * b ** 2 - Rz - Tz)
                + omega2
                * (
                    (TiDot + MtDot * (Mr - 1) * (b / M) ** 2)
                    - MtDot * ((rN / 2) ** 2 + (c - b * mu / Mr) ** 2)
                )
            )
        ) / (Ri + Ti + mu * b ** 2)
        alpha3 = (M3 - omega3 * (TzDot - MtDot * (rN ** 2) / 2)) / (Rz + Tz)
        # Euler parameters derivative
        e0Dot = 0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3)
        e1Dot = 0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3)
        e2Dot = 0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3)
        e3Dot = 0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2)

        # Linear acceleration
        L = [
            (R1 - b * Mt * (omega2 ** 2 + omega3 ** 2) - 2 * c * MtDot * omega2) / M,
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
        when rocket is flying under parachute. A 3 DOF aproximation is
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
        ma = ka * rho * (4 / 3) * np.pi * R ** 3
        mp = self.rocket.mass
        eta = 1
        Rdot = (6 * R * (1 - eta) / (1.2 ** 6)) * (
            (1 - eta) * t ** 5 + eta * (to ** 3) * (t ** 2)
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
        pseudoD = -0.5 * rho * CdS * freestreamSpeed - ka * rho * 4 * np.pi * (R ** 2) * Rdot
        Dx = pseudoD * freestreamX
        Dy = pseudoD * freestreamY
        Dz = pseudoD * freestreamZ
        ax = Dx / (mp + ma)
        ay = Dy / (mp + ma)
        az = (Dz - 9.8 * mp) / (mp + ma)

        if postProcessing:
            # Dynamics variables
            self.R1.append([t, Dx])
            self.R2.append([t, Dy])
            self.R3.append([t, Dz])
            self.M1.append([t, 0])
            self.M2.append([t, 0])
            self.M3.append([t, 0])
            # Atmospheric Conditions
            self.windVelocityX.append([t, self.env.windVelocityX(z)])
            self.windVelocityY.append([t, self.env.windVelocityY(z)])
            self.density.append([t, self.env.density(z)])
            self.dynamicViscosity.append([t, self.env.dynamicViscosity(z)])
            self.pressure.append([t, self.env.pressure(z)])
            self.speedOfSound.append([t, self.env.speedOfSound(z)])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def postProcess(self):
        """Post-process all Flight information produced during
        simulation. Includes the calculation of maximum values,
        calculation of secundary values such as energy and conversion
        of lists to Function objects to facilitate plotting.

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Process first type of outputs - state vector
        # Transform solution array into Functions
        sol = np.array(self.solution)
        self.x = Function(
            sol[:, [0, 1]], "Time (s)", "X (m)", "spline", extrapolation="natural"
        )
        self.y = Function(
            sol[:, [0, 2]], "Time (s)", "Y (m)", "spline", extrapolation="natural"
        )
        self.z = Function(
            sol[:, [0, 3]], "Time (s)", "Z (m)", "spline", extrapolation="natural"
        )
        self.vx = Function(
            sol[:, [0, 4]], "Time (s)", "Vx (m/s)", "spline", extrapolation="natural"
        )
        self.vy = Function(
            sol[:, [0, 5]], "Time (s)", "Vy (m/s)", "spline", extrapolation="natural"
        )
        self.vz = Function(
            sol[:, [0, 6]], "Time (s)", "Vz (m/s)", "spline", extrapolation="natural"
        )
        self.e0 = Function(
            sol[:, [0, 7]], "Time (s)", "e0", "spline", extrapolation="natural"
        )
        self.e1 = Function(
            sol[:, [0, 8]], "Time (s)", "e1", "spline", extrapolation="natural"
        )
        self.e2 = Function(
            sol[:, [0, 9]], "Time (s)", "e2", "spline", extrapolation="natural"
        )
        self.e3 = Function(
            sol[:, [0, 10]], "Time (s)", "e3", "spline", extrapolation="natural"
        )
        self.w1 = Function(
            sol[:, [0, 11]], "Time (s)", "ω1 (rad/s)", "spline", extrapolation="natural"
        )
        self.w2 = Function(
            sol[:, [0, 12]], "Time (s)", "ω2 (rad/s)", "spline", extrapolation="natural"
        )
        self.w3 = Function(
            sol[:, [0, 13]], "Time (s)", "ω3 (rad/s)", "spline", extrapolation="natural"
        )

        # Process second type of outputs - accelerations
        # Initialize acceleration arrays
        self.ax, self.ay, self.az = [], [], []
        self.alpha1, self.alpha2, self.alpha3 = [], [], []
        # Go throught each time step and calculate accelerations
        # Get fligth phases
        for phase_index, phase in self.timeIterator(self.flightPhases):
            initTime = phase.t
            finalTime = self.flightPhases[phase_index + 1].t
            currentDerivative = phase.derivative
            # Call callback functions
            for callback in phase.callbacks:
                callback(self)
            # Loop through time steps in flight phase
            for step in self.solution:  # Can be optmized
                if initTime < step[0] <= finalTime:
                    # Get derivatives
                    uDot = currentDerivative(step[0], step[1:])
                    # Get accelerations
                    ax, ay, az = uDot[3:6]
                    alpha1, alpha2, alpha3 = uDot[10:]
                    # Save accelerations
                    self.ax.append([step[0], ax])
                    self.ay.append([step[0], ay])
                    self.az.append([step[0], az])
                    self.alpha1.append([step[0], alpha1])
                    self.alpha2.append([step[0], alpha2])
                    self.alpha3.append([step[0], alpha3])
        # Convert accelerations to functions
        self.ax = Function(self.ax, "Time (s)", "Ax (m/s2)", "spline")
        self.ay = Function(self.ay, "Time (s)", "Ay (m/s2)", "spline")
        self.az = Function(self.az, "Time (s)", "Az (m/s2)", "spline")
        self.alpha1 = Function(self.alpha1, "Time (s)", "α1 (rad/s2)", "spline")
        self.alpha2 = Function(self.alpha2, "Time (s)", "α2 (rad/s2)", "spline")
        self.alpha3 = Function(self.alpha3, "Time (s)", "α3 (rad/s2)", "spline")

        # Process third type of outputs - temporary values calculated during integration
        # Initialize force and atmospheric arrays
        self.R1, self.R2, self.R3, self.M1, self.M2, self.M3 = [], [], [], [], [], []
        self.pressure, self.density, self.dynamicViscosity, self.speedOfSound = (
            [],
            [],
            [],
            [],
        )
        self.windVelocityX, self.windVelocityY = [], []
        # Go throught each time step and calculate forces and atmospheric values
        # Get fligth phases
        for phase_index, phase in self.timeIterator(self.flightPhases):
            initTime = phase.t
            finalTime = self.flightPhases[phase_index + 1].t
            currentDerivative = phase.derivative
            # Call callback functions
            for callback in phase.callbacks:
                callback(self)
            # Loop through time steps in flight phase
            for step in self.solution:  # Can be optmized
                if initTime < step[0] <= finalTime or (initTime == 0 and step[0] == 0):
                    # Call derivatives in post processing mode
                    uDot = currentDerivative(step[0], step[1:], postProcessing=True)
        # Convert forces and atmospheric arrays to functions
        self.R1 = Function(self.R1, "Time (s)", "R1 (N)", "spline")
        self.R2 = Function(self.R2, "Time (s)", "R2 (N)", "spline")
        self.R3 = Function(self.R3, "Time (s)", "R3 (N)", "spline")
        self.M1 = Function(self.M1, "Time (s)", "M1 (Nm)", "spline")
        self.M2 = Function(self.M2, "Time (s)", "M2 (Nm)", "spline")
        self.M3 = Function(self.M3, "Time (s)", "M3 (Nm)", "spline")
        self.windVelocityX = Function(
            self.windVelocityX, "Time (s)", "Wind Velocity X (East) (m/s)", "spline"
        )
        self.windVelocityY = Function(
            self.windVelocityY, "Time (s)", "Wind Velocity Y (North) (m/s)", "spline"
        )
        self.density = Function(self.density, "Time (s)", "Density (kg/m³)", "spline")
        self.pressure = Function(self.pressure, "Time (s)", "Pressure (Pa)", "spline")
        self.dynamicViscosity = Function(
            self.dynamicViscosity, "Time (s)", "Dynamic Viscosity (Pa s)", "spline"
        )
        self.speedOfSound = Function(
            self.speedOfSound, "Time (s)", "Speed of Sound (m/s)", "spline"
        )

        # Process fourth type of output - values calculated from previous outputs

        # Kinematicss functions and values
        # Velocity Magnitude
        self.speed = (self.vx ** 2 + self.vy ** 2 + self.vz ** 2) ** 0.5
        self.speed.setOutputs("Speed - Velocity Magnitude (m/s)")
        maxSpeedTimeIndex = np.argmax(self.speed[:, 1])
        self.maxSpeed = self.speed[maxSpeedTimeIndex, 1]
        self.maxSpeedTime = self.speed[maxSpeedTimeIndex, 0]
        # Acceleration
        self.acceleration = (self.ax ** 2 + self.ay ** 2 + self.az ** 2) ** 0.5
        self.acceleration.setOutputs("Acceleration Magnitude (m/s²)")
        maxAccelerationTimeIndex = np.argmax(self.acceleration[:, 1])
        self.maxAcceleration = self.acceleration[maxAccelerationTimeIndex, 1]
        self.maxAccelerationTime = self.acceleration[maxAccelerationTimeIndex, 0]
        # Path Angle
        self.horizontalSpeed = (self.vx ** 2 + self.vy ** 2) ** 0.5
        pathAngle = (180 / np.pi) * np.arctan2(
            self.vz[:, 1], self.horizontalSpeed[:, 1]
        )
        pathAngle = np.column_stack([self.vz[:, 0], pathAngle])
        self.pathAngle = Function(pathAngle, "Time (s)", "Path Angle (°)")
        # Attitude Angle
        self.attitudeVectorX = 2 * (self.e1 * self.e3 + self.e0 * self.e2)  # a13
        self.attitudeVectorY = 2 * (self.e2 * self.e3 - self.e0 * self.e1)  # a23
        self.attitudeVectorZ = 1 - 2 * (self.e1 ** 2 + self.e2 ** 2)  # a33
        horizontalAttitudeProj = (
            self.attitudeVectorX ** 2 + self.attitudeVectorY ** 2
        ) ** 0.5
        attitudeAngle = (180 / np.pi) * np.arctan2(
            self.attitudeVectorZ[:, 1], horizontalAttitudeProj[:, 1]
        )
        attitudeAngle = np.column_stack([self.attitudeVectorZ[:, 0], attitudeAngle])
        self.attitudeAngle = Function(attitudeAngle, "Time (s)", "Attitude Angle (°)")
        # Lateral Attitude Angle
        lateralVectorAngle = (np.pi / 180) * (self.heading - 90)
        lateralVectorX = np.sin(lateralVectorAngle)
        lateralVectorY = np.cos(lateralVectorAngle)
        attitudeLateralProj = (
            lateralVectorX * self.attitudeVectorX[:, 1]
            + lateralVectorY * self.attitudeVectorY[:, 1]
        )
        attitudeLateralProjX = attitudeLateralProj * lateralVectorX
        attitudeLateralProjY = attitudeLateralProj * lateralVectorY
        attiutdeLateralPlaneProjX = self.attitudeVectorX[:, 1] - attitudeLateralProjX
        attiutdeLateralPlaneProjY = self.attitudeVectorY[:, 1] - attitudeLateralProjY
        attiutdeLateralPlaneProjZ = self.attitudeVectorZ[:, 1]
        attiutdeLateralPlaneProj = (
            attiutdeLateralPlaneProjX ** 2
            + attiutdeLateralPlaneProjY ** 2
            + attiutdeLateralPlaneProjZ ** 2
        ) ** 0.5
        lateralAttitudeAngle = (180 / np.pi) * np.arctan2(
            attitudeLateralProj, attiutdeLateralPlaneProj
        )
        lateralAttitudeAngle = np.column_stack(
            [self.attitudeVectorZ[:, 0], lateralAttitudeAngle]
        )
        self.lateralAttitudeAngle = Function(
            lateralAttitudeAngle, "Time (s)", "Lateral Attitude Angle (°)"
        )
        # Euler Angles
        psi = (180 / np.pi) * (
            np.arctan2(self.e3[:, 1], self.e0[:, 1])
            + np.arctan2(-self.e2[:, 1], -self.e1[:, 1])
        )  # Precession angle
        psi = np.column_stack([self.e1[:, 0], psi])  # Precession angle
        self.psi = Function(psi, "Time (s)", "Precession Angle - ψ (°)")

        phi = (180 / np.pi) * (
            np.arctan2(self.e3[:, 1], self.e0[:, 1])
            - np.arctan2(-self.e2[:, 1], -self.e1[:, 1])
        )  # Spin angle
        phi = np.column_stack([self.e1[:, 0], phi])  # Spin angle
        self.phi = Function(phi, "Time (s)", "Spin Angle - φ (°)")

        theta = (
            (180 / np.pi)
            * 2
            * np.arcsin(-((self.e1[:, 1] ** 2 + self.e2[:, 1] ** 2) ** 0.5))
        )  # Nutation angle
        theta = np.column_stack([self.e1[:, 0], theta])  # Nutation angle
        self.theta = Function(theta, "Time (s)", "Nutation Angle - θ (°)")

        # Dynamics functions and variables
        # Rail Button Forces
        alpha = self.rocket.railButtons.angularPosition * (
            np.pi / 180
        )  # Rail buttons angular position
        D1 = self.rocket.railButtons.distanceToCM[
            0
        ]  # Distance from Rail Button 1 (upper) to CM
        D2 = self.rocket.railButtons.distanceToCM[
            1
        ]  # Distance from Rail Button 2 (lower) to CM
        F11 = (self.R1 * D2 - self.M2) / (
            D1 + D2
        )  # Rail Button 1 force in the 1 direction
        F12 = (self.R2 * D2 + self.M1) / (
            D1 + D2
        )  # Rail Button 1 force in the 2 direction
        F21 = (self.R1 * D1 + self.M2) / (
            D1 + D2
        )  # Rail Button 2 force in the 1 direction
        F22 = (self.R2 * D1 - self.M1) / (
            D1 + D2
        )  # Rail Button 2 force in the 2 direction
        outOfRailTimeIndex = np.searchsorted(
            F11[:, 0], self.outOfRailTime
        )  # Find out of rail time index
        # F11 = F11[:outOfRailTimeIndex + 1, :] # Limit force calculation to when rocket is in rail
        # F12 = F12[:outOfRailTimeIndex + 1, :] # Limit force calculation to when rocket is in rail
        # F21 = F21[:outOfRailTimeIndex + 1, :] # Limit force calculation to when rocket is in rail
        # F22 = F22[:outOfRailTimeIndex + 1, :] # Limit force calculation to when rocket is in rail
        self.railButton1NormalForce = F11 * np.cos(alpha) + F12 * np.sin(alpha)
        self.railButton1NormalForce.setOutputs("Upper Rail Button Normal Force (N)")
        self.railButton1ShearForce = F11 * -np.sin(alpha) + F12 * np.cos(alpha)
        self.railButton1ShearForce.setOutputs("Upper Rail Button Shear Force (N)")
        self.railButton2NormalForce = F21 * np.cos(alpha) + F22 * np.sin(alpha)
        self.railButton2NormalForce.setOutputs("Lower Rail Button Normal Force (N)")
        self.railButton2ShearForce = F21 * -np.sin(alpha) + F22 * np.cos(alpha)
        self.railButton2ShearForce.setOutputs("Lower Rail Button Shear Force (N)")
        # Rail Button Maximum Forces
        self.maxRailButton1NormalForce = np.amax(
            self.railButton1NormalForce[:outOfRailTimeIndex]
        )
        self.maxRailButton1ShearForce = np.amax(
            self.railButton1ShearForce[:outOfRailTimeIndex]
        )
        self.maxRailButton2NormalForce = np.amax(
            self.railButton2NormalForce[:outOfRailTimeIndex]
        )
        self.maxRailButton2ShearForce = np.amax(
            self.railButton2ShearForce[:outOfRailTimeIndex]
        )
        # Aerodynamic Lift and Drag
        self.aerodynamicLift = (self.R1 ** 2 + self.R2 ** 2) ** 0.5
        self.aerodynamicLift.setOutputs("Aerodynamic Lift Force (N)")
        self.aerodynamicDrag = -1 * self.R3
        self.aerodynamicDrag.setOutputs("Aerodynamic Drag Force (N)")
        self.aerodynamicBendingMoment = (self.M1 ** 2 + self.M2 ** 2) ** 0.5
        self.aerodynamicBendingMoment.setOutputs("Aerodynamic Bending Moment (N m)")
        self.aerodynamicSpinMoment = self.M3
        self.aerodynamicSpinMoment.setOutputs("Aerodynamic Spin Moment (N m)")
        # Energy
        b = -self.rocket.distanceRocketPropellant
        totalMass = self.rocket.totalMass
        mu = self.rocket.reducedMass
        Rz = self.rocket.inertiaZ
        Ri = self.rocket.inertiaI
        Tz = self.rocket.motor.inertiaZ
        Ti = self.rocket.motor.inertiaI
        I1, I2, I3 = (Ri + Ti + mu * b ** 2), (Ri + Ti + mu * b ** 2), (Rz + Tz)
        # Redefine I1, I2 and I3 grid
        grid = self.vx[:, 0]
        I1 = Function(np.column_stack([grid, I1(grid)]), "Time (s)")
        I2 = Function(np.column_stack([grid, I2(grid)]), "Time (s)")
        I3 = Function(np.column_stack([grid, I3(grid)]), "Time (s)")
        # Redefine total mass grid
        totalMass = Function(np.column_stack([grid, totalMass(grid)]), "Time (s)")
        # Redefine thrust grid
        thrust = Function(
            np.column_stack([grid, self.rocket.motor.thrust(grid)]), "Time (s)"
        )
        # Get some nicknames
        vx, vy, vz = self.vx, self.vy, self.vz
        w1, w2, w3 = self.w1, self.w2, self.w3
        # Kinetic Energy
        self.rotationalEnergy = 0.5 * (I1 * w1 ** 2 + I2 * w2 ** 2 + I3 * w3 ** 2)
        self.rotationalEnergy.setOutputs("Rotational Kinetic Energy (J)")
        self.translationalEnergy = 0.5 * totalMass * (vx ** 2 + vy ** 2 + vz ** 2)
        self.translationalEnergy.setOutputs("Translational Kinetic Energy (J)")
        self.kineticEnergy = self.rotationalEnergy + self.translationalEnergy
        self.kineticEnergy.setOutputs("Kinetic Energy (J)")
        # Potential Energy
        self.potentialEnergy = totalMass * self.env.g * self.z
        self.potentialEnergy.setInputs("Time (s)")
        # Total Mechanical Energy
        self.totalEnergy = self.kineticEnergy + self.potentialEnergy
        self.totalEnergy.setOutputs("Total Mechanical Energy (J)")
        # Thrust Power
        self.thrustPower = thrust * self.speed
        self.thrustPower.setOutputs("Thrust Power (W)")
        # Drag Power
        self.dragPower = self.R3 * self.speed
        self.dragPower.setOutputs("Drag Power (W)")

        # Stability and Control variables
        # Angular velocities frequency response - Fourier Analysis
        # Omega 1 - w1
        Fs = 100.0
        # sampling rate
        Ts = 1.0 / Fs
        # sampling interval
        t = np.arange(1, self.tFinal, Ts)  # time vector
        y = self.w1(t)
        y -= np.mean(y)
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(n // 2)]  # one side frequency range
        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(n // 2)]
        omega1FrequencyResponse = np.column_stack([frq, abs(Y)])
        self.omega1FrequencyResponse = Function(
            omega1FrequencyResponse, "Frequency (Hz)", "Omega 1 Angle Fourier Amplitude"
        )
        # Omega 2 - w2
        Fs = 100.0
        # sampling rate
        Ts = 1.0 / Fs
        # sampling interval
        t = np.arange(1, self.tFinal, Ts)  # time vector
        y = self.w2(t)
        y -= np.mean(y)
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(n // 2)]  # one side frequency range
        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(n // 2)]
        omega2FrequencyResponse = np.column_stack([frq, abs(Y)])
        self.omega2FrequencyResponse = Function(
            omega2FrequencyResponse, "Frequency (Hz)", "Omega 2 Angle Fourier Amplitude"
        )
        # Omega 3 - w3
        Fs = 100.0
        # sampling rate
        Ts = 1.0 / Fs
        # sampling interval
        t = np.arange(1, self.tFinal, Ts)  # time vector
        y = self.w3(t)
        y -= np.mean(y)
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(n // 2)]  # one side frequency range
        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(n // 2)]
        omega3FrequencyResponse = np.column_stack([frq, abs(Y)])
        self.omega3FrequencyResponse = Function(
            omega3FrequencyResponse, "Frequency (Hz)", "Omega 3 Angle Fourier Amplitude"
        )
        # Attitude Frequency Response
        Fs = 100.0
        # sampling rate
        Ts = 1.0 / Fs
        # sampling interval
        t = np.arange(1, self.tFinal, Ts)  # time vector
        y = self.attitudeAngle(t)
        y -= np.mean(y)
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(n // 2)]  # one side frequency range
        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(n // 2)]
        attitudeFrequencyResponse = np.column_stack([frq, abs(Y)])
        self.attitudeFrequencyResponse = Function(
            attitudeFrequencyResponse,
            "Frequency (Hz)",
            "Attitude Angle Fourier Amplitude",
        )
        # Static Margin
        self.staticMargin = self.rocket.staticMargin

        # Fluid Mechanics variables
        # Freestream Velocity
        self.streamVelocityX = self.windVelocityX - self.vx
        self.streamVelocityX.setOutputs("Freestream Velocity X (m/s)")
        self.streamVelocityY = self.windVelocityY - self.vy
        self.streamVelocityY.setOutputs("Freestream Velocity Y (m/s)")
        self.streamVelocityZ = -1 * self.vz
        self.streamVelocityZ.setOutputs("Freestream Velocity Z (m/s)")
        self.freestreamSpeed = (
            self.streamVelocityX ** 2
            + self.streamVelocityY ** 2
            + self.streamVelocityZ ** 2
        ) ** 0.5
        self.freestreamSpeed.setOutputs("Freestream Speed (m/s)")
        # Apogee Freestream speed
        self.apogeeFreestreamSpeed = self.freestreamSpeed(self.apogeeTime)
        # Mach Number
        self.MachNumber = self.freestreamSpeed / self.speedOfSound
        self.MachNumber.setOutputs("Mach Number")
        maxMachNumberTimeIndex = np.argmax(self.MachNumber[:, 1])
        self.maxMachNumberTime = self.MachNumber[maxMachNumberTimeIndex, 0]
        self.maxMachNumber = self.MachNumber[maxMachNumberTimeIndex, 1]
        # Reynolds Number
        self.ReynoldsNumber = (
            self.density * self.freestreamSpeed / self.dynamicViscosity
        ) * (2 * self.rocket.radius)
        self.ReynoldsNumber.setOutputs("Reynolds Number")
        maxReynoldsNumberTimeIndex = np.argmax(self.ReynoldsNumber[:, 1])
        self.maxReynoldsNumberTime = self.ReynoldsNumber[maxReynoldsNumberTimeIndex, 0]
        self.maxReynoldsNumber = self.ReynoldsNumber[maxReynoldsNumberTimeIndex, 1]
        # Dynamic Pressure
        self.dynamicPressure = 0.5 * self.density * self.freestreamSpeed ** 2
        self.dynamicPressure.setOutputs("Dynamic Pressure (Pa)")
        maxDynamicPressureTimeIndex = np.argmax(self.dynamicPressure[:, 1])
        self.maxDynamicPressureTime = self.dynamicPressure[
            maxDynamicPressureTimeIndex, 0
        ]
        self.maxDynamicPressure = self.dynamicPressure[maxDynamicPressureTimeIndex, 1]
        # Total Pressure
        self.totalPressure = self.pressure * (1 + 0.2 * self.MachNumber ** 2) ** (3.5)
        self.totalPressure.setOutputs("Total Pressure (Pa)")
        maxtotalPressureTimeIndex = np.argmax(self.totalPressure[:, 1])
        self.maxtotalPressureTime = self.totalPressure[maxtotalPressureTimeIndex, 0]
        self.maxtotalPressure = self.totalPressure[maxDynamicPressureTimeIndex, 1]
        # Angle of Attack
        angleOfAttack = []
        for i in range(len(self.attitudeVectorX[:, 1])):
            dotProduct = -(
                self.attitudeVectorX[i, 1] * self.streamVelocityX[i, 1]
                + self.attitudeVectorY[i, 1] * self.streamVelocityY[i, 1]
                + self.attitudeVectorZ[i, 1] * self.streamVelocityZ[i, 1]
            )
            if self.freestreamSpeed[i, 1] < 1e-6:
                angleOfAttack.append([self.freestreamSpeed[i, 0], 0])
            else:
                dotProductNormalized = dotProduct / self.freestreamSpeed[i, 1]
                dotProductNormalized = (
                    1 if dotProductNormalized > 1 else dotProductNormalized
                )
                dotProductNormalized = (
                    -1 if dotProductNormalized < -1 else dotProductNormalized
                )
                angleOfAttack.append(
                    [
                        self.freestreamSpeed[i, 0],
                        (180 / np.pi) * np.arccos(dotProductNormalized),
                    ]
                )
        self.angleOfAttack = Function(
            angleOfAttack, "Time (s)", "Angle Of Attack (°)", "linear"
        )

        # Post process other quantities

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
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of out of rail time
        outOfRailTimeIndexs = np.nonzero(self.x[:, 0] == self.outOfRailTime)
        outOfRailTimeIndex = (
            -1 if len(outOfRailTimeIndexs) == 0 else outOfRailTimeIndexs[0][0]
        )

        # Get index of time before parachute event
        if len(self.parachuteEvents) > 0:
            eventTime = self.parachuteEvents[0][0] + self.parachuteEvents[0][1].lag
            eventTimeIndex = np.nonzero(self.x[:, 0] == eventTime)[0][0]
        else:
            eventTime = self.tFinal
            eventTimeIndex = -1

        # Print surface wind conditions
        print("Surface Wind Conditions\n")
        print("Frontal Surface Wind Speed: {:.2f} m/s".format(self.frontalSurfaceWind))
        print("Lateral Surface Wind Speed: {:.2f} m/s".format(self.lateralSurfaceWind))

        # Print of rail conditions
        print("\n\n Rail Departure State\n")
        print("Rail Departure Time: {:.3f} s".format(self.outOfRailTime))
        print("Rail Departure Velocity: {:.3f} m/s".format(self.outOfRailVelocity))
        print(
            "Rail Departure Static Margin: {:.3f} c".format(
                self.staticMargin(self.outOfRailTime)
            )
        )
        print(
            "Rail Departure Angle of Attack: {:.3f}°".format(
                self.angleOfAttack(self.outOfRailTime)
            )
        )
        print(
            "Rail Departure Thrust-Weight Ratio: {:.3f}".format(
            self.rocket.thrustToWeight(self.outOfRailTime)
            )
        )
        print(
            "Rail Departure Reynolds Number: {:.3e}".format(
                self.ReynoldsNumber(self.outOfRailTime)
            )
        )


        # Print burnOut conditions
        print("\n\nBurnOut State\n")
        print("BurnOut time: {:.3f} s".format(self.rocket.motor.burnOutTime))
        print(
            "Altitude at burnOut: {:.3f} m (AGL)".format(
            self.z( self.rocket.motor.burnOutTime ) - self.env.elevation
            ) 
        )
        print("Rocket velocity at burnOut: {:.3f} m/s".format(
            self.speed( self.rocket.motor.burnOutTime )
            ) 
        )
        print(
            "Freestream velocity at burnOut: {:.3f} m/s".format(
                (self.streamVelocityX( self.rocket.motor.burnOutTime )**2 + 
                self.streamVelocityY( self.rocket.motor.burnOutTime )**2 + 
                self.streamVelocityZ( self.rocket.motor.burnOutTime )**2)**0.5
            )
        )
        print(
            "Mach Number at burnOut: {:.3f}".format(
                self.MachNumber( self.rocket.motor.burnOutTime))
        )
        print("Kinetic energy at burnOut: {:.3e} J".format(
            self.kineticEnergy(self.rocket.motor.burnOutTime)
            )
        )


        # Print apogee conditions
        print("\n\nApogee\n")
        print(
            "Apogee Altitude: {:.3f} m (ASL) | {:.3f} m (AGL)".format(
                self.apogee, self.apogee - self.env.elevation
            )
        )
        print("Apogee Time: {:.3f} s".format(self.apogeeTime))
        print("Apogee Freestream Speed: {:.3f} m/s".format(self.apogeeFreestreamSpeed))

        # Print events registered
        print("\n\nEvents\n")
        if len(self.parachuteEvents) == 0:
            print("No Parachute Events Were Triggered.")
        for event in self.parachuteEvents:
            triggerTime = event[0]
            parachute = event[1]
            openTime = triggerTime + parachute.lag
            velocity = self.freestreamSpeed(openTime)
            altitude = self.z(openTime)
            name = parachute.name.title()
            print(name + " Ejection Triggered at: {:.3f} s".format(triggerTime))
            print(name + " Parachute Inflated at: {:.3f} s".format(openTime))
            print(
                name
                + " Parachute Inflated with Freestream Speed of: {:.3f} m/s".format(
                    velocity
                )
            )
            print(name + " Parachute Inflated at Height of: {:.3f} m (AGL)".format(altitude - self.env.elevation))

        # Print impact conditions
        if len(self.impactState) != 0:
            print("\n\nImpact\n")
            print("X Impact: {:.3f} m".format(self.xImpact))
            print("Y Impact: {:.3f} m".format(self.yImpact))
            print("Time of Impact: {:.3f} s".format(self.tFinal))
            print("Velocity at Impact: {:.3f} m/s".format(self.impactVelocity))
        elif self.terminateOnApogee is False:
            print("\n\nEnd of Simulation\n")
            print("Time: {:.3f} s".format(self.solution[-1][0]))
            print("Altitude: {:.3f} m".format(self.solution[-1][3]))

        # Print maximum values
        print("\n\nMaximum Values\n")
        print(
            "Maximum Speed: {:.3f} m/s at {:.2f} s".format(
                self.maxSpeed, self.maxSpeedTime
            )
        )
        print(
            "Maximum Mach Number: {:.3f} Mach at {:.2f} s".format(
                self.maxMachNumber, self.maxMachNumberTime
            )
        )
        print(
            "Maximum Reynolds Number: {:.3e} at {:.2f} s".format(
                self.maxReynoldsNumber, self.maxReynoldsNumberTime
            )
        )
        print(
            "Maximum Dynamic Pressure: {:.3e} Pa at {:.2f} s".format(
                self.maxDynamicPressure, self.maxDynamicPressureTime
            )
        )
        print(
            "Maximum Acceleration: {:.3f} m/s² at {:.2f} s".format(
                self.maxAcceleration, self.maxAccelerationTime
            )
        )
        print(
            "Maximum Gs: {:.3f} g at {:.2f} s".format(
                self.maxAcceleration / self.env.g, self.maxAccelerationTime
            )
        )
        print(
            "Maximum Upper Rail Button Normal Force: {:.3f} N".format(
                self.maxRailButton1NormalForce
            )
        )
        print(
            "Maximum Upper Rail Button Shear Force: {:.3f} N".format(
                self.maxRailButton1ShearForce
            )
        )
        print(
            "Maximum Lower Rail Button Normal Force: {:.3f} N".format(
                self.maxRailButton2NormalForce
            )
        )
        print(
            "Maximum Lower Rail Button Shear Force: {:.3f} N".format(
                self.maxRailButton2ShearForce
            )
        )

        return None

    def printInitialConditionsData(self):
        """Prints all initial conditions data available about the flight 

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()
        
        print(
            "Position - x: {:.2f} m | y: {:.2f} m | z: {:.2f} m".format(
                self.x(0), self.y(0), self.z(0)
            )
        )
        print(
            "Velocity - Vx: {:.2f} m/s | Vy: {:.2f} m/s | Vz: {:.2f} m/s".format(
                self.vx(0), self.vy(0), self.vz(0)
            )
        )
        print(
            "Attitude - e0: {:.3f} | e1: {:.3f} | e2: {:.3f} | e3: {:.3f}".format(
                self.e0(0), self.e1(0), self.e2(0), self.e3(0)
            )
        )
        print(
            "Euler Angles - Spin φ : {:.2f}° | Nutation θ: {:.2f}° | Precession ψ: {:.2f}°".format(
                self.phi(0), self.theta(0), self.psi(0)
            )
        )
        print(
            "Angular Velocity - ω1: {:.2f} rad/s | ω2: {:.2f} rad/s| ω3: {:.2f} rad/s".format(
                self.w1(0), self.w2(0), self.w3(0)
            )
        )
        return None


    def printNumericalIntegrationSettings(self):
        """Prints out the Numerical Integration settings

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        print("Maximum Allowed Flight Time: {:f} s".format(self.maxTime))
        print("Maximum Allowed Time Step: {:f} s".format(self.maxTimeStep))
        print("Minimum Allowed Time Step: {:e} s".format(self.minTimeStep))
        print("Relative Error Tolerance: ", self.rtol)
        print("Absolute Error Tolerance: ", self.atol)
        print("Allow Event Overshoot: ", self.timeOvershoot)
        print("Terminate Simulation on Apogee: ", self.terminateOnApogee)
        print("Number of Time Steps Used: ", len(self.timeSteps))
        print(
            "Number of Derivative Functions Evaluation: ",
            sum(self.functionEvaluationsPerTimeStep),
        )
        print(
            "Average Function Evaluations per Time Step: {:3f}".format(
                sum(self.functionEvaluationsPerTimeStep) / len(self.timeSteps))
        )

        return None

    def calculateStallWindVelocity(self, stallAngle):
        """ Function to calculate the maximum wind velocity before the angle of
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
        tetha = self.inclination * np.pi /180
        stallAngle = stallAngle * np.pi /180

        c = (math.cos(stallAngle)**2 - math.cos(tetha)**2)/ math.sin(stallAngle)**2
        wV = (2*vF*math.cos(tetha)/c + (4*vF*vF*math.cos(tetha)*math.cos(tetha)/(c**2) + 4*1*vF*vF/c )**0.5 )/2
        
        # Convert stallAngle to degrees
        stallAngle = stallAngle * 180 / np.pi
        print("Maximum wind velocity at Rail Departure time before angle of attack exceeds {:.3f}°: {:.3f} m/s".format(stallAngle, wV))

        return None

    def plot3dTrajectory(self):
        """Plot a 3D graph of the trajectory

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get max and min x and y
        maxZ = max(self.z[:, 1] - self.env.elevation )
        maxX = max(self.x[:, 1])
        minX = min(self.x[:, 1])
        maxY = max(self.y[:, 1])
        minY = min(self.y[:, 1])
        maxXY = max(maxX, maxY)
        minXY = min(minX, minY)

        # Create figure
        fig1 = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111, projection="3d")
        ax1.plot(
            self.x[:, 1], self.y[:, 1], zs= 0, zdir="z", linestyle="--"
        )
        ax1.plot(self.x[:, 1], self.z[:, 1] - self.env.elevation, zs=minXY, zdir="y", linestyle="--")
        ax1.plot(self.y[:, 1], self.z[:, 1] - self.env.elevation, zs=minXY, zdir="x", linestyle="--")
        ax1.plot(self.x[:, 1], self.y[:, 1], self.z[:, 1] - self.env.elevation, linewidth='2')
        ax1.scatter(0, 0, 0)
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Z - Altitude Above Ground Level (m)")
        ax1.set_title("Flight Trajectory")
        ax1.set_zlim3d([0, maxZ])
        ax1.set_ylim3d([minXY, maxXY])
        ax1.set_xlim3d([minXY, maxXY])
        ax1.view_init(15, 45)
        plt.show()

        return None

    def plotLinearKinematicsData(self):
        """Prints out all Kinematics graphs available about the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Velocity and acceleration plots
        fig2 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(414)
        ax1.plot(self.vx[:, 0], self.vx[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.tFinal)
        ax1.set_title("Velocity X | Acceleration X")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity X (m/s)", color="#ff7f0e")
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(self.ax[:, 0], self.ax[:, 1], color="#1f77b4")
        ax1up.set_ylabel("Acceleration X (m/s²)", color="#1f77b4")
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(413)
        ax2.plot(self.vy[:, 0], self.vy[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, self.tFinal)
        ax2.set_title("Velocity Y | Acceleration Y")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity Y (m/s)", color="#ff7f0e")
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(self.ay[:, 0], self.ay[:, 1], color="#1f77b4")
        ax2up.set_ylabel("Acceleration Y (m/s²)", color="#1f77b4")
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(412)
        ax3.plot(self.vz[:, 0], self.vz[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, self.tFinal)
        ax3.set_title("Velocity Z | Acceleration Z")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity Z (m/s)", color="#ff7f0e")
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(self.az[:, 0], self.az[:, 1], color="#1f77b4")
        ax3up.set_ylabel("Acceleration Z (m/s²)", color="#1f77b4")
        ax3up.tick_params("y", colors="#1f77b4")

        ax4 = plt.subplot(411)
        ax4.plot(self.speed[:, 0], self.speed[:, 1], color="#ff7f0e")
        ax4.set_xlim(0, self.tFinal)
        ax4.set_title("Velocity Magnitude | Acceleration Magnitude")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity (m/s)", color="#ff7f0e")
        ax4.tick_params("y", colors="#ff7f0e")
        ax4.grid(True)

        ax4up = ax4.twinx()
        ax4up.plot(self.acceleration[:, 0], self.acceleration[:, 1], color="#1f77b4")
        ax4up.set_ylabel("Acceleration (m/s²)", color="#1f77b4")
        ax4up.tick_params("y", colors="#1f77b4")

        plt.subplots_adjust(hspace=0.5)
        plt.show()
        return None

    def plotAttitudeData(self):
        """Prints out all Angular position graphs available about the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of time before parachute event
        if len(self.parachuteEvents) > 0:
            eventTime = self.parachuteEvents[0][0] + self.parachuteEvents[0][1].lag
            eventTimeIndex = np.nonzero(self.x[:, 0] == eventTime)[0][0]
        else:
            eventTime = self.tFinal
            eventTimeIndex = -1

        # Angular position plots
        fig3 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.e0[:, 0], self.e0[:, 1], label="$e_0$")
        ax1.plot(self.e1[:, 0], self.e1[:, 1], label="$e_1$")
        ax1.plot(self.e2[:, 0], self.e2[:, 1], label="$e_2$")
        ax1.plot(self.e3[:, 0], self.e3[:, 1], label="$e_3$")
        ax1.set_xlim(0, eventTime)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Euler Parameters")
        ax1.set_title("Euler Parameters")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(412)
        ax2.plot(self.psi[:, 0], self.psi[:, 1])
        ax2.set_xlim(0, eventTime)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ψ (°)")
        ax2.set_title("Euler Precession Angle")
        ax2.grid(True)

        ax3 = plt.subplot(413)
        ax3.plot(self.theta[:, 0], self.theta[:, 1], label="θ - Nutation")
        ax3.set_xlim(0, eventTime)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("θ (°)")
        ax3.set_title("Euler Nutation Angle")
        ax3.grid(True)

        ax4 = plt.subplot(414)
        ax4.plot(self.phi[:, 0], self.phi[:, 1], label="φ - Spin")
        ax4.set_xlim(0, eventTime)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("φ (°)")
        ax4.set_title("Euler Spin Angle")
        ax4.grid(True)

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def plotFlightPathAngleData(self):
        """Prints out Flight path and Rocket Attitude angle graphs available
        about the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of time before parachute event
        if len(self.parachuteEvents) > 0:
            eventTime = self.parachuteEvents[0][0] + self.parachuteEvents[0][1].lag
            eventTimeIndex = np.nonzero(self.x[:, 0] == eventTime)[0][0]
        else:
            eventTime = self.tFinal
            eventTimeIndex = -1
        
        # Path, Attitude and Lateral Attitude Angle
        # Angular position plots
        fig5 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(self.pathAngle[:, 0], self.pathAngle[:, 1], label="Flight Path Angle")
        ax1.plot(
            self.attitudeAngle[:, 0],
            self.attitudeAngle[:, 1],
            label="Rocket Attitude Angle",
        )
        ax1.set_xlim(0, eventTime)
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.set_title("Flight Path and Attitude Angle")

        ax2 = plt.subplot(212)
        ax2.plot(self.lateralAttitudeAngle[:, 0], self.lateralAttitudeAngle[:, 1])
        ax2.set_xlim(0, eventTime)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Lateral Attitude Angle (°)")
        ax2.set_title("Lateral Attitude Angle")
        ax2.grid(True)

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def plotAngularKinematicsData(self):
        """Prints out all Angular veolcity and acceleration graphs available
        about the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of time before parachute event
        if len(self.parachuteEvents) > 0:
            eventTime = self.parachuteEvents[0][0] + self.parachuteEvents[0][1].lag
            eventTimeIndex = np.nonzero(self.x[:, 0] == eventTime)[0][0]
        else:
            eventTime = self.tFinal
            eventTimeIndex = -1
        
        # Angular velocity and acceleration plots
        fig4 = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(311)
        ax1.plot(self.w1[:, 0], self.w1[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, eventTime)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angular Velocity - ${\omega_1}$ (rad/s)", color="#ff7f0e")
        ax1.set_title(
            "Angular Velocity ${\omega_1}$ | Angular Acceleration ${\\alpha_1}$"
        )
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(self.alpha1[:, 0], self.alpha1[:, 1], color="#1f77b4")
        ax1up.set_ylabel(
            "Angular Acceleration - ${\\alpha_1}$ (rad/s²)", color="#1f77b4"
        )
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(312)
        ax2.plot(self.w2[:, 0], self.w2[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, eventTime)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angular Velocity - ${\omega_2}$ (rad/s)", color="#ff7f0e")
        ax2.set_title(
            "Angular Velocity ${\omega_2}$ | Angular Acceleration ${\\alpha_2}$"
        )
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(self.alpha2[:, 0], self.alpha2[:, 1], color="#1f77b4")
        ax2up.set_ylabel(
            "Angular Acceleration - ${\\alpha_2}$ (rad/s²)", color="#1f77b4"
        )
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(313)
        ax3.plot(self.w3[:, 0], self.w3[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, eventTime)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity - ${\omega_3}$ (rad/s)", color="#ff7f0e")
        ax3.set_title(
            "Angular Velocity ${\omega_3}$ | Angular Acceleration ${\\alpha_3}$"
        )
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(self.alpha3[:, 0], self.alpha3[:, 1], color="#1f77b4")
        ax3up.set_ylabel(
            "Angular Acceleration - ${\\alpha_3}$ (rad/s²)", color="#1f77b4"
        )
        ax3up.tick_params("y", colors="#1f77b4")

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def plotTrajectoryForceData(self):
        """Prints out all Forces and Moments graphs available about the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of out of rail time
        outOfRailTimeIndexs = np.nonzero(self.x[:, 0] == self.outOfRailTime)
        outOfRailTimeIndex = -1 if len(outOfRailTimeIndexs) == 0 else outOfRailTimeIndexs[0][0]

        # Get index of time before parachute event
        if len(self.parachuteEvents) > 0:
            eventTime = self.parachuteEvents[0][0] + self.parachuteEvents[0][1].lag
            eventTimeIndex = np.nonzero(self.x[:, 0] == eventTime)[0][0]
        else:
            eventTime = self.tFinal
            eventTimeIndex = -1
        
        # Rail Button Forces
        fig6 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(
            self.railButton1NormalForce[:outOfRailTimeIndex, 0],
            self.railButton1NormalForce[:outOfRailTimeIndex, 1],
            label="Upper Rail Button",
        )
        ax1.plot(
            self.railButton2NormalForce[:outOfRailTimeIndex, 0],
            self.railButton2NormalForce[:outOfRailTimeIndex, 1],
            label="Lower Rail Button",
        )
        ax1.set_xlim(0, self.outOfRailTime)
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normal Force (N)")
        ax1.set_title("Rail Buttons Normal Force")

        ax2 = plt.subplot(212)
        ax2.plot(
            self.railButton1ShearForce[:outOfRailTimeIndex, 0],
            self.railButton1ShearForce[:outOfRailTimeIndex, 1],
            label="Upper Rail Button",
        )
        ax2.plot(
            self.railButton2ShearForce[:outOfRailTimeIndex, 0],
            self.railButton2ShearForce[:outOfRailTimeIndex, 1],
            label="Lower Rail Button",
        )
        ax2.set_xlim(0, self.outOfRailTime)
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Shear Force (N)")
        ax2.set_title("Rail Buttons Shear Force")

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        # Aerodynamic force and moment plots
        fig7 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.aerodynamicLift[:eventTimeIndex, 0], self.aerodynamicLift[:eventTimeIndex, 1], label='Resultant')
        ax1.plot(self.R1[:eventTimeIndex, 0], self.R1[:eventTimeIndex, 1], label='R1')
        ax1.plot(self.R2[:eventTimeIndex, 0], self.R2[:eventTimeIndex, 1], label='R2')
        ax1.set_xlim(0, eventTime)
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Lift Force (N)")
        ax1.set_title("Aerodynamic Lift Resultant Force")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(self.aerodynamicDrag[:eventTimeIndex, 0], self.aerodynamicDrag[:eventTimeIndex, 1])
        ax2.set_xlim(0, eventTime)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Drag Force (N)")
        ax2.set_title("Aerodynamic Drag Force")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.aerodynamicBendingMoment[:eventTimeIndex, 0],
            self.aerodynamicBendingMoment[:eventTimeIndex, 1],
            label='Resultant',
        )
        ax3.plot(self.M1[:eventTimeIndex, 0], self.M1[:eventTimeIndex, 1], label='M1')
        ax3.plot(self.M2[:eventTimeIndex, 0], self.M2[:eventTimeIndex, 1], label='M2')
        ax3.set_xlim(0, eventTime)
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Bending Moment (N m)")
        ax3.set_title("Aerodynamic Bending Resultant Moment")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(self.aerodynamicSpinMoment[:eventTimeIndex, 0], self.aerodynamicSpinMoment[:eventTimeIndex, 1])
        ax4.set_xlim(0, eventTime)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Spin Moment (N m)")
        ax4.set_title("Aerodynamic Spin Moment")
        ax4.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def plotEnergyData(self):
        """Prints out all Energy components graphs available about the Flight

        Returns
        -------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of out of rail time
        outOfRailTimeIndexs = np.nonzero(self.x[:, 0] == self.outOfRailTime)
        outOfRailTimeIndex = -1 if len(outOfRailTimeIndexs) == 0 else outOfRailTimeIndexs[0][0]

        # Get index of time before parachute event
        if len(self.parachuteEvents) > 0:
            eventTime = self.parachuteEvents[0][0] + self.parachuteEvents[0][1].lag
            eventTimeIndex = np.nonzero(self.x[:, 0] == eventTime)[0][0]
        else:
            eventTime = self.tFinal
            eventTimeIndex = -1
        
        fig8 = plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(411)
        ax1.plot(
            self.kineticEnergy[:, 0], self.kineticEnergy[:, 1], label="Kinetic Energy"
        )
        ax1.plot(
            self.rotationalEnergy[:, 0],
            self.rotationalEnergy[:, 1],
            label="Rotational Energy",
        )
        ax1.plot(
            self.translationalEnergy[:, 0],
            self.translationalEnergy[:, 1],
            label="Translational Energy",
        )
        ax1.set_xlim(0, self.apogeeTime)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.set_title("Kinetic Energy Components")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Energy (J)")
        
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(self.totalEnergy[:, 0], self.totalEnergy[:, 1], label="Total Energy")
        ax2.plot(
            self.kineticEnergy[:, 0], self.kineticEnergy[:, 1], label="Kinetic Energy"
        )
        ax2.plot(
            self.potentialEnergy[:, 0],
            self.potentialEnergy[:, 1],
            label="Potential Energy",
        )
        ax2.set_xlim(0, self.apogeeTime)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.set_title("Total Mechanical Energy Components")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (J)")
        ax2.legend()
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(self.thrustPower[:, 0], self.thrustPower[:, 1], label="|Thrust Power|")
        ax3.set_xlim(0, self.rocket.motor.burnOutTime)
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax3.set_title("Thrust Absolute Power")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Power (W)")
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(self.dragPower[:, 0], -self.dragPower[:, 1], label="|Drag Power|")
        ax4.set_xlim(0, self.apogeeTime)
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax4.set_title("Drag Absolute Power")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Power (W)")
        ax4.legend()
        ax4.grid()

        plt.subplots_adjust(hspace=1)
        plt.show()

        return None

    def plotFluidMechanicsData(self):
        """Prints out a summary of the Fluid Mechanics graphs available about 
        the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Get index of out of rail time
        outOfRailTimeIndexs = np.nonzero(self.x[:, 0] == self.outOfRailTime)
        outOfRailTimeIndex = -1 if len(outOfRailTimeIndexs) == 0 else outOfRailTimeIndexs[0][0]

        
        # Trajectory Fluid Mechanics Plots
        fig10 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.MachNumber[:, 0], self.MachNumber[:, 1])
        ax1.set_xlim(0, self.tFinal)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(self.ReynoldsNumber[:, 0], self.ReynoldsNumber[:, 1])
        ax2.set_xlim(0, self.tFinal)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.dynamicPressure[:, 0],
            self.dynamicPressure[:, 1],
            label="Dynamic Pressure",
        )
        ax3.plot(
            self.totalPressure[:, 0], self.totalPressure[:, 1], label="Total Pressure"
        )
        ax3.plot(self.pressure[:, 0], self.pressure[:, 1], label="Static Pressure")
        ax3.set_xlim(0, self.tFinal)
        ax3.legend()
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax3.set_title("Total and Dynamic Pressure")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (Pa)")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(self.angleOfAttack[:, 0], self.angleOfAttack[:, 1])
        ax4.set_xlim(self.outOfRailTime, 10*self.outOfRailTime)
        ax4.set_ylim(0, self.angleOfAttack(self.outOfRailTime))
        ax4.set_title("Angle of Attack")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle of Attack (°)")
        ax4.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def calculateFinFlutterAnalysis(self, finThickness, shearModulus):
        """ Calculate, create and plot the Fin Flutter velocity, based on the
        pressure profile provided by Atmosferic model selected. It considers the
        Flutter Boundary Equation that is based on a calculation published in
        NACA Technical Paper 4197.
        Be careful, these results are only estimates of a real problem and may
        not be useful for fins made from non-isotropic materials. These results
        should not be used as a way to fully prove the safety of any rocket’s fins.
        IMPORTANT: This function works if only a single set of fins is added

        Parameters
        ----------
        finThickness : float
            The fin thickness, in meters
        shearModulus : float
            Shear Modulus of fins' material, must be given in Pascal
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        s = (self.rocket.tipChord + self.rocket.rootChord) * self.rocket.span /2
        ar = self.rocket.span * self.rocket.span / s
        la = self.rocket.tipChord / self.rocket.rootChord
 
        # Calculate the Fin Flutter Mach Number
        self.flutterMachNumber = ((shearModulus*2*(ar+2)*(finThickness/self.rocket.rootChord)**3)/(1.337 * (ar**3) *(la+1) * self.pressure ))**0.5 
        
        # Calculate difference between Fin Flutter Mach Number and the Rocket Speed
        self.difference =  self.flutterMachNumber - self.MachNumber

        # Calculate a safety factor for flutter
        self.safetyFactor = self.flutterMachNumber / self.MachNumber

        # Calculate the minimun Fin Flutter Mach Number and Velocity
        # Calculate the time and height of minimun Fin Flutter Mach Number
        minflutterMachNumberTimeIndex = np.argmin(self.flutterMachNumber[:,1])
        minflutterMachNumber = self.flutterMachNumber[minflutterMachNumberTimeIndex,1]
        minMFTime = self.flutterMachNumber[minflutterMachNumberTimeIndex,0]
        minMFHeight = self.z(minMFTime) - self.env.elevation
        minMFVelocity = minflutterMachNumber * self.env.speedOfSound(minMFHeight)

        # Calculate minimum difference between Fin Flutter Mach Number and the Rocket Speed
        # Calculate the time and height of the difference ...
        minDifferenceTimeIndex = np.argmin(self.difference[:,1])
        minDif = self.difference[minDifferenceTimeIndex,1]
        minDifTime = self.difference[minDifferenceTimeIndex,0]
        minDifHeight = self.z(minDifTime) - self.env.elevation
        minDifVelocity = minDif * self.env.speedOfSound(minDifHeight)

        # Calculate the minimun Fin Flutter Safety factor
        # Calculate the time and height of minimun Fin Flutter Safety factor
        minSFTimeIndex = np.argmin(self.safetyFactor[:,1])
        minSF = self.safetyFactor[minSFTimeIndex,1]
        minSFTime = self.safetyFactor[minSFTimeIndex,0]
        minSFHeight = self.z(minSFTime) - self.env.elevation

        # Print fin's geometric parameters
        print("Fin's geometric parameters")
        print("Surface area (S): {:.4f} m2".format(s))
        print("Aspect ratio (AR): {:.3f}".format(ar))
        print("TipChord/RootChord = \u03BB = {:.3f}".format(la))
        print("Fin Thickness: {:.5f} m".format(finThickness))

        # Print fin's material properties
        print("\n\nFin's material properties")
        print("Shear Modulus (G): {:.3e} Pa".format(shearModulus))

        # Print a summary of the Fin Flutter Analysis
        print("\n\nFin Flutter Analysis")
        print(
            "Minimum Fin Flutter Velocity: {:.3f} m/s at {:.2f} s".format(
                minMFVelocity, minMFTime 
            )
        )
        print(
            "Minimum Fin Flutter Mach Number: {:.3f} ".format(minflutterMachNumber)
        )
        #print(
        #    "Altitude of minimum Fin Flutter Velocity: {:.3f} m (AGL)".format(
        #        minMFHeight
        #    )
        #)
        print(
            "Minimum of (Fin Flutter Mach Number - Rocket Speed): {:.3f} m/s at {:.2f} s".format(
                minDifVelocity, minDifTime 
            )
        )
        print(
            "Minimum of (Fin Flutter Mach Number - Rocket Speed): {:.3f} Mach at {:.2f} s".format(
                minDif, minDifTime 
            )
        )
        #print(
        #    "Altitude of minimum (Fin Flutter Mach Number - Rocket Speed): {:.3f} m (AGL)".format(
        #        minDifHeight
        #    )
        #)
        print(
            "Minimum Fin Flutter Safety Factor: {:.3f} at {:.2f} s".format(
                minSF, minSFTime 
            )
        )
        print(
            "Altitude of minimum Fin Flutter Safety Factor: {:.3f} m (AGL)\n\n".format(
                minSFHeight
            )
        )

        #Create plots
        fig12 = plt.figure(figsize=(6, 9))
        ax1 = plt.subplot(311)
        ax1.plot()
        ax1.plot(self.flutterMachNumber[:,0] , self.flutterMachNumber[:,1], label = "Fin flutter Mach Number")
        ax1.plot(self.MachNumber[:,0], self.MachNumber[:,1], label= "Rocket Freestream Speed")
        ax1.set_xlim(0, self.apogeeTime)
        ax1.set_title("Fin Flutter Mach Number x Time(s)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(312)
        ax2.plot(self.difference[:,0], self.difference[:,1])
        ax2.set_xlim(0, self.apogeeTime)
        ax2.set_title("Mach flutter - Freestream velocity")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Mach")
        ax2.grid()

        ax3 = plt.subplot(313)
        ax3.plot(self.safetyFactor[:,0], self.safetyFactor[:,1])
        ax3.set_xlim(self.outOfRailTime, self.apogeeTime)
        ax3.set_ylim(0, 6)
        ax3.set_title("Fin Flutter Safety Factor")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Safety Factor")
        ax3.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def plotStabilityAndControlData(self):
        """Prints out Rocket Stability and Control parameters graphs available 
        about the Flight

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()
        
        fig9 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(self.staticMargin[:, 0], self.staticMargin[:, 1])
        ax1.set_xlim(0, self.staticMargin[:, 0][-1])
        ax1.set_title("Static Margin")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Static Margin (c)")
        ax1.grid()

        ax2 = plt.subplot(212)
        maxAttitude = max(self.attitudeFrequencyResponse[:, 1])
        maxAttitude = maxAttitude if maxAttitude != 0 else 1
        ax2.plot(
            self.attitudeFrequencyResponse[:, 0],
            self.attitudeFrequencyResponse[:, 1] / maxAttitude,
            label="Attitude Angle",
        )
        maxOmega1 = max(self.omega1FrequencyResponse[:, 1])
        maxOmega1 = maxOmega1 if maxOmega1 != 0 else 1
        ax2.plot(
            self.omega1FrequencyResponse[:, 0],
            self.omega1FrequencyResponse[:, 1] / maxOmega1,
            label="$\omega_1$",
        )
        maxOmega2 = max(self.omega2FrequencyResponse[:, 1])
        maxOmega2 = maxOmega2 if maxOmega2 != 0 else 1
        ax2.plot(
            self.omega2FrequencyResponse[:, 0],
            self.omega2FrequencyResponse[:, 1] / maxOmega2,
            label="$\omega_2$",
        )
        maxOmega3 = max(self.omega3FrequencyResponse[:, 1])
        maxOmega3 = maxOmega3 if maxOmega3 != 0 else 1
        ax2.plot(
            self.omega3FrequencyResponse[:, 0],
            self.omega3FrequencyResponse[:, 1] / maxOmega3,
            label="$\omega_3$",
        )
        ax2.set_title("Frequency Response")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude Magnitude Normalized")
        ax2.set_xlim(0, 5)
        ax2.legend()
        ax2.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def plotPressureSignals(self):
        """ Prints out all Parachute Trigger Pressure Signals.
        This function can be called also for plot pressure data for flights
        without Parachutes, in this case the Pressure Signals will be simply
        the pressure provided by the atmosfericModel, at Flight z positions.
        This means that no noise will be considered if at least one parachute 
        has not been added.

        This function aims to help the engineer to visually check if there
        isn't no anomalies with the Flight Simulation. 

        Parameters
        ----------
        None
        
        Return
        ------
        None
        """
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        if len(self.rocket.parachutes) == 0:
            plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(self.z[:,0], self.env.pressure(self.z[:,1]))
            ax1.set_title("Pressure at Rocket's Altitude")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Pressure (Pa)")
            ax1.set_xlim(0, self.tFinal)
            ax1.grid()

            plt.show()

        else:
            for parachute in self.rocket.parachutes:
                print('Parachute: ', parachute.name)
                parachute.noiseSignalFunction()
                parachute.noisyPressureSignalFunction()
                parachute.cleanPressureSignalFunction()

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

        This function was created especially for the Projeto Jupiter Eletronics
        Subsystems team and aims to help in configuring microcontrollers.

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
        if self.postProcessed is False:
            self.postProcess()
        
        timePoints = np.arange(0, self.tFinal, timeStep)
        
        # Create the file
        file = open(fileName, 'w')

        if len(self.rocket.parachutes) == 0:
            pressure = self.env.pressure(self.z(timePoints))
            for i in range(0, timePoints.size, 1):
                file.write("{:f}, {:.5f}\n".format(timePoints[i], pressure[i]))

        else:
            for parachute in self.rocket.parachutes:
                for i in range(0, timePoints.size, 1):
                    pCl = parachute.cleanPressureSignalFunction(timePoints[i])
                    pNs = parachute.noisyPressureSignalFunction(timePoints[i])
                    file.write("{:f}, {:.5f}, {:.5f}\n".format(timePoints[i], pCl, pNs))
                # We need to save only 1 parachute data
                pass

        file.close()
    
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
        # Post-process results
        if self.postProcessed is False:
            self.postProcess()

        # Print initial conditions
        print("Initial Conditions\n")
        self.printInitialConditionsData()

        # Print launch rail orientation
        print("\n\nLaunch Rail Orientation\n")
        print("Launch Rail Inclination: {:.2f}°".format(self.inclination))
        print("Launch Rail Heading: {:.2f}°\n\n".format(self.heading))

        # Print a summary of data about the flight
        self.info()
        
        print("\n\nNumerical Integration Information\n")
        self.printNumericalIntegrationSettings()

        print("\n\nTrajectory 3d Plot\n")
        self.plot3dTrajectory()

        print("\n\nTrajectory Kinematic Plots\n")
        self.plotLinearKinematicsData()
        
        print("\n\nAngular Position Plots\n")
        self.plotFlightPathAngleData()
        
        print("\n\nPath, Attitude and Lateral Attitude Angle plots\n")
        self.plotAttitudeData()
        
        print("\n\nTrajectory Angular Velocity and Acceleration Plots\n")
        self.plotAngularKinematicsData()
        
        print("\n\nTrajectory Force Plots\n")
        self.plotTrajectoryForceData()
        
        print("\n\nTrajectory Energy Plots\n")
        self.plotEnergyData()
        
        print("\n\nTrajectory Fluid Mechanics Plots\n")
        self.plotFluidMechanicsData()

        print("\n\nTrajectory Stability and Control Plots\n")
        self.plotStabilityAndControlData()

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
            # Caclulate rocket Mach number
            Vx = self.vx(t) / 340.40
            Vy = self.vy(t) / 340.40
            Vz = self.vz(t) / 340.40
            # Caculate wind Mach Number
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
                        "WARNING: Trying to add a flight phase starting together with the one proceding it."
                    )
                    print(
                        "This may be caused by more than when parachute being triggered simultaneously."
                    )
                    flightPhase.t += 1e-7
                    self.add(flightPhase, index + 1)
                elif flightPhase.t > nextPhase.t:
                    print(
                        "WARNING: Trying to add a flight phase starting after the one proceding it."
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
