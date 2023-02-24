# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np
from scipy import integrate

from rocketpy.Function import Function
from rocketpy.motors import SolidMotor, LiquidMotor


class HybridMotor(SolidMotor, LiquidMotor):
    """Class to specify characteristics and useful operations for Hybrid
    motors.

    Attributes
    ----------

        Geometrical attributes:
        Motor.nozzleRadius : float
            Radius of motor nozzle outlet in meters.
        Motor.throatRadius : float
            Radius of motor nozzle throat in meters.
        Motor.grainNumber : int
            Number of solid grains.
        Motor.grainSeparation : float
            Distance between two grains in meters.
        Motor.grainDensity : float
            Density of each grain in kg/meters cubed.
        Motor.grainOuterRadius : float
            Outer radius of each grain in meters.
        Motor.grainInitialInnerRadius : float
            Initial inner radius of each grain in meters.
        Motor.grainInitialHeight : float
            Initial height of each grain in meters.
        Motor.grainInitialVolume : float
            Initial volume of each grain in meters cubed.
        Motor.grainInnerRadius : Function
            Inner radius of each grain in meters as a function of time.
        Motor.grainHeight : Function
            Height of each grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.grainInitialMass : float
            Initial mass of each grain in kg.
        Motor.propellantInitialMass : float
            Total propellant initial mass in kg.
        Motor.mass : Function
            Propellant total mass in kg as a function of time.
        Motor.massDot : Function
            Time derivative of propellant total mass in kg/s as a function
            of time.
        Motor.inertiaI : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis
            perpendicular to axis of cylindrical symmetry of each grain,
            given as a function of time.
        Motor.inertiaIDot : Function
            Time derivative of inertiaI given in kg*meter^2/s as a function
            of time.
        Motor.inertiaZ : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis of
            cylindrical symmetry of each grain, given as a function of time.
        Motor.inertiaDot : Function
            Time derivative of inertiaZ given in kg*meter^2/s as a function
            of time.

        Thrust and burn attributes:
        Motor.thrust : Function
            Motor thrust force, in Newtons, as a function of time.
        Motor.totalImpulse : float
            Total impulse of the thrust curve in N*s.
        Motor.maxThrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.maxThrustTime : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.averageThrust : float
            Average thrust of the motor, given in N.
        Motor.burnOutTime : float
            Total motor burn out time, in seconds. Must include delay time
            when the motor takes time to ignite. Also seen as time to end thrust
            curve.
        Motor.exhaustVelocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burnArea : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burnArea divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burnRate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrustSource,
        burnOut,
        grainNumber,
        grainDensity,
        grainOuterRadius,
        grainInitialInnerRadius,
        grainInitialHeight,
        oxidizerTankRadius,
        oxidizerTankHeight,
        oxidizerInitialPressure,
        oxidizerDensity,
        oxidizerMolarMass,
        oxidizerInitialVolume,
        distanceGrainToTank,
        injectorArea,
        grainSeparation=0,
        nozzleRadius=0.0335,
        throatRadius=0.0114,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrustSource : int, float, callable, string, array
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Thrust units are Newtons.
        burnOut : int, float
            Motor burn out time in seconds.
        grainNumber : int
            Number of solid grains
        grainDensity : int, float
            Solid grain density in kg/m3.
        grainOuterRadius : int, float
            Solid grain outer radius in meters.
        grainInitialInnerRadius : int, float
            Solid grain initial inner radius in meters.
        grainInitialHeight : int, float
            Solid grain initial height in meters.
        oxidizerTankRadius :
            Oxidizer Tank inner radius.
        oxidizerTankHeight :
            Oxidizer Tank Height.
        oxidizerInitialPressure :
            Initial pressure of the oxidizer tank, could be equal to the pressure of the source cylinder in atm.
        oxidizerDensity :
            Oxidizer theoretical density in liquid state, for N2O is equal to 1.98 (Kg/m^3).
        oxidizerMolarMass :
            Oxidizer molar mass, for the N2O is equal to 44.01 (g/mol).
        oxidizerInitialVolume :
            Initial volume of oxidizer charged in the tank.
        distanceGrainToTank :
            Distance between the solid grain center of mass and the base of the oxidizer tank.
        injectorArea :
            injector outlet area.
        grainSeparation : int, float, optional
            Distance between grains, in meters. Default is 0.
        nozzleRadius : int, float, optional
            Motor's nozzle outlet radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not impact
            trajectory simulation.
        throatRadius : int, float, optional
            Motor's nozzle throat radius in meters. Its value has very low
            impact in trajectory simulation, only useful to analyze
            dynamic instabilities, therefore it is optional.
        reshapeThrustCurve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolationMethod : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".

        Returns
        -------
        None
        """
        super().__init__(
            thrustSource,
            burnOut,
            nozzleRadius,
            throatRadius,
            reshapeThrustCurve,
            interpolationMethod,
        )

        # Define motor attributes
        # Grain and nozzle parameters
        self.grainNumber = grainNumber
        self.grainSeparation = grainSeparation
        self.grainDensity = grainDensity
        self.grainOuterRadius = grainOuterRadius
        self.grainInitialInnerRadius = grainInitialInnerRadius
        self.grainInitialHeight = grainInitialHeight
        self.oxidizerTankRadius = oxidizerTankRadius
        self.oxidizerTankHeight = oxidizerTankHeight
        self.oxidizerInitialPressure = oxidizerInitialPressure
        self.oxidizerDensity = oxidizerDensity
        self.oxidizerMolarMass = oxidizerMolarMass
        self.oxidizerInitialVolume = oxidizerInitialVolume
        self.distanceGrainToTank = distanceGrainToTank
        self.injectorArea = injectorArea

        # Other quantities that will be computed
        self.zCM = None
        self.oxidizerInitialMass = None
        self.grainInnerRadius = None
        self.grainHeight = None
        self.burnArea = None
        self.burnRate = None
        self.Kn = None

        # Compute uncalculated quantities
        # Grains initial geometrical parameters
        self.grainInitialVolume = (
            self.grainInitialHeight
            * np.pi
            * (self.grainOuterRadius**2 - self.grainInitialInnerRadius**2)
        )
        self.grainInitialMass = self.grainDensity * self.grainInitialVolume
        self.propellantInitialMass = (
            self.grainNumber * self.grainInitialMass
            + self.oxidizerInitialVolume * self.oxidizerDensity
        )
        # Dynamic quantities
        self.evaluateMassDot()
        self.evaluateMass()
        self.evaluateGeometry()
        self.evaluateInertia()
        self.evaluateCenterOfMass()

    def evaluateCenterOfMass(self):
        """Calculates and returns the time derivative of motor center of mass.
        The formulas used are the Bernoulli equation, law of the ideal gases and Boyle's law.
        The result is a function of time, object of the Function class, which is stored in self.zCM.

        Parameters
        ----------
        None

        Returns
        -------
        zCM : Function
            Position of the center of mass as a function
            of time.
        """
        ...

    def evaluateInertia(self):
        """Calculates propellant inertia I, relative to directions
        perpendicular to the rocket body axis and its time derivative
        as a function of time. Also calculates propellant inertia Z,
        relative to the axial direction, and its time derivative as a
        function of time. Products of inertia are assumed null due to
        symmetry. The four functions are stored as an object of the
        Function class.

        Parameters
        ----------
        None

        Returns
        -------
        list of Functions
            The first argument is the Function representing inertia I,
            while the second argument is the Function representing
            inertia Z.
        """

        ...

    def allInfo(self):
        pass
