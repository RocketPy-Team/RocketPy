# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np
from scipy import integrate
from functools import cached_property

from rocketpy.Function import Function, funcify_method
from rocketpy.motors import SolidMotor, LiquidMotor, Motor


class HybridMotor(Motor):
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
        chamberPosition,
        grainNumber,
        grainDensity,
        grainOuterRadius,
        grainInitialInnerRadius,
        grainInitialHeight,
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
        self.liquid = LiquidMotor(
            thrustSource,
            burnOut,
            nozzleRadius,
            throatRadius,
            reshapeThrustCurve,
            interpolationMethod,
        )
        self.solid = SolidMotor(
            thrustSource,
            burnOut,
            chamberPosition,
            grainNumber,
            grainDensity,
            grainOuterRadius,
            grainInitialInnerRadius,
            grainInitialHeight,
            grainSeparation,
            nozzleRadius,
            throatRadius,
            reshapeThrustCurve,
            interpolationMethod,
        )

    def addTank(self, tank, position):
        self.liquid.addTank(tank, position)
        self.solid.massFlowRate = self.massDot - self.liquid.massFlowRate

    @cached_property
    def propellantInitialMass(self):
        return self.solid.propellantInitialMass + self.liquid.propellantInitialMass

    @funcify_method
    def mass(self):
        return self.solid.mass + self.liquid.mass

    @cached_property
    def massFlowRate(self):
        return self.solid.massFlowRate + self.liquid.massFlowRate

    @cached_property
    def centerOfMass(self):
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
        massBalance = (
            self.solid.mass * self.solid.centerOfMass
            + self.liquid.mass * self.liquid.centerOfMass
        )
        return massBalance / self.mass

    @cached_property
    def inertiaTensor(self):
        """Calculates the propellant principal moment of inertia relative
        to the tank center of mass. The z-axis correspond to the motor axis
        of symmetry while the x and y axes complete the right-handed coordinate
        system. The time derivatives of the products of inertia are also
        evaluated.
        Products of inertia are assumed null due to symmetry. 

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        tuple (of Functions)
            The two first arguments are equivalent and represent inertia Ix,
            and Iy. The third argument is inertia Iz.
        """
        solidCorrection = self.solid.mass * (self.solid.centerOfMass - self.centerOfMass) ** 2
        liquidCorrection = self.liquid.mass * (self.liquid.centerOfMass - self.centerOfMass) ** 2
        
        solidInertia = self.solid.inertiaTensor
        liquidInertia = self.liquid.inertiaTensor

        self.InertiaI = solidInertia[0] + solidCorrection + liquidInertia[0] + liquidCorrection
        self.InertiaZ = solidInertia[2] + solidCorrection + liquidInertia[2] + liquidCorrection

        # Set naming convention
        self.InertiaI.setInputs("time (s)")
        self.InertiaZ.setInputs("time (s)")
        self.InertiaI.setOutputs("inertia y (kg*m^2)")
        self.InertiaZ.setOutputs("inertia z (kg*m^2)")

        return self.InertiaI, self.InertiaI, self.InertiaZ
        


    def allInfo(self):
        """Prints out all data and graphs available about the Motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print nozzle details
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.nozzleRadius) + " m")
        print("Nozzle Throat Radius: " + str(self.throatRadius) + " m")

        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.grainNumber))
        print("Grain Spacing: " + str(self.grainSeparation) + " m")
        print("Grain Density: " + str(self.grainDensity) + " kg/m3")
        print("Grain Outer Radius: " + str(self.grainOuterRadius) + " m")
        print("Grain Inner Radius: " + str(self.grainInitialInnerRadius) + " m")
        print("Grain Height: " + str(self.grainInitialHeight) + " m")
        print("Grain Volume: " + "{:.3f}".format(self.grainInitialVolume) + " m3")
        print("Grain Mass: " + "{:.3f}".format(self.grainInitialMass) + " kg")

        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burnOutTime) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellantInitialMass)
            + " kg"
        )
        print(
            "Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaustVelocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.averageThrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.maxThrust)
            + " N at "
            + str(self.maxThrustTime)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.totalImpulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust()
        self.mass()
        self.massFlowRate()
        self.grainInnerRadius()
        self.grainHeight()
        self.burnRate()
        self.burnArea()
        self.Kn()
        self.inertiaI()
        self.InertiaZ()
        self.inertiaIDot()
        self.inertiaZDot()

        return None

