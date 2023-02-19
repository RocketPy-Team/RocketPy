# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano, Pedro Henrique Marinho Bressan, Patrick Bales, Lakshman Peri, Gautam Yarramreddy, Curtis Hu, and William Bradford"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from rocketpy.motors import Motor


class LiquidMotor(Motor):
    """Class to specify characteristics and useful operations for Liquid
    motors.
    """

    def __init__(
        self,
        thrustSource,
        burnOut,
        nozzleRadius,
        throatRadius,
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
        nozzleRadius : int, float, optional
            Motor's nozzle outlet radius in meters. Its value does not impact
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
        """
        super().__init__(
            thrustSource,
            burnOut,
            nozzleRadius,
            throatRadius,
            reshapeThrustCurve,
            interpolationMethod,
        )
        self.positioned_tanks = []

    def evaluateMassFlowRate(self):
        """Evaluates the mass flow rate of the motor as the sum of each tank
        mass flow rate.

        Returns
        -------
        float
            Mass flow rate of the motor, in kg/s.
        """
        massFlowRate = 0

        for positioned_tank in self.positioned_tanks:
            massFlowRate += positioned_tank.get("tank").netMassFlowRate

        return massFlowRate

    def evaluateCenterOfMass(self):
        """Evaluates the center of mass of the motor from each tank center of
        mass and positioning. The center of mass height is measured relative
        to the motor nozzle.

        Returns
        -------
        float
            Center of mass of the motor, in meters.
        """
        totalMass = 0
        massBalance = 0

        for positioned_tank in self.positioned_tanks:
            tank = positioned_tank.get("tank")
            tankPosition = positioned_tank.get("position")
            totalMass += tank.mass
            massBalance += tank.mass * (tankPosition + tank.centerOfMass)

        return massBalance / totalMass

    def evaluateInertiaTensor(self):
        """Evaluates the principal moment of inertia of the motor from each tank
        by the parallel axis theorem. The moment of inertia is measured relative
        to the motor center of mass with the z-axis being the motor symmetry axis
        and the x and y axes completing the right-handed coordinate system.

        Returns
        -------
        tuple
            Pricipal moment of inertia tensor of the motor, in kg*m^2.
        """
        inertia_x = inertia_y = inertia_z = 0
        centerOfMass = self.evaluateCenterOfMass()

        for positioned_tank in self.positioned_tanks:
            tank = positioned_tank.get("tank")
            tankPosition = positioned_tank.get("position")
            inertia_x += (
                tank.inertiaTensor
                + tank.mass * (tankPosition + tank.centerOfMass - centerOfMass) ** 2
            )
            inertia_y = inertia_x

        return inertia_x, inertia_y, inertia_z

    def addTank(self, tank, position):
        """Adds a tank to the rocket motor.

        Parameters
        ----------
        tank : Tank
            Tank object to be added to the rocket motor.
        position : float
            Position of the tank relative to the motor nozzle, in meters.
            Should be a positive value. The position is measured from the
            nozzle tip to the tank lowest point (including caps).
        """
        self.positioned_tanks.append({"tank": tank, "position": position})
