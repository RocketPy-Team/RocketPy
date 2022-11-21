# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano and Pedro Henrique Marinho Bressan"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


from rocketpy.motors import Motor


class LiquidMotor(Motor):
    def __init__(
        self,
        thrustSource,
        burnOut,
        nozzleRadius,
        throatRadius,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
    ):

        super.__init__()
        self.positioned_tanks = []
        pass

    def evaluateMassFlowRate(self):
        massFlowRate = 0

        for positioned_tank in self.positioned_tanks:
            massFlowRate += positioned_tank.get("tank").netMassFlowRate

        return massFlowRate

    def evaluateCenterOfMass(self):
        totalMass = 0
        massBalance = 0

        for positioned_tank in self.positioned_tanks:
            tank = positioned_tank.get("tank")
            tankPosition = positioned_tank.get("position")
            totalMass += tank.mass
            massBalance += tank.mass * (tankPosition + tank.centerOfMass)

        return massBalance / totalMass

    def evaluateInertiaTensor(self):
        pass

    def addTank(self, tank, position):
        """
        Adds a tank to the rocket motor.

        Parameters
        ----------
        tank : Tank
            Tank object to be added to the rocket motor.
        position : float
            Position of the tank in relation to the motor nozzle, in meters.
            Should be a positive value.
        """
        self.positioned_tanks.append({"tank": tank, "position": position})
