__author__ = " "
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _FlightPrints:
    """Class that holds prints methods for Flight class.

    Attributes
    ----------
    _FlightPrints.flight : Flight
        Flight object that will be used for the prints.

    """

    def __init__(
        self,
        flight,
    ):
        """Initializes _FlightPrints class

        Parameters
        ----------
        flight: Flight
            Instance of the Flight class.

        Returns
        -------
        None
        """
        self.flight = flight
        return None

    def initial_conditions(self):
        """Prints all initial conditions data available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        print("\nInitial Conditions\n")

        # Post-process results
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        print(
            "Position - x: {:.2f} m | y: {:.2f} m | z: {:.2f} m".format(
                self.flight.x(0), self.flight.y(0), self.flight.z(0)
            )
        )
        print(
            "Velocity - Vx: {:.2f} m/s | Vy: {:.2f} m/s | Vz: {:.2f} m/s".format(
                self.flight.vx(0), self.flight.vy(0), self.flight.vz(0)
            )
        )
        print(
            "Attitude - e0: {:.3f} | e1: {:.3f} | e2: {:.3f} | e3: {:.3f}".format(
                self.flight.e0(0),
                self.flight.e1(0),
                self.flight.e2(0),
                self.flight.e3(0),
            )
        )
        print(
            "Euler Angles - Spin φ : {:.2f}° | Nutation θ: {:.2f}° | Precession ψ: {:.2f}°".format(
                self.flight.phi(0), self.flight.theta(0), self.flight.psi(0)
            )
        )
        print(
            "Angular Velocity - ω1: {:.2f} rad/s | ω2: {:.2f} rad/s| ω3: {:.2f} rad/s".format(
                self.flight.w1(0), self.flight.w2(0), self.flight.w3(0)
            )
        )

        return None

    def numerical_integration_settings(self):
        """Prints out the Numerical Integration settings available about the
        flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        print("\nNumerical Integration Settings\n")
        print("Maximum Allowed Flight Time: {:f} s".format(self.flight.maxTime))
        print("Maximum Allowed Time Step: {:f} s".format(self.flight.maxTimeStep))
        print("Minimum Allowed Time Step: {:e} s".format(self.flight.minTimeStep))
        print("Relative Error Tolerance: ", self.flight.rtol)
        print("Absolute Error Tolerance: ", self.flight.atol)
        print("Allow Event Overshoot: ", self.flight.timeOvershoot)
        print("Terminate Simulation on Apogee: ", self.flight.terminateOnApogee)
        print("Number of Time Steps Used: ", len(self.flight.timeSteps))
        print(
            "Number of Derivative Functions Evaluation: ",
            sum(self.flight.functionEvaluationsPerTimeStep),
        )
        print(
            "Average Function Evaluations per Time Step: {:3f}".format(
                sum(self.flight.functionEvaluationsPerTimeStep)
                / len(self.flight.timeSteps)
            )
        )

        return None

    def surface_wind_conditions(self):
        """Prints out the Surface Wind Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        print("\nSurface Wind Conditions\n")
        print(
            "Frontal Surface Wind Speed: {:.2f} m/s".format(
                self.flight.frontalSurfaceWind
            )
        )
        print(
            "Lateral Surface Wind Speed: {:.2f} m/s".format(
                self.flight.lateralSurfaceWind
            )
        )

        return None

    def launch_rail_conditions(self):
        """Prints out the Launch Rail Conditions available about the flight.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        print("\nLaunch Rail Orientation\n")
        print("Launch Rail Inclination: {:.2f}°".format(self.flight.inclination))
        print("Launch Rail Heading: {:.2f}°".format(self.flight.heading))
        return None

    def out_of_rail_conditions(self):
        """Prints out the Out of Rail Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        print("\nRail Departure State\n")
        print("Rail Departure Time: {:.3f} s".format(self.flight.outOfRailTime))
        print(
            "Rail Departure Velocity: {:.3f} m/s".format(self.flight.outOfRailVelocity)
        )
        print(
            "Rail Departure Static Margin: {:.3f} c".format(
                self.flight.staticMargin(self.flight.outOfRailTime)
            )
        )
        print(
            "Rail Departure Angle of Attack: {:.3f}°".format(
                self.flight.angleOfAttack(self.flight.outOfRailTime)
            )
        )
        print(
            "Rail Departure Thrust-Weight Ratio: {:.3f}".format(
                self.flight.rocket.thrustToWeight(self.flight.outOfRailTime)
            )
        )
        print(
            "Rail Departure Reynolds Number: {:.3e}".format(
                self.flight.ReynoldsNumber(self.flight.outOfRailTime)
            )
        )

        return None

    def burn_out_conditions(self):
        """Prints out the Burn Out Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        print("\nBurnOut State\n")
        print("BurnOut time: {:.3f} s".format(self.flight.rocket.motor.burnOutTime))
        print(
            "Altitude at burnOut: {:.3f} m (AGL)".format(
                self.flight.z(self.flight.rocket.motor.burnOutTime)
                - self.flight.env.elevation
            )
        )
        print(
            "Rocket velocity at burnOut: {:.3f} m/s".format(
                self.flight.speed(self.flight.rocket.motor.burnOutTime)
            )
        )
        print(
            "Freestream velocity at burnOut: {:.3f} m/s".format(
                (
                    self.flight.streamVelocityX(self.flight.rocket.motor.burnOutTime)
                    ** 2
                    + self.flight.streamVelocityY(self.flight.rocket.motor.burnOutTime)
                    ** 2
                    + self.flight.streamVelocityZ(self.flight.rocket.motor.burnOutTime)
                    ** 2
                )
                ** 0.5
            )
        )
        print(
            "Mach Number at burnOut: {:.3f}".format(
                self.flight.MachNumber(self.flight.rocket.motor.burnOutTime)
            )
        )
        print(
            "Kinetic energy at burnOut: {:.3e} J".format(
                self.flight.kineticEnergy(self.flight.rocket.motor.burnOutTime)
            )
        )

        return None

    def apogee_conditions(self):
        """Prints out the Apogee Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        print("\nApogee State\n")
        print(
            "Apogee Altitude: {:.3f} m (ASL) | {:.3f} m (AGL)".format(
                self.flight.apogee, self.flight.apogee - self.flight.env.elevation
            )
        )
        print("Apogee Time: {:.3f} s".format(self.flight.apogeeTime))
        print(
            "Apogee Freestream Speed: {:.3f} m/s".format(
                self.flight.apogeeFreestreamSpeed
            )
        )

        return None

    def events_registered(self):
        """Prints out the Events Registered available about the flight.

        Returns
        -------
        None
        """
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        print("\nParachute Events\n")
        if len(self.flight.parachuteEvents) == 0:
            print("No Parachute Events Were Triggered.")
        for event in self.flight.parachuteEvents:
            triggerTime = event[0]
            parachute = event[1]
            openTime = triggerTime + parachute.lag
            velocity = self.flight.freestreamSpeed(openTime)
            altitude = self.flight.z(openTime)
            name = parachute.name.title()
            print(name + " Ejection Triggered at: {:.3f} s".format(triggerTime))
            print(name + " Parachute Inflated at: {:.3f} s".format(openTime))
            print(
                name
                + " Parachute Inflated with Freestream Speed of: {:.3f} m/s".format(
                    velocity
                )
            )
            print(
                name
                + " Parachute Inflated at Height of: {:.3f} m (AGL)".format(
                    altitude - self.flight.env.elevation
                )
            )
        return None

    def impact_conditions(self):
        """Prints out the Impact Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.postProcessed is False:
            self.flight.postProcess()
        if len(self.flight.impactState) != 0:
            print("\nImpact Conditions\n")
            print("X Impact: {:.3f} m".format(self.flight.xImpact))
            print("Y Impact: {:.3f} m".format(self.flight.yImpact))
            print("Time of Impact: {:.3f} s".format(self.flight.tFinal))
            print("Velocity at Impact: {:.3f} m/s".format(self.flight.impactVelocity))
        elif self.flight.terminateOnApogee is False:
            print("End of Simulation")
            print("Time: {:.3f} s".format(self.flight.solution[-1][0]))
            print("Altitude: {:.3f} m".format(self.flight.solution[-1][3]))

        return None

    def maximum_values(self):
        """Prints out the Maximum Values available about the flight.

        Returns
        -------
        None
        """
        print("\nMaximum Values\n")
        print(
            "Maximum Speed: {:.3f} m/s at {:.2f} s".format(
                self.flight.maxSpeed, self.flight.maxSpeedTime
            )
        )
        print(
            "Maximum Mach Number: {:.3f} Mach at {:.2f} s".format(
                self.flight.maxMachNumber, self.flight.maxMachNumberTime
            )
        )
        print(
            "Maximum Reynolds Number: {:.3e} at {:.2f} s".format(
                self.flight.maxReynoldsNumber, self.flight.maxReynoldsNumberTime
            )
        )
        print(
            "Maximum Dynamic Pressure: {:.3e} Pa at {:.2f} s".format(
                self.flight.maxDynamicPressure, self.flight.maxDynamicPressureTime
            )
        )
        print(
            "Maximum Acceleration: {:.3f} m/s² at {:.2f} s".format(
                self.flight.maxAcceleration, self.flight.maxAccelerationTime
            )
        )
        print(
            "Maximum Gs: {:.3f} g at {:.2f} s".format(
                self.flight.maxAcceleration
                / self.flight.env.gravity(
                    self.flight.z(self.flight.maxAccelerationTime)
                ),
                self.flight.maxAccelerationTime,
            )
        )
        print(
            "Maximum Upper Rail Button Normal Force: {:.3f} N".format(
                self.flight.maxRailButton1NormalForce
            )
        )
        print(
            "Maximum Upper Rail Button Shear Force: {:.3f} N".format(
                self.flight.maxRailButton1ShearForce
            )
        )
        print(
            "Maximum Lower Rail Button Normal Force: {:.3f} N".format(
                self.flight.maxRailButton2NormalForce
            )
        )
        print(
            "Maximum Lower Rail Button Shear Force: {:.3f} N".format(
                self.flight.maxRailButton2ShearForce
            )
        )
        return None

    def all(self):
        """Prints out all data available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Print initial conditions
        self.initial_conditions()
        print()

        # Print surface wind conditions
        self.surface_wind_conditions()
        print()

        # Print launch rail orientation
        self.launch_rail_conditions()
        print()

        # Print out of rail conditions
        self.out_of_rail_conditions()
        print()

        # Print burnOut conditions
        self.burn_out_conditions()
        print()

        # Print apogee conditions
        self.apogee_conditions()
        print()

        # Print events registered
        self.events_registered()
        print()

        # Print impact conditions
        self.impact_conditions()
        print()

        # Print maximum values
        self.maximum_values()
        print()

        # Print Numerical Integration Information
        self.numerical_integration_settings()
        print()

        return None
