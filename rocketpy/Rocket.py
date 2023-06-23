# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Franz Masatoshi Yuri, Mateus Stano Junqueira, Kaleb Ramos Wanderley, Calebe Gomes Teles, Matheus Doretto"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import warnings
from inspect import getsourcelines
from collections import namedtuple
from inspect import getsourcelines

import numpy as np

from .Function import Function, funcify_method
from .Parachute import Parachute
from .AeroSurfaces import AeroSurfaces
from .AeroSurfaces import NoseCone, TrapezoidalFins, EllipticalFins, Tail
from .motors.Motor import EmptyMotor

from .prints.rocket_prints import _RocketPrints
from .plots.rocket_plots import _RocketPlots


class Rocket:

    """Keeps all rocket and parachute information.

    Attributes
    ----------
        Geometrical attributes:
        Rocket.radius : float
            Rocket's largest radius in meters.
        Rocket.area : float
            Rocket's circular cross section largest frontal area in squared
            meters.
        Rocket.centerOfDryMassPosition : float
            Position, in m, of the rocket's center of dry mass (i.e. center of mass
            without propellant) relative to the rocket's coordinate system.
            See `Rocket.coordinateSystemOrientation` for more information regarding the
            rocket's coordinate system.
        Rocket.coordinateSystemOrientation : string
            String defining the orientation of the rocket's coordinate system. The
            coordinate system is defined by the rocket's axis of symmetry. The system's
            origin may be placed anywhere along such axis, such as in the nozzle or in
            the nose cone, and must be kept the same for all other positions specified.
            If "tailToNose", the coordinate system is defined with the rocket's axis of
            symmetry pointing from the rocket's tail to the rocket's nose cone.
            If "noseToTail", the coordinate system is defined with the rocket's axis of
            symmetry pointing from the rocket's nose cone to the rocket's tail.

        Mass and Inertia attributes:
        Rocket.mass : float
            Rocket's mass without propellant in kg.
        Rocket.centerOfMass : Function
            Position of the rocket's center of mass, including propellant, relative
            to the user defined rocket reference system.
            See `Rocket.coordinateSystemOrientation` for more information regarding the
            coordinate system.
            Expressed in meters as a function of time.
        Rocket.reducedMass : Function
            Function of time expressing the reduced mass of the rocket,
            defined as the product of the propellant mass and the mass
            of the rocket without propellant, divided by the sum of the
            propellant mass and the rocket mass.
        Rocket.totalMass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        Rocket.thrustToWeight : Function
            Function of time expressing the motor thrust force divided by rocket
            weight. The gravitational acceleration is assumed as 9.80665 m/s^2.

        Eccentricity attributes:
        Rocket.cpEccentricityX : float
            Center of pressure position relative to center of mass in the x
            axis, perpendicular to axis of cylindrical symmetry, in meters.
        Rocket.cpEccentricityY : float
            Center of pressure position relative to center of mass in the y
            axis, perpendicular to axis of cylindrical symmetry, in meters.
        Rocket.thrustEccentricityY : float
            Thrust vector position relative to center of mass in the y
            axis, perpendicular to axis of cylindrical symmetry, in meters.
        Rocket.thrustEccentricityX : float
            Thrust vector position relative to center of mass in the x
            axis, perpendicular to axis of cylindrical symmetry, in meters.

        Aerodynamic attributes
        Rocket.aerodynamicSurfaces : list
            List of aerodynamic surfaces of the rocket.
        Rocket.cpPosition : float
            Rocket's center of pressure position relative to the user defined rocket
            reference system. See `Rocket.coordinateSystemOrientation` for more information
            regarding the reference system.
            Expressed in meters.
        Rocket.staticMargin : float
            Float value corresponding to rocket static margin when
            loaded with propellant in units of rocket diameter or
            calibers.
        Rocket.powerOffDrag : Function
            Rocket's drag coefficient as a function of Mach number when the
            motor is off.
        Rocket.powerOnDrag : Function
            Rocket's drag coefficient as a function of Mach number when the
            motor is on.

        Motor attributes:
        Rocket.motor : Motor
            Rocket's motor. See Motor class for more details.
        Rocket.motorPosition : float
            Position, in m, of the motor's nozzle exit area relative to the user defined
            rocket coordinate system. See `Rocket.coordinateSystemOrientation` for more
            information regarding the rocket's coordinate system.
        Rocket.centerOfPropellantPosition : Function
            Position of the propellant's center of mass relative to the user defined
            rocket reference system. See `Rocket.coordinateSystemOrientation` for more
            information regarding the rocket's coordinate system.
            Expressed in meters as a function of time.
    """

    def __init__(
        self,
        radius,
        mass,
        inertia,
        powerOffDrag,
        powerOnDrag,
        center_of_mass_without_motor,
        coordinateSystemOrientation="tailToNose",
    ):
        """Initializes Rocket class, process inertial, geometrical and
        aerodynamic parameters.

        Parameters
        ----------
        radius : int, float
            Rocket largest outer radius in meters.
        mass : int, float
            Rocket total mass without motor in kg.
        inertia : tuple, list
            Tuple or list containing the rocket's dry mass inertia tensor
            components, in kg*m^2.
            Assuming e_3 is the rocket's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the dry mass
            inertia tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23), where I_ij is the
            component of the inertia tensor in the direction of e_i x e_j.
            Alternatively, the inertia tensor can be given as (I_11, I_22, I_33),
            where I_12 = I_13 = I_23 = 0.
        powerOffDrag : int, float, callable, string, array
            Rocket's drag coefficient when the motor is off. Can be given as an
            entry to the Function class. See help(Function) for more
            information. If int or float is given, it is assumed constant. If
            callable, string or array is given, it must be a function of Mach
            number only.
        powerOnDrag : int, float, callable, string, array
            Rocket's drag coefficient when the motor is on. Can be given as an
            entry to the Function class. See help(Function) for more
            information. If int or float is given, it is assumed constant. If
            callable, string or array is given, it must be a function of Mach
            number only.
        center_of_mass_without_motor : int, float
            Position, in m, of the rocket's center of mass without motor
            relative to the rocket's coordinate system. Default is 0, which
            means the center of dry mass is chosen as the origin, to comply
            with the legacy behavior of versions 0.X.Y.
            See `Rocket.coordinateSystemOrientation` for more information
            regarding the rocket's coordinate system.
        coordinateSystemOrientation : string, optional
            String defining the orientation of the rocket's coordinate system. The
            coordinate system is defined by the rocket's axis of symmetry. The system's
            origin may be placed anywhere along such axis, such as in the nozzle or in
            the nose cone, and must be kept the same for all other positions specified.
            The two options available are: "tailToNose" and "noseToTail". The first
            defines the coordinate system with the rocket's axis of symmetry pointing
            from the rocket's tail to the rocket's nose cone. The second option defines
            the coordinate system with the rocket's axis of symmetry pointing from the
            rocket's nose cone to the rocket's tail. Default is "tailToNose".

        Returns
        -------
        None
        """
        # Define coordinate system orientation
        self.coordinateSystemOrientation = coordinateSystemOrientation
        if coordinateSystemOrientation == "tailToNose":
            self._csys = 1
        elif coordinateSystemOrientation == "noseToTail":
            self._csys = -1

        # Define rocket inertia attributes in SI units
        self.mass = mass
        inertia = (*inertia, 0, 0, 0) if len(inertia) == 3 else inertia
        self.dry_I_11 = inertia[0]
        self.dry_I_22 = inertia[1]
        self.dry_I_33 = inertia[2]
        self.dry_I_12 = inertia[3]
        self.dry_I_13 = inertia[4]
        self.dry_I_23 = inertia[5]

        # Define rocket geometrical parameters in SI units
        self.center_of_mass_without_motor = center_of_mass_without_motor
        self.radius = radius
        self.area = np.pi * self.radius**2

        # Eccentricity data initialization
        self.cpEccentricityX = 0
        self.cpEccentricityY = 0
        self.thrustEccentricityY = 0
        self.thrustEccentricityX = 0

        # Parachute data initialization
        self.parachutes = []

        # Rail button data initialization
        self.railButtons = None

        # Aerodynamic data initialization
        self.aerodynamicSurfaces = AeroSurfaces()
        self.nosecone = AeroSurfaces()
        self.fins = AeroSurfaces()
        self.tail = AeroSurfaces()
        self.cpPosition = 0
        self.staticMargin = Function(
            lambda x: 0, inputs="Time (s)", outputs="Static Margin (c)"
        )

        # Define aerodynamic drag coefficients
        self.powerOffDrag = Function(
            powerOffDrag,
            "Mach Number",
            "Drag Coefficient with Power Off",
            "linear",
            "constant",
        )
        self.powerOnDrag = Function(
            powerOnDrag,
            "Mach Number",
            "Drag Coefficient with Power On",
            "linear",
            "constant",
        )
        self.cpPosition = 0  # Set by self.evaluateStaticMargin()

        # Create a, possibly, temporary empty motor
        self.addMotor(motor=EmptyMotor(), position=0)

        # Important dynamic inertial quantities
        self.centerOfMass = None
        self.reducedMass = None
        self.totalMass = None
        self.dryMass = None

        # Calculate dynamic inertial quantities
        self.evaluateDryMass()
        self.evaluateTotalMass()
        self.evaluateCenterOfDryMass()
        self.evaluateCenterOfMass()
        self.evaluateReducedMass()
        self.evaluateThrustToWeight()

        # Evaluate static margin (even though no aerodynamic surfaces are present yet)
        self.evaluateStaticMargin()

        # Initialize plots and prints object
        self.prints = _RocketPrints(self)
        self.plots = _RocketPlots(self)

        return None

    def evaluateTotalMass(self):
        """Calculates and returns the rocket's total mass. The total
        mass is defined as the sum of the motor mass with propellant and the
        rocket mass without propellant. The function returns an object
        of the Function class and is defined as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.totalMass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Calculate total mass by summing up propellant and dry mass
        self.totalMass = self.mass + self.motor.totalMass
        self.totalMass.setOutputs("Total Mass (Rocket + Propellant) (kg)")

        # Return total mass
        return self.totalMass

    def evaluateDryMass(self):
        """Calculates and returns the rocket's dry mass. The dry
        mass is defined as the sum of the motor's dry mass and the
        rocket mass without motor. The function returns an object
        of the Function class and is defined as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.totalMass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Calculate total mass by summing up propellant and dry mass
        self.dryMass = self.mass + self.motor.dry_mass

        # Return total mass
        return self.dryMass

    def evaluateCenterOfMass(self):
        """Evaluates rocket center of mass position relative to user defined rocket
        reference system.

        Parameters
        ----------
        None

        Returns
        -------
        self.centerOfMass : Function
            Function of time expressing the rocket's center of mass position relative to
            user defined rocket reference system.
            See `Rocket.coordinateSystemOrientation` for more information.
        """
        # Compute center of mass position
        self.centerOfMass = (
            self.center_of_mass_without_motor * self.mass
            + (self.motor.centerOfMass + self.motorPosition) * self.motor.totalMass
        ) / self.totalMass
        self.centerOfMass.setInputs("Time (s)")
        self.centerOfMass.setOutputs("Center of Mass Position (m)")

        return self.centerOfMass

    def evaluateCenterOfDryMass(self):
        """Evaluates rocket center dry of mass (i.e. without propellant)
        position relative to user defined rocket reference system.

        Parameters
        ----------
        None

        Returns
        -------
        self.centerOfDryMass : int, float
            Rocket's center of dry mass position relative to user defined rocket
            reference system. See `Rocket.coordinateSystemOrientation` for more
            information.
        """
        # Compute center of mass position
        self.centerOfDryMassPosition = (
            self.center_of_mass_without_motor * self.mass
            + self.motor.center_of_dry_mass * self.motor.dry_mass
        ) / self.dryMass

        return self.centerOfDryMassPosition

    def evaluateReducedMass(self):
        """Calculates and returns the rocket's total reduced mass. The
        reduced mass is defined as the product of the propellant mass
        and the mass of the rocket without propellant, divided by the
        sum of the propellant mass and the rocket mass. The function
        returns an object of the Function class and is defined as a
        function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.reducedMass : Function
            Function of time expressing the reduced mass of the rocket,
            defined as the product of the propellant mass and the mass
            of the rocket without propellant, divided by the sum of the
            propellant mass and the rocket mass.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Retrieve propellant mass as a function of time
        motorMass = self.motor.propellantMass

        # Retrieve constant rocket mass without propellant
        mass = self.dryMass

        # Calculate reduced mass
        self.reducedMass = motorMass * mass / (motorMass + mass)
        self.reducedMass.setOutputs("Reduced Mass (kg)")

        # Return reduced mass
        return self.reducedMass

    def evaluateThrustToWeight(self):
        """Evaluates thrust to weight as a Function of time.

        Uses g = 9.80665 m/s² as nominal gravity for weight calculation.

        Returns
        -------
        None
        """
        self.thrustToWeight = self.motor.thrust / (9.80665 * self.totalMass)
        self.thrustToWeight.setInputs("Time (s)")
        self.thrustToWeight.setOutputs("Thrust/Weight")

    def evaluateStaticMargin(self):
        """Calculates and returns the rocket's static margin when
        loaded with propellant. The static margin is saved and returned
        in units of rocket diameter or calibers. This function also calculates
        the rocket center of pressure and total lift coefficients.

        Parameters
        ----------
        None

        Returns
        -------
        self.staticMargin : float
            Float value corresponding to rocket static margin when
            loaded with propellant in units of rocket diameter or
            calibers.
        """
        # Initialize total lift coefficient derivative and center of pressure position
        self.totalLiftCoeffDer = 0
        self.cpPosition = 0

        # Calculate total lift coefficient derivative and center of pressure
        if len(self.aerodynamicSurfaces) > 0:
            for aeroSurface, position in self.aerodynamicSurfaces:
                self.totalLiftCoeffDer += Function(
                    lambda alpha: aeroSurface.cl(alpha, 0)
                ).differentiate(x=1e-2, dx=1e-3)
                self.cpPosition += Function(
                    lambda alpha: aeroSurface.cl(alpha, 0)
                ).differentiate(x=1e-2, dx=1e-3) * (
                    position - self._csys * aeroSurface.cpz
                )
            self.cpPosition /= self.totalLiftCoeffDer

        # Calculate static margin
        self.staticMargin = (self.centerOfMass - self.cpPosition) / (2 * self.radius)
        self.staticMargin *= (
            self._csys
        )  # Change sign if coordinate system is upside down
        self.staticMargin.setInputs("Time (s)")
        self.staticMargin.setOutputs("Static Margin (c)")
        self.staticMargin.setDiscrete(
            lower=0, upper=self.motor.burnOutTime, samples=200
        )

        # Return self
        return self

    @funcify_method("Time (s)", "Inertia I_11 (kg m²)")
    def I_11(self):
        """Inertia tensor 11 component, which corresponds to the inertia
        relative to the e_1 axis, centered at the instantaneous center of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total inertia tensor 11 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        rocket axial direction.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Get masses
        prop_mass = self.motor.propellantMass  # Propellant mass as a function of time
        dry_mass = self.dryMass  # Constant rocket dry mass without propellant

        # Compute axes distances
        csys = self._csys
        dry_dz = (self.centerOfMass - self.centerOfDryMassPosition) * csys
        prop_dz = (self.centerOfMass - self.centerOfPropellantPosition) * csys

        # Deconstruct tensor into components
        return (
            self.dry_I_11
            + self.motor.I_11
            + dry_mass * dry_dz**2
            + prop_mass * prop_dz**2
        )

    @funcify_method("Time (s)", "Inertia I_22 (kg m²)")
    def I_22(self):
        """Inertia tensor 22 component, which corresponds to the inertia
        relative to the e_2 axis, centered at the instantaneous center of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total inertia tensor 22 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        rocket axial direction.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Get masses
        prop_mass = self.motor.propellantMass  # Propellant mass as a function of time
        dry_mass = self.dryMass  # Constant rocket dry mass without propellant

        # Compute axes distances
        csys = self._csys
        dry_dz = (self.centerOfMass - self.centerOfDryMassPosition) * csys
        prop_dz = (self.centerOfMass - self.centerOfPropellantPosition) * csys

        # Deconstruct tensor into components
        return (
            self.dry_I_22
            + self.motor.I_22
            + dry_mass * dry_dz**2
            + prop_mass * prop_dz**2
        )

    @funcify_method("Time (s)", "Inertia I_33 (kg m²)")
    def I_33(self):
        """Inertia tensor 33 component, which corresponds to the inertia
        relative to the e_3 axis, centered at the instantaneous center of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total inertia tensor 33 component at time t.

        Notes
        -----
        The e_3 direction is assumed to be the direction parallel to the axis
        of symmetry of the rocket.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.dry_I_33 + self.motor.I_33

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def I_12(self):
        """Inertia tensor 12 component, which corresponds to the inertia
        relative to the e_1 and e_2 axes, centered at the instantaneous center
        of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total inertia tensor 12 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        rocket body axis of symmetry.
        The e_2 direction is assumed to be the direction perpendicular to the
        rocket body axis of symmetry, and perpendicular to e_1.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.dry_I_12 + self.motor.I_12

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def I_13(self):
        """Inertia tensor 13 component, which corresponds to the inertia
        relative to the e_1 and e_3 axes, centered at the instantaneous center
        of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total inertia tensor 13 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        rocket body axis of symmetry.
        The e_3 direction is assumed to be the axial direction of the rocket.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.dry_I_13 + self.motor.I_13

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def I_23(self):
        """Inertia tensor 23 component, which corresponds to the inertia
        relative to the e_2 and e_3 axes, centered at the instantaneous center
        of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total inertia tensor 23 component at time t.

        Notes
        -----
        The e_2 direction is assumed to be the direction perpendicular to the
        rocket body axis of symmetry.
        The e_3 direction is assumed to be the axial direction of the rocket.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.dry_I_23 + self.motor.I_23

    def evaluateNozzleGyrationTensor(self):
        pass

    def addMotor(self, motor, position):
        """Adds a motor to the rocket.

        Parameters
        ----------
        motor : Motor, SolidMotor, HybridMotor, EmptyMotor
            Motor to be added to the rocket. See Motor class for more information.
        position : int, float
            Position, in m, of the motor's nozzle exit area relative to the user defined
            rocket coordinate system. See `Rocket.coordinateSystemOrientation` for more
            information regarding the rocket's coordinate system.

        Returns
        -------
        None
        """
        if hasattr(self, "motor") and not isinstance(self.motor, EmptyMotor):
            print(
                "Only one motor per rocket is currently supported. "
                + "Overwriting previous motor."
            )
        self.motor = motor
        self.motorPosition = position
        _ = self._csys * self.motor._csys
        self.centerOfPropellantPosition = (
            self.motor.centerOfPropellantMass - self.motor.nozzlePosition
        ) * _ + self.motorPosition
        self.evaluateDryMass()
        self.evaluateTotalMass()
        self.evaluateCenterOfDryMass()
        self.evaluateCenterOfMass()
        self.evaluateReducedMass()
        self.evaluateThrustToWeight()
        self.evaluateStaticMargin()
        return None

    def addTail(
        self, topRadius, bottomRadius, length, position, radius=None, name="Tail"
    ):
        """Create a new tail or rocket diameter change, storing its
        parameters as part of the aerodynamicSurfaces list. Its
        parameters are the axial position along the rocket and its
        derivative of the coefficient of lift in respect to angle of
        attack.
        Parameters
        ----------
        topRadius : int, float
            Tail top radius in meters, considering positive direction
            from center of mass to nose cone.
        bottomRadius : int, float
            Tail bottom radius in meters, considering positive direction
            from center of mass to nose cone.
        length : int, float
            Tail length or height in meters. Must be a positive value.
        position : int, float
            Tail position relative to the rocket's coordinate system.
            By tail position, understand the point belonging to the tail which is
            highest in the rocket coordinate system (i.e. generally the point closest
            to the nose cone).
            See `Rocket.coordinateSystemOrientation` for more information.
        Returns
        -------
        tail : Tail
            Tail object created.
        """

        # Modify reference radius if not provided
        radius = self.radius if radius is None else radius

        # Create new tail as an object of the Tail class
        tail = Tail(topRadius, bottomRadius, length, radius, name)
        # Saves position on object for practicality
        tail.position = position

        # Add tail to aerodynamic surfaces and tail list
        self.aerodynamicSurfaces.append(aeroSurface=tail, position=position)
        self.tail.append(tail)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return self
        return tail

    def addNose(self, length, kind, position, name="Nose Cone"):
        """Creates a nose cone, storing its parameters as part of the
        aerodynamicSurfaces list. Its parameters are the axial position
        along the rocket and its derivative of the coefficient of lift
        in respect to angle of attack.


        Parameters
        ----------
        length : int, float
            Nose cone length or height in meters. Must be a positive
            value.
        kind : string
            Nose cone type. Von Karman, conical, ogive, and lvhaack are
            supported.
        position : int, float
            Nose cone tip coordinate relative to the rocket's coordinate system.
            See `Rocket.coordinateSystemOrientation` for more information.
        name : string
            Nose cone name. Default is "Nose Cone".

        Returns
        -------
        nose : Nose
            Nose cone object created.
        """
        # Create a nose as an object of NoseCone class
        nose = NoseCone(length, kind, self.radius, self.radius, name)
        # Saves position on object for practicality
        nose.position = position

        # Add nose to the list of aerodynamic surfaces
        self.aerodynamicSurfaces.append(aeroSurface=nose, position=position)
        self.nosecone.append(nose)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return self
        return nose

    def addFins(self, *args, **kwargs):
        """See Rocket.addTrapezoidalFins for documentation.
        This method is set to be deprecated in version 1.0.0 and fully removed
        by version 2.0.0. Use Rocket.addTrapezoidalFins instead. It keeps the
        same arguments and signature."""
        warnings.warn(
            "This method is set to be deprecated in version 1.0.0 and fully "
            "removed by version 2.0.0. Use Rocket.addTrapezoidalFins instead",
            PendingDeprecationWarning,
        )
        return self.addTrapezoidalFins(*args, **kwargs)

    def addTrapezoidalFins(
        self,
        n,
        rootChord,
        tipChord,
        span,
        position,
        cantAngle=0,
        sweepLength=None,
        sweepAngle=None,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Create a trapezoidal fin set, storing its parameters as part of the
        aerodynamicSurfaces list. Its parameters are the axial position
        along the rocket and its derivative of the coefficient of lift
        in respect to angle of attack.
        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        span : int, float
            Fin span in meters.
        rootChord : int, float
            Fin root chord in meters.
        tipChord : int, float
            Fin tip chord in meters.
        position : int, float
            Fin set position relative to the rocket's coordinate system.
            By fin set position, understand the point belonging to the root chord which
            is highest in the rocket coordinate system (i.e. generally the point closest
            to the nose cone tip).
            See `Rocket.coordinateSystemOrientation` for more information.
        cantAngle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        sweepLength : int, float, optional
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline. If not given, the sweep length is
            assumed to be equal the root chord minus the tip chord, in which case the
            fin is a right trapezoid with its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweepAngle.
        sweepAngle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid with
            its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweepLength.
        radius : int, float, optional
            Reference radius to calculate lift coefficient. If None, which
            is default, use rocket radius.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files must contain no headers
            and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".
        Returns
        -------
        finSet : TrapezoidalFins
            Fin set object created.
        """

        # Modify radius if not given, use rocket radius, otherwise use given.
        radius = radius if radius is not None else self.radius

        # Create a fin set as an object of TrapezoidalFins class
        finSet = TrapezoidalFins(
            n,
            rootChord,
            tipChord,
            span,
            radius,
            cantAngle,
            sweepLength,
            sweepAngle,
            airfoil,
            name,
        )
        # Saves position on object for practicality
        finSet.position = position

        # Add fin set to the list of aerodynamic surfaces
        self.aerodynamicSurfaces.append(aeroSurface=finSet, position=position)
        self.fins.append(finSet)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return the created aerodynamic surface
        return finSet

    def addEllipticalFins(
        self,
        n,
        rootChord,
        span,
        position,
        cantAngle=0,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Create an elliptical fin set, storing its parameters as part of the
        aerodynamicSurfaces list. Its parameters are the axial position
        along the rocket and its derivative of the coefficient of lift
        in respect to angle of attack.
        Parameters
        ----------
        type: string
            Type of fin selected to the rocket. Must be either "trapezoid"
            or "elliptical".
        span : int, float
            Fin span in meters.
        rootChord : int, float
            Fin root chord in meters.
        n : int
            Number of fins, from 2 to infinity.
        position : int, float
            Fin set position relative to the rocket's coordinate system.
            By fin set position, understand the point belonging to the root chord which
            is highest in the rocket coordinate system (i.e. generally the point closest
            to the nose cone tip).
            See `Rocket.coordinateSystemOrientation` for more information.
        cantAngle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        radius : int, float, optional
            Reference radius to calculate lift coefficient. If None, which
            is default, use rocket radius.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files must contain no headers
            and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".
        Returns
        -------
        finSet : EllipticalFins
            Fin set object created.
        """

        # Modify radius if not given, use rocket radius, otherwise use given.
        radius = radius if radius is not None else self.radius

        # Create a fin set as an object of EllipticalFins class
        finSet = EllipticalFins(n, rootChord, span, radius, cantAngle, airfoil, name)
        # Saves position on object for practicality
        finSet.position = position

        # Add fin set to the list of aerodynamic surfaces
        self.aerodynamicSurfaces.append(aeroSurface=finSet, position=position)
        self.fins.append(finSet)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return self
        return finSet

    def addParachute(
        self, name, CdS, trigger, samplingRate=100, lag=0, noise=(0, 0, 0)
    ):
        """Creates a new parachute, storing its parameters such as
        opening delay, drag coefficients and trigger function.

        Parameters
        ----------
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        CdS : float
            Drag coefficient times reference area for parachute. It is
            used to compute the drag force exerted on the parachute by
            the equation F = ((1/2)*rho*V^2)*CdS, that is, the drag
            force is the dynamic pressure computed on the parachute
            times its CdS coefficient. Has units of area and must be
            given in squared meters.
        trigger : function
            Function which defines if the parachute ejection system is
            to be triggered. It must take as input the freestream
            pressure in pascal and the state vector of the simulation,
            which is defined by [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].
            It will be called according to the sampling rate given next.
            It should return True if the parachute ejection system is
            to be triggered and False otherwise.
        samplingRate : float, optional
            Sampling rate in which the trigger function works. It is used to
            simulate the refresh rate of onboard sensors such as barometers.
            Default value is 100. Value must be given in hertz.
        lag : float, optional
            Time between the parachute ejection system is triggered and the
            parachute is fully opened. During this time, the simulation will
            consider the rocket as flying without a parachute. Default value
            is 0. Must be given in seconds.
        noise : tuple, list, optional
            List in the format (mean, standard deviation, time-correlation).
            The values are used to add noise to the pressure signal which is
            passed to the trigger function. Default value is (0, 0, 0). Units
            are in pascal.

        Returns
        -------
        parachute : Parachute
            Parachute  containing trigger, samplingRate, lag, CdS, noise
            and name. Furthermore, it stores cleanPressureSignal,
            noiseSignal and noisyPressureSignal which are filled in during
            Flight simulation.
        """
        # Create a parachute
        parachute = Parachute(name, CdS, trigger, samplingRate, lag, noise)

        # Add parachute to list of parachutes
        self.parachutes.append(parachute)

        # Return self
        return self.parachutes[-1]

    def setRailButtons(self, position, angularPosition=45):
        """Adds rail buttons to the rocket, allowing for the
        calculation of forces exerted by them when the rocket is
        sliding in the launch rail. Furthermore, rail buttons are
        also needed for the simulation of the planar flight phase,
        when the rocket experiences 3 degrees of freedom motion while
        only one rail button is still in the launch rail.

        Parameters
        ----------
        position : tuple, list, array
            Two values organized in a tuple, list or array which
            represent the position of each of the two rail buttons
            in the rocket coordinate system
            The order does not matter. All values should be in meters.
            See `Rocket.coordinateSystemOrientation` for more information.
        angularPosition : float
            Angular position of the rail buttons in degrees measured
            as the rotation around the symmetry axis of the rocket
            relative to one of the other principal axis.
            Default value is 45 degrees, generally used in rockets with
            4 fins.

        Returns
        -------
        None
        """
        # Place top most rail button as the first element of the list
        if self._csys * position[0] < self._csys * position[1]:
            position.reverse()
        # Save important attributes
        self.railButtons = self.railButtonPair(position, angularPosition)

        return None

    def addCMEccentricity(self, x, y):
        """Moves line of action of aerodynamic and thrust forces by
        equal translation amount to simulate an eccentricity in the
        position of the center of mass of the rocket relative to its
        geometrical center line. Should not be used together with
        addCPEccentricity and addThrustEccentricity.

        Parameters
        ----------
        x : float
            Distance in meters by which the CM is to be translated in
            the x direction relative to geometrical center line.
        y : float
            Distance in meters by which the CM is to be translated in
            the y direction relative to geometrical center line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move center of pressure to -x and -y
        self.cpEccentricityX = -x
        self.cpEccentricityY = -y

        # Move thrust center by -x and -y
        self.thrustEccentricityY = -x
        self.thrustEccentricityX = -y

        # Return self
        return self

    def addCPEccentricity(self, x, y):
        """Moves line of action of aerodynamic forces to simulate an
        eccentricity in the position of the center of pressure relative
        to the center of mass of the rocket.

        Parameters
        ----------
        x : float
            Distance in meters by which the CP is to be translated in
            the x direction relative to the center of mass axial line.
        y : float
            Distance in meters by which the CP is to be translated in
            the y direction relative to the center of mass axial line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move center of pressure by x and y
        self.cpEccentricityX = x
        self.cpEccentricityY = y

        # Return self
        return self

    def addThrustEccentricity(self, x, y):
        """Moves line of action of thrust forces to simulate a
        misalignment of the thrust vector and the center of mass.

        Parameters
        ----------
        x : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of mass axial line.
        y : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of mass axial line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move thrust line by x and y
        self.thrustEccentricityY = x
        self.thrustEccentricityX = y

        # Return self
        return self

    def info(self):
        """Prints out a summary of the data and graphs available about
        the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # All prints
        self.prints.all()

        return None

    def allInfo(self):
        """Prints out all data and graphs available about the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # All prints and plots
        self.info()
        self.plots.all()

        return None

    def addFin(
        self,
        numberOfFins=4,
        cl=2 * np.pi,
        cpr=1,
        cpz=1,
        gammas=[0, 0, 0, 0],
        angularPositions=None,
    ):
        "Hey! I will document this function later"
        self.aerodynamicSurfaces = []
        pi = np.pi
        # Calculate angular positions if not given
        if angularPositions is None:
            angularPositions = np.array(range(numberOfFins)) * 2 * pi / numberOfFins
        else:
            angularPositions = np.array(angularPositions) * pi / 180
        # Convert gammas to degree
        if isinstance(gammas, (int, float)):
            gammas = [(pi / 180) * gammas for i in range(numberOfFins)]
        else:
            gammas = [(pi / 180) * gamma for gamma in gammas]
        for i in range(numberOfFins):
            # Get angular position and inclination for current fin
            angularPosition = angularPositions[i]
            gamma = gammas[i]
            # Calculate position vector
            cpx = cpr * np.cos(angularPosition)
            cpy = cpr * np.sin(angularPosition)
            positionVector = np.array([cpx, cpy, cpz])
            # Calculate chord vector
            auxVector = np.array([cpy, -cpx, 0]) / (cpr)
            chordVector = (
                np.cos(gamma) * np.array([0, 0, 1]) - np.sin(gamma) * auxVector
            )
            self.aerodynamicSurfaces.append([positionVector, chordVector])
        return None

    # Variables
    railButtonPair = namedtuple("railButtonPair", "position angularPosition")
