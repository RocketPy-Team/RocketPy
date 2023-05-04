# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import re
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from scipy import integrate

from .Function import Function


class Motor(ABC):
    """Abstract class to specify characteristics and useful operations for
    motors. Cannot be instantiated.

    Attributes
    ----------

        Geometrical attributes:
        Motor.coordinate_system_orientation : str
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as at the
            nozzle area, and must be kept the same for all other positions specified.
            Options are "nozzle_to_combustion_chamber" and "combustion_chamber_to_nozzle".
        Motor.nozzle_radius : float
            Radius of motor nozzle outlet in meters.
        Motor.nozzle_position : float
            Motor's nozzle outlet position in meters. More specifically, the coordinate
            of the nozzle outlet specified in the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
        Motor.throat_radius : float
            Radius of motor nozzle throat in meters.
        Motor.grain_number : int
            Number of solid grains.
        Motor.grain_separation : float
            Distance between two grains in meters.
        Motor.grain_density : float
            Density of each grain in kg/meters cubed.
        Motor.grain_outer_radius : float
            Outer radius of each grain in meters.
        Motor.grain_initial_inner_radius : float
            Initial inner radius of each grain in meters.
        Motor.grain_initial_height : float
            Initial height of each grain in meters.
        Motor.grain_initial_volume : float
            Initial volume of each grain in meters cubed.
        Motor.grain_inner_radius : Function
            Inner radius of each grain in meters as a function of time.
        Motor.grain_height : Function
            Height of each grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.grain_initial_mass : float
            Initial mass of each grain in kg.
        Motor.propellant_initial_mass : float
            Total propellant initial mass in kg.
        Motor.mass : Function
            Propellant total mass in kg as a function of time.
        Motor.mass_dot : Function
            Time derivative of propellant total mass in kg/s as a function
            of time.
        Motor.inertia_i : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis
            perpendicular to axis of cylindrical symmetry of each grain,
            given as a function of time.
        Motor.inertia_i_dot : Function
            Time derivative of inertia_i given in kg*meter^2/s as a function
            of time.
        Motor.inertia_z : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis of
            cylindrical symmetry of each grain, given as a function of time.
        Motor.inertia_z_dot : Function
            Time derivative of inertia_z given in kg*meter^2/s as a function
            of time.

        Thrust and burn attributes:
        Motor.thrust : Function
            Motor thrust force, in Newtons, as a function of time.
        Motor.total_impulse : float
            Total impulse of the thrust curve in N*s.
        Motor.max_thrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.max_thrust_time : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.average_thrust : float
            Average thrust of the motor, given in N.
        Motor.burn_out_time : float
            Total motor burn out time, in seconds. Must include delay time
            when the motor takes time to ignite. Also seen as time to end thrust
            curve.
        Motor.exhaust_velocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burn_area : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burn_area divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burn_rate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrust_source,
        burn_out,
        nozzle_radius=0.0335,
        nozzle_position=0,
        throat_radius=0.0114,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, array
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Thrust units are Newtons.
        burn_out : int, float
            Motor burn out time in seconds.
        nozzle_radius : int, float, optional
            Motor's nozzle outlet radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not impact
            trajectory simulation.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters. More specifically, the coordinate
            of the nozzle outlet specified in the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
            Default is 0, in which case the origin of the motor's coordinate system
            is placed at the motor's nozzle outlet.
        throat_radius : int, float, optional
            Motor's nozzle throat radius in meters. Its value has very low
            impact in trajectory simulation, only useful to analyze
            dynamic instabilities, therefore it is optional.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolation_method : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
        coordinate_system_orientation : string, optional
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as at the
            nozzle area, and must be kept the same for all other positions specified.
            Options are "nozzle_to_combustion_chamber" and "combustion_chamber_to_nozzle".
            Default is "nozzle_to_combustion_chamber".

        Returns
        -------
        None
        """
        # Define coordinate system orientation
        self.coordinate_system_orientation = coordinate_system_orientation
        if coordinate_system_orientation == "nozzle_to_combustion_chamber":
            self._csys = 1
        elif coordinate_system_orientation == "combustion_chamber_to_nozzle":
            self._csys = -1

        # Thrust parameters
        self.interpolate = interpolation_method
        self.burn_out_time = burn_out

        # Check if thrust_source is csv, eng, function or other
        if isinstance(thrust_source, str):
            # Determine if csv or eng
            if thrust_source[-3:] == "eng":
                # Import content
                comments, desc, points = self.import_eng(thrust_source)
                # Process description and points
                # diameter = float(desc[1])/1000
                # height = float(desc[2])/1000
                # mass = float(desc[4])
                # nozzle_radius = diameter/4
                # throat_radius = diameter/8
                # grain_number = grain_number
                # grainVolume = height*np.pi*((diameter/2)**2 -(diameter/4)**2)
                # grain_density = mass/grainVolume
                # grain_outer_radius = diameter/2
                # grain_initial_inner_radius = diameter/4
                # grain_initial_height = height
                thrust_source = points
                self.burn_out_time = points[-1][0]

        # Create thrust function
        self.thrust = Function(
            thrust_source, "Time (s)", "Thrust (N)", self.interpolate, "zero"
        )
        if callable(thrust_source) or isinstance(thrust_source, (int, float)):
            self.thrust.set_discrete(0, burn_out, 50, self.interpolate, "zero")

        # Reshape curve and calculate impulse
        if reshape_thrust_curve:
            self.reshape_thrust_curve(*reshape_thrust_curve)
        else:
            self.evaluate_total_impulse()

        # Define motor attributes
        # Grain and nozzle parameters
        self.nozzle_radius = nozzle_radius
        self.nozzle_position = nozzle_position
        self.throat_radius = throat_radius

        # Other quantities that will be computed
        self.mass_dot = None
        self.mass = None
        self.inertia_i = None
        self.inertia_i_dot = None
        self.inertia_z = None
        self.inertia_z_dot = None
        self.max_thrust = None
        self.max_thrust_time = None
        self.average_thrust = None

        # Compute quantities
        # Thrust information - maximum and average
        self.max_thrust = np.amax(self.thrust.source[:, 1])
        max_thrust_index = np.argmax(self.thrust.source[:, 1])
        self.max_thrust_time = self.thrust.source[max_thrust_index, 0]
        self.average_thrust = self.total_impulse / self.burn_out_time

        self.propellant_initial_mass = None

    def reshape_thrust_curve(
        self, burn_time, total_impulse, old_total_impulse=None, start_at_zero=True
    ):
        """Transforms the thrust curve supplied by changing its total
        burn time and/or its total impulse, without altering the
        general shape of the curve. May translate the curve so that
        thrust starts at time equals 0, without any delays.

        Parameters
        ----------
        burn_time : float
            New desired burn out time in seconds.
        total_impulse : float
            New desired total impulse.
        old_total_impulse : float, optional
            Specify the total impulse of the given thrust curve,
            overriding the value calculated by numerical integration.
            If left as None, the value calculated by numerical
            integration will be used in order to reshape the curve.
        start_at_zero: bool, optional
            If True, trims the initial thrust curve points which
            are 0 Newtons, translating the thrust curve so that
            thrust starts at time equals 0. If False, no translation
            is applied.

        Returns
        -------
        None
        """
        # Retrieve current thrust curve data points
        time_array = self.thrust.source[:, 0]
        thrust_array = self.thrust.source[:, 1]
        # Move start to time = 0
        if start_at_zero and time_array[0] != 0:
            time_array = time_array - time_array[0]

        # Reshape time - set burn time to burn_time
        self.thrust.source[:, 0] = (burn_time / time_array[-1]) * time_array
        self.burn_out_time = burn_time
        self.thrust.set_interpolation(self.interpolate)

        # Reshape thrust - set total impulse
        if old_total_impulse is None:
            old_total_impulse = self.evaluate_total_impulse()
        self.thrust.source[:, 1] = (total_impulse / old_total_impulse) * thrust_array
        self.thrust.set_interpolation(self.interpolate)

        # Store total impulse
        self.total_impulse = total_impulse

        # Return reshaped curve
        return self.thrust

    def evaluate_total_impulse(self):
        """Calculates and returns total impulse by numerical
        integration of the thrust curve in SI units. The value is
        also stored in self.total_impulse.

        Parameters
        ----------
        None

        Returns
        -------
        self.total_impulse : float
            Motor total impulse in Ns.
        """
        # Calculate total impulse
        self.total_impulse = self.thrust.integral(0, self.burn_out_time)

        # Return total impulse
        return self.total_impulse

    @abstractproperty
    def exhaust_velocity(self):
        """Calculates and returns exhaust velocity by assuming it
        as a constant. The formula used is total impulse/propellant
        initial mass. The value is also stored in
        self.exhaust_velocity.

        Parameters
        ----------
        None

        Returns
        -------
        self.exhaust_velocity : float
            Constant gas exhaust velocity of the motor.
        """
        pass

    @abstractmethod
    def evaluate_mass_dot(self):
        """Calculates and returns the time derivative of propellant
        mass by assuming constant exhaust velocity. The formula used
        is the opposite of thrust divided by exhaust velocity. The
        result is a function of time, object of the Function class,
        which is stored in self.mass_dot.

        Parameters
        ----------
        None

        Returns
        -------
        self.mass_dot : Function
            Time derivative of total propellant mas as a function
            of time.
        """
        pass

    @abstractmethod
    def evaluate_center_of_mass(self):
        """Calculates and returns the time derivative of motor center of mass.
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

        pass

    def evaluate_mass(self):
        """Calculates and returns the total propellant mass curve by
        numerically integrating the mass_dot curve, calculated in
        evaluate_mass_dot. Numerical integration is done with the
        Trapezoidal Rule, giving the same result as scipy.integrate.
        odeint but 100x faster. The result is a function of time,
        object of the class Function, which is stored in self.mass.

        Parameters
        ----------
        None

        Returns
        -------
        self.mass : Function
            Total propellant mass as a function of time.
        """
        # Retrieve mass dot curve data
        t = self.mass_dot.source[:, 0]
        ydot = self.mass_dot.source[:, 1]

        # Set initial conditions
        T = [0]
        y = [self.propellant_initial_mass]

        # Solve for each time point
        for i in range(1, len(t)):
            T += [t[i]]
            y += [y[i - 1] + 0.5 * (t[i] - t[i - 1]) * (ydot[i] + ydot[i - 1])]

        # Create Function
        self.mass = Function(
            np.concatenate(([T], [y])).transpose(),
            "Time (s)",
            "Propellant Total Mass (kg)",
            self.interpolate,
            "constant",
        )

        # Return Mass Function
        return self.mass

    @property
    def throat_area(self):
        return np.pi * self.throat_radius**2

    @abstractmethod
    def evaluate_inertia(self):
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

        pass

    def import_eng(self, file_name):
        """Read content from .eng file and process it, in order to
        return the comments, description and data points.

        Parameters
        ----------
        file_name : string
            Name of the .eng file. E.g. 'test.eng'.
            Note that the .eng file must not contain the 0 0 point.

        Returns
        -------
        comments : list
            All comments in the .eng file, separated by line in a list. Each
            line is an entry of the list.
        description: list
            Description of the motor. All attributes are returned separated in
            a list. E.g. "F32 24 124 5-10-15 .0377 .0695 RV\n" is return as
            ['F32', '24', '124', '5-10-15', '.0377', '.0695', 'RV\n']
        dataPoints: list
            List of all data points in file. Each data point is an entry in
            the returned list and written as a list of two entries.
        """

        # Initialize arrays
        comments = []
        description = []
        dataPoints = [[0, 0]]

        # Open and read .eng file
        with open(file_name) as file:
            for line in file:
                if re.search(r";.*", line):
                    # Extract comment
                    comments.append(re.findall(r";.*", line)[0])
                    line = re.sub(r";.*", "", line)
                if line.strip():
                    if description == []:
                        # Extract description
                        description = line.strip().split(" ")
                    else:
                        # Extract thrust curve data points
                        time, thrust = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        dataPoints.append([float(time), float(thrust)])

        # Return all extract content
        return comments, description, dataPoints

    def export_eng(self, file_name, motor_name):
        """Exports thrust curve data points and motor description to
        .eng file format. A description of the format can be found
        here: http://www.thrustcurve.org/raspformat.shtml

        Parameters
        ----------
        file_name : string
            Name of the .eng file to be exported. E.g. 'test.eng'
        motor_name : string
            Name given to motor. Will appear in the description of the
            .eng file. E.g. 'Mandioca'

        Returns
        -------
        None
        """
        # Open file
        file = open(file_name, "w")

        # Write first line
        file.write(
            motor_name
            + " {:3.1f} {:3.1f} 0 {:2.3} {:2.3} RocketPy\n".format(
                2000 * self.grain_outer_radius,
                1000
                * self.grain_number
                * (self.grain_initial_height + self.grain_separation),
                self.propellant_initial_mass,
                self.propellant_initial_mass,
            )
        )

        # Write thrust curve data points
        for time, thrust in self.thrust.source[1:-1, :]:
            # time, thrust = item
            file.write("{:.4f} {:.3f}\n".format(time, thrust))

        # Write last line
        file.write("{:.4f} {:.3f}\n".format(self.thrust.source[-1, 0], 0))

        # Close file
        file.close()

        return None

    def info(self):
        """Prints out a summary of the data and graphs available about
        the Motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burn_out_time) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellant_initial_mass)
            + " kg"
        )
        print(
            "Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaust_velocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.average_thrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.max_thrust)
            + " N at "
            + str(self.max_thrust_time)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.total_impulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust()

        return None

    def allinfo(self):
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
        print("Nozzle Radius: " + str(self.nozzle_radius) + " m")
        print("Nozzle Throat Radius: " + str(self.throat_radius) + " m")

        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.grain_number))
        print("Grain Spacing: " + str(self.grain_separation) + " m")
        print("Grain Density: " + str(self.grain_density) + " kg/m3")
        print("Grain Outer Radius: " + str(self.grain_outer_radius) + " m")
        print("Grain Inner Radius: " + str(self.grain_initial_inner_radius) + " m")
        print("Grain Height: " + str(self.grain_initial_height) + " m")
        print("Grain Volume: " + "{:.3f}".format(self.grain_initial_volume) + " m3")
        print("Grain Mass: " + "{:.3f}".format(self.grain_initial_mass) + " kg")

        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burn_out_time) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellant_initial_mass)
            + " kg"
        )
        print(
            "Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaust_velocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.average_thrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.max_thrust)
            + " N at "
            + str(self.max_thrust_time)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.total_impulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust()
        self.mass()
        self.mass_dot()
        self.grain_inner_radius()
        self.grain_height()
        self.burn_rate()
        self.burn_area()
        self.Kn()
        self.inertia_i()
        self.inertia_i_dot()
        self.inertia_z()
        self.inertia_z_dot()

        return None


class SolidMotor(Motor):
    """Class to specify characteristics and useful operations for solid
    motors.

    Attributes
    ----------

        Geometrical attributes:
        Motor.coordinate_system_orientation : str
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as at the
            nozzle area, and must be kept the same for all other positions specified.
            Options are "nozzle_to_combustion_chamber" and "combustion_chamber_to_nozzle".
        Motor.nozzle_radius : float
            Radius of motor nozzle outlet in meters.
        Motor.nozzle_position : float
            Motor's nozzle outlet position in meters. More specifically, the coordinate
            of the nozzle outlet specified in the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
        Motor.throat_radius : float
            Radius of motor nozzle throat in meters.
        Motor.grain_number : int
            Number of solid grains.
        Motor.grains_center_of_mass_position : float
            Position of the center of mass of the grains in meters. More specifically,
            the coordinate of the center of mass specified in the motor's coordinate
            system. See `Motor.coordinate_system_orientation` for more information.
        Motor.grain_separation : float
            Distance between two grains in meters.
        Motor.grain_density : float
            Density of each grain in kg/meters cubed.
        Motor.grain_outer_radius : float
            Outer radius of each grain in meters.
        Motor.grain_initial_inner_radius : float
            Initial inner radius of each grain in meters.
        Motor.grain_initial_height : float
            Initial height of each grain in meters.
        Motor.grain_initial_volume : float
            Initial volume of each grain in meters cubed.
        Motor.grain_inner_radius : Function
            Inner radius of each grain in meters as a function of time.
        Motor.grain_height : Function
            Height of each grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.center_of_mass : Function
            Position of the center of mass in meters as a function of time. Constant for
            solid motors, as the grains are assumed to be fixed.
            See `Motor.coordinate_system_orientation` for more information regarding
            the motor's coordinate system
        Motor.grain_initial_mass : float
            Initial mass of each grain in kg.
        Motor.propellant_initial_mass : float
            Total propellant initial mass in kg.
        Motor.mass : Function
            Propellant total mass in kg as a function of time.
        Motor.mass_dot : Function
            Time derivative of propellant total mass in kg/s as a function
            of time.
        Motor.inertia_i : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis
            perpendicular to axis of cylindrical symmetry of each grain,
            given as a function of time.
        Motor.inertia_i_dot : Function
            Time derivative of inertia_i given in kg*meter^2/s as a function
            of time.
        Motor.inertia_z : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis of
            cylindrical symmetry of each grain, given as a function of time.
        Motor.inertia_dot : Function
            Time derivative of inertia_z given in kg*meter^2/s as a function
            of time.

        Thrust and burn attributes:
        Motor.thrust : Function
            Motor thrust force, in Newtons, as a function of time.
        Motor.total_impulse : float
            Total impulse of the thrust curve in N*s.
        Motor.max_thrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.max_thrust_time : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.average_thrust : float
            Average thrust of the motor, given in N.
        Motor.burn_out_time : float
            Total motor burn out time, in seconds. Must include delay time
            when the motor takes time to ignite. Also seen as time to end thrust
            curve.
        Motor.exhaust_velocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burn_area : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burn_area divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burn_rate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrust_source,
        burn_out,
        grains_center_of_mass_position,
        grain_number,
        grain_density,
        grain_outer_radius,
        grain_initial_inner_radius,
        grain_initial_height,
        grain_separation=0,
        nozzle_radius=0.0335,
        nozzle_position=0,
        throat_radius=0.0114,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, array
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Thrust units are Newtons.
        burn_out : int, float
            Motor burn out time in seconds.
        grains_center_of_mass_position : float
            Position of the center of mass of the grains in meters. More specifically,
            the coordinate of the center of mass specified in the motor's coordinate
            system. See `Motor.coordinate_system_orientation` for more information.
        grain_number : int
            Number of solid grains
        grain_density : int, float
            Solid grain density in kg/m3.
        grain_outer_radius : int, float
            Solid grain outer radius in meters.
        grain_initial_inner_radius : int, float
            Solid grain initial inner radius in meters.
        grain_initial_height : int, float
            Solid grain initial height in meters.
        grain_separation : int, float, optional
            Distance between grains, in meters. Default is 0.
        nozzle_radius : int, float, optional
            Motor's nozzle outlet radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not impact
            trajectory simulation.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters. More specifically, the coordinate
            of the nozzle outlet specified in the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
            Default is 0, in which case the origin of the motor's coordinate system
            is placed at the motor's nozzle outlet.
        throat_radius : int, float, optional
            Motor's nozzle throat radius in meters. Its value has very low
            impact in trajectory simulation, only useful to analyze
            dynamic instabilities, therefore it is optional.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolation_method : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
        coordinate_system_orientation : string, optional
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as at the
            nozzle area, and must be kept the same for all other positions specified.
            Options are "nozzle_to_combustion_chamber" and "combustion_chamber_to_nozzle".
            Default is "nozzle_to_combustion_chamber".

        Returns
        -------
        None
        """
        super().__init__(
            thrust_source,
            burn_out,
            nozzle_radius,
            nozzle_position,
            throat_radius,
            reshape_thrust_curve,
            interpolation_method,
            coordinate_system_orientation,
        )
        # Define motor attributes
        # Grain parameters
        self.grains_center_of_mass_position = grains_center_of_mass_position
        self.grain_number = grain_number
        self.grain_separation = grain_separation
        self.grain_density = grain_density
        self.grain_outer_radius = grain_outer_radius
        self.grain_initial_inner_radius = grain_initial_inner_radius
        self.grain_initial_height = grain_initial_height

        # Other quantities that will be computed
        self.grain_inner_radius = None
        self.grain_height = None
        self.burn_area = None
        self.burn_rate = None
        self.Kn = None

        # Grains initial geometrical parameters
        self.grain_initial_volume = (
            self.grain_initial_height
            * np.pi
            * (self.grain_outer_radius**2 - self.grain_initial_inner_radius**2)
        )
        self.grain_initial_mass = self.grain_density * self.grain_initial_volume
        self.propellant_initial_mass = self.grain_number * self.grain_initial_mass

        # Dynamic quantities
        self.evaluate_mass_dot()
        self.evaluate_mass()
        self.evaluate_geometry()
        self.evaluate_inertia()
        self.evaluate_center_of_mass()

    @property
    def exhaust_velocity(self):
        """Calculates and returns exhaust velocity by assuming it
        as a constant. The formula used is total impulse/propellant
        initial mass. The value is also stored in
        self.exhaust_velocity.

        Parameters
        ----------
        None

        Returns
        -------
        self.exhaust_velocity : float
            Constant gas exhaust velocity of the motor.
        """
        return self.total_impulse / self.propellant_initial_mass

    def evaluate_mass_dot(self):
        """Calculates and returns the time derivative of propellant
        mass by assuming constant exhaust velocity. The formula used
        is the opposite of thrust divided by exhaust velocity. The
        result is a function of time, object of the Function class,
        which is stored in self.mass_dot.

        Parameters
        ----------
        None

        Returns
        -------
        self.mass_dot : Function
            Time derivative of total propellant mas as a function
            of time.
        """
        # Create mass dot Function
        self.mass_dot = self.thrust / (-self.exhaust_velocity)
        self.mass_dot.set_outputs("Mass Dot (kg/s)")
        self.mass_dot.set_extrapolation("zero")

        # Return Function
        return self.mass_dot

    def evaluate_center_of_mass(self):
        """Calculates and returns the time derivative of motor center of mass.
        The result is a function of time, object of the Function class, which is stored in self.zCM.

        Parameters
        ----------
        None

        Returns
        -------
        self.center_of_mass : Function
            Position of the center of mass as a function of time. Constant for solid
            motors, as the grains are assumed to be fixed.
        """

        self.center_of_mass = Function(
            self.grains_center_of_mass_position, "Time (s)", "Center of Mass (m)"
        )

        return self.center_of_mass

    def evaluate_geometry(self):
        """Calculates grain inner radius and grain height as a
        function of time by assuming that every propellant mass
        burnt is exhausted. In order to do that, a system of
        differential equations is solved using scipy.integrate.
        odeint. Furthermore, the function calculates burn area,
        burn rate and Kn as a function of time using the previous
        results. All functions are stored as objects of the class
        Function in self.grain_inner_radius, self.grain_height, self.
        burn_area, self.burn_rate and self.Kn.

        Parameters
        ----------
        None

        Returns
        -------
        geometry : list of Functions
            First element is the Function representing the inner
            radius of a grain as a function of time. Second
            argument is the Function representing the height of a
            grain as a function of time.
        """
        # Define initial conditions for integration
        y0 = [self.grain_initial_inner_radius, self.grain_initial_height]

        # Define time mesh
        t = self.mass_dot.source[:, 0]

        density = self.grain_density
        rO = self.grain_outer_radius

        # Define system of differential equations
        def geometry_dot(y, t):
            grain_mass_dot = self.mass_dot(t) / self.grain_number
            rI, h = y
            rIDot = (
                -0.5 * grain_mass_dot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            )
            hDot = (
                1.0 * grain_mass_dot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            )
            return [rIDot, hDot]

        # Solve the system of differential equations
        sol = integrate.odeint(geometry_dot, y0, t)

        # Write down functions for innerRadius and height
        self.grain_inner_radius = Function(
            np.concatenate(([t], [sol[:, 0]])).transpose().tolist(),
            "Time (s)",
            "Grain Inner Radius (m)",
            self.interpolate,
            "constant",
        )
        self.grain_height = Function(
            np.concatenate(([t], [sol[:, 1]])).transpose().tolist(),
            "Time (s)",
            "Grain Height (m)",
            self.interpolate,
            "constant",
        )

        # Create functions describing burn rate, Kn and burn area
        self.evaluate_burn_area()
        self.evaluate_kn()
        self.evaluate_burn_rate()

        return [self.grain_inner_radius, self.grain_height]

    def evaluate_burn_area(self):
        """Calculates the burn_area of the grain for
        each time. Assuming that the grains are cylindrical
        BATES grains.

        Parameters
        ----------
        None

        Returns
        -------
        burn_area : Function
        Function representing the burn area progression with the time.
        """
        self.burn_area = (
            2
            * np.pi
            * (
                self.grain_outer_radius**2
                - self.grain_inner_radius**2
                + self.grain_inner_radius * self.grain_height
            )
            * self.grain_number
        )
        self.burn_area.set_outputs("Burn Area (m2)")
        return self.burn_area

    def evaluate_burn_rate(self):
        """Calculates the burn_rate with respect to time.
        This evaluation assumes that it was already
        calculated the mass_dot, burn_area timeseries.

        Parameters
        ----------
        None

        Returns
        -------
        burn_rate : Function
        Rate of progression of the inner radius during the combustion.
        """
        self.burn_rate = (-1) * self.mass_dot / (self.burn_area * self.grain_density)
        self.burn_rate.set_outputs("Burn Rate (m/s)")
        return self.burn_rate

    def evaluate_kn(self):
        KnSource = (
            np.concatenate(
                (
                    [self.grain_inner_radius.source[:, 1]],
                    [self.burn_area.source[:, 1] / self.throat_area],
                )
            ).transpose()
        ).tolist()
        self.Kn = Function(
            KnSource,
            "Grain Inner Radius (m)",
            "Kn (m2/m2)",
            self.interpolate,
            "constant",
        )
        return self.Kn

    def evaluate_inertia(self):
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

        # Inertia I
        # Calculate inertia I for each grain
        grain_mass = self.mass / self.grain_number
        grain_mass_dot = self.mass_dot / self.grain_number
        grain_number = self.grain_number
        grain_inertia_i = grain_mass * (
            (1 / 4) * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
            + (1 / 12) * self.grain_height**2
        )

        # Calculate each grain's distance d to propellant center of mass
        initial_value = (grain_number - 1) / 2
        d = np.linspace(-initial_value, initial_value, grain_number)
        d = d * (self.grain_initial_height + self.grain_separation)

        # Calculate inertia for all grains
        self.inertia_i = grain_number * grain_inertia_i + grain_mass * np.sum(d**2)
        self.inertia_i.set_outputs("Propellant Inertia I (kg*m2)")

        # Inertia I Dot
        # Calculate each grain's inertia I dot
        grain_inertia_i_dot = (
            grain_mass_dot
            * (
                (1 / 4) * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
                + (1 / 12) * self.grain_height**2
            )
            + grain_mass
            * ((1 / 2) * self.grain_inner_radius - (1 / 3) * self.grain_height)
            * self.burn_rate
        )

        # Calculate inertia I dot for all grains
        self.inertia_i_dot = (
            grain_number * grain_inertia_i_dot + grain_mass_dot * np.sum(d**2)
        )
        self.inertia_i_dot.set_outputs("Propellant Inertia I Dot (kg*m2/s)")

        # Inertia Z
        self.inertia_z = (
            (1 / 2.0)
            * self.mass
            * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
        )
        self.inertia_z.set_outputs("Propellant Inertia Z (kg*m2)")

        # Inertia Z Dot
        self.inertia_z_dot = (1 / 2.0) * self.mass_dot * (
            self.grain_outer_radius**2 + self.grain_inner_radius**2
        ) + self.mass * self.grain_inner_radius * self.burn_rate
        self.inertia_z_dot.set_outputs("Propellant Inertia Z Dot (kg*m2/s)")

        return [self.inertia_i, self.inertia_z]

    def allinfo(self):
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
        print("Nozzle Radius: " + str(self.nozzle_radius) + " m")
        print("Nozzle Throat Radius: " + str(self.throat_radius) + " m")

        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.grain_number))
        print("Grain Spacing: " + str(self.grain_separation) + " m")
        print("Grain Density: " + str(self.grain_density) + " kg/m3")
        print("Grain Outer Radius: " + str(self.grain_outer_radius) + " m")
        print("Grain Inner Radius: " + str(self.grain_initial_inner_radius) + " m")
        print("Grain Height: " + str(self.grain_initial_height) + " m")
        print("Grain Volume: " + "{:.3f}".format(self.grain_initial_volume) + " m3")
        print("Grain Mass: " + "{:.3f}".format(self.grain_initial_mass) + " kg")

        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burn_out_time) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellant_initial_mass)
            + " kg"
        )
        print(
            "Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaust_velocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.average_thrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.max_thrust)
            + " N at "
            + str(self.max_thrust_time)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.total_impulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust()
        self.mass()
        self.mass_dot()
        self.grain_inner_radius()
        self.grain_height()
        self.burn_rate()
        self.burn_area()
        self.Kn()
        self.inertia_i()
        self.inertia_i_dot()
        self.inertia_z()
        self.inertia_z_dot()

        return None


class HybridMotor(Motor):
    """Class to specify characteristics and useful operations for Hybrid
    motors.

    Attributes
    ----------

        Geometrical attributes:
        Motor.coordinate_system_orientation : str
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as at the
            nozzle area, and must be kept the same for all other positions specified.
            Options are "nozzle_to_combustion_chamber" and "combustion_chamber_to_nozzle".
        Motor.nozzle_radius : float
            Radius of motor nozzle outlet in meters.
        Motor.nozzle_position : float
            Motor's nozzle outlet position in meters. More specifically, the coordinate
            of the nozzle outlet specified in the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
        Motor.throat_radius : float
            Radius of motor nozzle throat in meters.
        Motor.grain_number : int
            Number of solid grains.
        Motor.grain_separation : float
            Distance between two grains in meters.
        Motor.grain_density : float
            Density of each grain in kg/meters cubed.
        Motor.grain_outer_radius : float
            Outer radius of each grain in meters.
        Motor.grain_initial_inner_radius : float
            Initial inner radius of each grain in meters.
        Motor.grain_initial_height : float
            Initial height of each grain in meters.
        Motor.grain_initial_volume : float
            Initial volume of each grain in meters cubed.
        Motor.grain_inner_radius : Function
            Inner radius of each grain in meters as a function of time.
        Motor.grain_height : Function
            Height of each grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.grain_initial_mass : float
            Initial mass of each grain in kg.
        Motor.propellant_initial_mass : float
            Total propellant initial mass in kg.
        Motor.mass : Function
            Propellant total mass in kg as a function of time.
        Motor.mass_dot : Function
            Time derivative of propellant total mass in kg/s as a function
            of time.
        Motor.inertia_i : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis
            perpendicular to axis of cylindrical symmetry of each grain,
            given as a function of time.
        Motor.inertia_i_dot : Function
            Time derivative of inertia_i given in kg*meter^2/s as a function
            of time.
        Motor.inertia_z : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis of
            cylindrical symmetry of each grain, given as a function of time.
        Motor.inertia_dot : Function
            Time derivative of inertia_z given in kg*meter^2/s as a function
            of time.

        Thrust and burn attributes:
        Motor.thrust : Function
            Motor thrust force, in Newtons, as a function of time.
        Motor.total_impulse : float
            Total impulse of the thrust curve in N*s.
        Motor.max_thrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.max_thrust_time : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.average_thrust : float
            Average thrust of the motor, given in N.
        Motor.burn_out_time : float
            Total motor burn out time, in seconds. Must include delay time
            when the motor takes time to ignite. Also seen as time to end thrust
            curve.
        Motor.exhaust_velocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burn_area : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burn_area divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burn_rate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrust_source,
        burn_out,
        grain_number,
        grain_density,
        grain_outer_radius,
        grain_initial_inner_radius,
        grain_initial_height,
        oxidizer_tank_radius,
        oxidizer_tank_height,
        oxidizer_initial_pressure,
        oxidizer_density,
        oxidizer_molar_mass,
        oxidizer_initial_volume,
        distance_grain_to_tank,
        injector_area,
        grain_separation=0,
        nozzle_radius=0.0335,
        nozzle_position=0,
        throat_radius=0.0114,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, array
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Thrust units are Newtons.
        burn_out : int, float
            Motor burn out time in seconds.
        grain_number : int
            Number of solid grains
        grain_density : int, float
            Solid grain density in kg/m3.
        grain_outer_radius : int, float
            Solid grain outer radius in meters.
        grain_initial_inner_radius : int, float
            Solid grain initial inner radius in meters.
        grain_initial_height : int, float
            Solid grain initial height in meters.
        oxidizer_tank_radius :
            Oxidizer Tank inner radius.
        oxidizer_tank_height :
            Oxidizer Tank Height.
        oxidizer_initial_pressure :
            Initial pressure of the oxidizer tank, could be equal to the pressure of the source cylinder in atm.
        oxidizer_density :
            Oxidizer theoretical density in liquid state, for N2O is equal to 1.98 (Kg/m^3).
        oxidizer_molar_mass :
            Oxidizer molar mass, for the N2O is equal to 44.01 (g/mol).
        oxidizer_initial_volume :
            Initial volume of oxidizer charged in the tank.
        distance_grain_to_tank :
            Distance between the solid grain center of mass and the base of the oxidizer tank.
        injector_area :
            injector outlet area.
        grain_separation : int, float, optional
            Distance between grains, in meters. Default is 0.
        nozzle_radius : int, float, optional
            Motor's nozzle outlet radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not impact
            trajectory simulation.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters. More specifically, the coordinate
            of the nozzle outlet specified in the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
            Default is 0, in which case the origin of the motor's coordinate system
            is placed at the motor's nozzle outlet.
        throat_radius : int, float, optional
            Motor's nozzle throat radius in meters. Its value has very low
            impact in trajectory simulation, only useful to analyze
            dynamic instabilities, therefore it is optional.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolation_method : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
        coordinate_system_orientation : string, optional
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as at the
            nozzle area, and must be kept the same for all other positions specified.
            Options are "nozzle_to_combustion_chamber" and "combustion_chamber_to_nozzle".
            Default is "nozzle_to_combustion_chamber".

        Returns
        -------
        None
        """
        super().__init__(
            thrust_source,
            burn_out,
            nozzle_radius,
            nozzle_position,
            throat_radius,
            reshape_thrust_curve,
            interpolation_method,
        )

        # Define motor attributes
        # Grain and nozzle parameters
        self.grain_number = grain_number
        self.grain_separation = grain_separation
        self.grain_density = grain_density
        self.grain_outer_radius = grain_outer_radius
        self.grain_initial_inner_radius = grain_initial_inner_radius
        self.grain_initial_height = grain_initial_height
        self.oxidizer_tank_radius = oxidizer_tank_radius
        self.oxidizer_tank_height = oxidizer_tank_height
        self.oxidizer_initial_pressure = oxidizer_initial_pressure
        self.oxidizer_density = oxidizer_density
        self.oxidizer_molar_mass = oxidizer_molar_mass
        self.oxidizer_initial_volume = oxidizer_initial_volume
        self.distance_grain_to_tank = distance_grain_to_tank
        self.injector_area = injector_area

        # Other quantities that will be computed
        self.zCM = None
        self.oxidizer_initial_mass = None
        self.grain_inner_radius = None
        self.grain_height = None
        self.burn_area = None
        self.burn_rate = None
        self.Kn = None

        # Compute uncalculated quantities
        # Grains initial geometrical parameters
        self.grain_initial_volume = (
            self.grain_initial_height
            * np.pi
            * (self.grain_outer_radius**2 - self.grain_initial_inner_radius**2)
        )
        self.grain_initial_mass = self.grain_density * self.grain_initial_volume
        self.propellant_initial_mass = (
            self.grain_number * self.grain_initial_mass
            + self.oxidizer_initial_volume * self.oxidizer_density
        )
        # Dynamic quantities
        self.evaluate_mass_dot()
        self.evaluate_mass()
        self.evaluate_geometry()
        self.evaluate_inertia()
        self.evaluate_center_of_mass()

    @property
    def exhaust_velocity(self):
        """Calculates and returns exhaust velocity by assuming it
        as a constant. The formula used is total impulse/propellant
        initial mass. The value is also stored in
        self.exhaust_velocity.

        Parameters
        ----------
        None

        Returns
        -------
        self.exhaust_velocity : float
            Constant gas exhaust velocity of the motor.
        """
        return self.total_impulse / self.propellant_initial_mass

    def evaluate_mass_dot(self):
        """Calculates and returns the time derivative of propellant
        mass by assuming constant exhaust velocity. The formula used
        is the opposite of thrust divided by exhaust velocity. The
        result is a function of time, object of the Function class,
        which is stored in self.mass_dot.

        Parameters
        ----------
        None

        Returns
        -------
        self.mass_dot : Function
            Time derivative of total propellant mass as a function
            of time.
        """
        # Create mass dot Function
        self.mass_dot = self.thrust / (-self.exhaust_velocity)
        self.mass_dot.set_outputs("Mass Dot (kg/s)")
        self.mass_dot.set_extrapolation("zero")

        # Return Function
        return self.mass_dot

    def evaluate_center_of_mass(self):
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

        self.zCM = 0

        return self.zCM

    def evaluate_geometry(self):
        """Calculates grain inner radius and grain height as a
        function of time by assuming that every propellant mass
        burnt is exhausted. In order to do that, a system of
        differential equations is solved using scipy.integrate.
        odeint. Furthermore, the function calculates burn area,
        burn rate and Kn as a function of time using the previous
        results. All functions are stored as objects of the class
        Function in self.grain_inner_radius, self.grain_height, self.
        burn_area, self.burn_rate and self.Kn.

        Parameters
        ----------
        None

        Returns
        -------
        geometry : list of Functions
            First element is the Function representing the inner
            radius of a grain as a function of time. Second
            argument is the Function representing the height of a
            grain as a function of time.
        """
        # Define initial conditions for integration
        y0 = [self.grain_initial_inner_radius, self.grain_initial_height]

        # Define time mesh
        t = self.mass_dot.source[:, 0]

        density = self.grain_density
        rO = self.grain_outer_radius

        # Define system of differential equations
        def geometry_dot(y, t):
            grain_mass_dot = self.mass_dot(t) / self.grain_number
            rI, h = y
            rIDot = (
                -0.5 * grain_mass_dot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            )
            hDot = (
                1.0 * grain_mass_dot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            )
            return [rIDot, hDot]

        # Solve the system of differential equations
        sol = integrate.odeint(geometry_dot, y0, t)

        # Write down functions for innerRadius and height
        self.grain_inner_radius = Function(
            np.concatenate(([t], [sol[:, 0]])).transpose().tolist(),
            "Time (s)",
            "Grain Inner Radius (m)",
            self.interpolate,
            "constant",
        )
        self.grain_height = Function(
            np.concatenate(([t], [sol[:, 1]])).transpose().tolist(),
            "Time (s)",
            "Grain Height (m)",
            self.interpolate,
            "constant",
        )

        # Create functions describing burn rate, Kn and burn area
        self.evaluate_burn_area()
        self.evaluate_kn()
        self.evaluate_burn_rate()

        return [self.grain_inner_radius, self.grain_height]

    def evaluate_burn_area(self):
        """Calculates the burn_area of the grain for
        each time. Assuming that the grains are cylindrical
        BATES grains.

        Parameters
        ----------
        None

        Returns
        -------
        burn_area : Function
        Function representing the burn area progression with the time.
        """
        self.burn_area = (
            2
            * np.pi
            * (
                self.grain_outer_radius**2
                - self.grain_inner_radius**2
                + self.grain_inner_radius * self.grain_height
            )
            * self.grain_number
        )
        self.burn_area.set_outputs("Burn Area (m2)")
        return self.burn_area

    def evaluate_burn_rate(self):
        """Calculates the burn_rate with respect to time.
        This evaluation assumes that it was already
        calculated the mass_dot, burn_area timeseries.

        Parameters
        ----------
        None

        Returns
        -------
        burn_rate : Function
        Rate of progression of the inner radius during the combustion.
        """
        self.burn_rate = (-1) * self.mass_dot / (self.burn_area * self.grain_density)
        self.burn_rate.set_outputs("Burn Rate (m/s)")
        return self.burn_rate

    def evaluate_kn(self):
        KnSource = (
            np.concatenate(
                (
                    [self.grain_inner_radius.source[:, 1]],
                    [self.burn_area.source[:, 1] / self.throat_area],
                )
            ).transpose()
        ).tolist()
        self.Kn = Function(
            KnSource,
            "Grain Inner Radius (m)",
            "Kn (m2/m2)",
            self.interpolate,
            "constant",
        )
        return self.Kn

    def evaluate_inertia(self):
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

        # Inertia I
        # Calculate inertia I for each grain
        grain_mass = self.mass / self.grain_number
        grain_mass_dot = self.mass_dot / self.grain_number
        grain_number = self.grain_number
        grain_inertia_i = grain_mass * (
            (1 / 4) * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
            + (1 / 12) * self.grain_height**2
        )

        # Calculate each grain's distance d to propellant center of mass
        initial_value = (grain_number - 1) / 2
        d = np.linspace(-initial_value, initial_value, grain_number)
        d = d * (self.grain_initial_height + self.grain_separation)

        # Calculate inertia for all grains
        self.inertia_i = grain_number * grain_inertia_i + grain_mass * np.sum(d**2)
        self.inertia_i.set_outputs("Propellant Inertia I (kg*m2)")

        # Inertia I Dot
        # Calculate each grain's inertia I dot
        grain_inertia_i_dot = (
            grain_mass_dot
            * (
                (1 / 4) * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
                + (1 / 12) * self.grain_height**2
            )
            + grain_mass
            * ((1 / 2) * self.grain_inner_radius - (1 / 3) * self.grain_height)
            * self.burn_rate
        )

        # Calculate inertia I dot for all grains
        self.inertia_i_dot = (
            grain_number * grain_inertia_i_dot + grain_mass_dot * np.sum(d**2)
        )
        self.inertia_i_dot.set_outputs("Propellant Inertia I Dot (kg*m2/s)")

        # Inertia Z
        self.inertia_z = (
            (1 / 2.0)
            * self.mass
            * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
        )
        self.inertia_z.set_outputs("Propellant Inertia Z (kg*m2)")

        # Inertia Z Dot
        self.inertia_z_dot = (1 / 2.0) * self.mass_dot * (
            self.grain_outer_radius**2 + self.grain_inner_radius**2
        ) + self.mass * self.grain_inner_radius * self.burn_rate
        self.inertia_z_dot.set_outputs("Propellant Inertia Z Dot (kg*m2/s)")

        return [self.inertia_i, self.inertia_z]

    def allinfo(self):
        pass


class EmptyMotor:
    """Class that represents an empty motor with no mass and no thrust."""

    # TODO: This is a temporary solution. It should be replaced by a class that
    # inherits from the abstract Motor class. Currently cannot be done easily.
    def __init__(self):
        """Initializes an empty motor with no mass and no thrust."""
        self._csys = 1
        self.nozzle_radius = 0
        self.thrust = Function(0, "Time (s)", "Thrust (N)")
        self.mass = Function(0, "Time (s)", "Mass (kg)")
        self.mass_dot = Function(0, "Time (s)", "Mass Depletion Rate (kg/s)")
        self.burn_out_time = 1
        self.nozzle_position = 0
        self.center_of_mass = Function(0, "Time (s)", "Mass (kg)")
        self.inertia_z = Function(0, "Time (s)", "Moment of Inertia Z (kg m²)")
        self.inertia_i = Function(0, "Time (s)", "Moment of Inertia I (kg m²)")
        self.inertia_z_dot = Function(
            0, "Time (s)", "Propellant Inertia Z Dot (kgm²/s)"
        )
        self.inertia_i_dot = Function(
            0, "Time (s)", "Propellant Inertia I Dot (kgm²/s)"
        )
