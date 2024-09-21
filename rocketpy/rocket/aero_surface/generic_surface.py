import copy
import csv
import math

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector


class GenericSurface:
    """Defines a generic aerodynamic surface with custom force and moment
    coefficients. The coefficients can be nonlinear functions of the angle of
    attack, sideslip angle, Mach number, Reynolds number, pitch rate, yaw rate
    and roll rate."""

    def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure=(0, 0, 0),
        name="Generic Surface",
    ):
        """Create a generic aerodynamic surface, defined by its aerodynamic
        coefficients. This surface is used to model any aerodynamic surface
        that does not fit the predefined classes.

        Important
        ---------
        All the aerodynamic coefficients can be input as callable functions of
        angle of attack, angle of sideslip, Mach number, Reynolds number,
        pitch rate, yaw rate and roll rate. For CSV files, the header must
        contain at least one of the following: "alpha", "beta", "mach",
        "reynolds", "pitch_rate", "yaw_rate" and "roll_rate".

        See Also
        --------
        :ref:`genericsurfaces`.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface. Has the unit of meters.
            Commonly defined as the rocket's diameter.
        coefficients: dict
            List of coefficients. If a coefficient is omitted, it is set to 0.
            The valid coefficients are:\n
            cL: str, callable, optional
                Lift coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cQ: str, callable, optional
                Side force coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cD: str, callable, optional
                Drag coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cm: str, callable, optional
                Pitch moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cn: str, callable, optional
                Yaw moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cl: str, callable, optional
                Roll moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
        center_of_pressure : tuple, list, optional
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str, optional
            Name of the aerodynamic surface. Default is 'GenericSurface'.
        """

        self.reference_area = reference_area
        self.reference_length = reference_length
        self.center_of_pressure = center_of_pressure
        self.cp = center_of_pressure
        self.cpx = center_of_pressure[0]
        self.cpy = center_of_pressure[1]
        self.cpz = center_of_pressure[2]
        self.name = name

        default_coefficients = self._get_default_coefficients()
        self._check_coefficients(coefficients, default_coefficients)
        coefficients = self._complete_coefficients(coefficients, default_coefficients)
        for coeff, coeff_value in coefficients.items():
            value = self._process_input(coeff_value, coeff)
            setattr(self, coeff, value)

    def _get_default_coefficients(self):
        """Returns default coefficients

        Returns
        -------
        default_coefficients: dict
            Dictionary whose keys are the coefficients names and keys
            are the default values.
        """
        default_coefficients = {
            "cL": 0,
            "cQ": 0,
            "cD": 0,
            "cm": 0,
            "cn": 0,
            "cl": 0,
        }
        return default_coefficients

    def _complete_coefficients(self, input_coefficients, default_coefficients):
        """Creates a copy of the input coefficients dict and fill it with missing
        keys with default values

        Parameters
        ----------
        input_coefficients : str, dict
            Coefficients dictionary passed by the user. If the user only specifies some
            of the coefficients, the remaining are completed with class default
            values
        default_coefficients : dict
            Default coefficients of the class

        Returns
        -------
        coefficients : dict
            Coefficients dictionary used to setup coefficient attributes
        """
        coefficients = copy.deepcopy(input_coefficients)
        for coeff, value in default_coefficients.items():
            if coeff not in coefficients.keys():
                coefficients[coeff] = value

        return coefficients

    def _check_coefficients(self, input_coefficients, default_coefficients):
        """Check if input coefficients have only valid keys

        Parameters
        ----------
        input_coefficients : str, dict
            Coefficients dictionary passed by the user. If the user only specifies some
            of the coefficients, the remaining are completed with class default
            values
        default_coefficients : dict
            Default coefficients of the class

        Raises
        ------
        ValueError
            Raises a value error if the input coefficient has an invalid key
        """
        invalid_keys = set(input_coefficients) - set(default_coefficients)
        if invalid_keys:
            raise ValueError(
                f"Invalid coefficient name(s) used in key(s): {', '.join(invalid_keys)}. "
                "Check the documentation for valid names."
            )

    def _compute_from_coefficients(
        self,
        rho,
        stream_speed,
        alpha,
        beta,
        mach,
        reynolds,
        pitch_rate,
        yaw_rate,
        roll_rate,
    ):
        """Compute the aerodynamic forces and moments from the aerodynamic
        coefficients.

        Parameters
        ----------
        rho : float
            Air density.
        stream_speed : float
            Magnitude of the airflow speed.
        alpha : float
            Angle of attack in radians.
        beta : float
            Sideslip angle in radians.
        mach : float
            Mach number.
        reynolds : float
            Reynolds number.
        pitch_rate : float
            Pitch rate in radians per second.
        yaw_rate : float
            Yaw rate in radians per second.
        roll_rate : float
            Roll rate in radians per second.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        # Precompute common values
        dyn_pressure_area = 0.5 * rho * stream_speed**2 * self.reference_area
        dyn_pressure_area_length = dyn_pressure_area * self.reference_length

        # Compute aerodynamic forces
        lift = dyn_pressure_area * self.cL(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )
        side = dyn_pressure_area * self.cQ(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )
        drag = dyn_pressure_area * self.cD(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        # Compute aerodynamic moments
        pitch = dyn_pressure_area_length * self.cm(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )
        yaw = dyn_pressure_area_length * self.cn(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )
        roll = dyn_pressure_area_length * self.cl(
            alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate
        )

        return lift, side, drag, pitch, yaw, roll

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        reynolds,
    ):
        """Computes the forces and moments acting on the aerodynamic surface.
        Used in each time step of the simulation.  This method is valid for
        both linear and nonlinear aerodynamic coefficients.

        Parameters
        ----------
        stream_velocity : tuple of float
            The velocity of the airflow relative to the surface.
        stream_speed : float
            The magnitude of the airflow speed.
        stream_mach : float
            The Mach number of the airflow.
        rho : float
            Air density.
        cp : Vector
            Center of pressure coordinates in the body frame.
        omega: tuple[float, float, float]
            Tuple containing angular velocities around the x, y, z axes.
        reynolds : float
            Reynolds number.
        omega: tuple of float
            Tuple containing angular velocities around the x, y, z axes.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        # Stream velocity in standard aerodynamic frame
        stream_velocity = -stream_velocity

        # Angles of attack and sideslip
        alpha = np.arctan2(stream_velocity[1], stream_velocity[2])
        beta = np.arctan2(-stream_velocity[0], stream_velocity[2])

        # Compute aerodynamic forces and moments
        lift, side, drag, pitch, yaw, roll = self._compute_from_coefficients(
            rho,
            stream_speed,
            alpha,
            beta,
            stream_mach,
            reynolds,
            omega[0],
            omega[1],
            omega[2],
        )

        # Conversion from aerodynamic frame to body frame
        rotation_matrix = Matrix(
            [
                [1, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha)],
                [0, math.sin(alpha), math.cos(alpha)],
            ]
        ) @ Matrix(
            [
                [math.cos(beta), 0, -math.sin(beta)],
                [0, 1, 0],
                [math.sin(beta), 0, math.cos(beta)],
            ]
        )
        R1, R2, R3 = rotation_matrix @ Vector([side, -lift, -drag])

        # Dislocation of the aerodynamic application point to CDM
        M1, M2, M3 = Vector([pitch, yaw, roll]) + (cp ^ Vector([R1, R2, R3]))

        return R1, R2, R3, M1, M2, M3

    def _process_input(self, input_data, coeff_name):
        """Process the input data, either as a CSV file or a callable function.

        Parameters
        ----------
        input_data : str or callable
            Input data to be processed, either a path to a CSV or a callable.
        coeff_name : str
            Name of the coefficient being processed for error reporting.

        Returns
        -------
        Function
            Function object with 7 input arguments (alpha, beta, mach, reynolds,
            pitch_rate, yaw_rate, roll_rate).
        """
        if isinstance(input_data, str):
            # Input is assumed to be a file path to a CSV
            return self.__load_csv(input_data, coeff_name)
        elif isinstance(input_data, Function):
            if input_data.__dom_dim__ != 7:
                raise ValueError(
                    f"{coeff_name} function must have 7 input arguments"
                    " (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)."
                )
            return input_data
        elif callable(input_data):
            # Check if callable has 7 inputs (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
            if input_data.__code__.co_argcount != 7:
                raise ValueError(
                    f"{coeff_name} function must have 7 input arguments"
                    " (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)."
                )
            return input_data
        elif input_data == 0:
            return Function(
                lambda alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate: 0,
                [
                    'alpha',
                    'beta',
                    'mach',
                    'reynolds',
                    'pitch_rate',
                    'yaw_rate',
                    'roll_rate',
                ],
                [coeff_name],
                interpolation='linear',
                extrapolation='natural',
            )
        else:
            raise TypeError(
                f"Invalid input for {coeff_name}: must be a CSV file path"
                " or a callable."
            )

    def __load_csv(self, file_path, coeff_name):
        """Load a CSV file and create a Function object with the correct number
        of arguments. The CSV file must have a header that specifies the
        independent variables that are used.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        coeff_name : str
            Name of the coefficient being processed.

        Returns
        -------
        Function
            Function object with 7 input arguments (alpha, beta, mach, reynolds,
            pitch_rate, yaw_rate, roll_rate).
        """
        try:
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                header = next(reader)
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading {coeff_name} CSV file: {e}") from e

        if not header:
            raise ValueError(f"Invalid or empty CSV file for {coeff_name}.")

        # TODO make header strings flexible (e.g. 'alpha', 'Alpha', 'ALPHA')
        independent_vars = [
            'alpha',
            'beta',
            'mach',
            'reynolds',
            'pitch_rate',
            'yaw_rate',
            'roll_rate',
        ]
        present_columns = [col for col in independent_vars if col in header]

        # Check that the last column is not an independent variable
        if header[-1] in independent_vars:
            raise ValueError(
                f"Last column in {coeff_name} CSV must be the coefficient"
                " value, not an independent variable."
            )

        # Ensure that at least one independent variable is present
        if not present_columns:
            raise ValueError(f"No independent variables found in {coeff_name} CSV.")

        # Initialize the CSV-based function
        csv_func = Function(
            file_path,
            interpolation='linear',
            extrapolation='natural',
        )

        # Create a mask for the presence of each independent variable
        # save on self to avoid loss of scope
        _mask = [1 if col in present_columns else 0 for col in independent_vars]

        # Generate a lambda that applies only the relevant arguments to csv_func
        def wrapper(alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate):
            args = [alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate]
            # Select arguments that correspond to present variables
            selected_args = [arg for arg, m in zip(args, _mask) if m]
            return csv_func(*selected_args)

        # Create the interpolation function
        func = Function(
            wrapper,
            independent_vars,
            [coeff_name],
            interpolation='linear',
            extrapolation='natural',
        )
        return func
