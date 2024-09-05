import csv
from .aero_surface import AeroSurface
from rocketpy.mathutils import Function
import numpy as np


class GenericSurface(AeroSurface):
    """Defines a generic aerodynamic surface with custom force and moment coefficients.
    The coefficients can be nonlinear functions of the angle of attack, sideslip angle,
    Mach number, and altitude."""

    def __init__(
        self,
        reference_area,
        reference_length,
        cL=0,
        cD=0,
        cQ=0,
        cm=0,
        cm_d=0,
        cn=0,
        cn_d=0,
        cl=0,
        cl_d=0,
        center_of_pressure=(0, 0, 0),
        name="Generic Surface",
    ):
        """Initializes the generic aerodynamic surface object. The coefficients
        can be defined as a CSV file or as a callable function. The function
        must have 4 input arguments in the form (alpha, beta, mach, height). The
        CSV file must have a header with the following columns: 'alpha', 'beta',
        'mach', 'height', and the last column must be the coefficient value,
        which can have any name in the header. Not all columns are required, but
        at least one independent variable and the coefficient value are
        required.

        See Also
        --------
        LINK TO DOCS


        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface. Has the unit of meters.
            Commonly defined as the rocket's diameter.
        cL : str, callable
            Lift coefficient. Can be a path to a CSV file or a callable
            function.
        cD : str, callable
            Drag coefficient. Can be a path to a CSV file or a callable
            function.
        cQ : str, callable
            Side force coefficient. Can be a path to a CSV file or a callable
            function.
        cm : str, callable
            Pitch moment coefficient. Can be a path to a CSV file or a callable
            function.
        cm_d : str, callable
            Pitch damping moment coefficient. Can be a path to a CSV file or a
            callable function. Also known as the pitch moment derivative with
            respect to the pitch rate.
        cn : str, callable
            Yaw moment coefficient. Can be a path to a CSV file or a callable
            function.
        cn_d : str, callable
            Yaw damping moment coefficient. Can be a path to a CSV file or a
            callable function. Also known as the yaw moment derivative with
            respect to the yaw rate.
        cl : str, callable
            Roll moment coefficient. Can be a path to a CSV file or a callable
            function.
        cl_d : str, callable
            Roll damping moment coefficient. Can be a path to a CSV file or a
            callable function. Also known as the roll moment derivative with
            respect to the roll rate.
        center_of_pressure : tuple, list
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str
            Name of the aerodynamic surface. Default is 'GenericSurface'.
        """
        super().__init__(name, reference_area, reference_length)
        self.center_of_pressure = center_of_pressure
        self.cp = center_of_pressure
        self.cpx = center_of_pressure[0]
        self.cpy = center_of_pressure[1]
        self.cpz = center_of_pressure[2]

        self.cL = self.__process_input(cL, "cL")
        self.cD = self.__process_input(cD, "cD")
        self.cQ = self.__process_input(cQ, "cQ")
        self.cm = self.__process_input(cm, "cm")
        self.cm_d = self.__process_input(cm_d, "cm_d")
        self.cl = self.__process_input(cl, "cl")
        self.cl_d = self.__process_input(cl_d, "cl_d")
        self.cn = self.__process_input(cn, "cn")
        self.cn_d = self.__process_input(cn_d, "cn_d")

    def __process_input(self, input_data, coeff_name):
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
            Function object with 4 input arguments (alpha, beta, mach, height).
        """
        if isinstance(input_data, str):
            # Input is assumed to be a file path to a CSV
            return self.__load_csv(input_data, coeff_name)
        elif isinstance(input_data, Function):
            if input_data.__dom_dim__ != 4:
                raise ValueError(
                    f"{coeff_name} function must have 4 input arguments"
                    " (alpha, beta, mach, height)."
                )
            return input_data
        elif callable(input_data):
            # Check if callable has 4 inputs (alpha, beta, mach, height)
            if input_data.__code__.co_argcount != 4:
                raise ValueError(
                    f"{coeff_name} function must have 4 input arguments"
                    " (alpha, beta, mach, height)."
                )
            return input_data
        else:
            raise TypeError(
                f"Invalid input for {coeff_name}: must be a CSV file path"
                " or a callable."
            )

    def __load_csv(self, file_path, coeff_name):
        """Load a CSV file and create a Function object with the correct number
        of arguments. The CSV file must have a header with the following columns:
        'alpha', 'beta', 'mach', 'height', and the last column must be the
        coefficient value, which can have any name in the header. Not all columns
        are required, but at least one independent variable and the coefficient
        value are required.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        coeff_name : str
            Name of the coefficient being processed.

        Returns
        -------
        Function
            Function object with 4 input arguments (alpha, beta, mach, height).
        """
        try:
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                header = next(reader)
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading {coeff_name} CSV file: {e}")

        if not header:
            raise ValueError(f"Invalid or empty CSV file for {coeff_name}.")

        # TODO make header strings flexible (e.g. 'alpha', 'Alpha', 'ALPHA')
        independent_vars = ['alpha', 'beta', 'mach', 'height']
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
        csv_func = Function(file_path, interpolation='linear', extrapolation='natural')

        # Create a mask for the presence of each independent variable
        mask = [1 if col in present_columns else 0 for col in independent_vars]

        # Generate a lambda that applies only the relevant arguments to csv_func
        def wrapper(alpha, beta, mach, height):
            args = [alpha, beta, mach, height]
            # Select arguments that correspond to present variables
            selected_args = [arg for arg, m in zip(args, mask) if m]
            return csv_func(*selected_args)

        # Create the interpolation function
        func = Function(
            wrapper, independent_vars, [coeff_name], extrapolation='natural'
        )

        return func

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        _,
        height,
        omega1,
        omega2,
        omega3,
        *args,
        **kwargs,
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
        height : float
            Altitude of the surface.
        omega1, omega2, omega3 : float
            Angular velocities around the x, y, z axes.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments (pitch, yaw, roll).
        """
        # Calculate aerodynamic angles
        alpha = np.arctan2(-stream_velocity[0], stream_velocity[2])
        beta = np.arctan2(-stream_velocity[1], stream_velocity[2])

        # Compute aerodynamic forces
        lift = self.__compute_force(
            self.cL, alpha, beta, stream_mach, height, rho, stream_speed
        )
        side_force = self.__compute_force(
            self.cQ, alpha, beta, stream_mach, height, rho, stream_speed
        )
        drag = self.__compute_force(
            self.cD, alpha, beta, stream_mach, height, rho, stream_speed
        )

        # Compute aerodynamic moments
        pitch = self.__compute_moment(
            self.cm,
            self.cm_d,
            alpha,
            beta,
            stream_mach,
            height,
            rho,
            stream_speed,
            omega1,
        )
        yaw = self.__compute_moment(
            self.cn,
            self.cn_d,
            alpha,
            beta,
            stream_mach,
            height,
            rho,
            stream_speed,
            omega2,
        )
        roll = self.__compute_moment(
            self.cl,
            self.cl_d,
            alpha,
            beta,
            stream_mach,
            height,
            rho,
            stream_speed,
            omega3,
        )

        return lift, side_force, drag, pitch, yaw, roll

    def __compute_force(
        self, coeff_function, alpha, beta, mach, height, rho, stream_speed
    ):
        """Helper function to compute aerodynamic forces."""
        return (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * coeff_function(alpha, beta, mach, height)
        )

    def __compute_moment(
        self,
        forcing_coefficient,
        damping_coefficient,
        alpha,
        beta,
        mach,
        height,
        rho,
        stream_speed,
        angular_velocity,
    ):
        """
        Helper function to compute aerodynamic moments.
        """
        forcing_moment = (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * self.reference_length
            * forcing_coefficient(alpha, beta, mach, height)
        )
        damping_moment = (
            0.5
            * rho
            * stream_speed
            * self.reference_area
            * self.reference_length**2
            * damping_coefficient(alpha, beta, mach, height)
            * angular_velocity
            / 2
        )
        return forcing_moment - damping_moment
