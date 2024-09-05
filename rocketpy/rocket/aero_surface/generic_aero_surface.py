import csv
from .aero_surface import AeroSurface
from rocketpy.mathutils import Function
import numpy as np


class GenericSurface(AeroSurface):
    """Class that defines a generic aerodynamic surface. This class is used to
    define aerodynamic surfaces with custom force and moment coefficients. The
    coefficients can be nonlinear functions of the angle of attack, sideslip
    angle, mach number, and altitude.
    """

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
        input_data : str, callable
            Input data to be processed.
        coeff_name : str
            Name of the coefficient being processed.
        """
        if isinstance(input_data, str):
            # Assume input_data is a file path to a CSV
            return self.__load_csv(input_data, coeff_name)
        elif isinstance(input_data, Function):
            # Check if it has 4 inputs (alpha, beta, mach, height)
            if input_data.__dom_dim__ != 4:
                raise ValueError(
                    f"Function for {coeff_name} must have 4 input arguments in"
                    " the form (alpha, beta, mach, height)."
                )
        elif callable(input_data):
            # Check if it has 4 inputs (alpha, beta, mach, height)
            if input_data.__code__.co_argcount != 4:
                raise ValueError(
                    f"Function for {coeff_name} must have 4 input arguments in"
                    " the form (alpha, beta, mach, height)."
                )
            return input_data
        else:
            raise ValueError(
                f"Input for {coeff_name} must be a file path or a callable."
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
            Function object that interpolates the data from the CSV file.
        """
        try:
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Read only the first row (the header)
        except Exception as e:
            raise ValueError(f"Could not read CSV file for {coeff_name}: {e}")

        if not header:
            raise ValueError(f"CSV file for {coeff_name} is empty or invalid.")

        # TODO make header strings flexible (e.g. 'alpha', 'Alpha', 'ALPHA')

        # Define possible independent variables
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
        stream_velocity : tuple
            Tuple containing the stream velocity components in the body frame.
        stream_speed : int, float
            Speed of the stream in m/s.
        stream_mach : int, float
            Mach number of the stream.
        rho : int, float
            Density of the stream in kg/m^3.
        cpz : int, float
            Distance between the center of pressure and the center of dry mass
            in the z direction in meters.
        height : int, float
            Altitude of the rocket in meters.
        omega1 : int, float
            Angular velocity of the rocket in the x direction in rad/s.
        omega2 : int, float
            Angular velocity of the rocket in the y direction in rad/s.
        omega3 : int, float
            Angular velocity of the rocket in the z direction in rad/s.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        R1 : int, float
            Force component in the x direction in Newtons.
        R2 : int, float
            Force component in the y direction in Newtons.
        R3 : int, float
            Force component in the z direction in Newtons.
        M1 : int, float
            Moment component in the x direction in Newton-meters.
        M2 : int, float
            Moment component in the y direction in Newton-meters.
        M3 : int, float
            Moment component in the z direction in Newton-meters.
        """

        # calculate the aerodynamic angles
        alpha = np.arctan(-stream_velocity[0] / stream_velocity[2])  # X-Z plane
        beta = np.arctan(-stream_velocity[1] / stream_velocity[2])  # Y-Z plane

        # calculate the forces
        lift = (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * self.cL(alpha, beta, stream_mach, height)
        )

        side_force = (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * self.cQ(alpha, beta, stream_mach, height)
        )

        drag = (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * self.cD(alpha, beta, stream_mach, height)
        )

        pitch = self.__compute_moment(
            self.cm,
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
            alpha,
            beta,
            stream_mach,
            height,
            rho,
            stream_speed,
            omega3,
        )

        return lift, side_force, drag, pitch, yaw, roll

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
