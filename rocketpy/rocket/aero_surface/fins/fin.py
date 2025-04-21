import math

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket.aero_surface.fins.base_fin import _BaseFin

from ..aero_surface import AeroSurface


class Fin(_BaseFin):
    """Abstract class that holds common methods for the individual fin classes.
    Cannot be instantiated.

    Note
    ----
    Local coordinate system:
        - Origin located at the top of the root chord.
        - Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).
        - Y axis perpendicular to the Z axis, in the span direction, positive upwards.
        - X axis completes the right-handed coordinate system.

    Attributes
    ----------
    Fin.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization,
        in meters.
    Fin.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    Fin.cant_angle : float
        Fin cant angle with respect to the rocket centerline, in degrees.
    Fin.changing_attribute_dict : dict
        Dictionary that stores the name and the values of the attributes that
        may be changed during a simulation. Useful for control systems.
    Fin.cant_angle_rad : float
        Fin cant angle with respect to the rocket centerline, in radians.
    Fin.root_chord : float
        Fin root chord in meters.
    Fin.tip_chord : float
        Fin tip chord in meters.
    Fin.span : float
        Fin span in meters.
    Fin.name : string
        Name of fin set.
    Fin.sweep_length : float
        Fin sweep length in meters. By sweep length, understand the axial
        distance between the fin root leading edge and the fin tip leading edge
        measured parallel to the rocket centerline.
    Fin.sweep_angle : float
        Fin sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    Fin.d : float
        Reference diameter of the rocket. Has units of length and is given
        in meters.
    Fin.ref_area : float
        Reference area of the rocket.
    Fin.Af : float
        Area of the longitudinal section of each fin in the set.
    Fin.AR : float
        Aspect ratio of each fin in the set.
    Fin.gamma_c : float
        Fin mid-chord sweep angle.
    Fin.Yma : float
        Span wise position of the mean aerodynamic chord.
    Fin.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    Fin.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    Fin.lift_interference_factor : float
        Factor of Fin-Body interference in the lift coefficient.
    Fin.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of
        pressure. Has units of length and is given in meters.
    Fin.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and
        is given in meters.
    Fin.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and
        is given in meters.
    Fin.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and
        is given in meters.
    Fin.cl : Function
        Function which defines the lift coefficient as a function of the angle
        of attack and the Mach number. Takes as input the angle of attack in
        radians and the Mach number. Returns the lift coefficient.
    Fin.clalpha : float
        Lift coefficient slope. Has units of 1/rad.
    Fin.roll_parameters : list
        List containing the roll moment lift coefficient, the roll moment
        damping coefficient and the cant angle in radians.
    """

    def __init__(
        self,
        angular_position,
        root_chord,
        span,
        rocket_radius,
        cant_angle=0,
        airfoil=None,
        name="Fin",
    ):
        """Initialize Fin class.

        Parameters
        ----------
        angular_position : int, float
            Angular position of the fin in degrees.
        root_chord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        rocket_radius : int, float
            Reference rocket radius used for lift coefficient normalization.
        cant_angle : int, float, optional
            Fin cant angle with respect to the rocket centerline. Must
            be given in degrees.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files can contain a single line
            header and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".
        name : str
            Name of fin.
        """
        super().__init__(
            name=name,
            rocket_radius=rocket_radius,
            root_chord=root_chord,
            span=span,
            airfoil=airfoil,
            cant_angle=cant_angle,
        )

        # Store values
        self._angular_position = angular_position
        self._angular_position_rad = math.radians(angular_position)

    @property
    def cant_angle(self):
        return self._cant_angle

    @cant_angle.setter
    def cant_angle(self, value):
        self._cant_angle = value
        self.cant_angle_rad = math.radians(value)

    @property
    def cant_angle_rad(self):
        return self._cant_angle_rad

    @cant_angle_rad.setter
    def cant_angle_rad(self, value):
        self._cant_angle_rad = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()
        self.evaluate_rotation_matrix()

    @property
    def angular_position(self):
        return self._angular_position

    @angular_position.setter
    def angular_position(self, value):
        self._angular_position = value
        self.angular_position_rad = math.radians(value)

    @property
    def angular_position_rad(self):
        return self._angular_position_rad

    @angular_position_rad.setter
    def angular_position_rad(self, value):
        self._angular_position_rad = value
        self.evaluate_rotation_matrix()

    def evaluate_lift_coefficient(self):
        """Calculates and returns the fin set's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves the lift coefficient derivative
        for a single fin and the lift coefficient derivative for
        a number of n fins corrected for Fin-Body interference.

        Returns
        -------
        None
        """
        self.evaluate_single_fin_lift_coefficient()

        self.clalpha = self.clalpha_single_fin * self.lift_interference_factor

        # Cl = clalpha * alpha
        self.cl = Function(
            lambda alpha, mach: alpha * self.clalpha(mach),
            ["Alpha (rad)", "Mach"],
            "Lift coefficient",
        )

        return self.cl

    def evaluate_roll_parameters(self):
        """Calculates and returns the fin set's roll coefficients.
        The roll coefficients are saved in a list.

        Returns
        -------
        self.roll_parameters : list
            List containing the roll moment lift coefficient, the
            roll moment damping coefficient and the cant angle in
            radians
        """
        clf_delta = 0  # Not used in this class
        cld_omega = -(
            2
            * self.roll_damping_interference_factor
            * self.clalpha_single_fin
            * np.cos(self.cant_angle_rad)
            * self.roll_geometrical_constant
            / (self.ref_area * self.d**2)
        )  # Function of mach number
        cld_omega.set_inputs("Mach")
        cld_omega.set_outputs("Roll moment damping coefficient derivative")
        self.roll_parameters = [clf_delta, cld_omega, self.cant_angle_rad]
        return self.roll_parameters

    def evaluate_rotation_matrix(self):
        """Calculates and returns the rotation matrix from the rocket body frame
        to the fin frame.

        Note
        ----
        Local coordinate system:
        - Origin located at the leading edge of the root chord.
        - Z axis along the longitudinal axis of the fin, positive downwards
            (leading edge -> trailing edge).
        - Y axis perpendicular to the Z axis, in the span direction,
            positive upwards (root chord -> tip chord).
        - X axis completes the right-handed coordinate system.


        Returns
        -------
        None

        References
        ----------
        [1] TODO link to docs
        """
        phi = self.angular_position_rad
        delta = self.cant_angle_rad

        # Rotation about body Z by angular position
        R_phi = Matrix(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        )

        # Cant rotation about body Y
        R_delta = Matrix(
            [
                [np.cos(delta), 0, -np.sin(delta)],
                [0, 1, 0],
                [np.sin(delta), 0, np.cos(delta)],
            ]
        )

        # 180 flip about Y to align fin leading/trailing edge
        R_pi = Matrix(
            [
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]
        )

        # Uncanted body to fin, then apply cant
        R_uncanted = R_phi @ R_pi
        R_body_to_fin = R_delta @ R_uncanted

        # Store for downstream transforms
        self._rotation_fin_to_body_uncanted = R_uncanted.transpose
        self._rotation_body_to_fin = R_body_to_fin
        self._rotation_fin_to_body = R_body_to_fin.transpose

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        *args,
    ):  # pylint: disable=arguments-differ
        """Computes the forces and moments acting on the aerodynamic surface.

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

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

        # stream velocity in fin frame
        stream_velocity_f = self._rotation_body_to_fin @ stream_velocity

        attack_angle = np.arctan2(stream_velocity_f[0], stream_velocity_f[2])
        # Force in the X direction of the fin
        X = (
            0.5
            * rho
            * stream_speed**2
            * self.reference_area
            * self.cl.get_value_opt(attack_angle, stream_mach)
        )
        # Force in body frame
        R1, R2, R3 = self._rotation_fin_to_body @ Vector([X, 0, 0])
        # Moments
        M1, M2, M3 = cp ^ Vector([R1, R2, R3])
        # Apply roll interference factor, disregarding lift interference factor
        M3 *= self.roll_forcing_interference_factor / self.lift_interference_factor

        # Roll damping
        _, cld_omega, _ = self.roll_parameters
        M3_damping = (
            (1 / 2 * rho * stream_speed)
            * self.reference_area
            * (self.reference_length) ** 2
            * cld_omega.get_value_opt(stream_mach)
            * omega[2]  # omega3
            / 2
        )
        M3 += M3_damping
        return R1, R2, R3, M1, M2, M3

    def to_dict(self, include_outputs=False):
        data = {
            "angular_position": self.angular_position,
            "root_chord": self.root_chord,
            "span": self.span,
            "rocket_radius": self.rocket_radius,
            "cant_angle": self.cant_angle,
            "airfoil": self.airfoil,
            "name": self.name,
        }

        if include_outputs:
            data.update(
                {
                    "cp": self.cp,
                    "cl": self.cl,
                    "roll_parameters": self.roll_parameters,
                    "d": self.d,
                    "ref_area": self.ref_area,
                }
            )
        return data

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information, including
        the center line, the quarter line and the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        """
        self.plots.draw(filename=filename)
