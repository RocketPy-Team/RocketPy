from abc import ABC, abstractmethod

import numpy as np


class AeroSurface(ABC):
    """Abstract class used to define aerodynamic surfaces."""

    def __init__(self, name, reference_area, reference_length):
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.name = name

        self.cpx = 0
        self.cpy = 0
        self.cpz = 0

    @staticmethod
    def _beta(mach):
        """Defines a parameter that is often used in aerodynamic
        equations. It is commonly used in the Prandtl factor which
        corrects subsonic force coefficients for compressible flow.
        This is applied to the lift coefficient of the nose cone,
        fins and tails/transitions as in [1].

        Parameters
        ----------
        mach : int, float
            Number of mach.

        Returns
        -------
        beta : int, float
            Value that characterizes flow speed based on the mach number.

        References
        ----------
        [1] Barrowman, James S. https://arc.aiaa.org/doi/10.2514/6.1979-504
        """

        if mach < 0.8:
            return np.sqrt(1 - mach**2)
        elif mach < 1.1:
            return np.sqrt(1 - 0.8**2)
        else:
            return np.sqrt(mach**2 - 1)

    @abstractmethod
    def evaluate_center_of_pressure(self):
        """Evaluates the center of pressure of the aerodynamic surface in local
        coordinates.

        Returns
        -------
        None
        """

    @abstractmethod
    def evaluate_lift_coefficient(self):
        """Evaluates the lift coefficient curve of the aerodynamic surface.

        Returns
        -------
        None
        """

    @abstractmethod
    def evaluate_geometrical_parameters(self):
        """Evaluates the geometrical parameters of the aerodynamic surface.

        Returns
        -------
        None
        """

    @abstractmethod
    def info(self):
        """Prints and plots summarized information of the aerodynamic surface.

        Returns
        -------
        None
        """

    @abstractmethod
    def all_info(self):
        """Prints and plots all the available information of the aero surface.

        Returns
        -------
        None
        """

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        *args,
    ):  # pylint: disable=unused-argument
        """Computes the forces and moments acting on the aerodynamic surface.
        Used in each time step of the simulation. This method is valid for
        the barrowman aerodynamic models.

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
        cp : Vector
            Center of pressure coordinates in the body frame.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0
        cpz = cp[2]
        stream_vx, stream_vy, stream_vz = stream_velocity
        if stream_vx**2 + stream_vy**2 != 0:  # TODO: maybe try/except
            # Normalize component stream velocity in body frame
            stream_vzn = stream_vz / stream_speed
            if -1 * stream_vzn < 1:
                attack_angle = np.arccos(-stream_vzn)
                c_lift = self.cl.get_value_opt(attack_angle, stream_mach)
                # Component lift force magnitude
                lift = 0.5 * rho * (stream_speed**2) * self.reference_area * c_lift
                # Component lift force components
                lift_dir_norm = (stream_vx**2 + stream_vy**2) ** 0.5
                lift_xb = lift * (stream_vx / lift_dir_norm)
                lift_yb = lift * (stream_vy / lift_dir_norm)
                # Total lift force
                R1, R2, R3 = lift_xb, lift_yb, 0
                # Total moment
                M1, M2, M3 = -cpz * lift_yb, cpz * lift_xb, 0
        return R1, R2, R3, M1, M2, M3
