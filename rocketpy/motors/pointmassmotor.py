from functools import cached_property
import numpy as np
import csv
import math 
from typing import Callable # Import Callable from the typing module

from rocketpy.mathutils.function import Function, funcify_method
from .motor import Motor 

class PointMassMotor(Motor): 
    """Class representing a motor modeled as a point mass.

    Inherits from the Motor class and simplifies the model to a thrust-producing
    object without detailed structural components. The total mass of the motor
    will vary with propellant consumption, similar to a standard motor. However,
    its inertia components and the center of propellant mass are considered zero
    and fixed at the motor's reference point, respectively.
    """

    def __init__(
        self,
        thrust_source,
        dry_mass,
        propellant_initial_mass,
        burn_time = None,
        propellant_final_mass = None,
        reshape_thrust_curve = False,
        interpolation_method = "linear",
    ):
        """Initialize the PointMassMotor class.

        This motor simplifies the physical model by considering its mass to be
        concentrated at a single point, effectively setting all its inertia
        components to zero. Propellant mass variation and exhaust velocity
        are still simulated and derived from the thrust and propellant consumption
        characteristics, similar to the base Motor class.

        Parameters
        ----------
        thrust_source : int, float, callable, str, numpy.ndarray, Function
            Thrust source. Can be a constant value (int, float), a callable
            function of time, a path to a CSV file, a NumPy array, or a
            RocketPy `Function` object.
        dry_mass : float
            Total dry mass of the motor in kg.
        propellant_initial_mass : float
            Initial mass of the propellant in kg. This is a required parameter
            as the point mass motor will still simulate propellant consumption.
        burn_time : float, optional
            Total burn time of the motor in seconds. Required if `thrust_source`
            is a constant value or a callable function, and `propellant_final_mass`
            is not provided. If `thrust_source` is a CSV, array, or Function,
            the burn time is derived from it.
        propellant_final_mass : float, optional
            Final mass of the propellant in kg. Required if `thrust_source`
            is a callable function and `burn_time` is not provided. If not
            provided, it is calculated by the base Motor class.
        reshape_thrust_curve : bool, optional
            Whether to reshape the thrust curve to start at t=0 and end at
            burn_time. Defaults to False.
        interpolation_method : str, optional
            Interpolation method for the thrust curve, if applicable.
            Defaults to 'linear'.

        Raises
        ------
        ValueError
            If insufficient data is provided for mass flow rate calculation.
        TypeError
            If an invalid type is provided for `thrust_source`.
        """
        if isinstance(thrust_source, (int, float, Callable)):
            if propellant_initial_mass is None:
                raise ValueError(
                    "For constant or callable thrust, 'propellant_initial_mass' is required."
                )
            if burn_time is None and propellant_final_mass is None:
                raise ValueError(
                    "For constant or callable thrust, either 'burn_time' or "
                    "'propellant_final_mass' must be provided."
                )
        
        elif isinstance(thrust_source, (Function, np.ndarray, str)):
            if propellant_initial_mass is None:
                raise ValueError(
                    "For thrust from a Function, NumPy array, or CSV, 'propellant_initial_mass' is required."
                )
        else:
             raise TypeError(
                "Invalid 'thrust_source' type. Must be int, float, callable, str, numpy.ndarray, or Function."
            )

        self._propellant_initial_mass = propellant_initial_mass
        self.propellant_final_mass = propellant_final_mass

        super().__init__(
            thrust_source=thrust_source,
            dry_inertia=(0, 0, 0), # Inertia is zero for a point mass
            nozzle_radius=0, # Nozzle radius is irrelevant for a point mass model
            center_of_dry_mass_position=0, # Pass 0 directly to the superclass
            dry_mass=dry_mass,
            nozzle_position=0, # Nozzle is at the motor's origin
            burn_time=burn_time,
            reshape_thrust_curve=reshape_thrust_curve, 
            interpolation_method=interpolation_method,
            coordinate_system_orientation="nozzle_to_combustion_chamber", # Standard orientation
        )

    # Removed the thrust method. It will now be inherited directly from the Motor base class,
    # which already correctly handles the conversion of thrust_source to a Function
    # and exposes it as a cached property.

    # Removed the total_mass override. The base Motor class's total_mass property
    # will now correctly calculate dry_mass + propellant_mass(t), which is the desired
    # varying mass behavior for the point mass motor.

    # Removed the center_of_dry_mass_position override. It is now passed directly
    # to the super().__init__ as 0.
    @property
    def propellant_initial_mass(self):
        """Returns the initial propellant mass for a point mass motor.

        This property retrieves the value set during initialization. This implementation
        is required as 'propellant_initial_mass' is an abstract method in the parent Motor class.

        Returns
        -------
        float
            Propellant initial mass in kg.
        """
        return self._propellant_initial_mass
    
    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaust_velocity(self):
        """Exhaust velocity by assuming it as a constant. The formula used is
        total impulse/propellant initial mass.

        Returns
        -------
        self.exhaust_velocity : Function
            Gas exhaust velocity of the motor.

        Notes
        -----
        This corresponds to the actual exhaust velocity only when the nozzle
        exit pressure equals the atmospheric pressure.
        """
        return Function(
            self.total_impulse / self.propellant_initial_mass
        ).set_discrete_based_on_model(self.thrust)
    
    @cached_property
    @funcify_method("Time (s)", "Mass flow rate (kg/s)", extrapolation="zero")
    def total_mass_flow_rate(self) -> Function:
        """Time derivative of the propellant mass as a function of time.

        It calculates mass flow rate as the negative of thrust divided by exhaust velocity,
        consistent with the fundamental rocket equation.

        Returns
        -------
        Function
            Time derivative of total propellant mass a function of time.
        """

        exhaust_vel_func = self.exhaust_velocity
        return -self.thrust / exhaust_vel_func

    @cached_property
    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def center_of_propellant_mass(self):
        """Returns the position of the center of mass of the propellant.

        For a point mass motor, the propellant's center of mass is considered
        to be at the origin (0) of the motor's coordinate system.

        Returns
        -------
        Function
            A Function object representing the center of propellant mass (always 0).
        """
        return 0

    @cached_property
    @funcify_method("Time (s)", "Inertia (kg·m²)")
    def propellant_I_11(self):
        """Returns the propellant moment of inertia around the x-axis.

        For a point mass motor, this is always zero.

        Returns
        -------
        Function
            A Function object representing zero propellant inertia.
        """
        return 0

    @cached_property
    @funcify_method("Time (s)", "Inertia (kg·m²)")
    def propellant_I_12(self):
        """Returns the propellant product of inertia I_xy.

        For a point mass motor, this is always zero.

        Returns
        -------
        Function
            A Function object representing zero propellant inertia.
        """
        return 0

    @cached_property
    @funcify_method("Time (s)", "Inertia (kg·m²)")
    def propellant_I_13(self):
        """Returns the propellant product of inertia I_xz.

        For a point mass motor, this is always zero.

        Returns
        -------
        Function
            A Function object representing zero propellant inertia.
        """
        return 0

    @cached_property
    @funcify_method("Time (s)", "Inertia (kg·m²)")
    def propellant_I_22(self):
        """Returns the propellant moment of inertia around the y-axis.

        For a point mass motor, this is always zero.

        Returns
        -------
        Function
            A Function object representing zero propellant inertia.
        """
        return 0

    @cached_property
    @funcify_method("Time (s)", "Inertia (kg·m²)")
    def propellant_I_23(self):
        """Returns the propellant product of inertia I_yz.

        For a point mass motor, this is always zero.

        Returns
        -------
        Function
            A Function object representing zero propellant inertia.
        """
        return 0

    @cached_property
    @funcify_method("Time (s)", "Inertia (kg·m²)")
    def propellant_I_33(self):
        """Returns the propellant moment of inertia around the z-axis.

        For a point mass motor, this is always zero.

        Returns
        -------
        Function
            A Function object representing zero propellant inertia.
        """
        return 0
