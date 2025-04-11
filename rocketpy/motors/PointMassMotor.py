from functools import cached_property
import numpy as np
from ..mathutils.function import Function, funcify_method
from .motor import Motor

class PointMassMotor(Motor):
    """Class representing a motor modeled as a point mass.
    Inherits from the Motor class and simplifies the model to a thrust-producing
    object without detailed structural components."""

    def __init__(
        self,
        thrust_source,
        dry_mass,
        thrust_curve=None,
        propellant_initial_mass=None,
        propellant_final_mass=None,
        burn_time=None,
        center_of_dry_mass_position=0,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Initialize the PointMassMotor class.
        
        Parameters
        ----------
        thrust_source : int, float, callable, string, array, Function
            Thrust source similar to the Motor class.
        dry_mass : float
            Total dry mass of the motor in kg.
        thrust_curve : Function, np.array, or str (csv file), optional
            Required if thrust_source is a csv file, Function, or np.array.
        propellant_initial_mass : float, optional
            Required if thrust_source is a csv file, Function, or np.array.
        propellant_final_mass : float, optional
            Required if thrust_source is callable.
        burn_time : float or tuple of float, optional
            Required if thrust_source is callable or if a thrust value is given.
        center_of_dry_mass_position : float, optional
            Initial position of the motor, default is 0.
        interpolation_method : string, optional
            Interpolation method for thrust curve, default is 'linear'.
        """
        if isinstance(thrust_source, (Function, np.ndarray, str)):
            if thrust_curve is None or propellant_initial_mass is None:
                raise ValueError("thrust_curve and propellant_initial_mass are required for csv, Function, or np.array inputs.")
        elif callable(thrust_source):
            if any(param is None for param in [thrust_curve, propellant_initial_mass, burn_time, propellant_final_mass]):
                raise ValueError("thrust_curve, propellant_initial_mass, burn_time, and propellant_final_mass are required for callable inputs.")
        elif isinstance(thrust_source, (int, float)):
            if any(param is None for param in [thrust_curve, propellant_initial_mass, burn_time]):
                raise ValueError("thrust_curve, propellant_initial_mass, and burn_time are required when a thrust value is given.")
        
        self._propellant_initial_mass = propellant_initial_mass
        super().__init__(
            thrust_source=thrust_source,
            dry_inertia=(0, 0, 0),
            nozzle_radius=0,
        center_of_dry_mass_position=center_of_dry_mass_position,
            dry_mass=dry_mass,
            nozzle_position=0,
            burn_time=burn_time,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation=coordinate_system_orientation,
        )       
    @funcify_method("Time (s)", "Thrust (N)")
    def thrust(self):
        """Returns the thrust of the motor as a function of time."""
        return self.thrust_source

    @funcify_method("Time (s)", "Acceleration (m/s^2)")
    def total_mass(self):
        """Returns the constant total mass of the point mass motor."""
        return self.dry_mass

    @funcify_method("Time (s)", "Acceleration (m/s^2)")
    def acceleration(self):
        """Computes the acceleration of the motor as thrust divided by mass."""
        return self.thrust() / self.total_mass

    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def center_of_propellant_mass(self):
        return 0

    @funcify_method("Time (s)", "Exhaust Velocity (m/s)")
    def exhaust_velocity(self):
        return 2000  # m/s, estimated value

    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_initial_mass(self):
        return self._propellant_initial_mass

    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_I_11(self): 
        return 0
    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_I_12(self): 
        return 0
    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_I_13(self): 
        return 0
    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_I_22(self): 
        return 0
    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_I_23(self): 
        return 0
    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_I_33(self): 
        return 0
