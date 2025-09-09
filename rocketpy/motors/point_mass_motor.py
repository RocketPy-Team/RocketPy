from functools import cached_property
from typing import Callable

import numpy as np

from rocketpy.mathutils.function import Function, funcify_method

from .motor import Motor


class PointMassMotor(Motor):
    """Motor modeled as a point mass for 3-DOF simulations."""

    def __init__(
        self,
        thrust_source,
        dry_mass,
        propellant_initial_mass,
        burn_time=None,
        propellant_final_mass=None,
        reshape_thrust_curve=False,
        interpolation_method="linear",
    ):
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
            dry_inertia=(0, 0, 0),
            nozzle_radius=0,
            center_of_dry_mass_position=0,
            dry_mass=dry_mass,
            nozzle_position=0,
            burn_time=burn_time,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )

    @property
    def propellant_initial_mass(self):
        return self._propellant_initial_mass

    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaust_velocity(self):
        """Assume constant exhaust velocity: total impulse / propellant mass"""
        v_e = self.total_impulse / self.propellant_initial_mass
        return Function(v_e).set_discrete_based_on_model(self.thrust)

    @cached_property
    def total_mass_flow_rate(self) -> Function:
        """Mass flow rate: -thrust / exhaust_velocity"""
        return -self.thrust / self.exhaust_velocity

    @cached_property
    def center_of_propellant_mass(self):
        """Center of propellant mass is always zero"""
        return Function(0.0)

    # Propellant inertias: always zero, but return as Function objects
    def _zero_inertia_func(self):
        return Function(0.0)

    @cached_property
    def propellant_I_11(self):
        return self._zero_inertia_func()

    @cached_property
    def propellant_I_12(self):
        return self._zero_inertia_func()

    @cached_property
    def propellant_I_13(self):
        return self._zero_inertia_func()

    @cached_property
    def propellant_I_22(self):
        return self._zero_inertia_func()

    @cached_property
    def propellant_I_23(self):
        return self._zero_inertia_func()

    @cached_property
    def propellant_I_33(self):
        return self._zero_inertia_func()
