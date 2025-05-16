from rocketpy.mathutils import Function, funcify_method
from rocketpy.motors.motor import Motor


class EmptyMotor(Motor):
    """Class that represents an empty motor with no mass and no thrust."""

    def __init__(self):
        """Initializes an empty motor with no mass and no thrust."""

        super().__init__(
            thrust_source=0,
            dry_inertia=(0, 0, 0),
            nozzle_radius=0,
            center_of_dry_mass_position=0,
            dry_mass=0,
            nozzle_position=0,
            burn_time=1,
            reshape_thrust_curve=False,
            interpolation_method="linear",
            coordinate_system_orientation="nozzle_to_combustion_chamber",
            reference_pressure=0,
        )

        # Mass properties
        self.propellant_mass = Function(0, "Time (s)", "Propellant Mass (kg)")
        self.total_mass = Function(0, "Time (s)", "Total Mass (kg)")
        self.total_mass_flow_rate = Function(
            0, "Time (s)", "Mass Depletion Rate (kg/s)"
        )
        self.center_of_mass = Function(0, "Time (s)", "Center of Mass (kg)")

        # Inertia properties
        self.I_11 = Function(0)
        self.I_22 = Function(0)
        self.I_33 = Function(0)
        self.I_12 = Function(0)
        self.I_13 = Function(0)
        self.I_23 = Function(0)

    @funcify_method("Time (s)", "Center of Propellant Mass (kg)", "linear", "zero")
    def center_of_propellant_mass(self):
        return 0

    @funcify_method("Time (s)", "Exhaust Velocity (m/s)", "linear", "zero")
    def exhaust_velocity(self):
        return 0

    @property
    def propellant_initial_mass(self):
        return 0

    @funcify_method("Time (s)", "Propellant I_11 (kg m²)", "linear", "zero")
    def propellant_I_11(self):
        return 0

    @funcify_method("Time (s)", "Propellant I_12 (kg m²)", "linear", "zero")
    def propellant_I_12(self):
        return 0

    @funcify_method("Time (s)", "Propellant I_13 (kg m²)", "linear", "zero")
    def propellant_I_13(self):
        return 0

    @funcify_method("Time (s)", "Propellant I_22 (kg m²)", "linear", "zero")
    def propellant_I_22(self):
        return 0

    @funcify_method("Time (s)", "Propellant I_23 (kg m²)", "linear", "zero")
    def propellant_I_23(self):
        return 0

    @funcify_method("Time (s)", "Propellant I_33 (kg m²)", "linear", "zero")
    def propellant_I_33(self):
        return 0

    @property
    def structural_mass_ratio(self):
        return 0
