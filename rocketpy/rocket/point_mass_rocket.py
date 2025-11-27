"""Point mass rocket class for simplified 3-DOF trajectory simulations."""

from rocketpy.rocket.rocket import Rocket


class PointMassRocket(Rocket):
    """A simplified rocket class for trajectory simulations where the rocket
    is modeled as a point mass.

    This class omits rotational dynamics and complex inertial properties,
    focusing solely on translational (3-DOF) motion based on mass and
    aerodynamics. Appropriate for educational use, quick analyses, or when
    rotational effects are negligible.

    Parameters
    ----------
    radius : float
        Rocket's largest radius in meters.
    mass : float
        Rocket's mass without motor in kg.
    center_of_mass_without_motor : float
        Position, in meters, of the rocket's center of mass without motor
        relative to the rocket's coordinate system.
    power_off_drag : float, callable, array, string, Function
        Drag coefficient as a function of Mach number when the motor is off.
    power_on_drag : float, callable, array, string, Function
        Drag coefficient as a function of Mach number when the motor is on.

    Attributes
    ----------
    radius : float
        Rocket's largest radius in meters.
    mass : float
        Rocket's mass without motor in kg.
    center_of_mass_without_motor : float
        Position, in meters, of the rocket's center of mass without motor
        relative to the rocket's coordinate system.
    power_off_drag : Function
        Drag coefficient as a function of Mach number when the motor is off.
    power_on_drag : Function
        Drag coefficient as a function of Mach number when the motor is on.
    """

    def __init__(
        self,
        radius: float,
        mass: float,
        center_of_mass_without_motor: float,
        power_off_drag,
        power_on_drag,
    ):
        self._center_of_mass_without_motor_pointmass = center_of_mass_without_motor
        self._center_of_dry_mass_position = center_of_mass_without_motor
        self._center_of_mass = center_of_mass_without_motor
        # Dry inertias are zero for point mass
        self.dry_I_11 = 0.0
        self.dry_I_22 = 0.0
        self.dry_I_33 = 0.0
        self.dry_I_12 = 0.0
        self.dry_I_13 = 0.0
        self.dry_I_23 = 0.0

        # Call base init with safe defaults
        super().__init__(
            radius=radius,
            mass=mass,
            inertia=(0, 0, 0),
            power_off_drag=power_off_drag,
            power_on_drag=power_on_drag,
            center_of_mass_without_motor=center_of_mass_without_motor,
        )

    def evaluate_dry_inertias(self):
        """Override to ensure inertias remain zero for point mass model.

        Returns
        -------
        tuple
            All inertia components as zeros.
        """
        self.dry_I_11 = 0.0
        self.dry_I_22 = 0.0
        self.dry_I_33 = 0.0
        self.dry_I_12 = 0.0
        self.dry_I_13 = 0.0
        self.dry_I_23 = 0.0
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def evaluate_inertias(self):
        """Override to ensure inertias remain zero for point mass model.

        Returns
        -------
        tuple
            All inertia components as zeros.
        """
        self.I_11 = 0.0
        self.I_22 = 0.0
        self.I_33 = 0.0
        self.I_12 = 0.0
        self.I_13 = 0.0
        self.I_23 = 0.0
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
