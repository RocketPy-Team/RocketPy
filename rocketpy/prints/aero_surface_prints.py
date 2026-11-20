import numpy as np

from rocketpy.tools import snake_case_key


# TODO: this file could be separated into different, smaller files.
class _GenericSurfacePrints:
    """Base prints for a generic aerodynamic surface."""

    def __init__(self, aero_surface):
        self.aero_surface = aero_surface

    def coefficients(self):
        """Prints the surface's aerodynamic coefficients.

        Reports the six body-frame coefficients the surface holds (``cN``,
        ``cY``, ``cA``, ``cm``, ``cn``, ``cl``) at a reference condition (5°
        angle of attack and sideslip, Mach 0.3) with the variables each one
        depends on, marking the ones that are zero. A surface given in the wind
        frame also reports ``cL``, ``cQ`` and ``cD``.
        """
        surface = self.aero_surface
        independent_vars = getattr(surface, "independent_vars", None)
        if independent_vars is None:
            return
        index = {name: i for i, name in enumerate(independent_vars)}
        if "mach" not in index:
            return

        print("Aerodynamic coefficients (AoA 5°, sideslip 5°, Mach 0.3):")
        print("---------------------------------------------------------")
        args = surface._reference_arguments()  # pylint: disable=protected-access
        args[index["mach"]] = 0.3
        if "alpha" in index:
            args[index["alpha"]] = np.deg2rad(5)
        if "beta" in index:
            args[index["beta"]] = np.deg2rad(5)
        names = ["cN", "cY", "cA", "cm", "cn", "cl"]
        if getattr(surface, "force_convention", "body") == "wind":
            names += ["cL", "cQ", "cD"]
        for name in names:
            coeff = getattr(surface, name, None)
            if coeff is None:
                continue
            if getattr(coeff, "is_zero", False):
                print(f"  {name} = 0 (zero)")
                continue
            depends = getattr(coeff, "depends_on", None)
            suffix = f" [depends on {', '.join(depends)}]" if depends else ""
            print(f"  {name} = {coeff(*args):.4f}{suffix}")
        print()

    def identity(self):
        """Prints the identity of the aero surface.

        Returns
        -------
        None
        """
        print("Identification of the AeroSurface:")
        print("----------------------------------")
        print(f"Name: {self.aero_surface.name}")
        print(f"Python Class: {str(self.aero_surface.__class__)}\n")

    def geometry(self):
        """Prints the reference geometry of the generic surface."""
        print("Geometric information of the Surface:")
        print("----------------------------------")
        print(f"Reference Area: {self.aero_surface.reference_area:.3f} m^2")
        print(f"Reference length: {self.aero_surface.reference_length:.3f} m\n")

    def all(self):
        """Prints all information of the generic surface.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.coefficients()


class _LinearGenericSurfacePrints(_GenericSurfacePrints):
    """Prints for a linear generic surface; same reporting as the generic
    base."""


class _BarrowmanSurfacePrints(_GenericSurfacePrints):
    """Prints shared by the geometry-defined (Barrowman) surfaces: adds the
    center-of-pressure / lift-curve-slope report on top of the generic base."""

    def lift(self):
        """Prints the lift information of the aero surface.

        Returns
        -------
        None
        """
        print("Lift information of the AeroSurface:")
        print("-----------------------------------")
        print(
            "Center of Pressure position in local coordinates: "
            f"({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, "
            f"{self.aero_surface.cpz:.3f})"
        )
        print(
            "Lift coefficient derivative at Mach 0 and AoA 0: "
            f"{self.aero_surface.clalpha(0):.3f} 1/rad\n"
        )

    def all(self):
        """Prints all information of the aero surface.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.lift()
        self.coefficients()


class _NoseConePrints(_BarrowmanSurfacePrints):
    """Class that contains all nosecone prints."""

    def geometry(self):
        """Prints the geometric information of the nosecone.

        Returns
        -------
        None
        """
        print("Geometric information of NoseCone:")
        print("----------------------------------")
        print(f"Length: {self.aero_surface.length:.3f} m")
        print(f"Kind: {self.aero_surface.kind}")
        print(f"Base radius: {self.aero_surface.base_radius:.3f} m")
        print(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        print(f"Reference radius ratio: {self.aero_surface.radius_ratio:.3f}\n")


class _FinsPrints(_BarrowmanSurfacePrints):
    def geometry(self):
        print("Geometric information of the fin set:")
        print("-------------------------------------")
        print(f"Number of fins: {self.aero_surface.n}")
        print(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        if hasattr(self.aero_surface, "tip_chord"):
            print(f"Tip chord: {self.aero_surface.tip_chord:.3f} m")
        print(f"Root chord: {self.aero_surface.root_chord:.3f} m")
        print(f"Span: {self.aero_surface.span:.3f} m")
        print(
            f"Cant angle: {self.aero_surface.cant_angle:.3f} ° or "
            f"{self.aero_surface.cant_angle_rad:.3f} rad"
        )
        print(f"Longitudinal section area: {self.aero_surface.Af:.3f} m²")
        print(f"Aspect ratio: {self.aero_surface.AR:.3f} ")
        print(f"Gamma_c: {self.aero_surface.gamma_c:.3f} m")
        print(f"Mean aerodynamic chord: {self.aero_surface.Yma:.3f} m\n")

    def airfoil(self):
        """Prints out airfoil related information of the fin set.

        Returns
        -------
        None
        """
        if self.aero_surface.airfoil:
            print("Airfoil information:")
            print("--------------------")
            print(
                "Number of points defining the lift curve: "
                f"{len(self.aero_surface.airfoil_cl.x_array)}"
            )
            print(
                "Lift coefficient derivative at Mach 0 and AoA 0: "
                f"{self.aero_surface.clalpha(0):.5f} 1/rad\n"
            )

    def roll(self):
        """Prints out information about roll parameters
        of the fin set.

        Returns
        -------
        None
        """
        print("Roll information of the fin set:")
        print("--------------------------------")
        print(
            f"Geometric constant: {self.aero_surface.roll_geometrical_constant:.3f} m"
        )
        print(
            "Damping interference factor: "
            f"{self.aero_surface.roll_damping_interference_factor:.3f} rad"
        )
        print(
            "Forcing interference factor: "
            f"{self.aero_surface.roll_forcing_interference_factor:.3f} rad\n"
        )

    def lift(self):
        """Prints out information about lift parameters
        of the fin set.

        Returns
        -------
        None
        """
        print("Lift information of the fin set:")
        print("--------------------------------")
        print(
            "Lift interference factor: "
            f"{self.aero_surface.lift_interference_factor:.3f} m"
        )
        print(
            "Center of Pressure position in local coordinates: "
            f"({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, "
            f"{self.aero_surface.cpz:.3f})"
        )
        print(
            "Lift Coefficient derivative (single fin) at Mach 0 and AoA 0: "
            f"{self.aero_surface.clalpha_single_fin(0):.3f}"
        )

    def all(self):
        """Prints all information of the fin set.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.airfoil()
        self.roll()
        self.lift()


class _FinPrints(_BarrowmanSurfacePrints):
    def geometry(self):
        print("Geometric information of the fin set:")
        print("-------------------------------------")
        print(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        if hasattr(self.aero_surface, "tip_chord"):
            print(f"Tip chord: {self.aero_surface.tip_chord:.3f} m")
        print(f"Root chord: {self.aero_surface.root_chord:.3f} m")
        print(f"Span: {self.aero_surface.span:.3f} m")
        print(
            f"Cant angle: {self.aero_surface.cant_angle:.3f} ° or "
            f"{self.aero_surface.cant_angle_rad:.3f} rad"
        )
        print(f"Longitudinal section area: {self.aero_surface.Af:.3f} m²")
        print(f"Aspect ratio: {self.aero_surface.AR:.3f} ")
        print(f"Gamma_c: {self.aero_surface.gamma_c:.3f} m")
        print(f"Mean aerodynamic chord: {self.aero_surface.Yma:.3f} m\n")

    def airfoil(self):
        """Prints out airfoil related information of the fin set.

        Returns
        -------
        None
        """
        if self.aero_surface.airfoil:
            print("Airfoil information:")
            print("--------------------")
            print(
                "Number of points defining the lift curve: "
                f"{len(self.aero_surface.airfoil_cl.x_array)}"
            )
            print(
                "Lift coefficient derivative at Mach 0 and AoA 0: "
                f"{self.aero_surface.clalpha(0):.5f} 1/rad\n"
            )

    def roll(self):
        """Prints out information about roll parameters
        of the fin set.

        Returns
        -------
        None
        """
        print("Roll information of the fin set:")
        print("--------------------------------")
        print(
            f"Geometric constant: {self.aero_surface.roll_geometrical_constant:.3f} m"
        )
        print(
            "Damping interference factor: "
            f"{self.aero_surface.roll_damping_interference_factor:.3f} rad"
        )
        print(
            "Forcing interference factor: "
            f"{self.aero_surface.roll_forcing_interference_factor:.3f} rad\n"
        )

    def lift(self):
        """Prints out information about lift parameters
        of the fin set.

        Returns
        -------
        None
        """
        print("Lift information of the fin set:")
        print("--------------------------------")
        print(
            "Lift interference factor: "
            f"{self.aero_surface.lift_interference_factor:.3f} m"
        )
        print(
            "Center of Pressure position in local coordinates: "
            f"({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, "
            f"{self.aero_surface.cpz:.3f})"
        )
        print(
            "Lift Coefficient derivative (single fin) at Mach 0 and AoA 0: "
            f"{self.aero_surface.clalpha_single_fin(0):.3f}"
        )

    def all(self):
        """Prints all information of the fin set.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.airfoil()
        self.roll()
        self.lift()


class _TrapezoidalFinsPrints(_FinsPrints):
    """Class that contains all trapezoidal fins prints."""


class _TrapezoidalFinPrints(_FinPrints):
    """Class that contains all trapezoidal fin prints."""


class _EllipticalFinsPrints(_FinsPrints):
    """Class that contains all elliptical fins prints."""


class _EllipticalFinPrints(_FinPrints):
    """Class that contains all elliptical fin prints."""


class _FreeFormFinsPrints(_FinsPrints):
    """Class that contains all free form fins prints."""


class _FreeFormFinPrints(_FinPrints):
    """Class that contains all free form fins prints."""


class _TailPrints(_BarrowmanSurfacePrints):
    """Class that contains all tail prints."""

    def geometry(self):
        """Prints the geometric information of the tail.

        Returns
        -------
        None
        """
        print("Geometric information of the Tail:")
        print("----------------------------------")
        print(f"Top radius: {self.aero_surface.top_radius:.3f} m")
        print(f"Bottom radius: {self.aero_surface.bottom_radius:.3f} m")
        print(f"Reference radius: {2 * self.aero_surface.rocket_radius:.3f} m")
        print(f"Length: {self.aero_surface.length:.3f} m")
        print(f"Slant length: {self.aero_surface.slant_length:.3f} m")
        print(f"Surface area: {self.aero_surface.surface_area:.6f} m²\n")


class _RailButtonsPrints(_GenericSurfacePrints):
    """Class that contains all rail buttons prints."""

    def geometry(self):
        print("Geometric information of the RailButtons:")
        print("-----------------------------------------")
        print(
            "Distance from one button to the other: "
            f"{self.aero_surface.buttons_distance:.3f} m"
        )
        print(
            "Angular position of the buttons: "
            f"{self.aero_surface.angular_position:.3f} deg\n"
        )


class _AirBrakesPrints(_GenericSurfacePrints):
    """Prints of air brakes: their settings and their drag coefficient."""

    def geometry(self):
        """Print the air brakes' settings: control input, reference area,
        limits, starting level and whether the curve replaces the rocket's
        drag."""
        brakes = self.aero_surface
        print("Air brakes:")
        print("-----------")
        control = f"{snake_case_key(brakes.name)}.deployment_level"
        print(f"Control input: {control} (0 retracted, 1 fully open)")
        print(f"Reference area: {brakes.reference_area:.4g} m^2")
        print(f"Starts each flight at: {brakes.initial_deployment_level:.4g}")
        print(f"Deployment level kept between 0 and 1: {brakes.clamp}")
        drag = (
            "the drag of the whole rocket at each deployment level"
            if brakes.override_rocket_drag
            else "added to the rocket's drag"
        )
        print(f"Drag coefficient curve: {drag}")
        print(f"Active: {brakes.active_during}, starts switched on: {brakes.active}\n")

    def drag_coefficient(self, levels=(0.25, 0.5, 0.75, 1.0), machs=(0.3, 0.5, 0.7)):
        """Print the drag coefficient at a few deployment levels and Mach
        numbers, as a table.

        Parameters
        ----------
        levels : sequence of float, optional
            The deployment levels of the rows. Default 0.25, 0.5, 0.75, 1.
        machs : sequence of float, optional
            The Mach numbers of the columns. Default 0.3, 0.5, 0.7.
        """
        curve = self.aero_surface.drag_coefficient
        print("Drag coefficient by deployment level (rows) and Mach (columns):")
        print("----------------------------------------------------------------")
        print(f"{'Level':>6s}  " + "  ".join(f"M {mach:<6.2f}" for mach in machs))
        for level in levels:
            values = "  ".join(f"{curve(level, mach):8.4f}" for mach in machs)
            print(f"{level:6.2f}  {values}")
        print()

    def all(self):
        """Print the identity, the settings and the drag coefficient table."""
        self.identity()
        self.geometry()
        self.drag_coefficient()


class _ControlFinsPrints(_GenericSurfacePrints):
    """Prints of a set of control fins: which fin is which, then the generic
    report."""

    def fins(self):
        """Print each fin's control input, angular position and default
        deflection, so a controller knows which fin it turns."""
        surface = self.aero_surface
        print("Fins of the set:")
        print("----------------")
        limits = surface.deflection_limits
        if limits is None:
            print("Deflection limits: none")
        else:
            print(f"Deflection limits: {limits[0]:.4g} to {limits[1]:.4g} rad")
        print(
            f"Hinge line: at {100 * surface.hinge_position:.4g}% of the root chord "
            "from the leading edge"
        )
        print(f"{'Control':<10s}  {'Angular position':>16s}  {'Default':>12s}")
        for control, angle in zip(surface.controls, surface.angular_positions):
            default = surface.control_defaults[control]
            print(f"{control:<10s}  {angle:>12.2f} deg  {default:>8.4g} rad")
        print(
            "A positive deflection turns a fin the way a positive cant angle does: "
            "the same\ndeflection on every fin gives a negative roll moment. "
            "The commands of\ndeflections() take care of the signs.\n"
        )

    def effectiveness(self, mach=0.3):
        """Print what the fins do per radian of each command of
        :meth:`rocketpy.ControlFins.deflections`, at one Mach number."""
        effectiveness = self.aero_surface.effectiveness()
        print(f"Effectiveness at Mach {mach:g}, zero angle of attack:")
        print("--------------------------------------------")
        print(
            "Roll moment coefficient per rad of the roll command: "
            f"{effectiveness['roll'](mach):+.4f}"
        )
        if "pitch" not in effectiveness:
            print(
                "Pitch and yaw: add the set to a rocket to see them, they depend "
                "on where it sits.\n"
            )
            return
        print(
            "Pitch moment coefficient per rad of the pitch command: "
            f"{effectiveness['pitch'](mach):+.4f}"
        )
        print(
            "Yaw moment coefficient per rad of the yaw command: "
            f"{effectiveness['yaw'](mach):+.4f}"
        )
        print("Pitch and yaw moments are about the rocket's center of dry mass.\n")

    def all(self):
        """Prints all information of the set of control fins.

        Returns
        -------
        None
        """
        self.identity()
        self.fins()
        self.effectiveness()
        self.geometry()
        self.coefficients()
