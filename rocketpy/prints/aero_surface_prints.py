import logging

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod


# TODO: the rocketpy/prints/aero_surface_prints.py file could be separated into different, smaller files.
class _AeroSurfacePrints(ABC):
    def __init__(self, aero_surface):
        self.aero_surface = aero_surface

    def identity(self):
        """Prints the identity of the aero surface.

        Returns
        -------
        None
        """
        logger.info("Identification of the AeroSurface:")
        logger.info("----------------------------------")
        logger.info(f"Name: {self.aero_surface.name}")
        logger.info(f"Python Class: {str(self.aero_surface.__class__)}\n")

    @abstractmethod
    def geometry(self):
        pass

    def lift(self):
        """Prints the lift information of the aero surface.

        Returns
        -------
        None
        """
        logger.info("Lift information of the AeroSurface:")
        logger.info("-----------------------------------")
        logger.info(
            "Center of Pressure position in local coordinates: "
            f"({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, "
            f"{self.aero_surface.cpz:.3f})"
        )
        logger.info(
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


class _NoseConePrints(_AeroSurfacePrints):
    """Class that contains all nosecone prints."""

    def geometry(self):
        """Prints the geometric information of the nosecone.

        Returns
        -------
        None
        """
        logger.info("Geometric information of NoseCone:")
        logger.info("----------------------------------")
        logger.info(f"Length: {self.aero_surface.length:.3f} m")
        logger.info(f"Kind: {self.aero_surface.kind}")
        logger.info(f"Base radius: {self.aero_surface.base_radius:.3f} m")
        logger.info(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        logger.info(f"Reference radius ratio: {self.aero_surface.radius_ratio:.3f}\n")


class _FinsPrints(_AeroSurfacePrints):
    def geometry(self):
        logger.info("Geometric information of the fin set:")
        logger.info("-------------------------------------")
        logger.info(f"Number of fins: {self.aero_surface.n}")
        logger.info(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        if hasattr(self.aero_surface, "tip_chord"):
            logger.info(f"Tip chord: {self.aero_surface.tip_chord:.3f} m")
        logger.info(f"Root chord: {self.aero_surface.root_chord:.3f} m")
        logger.info(f"Span: {self.aero_surface.span:.3f} m")
        logger.info(
            f"Cant angle: {self.aero_surface.cant_angle:.3f} ° or "
            f"{self.aero_surface.cant_angle_rad:.3f} rad"
        )
        logger.info(f"Longitudinal section area: {self.aero_surface.Af:.3f} m²")
        logger.info(f"Aspect ratio: {self.aero_surface.AR:.3f} ")
        logger.info(f"Gamma_c: {self.aero_surface.gamma_c:.3f} m")
        logger.info(f"Mean aerodynamic chord: {self.aero_surface.Yma:.3f} m\n")

    def airfoil(self):
        """Prints out airfoil related information of the fin set.

        Returns
        -------
        None
        """
        if self.aero_surface.airfoil:
            logger.info("Airfoil information:")
            logger.info("--------------------")
            logger.info(
                "Number of points defining the lift curve: "
                f"{len(self.aero_surface.airfoil_cl.x_array)}"
            )
            logger.info(
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
        logger.info("Roll information of the fin set:")
        logger.info("--------------------------------")
        logger.info(
            f"Geometric constant: {self.aero_surface.roll_geometrical_constant:.3f} m"
        )
        logger.info(
            "Damping interference factor: "
            f"{self.aero_surface.roll_damping_interference_factor:.3f} rad"
        )
        logger.info(
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
        logger.info("Lift information of the fin set:")
        logger.info("--------------------------------")
        logger.info(
            "Lift interference factor: "
            f"{self.aero_surface.lift_interference_factor:.3f} m"
        )
        logger.info(
            "Center of Pressure position in local coordinates: "
            f"({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, "
            f"{self.aero_surface.cpz:.3f})"
        )
        logger.info(
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


class _FinPrints(_AeroSurfacePrints):
    def geometry(self):
        logger.info("Geometric information of the fin set:")
        logger.info("-------------------------------------")
        logger.info(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        if hasattr(self.aero_surface, "tip_chord"):
            logger.info(f"Tip chord: {self.aero_surface.tip_chord:.3f} m")
        logger.info(f"Root chord: {self.aero_surface.root_chord:.3f} m")
        logger.info(f"Span: {self.aero_surface.span:.3f} m")
        logger.info(
            f"Cant angle: {self.aero_surface.cant_angle:.3f} ° or "
            f"{self.aero_surface.cant_angle_rad:.3f} rad"
        )
        logger.info(f"Longitudinal section area: {self.aero_surface.Af:.3f} m²")
        logger.info(f"Aspect ratio: {self.aero_surface.AR:.3f} ")
        logger.info(f"Gamma_c: {self.aero_surface.gamma_c:.3f} m")
        logger.info(f"Mean aerodynamic chord: {self.aero_surface.Yma:.3f} m\n")

    def airfoil(self):
        """Prints out airfoil related information of the fin set.

        Returns
        -------
        None
        """
        if self.aero_surface.airfoil:
            logger.info("Airfoil information:")
            logger.info("--------------------")
            logger.info(
                "Number of points defining the lift curve: "
                f"{len(self.aero_surface.airfoil_cl.x_array)}"
            )
            logger.info(
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
        logger.info("Roll information of the fin set:")
        logger.info("--------------------------------")
        logger.info(
            f"Geometric constant: {self.aero_surface.roll_geometrical_constant:.3f} m"
        )
        logger.info(
            "Damping interference factor: "
            f"{self.aero_surface.roll_damping_interference_factor:.3f} rad"
        )
        logger.info(
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
        logger.info("Lift information of the fin set:")
        logger.info("--------------------------------")
        logger.info(
            "Lift interference factor: "
            f"{self.aero_surface.lift_interference_factor:.3f} m"
        )
        logger.info(
            "Center of Pressure position in local coordinates: "
            f"({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, "
            f"{self.aero_surface.cpz:.3f})"
        )
        logger.info(
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


class _TailPrints(_AeroSurfacePrints):
    """Class that contains all tail prints."""

    def geometry(self):
        """Prints the geometric information of the tail.

        Returns
        -------
        None
        """
        logger.info("Geometric information of the Tail:")
        logger.info("----------------------------------")
        logger.info(f"Top radius: {self.aero_surface.top_radius:.3f} m")
        logger.info(f"Bottom radius: {self.aero_surface.bottom_radius:.3f} m")
        logger.info(f"Reference radius: {2 * self.aero_surface.rocket_radius:.3f} m")
        logger.info(f"Length: {self.aero_surface.length:.3f} m")
        logger.info(f"Slant length: {self.aero_surface.slant_length:.3f} m")
        logger.info(f"Surface area: {self.aero_surface.surface_area:.6f} m²\n")


class _RailButtonsPrints(_AeroSurfacePrints):
    """Class that contains all rail buttons prints."""

    def geometry(self):
        logger.info("Geometric information of the RailButtons:")
        logger.info("-----------------------------------------")
        logger.info(
            "Distance from one button to the other: "
            f"{self.aero_surface.buttons_distance:.3f} m"
        )
        logger.info(
            "Angular position of the buttons: "
            f"{self.aero_surface.angular_position:.3f} deg\n"
        )


class _AirBrakesPrints(_AeroSurfacePrints):
    """Class that contains all air_brakes prints. Not yet implemented."""

    def geometry(self):
        pass

    def all(self):
        pass


class _GenericSurfacePrints(_AeroSurfacePrints):
    """Class that contains all generic surface prints."""

    def geometry(self):
        logger.info("Geometric information of the Surface:")
        logger.info("----------------------------------")
        logger.info(f"Reference Area: {self.generic_surface.reference_area:.3f} m")
        logger.info(f"Reference length: {2 * self.generic_surface.rocket_radius:.3f} m")

    def all(self):
        """Prints all information of the generic surface.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.lift()


class _LinearGenericSurfacePrints(_AeroSurfacePrints):
    """Class that contains all linear generic surface prints."""

    def geometry(self):
        logger.info("Geometric information of the Surface:")
        logger.info("----------------------------------")
        logger.info(f"Reference Area: {self.generic_surface.reference_area:.3f} m")
        logger.info(f"Reference length: {2 * self.generic_surface.rocket_radius:.3f} m")

    def all(self):
        """Prints all information of the linear generic surface.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.lift()
