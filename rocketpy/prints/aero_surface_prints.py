from abc import ABC, abstractmethod


class _AeroSurfacePrints(ABC):
    def __init__(self, aero_surface):
        self.aero_surface = aero_surface
        return None

    def identity(self):
        """Prints the identity of the aero surface.

        Returns
        -------
        None
        """
        print(f"Identification of the AeroSurface:")
        print(f"----------------------------------")
        print(f"Name: {self.aero_surface.name}")
        print(f"Python Class: {str(self.aero_surface.__class__)}\n")
        return None

    @abstractmethod
    def geometry(self):
        pass

    def lift(self):
        """Prints the lift information of the aero surface.

        Returns
        -------
        None
        """
        print(f"Lift information of the AeroSurface:")
        print(f"-----------------------------------")
        print(
            f"Center of Pressure position in local coordinates: ({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, {self.aero_surface.cpz:.3f})"
        )
        print(
            f"Lift coefficient derivative at Mach 0 and AoA 0: {self.aero_surface.clalpha(0):.3f} 1/rad\n"
        )
        return None

    def all(self):
        """Prints all information of the aero surface.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.lift()
        return None


class _NoseConePrints(_AeroSurfacePrints):
    """Class that contains all nosecone prints."""

    def __init__(self, nosecone):
        """Initialize the class

        Parameters
        ----------
        nosecone : rocketpy.AeroSurface.NoseCone
            Nosecone object to be printed

        Returns
        -------
        None
        """
        super().__init__(nosecone)
        return None

    def geometry(self):
        """Prints the geometric information of the nosecone.

        Returns
        -------
        None
        """
        print(f"Geometric information of NoseCone:")
        print(f"----------------------------------")
        print(f"Length: {self.aero_surface.length:.3f} m")
        print(f"Kind: {self.aero_surface.kind}")
        print(f"Base radius: {self.aero_surface.base_radius:.3f} m")
        print(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        print(f"Reference radius ratio: {self.aero_surface.radius_ratio:.3f}\n")
        return None


class _FinsPrints(_AeroSurfacePrints):
    def __init__(self, fin_set):
        """Initialize the class

        Parameters
        ----------
        fin_set : rocketpy.AeroSurface.fin_set
            fin_set object to be printed

        Returns
        -------
        None
        """
        super().__init__(fin_set)
        return None

    def geometry(self):
        print(f"Geometric information of the fin set:")
        print(f"-------------------------------------")
        print(f"Number of fins: {self.aero_surface.n}")
        print(f"Reference rocket radius: {self.aero_surface.rocket_radius:.3f} m")
        try:
            print(f"Tip chord: {self.aero_surface.tip_chord:.3f} m")
        except AttributeError:
            pass  # it isn't a trapezoidal fin, just don't worry about tip chord
        print(f"Root chord: {self.aero_surface.root_chord:.3f} m")
        print(f"Span: {self.aero_surface.span:.3f} m")
        print(
            f"Cant angle: {self.aero_surface.cant_angle:.3f} ° or {self.aero_surface.cant_angle_rad:.3f} rad"
        )
        print(f"Longitudinal section area: {self.aero_surface.Af:.3f} m²")
        print(f"Aspect ratio: {self.aero_surface.AR:.3f} ")
        print(f"Gamma_c: {self.aero_surface.gamma_c:.3f} m")
        print(f"Mean aerodynamic chord: {self.aero_surface.Yma:.3f} m\n")
        return None

    def airfoil(self):
        """Prints out airfoil related information of the fin set.

        Returns
        -------
        None
        """
        if self.aero_surface.airfoil:
            print(f"Airfoil information:")
            print(f"--------------------")
            print(
                f"Number of points defining the lift curve: {len(self.aero_surface.airfoil_cl.x_array)}"
            )
            print(
                f"Lift coefficient derivative at Mach 0 and AoA 0: {self.aero_surface.clalpha(0):.5f} 1/rad\n"
            )
        return None

    def roll(self):
        """Prints out information about roll parameters
        of the fin set.

        Returns
        -------
        None
        """
        print(f"Roll information of the fin set:")
        print(f"--------------------------------")
        print(
            f"Geometric constant: {self.aero_surface.roll_geometrical_constant:.3f} m"
        )
        print(
            f"Damping interference factor: {self.aero_surface.roll_damping_interference_factor:.3f} rad"
        )
        print(
            f"Forcing interference factor: {self.aero_surface.roll_forcing_interference_factor:.3f} rad\n"
        )
        return None

    def lift(self):
        """Prints out information about lift parameters
        of the fin set.

        Returns
        -------
        None
        """
        print(f"Lift information of the fin set:")
        print(f"--------------------------------")
        print(
            f"Lift interference factor: {self.aero_surface.lift_interference_factor:.3f} m"
        )
        print(
            f"Center of Pressure position in local coordinates: ({self.aero_surface.cpx:.3f}, {self.aero_surface.cpy:.3f}, {self.aero_surface.cpz:.3f})"
        )
        print(
            f"Lift Coefficient derivative (single fin) at Mach 0 and AoA 0: {self.aero_surface.clalpha_single_fin(0):.3f}"
        )
        print(
            f"Lift Coefficient derivative (fin set) at Mach 0 and AoA 0: {self.aero_surface.clalpha_multiple_fins(0):.3f}"
        )
        return None

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
        return None


class _TrapezoidalFinsPrints(_FinsPrints):
    def __init__(self, fin_set):
        """Initialize the class

        Parameters
        ----------
        fin_set : rocketpy.AeroSurface.fin_set
            fin_set object to be printed

        Returns
        -------
        None
        """
        super().__init__(fin_set)
        return None


class _EllipticalFinsPrints(_FinsPrints):
    """Class that contains all elliptical fins prints."""

    def __init__(self, fin_set):
        """Initialize the class

        Parameters
        ----------
        fin_set : rocketpy.AeroSurface.fin_set
            fin_set object to be printed

        Returns
        -------
        None
        """
        super().__init__(fin_set)
        return None


class _TailPrints(_AeroSurfacePrints):
    """Class that contains all tail prints."""

    def __init__(self, tail):
        """Initialize the class

        Parameters
        ----------
        tail : rocketpy.AeroSurface.Tail
            Tail object to be printed

        Returns
        -------
        None
        """
        super().__init__(tail)
        return None

    def geometry(self):
        """Prints the geometric information of the tail.

        Returns
        -------
        None
        """
        print(f"Geometric information of the Tail:")
        print(f"----------------------------------")
        print(f"Top radius: {self.aero_surface.top_radius:.3f} m")
        print(f"Bottom radius: {self.aero_surface.bottom_radius:.3f} m")
        print(f"Reference radius: {2*self.aero_surface.rocket_radius:.3f} m")
        print(f"Length: {self.aero_surface.length:.3f} m")
        print(f"Slant length: {self.aero_surface.slant_length:.3f} m")
        print(f"Surface area: {self.aero_surface.surface_area:.6f} m²\n")
        return None


class _RailButtonsPrints(_AeroSurfacePrints):
    """Class that contains all rail buttons prints."""

    def __init__(self, rail_buttons):
        super().__init__(rail_buttons)
        return None

    def geometry(self):
        print(f"Geometric information of the RailButtons:")
        print(f"-----------------------------------------")
        print(
            f"Distance from one button to the other: {self.aero_surface.buttons_distance:.3f} m"
        )
        print(
            f"Angular position of the buttons: {self.aero_surface.angular_position:.3f} deg\n"
        )
        return None
