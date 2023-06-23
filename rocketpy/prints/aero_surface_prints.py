__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

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
        self.nosecone = nosecone
        return None

    def identity(self):
        """Prints the identity of the nosecone.

        Returns
        -------
        None
        """
        print(f"Identification of the NoseCone:")
        print(f"-------------------------------")
        print(f"Name: {self.nosecone.name}")
        print(f"Python Class: {str(self.nosecone.__class__)}\n")
        return None

    def geometry(self):
        """Prints the geometric information of the nosecone.

        Returns
        -------
        None
        """
        print(f"Geometric information of NoseCone:")
        print(f"----------------------------------")
        print(f"Length: {self.nosecone.length:.3f} m")
        print(f"Kind: {self.nosecone.kind}")
        print(f"Base radius: {self.nosecone.baseRadius:.3f} m")
        print(f"Reference rocket radius: {self.nosecone.rocketRadius:.3f} m")
        print(f"Radius ratio: {self.nosecone.radiusRatio:.3f}\n")
        return None

    def aerodynamic(self):
        """Prints the aerodynamic information of the nosecone.

        Returns
        -------
        None
        """
        print(f"Aerodynamic information of NoseCone:")
        print(f"------------------------------------")
        print(f"Center of Pressure position in local coordinates: {self.nosecone.cp} m")
        print(
            f"Lift coefficient derivative at Mach 0 and AoA 0: {self.nosecone.clalpha(0):.3f} 1/rad\n"
        )
        return None

    def all(self):
        """Prints all information of the nosecone.

        Returns
        -------
        None
        """
        self.identity()
        self.geometry()
        self.aerodynamic()
        return None
