class _PlatePrints:
    """Class that holds print methods for the Plate class.

    Attributes
    ----------
    _PlatePrints.plate : Plate
        Plate object that will be used for the prints.
    """

    def __init__(self, plate) -> None:
        """Initializes _PlatePrints class.

        Parameters
        ----------
        plate: Plate
            Plate instance.
        """
        self.plate = plate

    def len_points_print(self) -> None:
        """Prints the number of points that form the plate."""
        len_points = len(self.plate.points)
        print(f"Number of points that form the plate: {len_points}")

    def properties_material(self) -> None:
        """Prints material properties including relative magnetic permeability,
        thickness, area, and volume.
        """
        print(
            f"Relative magnetic permeability of the material: {self.plate.relative_magnetic_permeability}"
        )
        print(
            f"Absolute magnetic permeability: {self.plate.absolute_magnetic_permeability} H/m"
        )
        print(f"Thickness: {self.plate.thickness} m")
        print(f"Area: {self.plate.area} m^2")
        print(f"Volume: {self.plate.volume} m^3")

    def magnetic_distortion_matrix(self) -> None:
        """Prints the magnetic distortion matrix at all recorded positions."""
        for key in self.plate._magnetic_distortion_matrices:
            print(
                f"Magnetic distortion matrix at position: {key} is {self.plate._magnetic_distortion_matrices[key]}"
            )

    def all(self) -> None:
        """Prints all print methods about the Plate."""
        print(f"\n{self.plate.name} information: ")
        self.properties_material()
        self.len_points_print()
        self.magnetic_distortion_matrix()
