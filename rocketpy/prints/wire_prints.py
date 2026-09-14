class _WirePrints:
    """Class that holds print methods for the Wire class.

    Attributes
    ----------
    _WirePrints.wire : Wire
        Wire object that will be used for the prints.
    """

    def __init__(self, wire) -> None:
        """Initializes _WirePrints class.

        Parameters
        ----------
        wire : Wire
            Wire instance.
        """
        self.wire = wire

    def current(self) -> None:
        """Prints the current flowing through the wire."""
        print(f"Current: {self.wire.current} A")

    def endpoints(self) -> None:
        """Prints the wire endpoints coordinates in the body axis coordinate system."""
        print(
            f"Endpoint A is {self.wire._wire_endpoints_bacs[0]}, Endpoint B is {self.wire._wire_endpoints_bacs[1]} in the body axis coordinate system"
        )

    def wire_type(self) -> None:
        """Prints the wire type and, if applicable, the ignition wire function."""
        if self.wire.wire_type == "communications":
            print("Wire type: communications type")
        else:
            print("Wire type: ignition type")
            print(f"Ignition wire function: {self.wire.ignition_wire_function}")

    def magnetic_field_vectors(self) -> None:
        """Prints the magnetic field vectors stored in the wire for all recorded positions."""
        for position, b_field in self.wire.magnetic_field.items():
            if position is not None:
                print(
                    f"Magnetic field vector at position {position} due to the wire is {b_field} T"
                )

    def all(self) -> None:
        """Prints all print methods about the Wire."""
        print(f"\n{self.wire.name} information: ")
        self.wire_type()
        self.current()
        self.magnetic_field_vectors()
        self.endpoints()
