from dataclasses import dataclass

from ..plots.fluid_plots import _FluidPlots
from ..prints.fluid_prints import _FluidPrints


@dataclass
class Fluid:
    """Class that represents a fluid.

    Attributes
    ----------
    name : str
        Name of the fluid.
    density : float
        Density of the fluid in kg/m³.
    """

    name: str
    density: float

    def __post_init__(self):
        """Post initialization method.

        Raises
        ------
        ValueError
            If the name is not a string.
        ValueError
            If the density is not a positive number.
        """

        if not isinstance(self.name, str):  # pragma: no cover
            raise ValueError("The name must be a string.")
        if self.density < 0:  # pragma: no cover
            raise ValueError("The density must be a positive number.")

        # Initialize plots and prints object
        self.prints = _FluidPrints(self)
        self.plots = _FluidPlots(self)

    def __repr__(self):
        """Representation method.

        Returns
        -------
        str
            String representation of the class.
        """

        return f"Fluid(name={self.name}, density={self.density})"

    def __str__(self):
        """String method.

        Returns
        -------
        str
            String representation of the class.
        """

        return f"Fluid: {self.name}"

    def to_dict(self, include_outputs=False):  # pylint: disable=unused-argument
        return {"name": self.name, "density": self.density}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["density"])
