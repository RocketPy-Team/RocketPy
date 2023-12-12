from rocketpy.rocket.aero_surface import (
    EllipticalFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)

from .dispersion_model import DispersionModel


class McNoseCone(DispersionModel):
    def __init__(
        self,
        nosecone,
        length=None,
        kind=None,
        base_radius=None,
        bluffness=None,
        rocket_radius=None,
    ):
        self._validate_kind(kind)
        super().__init__(
            nosecone,
            length=length,
            kind=kind,
            base_radius=base_radius,
            bluffness=bluffness,
            rocket_radius=rocket_radius,
            name=None,
        )

    def _validate_kind(self, kind):
        """Validates the kind input. If the kind input argument is not None, it
        must be a list of strings."""
        if kind is not None:
            assert isinstance(kind, list) and all(
                isinstance(member, str) for member in kind
            ), "`kind` must be a list of strings"

    def create_object(self):
        generated_dict = next(self.dict_generator())
        nosecone = NoseCone(
            length=generated_dict["length"],
            kind=generated_dict["kind"],
            base_radius=generated_dict["base_radius"],
            bluffness=generated_dict["bluffness"],
            rocket_radius=generated_dict["rocket_radius"],
            name=generated_dict["name"],
        )
        return nosecone


class McTrapezoidalFins(DispersionModel):
    # special validation for
    # - n
    # - should we vary airfoil somehow? airfoil: List[Union[Tuple[FilePath, StrictStr], None]]
    # - should we vary sweepLength or sweepAngle or both?
    def __init__(
        self,
        trapezoidal_fins,
        n=None,
        root_chord=None,
        tip_chord=None,
        span=None,
        rocket_radius=None,
        cant_angle=None,
        sweep_length=None,
        sweep_angle=None,
        airfoil=None,
    ):
        self._validate_positive_int_list("n", n)
        self._validate_airfoil(airfoil)
        super().__init__(
            trapezoidal_fins,
            n=n,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=span,
            rocket_radius=rocket_radius,
            cant_angle=cant_angle,
            sweep_length=sweep_length,
            sweep_angle=sweep_angle,
            airfoil=airfoil,
            name=None,
        )

    def create_object(self):
        generated_dict = next(self.dict_generator())
        fins = TrapezoidalFins(
            n=generated_dict["n"],
            root_chord=generated_dict["root_chord"],
            tip_chord=generated_dict["tip_chord"],
            span=generated_dict["span"],
            rocket_radius=generated_dict["rocket_radius"],
            cant_angle=generated_dict["cant_angle"],
            sweep_length=generated_dict["sweep_length"],
            airfoil=generated_dict["airfoil"],
            name=generated_dict["name"],
        )
        return fins


class McEllipticalFins(DispersionModel):
    def __init__(
        self,
        elliptical_fins=None,
        n=None,
        root_chord=None,
        span=None,
        rocket_radius=None,
        cant_angle=None,
        airfoil=None,
    ):
        self._validate_positive_int_list("n", n)
        self._validate_airfoil(airfoil)
        super().__init__(
            elliptical_fins,
            n=n,
            root_chord=root_chord,
            span=span,
            rocket_radius=rocket_radius,
            cant_angle=cant_angle,
            airfoil=airfoil,
            name=None,
        )

    def create_object(self):
        generated_dict = next(self.dict_generator())
        fins = EllipticalFins(
            n=generated_dict["n"],
            root_chord=generated_dict["root_chord"],
            span=generated_dict["span"],
            rocket_radius=generated_dict["rocket_radius"],
            cant_angle=generated_dict["cant_angle"],
            airfoil=generated_dict["airfoil"],
            name=generated_dict["name"],
        )
        return fins


class McTail(DispersionModel):
    def __init__(
        self,
        tail,
        top_radius=None,
        bottom_radius=None,
        length=None,
        rocket_radius=None,
    ):
        super().__init__(
            tail,
            top_radius=top_radius,
            bottom_radius=bottom_radius,
            length=length,
            rocket_radius=rocket_radius,
            name=None,
        )

    def create_object(self):
        generated_dict = next(self.dict_generator())
        tail = Tail(
            top_radius=generated_dict["top_radius"],
            bottom_radius=generated_dict["bottom_radius"],
            length=generated_dict["length"],
            rocket_radius=generated_dict["rocket_radius"],
            name=generated_dict["name"],
        )
        return tail


class McRailButtons(DispersionModel):
    def __init__(
        self,
        rail_buttons=None,
        buttons_distance=None,
        angular_position=None,
    ):
        super().__init__(
            rail_buttons,
            buttons_distance=buttons_distance,
            angular_position=angular_position,
            name=None,
        )

    def create_object(self):
        generated_dict = next(self.dict_generator())
        rail_buttons = RailButtons(
            buttons_distance=generated_dict["buttons_distance"],
            angular_position=generated_dict["angular_position"],
            name=generated_dict["name"],
        )
        return rail_buttons
