from rocketpy.rocket.aero_surface import (
    EllipticalFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)

from .stochastic_model import StochasticModel


class StochasticNoseCone(StochasticModel):
    """A Stochastic Nose Cone class that inherits from StochasticModel. This
    class is used to receive a NoseCone object and information about the
    dispersion of its parameters and generate a random nose cone object based
    on the provided information.

    See Also
    --------
    :ref:`stochastic_model`

    Attributes
    ----------
    object : NoseCone
        NoseCone object to be used for validation.
    length : tuple, list, int, float
        Length of the nose cone in meters. Follows the standard input format of
        Stochastic Models.
    kind : list
        List of strings representing the kind of nose cone. Follows the standard
        input format of Stochastic Models.
    base_radius : tuple, list, int, float
        Base radius of the nose cone in meters. Follows the standard input
        format of Stochastic Models.
    bluffness : tuple, list, int, float
        Bluffness of the nose cone. Follows the standard input format of
        Stochastic Models.
    rocket_radius : tuple, list, int, float
        Rocket radius of the nose cone in meters. Follows the standard input
        format of Stochastic Models.
    name : list
        List of names. This attribute can not be randomized.
    """

    def __init__(
        self,
        nosecone,
        length=None,
        kind=None,
        base_radius=None,
        bluffness=None,
        rocket_radius=None,
    ):
        """Initializes the Stochastic Nose Cone class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        nosecone : NoseCone
            NoseCone object to be used for validation.
        length : tuple, list, int, float
            Length of the nose cone in meters. Follows the standard input format
            of Stochastic Models.
        kind : list
            List of strings representing the kind of nose cone. Follows the
            standard input format of Stochastic Models.
        base_radius : tuple, list, int, float
            Base radius of the nose cone in meters. Follows the standard input
            format of Stochastic Models.
        bluffness : tuple, list, int, float
            Bluffness of the nose cone. Follows the standard input format of
            Stochastic Models.
        rocket_radius : tuple, list, int, float
            Rocket radius of the nose cone in meters. Follows the standard input
            format of Stochastic Models.
        """
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
        """Creates and returns a NoseCone object from the randomly generated
        input arguments.

        Returns
        -------
        nosecone : NoseCone
            NoseCone object with the randomly generated input arguments.
        """
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


class StochasticTrapezoidalFins(StochasticModel):
    """A Stochastic Trapezoidal Fins class that inherits from StochasticModel.
    This class is used to receive a TrapezoidalFins object and information about
    the dispersion of its parameters and generate a random trapezoidal fins
    object based on the provided information.

    See Also
    --------
    :ref:`stochastic_model`

    Attributes
    ----------
    object : TrapezoidalFins
        TrapezoidalFins object to be used for validation.
    n : list of ints
        List of integers representing the number of fins. Follows the standard
        input format of Stochastic Models.
    root_chord : tuple, list, int, float
        Root chord of the fins in meters. Follows the standard input format of
        Stochastic Models.
    tip_chord : tuple, list, int, float
        Tip chord of the fins in meters. Follows the standard input format of
        Stochastic Models.
    span : tuple, list, int, float
        Span of the fins in meters. Follows the standard input format of
        Stochastic Models.
    rocket_radius : tuple, list, int, float
        Rocket radius of the fins in meters. Follows the standard input format
        of Stochastic Models.
    cant_angle : tuple, list, int, float
        Cant angle of the fins in degrees. Follows the standard input format of
        Stochastic Models.
    sweep_length : tuple, list, int, float
        Sweep length of the fins in meters. Follows the standard input format of
        Stochastic Models.
    sweep_angle : tuple, list, int, float
        Sweep angle of the fins in degrees. Follows the standard input format of
        Stochastic Models.
    airfoil : list
        List of tuples in the form of (airfoil file path, airfoil name).
    name : list
        List of names. This attribute can not be randomized.
    """

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
        """Initializes the Stochastic Trapezoidal Fins class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        trapezoidal_fins : TrapezoidalFins
            TrapezoidalFins object to be used for validation.
        n : list of ints
            List of integers representing the number of fins. Follows the
            standard input format of Stochastic Models.
        root_chord : tuple, list, int, float
            Root chord of the fins in meters. Follows the standard input format
            of Stochastic Models.
        tip_chord : tuple, list, int, float
            Tip chord of the fins in meters. Follows the standard input format
            of Stochastic Models.
        span : tuple, list, int, float
            Span of the fins in meters. Follows the standard input format of
            Stochastic Models.
        rocket_radius : tuple, list, int, float
            Rocket radius of the fins in meters. Follows the standard input
            format of Stochastic Models.
        cant_angle : tuple, list, int, float
            Cant angle of the fins in degrees. Follows the standard input format
            of Stochastic Models.
        sweep_length : tuple, list, int, float
            Sweep length of the fins in meters. Follows the standard input
            format of Stochastic Models.
        sweep_angle : tuple, list, int, float
            Sweep angle of the fins in degrees. Follows the standard input
            format of Stochastic Models.
        airfoil : list
            List of tuples in the form of (airfoil file path, airfoil name).
        """
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
        """Creates and returns a TrapezoidalFins object from the randomly
        generated input arguments.

        Returns
        -------
        fins : TrapezoidalFins
            TrapezoidalFins object with the randomly generated input arguments.
        """
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


class StochasticEllipticalFins(StochasticModel):
    """A Stochastic Elliptical Fins class that inherits from StochasticModel.
    This class is used to receive a EllipticalFins object and information about
    the dispersion of its parameters and generate a random elliptical fins
    object based on the provided information.

    See Also
    --------
    :ref:`stochastic_model`

    Attributes
    ----------
    object : EllipticalFins
        EllipticalFins object to be used for validation.
    n : list of ints
        List of integers representing the number of fins. Follows the standard
        input format of Stochastic Models.
    root_chord : tuple, list, int, float
        Root chord of the fins in meters. Follows the standard input format of
        Stochastic Models.
    span : tuple, list, int, float
        Span of the fins in meters. Follows the standard input format of
        Stochastic Models.
    rocket_radius : tuple, list, int, float
        Rocket radius of the fins in meters. Follows the standard input format
        of Stochastic Models.
    cant_angle : tuple, list, int, float
        Cant angle of the fins in degrees. Follows the standard input format of
        Stochastic Models.
    airfoil : list
        List of tuples in the form of (airfoil file path, airfoil name).
    name : list
        List of names. This attribute can not be randomized.
    """

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
        """Initializes the Stochastic Elliptical Fins class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        elliptical_fins : EllipticalFins
            EllipticalFins object to be used for validation.
        n : list of ints
            List of integers representing the number of fins. Follows the
            standard input format of Stochastic Models.
        root_chord : tuple, list, int, float
            Root chord of the fins in meters. Follows the standard input format
            of Stochastic Models.
        span : tuple, list, int, float
            Span of the fins in meters. Follows the standard input format of
            Stochastic Models.
        rocket_radius : tuple, list, int, float
            Rocket radius of the fins in meters. Follows the standard input
            format of Stochastic Models.
        cant_angle : tuple, list, int, float
            Cant angle of the fins in degrees. Follows the standard input format
            of Stochastic Models.
        airfoil : list
            List of tuples in the form of (airfoil file path, airfoil name).
        """
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
        """Creates and returns a EllipticalFins object from the randomly
        generated input arguments.

        Returns
        -------
        fins : EllipticalFins
            EllipticalFins object with the randomly generated input arguments.
        """
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


class StochasticTail(StochasticModel):
    """A Stochastic Tail class that inherits from StochasticModel. This class
    is used to receive a Tail object and information about the dispersion of its
    parameters and generate a random tail object based on the provided
    information.

    See Also
    --------
    :ref:`stochastic_model`

    Attributes
    ----------
    object : Tail
        Tail object to be used for validation.
    top_radius : tuple, list, int, float
        Top radius of the tail in meters. Follows the standard input format of
        Stochastic Models.
    bottom_radius : tuple, list, int, float
        Bottom radius of the tail in meters. Follows the standard input format
        of Stochastic Models.
    length : tuple, list, int, float
        Length of the tail in meters. Follows the standard input format of
        Stochastic Models.
    rocket_radius : tuple, list, int, float
        Rocket radius of the tail in meters. Follows the standard input format
        of Stochastic Models.
    name : list
        List of names. This attribute can not be randomized.
    """

    def __init__(
        self,
        tail,
        top_radius=None,
        bottom_radius=None,
        length=None,
        rocket_radius=None,
    ):
        """Initializes the Stochastic Tail class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        tail : Tail
            Tail object to be used for validation.
        top_radius : tuple, list, int, float
            Top radius of the tail in meters. Follows the standard input format
            of Stochastic Models.
        bottom_radius : tuple, list, int, float
            Bottom radius of the tail in meters. Follows the standard input
            format of Stochastic Models.
        length : tuple, list, int, float
            Length of the tail in meters. Follows the standard input format of
            Stochastic Models.
        rocket_radius : tuple, list, int, float
            Rocket radius of the tail in meters. Follows the standard input
            format of Stochastic Models.
        """
        super().__init__(
            tail,
            top_radius=top_radius,
            bottom_radius=bottom_radius,
            length=length,
            rocket_radius=rocket_radius,
            name=None,
        )

    def create_object(self):
        """Creates and returns a Tail object from the randomly generated input
        arguments.

        Returns
        -------
        tail : Tail
            Tail object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        tail = Tail(
            top_radius=generated_dict["top_radius"],
            bottom_radius=generated_dict["bottom_radius"],
            length=generated_dict["length"],
            rocket_radius=generated_dict["rocket_radius"],
            name=generated_dict["name"],
        )
        return tail


class StochasticRailButtons(StochasticModel):
    """A Stochastic RailButtons class that inherits from StochasticModel. This
    class is used to receive a RailButtons object and information about the
    dispersion of its parameters and generate a random rail buttons object based
    on the provided information.

    See Also
    --------
    :ref:`stochastic_model`

    Attributes
    ----------
    object : RailButtons
        RailButtons object to be used for validation.
    rail_buttons : list
        List of RailButton objects. Follows the standard input format of
        Stochastic Models.
    buttons_distance : tuple, list, int, float
        Distance between the buttons in meters. Follows the standard input
        format of Stochastic Models.
    angular_position : tuple, list, int, float
        Angular position of the buttons in degrees. Follows the standard input
        format of Stochastic Models.
    name : list
        List of names. This attribute can not be randomized.
    """

    def __init__(
        self,
        rail_buttons=None,
        buttons_distance=None,
        angular_position=None,
    ):
        """Initializes the Stochastic RailButtons class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        rail_buttons : RailButtons
            RailButtons object to be used for validation.
        buttons_distance : tuple, list, int, float
            Distance between the buttons in meters. Follows the standard input
            format of Stochastic Models.
        angular_position : tuple, list, int, float
            Angular position of the buttons in degrees. Follows the standard
            input format of Stochastic Models.
        """
        super().__init__(
            rail_buttons,
            buttons_distance=buttons_distance,
            angular_position=angular_position,
            name=None,
        )

    def create_object(self):
        """Creates and returns a RailButtons object from the randomly generated
        input arguments.

        Returns
        -------
        rail_buttons : RailButtons
            RailButtons object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        rail_buttons = RailButtons(
            buttons_distance=generated_dict["buttons_distance"],
            angular_position=generated_dict["angular_position"],
            name=generated_dict["name"],
        )
        return rail_buttons
