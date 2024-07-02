"""
Defines the StochasticNoseCone, StochasticTrapezoidalFins,
StochasticEllipticalFins, StochasticTail and StochasticRailButtons classes.
"""

from rocketpy.rocket.aero_surface import (
    EllipticalFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)

from .stochastic_model import StochasticModel


class StochasticNoseCone(StochasticModel):
    """The ``StochasticNoseCone`` class inherits from the StochasticModel. This
    class is used to store a ``NoseCone`` object along with the uncertainty
    of its parameters.

    See Also
    --------
    :ref:`stochastic_model` and
    :class:`NoseCone <rocketpy.rocket.aero_surface.NoseCone>`

    Attributes
    ----------
    object : NoseCone
        NoseCone object to be used as a base for the Stochastic nose cone.
    length : tuple, list, int, float
        Length of the nose cone in meters.
    kind : list[str]
        List with a string representing the kind of nose cone.
    base_radius : tuple, list, int, float
        Nose cone base radius in meters.
    bluffness : tuple, list, int, float
        Bluffness of the nose cone.
    rocket_radius : tuple, list, int, float
        The reference rocket radius used for lift coefficient normalization, in
        meters.
    name : list[str]
        List with the name of the NoseCone. This attribute can not be randomized.
    """

    # TODO: Never vary the kind of the nose cone. Fixed parameter.

    def __init__(
        self,
        nosecone,
        length=None,
        kind=None,  # TODO: Never vary the kind of the nose cone. Fixed parameter.
        base_radius=None,
        bluffness=None,
        rocket_radius=None,
    ):
        """Initializes the Stochastic Nose Cone class.

        See Also
        --------
        See the :ref:`stochastic_model` and :ref:`nose_cone_class` for further
        information.

        Parameters
        ----------
        nosecone : NoseCone
            NoseCone object to be used as a base for the Stochastic nose cone.
        length : tuple, list, int, float
            Length of the nose cone in meters.
        base_radius : tuple, list, int, float
            Nose cone base radius in meters.
        bluffness : tuple, list, int, float
            Bluffness of the nose cone.
        rocket_radius : tuple, list, int, float
            The reference rocket radius used for lift coefficient normalization,
            in meters.
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
            # TODO: Never vary the kind of the nose cone. It is a fixed parameter.
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
        return NoseCone(**generated_dict)


class StochasticTrapezoidalFins(StochasticModel):
    """A Stochastic Trapezoidal Fins class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and
    :class:`TrapezoidalFins <rocketpy.TrapezoidalFins>`

    Attributes
    ----------
    object : TrapezoidalFins
        TrapezoidalFins object to be used for validation.
    n : list[int]
        List with an integer representing the number of fins. This attribute
        can be randomized.
    root_chord : tuple, list, int, float
        Root chord of the fins in meters.
    tip_chord : tuple, list, int, float
        Tip chord of the fins in meters.
    span : tuple, list, int, float
        Span of the fins in meters.
    rocket_radius : tuple, list, int, float
        Rocket radius of the fins in meters.
    cant_angle : tuple, list, int, float
        Cant angle of the fins in degrees.
    sweep_length : tuple, list, int, float
        Sweep length of the fins in meters.
    sweep_angle : tuple, list, int, float
        Sweep angle of the fins in degrees.
    airfoil : list
        List of tuples in the form of (airfoil file path, airfoil name).
    name : list[str]
        List with the object name. This attribute can not be randomized.
    """

    def __init__(
        self,
        trapezoidal_fins,
        n=None,  # TODO: Never vary the number of fins. It is a fixed parameter.
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
        root_chord : tuple, list, int, float
            Root chord of the fins in meters.
        tip_chord : tuple, list, int, float
            Tip chord of the fins in meters.
        span : tuple, list, int, float
            Span of the fins in meters.
        rocket_radius : tuple, list, int, float
            Rocket radius of the fins in meters.
        cant_angle : tuple, list, int, float
            Cant angle of the fins in degrees.
        sweep_length : tuple, list, int, float
            Sweep length of the fins in meters.
        sweep_angle : tuple, list, int, float
            Sweep angle of the fins in degrees.
        airfoil : list[tuple]
            List of tuples in the form of (airfoil file path, airfoil name).
        """
        # TODO: never vary the number of fins. It is a fixed parameter.
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
        return TrapezoidalFins(**generated_dict)


class StochasticEllipticalFins(StochasticModel):
    """A Stochastic Elliptical Fins class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and
    :class:`EllipticalFins <rocketpy.EllipticalFins>`

    Attributes
    ----------
    object : EllipticalFins
        EllipticalFins object to be used for validation.
    n : list[int]
        List with an integer representing the number of fins. This attribute
        can be randomized.
    root_chord : tuple, list, int, float
        Root chord of the fins in meters.
    span : tuple, list, int, float
        Span of the fins in meters.
    rocket_radius : tuple, list, int, float
        Rocket radius of the fins in meters.
    cant_angle : tuple, list, int, float
        Cant angle of the fins in degrees.
    airfoil : list
        List of tuples in the form of (airfoil file path, airfoil name).
    name : list[str]
        List with the fins object name. This attribute can not be randomized.
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
        root_chord : tuple, list, int, float
            Root chord of the fins in meters.
        span : tuple, list, int, float
            Span of the fins in meters.
        rocket_radius : tuple, list, int, float
            Rocket radius of the fins in meters.
        cant_angle : tuple, list, int, float
            Cant angle of the fins in degrees.
        airfoil : list[tuple]
            List of tuples in the form of (airfoil file path, airfoil name).
        """
        # TODO: never vary the number of fins. It is a fixed parameter.
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
        return EllipticalFins(**generated_dict)


class StochasticTail(StochasticModel):
    """A Stochastic Tail class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`Tail <rocketpy.Tail>`

    Attributes
    ----------
    object : Tail
        Tail object to be used for validation.
    top_radius : tuple, list, int, float
        Top radius of the tail in meters.
    bottom_radius : tuple, list, int, float
        Bottom radius of the tail in meters.
    length : tuple, list, int, float
        Length of the tail in meters.
    rocket_radius : tuple, list, int, float
        Rocket radius of the tail in meters.
    name : list[str]
        List with the name of the tail object. This cannot be randomized.
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
        :ref:`stochastic_model` and :class:`Tail <rocketpy.Tail>`

        Parameters
        ----------
        tail : Tail
            Tail object to be used for validation.
        top_radius : tuple, list, int, float
            Top radius of the tail in meters.
        bottom_radius : tuple, list, int, float
            Bottom radius of the tail in meters.
        length : tuple, list, int, float
            Length of the tail in meters.
        rocket_radius : tuple, list, int, float
            Rocket radius of the tail in meters.
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
        Tail
            Tail object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        return Tail(**generated_dict)


class StochasticRailButtons(StochasticModel):
    """A Stochastic RailButtons class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`RailButtons <rocketpy.rocket.RailButtons>`

    Attributes
    ----------
    object : RailButtons
        RailButtons object to be used for validation.
    rail_buttons : list
        List of RailButton objects.
    buttons_distance : tuple, list, int, float
        Distance between the buttons in meters.
    angular_position : tuple, list, int, float
        Angular position of the buttons in degrees.
    name : list[str]
        List with the name of the object. This attribute can not be randomized.
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
            Distance between the buttons in meters.
        angular_position : tuple, list, int, float
            Angular position of the buttons in degrees.
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
        return RailButtons(**generated_dict)
