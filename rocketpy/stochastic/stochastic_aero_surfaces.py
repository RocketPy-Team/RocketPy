"""
Defines the StochasticNoseCone, StochasticTrapezoidalFins,
StochasticEllipticalFins, StochasticFreeFormFins, StochasticTail and
StochasticRailButtons classes.
"""

import numpy as np

from rocketpy.rocket.aero_surface import (
    AirBrakes,
    EllipticalFins,
    FreeFormFins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)

from .custom_sampler import CustomSampler
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
        power=None,
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
            power=power,
            name=None,
        )

    def _validate_kind(self, kind):
        """Validates the kind input. If the kind input argument is not None, it
        must be a list of strings."""
        if kind is not None:
            # TODO: Never vary the kind of the nose cone. It is a fixed parameter.
            if not (
                isinstance(kind, list)
                and all(isinstance(member, str) for member in kind)
            ):
                raise AssertionError("`kind` must be a list of strings")

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


class StochasticFreeFormFins(StochasticModel):
    """A Stochastic Free Form Fins class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and
    :class:`FreeFormFins <rocketpy.FreeFormFins>`

    Attributes
    ----------
    object : FreeFormFins
        FreeFormFins object to be used for validation.
    n : list[int]
        List with an integer representing the number of fins. This attribute
        can be randomized.
    shape_points : tuple, list, numpy.ndarray, int, float
        The (x, y) points defining the fin outline, in meters. Unlike the other
        fin sets, this geometry is a whole list of points rather than a single
        scalar, so the deviation given applies to the outline as a block: every
        coordinate of every point is perturbed, each by its own draw. See the
        ``shape_points`` parameter of :meth:`__init__` for the accepted formats.
    rocket_radius : tuple, list, int, float
        Rocket radius of the fins in meters.
    cant_angle : tuple, list, int, float
        Cant angle of the fins in degrees.
    airfoil : list
        List of tuples in the form of (airfoil file path, airfoil name).
    name : list[str]
        List with the fins object name. This attribute can not be randomized.
    """

    # The outline is the whole nominal value of this input, not a single number.
    array_valued_inputs = ("shape_points",)

    # The distributions that can mean a deviation around a nominal coordinate,
    # which is what perturbing an outline asks of them. The rest of what
    # ``get_distribution`` offers reads its arguments as bounds (``uniform``) or
    # as shape parameters (``wald``, ``gamma``, ``poisson``, ...), neither of
    # which an outline of coordinates can be.
    _outline_distributions = ("normal", "gumbel", "laplace", "logistic")

    def __init__(
        self,
        free_form_fins=None,
        n=None,
        shape_points=None,
        rocket_radius=None,
        cant_angle=None,
        airfoil=None,
    ):
        """Initializes the Stochastic Free Form Fins class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        free_form_fins : FreeFormFins
            FreeFormFins object to be used for validation.
        shape_points : tuple, list, numpy.ndarray, int, float, optional
            The (x, y) points defining the fin outline, in meters. The whole
            outline is perturbed as a block, since a fin shape is only
            meaningful as a complete set of points: the deviation given applies
            to every coordinate of every point, each drawn independently of the
            others. The fin root is held on the body line, so a point nominally
            at ``y = 0`` keeps that value and no point is moved inside the
            airframe. The accepted formats are:

            - ``int`` or ``float``: standard deviation applied to every
              coordinate of the nominal outline, drawn from a normal
              distribution.
            - ``tuple``: ``(standard deviation, distribution name)``, or
              ``(nominal outline, standard deviation[, distribution name])``.
              The distribution must be one of ``"normal"``, ``"gumbel"``,
              ``"laplace"`` or ``"logistic"``, the ones that take the nominal
              coordinate as their centre.
            - ``list`` or ``numpy.ndarray``: either one fixed outline, e.g.
              ``[(0, 0), (0.1, 0.1), (0.1, 0)]``, which is used as given and
              not randomized; or a list of candidate outlines, e.g.
              ``[[(0, 0), (0.1, 0.1), (0.1, 0)], [(0, 0), (0.1, 0.12), (0.1, 0)]]``,
              one of which is chosen per simulation. The candidates need not
              all have the same number of points. An empty list means the
              nominal outline of the object passed, unrandomized, as it does
              for every other argument.
            - ``CustomSampler``: has to yield a whole outline per sample, since
              the value it returns replaces the outline instead of perturbing
              it.
        rocket_radius : tuple, list, int, float, optional
            Rocket radius of the fins in meters.
        cant_angle : tuple, list, int, float, optional
            Cant angle of the fins in degrees.
        airfoil : list[tuple], optional
            List of tuples in the form of (airfoil file path, airfoil name).
        """
        # TODO: never vary the number of fins. It is a fixed parameter.
        self._validate_positive_int_list("n", n)
        self._validate_airfoil(airfoil)
        shape_points = self._validate_shape_points(shape_points)
        super().__init__(
            free_form_fins,
            n=n,
            shape_points=shape_points,
            rocket_radius=rocket_radius,
            cant_angle=cant_angle,
            airfoil=airfoil,
            name=None,
        )

    def _validate_shape_points(self, shape_points):
        """Validate the ``shape_points`` input and normalize it to a form the
        base class can randomize.

        A fin outline is a sequence of points, so it does not fit the
        scalar-per-input assumption the base class makes. Two formats would be
        silently misread if passed straight through, and both are the natural
        thing for a user to write:

        - a bare outline ``[(0, 0), (0.1, 0.1), (0.1, 0)]`` is a ``list``, which
          the base class reads as a list of candidate values and would sample a
          single ``(x, y)`` point from. It is wrapped here so it is treated as
          the one candidate outline it is.
        - a ``(nominal outline, standard deviation)`` tuple carries an outline
          where the base class reads a distribution argument, so the outline is
          checked here and the rest is left to the base class.

        Parameters
        ----------
        shape_points : tuple, list, numpy.ndarray, int, float, optional
            Value of the ``shape_points`` input argument.

        Returns
        -------
        tuple, list, int, float or None
            The input, normalized so the base class randomizes the outline as a
            block. Outlines come back as ``(n, 2)`` arrays of floats.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        if shape_points is None or isinstance(
            shape_points, (int, float, CustomSampler)
        ):
            # A number is a standard deviation around the nominal outline, which
            # the base class looks up and hands to the distribution as an array.
            # A sampler yields whole outlines, so it replaces that machinery.
            return shape_points

        if isinstance(shape_points, tuple):
            return self._validate_shape_points_tuple(shape_points)

        if isinstance(shape_points, (list, np.ndarray)):
            if len(shape_points) == 0:
                # An empty list means the nominal value everywhere else, and
                # nothing about this argument makes it mean something else.
                return []
            if self._is_outline(shape_points):
                # A bare outline is the one candidate it describes. Left as a
                # list of points it would be read as a list of candidates and
                # sampled down to a single (x, y) point.
                return [self._validate_outline(shape_points)]
            return [self._validate_outline(outline) for outline in shape_points]

        raise AssertionError(
            "`shape_points` must be a tuple, list, numpy array, int, or float "
            "or a custom sampler"
        )

    def _validate_shape_points_tuple(self, shape_points):
        """Validate a ``shape_points`` tuple.

        Accepts ``(standard deviation, distribution name)``, in which case the
        nominal outline comes from the object passed, and
        ``(nominal outline, standard deviation[, distribution name])``.

        Parameters
        ----------
        shape_points : tuple
            Value of the ``shape_points`` input argument.

        Returns
        -------
        tuple
            The input tuple, with any nominal outline converted to an ``(n, 2)``
            array of floats so the standard deviation broadcasts over every
            coordinate.

        Raises
        ------
        AssertionError
            If the input is not in a valid format.
        """
        if len(shape_points) not in (2, 3):
            raise AssertionError("'shape_points': tuple must have length 2 or 3")

        if isinstance(shape_points[0], (int, float)):
            # (standard deviation, distribution name), the nominal outline being
            # taken from the object passed. A number in the second item would
            # make the first one the nominal value, which for this argument is
            # an outline rather than a number.
            if not isinstance(shape_points[1], str):
                raise AssertionError(
                    "'shape_points': when the first item of a tuple is a "
                    "standard deviation, the second must be a string naming a "
                    "valid numpy.random distribution function."
                )
            self._validate_outline_distribution(shape_points[1])
            return shape_points

        # (nominal outline, standard deviation[, distribution name]). The second
        # item is checked here rather than left to the base class, which also
        # accepts a string there and would read the outline as the deviation.
        outline = self._validate_outline(shape_points[0])
        if not isinstance(shape_points[1], (int, float)):
            raise AssertionError(
                "'shape_points': second item of tuple must be an int or float "
                "standard deviation."
            )
        if len(shape_points) == 3:
            if not isinstance(shape_points[2], str):
                raise AssertionError(
                    "'shape_points': Third item of tuple must be a string containing "
                    "the name of a valid numpy.random distribution function."
                )
            self._validate_outline_distribution(shape_points[2])
        return (outline,) + tuple(shape_points[1:])

    @classmethod
    def _validate_outline_distribution(cls, distribution_name):
        """Reject distributions that cannot mean a deviation around a coordinate.

        The distribution is called as ``dist_func(nominal_outline, std_dev)``, so
        only the ones that read the first argument as the centre of the draw can
        perturb an outline. ``uniform`` would read the outline as its lower bound
        and the deviation as a single upper bound, leaving an empty range for
        every coordinate above it, and ``wald`` and the shape-parameter
        distributions reject the zeros that a root point has.

        Raises
        ------
        AssertionError
            If the distribution cannot be applied to an outline.
        """
        if distribution_name not in cls._outline_distributions:
            accepted = ", ".join(repr(name) for name in cls._outline_distributions)
            raise AssertionError(
                f"'shape_points': the '{distribution_name}' distribution cannot "
                f"be applied to an outline. Use one of {accepted}, which take "
                "the nominal coordinate as the centre of the deviation."
            )

    @staticmethod
    def _is_outline(value):
        """Return True if ``value`` is a single (x, y) outline.

        Used to tell a bare outline apart from a list of candidate outlines,
        which are the two things a list input can mean. The conversion is what
        decides it: numpy refuses a ragged or non-numeric sequence, and a list
        of candidates whose outlines have different numbers of points is exactly
        that, so those are left for the caller to check one at a time.
        """
        try:
            array = np.asarray(value, dtype=float)
        except (ValueError, TypeError):
            return False
        return array.ndim == 2 and array.shape[1] == 2

    @staticmethod
    def _validate_outline(outline):
        """Validate a single (x, y) fin outline.

        Returns
        -------
        numpy.ndarray
            The outline as an ``(n, 2)`` array of floats.

        Raises
        ------
        AssertionError
            If the outline is not a sequence of at least three (x, y) numbers.
        """
        if not StochasticFreeFormFins._is_outline(outline):
            raise AssertionError(
                "`shape_points` outlines must be sequences of (x, y) numbers, "
                "i.e. have shape (n, 2)."
            )
        array = np.asarray(outline, dtype=float)
        if array.shape[0] < 3:
            raise AssertionError(
                "`shape_points` outlines must have at least 3 points to "
                "enclose an area."
            )
        return array

    # pylint: disable=stop-iteration-return
    def dict_generator(self):
        """Generate the input arguments, with the fin root kept on the body line.

        Yields
        ------
        dict
            Dictionary with the randomly generated input arguments.
        """
        generated_dict = next(super().dict_generator())
        if isinstance(self.shape_points, tuple):
            # Only a perturbed outline can have drifted off the body line. One
            # chosen from a list of candidates, or one a sampler produced, is
            # used exactly as it was given.
            generated_dict["shape_points"] = self._keep_root_on_body_line(
                self.shape_points[0], generated_dict["shape_points"]
            )
            # The outline moved after the base class recorded it, and what the
            # fins are built from is what the record has to hold.
            self._record_draw(generated_dict)
        yield generated_dict

    @staticmethod
    def _keep_root_on_body_line(nominal_outline, sampled_outline):
        """Hold the root of a perturbed outline on the body line.

        :class:`FreeFormFins <rocketpy.FreeFormFins>` measures the span from
        ``y = 0`` and slices the chords over that interval, so a root point that
        drifts off the line puts part of the fin inside the airframe and inflates
        the span those chords are measured against. Points nominally on the line
        are kept there, and no other point is allowed to cross it.

        Parameters
        ----------
        nominal_outline : numpy.ndarray
            The unperturbed outline, which says which points are on the line.
        sampled_outline : numpy.ndarray
            The perturbed outline.

        Returns
        -------
        numpy.ndarray
            The perturbed outline, with its root back on the body line.
        """
        nominal_outline = np.asarray(nominal_outline, dtype=float)
        sampled_outline = np.array(sampled_outline, dtype=float)
        sampled_outline[nominal_outline[:, 1] == 0, 1] = 0.0
        sampled_outline[:, 1] = np.maximum(sampled_outline[:, 1], 0.0)
        return sampled_outline

    def create_object(self):
        """Creates and returns a FreeFormFins object from the randomly
        generated input arguments.

        Returns
        -------
        fins : FreeFormFins
            FreeFormFins object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        return FreeFormFins(**generated_dict)


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


class StochasticAirBrakes(StochasticModel):
    """A Stochastic Air Brakes class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and
    :class:`AirBrakes <rocketpy.AirBrakes>`

    Attributes
    ----------
    object : AirBrakes
        AirBrakes object to be used for validation.
    drag_coefficient_curve : list, str
        The drag coefficient curve of the air brakes can account for
        either the air brakes' drag alone or the combined drag of both
        the rocket and the air brakes.
    drag_coefficient_curve_factor : tuple, list, int, float
        The drag curve factor of the air brakes. This value scales the
        drag coefficient curve to introduce stochastic variability.
    reference_area : tuple, list, int, float
        Reference area used to non-dimensionalize the drag coefficients.
    clamp : bool
        If True, the simulation will clamp the deployment level to 0 or 1 if
        the deployment level is out of bounds. If False, the simulation will
        not clamp the deployment level and will instead raise a warning if
        the deployment level is out of bounds.
    override_rocket_drag : bool
        If False, the air brakes drag coefficient will be added to the
        rocket's power off drag coefficient curve. If True, during the
        simulation, the rocket's power off drag will be ignored and the air
        brakes drag coefficient will be used for the entire rocket instead.
    deployment_level : tuple, list, int, float
        Initial deployment level, ranging from 0 to 1.
    name : list[str]
        List with the air brakes object name. This attribute can't be randomized.
    """

    def __init__(
        self,
        air_brakes,
        drag_coefficient_curve=None,
        drag_coefficient_curve_factor=(1, 0),
        reference_area=None,
        clamp=None,
        override_rocket_drag=None,
        deployment_level=(0, 0),
    ):
        """Initializes the Stochastic AirBrakes class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        air_brakes : AirBrakes
            AirBrakes object to be used for validation.
        drag_coefficient_curve : list, str, optional
            The drag coefficient curve of the air brakes can account for
            either the air brakes' drag alone or the combined drag of both
            the rocket and the air brakes.
        drag_coefficient_curve_factor : tuple, list, int, float, optional
            The drag curve factor of the air brakes. This value scales the
            drag coefficient curve to introduce stochastic variability.
        reference_area : tuple, list, int, float, optional
            Reference area used to non-dimensionalize the drag coefficients.
        clamp : bool, optional
            If True, the simulation will clamp the deployment level to 0 or 1 if
            the deployment level is out of bounds. If False, the simulation will
            not clamp the deployment level and will instead raise a warning if
            the deployment level is out of bounds.
        override_rocket_drag : bool, optional
            If False, the air brakes drag coefficient will be added to the
            rocket's power off drag coefficient curve. If True, during the
            simulation, the rocket's power off drag will be ignored and the air
            brakes drag coefficient will be used for the entire rocket instead.
        deployment_level : tuple, list, int, float, optional
            Initial deployment level, ranging from 0 to 1.
        """
        super().__init__(
            air_brakes,
            drag_coefficient_curve=drag_coefficient_curve,
            drag_coefficient_curve_factor=drag_coefficient_curve_factor,
            reference_area=reference_area,
            clamp=clamp,
            override_rocket_drag=override_rocket_drag,
            deployment_level=deployment_level,
            name=None,
        )

    def create_object(self):
        """Creates and returns an AirBrakes object from the randomly generated
        input arguments.

        Returns
        -------
        air_brake : AirBrakes
            AirBrakes object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        air_brakes = AirBrakes(
            drag_coefficient_curve=generated_dict["drag_coefficient_curve"],
            reference_area=generated_dict["reference_area"],
            clamp=generated_dict["clamp"],
            override_rocket_drag=generated_dict["override_rocket_drag"],
            deployment_level=generated_dict["deployment_level"],
        )
        air_brakes.drag_coefficient *= generated_dict["drag_coefficient_curve_factor"]
        return air_brakes
