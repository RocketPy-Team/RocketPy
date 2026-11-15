import math
from copy import deepcopy

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.plots.aero_surface_plots import _ControlFinsPlots
from rocketpy.prints.aero_surface_prints import _ControlFinsPrints
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket.aero_surface.fins.fin import Fin
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface
from rocketpy.tools import point_around

# The variables the coefficients of a set of control fins depend on, before
# the deflection of each fin. The Reynolds number plays no part in the fin model.
_FLOW_VARIABLES = ("alpha", "beta", "mach", "pitch_rate", "yaw_rate", "roll_rate")


class ControlFins(GenericSurface):
    """A set of fins a controller turns one by one, such as canards or movable
    tail fins.

    The set is ``n`` copies of one fin, spaced evenly around the rocket, or a
    list of fins given one by one. Each fin has one control, its deflection
    in radians: the angle it is turned by about its hinge line, exactly as a
    cant angle turns a fixed fin. The fins are numbered from the one given,
    the way the angular position grows: with 4 fins at 0 degrees, ``fin_1``
    is at 0, ``fin_2`` at 90, ``fin_3`` at 180 and ``fin_4`` at 270.
    :meth:`info` prints this table. In a flight each deflection is a control
    input named after the set, ``canards.fin_1`` for a set named
    ``"Canards"``. :meth:`deflections` turns a roll command and two sideways
    commands into one deflection per fin; :meth:`effectiveness` gives the
    moment and force coefficients per radian of each, for the controller's
    design.

    The same positive deflection on every fin gives a negative roll moment
    (``w3`` decreases), as a positive cant angle does. One fin deflected
    alone pushes the rocket sideways at that fin: the fin at 0 degrees (on
    the ``+y`` axis) toward ``+x``, the fin at 90 degrees toward ``+y``. On
    canards the nose follows the push, on tail fins it goes the other way.

    The fin model does not stall, so keep the deflection plus the angle of
    attack within a few tenths of a radian; ``deflection_limits`` enforces
    it. A set held at a deflection gives the same forces as the same fins
    added to the rocket with that cant angle.

    Attributes
    ----------
    ControlFins.fin : Fin or None
        The fin the set is copied from, or ``None`` when the fins were given
        one by one.
    ControlFins.n : int
        The number of fins.
    ControlFins.fins : tuple of Fin
        The fins of the set, in order: ``fins[0]`` is ``fin_1``. Each is
        uncanted: its deflection turns it during the flight.
    ControlFins.angular_positions : tuple of float
        The angular position of each fin, in degrees, in order.
    ControlFins.hinge_position : float
        Where the hinge line crosses the root chord, as a fraction of its
        length from the leading edge.
    ControlFins.controls : tuple of str
        ``("fin_1", ..., "fin_n")``: the deflection of each fin.
    ControlFins.control_defaults : dict
        Deflection of each fin at the start of every flight, in radians.
    ControlFins.control_limits : dict
        ``(low, high)`` deflection of each fin, in radians, or ``None``.
    """

    # The fins of the set act at different points around the rocket, and a set
    # deflected unevenly pushes harder in one plane than in the other
    is_axisymmetric = False

    def __init__(  # pylint: disable=too-many-arguments
        self,
        fin,
        n=None,
        hinge_position=0.5,
        deflection_limits=None,
        name="ControlFins",
        active_during="always",
        active=True,
    ):
        """Build a set of control fins.

        Parameters
        ----------
        fin : TrapezoidalFin, EllipticalFin or FreeFormFin, or a sequence of them
            One fin: the fin every fin of the set is a copy of. Its
            ``angular_position`` is where ``fin_1`` sits, and its
            ``cant_angle`` is the deflection every fin starts each flight at
            (0 for an uncanted fin). A sequence of fins: the fins of the set,
            one by one, ``fin_1`` first, each at its own ``angular_position``
            and starting each flight at its own ``cant_angle``; they may
            differ in shape and size. Angles are in degrees, while the
            deflections a controller writes are in radians. The fins given
            are not changed or added to the rocket themselves. The set's
            reference area and length are the first fin's.
        n : int, optional
            The number of fins. With one fin, how many copies to space evenly
            around the rocket, 4 by default. With a sequence of fins it is
            their count, and need not be given.
        hinge_position : float, optional
            Where the hinge line, the axis the fin turns about, crosses the
            root chord: a fraction of the root chord from the leading edge,
            from 0 (the leading edge) to 1 (the trailing edge). It sets where
            the fin moves to when it turns. Default 0.5, the middle of the
            root chord, which is also where a cant angle turns a fixed fin.
        deflection_limits : tuple of float, optional
            ``(low, high)``: the smallest and largest deflection of each fin,
            in radians, for example ``(-0.2, 0.2)``: how far the fins can
            turn, and the way to keep them where the fin model holds, within
            a few tenths of a radian. A deflection asked for outside them is
            moved to the nearest limit. Default ``None``, no limits.
        name : str, optional
            Name of the set. It also names the control inputs, in snake_case:
            ``"Canards"`` gives ``canards.fin_1``, ``canards.fin_2`` and so on.
            Default ``"ControlFins"``, which gives ``control_fins.fin_1``.
        active_during : str, optional
            The motor phase the fins produce force in: ``"always"`` (the
            default), ``"power_on"`` or ``"power_off"``.
        active : bool, optional
            Whether the fins start each flight switched on. Default ``True``.

        Raises
        ------
        TypeError
            If ``fin`` is neither an individual fin nor a sequence of them, or
            ``n`` is not a whole number.
        ValueError
            If ``n`` is smaller than 1 or differs from the number of fins
            given, ``deflection_limits`` is not ``(low, high)`` with
            ``low <= high``, or ``hinge_position`` is not between 0 and 1.
        """
        given = self.__check_fins(fin, n)
        n = len(given)
        if not 0 <= hinge_position <= 1:
            raise ValueError(
                "The hinge position is a fraction of the root chord from the "
                f"leading edge, between 0 and 1, got {hinge_position!r}."
            )

        self.fin = deepcopy(fin) if isinstance(fin, Fin) else None
        self.n = n
        self._hinge_position = hinge_position
        self.deflection_limits = (
            None if deflection_limits is None else tuple(deflection_limits)
        )
        self.angular_positions = tuple(each.angular_position for each in given)
        self.fins = tuple(
            self._copy_fin(each, each.angular_position, f"{name} fin {i + 1}")
            for i, each in enumerate(given)
        )
        # The fins as given, canted, to save the set
        self._given = tuple(deepcopy(each) for each in given)
        # Where each fin's root leading edge sits without a deflection, from
        # the point the set is added at, in the body frame
        self._roots = tuple(
            Vector(point_around(0, each.angular_position, each.rocket_radius))
            for each in self.fins
        )
        # The frame of each fin at the deflection it was last flown with, so a
        # deflection held for many steps is not turned again at each one
        self._frames = [None] * n

        controls = tuple(f"fin_{i + 1}" for i in range(n))
        defaults = {
            control: math.radians(each.cant_angle)
            for control, each in zip(controls, given)
        }
        super().__init__(
            reference_area=given[0].reference_area,
            reference_length=given[0].reference_length,
            coefficients={},
            center_of_pressure=(0, 0, 0),
            name=name,
            active_during=active_during,
            active=active,
            controls=controls,
            control_defaults=defaults,
            control_limits=(
                None
                if deflection_limits is None
                else dict.fromkeys(controls, deflection_limits)
            ),
        )
        self.prints = _ControlFinsPrints(self)
        self.plots = _ControlFinsPlots(self)

    @property
    def hinge_position(self):
        """Where the hinge line crosses the root chord, as a fraction of its
        length from the leading edge, from 0 to 1."""
        return self._hinge_position

    @hinge_position.setter
    def hinge_position(self, value):
        self._hinge_position = value
        # The frames kept were turned about the old hinge line, and so were
        # the forces the coefficients kept for the point they were last read at
        self._frames = [None] * self.n
        self._bind_coefficients()
        self._version += 1

    @staticmethod
    def __check_fins(fin, n):
        """The fins of the set as given: ``n`` evenly spaced copies of one fin,
        or the sequence given, with ``n`` checked against it."""
        if isinstance(fin, Fin):
            n = 4 if n is None else n
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError(
                    f"The number of fins must be a whole number, got {n!r}."
                )
            if n < 1:
                raise ValueError(f"The number of fins must be at least 1, got {n}.")
            copies = []
            for i in range(n):
                copy = deepcopy(fin)
                copy.angular_position = (fin.angular_position + 360 * i / n) % 360
                copies.append(copy)
            return tuple(copies)
        if (
            isinstance(fin, (list, tuple))
            and fin
            and all(isinstance(each, Fin) for each in fin)
        ):
            if n is not None and n != len(fin):
                raise ValueError(
                    f"{len(fin)} fins were given, but n={n!r}. Leave n out when "
                    "giving the fins one by one."
                )
            return tuple(fin)
        raise TypeError(
            "ControlFins is built from one individual fin, such as "
            f"TrapezoidalFin(...), or a sequence of them, got {fin!r}."
        )

    @staticmethod
    def _copy_fin(fin, angular_position, name):
        """A copy of ``fin`` at ``angular_position``, uncanted: the deflection
        turns it during the flight."""
        copy = deepcopy(fin)
        copy.name = name
        if copy.cant_angle != 0:
            copy.cant_angle = 0
        if copy.angular_position != angular_position:
            copy.angular_position = angular_position
        return copy

    def _frame(self, index, deflection):
        """The rotations of fin ``index`` turned by ``deflection`` and where its
        center of pressure is, from the point the set is added at, in the body
        frame."""
        frame = self._frames[index]
        if frame is not None and frame[0] == deflection:
            return frame[1:]
        fin = self.fins[index]
        uncanted, body_to_fin = fin._rotations(deflection)
        fin_to_body = body_to_fin.transpose
        offset = (
            self._roots[index]
            + fin._cant_shift(deflection, uncanted.transpose, self.hinge_position)
            + fin_to_body @ fin.force_application_point
        )
        self._frames[index] = (deflection, body_to_fin, fin_to_body, offset)
        return body_to_fin, fin_to_body, offset

    def _fins_in_air(
        self, stream_velocity, stream_speed, stream_mach, omega, deflections
    ):
        """Each fin turned by its deflection, with its frame and the air it
        meets where it sits: ``(fin, deflection, body_to_fin, fin_to_body,
        offset, stream, speed, mach)``. The rotation of the rocket moves each
        fin through the air on its own, as it would an individual fin added to
        the rocket."""
        stream_velocity = Vector(stream_velocity)
        omega = Vector(omega)
        for index, (fin, deflection) in enumerate(zip(self.fins, deflections)):
            body_to_fin, fin_to_body, offset = self._frame(index, deflection)
            stream = stream_velocity - (omega ^ offset)
            speed = abs(stream)
            mach = stream_mach * speed / stream_speed if stream_speed > 0 else 0.0
            yield fin, deflection, body_to_fin, fin_to_body, offset, stream, speed, mach

    def _fin_forces(  # pylint: disable=too-many-arguments
        self, stream_velocity, stream_speed, stream_mach, rho, cp, omega, deflections
    ):
        """Sum the forces and moments of the fins, each turned by its
        deflection and fed the air it meets where it sits."""
        totals = [0.0] * 6
        for (
            fin,
            deflection,
            body_to_fin,
            fin_to_body,
            offset,
            *air,
        ) in self._fins_in_air(
            stream_velocity, stream_speed, stream_mach, omega, deflections
        ):
            forces = fin._forces_in_frame(
                *air, rho, cp + offset, omega, body_to_fin, fin_to_body, deflection
            )
            for k in range(6):
                totals[k] += forces[k]
        return tuple(totals)

    @staticmethod
    def _stream(alpha, beta):
        """The unit air velocity in the body frame for the angles given, in
        radians: from the nose toward the tail (``-z``), turned by ``alpha``
        toward ``-y`` and by ``beta`` toward ``-x``, as
        :meth:`GenericSurface.compute_forces_and_moments` defines the angles."""
        stream = Vector([-math.tan(beta), -math.tan(alpha), -1.0])
        return stream / abs(stream)

    def compute_forces_and_moments(  # pylint: disable=arguments-differ
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        density=None,
        dynamic_viscosity=None,
        z=None,
        controls=None,
    ):
        """Compute the forces and moments of the fins, each turned by its
        deflection. Called at each step of the simulation.

        Parameters
        ----------
        stream_velocity : Vector
            Velocity of the air relative to the point the set is added at, in
            the body frame, in m/s. Flying nose first it points toward ``-z``,
            from the nose to the tail.
        stream_speed : float
            Its magnitude, in m/s.
        stream_mach : float
            Its Mach number.
        rho : float
            Air density, in kg/m³.
        cp : Vector
            The point the set is added at, relative to the rocket's center of
            dry mass, in the body frame, in meters.
        omega : tuple of float
            Angular velocity of the rocket about the body axes, in rad/s.
        density, dynamic_viscosity, z : optional
            Accepted for the same call as the other surfaces. Unused: the fin
            model does not depend on the Reynolds number.
        controls : sequence of float, optional
            The deflection of each fin, in radians, in the order of
            :attr:`controls`. ``None`` (the default) uses the default
            deflections, as outside a flight.

        Returns
        -------
        tuple of float
            The forces ``(R1, R2, R3)`` along the body axes and the moments
            ``(M1, M2, M3)`` about them, taken about the rocket's center of dry
            mass.
        """
        deflections = self._default_control_values() if controls is None else controls
        return self._fin_forces(
            stream_velocity, stream_speed, stream_mach, rho, cp, omega, deflections
        )

    def evaluate_coefficients(self):
        """Write the forces of the fins as the six body-frame coefficients,
        taken about the point the set is added at, as functions of the angles,
        Mach, the reduced rates and the deflection of each fin. They are what
        the static margin, the plots and :meth:`rocketpy.Rocket.to_surface`
        read; the flight computes the same forces directly."""
        variables = (*_FLOW_VARIABLES, *self.controls)
        last = {}

        def coefficients(*args):
            # The six coefficients come from one sum over the fins, so the sum
            # is kept for the next coefficient read at the same point
            if last.get("args") != args:
                last["args"], last["values"] = args, self._coefficients_at(*args)
            return last["values"]

        for index, name in enumerate(("cN", "cY", "cA", "cm", "cn", "cl")):
            setattr(
                self,
                name,
                AeroCoefficient(
                    lambda *args, index=index: coefficients(*args)[index],
                    depends_on=variables,
                    control_variables=self.controls,
                    name=name,
                ),
            )

    def _bind_coefficients(self):
        """Rebuild the coefficients, computed from the fins of the set they
        were built for, so that a deep copy reads its own fins."""
        self.evaluate_coefficients()
        self._evaluate_stability_derivatives()

    def _coefficients_at(  # pylint: disable=too-many-arguments
        self, alpha, beta, mach, pitch_rate, yaw_rate, roll_rate, *deflections
    ):
        """``(cN, cY, cA, cm, cn, cl)`` of the set at unit speed and density,
        about the point the set is added at, in the air :meth:`_stream` gives
        for the angles."""
        # The reduced rates are omega * L / (2 * V), at a speed of 1
        rate_factor = 2 / self.reference_length
        omega = Vector([pitch_rate, yaw_rate, roll_rate]) * rate_factor
        R1, R2, R3, M1, M2, M3 = self._fin_forces(
            self._stream(alpha, beta),
            1.0,
            mach,
            1.0,
            Vector([0, 0, 0]),
            omega,
            deflections,
        )
        force = 0.5 * self.reference_area
        moment = force * self.reference_length
        return (
            -R2 / force,
            R1 / force,
            -R3 / force,
            M1 / moment,
            M2 / moment,
            M3 / moment,
        )

    def deflections(self, roll=0.0, x=0.0, y=0.0):
        """Turn a roll command and two sideways commands into the deflection
        of every fin.

        Each fin gets ``roll + x * cos(phi) + y * sin(phi)``, with ``phi`` its
        angular position, so the three add up. The same ``roll`` on every fin
        gives a roll moment; ``x`` turns the fins so that the set pushes
        toward ``+x``, ``y`` so that it pushes toward ``+y``, in the body frame
        (see the class description). On canards the nose follows
        the push, on tail fins it goes the other way.

        Parameters
        ----------
        roll : float, optional
            Deflection of every fin, in radians. Positive gives a negative
            roll moment, so ``roll`` with the sign of the roll rate opposes
            it. Default 0.
        x : float, optional
            Deflection, in radians, of a fin on the ``+y`` axis that pushes the
            set toward ``+x``; the other fins follow with their cosine.
            Default 0.
        y : float, optional
            The same for a push toward ``+y``, from the fin at 90 degrees of
            angular position. Default 0.

        Returns
        -------
        dict
            The deflection of each fin, in radians, by control name:
            ``{"fin_1": ..., "fin_2": ..., ...}``. Write them all at once
            with ``context.controls.<set>.update(...)``.

        Examples
        --------
        >>> from rocketpy import ControlFins, TrapezoidalFin
        >>> fin = TrapezoidalFin(root_chord=0.1, tip_chord=0.05, span=0.08,
        ...                      rocket_radius=0.06, angular_position=0)
        >>> canards = ControlFins(fin, n=4, name="Canards")
        >>> canards.deflections(x=0.1)
        {'fin_1': 0.1, 'fin_2': 0.0, 'fin_3': -0.1, 'fin_4': 0.0}
        """
        # Rounded so cos(90 deg) gives 0 rather than 6e-17; + 0.0 turns -0.0 to 0.0
        return {
            control: round(
                roll
                + x * math.cos(math.radians(phi))
                + y * math.sin(math.radians(phi)),
                12,
            )
            + 0.0
            for control, phi in zip(self.controls, self.angular_positions)
        }

    def effectiveness(self, alpha=0.0, beta=0.0):
        """Give how much roll moment and sideways force the fins produce per
        radian of each command of :meth:`deflections`, as functions of Mach.

        The numbers are coefficients: multiply by the dynamic pressure and the
        set's reference area (and its reference length for the roll moment) to
        get a moment in N m or a force in N per radian of deflection. They are
        what sets the gain of a roll or attitude controller. The sideways
        forces act at the fins, so their moment about the center of mass is
        the force times the distance from the fins to it.

        The air comes head on, from the nose toward the tail (along ``-z`` in
        the body frame), turned by the two angles below. The rocket is not
        rotating. :meth:`rocketpy.ControlFins.info` prints these numbers at
        one Mach number and ``plots.effectiveness()`` draws them.

        Parameters
        ----------
        alpha, beta : float, optional
            The angle of attack and the sideslip angle, in radians, the
            effectiveness is taken at. A positive ``alpha`` means the rocket
            moves toward ``+y`` through the air, so the air flows across it
            toward ``-y``; a positive ``beta`` the same toward ``+x`` and
            ``-x``. Default 0, head on.

        Returns
        -------
        dict of Function
            ``"roll"``: the roll moment coefficient per radian of the same
            deflection on every fin, negative since a positive deflection
            rolls the rocket toward negative ``w3``. ``"x"``: the force
            coefficient toward ``+x`` per radian of the ``x`` command, and
            ``"y"`` toward ``+y`` per radian of the ``y`` command, both
            positive. Each is a :class:`rocketpy.Function` of the Mach number.

        Examples
        --------
        >>> from rocketpy import ControlFins, TrapezoidalFin
        >>> fin = TrapezoidalFin(root_chord=0.1, tip_chord=0.05, span=0.08,
        ...                      rocket_radius=0.06, angular_position=0)
        >>> canards = ControlFins(fin, n=4, name="Canards")
        >>> roll_per_rad = canards.effectiveness()["roll"]
        >>> bool(roll_per_rad(0.3) < 0)
        True
        """
        stream = self._stream(alpha, beta)
        still = Vector([0, 0, 0])
        force = 0.5 * self.reference_area
        moment = force * self.reference_length
        step = 0.01  # rad, small enough for the slope, large enough for the sum

        def forces(mach, deflections):
            # At unit speed and density, about the point the set is added at
            return self._fin_forces(stream, 1.0, mach, 1.0, still, still, deflections)

        def slope(command, index, scale):
            up = list(self.deflections(**{command: step}).values())
            down = list(self.deflections(**{command: -step}).values())
            return lambda mach: (
                (forces(mach, up)[index] - forces(mach, down)[index])
                / (2 * step)
                / scale
            )

        return {
            "roll": Function(
                slope("roll", 5, moment),
                "Mach Number",
                "Roll Moment Coefficient per rad",
            ),
            "x": Function(
                slope("x", 0, force),
                "Mach Number",
                "Force Coefficient toward +x per rad",
            ),
            "y": Function(
                slope("y", 1, force),
                "Mach Number",
                "Force Coefficient toward +y per rad",
            ),
        }

    def hinge_moments(self, dynamic_pressure, deflections=None, alpha=0.0, beta=0.0):
        """Give the torque the air puts on each fin about its hinge line, in
        N m, as a function of Mach: the load the mechanism turning the fin
        has to hold.

        Each fin's torque is its normal force times the distance along the
        chord from the hinge line to the fin's center of pressure. The center
        of pressure is the one point the fin model gives, the quarter chord of
        the mean chord at low speed. The real one moves back at transonic and
        supersonic speeds and with the angle of attack, so when the hinge is
        placed close to it, as it usually is to keep this torque small, the
        size is right but the sign is not to be trusted. Size the mechanism
        for the largest torque over the deflections and speeds it will see.

        ``alpha`` and ``beta`` set the direction the air comes from: both zero
        means straight along the rocket, from the nose toward the tail
        (``-z``). The rocket is not rotating.

        Parameters
        ----------
        dynamic_pressure : float
            Half the air density times the airspeed squared, in Pa, the
            torques are given at.
        deflections : float or dict, optional
            The deflection of the fins, in radians: one number for every fin,
            or one per control input as :meth:`deflections` returns. A fin
            left out keeps its default deflection. Default ``None``, every fin
            at its default.
        alpha, beta : float, optional
            The angle of attack and the sideslip angle, in radians. A
            positive ``alpha`` means the rocket moves toward ``+y`` through
            the air, so the air flows across it toward ``-y``; a positive
            ``beta`` the same toward ``+x`` and ``-x``. Default 0, head on.

        Returns
        -------
        dict of Function
            One :class:`rocketpy.Function` of the Mach number per control
            input, keyed by its name, giving the torque in N m. Positive
            pushes the fin toward a larger deflection, negative toward a
            smaller one, so a fin with its hinge ahead of the center of
            pressure gets a negative torque at a positive deflection.

        Examples
        --------
        >>> from rocketpy import ControlFins, TrapezoidalFin
        >>> fin = TrapezoidalFin(root_chord=0.1, tip_chord=0.05, span=0.08,
        ...                      rocket_radius=0.06, angular_position=0)
        >>> canards = ControlFins(fin, n=4, hinge_position=0.25, name="Canards")
        >>> torque = canards.hinge_moments(20e3, deflections=0.1)["fin_1"]
        >>> bool(torque(0.3) < 0)
        True
        """
        stream = self._stream(alpha, beta)
        still = Vector([0, 0, 0])
        if deflections is None:
            deflections = {}
        elif not isinstance(deflections, dict):
            deflections = dict.fromkeys(self.controls, float(deflections))
        values = [
            deflections.get(control, self.control_defaults[control])
            for control in self.controls
        ]
        # Unit speed and a density of twice the dynamic pressure give
        # 0.5 * rho * speed**2 = dynamic_pressure
        rho = 2 * dynamic_pressure

        def torque(index):
            # Function reads the number of inputs off the callable, so the
            # index is bound here rather than as a default argument
            return lambda mach: self._hinge_moments_at(
                stream, 1.0, mach, rho, still, values
            )[index]

        return {
            control: Function(
                torque(index), "Mach Number", f"Hinge Moment of {control} (N m)"
            )
            for index, control in enumerate(self.controls)
        }

    def _hinge_moments_at(
        self, stream_velocity, stream_speed, stream_mach, rho, omega, deflections
    ):
        """The torque on each fin about its hinge line, in N m, in the air
        given, fed to each fin as :meth:`_fin_forces` feeds it: the fin's
        normal force times the chordwise arm from the hinge line to its
        center of pressure, positive toward a larger deflection."""
        return [
            fin._normal_force(*air, rho, body_to_fin)
            * (fin.cpz - self.hinge_position * fin.root_chord)
            for fin, _, body_to_fin, _, _, *air in self._fins_in_air(
                stream_velocity, stream_speed, stream_mach, omega, deflections
            )
        ]

    def _hinge_moments_in(self, flight):
        """The torque on each fin about its hinge line over ``flight``, in
        N m: one Function of time per control input, from the air the set met
        at each stored row of the flight and the deflections then in force.

        Private for now: where post-flight analyses of a part live is still
        open, see ``docs/development/post_events_issues/issue_flight_analysis.md``.
        ``plots.hinge_moments(flight)`` draws it.

        Raises
        ------
        ValueError
            If the set is not on the flight's rocket.
        """
        on_rocket = self._on_rocket_of(flight)
        cp = flight.rocket.surfaces_cp_to_cdm[on_rocket]
        history = flight.controls[on_rocket.name]
        times = flight.time
        # The air at the rocket's center of dry mass, moved to the set by the
        # rotation; the wind at the set's own altitude is not told apart
        streams = np.atleast_2d(flight.stream_velocity_body_frame)
        rows = []
        for t, stream in zip(times, streams):
            if not on_rocket.is_active(t, flight):
                rows.append([0.0] * on_rocket.n)
                continue
            omega = Vector(
                [
                    flight.w1.get_value_opt(t),
                    flight.w2.get_value_opt(t),
                    flight.w3.get_value_opt(t),
                ]
            )
            stream = Vector(stream) - (omega ^ cp)
            speed = abs(stream)
            mach = speed / flight.speed_of_sound.get_value_opt(t)
            rho = flight.density.get_value_opt(t)
            deflections = [
                history[control].get_value_opt(t) for control in on_rocket.controls
            ]
            rows.append(
                on_rocket._hinge_moments_at(
                    stream, speed, mach, rho, omega, deflections
                )
            )
        rows = np.array(rows)
        return {
            control: Function(
                np.column_stack((times, rows[:, index])),
                "Time (s)",
                f"Hinge Moment of {on_rocket.name} {control} (N m)",
                "linear",
                "zero",
            )
            for index, control in enumerate(on_rocket.controls)
        }

    def info(self):
        """Print which fin is which (its control input, angular position and
        default deflection), the effectiveness, then the geometry and the
        coefficients.

        Returns
        -------
        None
        """
        self.prints.fins()
        self.prints.effectiveness()
        self.prints.geometry()
        self.prints.coefficients()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "fin": self.fin if self.fin is not None else list(self._given),
            "n": self.n,
            "name": self.name,
            "deflection_limits": self.deflection_limits,
            "hinge_position": self.hinge_position,
            "active_during": self.active_during,
            "active": self.active,
            # As they are now, if changed since the set was built
            **self._control_settings(),
        }

    @classmethod
    def from_dict(cls, data):
        limits = data.get("deflection_limits")
        fins = cls(
            data["fin"],
            n=data.get("n"),
            name=data.get("name", "ControlFins"),
            deflection_limits=None if limits is None else tuple(limits),
            hinge_position=data.get("hinge_position", 0.5),
            active_during=data.get("active_during", "always"),
            active=data.get("active", True),
        )
        if "control_defaults" in data:
            fins._set_controls(
                fins.controls, data["control_defaults"], data["control_limits"]
            )
        return fins
