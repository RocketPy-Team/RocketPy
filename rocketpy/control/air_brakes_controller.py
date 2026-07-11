from ..plots.controller_plots import _AirBrakesControllerPlots
from ..prints.controller_prints import _AirBrakesControllerPrints
from .surface_controller import SurfaceController


class AirBrakesController(SurfaceController):
    """A :class:`SurfaceController` specialized for a single
    :class:`rocketpy.AirBrakes` surface.

    The air brakes are exposed to the controller function as
    ``kwargs["air_brakes"]``; the controller applies control by setting
    ``air_brakes.deployment_level``. The recorded deployment history is
    available post-flight through :attr:`deployment_level_history`.

    See Also
    --------
    rocketpy.Rocket.add_air_brakes : Builds the air brakes and this
        controller in one call.
    """

    def __init__(
        self,
        controller_function,
        air_brakes,
        sampling_rate,
        context=None,
        name="AirBrakes Controller",
        enabled=True,
        disable_on=None,
        enable_on=None,
        needs=None,
    ):
        """Initialize the air brakes controller.

        Parameters
        ----------
        controller_function : callable
            Control logic with signature ``controller_function(**kwargs)``;
            set ``kwargs["air_brakes"].deployment_level`` to apply the
            control action. See :class:`Controller` for the full list of
            available keyword arguments.
        air_brakes : AirBrakes
            The air brakes surface this controller drives.
        sampling_rate : float
            Rate in hertz at which the controller executes.
        context : dict, optional
            Initial persistent state; see :class:`Controller`.
        name : str, optional
            Controller name. Defaults to ``"AirBrakes Controller"``.
        enabled : bool, optional
            Initial enabled state. Defaults to ``True``.
        disable_on, enable_on : str or int or float or callable, optional
            Automatic disable/enable conditions; see :class:`Controller`.
        needs : list or frozenset of str or None, optional
            Expensive simulation values the controller function accesses;
            see :class:`Controller`.
        """
        super().__init__(
            controller_function,
            air_brakes,
            sampling_rate,
            context=context,
            name=name,
            controlled_objects_name="air_brakes",
            enabled=enabled,
            disable_on=disable_on,
            enable_on=enable_on,
            needs=needs,
        )
        self.prints = _AirBrakesControllerPrints(self)
        self.plots = _AirBrakesControllerPlots(self)

    @classmethod
    def _construct(
        cls, controller_function, controlled_objects, sampling_rate, **kwargs
    ):
        # __init__ pins controlled_objects_name to "air_brakes" itself.
        kwargs.pop("controlled_objects_name", None)
        return cls(controller_function, controlled_objects, sampling_rate, **kwargs)

    @property
    def air_brakes(self):
        """The controlled AirBrakes surface."""
        return self._controlled_objects_list[0]

    @property
    def deployment_level_history(self):
        """Recorded deployment level as a ``Function`` of time (linear
        interpolation, constant extrapolation). Raises if no history has been
        recorded yet."""
        schedule = self.recorded_schedule
        try:
            return schedule["air_brakes"]["deployment_level"]
        except KeyError as exc:
            raise ValueError(
                "No recorded deployment level history - run a Flight with "
                "this controller first."
            ) from exc
