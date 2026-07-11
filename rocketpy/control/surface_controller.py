from .controller import Controller


class SurfaceController(Controller):
    """A :class:`Controller` that drives one or more
    :class:`rocketpy.ControllableGenericSurface` objects.

    The controller function applies control actions through the surfaces'
    ``set_control`` (or convenience attributes such as
    ``AirBrakes.deployment_level``); the base class records each surface's
    control state after every execution and restores it between flights.

    See Also
    --------
    Controller : Base class holding the tracking/reset machinery.
    AirBrakesController : Specialization for air brakes.
    """

    def __init__(self, controller_function, surfaces, sampling_rate, **kwargs):
        """Initialize the surface controller.

        Parameters
        ----------
        controller_function : callable
            Control logic with signature ``controller_function(**kwargs)``;
            see :class:`Controller` for the available keyword arguments.
        surfaces : ControllableGenericSurface or list
            Controllable surface(s) this controller drives.
        sampling_rate : float
            Rate in hertz at which the controller executes.
        **kwargs : dict
            Remaining :class:`Controller` options (``context``, ``name``,
            ``controlled_objects_name``, ``enabled``, ``disable_on``,
            ``enable_on``, ``needs``).
        """
        self._validate_surfaces(surfaces)
        super().__init__(controller_function, surfaces, sampling_rate, **kwargs)

    @staticmethod
    def _validate_surfaces(surfaces):
        """Require every controlled object to be a ControllableGenericSurface."""
        # Imported here to avoid a circular import with the rocket package.
        from rocketpy.rocket.aero_surface.controllable_generic_surface import (  # pylint: disable=import-outside-toplevel
            ControllableGenericSurface,
        )

        candidates = surfaces if isinstance(surfaces, (list, tuple)) else [surfaces]
        for surface in candidates:
            if not isinstance(surface, ControllableGenericSurface):
                raise TypeError(
                    f"SurfaceController can only control ControllableGenericSurface "
                    f"objects, but got {type(surface).__name__} "
                    f"('{getattr(surface, 'name', surface)}'). Use the base "
                    "Controller class for other controlled objects."
                )

    @property
    def surfaces(self):
        """List of the controlled surfaces."""
        return self._controlled_objects_list
