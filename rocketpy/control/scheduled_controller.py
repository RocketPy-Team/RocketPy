from rocketpy.mathutils.function import Function

from .controller import Controller


class ScheduledController(Controller):
    """A :class:`Controller` that replays a pre-defined control schedule
    open-loop.

    At every execution the scheduled value of each control variable is looked
    up at the current simulation time and applied to the controlled objects
    through ``set_control`` (so clamping still applies). Its main use is
    reconstructing/replaying a recorded flight: pass a controller's
    ``recorded_schedule`` (or build one with :meth:`from_controller`) and the
    flight is re-flown with the same control inputs without the original
    controller function.

    Attributes
    ----------
    ScheduledController.schedule : dict
        ``{object_name: {variable: Function}}`` giving each control variable
        as a function of time.
    """

    def __init__(
        self,
        schedule,
        controlled_objects,
        sampling_rate,
        context=None,
        name="Scheduled Controller",
        controlled_objects_name=None,
        enabled=True,
        disable_on=None,
        enable_on=None,
    ):
        """Initialize the scheduled controller.

        Parameters
        ----------
        schedule : dict
            Control values over time. Either nested,
            ``{object_name: {variable: source}}`` (object names as tracked by
            the controller, see ``Controller.control_history``), or flat,
            ``{variable: source}``, when there is a single controlled object.
            Each source may be a :class:`Function` of time, a callable, a
            constant, or a list/array of ``(time, value)`` points
            (interpolated linearly, with constant extrapolation).
        controlled_objects : object or list of object
            Object(s) the schedule is applied to; see :class:`Controller`.
        sampling_rate : float
            Rate in hertz at which the schedule is applied.
        context, name, controlled_objects_name, enabled, disable_on,
        enable_on :
            Same as in :class:`Controller`.
        """
        super().__init__(
            self._apply_schedule,
            controlled_objects,
            sampling_rate,
            context=context,
            name=name,
            controlled_objects_name=controlled_objects_name,
            enabled=enabled,
            disable_on=disable_on,
            enable_on=enable_on,
        )
        self.schedule = self._normalize_schedule(schedule)

    def _normalize_schedule(self, schedule):
        """Normalize the schedule to ``{object_name: {variable: Function}}``,
        resolving the flat single-object form against the tracked objects."""
        if not isinstance(schedule, dict):
            raise TypeError(
                "schedule must be a dict of {object_name: {variable: source}} "
                "or {variable: source} for a single controlled object."
            )
        is_nested = all(isinstance(value, dict) for value in schedule.values())
        if not is_nested:
            if len(self._tracked) != 1:
                raise ValueError(
                    "A flat {variable: source} schedule requires exactly one "
                    "controlled object with a control state; got "
                    f"{len(self._tracked)}. Use the nested "
                    "{object_name: {variable: source}} form instead."
                )
            schedule = {self._tracked[0][0]: schedule}
        return {
            object_name: {
                variable: self._as_function(source, variable)
                for variable, source in variables.items()
            }
            for object_name, variables in schedule.items()
        }

    @staticmethod
    def _as_function(source, variable):
        if isinstance(source, Function):
            return source
        return Function(
            source,
            inputs="Time (s)",
            outputs=variable,
            interpolation="linear",
            extrapolation="constant",
        )

    def _apply_schedule(self, **kwargs):
        """Apply the scheduled control values at the current time."""
        time = kwargs["time"]
        for object_name, obj in self._tracked:
            for variable, function in self.schedule.get(object_name, {}).items():
                obj.set_control(variable, function.get_value_opt(time))

    @classmethod
    def from_controller(
        cls, controller, controlled_objects=None, sampling_rate=None, name=None
    ):
        """Build an open-loop replay of another controller's recorded
        schedule.

        Parameters
        ----------
        controller : Controller
            Controller whose ``recorded_schedule`` (from a previous flight)
            is replayed.
        controlled_objects : object or list, optional
            Objects to drive; defaults to the source controller's.
        sampling_rate : float, optional
            Defaults to the source controller's sampling rate.
        name : str, optional
            Defaults to ``"<controller name> (replay)"``.

        Returns
        -------
        ScheduledController
        """
        schedule = controller.recorded_schedule
        if not schedule:
            raise ValueError(
                f"Controller '{controller.name}' has no recorded control "
                "history to replay - run a Flight with it first."
            )
        return cls(
            schedule=schedule,
            controlled_objects=(
                controlled_objects
                if controlled_objects is not None
                else controller.controlled_objects
            ),
            sampling_rate=(
                sampling_rate if sampling_rate is not None else controller.sampling_rate
            ),
            controlled_objects_name=controller.controlled_objects_name,
            name=name or f"{controller.name} (replay)",
        )

    def to_dict(self, **kwargs):
        """Serialize the scheduled controller; the schedule itself is stored
        (as Functions) instead of a pickled controller function."""
        data = super().to_dict(**kwargs)
        data["controller_function"] = None
        data["schedule"] = self.schedule
        return data

    @classmethod
    def from_dict(cls, data, controlled_objects=None):
        pending_name = None
        controlled_objects_name = data.get("controlled_objects_name")
        if controlled_objects is None:
            controlled_objects = []
            pending_name = controlled_objects_name
            controlled_objects_name = None

        controller = cls(
            schedule=data["schedule"],
            controlled_objects=controlled_objects,
            sampling_rate=data.get("sampling_rate"),
            context=data.get("context", {}),
            name=data.get("name", "Scheduled Controller"),
            controlled_objects_name=controlled_objects_name,
            enabled=data.get("enabled", True),
        )
        if pending_name is not None:
            controller.controlled_objects_name = pending_name
        controller._controlled_objects_ref = data.get("controlled_objects_ref", [])
        return controller
