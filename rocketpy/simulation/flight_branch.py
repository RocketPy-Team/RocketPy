"""FlightBranch – a single contiguous integration arc in a multistage flight."""


class FlightBranch:
    """Represents one contiguous integration arc in a branching flight tree.
    When a :class:`~rocketpy.mission.Stage` separates or a
    :class:`~rocketpy.mission.Deployable` is released, the simulation engine
    creates one or more new :class:`FlightBranch` objects.  Each branch tracks
    its own body, initial state, and list of pending events.
    The branches together form a directed acyclic graph (DAG) rooted at the
    first-stage branch.  A :class:`~rocketpy.simulation.Flight` object holds
    the root branch and drives the full simulation.
    Parameters
    ----------
    body : :class:`~rocketpy.body.BodyLike`
        The flight-ready body for this branch.  A simulation-safe deep copy
        should be provided (see
        :meth:`~rocketpy.body.BodyLike.to_branch_ready_copy`).
    start_state : object
        Initial state vector (position, velocity, attitude, …) at the
        start of this branch.
    start_time : float
        Simulation time, in seconds, at which this branch begins.
    parent : :class:`FlightBranch` or None, optional
        The branch from which this one was spawned.  ``None`` for the root.
    events : list[:class:`~rocketpy.mission.Event`], optional
        Events that are pending on this branch.
    Attributes
    ----------
    body : BodyLike
    parent : FlightBranch or None
    children : list[FlightBranch]
    start_state : object
    start_time : float
    events : list[Event]
    """

    def __init__(
        self,
        body,
        start_state=None,
        start_time: float = 0.0,
        parent=None,
        events=None,
    ):