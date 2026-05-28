"""Stage – a mission item representing a rocket stage."""

from rocketpy.mission.attached_item import AttachedItem
from rocketpy.mission.parent_update import NoOpParentUpdate
from rocketpy.mission.separation_model import InstantaneousSeparation
from rocketpy.mission.stage_state import StageState


class Stage(AttachedItem):
    """A rocket stage that can ignite and/or separate from its parent.

    A :class:`Stage` is the mission-layer wrapper around the physical
    body of a rocket stage.  It tracks the stage lifecycle via
    :class:`~rocketpy.mission.StageState` and holds references to the
    events that govern separation and ignition.

    Separation and ignition are kept as independent concerns: a stage may
    separate without igniting (e.g. a fairing) or may ignite before
    separating (hot staging).

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"second_stage"``).
    body : :class:`~rocketpy.body.BodyLike`
        Physical body of the stage.
    attachment : :class:`~rocketpy.mission.Attachment`
        Mechanical link to the parent body.
    separation_event : :class:`~rocketpy.mission.StageSeparationEvent`, optional
        Event that triggers mechanical separation.  ``None`` if the stage
        never separates during the simulated mission.
    ignition_event : :class:`~rocketpy.mission.IgnitionEvent`, optional
        Event that triggers motor ignition.  ``None`` if the stage is
        already burning when the simulation starts (i.e. it is the
        first stage).
    separation : :class:`~rocketpy.mission.SeparationModel`, optional
        Separation dynamics model.  Defaults to
        :class:`~rocketpy.mission.InstantaneousSeparation`.
    parent_update : :class:`~rocketpy.mission.ParentUpdate`, optional
        Updates applied to the parent body after separation.  Defaults
        to :class:`~rocketpy.mission.NoOpParentUpdate`.
    events : list[:class:`~rocketpy.mission.Event`], optional
        Additional events besides *separation_event* and *ignition_event*.

    Attributes
    ----------
    separation_event : StageSeparationEvent or None
    ignition_event : IgnitionEvent or None
    separation : SeparationModel
    parent_update : ParentUpdate
    state : StageState
    """

    def __init__(
        self,
        name,
        body,
        attachment,
        separation_event=None,
        ignition_event=None,
        separation=None,
        parent_update=None,
        events=None,
    ):