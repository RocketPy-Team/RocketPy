"""Deployable – a mission item that can be released during flight."""

from rocketpy.mission.attached_item import AttachedItem
from rocketpy.mission.parent_update import NoOpParentUpdate
from rocketpy.mission.separation_model import InstantaneousSeparation


class Deployable(AttachedItem):
    """A body that can be deployed (released) from its parent during flight.
    A :class:`Deployable` is the mission-layer wrapper around a physical
    body that should be released at some point in flight (e.g. a nose
    cone, a payload bay, or a secondary vehicle).  It is *not* the physical
    body itself – that is held in :attr:`body`.
    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"nose_cone_payload"``).
    body : :class:`~rocketpy.body.BodyLike`
        The physical body of the deployable.
    attachment : :class:`~rocketpy.mission.Attachment`
        Mechanical link to the parent body.
    deployment_event : :class:`~rocketpy.mission.DeploymentEvent`
        The event that triggers this deployable's release.
    separation : :class:`~rocketpy.mission.SeparationModel`, optional
        Model for the separation dynamics.  Defaults to
        :class:`~rocketpy.mission.InstantaneousSeparation`.
    parent_update : :class:`~rocketpy.mission.ParentUpdate`, optional
        Callback that modifies the parent body after separation.  Defaults
        to :class:`~rocketpy.mission.NoOpParentUpdate`.
    events : list[:class:`~rocketpy.mission.Event`], optional
        Additional events besides *deployment_event*.
    Attributes
    ----------
    deployment_event : DeploymentEvent
    separation : SeparationModel
    parent_update : ParentUpdate
    """

    def __init__(
        self,
        name,
        body,
        attachment,
        deployment_event,
        separation=None,
        parent_update=None,
        events=None,
    ):