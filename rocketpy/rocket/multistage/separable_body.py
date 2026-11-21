"""The body a separation event turns loose from the vehicle."""


class SeparableBody:
    """A body that starts attached to the vehicle and becomes a free body
    when its separation event fires. Base class for Stage and Deployable.

    Parameters
    ----------
    name : str
        Unique body name within the vehicle; used to group Mission
        results (e.g. ``mission.flights["booster"]``).
    separation_delta_v : float, optional
        Relative separation speed along the stack's longitudinal axis,
        in m/s, split by momentum conservation. Default 0.

    Notes
    -----
    Subclasses each hold their own release-event attribute under its own
    name (``Stage.separation``, ``Deployable.ejection``) rather than a
    shared base attribute, since the two events are conceptually similar
    but not interchangeable.
    """

    def __init__(self, name, separation_delta_v=0.0):
        self.name = name
        self.separation_delta_v = separation_delta_v
