"""Frame-labelled vectors used by physical interaction callbacks."""

from __future__ import annotations

import numpy as np

from .reference_frame import ReferenceFrame


class FrameVector(np.ndarray):
    """A three-component NumPy vector with an explicit coordinate frame.

    The class is an ``ndarray`` subclass, so existing callbacks may index it,
    iterate over it, or pass it directly to NumPy. The additional ``frame``
    attribute identifies the basis in which its components are expressed.

    Relative velocity is unchanged by a common translational boost, but its
    components still rotate when changing between terrestrial and celestial
    axes. Labelling the basis is therefore necessary for projected-area models
    that combine the vector with vehicle attitude.
    """

    def __new__(cls, components, frame):
        values = np.asarray(components, dtype=float)
        if values.shape != (3,):
            raise ValueError("FrameVector must contain exactly three components.")
        instance = np.array(values, dtype=float, copy=True).view(cls)
        instance.frame = ReferenceFrame.coerce(frame)
        instance.setflags(write=False)
        return instance

    def __array_finalize__(self, source):
        self.frame = getattr(source, "frame", None)

    def to_frame(self, epoch, datum, target):
        """Return this free vector resolved in another datum-owned frame."""
        target = ReferenceFrame.coerce(target)
        if target == self.frame:
            return FrameVector(self, self.frame)
        transformed = datum.transform_vector(
            epoch,
            self,
            source=self.frame,
            target=target,
        )
        return FrameVector(transformed, target)
