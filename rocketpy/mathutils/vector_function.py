"""Vector-valued views composed from RocketPy scalar Functions."""

from __future__ import annotations

import numpy as np

from .function import Function


class VectorFunction:
    """A sampled three-component function of one scalar input.

    RocketPy's scalar :class:`Function` remains the implementation for each
    component, while this view provides vector calls and a shared ``(n, 4)``
    history containing input, x, y, and z columns.
    """

    def __init__(self, source, *, inputs="Time (s)", outputs=None):
        source = np.asarray(source, dtype=float)
        if source.ndim != 2 or source.shape[1] != 4:
            raise ValueError("VectorFunction source must have shape (n, 4).")
        self.source = source.copy()
        labels = outputs or ("X", "Y", "Z")
        if len(labels) != 3:
            raise ValueError("VectorFunction requires three output labels.")
        self.components = tuple(
            Function(
                self.source[:, [0, index]],
                inputs=inputs,
                outputs=label,
                interpolation="linear",
                extrapolation="constant",
            )
            for index, label in enumerate(labels, start=1)
        )

    def __call__(self, value):
        """Evaluate all three components at one or more input values."""
        evaluated = np.asarray([component(value) for component in self.components])
        return evaluated if evaluated.ndim == 1 else np.moveaxis(evaluated, 0, -1)

    def __getitem__(self, item):
        return self.source[item]

    def __len__(self):
        return len(self.source)

    @property
    def x(self):
        return self.components[0]

    @property
    def y(self):
        return self.components[1]

    @property
    def z(self):
        return self.components[2]

    @property
    def magnitude(self):
        """Magnitude as a scalar RocketPy Function."""
        return Function(
            np.column_stack(
                (self.source[:, 0], np.linalg.norm(self.source[:, 1:4], axis=1))
            ),
            inputs="Time (s)",
            outputs="Magnitude",
            interpolation="linear",
            extrapolation="constant",
        )
