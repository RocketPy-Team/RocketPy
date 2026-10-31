"""Deterministic geometry and coefficient tables for DATCOM regression tests."""

import numpy as np
import pytest

from rocketpy.rocket.datcom import BODY_KEYS, DAMPING_KEYS, DatcomResult


@pytest.fixture
def datcom_table_factory():
    """Make complete signed tables without requiring an external executable."""

    def make(values=None):
        values = values or {}
        mach = np.array([0.2, 0.6])
        alpha = np.array([-2.0, 0.0, 2.0])
        zeros = np.zeros((2, 3))
        body = {key: zeros + values.get(key, 0) for key in BODY_KEYS}
        derivatives = {
            key: zeros + values.get(key, 0)
            for key in ("CNA", "CMA", "CYB", "CLNB", "CLLB")
        }
        damping = {key: zeros + values.get(key, 0) for key in DAMPING_KEYS}
        return DatcomResult(
            (
                zeros.copy(),
                zeros + values.get("CD", 0.4),
                body["CM"],
                zeros.copy(),
                mach,
                alpha,
            ),
            body,
            derivatives,
            damping,
        )

    return make


@pytest.fixture
def datcom_rocket(calisto, tmp_path):
    """An assembled rocket with a fake executable, nose, DATCOM fins and tail."""
    executable = tmp_path / "datcom"
    executable.write_text("not executed by unit tests")
    calisto.add_nose(0.55, "von karman", position=1.2)
    calisto.add_datcom_hex_fins(
        4,
        root_chord=0.25,
        tip_chord=0.1,
        span=0.1,
        position=-0.9,
        sweep_length=0.1,
        path=executable,
        datcom_options={"mach": [0.2, 0.6], "alpha": [-2, 0, 2]},
    )
    calisto.add_tail(calisto.radius, 0.045, 0.1, position=-1.2)
    return calisto
