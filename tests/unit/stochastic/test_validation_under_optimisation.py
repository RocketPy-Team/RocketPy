"""Validation in `stochastic/` has to survive `python -O`.

`assert` is removed by the optimiser, so every check written that way stops
running under `-O` and malformed user input reaches the model instead. The
module carried a TODO asking for this; these tests are what keeps it done.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

MODULES = [
    "stochastic_model",
    "stochastic_environment",
    "stochastic_flight",
    "stochastic_aero_surfaces",
    "stochastic_parachute",
]


@pytest.mark.parametrize("module", MODULES)
def test_no_production_asserts_remain(module):
    """The mechanism. A single `assert` reintroduced here is a check that stops
    existing under `-O`, which is exactly what this file exists to prevent."""
    source = (REPO / "rocketpy" / "stochastic" / f"{module}.py").read_text()
    offenders = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(source.splitlines(), 1)
        if line.strip().startswith("assert ")
    ]

    assert not offenders, f"{module}.py still asserts: {offenders}"


REFUSALS = [
    ("a tuple element of the wrong type", "mass=('not a number', 0.5)"),
    ("a tuple of the wrong length", "mass=(1.0, 0.5, 'normal', 'extra')"),
]


@pytest.mark.parametrize("label, kwargs", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_malformed_input_is_refused_under_optimisation(label, kwargs):
    """The behaviour, in a child interpreter under -O.

    Run in-process this passes either way, because the assert is still compiled
    in. The optimiser only strips it at compile time, so the check has to be a
    separate interpreter to mean anything.
    """
    program = (
        "from types import SimpleNamespace;"
        "from rocketpy.stochastic.stochastic_model import StochasticModel;"
        "obj = SimpleNamespace(mass=1.0)\n"
        "try:\n"
        f"    StochasticModel(obj, {kwargs})\n"
        "    print('accepted')\n"
        "except AssertionError:\n"
        "    print('refused')\n"
    )
    done = subprocess.run(
        [sys.executable, "-O", "-c", program],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )

    assert "refused" in done.stdout, f"{label}: {done.stdout!r} {done.stderr!r}"
