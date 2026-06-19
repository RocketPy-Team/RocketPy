"""Interpolation and extrapolation submodule for the Function class.

This package provides factory functions that build strategy objects for
interpolation and extrapolation.

The registry helper :func:`build_interpolation_evaluator` is the main
entry point consumed by :class:`rocketpy.mathutils.function.Function`.
"""

from .registry import build_interpolation_evaluator

__all__ = ["build_interpolation_evaluator"]
