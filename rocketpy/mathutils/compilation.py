"""Optional eager Numba compilation for numerical RocketPy kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

try:  # pragma: no cover - availability depends on the installation extra
    from numba import jit as _numba_jit
except (ImportError, OSError):  # pragma: no cover - installation dependent
    _numba_jit = None


F = TypeVar("F", bound=Callable[..., Any])
NUMBA_AVAILABLE = _numba_jit is not None


def numbify(
    func: F | None = None,
    *,
    signature: str | None = None,
    **jit_options: Any,
):
    """Compile a numerical function with Numba when it is installed.

    A string ``signature`` is passed as the first argument to ``numba.jit``.
    Numba therefore compiles the specialization eagerly while the containing
    module is imported. Without Numba, or if a particular kernel cannot be
    compiled, the original Python function is returned unchanged.

    The decorator supports both ``@numbify`` and configured use such as
    ``@numbify(signature="float64(float64)", nopython=True, cache=True)``.
    """
    if signature is not None and not isinstance(signature, str):
        raise TypeError("numbify signature must be a Numba signature string.")

    def decorate(function: F):
        if _numba_jit is None:
            return _mark_python_fallback(function, signature)
        try:
            if signature is None:
                compiled = _numba_jit(**jit_options)(function)
            else:
                compiled = _numba_jit(signature, **jit_options)(function)
        except Exception as error:  # pragma: no cover - version/kernel dependent
            return _mark_python_fallback(function, signature, error)
        compiled.__numbified__ = True
        compiled.__numba_signature__ = signature
        return compiled

    return decorate(func) if func is not None else decorate


def _mark_python_fallback(function, signature, error=None):
    function.__numbified__ = False
    function.__numba_signature__ = signature
    if error is not None:
        function.__numbify_error__ = error
    return function
