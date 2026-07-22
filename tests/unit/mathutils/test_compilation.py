import pytest

from rocketpy.mathutils import compilation


def test_numbify_passes_string_signature_for_eager_compilation(monkeypatch):
    """A configured decorator passes its signature directly to Numba JIT."""
    # Arrange
    calls = []

    def fake_jit(signature, **options):
        calls.append((signature, options))
        return lambda function: function

    monkeypatch.setattr(compilation, "_numba_jit", fake_jit)

    # Act
    @compilation.numbify(signature="float64(float64)", nopython=True, cache=True)
    def square(value):
        return value * value

    # Assert
    assert calls == [("float64(float64)", {"nopython": True, "cache": True})]
    assert square(3.0) == 9.0
    assert square.__numbified__ is True
    assert square.__numba_signature__ == "float64(float64)"


def test_numbify_returns_python_function_without_numba(monkeypatch):
    """The optional compiler has a behaviorally identical Python fallback."""
    # Arrange
    monkeypatch.setattr(compilation, "_numba_jit", None)

    # Act
    @compilation.numbify(signature="float64(float64)", nopython=True)
    def increment(value):
        return value + 1.0

    # Assert
    assert increment(2.0) == 3.0
    assert increment.__numbified__ is False
    assert increment.__numba_signature__ == "float64(float64)"


def test_numbify_rejects_non_string_eager_signature():
    """The eager compilation contract rejects ambiguous signature objects."""
    # Arrange
    invalid_signature = ("float64", "float64")

    # Act / Assert
    with pytest.raises(TypeError, match="signature string"):
        compilation.numbify(signature=invalid_signature)
