import pytest

from rocketpy import PiecewiseFunction


@pytest.mark.parametrize(
    "source",
    [
        ((0, 4), lambda x: x),
        {"0-4": lambda x: x},
        {(0, 4): lambda x: x, (3, 5): lambda x: 2 * x},
    ],
)
def test_invalid_source(source):
    """Test an error is raised when the source parameter is invalid"""
    with pytest.raises((TypeError, ValueError)):
        PiecewiseFunction(source)


@pytest.mark.parametrize(
    "source",
    [
        {(-1, 0): lambda x: -x, (0, 1): lambda x: x},
        {
            (0, 1): lambda x: x,
            (1, 2): lambda x: 1,
            (2, 3): lambda x: 3 - x,
        },
    ],
)
@pytest.mark.parametrize("inputs", [None, "X"])
@pytest.mark.parametrize("outputs", [None, "Y"])
def test_new(source, inputs, outputs):
    """Test if PiecewiseFunction.__new__ runs correctly"""
    PiecewiseFunction(source, inputs, outputs)
