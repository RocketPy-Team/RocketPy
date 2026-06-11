"""Strategy interfaces for interpolation/extrapolation evaluation."""

from abc import ABC, abstractmethod


class PolationBase(ABC):
    """Base strategy for evaluation and optional derivatives/integrals."""

    @abstractmethod
    def evaluate(self, x, _is_iterable=None):
        """Evaluate the strategy at the given input(s)."""

    def derivative(self, x, _is_iterable=None):
        raise NotImplementedError("Analytical derivative not available.")

    def second_derivative(self, x, _is_iterable=None):
        raise NotImplementedError("Analytical 2nd derivative not available.")

    def integral(self, x, _is_iterable=None):
        raise NotImplementedError("Analytical antiderivative not available.")

    def definite_integral(self, a, b):
        """Evaluate the definite integral from a to b."""
        return self.integral(b) - self.integral(a)

    def coefficients(self):
        return None

    def expose(self):
        """Returns a fast callable. By default, returns self.evaluate."""
        return self.evaluate
