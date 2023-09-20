from cmath import isclose

from rocketpy.tools import cached_property
from rocketpy.mathutils import Matrix


class Vector:
    """Pure python basic R3 vector class designed for simple operations.

    Notes
    -----
    Instances of the Vector class are immutable.
    Real and complex components are supported.

    Examples
    --------
    Creating a Vector instance requires passing its components as an iterable:

    >>> v = Vector([1, 2, 3])
    >>> v
    Vector(1, 2, 3)

    Vector components can be accessed by x, y and z or by indexing:

    >>> v.x, v.y, v.z
    (1, 2, 3)

    >>> v[0], v[1], v[2]
    (1, 2, 3)

    Vector instances can be added, subtracted, multiplied by a scalar, divided
    by a scalar, negated, and cross and dot product can be computed:

    >>> v + v
    Vector(2, 4, 6)

    >>> v - v
    Vector(0, 0, 0)

    >>> v * 2
    Vector(2, 4, 6)

    >>> v / 2
    Vector(0.5, 1.0, 1.5)

    >>> -v
    Vector(-1, -2, -3)

    >>> v @ v # Dot product
    14

    Cross products need to be wrapped in parentheses to ensure the ^ operator
    precedence:

    >>> (v ^ v)
    Vector(0, 0, 0)

    Vector instances can be called as functions if their elements are callable:

    >>> v = Vector([lambda x: x**2, lambda x: x**3, lambda x: x**4])
    >>> v(2)
    Vector(4, 8, 16)

    Vector instances magnitudes can be accessed as its absolute value:

    >>> v = Vector([1, 2, 3])
    >>> abs(v)
    3.7416573867739413

    Vector instances can be normalized:

    >>> v.unit_vector
    Vector(0.2672612419124244, 0.5345224838248488, 0.8017837257372732)

    Vector instances can be compared for equality:

    >>> v = Vector([1, 2, 3])
    >>> u = Vector([1, 2, 3])
    >>> v == u
    True
    >>> v != u
    False

    And last, but not least, it is also possible to check if two vectors are
    parallel or orthogonal:

    >>> v = Vector([1, 2, 3])
    >>> u = Vector([2, 4, 6])
    >>> v.is_parallel_to(u)
    True
    >>> v.is_orthogonal_to(u)
    False
    """

    __array_ufunc__ = None

    def __init__(self, components):
        """Vector class constructor.

        Parameters
        ----------
        components : array-like, iterable
            An iterable with length equal to 3, corresponding to x, y and z
            components.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> v
        Vector(1, 2, 3)
        """
        self.components = components
        self.x, self.y, self.z = self.components

    def __getitem__(self, i):
        """Access vector components by indexing."""
        return self.components[i]

    def __iter__(self):
        """Adds support for iteration."""
        return iter(self.components)

    def __call__(self, *args):
        """Adds support for calling a vector as a function, if its elements are
        callable.

        Parameters
        ----------
        args : arguments
            Arguments to be passed to the vector elements.

        Returns
        -------
        Vector
            Vector with the return of each element called with the given
            arguments.

        Examples
        --------
        >>> v = Vector([lambda x: x**2, lambda x: x**3, lambda x: x**4])
        >>> v(2)
        Vector(4, 8, 16)
        """
        try:
            return self.element_wise(lambda f: f(*args))
        except TypeError as exc:
            msg = "One or more elements of this vector is not callable."
            raise TypeError(msg) from exc

    def __len__(self):
        return 3

    @cached_property
    def unit_vector(self):
        """R3 vector with the same direction of self, but normalized."""
        return self / abs(self)

    @cached_property
    def cross_matrix(self):
        """Skew symmetric matrix used for cross product.

        Notes
        -----
        The cross product between two vectors can be computed as the matrix
        product between the cross matrix of the first vector and the second
        vector.

        Examples
        --------
        >>> v = Vector([1, 7, 3])
        >>> u = Vector([2, 5, 6])
        >>> (v ^ u) == v.cross_matrix @ u
        True
        """
        return Matrix(
            [[0, -self.z, self.y], [self.z, 0, -self.x], [-self.y, self.x, 0]]
        )

    def __abs__(self):
        """R3 vector norm, magnitude or absolute value."""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def __neg__(self):
        """-1 times R3 vector self."""
        return Vector([-self.x, -self.y, -self.z])

    def __add__(self, other):
        """Sum two R3 vectors."""
        return Vector([self.x + other.x, self.y + other.y, self.z + other.z])

    def __sub__(self, other):
        """Subtract two R3 vectors."""
        return Vector([self.x - other.x, self.y - other.y, self.z - other.z])

    def __mul__(self, other):
        """Component wise multiplication between R3 vector and scalar other."""
        return self.__rmul__(other)

    def __rmul__(self, other):
        """Component wise multiplication between R3 vector and scalar other."""
        return Vector([other * self.x, other * self.y, other * self.z])

    def __truediv__(self, other):
        """Component wise division between R3 vector and scalar other."""
        return Vector([self.x / other, self.y / other, self.z / other])

    def __xor__(self, other):
        """Cross product between self and other.

        Parameters
        ----------
        other : Vector
            R3 vector to be crossed with self.

        Returns
        -------
        Vector
            R3 vector resulting from the cross product between self and other.

        Examples
        --------
        >>> v = Vector([1, 7, 3])
        >>> u = Vector([2, 5, 6])
        >>> (v ^ u)
        Vector(27, 0, -9)

        Notes
        -----
        Parameters order matters, since cross product is not commutative.
        Parentheses are required when using cross product with the ^ operator
        to avoid ambiguity with the bitwise xor operator and keep the
        precedence of the operators.
        """
        return Vector(
            [
                self.y * other.z - self.z * other.y,
                -self.x * other.z + self.z * other.x,
                self.x * other.y - self.y * other.x,
            ]
        )

    def __matmul__(self, other):
        """Dot product between two R3 vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __eq__(self, other):
        """Check if two R3 vectors are equal.

        Parameters
        ----------
        other : Vector
            R3 vector to be compared with self.

        Returns
        -------
        bool
            True if self and other are equal. False otherwise.

        Examples
        --------
        >>> v = Vector([1, 7, 3])
        >>> u = Vector([1, 7, 3])
        >>> v == u
        True

        Notes
        -----
        Two R3 vectors are equal if their components are equal or almost equal.
        Python's cmath.isclose function is used to compare the components.
        """
        return (
            len(other) == 3
            and isclose(self.x, other[0], rel_tol=0, abs_tol=1e-9)
            and isclose(self.y, other[1], rel_tol=0, abs_tol=1e-9)
            and isclose(self.z, other[2], rel_tol=0, abs_tol=1e-9)
        )

    def is_parallel_to(self, other):
        """Returns True if self is parallel to R3 vector other. False otherwise.

        Parameters
        ----------
        other : Vector
            R3 vector to be compared with self.

        Returns
        -------
        bool
            True if self and other are parallel. False otherwise.

        Notes
        -----
        Two R3 vectors are parallel if their cross product is the zero vector.
        Python's cmath.isclose function is used to assert this.
        """
        return self ^ other == Vector([0, 0, 0])

    def is_orthogonal_to(self, other):
        """Returns True if self is perpendicular to R3 vector other. False
        otherwise.

        Parameters
        ----------
        other : Vector
            R3 vector to be compared with self.

        Returns
        -------
        bool
            True if self and other are perpendicular. False otherwise.

        Notes
        -----
        Two R3 vectors are perpendicular if their dot product is zero.
        Python's cmath.isclose function is used to assert this with absolute
        tolerance of 1e-9.
        """
        return isclose(self @ other, 0, rel_tol=0, abs_tol=1e-9)

    def element_wise(self, operation):
        """Element wise operation.

        Parameters
        ----------
        operation : callable
            Callable with a single argument, which should take an element and
            return the result of the desired operation.

        Examples
        --------
        >>> v = Vector([1, 7, 3])
        >>> v.element_wise(lambda x: x**2)
        Vector(1, 49, 9)
        """
        return Vector([operation(self.x), operation(self.y), operation(self.z)])

    def dot(self, other):
        """Dot product between two R3 vectors."""
        return self.__matmul__(other)

    def cross(self, other):
        """Cross product between two R3 vectors."""
        return self.__xor__(other)

    def proj(self, other):
        """Scalar projection of R3 vector self onto R3 vector other.

        Parameters
        ----------
        other : Vector
            R3 vector to be projected onto.

        Returns
        -------
        float
            Scalar projection of self onto other.

        Examples
        --------
        >>> v = Vector([1, 7, 3])
        >>> u = Vector([2, 5, 6])
        >>> v.proj(u)
        6.821910402406465
        """
        return (self @ other) / abs(other)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"

    @staticmethod
    def zeros():
        """Returns the zero vector."""
        return Vector([0, 0, 0])

    @staticmethod
    def i():
        """Returns the i vector, [1, 0, 0]."""
        return Vector([1, 0, 0])

    @staticmethod
    def j():
        """Returns the j vector, [0, 1, 0]."""
        return Vector([0, 1, 0])

    @staticmethod
    def k():
        """Returns the k vector, [0, 0, 1]."""
        return Vector([0, 0, 1])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
