from cmath import isclose
from itertools import product

from rocketpy.tools import cached_property
from rocketpy.mathutils import Vector


class Matrix:
    """Pure Python 3x3 Matrix class for simple matrix-matrix and matrix-vector
    operations.

    Notes
    -----
    Instances of the Matrix class are immutable.
    Real and complex components are supported.

    Examples
    --------
    Creating a Matrix instance requires passing its components as a nested
    iterable:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M
    Matrix([1, 2, 3],
           [4, 5, 6],
           [7, 8, 9])

    Matrix instances can be indexed and sliced like lists:

    >>> M[0]
    [1, 2, 3]

    >>> M[0][0]
    1

    >>> M[0, 0]
    1

    >>> M[0, 0:2]
    [1, 2]

    Matrix instances components can be accessed as attributes:

    >>> M.xx, M.xy, M.xz
    (1, 2, 3)

    Matrix instances can be called as functions, if their elements are
    callable:

    >>> M = Matrix([[lambda x: x**1, lambda x: x**2, lambda x: x**3],
    ...             [lambda x: x**4, lambda x: x**5, lambda x: x**6],
    ...             [lambda x: x**7, lambda x: x**8, lambda x: x**9]])
    >>> M(2)
    Matrix([2, 4, 8],
           [16, 32, 64],
           [128, 256, 512])

    Matrix instances can be added, subtracted, multiplied and divided by
    scalars:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M + M
    Matrix([2, 4, 6],
           [8, 10, 12],
           [14, 16, 18])

    >>> M - M
    Matrix([0, 0, 0],
           [0, 0, 0],
           [0, 0, 0])

    >>> M * 2
    Matrix([2, 4, 6],
           [8, 10, 12],
           [14, 16, 18])

    >>> M / 2
    Matrix([0.5, 1.0, 1.5],
           [2.0, 2.5, 3.0],
           [3.5, 4.0, 4.5])

    Matrix instances can be multiplied (inner product) by other matrices:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> N = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M @ N
    Matrix([30, 36, 42],
           [66, 81, 96],
           [102, 126, 150])

    Matrix instances can be used to transform vectors by the inner product:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> v = Vector([1, 2, 3])
    >>> M @ v
    Vector(14, 32, 50)

    Matrix instances can be transposed and inverted:

    >>> M = Matrix([[1, 2, 3], [4, 0, 6], [7, 8, 9]])
    >>> M.transpose
    Matrix([1, 4, 7],
           [2, 0, 8],
           [3, 6, 9])
    >>> M.inverse
    Matrix([-0.8, 0.1, 0.2],
           [0.1, -0.2, 0.1],
           [0.5333333333333333, 0.1, -0.13333333333333333])

    Matrix instances can be element-wise operated on by callables:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.element_wise(lambda x: x**2)
    Matrix([1, 4, 9],
           [16, 25, 36],
           [49, 64, 81])

    Determinants can be calculated:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.det
    0
    >>> abs(M)
    0

    Matrices can be compared for equality:

    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> N = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M == N
    True
    >>> M != N
    False
    """

    __array_ufunc__ = None

    def __init__(self, components):
        """Matrix class constructor.

        Parameters
        ----------
        components : 3x3 array-like
            3x3 array-like with matrix components. Indexing must be
            [row, column].

        Examples
        --------
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> M
        Matrix([1, 2, 3],
               [4, 5, 6],
               [7, 8, 9])
        """
        self.components = components
        self.x, self.y, self.z = self.components
        self.xx, self.xy, self.xz = self.x
        self.yx, self.yy, self.yz = self.y
        self.zx, self.zy, self.zz = self.z

    def __getitem__(self, args):
        """Adds support for indexing and slicing."""
        if isinstance(args, int):
            return self.components[args]
        else:
            return self.components[args[0]][args[1]]

    def __iter__(self):
        """Adds support for iteration."""
        return iter(self.components)

    def __call__(self, *args):
        """Adds support for calling a matrix as a function, if its elements are
        callable.

        Parameters
        ----------
        args : tuple
            Arguments to be passed to the matrix elements.

        Returns
        -------
        Matrix
            Matrix with the same shape as the original, but with its elements
            replaced by the result of calling them with the given arguments.

        Examples
        --------
        >>> M = Matrix([[lambda x: x**1, lambda x: x**2, lambda x: x**3],
        ...             [lambda x: x**4, lambda x: x**5, lambda x: x**6],
        ...             [lambda x: x**7, lambda x: x**8, lambda x: x**9]])
        >>> M(2)
        Matrix([2, 4, 8],
               [16, 32, 64],
               [128, 256, 512])
        """
        try:
            return self.element_wise(lambda f: f(*args))
        except TypeError as exc:
            msg = "One or more elements of this matrix is not callable."
            raise TypeError(msg) from exc

    def __len__(self):
        """Adds support for the len() function."""
        return 3

    @cached_property
    def shape(self):
        """tuple: Shape of the matrix."""
        return (3, 3)

    @cached_property
    def trace(self):
        """Matrix trace, sum of its diagonal components."""
        return self.xx + self.yy + self.zz

    @cached_property
    def transpose(self):
        """Matrix transpose."""
        return Matrix(
            [
                [self.xx, self.yx, self.zx],
                [self.xy, self.yy, self.zy],
                [self.xz, self.yz, self.zz],
            ]
        )

    @cached_property
    def det(self):
        """Matrix determinant."""
        return self.__abs__()

    @cached_property
    def is_diagonal(self, tol=1e-6):
        """Boolean indicating if matrix is diagonal.

        Parameters
        ----------
        tol : float, optional
            Tolerance used to determine if non-diagonal elements are negligible.
            Defaults to 1e-6.

        Returns
        -------
        bool
            True if matrix is diagonal, False otherwise.

        Examples
        --------
        >>> M = Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        >>> M.is_diagonal
        True

        >>> M = Matrix([[1, 0, 0], [0, 2, 0], [0, 1e-7, 3]])
        >>> M.is_diagonal
        True

        >>> M = Matrix([[1, 0, 0], [0, 2, 0], [0, 1e-5, 3]])
        >>> M.is_diagonal
        False
        """
        for i, j in product(range(3), range(3)):
            if i == j:
                continue
            if abs(self[i, j]) > tol:
                return False
        return True

    @cached_property
    def inverse(self):
        """Matrix inverse.

        Returns
        -------
        Matrix
            Inverse of the matrix.

        Notes
        -----
        If the matrix is diagonal, the inverse is computed by inverting its
        diagonal elements. If not, the inverse is computed using the adjugate
        matrix.

        Raises
        ------
        ZeroDivisionError
            If the matrix is singular.
        """
        ixx = self.yy * self.zz - self.zy * self.yz
        iyx = self.zx * self.yz - self.yx * self.zz
        izx = self.yx * self.zy - self.zx * self.yy
        ixy = self.zy * self.xz - self.xy * self.zz
        iyy = self.xx * self.zz - self.zx * self.xz
        izy = self.zx * self.xy - self.xx * self.zy
        ixz = self.xy * self.yz - self.yy * self.xz
        iyz = self.yx * self.xz - self.yz * self.xx
        izz = self.xx * self.yy - self.yx * self.xy
        det = self.xx * ixx + self.xy * iyx + self.xz * izx
        return Matrix(
            [
                [ixx / det, ixy / det, ixz / det],
                [iyx / det, iyy / det, iyz / det],
                [izx / det, izy / det, izz / det],
            ]
        )

    def __abs__(self):
        """Matrix determinant."""
        ixx = self.yy * self.zz - self.zy * self.yz
        iyx = self.zx * self.yz - self.yx * self.zz
        izx = self.yx * self.zy - self.zx * self.yy
        det = self.xx * ixx + self.xy * iyx + self.xz * izx
        return det

    def __neg__(self):
        """-1 times 3x3 matrix self."""
        return Matrix(
            [
                [-self.xx, -self.xy, -self.xz],
                [-self.yx, -self.yy, -self.yz],
                [-self.zx, -self.zy, -self.zz],
            ]
        )

    def __add__(self, other):
        """Sum two 3x3 matrices."""
        return Matrix(
            [
                [self.xx + other.xx, self.xy + other.xy, self.xz + other.xz],
                [self.yx + other.yx, self.yy + other.yy, self.yz + other.yz],
                [self.zx + other.zx, self.zy + other.zy, self.zz + other.zz],
            ]
        )

    def __sub__(self, other):
        """Subtract two 3x3 matrices."""
        return Matrix(
            [
                [self.xx - other.xx, self.xy - other.xy, self.xz - other.xz],
                [self.yx - other.yx, self.yy - other.yy, self.yz - other.yz],
                [self.zx - other.zx, self.zy - other.zy, self.zz - other.zz],
            ]
        )

    def __mul__(self, other):
        """Element wise multiplication of 3x3 matrix self by scalar other."""
        return Matrix(
            [
                [other * self.xx, other * self.xy, other * self.xz],
                [other * self.yx, other * self.yy, other * self.yz],
                [other * self.zx, other * self.zy, other * self.zz],
            ]
        )

    def __rmul__(self, other):
        """Element wise multiplication of 3x3 matrix self by scalar other."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element wise division is carried out."""
        return Matrix(
            [
                [self.xx / other, self.xy / other, self.xz / other],
                [self.yx / other, self.yy / other, self.yz / other],
                [self.zx / other, self.zy / other, self.zz / other],
            ]
        )

    def __matmul__(self, other):
        """Dot (inner) product between two 3x3 matrices or between 3x3 matrix
        and R3 vector.

        Parameters
        ----------
        other : Matrix or Vector
            The other matrix or vector.

        Returns
        -------
        Matrix or Vector
            The result of the dot product. A Matrix if other if Matrix, and
            a Vector if other is Vector.

        Examples
        --------
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Vector([1, 2, 3])
        >>> M @ v
        Vector(14, 32, 50)

        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> N = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> M @ N
        Matrix([30, 36, 42],
               [66, 81, 96],
               [102, 126, 150])
        """
        if isinstance(other, Vector):
            return Vector(
                [
                    self.xx * other.x + self.xy * other.y + self.xz * other.z,
                    self.yx * other.x + self.yy * other.y + self.yz * other.z,
                    self.zx * other.x + self.zy * other.y + self.zz * other.z,
                ]
            )
        elif isinstance(other, Matrix):
            return Matrix(
                [
                    [
                        self.xx * other.xx + self.xy * other.yx + self.xz * other.zx,
                        self.xx * other.xy + self.xy * other.yy + self.xz * other.zy,
                        self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
                    ],
                    [
                        self.yx * other.xx + self.yy * other.yx + self.yz * other.zx,
                        self.yx * other.xy + self.yy * other.yy + self.yz * other.zy,
                        self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
                    ],
                    [
                        self.zx * other.xx + self.zy * other.yx + self.zz * other.zx,
                        self.zx * other.xy + self.zy * other.yy + self.zz * other.zy,
                        self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
                    ],
                ]
            )
        else:
            raise TypeError("Can only dot product with Matrix or Vector.")

    def __pow__(self, other):
        """Exponentiation of 3x3 matrix by integer other.

        Parameters
        ----------
        other : int
            The exponent.

        Returns
        -------
        Matrix
            The result of exponentiation.

        Examples
        --------
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> M ** 2
        Matrix([30, 36, 42],
               [66, 81, 96],
               [102, 126, 150])
        """
        result = Matrix.identity()
        for _ in range(other):
            result = result @ self
        return result

    def __eq__(self, other):
        """Equality of two 3x3 matrices.

        Parameters
        ----------
        other : Matrix
            The other matrix.

        Returns
        -------
        bool
            True if the two matrices are equal, False otherwise.

        Examples
        --------
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> N = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> M == N
        True

        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> N = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        >>> M == N
        False

        Notes
        -----
        Equality is determined by comparing each element of the two matrices
        with an absolute tolerance of 1e-9 using Python's cmath.isclose.
        """
        return (
            len(other) == 3
            and isclose(self.xx, other[0][0], rel_tol=0, abs_tol=1e-9)
            and isclose(self.xy, other[0][1], rel_tol=0, abs_tol=1e-9)
            and isclose(self.xz, other[0][2], rel_tol=0, abs_tol=1e-9)
            and isclose(self.yx, other[1][0], rel_tol=0, abs_tol=1e-9)
            and isclose(self.yy, other[1][1], rel_tol=0, abs_tol=1e-9)
            and isclose(self.yz, other[1][2], rel_tol=0, abs_tol=1e-9)
            and isclose(self.zx, other[2][0], rel_tol=0, abs_tol=1e-9)
            and isclose(self.zy, other[2][1], rel_tol=0, abs_tol=1e-9)
            and isclose(self.zz, other[2][2], rel_tol=0, abs_tol=1e-9)
        )

    def element_wise(self, operation):
        """Element wise operation.

        Parameters
        ----------
        operation : callable
            Callable with a single argument, which should take an element and
            return the result of the desired operation.

        Returns
        -------
        Matrix
            The result of the element wise operation.

        Examples
        --------
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> M.element_wise(lambda x: x ** 2)
        Matrix([1, 4, 9],
               [16, 25, 36],
               [49, 64, 81])
        """
        return Matrix(
            [
                [operation(self.xx), operation(self.xy), operation(self.xz)],
                [operation(self.yx), operation(self.yy), operation(self.yz)],
                [operation(self.zx), operation(self.zy), operation(self.zz)],
            ]
        )

    def dot(self, other):
        """Dot product between two 3x3 matrices or between 3x3 matrix and R3
        vector.

        See Also
        --------
        Matrix.__matmul__
        """
        return self.__matmul__(other)

    def __str__(self):
        return (
            f"[{self.xx}, {self.xy}, {self.xz}]\n"
            + f"[{self.yx}, {self.yy}, {self.yz}]\n"
            + f"[{self.zx}, {self.zy}, {self.zz}]]"
        )

    def __repr__(self):
        return (
            f"Matrix([{self.xx}, {self.xy}, {self.xz}],\n"
            + f"       [{self.yx}, {self.yy}, {self.yz}],\n"
            + f"       [{self.zx}, {self.zy}, {self.zz}])"
        )

    @staticmethod
    def identity():
        """Returns the 3x3 identity matrix."""
        return Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @staticmethod
    def zeros():
        """Returns the 3x3 zero matrix."""
        return Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    @staticmethod
    def transformation(quaternion):
        """Returns the transformation Matrix from frame B to frame A, where B
        is rotated by the quaternion q with respect to A.

        Parameters
        ----------
        q : tuple of 4 floats
            The quaternion representing the rotation from frame A to frame B.
            Example: (cos(phi/2), 0, 0, sin(phi/2)) represents a rotation of
            phi around the z-axis.
            Note: the quaternion must be normalized.

        Returns
        -------
        Matrix
            The transformation matrix from frame B to frame A.
        """
        q_w, q_x, q_y, q_z = quaternion
        return Matrix(
            [
                [
                    1 - 2 * (q_y**2 + q_z**2),
                    2 * (q_x * q_y - q_w * q_z),
                    2 * (q_x * q_z + q_w * q_y),
                ],
                [
                    2 * (q_x * q_y + q_w * q_z),
                    1 - 2 * (q_x**2 + q_z**2),
                    2 * (q_y * q_z - q_w * q_x),
                ],
                [
                    2 * (q_x * q_z - q_w * q_y),
                    2 * (q_y * q_z + q_w * q_x),
                    1 - 2 * (q_x**2 + q_y**2),
                ],
            ]
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
