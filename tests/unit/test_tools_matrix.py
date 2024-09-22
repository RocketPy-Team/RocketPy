import numpy as np
import pytest

from rocketpy import Function
from rocketpy.mathutils import Matrix, Vector

test_matrix_1 = [[-7, 2, 3], [4, 5, -6], [1, -8, 9]]

test_matrix_2 = [[np.pi, 2.5, 3.7], [4.2, np.e, -6.7], [1.1, -8.0, 0]]

test_matrix_3 = [
    [0.1 + 1.0j, 3.1 + 1.2j, 2 + 0.5j],
    [2.1 + 0.5j, 1.5 + 0.5j, 1 + 1.8j],
    [5.2 + 1.3j, 4.2 + 7.7j, 7 + 5.3j],
]

test_matrix_4 = [[-7, 0, 0], [0, 5, 0], [0, 0, 9]]

test_matrix_5 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

test_matrices = [
    test_matrix_1,
    test_matrix_2,
    test_matrix_3,
    test_matrix_4,
    test_matrix_5,
]


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_constructor(components):
    matrix = Matrix(components)
    assert matrix.components == components


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_getitem(components):
    matrix = Matrix(components)
    for i, j in [(i, j) for i in range(3) for j in range(3)]:
        assert matrix[i, j] == components[i][j]


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_iter(components):
    matrix = Matrix(components)
    for i, j in zip(matrix, components):
        assert i == j


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_call(components):
    f = Function(lambda x: x**2)
    matrix = Matrix(components)
    callable_matrix = matrix * f
    assert callable_matrix(1) == matrix
    assert callable_matrix(2) == 4 * matrix


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_len(components):
    matrix = Matrix(components)
    assert len(matrix) == 3


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_shape(components):
    matrix = Matrix(components)
    assert matrix.shape == (3, 3)


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_trace(components):
    matrix = Matrix(components)
    assert matrix.trace == np.trace(matrix)


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_transpose(components):
    matrix = Matrix(components)
    assert matrix.transpose == np.transpose(matrix)


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_det(components):
    matrix = Matrix(components)
    assert matrix.det == pytest.approx(np.linalg.det(matrix))


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_is_diagonal(components):
    matrix = Matrix(components)
    assert matrix.is_diagonal == (matrix.det == matrix.xx * matrix.yy * matrix.zz)


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_inverse(components):
    matrix = Matrix(components)
    if matrix.det == 0:
        with pytest.raises(ZeroDivisionError):
            assert matrix.inverse
    else:
        assert matrix.inverse == np.linalg.inv(matrix)


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_abs(components):
    matrix = Matrix(components)
    assert abs(matrix) == pytest.approx(np.linalg.det(matrix))


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_neg(components):
    assert -Matrix(components) + Matrix(components) == Matrix.zeros()


@pytest.mark.parametrize("A", test_matrices)
@pytest.mark.parametrize("B", test_matrices)
def test_matrix_add(A, B):
    expected_result = np.array(A) + np.array(B)
    assert Matrix(A) + Matrix(B) == expected_result


@pytest.mark.parametrize("A", test_matrices)
@pytest.mark.parametrize("B", test_matrices)
def test_matrix_sub(A, B):
    expected_result = np.array(A) - np.array(B)
    assert Matrix(A) - Matrix(B) == expected_result


@pytest.mark.parametrize("k", [-1, 0, 1, np.pi])
@pytest.mark.parametrize("A", test_matrices)
def test_matrix_mul(A, k):
    A = Matrix(A)
    assert A * k == k * np.array(A)


@pytest.mark.parametrize("k", [-1, 0, 1, np.pi])
@pytest.mark.parametrize("A", test_matrices)
def test_matrix_rmul(A, k):
    np_array = np.array(A)
    A = Matrix(A)
    assert k * A == k * np_array


@pytest.mark.parametrize("A", test_matrices)
@pytest.mark.parametrize("k", [-1, 1, np.pi, np.e])
def test_matrix_truediv(A, k):
    A = Matrix(A)
    assert A / k == np.array(A) / k


@pytest.mark.parametrize("A", test_matrices)
@pytest.mark.parametrize("B", test_matrices)
def test_matrix_matmul_matrices(A, B):
    expected_result = np.dot(A, B)
    assert Matrix(A) @ Matrix(B) == expected_result


@pytest.mark.parametrize("A", test_matrices)
@pytest.mark.parametrize("B", [[1, 2, 3], [-np.pi, 1, np.e], [3 * 1j, -2j, 0j]])
def test_matrix_matmul_vectors(A, B):
    expected_result = np.dot(A, B)
    assert Matrix(A) @ Vector(B) == expected_result


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("A", test_matrices)
def test_matrix_pow(A, k):
    A = Matrix(A)
    assert A**k == np.linalg.matrix_power(A, k)


@pytest.mark.parametrize("matrix_components", test_matrices)
def test_matrix_eq(matrix_components):
    matrix = Matrix(matrix_components)
    assert matrix == matrix_components
    assert (matrix == 2 * matrix) is False


@pytest.mark.parametrize("operation", [lambda i: i**2, lambda i: 1 / (i + 1.1)])
@pytest.mark.parametrize("matrix_components", test_matrices)
def test_matrix_element_wise(matrix_components, operation):
    matrix = Matrix(matrix_components)
    matrix = matrix.element_wise(operation)
    assert Matrix(
        [
            [operation(matrix.xx), operation(matrix.xy), operation(matrix.xz)],
            [operation(matrix.yx), operation(matrix.yy), operation(matrix.yz)],
            [operation(matrix.zx), operation(matrix.zy), operation(matrix.zz)],
        ]
    )


@pytest.mark.parametrize("A", test_matrices)
@pytest.mark.parametrize("B", test_matrices)
def test_matrix_dot(A, B):
    A, B = Matrix(A), Matrix(B)
    assert A.dot(B) == np.dot(A, B)


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_str(components):
    matrix = Matrix(components)
    assert str(matrix) == (
        f"[{components[0][0]}, {components[0][1]}, {components[0][2]}]\n"
        + f"[{components[1][0]}, {components[1][1]}, {components[1][2]}]\n"
        + f"[{components[2][0]}, {components[2][1]}, {components[2][2]}]]"
    )


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_repr(components):
    matrix = Matrix(components)
    assert repr(matrix) == (
        f"Matrix([{components[0][0]}, {components[0][1]}, {components[0][2]}],\n"
        + f"       [{components[1][0]}, {components[1][1]}, {components[1][2]}],\n"
        + f"       [{components[2][0]}, {components[2][1]}, {components[2][2]}])"
    )


def test_matrix_identity():
    assert Matrix.identity() == np.eye(3)


def test_matrix_zeros():
    assert Matrix.zeros() == np.zeros((3, 3))


def test_matrix_transformation():
    # Check that the matrix is orthogonal
    phi = 45 * np.pi / 180
    n = Vector([1, 1, 1])
    q0 = np.cos(phi / 2)
    q1, q2, q3 = np.sin(phi / 2) * n.unit_vector
    matrix = Matrix.transformation((q0, q1, q2, q3))
    assert matrix @ matrix.transpose == Matrix.identity()

    # Check that the matrix rotates the vector correctly
    phi = np.pi / 2
    n = Vector([1, 0, 0])
    q0 = np.cos(phi / 2)
    q1, q2, q3 = np.sin(phi / 2) * n
    matrix = Matrix.transformation((q0, q1, q2, q3))
    assert matrix @ Vector([0, 0, 1]) == Vector([0, -1, 0])


def test_matrix_transformation_euler_angles():
    roll = np.pi / 2
    pitch = np.pi / 2
    roll2 = np.pi / 2
    matrix = Matrix.transformation_euler_angles(roll, pitch, roll2)
    matrix = matrix.round(12)
    # Check that the matrix is orthogonal
    assert matrix @ matrix.transpose == Matrix.identity()
    # Check that the matrix rotates the vector correctly
    assert matrix @ Vector([0, 0, 1]) == Vector([1, 0, 0])


def test_matrix_round():
    matrix = [[2e-10, -2e-10, 0], [5.1234, -5.1234, 0], [0, 0, 9]]
    matrix = Matrix(matrix).round(3)
    assert matrix == Matrix([[0, 0, 0], [5.123, -5.123, 0], [0, 0, 9]])


@pytest.mark.parametrize("components", test_matrices)
def test_matrix_x_y_z(components):
    matrix = Matrix(components)
    assert matrix.xx == components[0][0]
    assert matrix.xy == components[0][1]
    assert matrix.xz == components[0][2]
    assert matrix.yx == components[1][0]
    assert matrix.yy == components[1][1]
    assert matrix.yz == components[1][2]
    assert matrix.zx == components[2][0]
    assert matrix.zy == components[2][1]
    assert matrix.zz == components[2][2]
