import pytest
import numpy as np

from rocketpy.tools import Vector, Matrix

example_matrix_1 = [
    [-7,  2,  3],
    [ 4,  5, -6],
    [ 1, -8,  9]
]

example_matrix_2 = [
    [-7,  2,  3],
    [ 4,  5, -6],
    [ 1, -8,  9]
]

example_matrix_3 = [
    [-7,  2,  3],
    [ 4,  5, -6],
    [ 1, -8,  9]
]

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_constructor(components):
    matrix = Matrix(components)
    assert matrix.components == components

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_getitem(components):
    matrix = Matrix(components)
    for i, j in [i, j for i in range(3) for j in range(3)]
        assert matrix[i, j] == components[i][j]

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_constructor(components):
    matrix = Matrix(components)
    assert matrix.shape == (3, 3)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
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

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_trace(components):
    matrix = Matrix(components)
    assert matrix.trace == np.trace(matrix)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_transpose(components):
    matrix = Matrix(components)
    assert matrix.transpose == np.transpose(matrix)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_inverse(components):
    matrix = Matrix(components)
    assert matrix.inverse == np.linalg.inv(matrix)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_det(components):
    matrix = Matrix(components)
    assert matrix.det == np.linalg.det(matrix)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_is_diagonal(components):
    matrix = Matrix(components)
    assert matrix.is_diagonal == (matrix.det == matrix.xx * matrix.yy * matrix.zz)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_abs(components):
    matrix = Matrix(components)
    assert abs(matrix) == np.linalg.det(matrix)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_neg(components):
    matrix = -Matrix(components)
    assert matrix == -np.array(components)

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_add(A_c, B_c):
    A, B = Matrix(A_c), Matrix(B_c)
    assert A + B == np.array(A_c) + np.array(B_c)
    A, C = Matrix(A_c), np.array(B_c)
    assert A + C == np.array(A_c) + C

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_radd(A_c, B_c):
    A, B = Matrix(A_c), np.array(B_c)
    assert B + A == np.array(A_c) + np.array(B_c)

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_sub(A_c, B_c):
    A, B = Matrix(A_c), Matrix(B_c)
    assert A - B == np.array(A_c) - np.array(B_c)
    A, C = Matrix(A_c), np.array(B_c)
    assert A - C == np.array(A_c) - C

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_rsub(A_c, B_c):
    A, B = Matrix(A_c), np.array(B_c)
    assert B - A == np.array(A_c) - np.array(B_c)

@pytest.mark.parametrize("k", [-1, 0, 1, np.pi, -np.inf, np.inf])
@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_mul(A_c, k):
    A = Matrix(A_c)
    assert k * A == k*np.array(A_c)

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("k", [-1, 0, 1, np.pi, -np.inf, np.inf])
def test_matrix_truediv_scalar(A_c, k):
    A = Matrix(A_c)
    assert A / k == np.array(A)/k

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_truediv_inverse(A_c, B_c):
    A, B = Matrix(A_c), Matrix(B_c)
    assert A / B == np.dot(A, np.linalg.inv(B))

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_rtruediv(A_c, B_c):
    A, B = Matrix(A_c), np.array(B_c)
    assert A / B == np.dot(A, np.linalg.inv(B))

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3, [1, 2, 3], [-np.pi, np.inf, np.e], [3*1j, -2j, 0j]])
def test_matrix_matmul(A_c, B_c):
    A, B = Matrix(A_c), np.array(B_c)
    assert A @ B == np.dot(A, B)

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3, [1, 2, 3], [-np.pi, np.inf, np.e], [3*1j, -2j, 0j]])
def test_matrix_matmul(A_c, B_c):
    A, B = Matrix(A_c), np.array(B_c)
    assert B @ A == np.dot(B, A)

@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 10])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_pow(A_c, k):
    A = Matrix(A_c)
    assert A**k == np.array(A)**k

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_eq(matrix_components):
    matrix = Matrix(matrix_components)
    assert matrix == matrix
    assert matrix == matrix_components
    assert (matrix == 2*matrix) == False


@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_neq(matrix_components):
    matrix = Matrix(matrix_components)
    assert matrix != 2*matrix
    assert 2*matrix != matrix_components
    assert (matrix != matrix) == False

@pytest.mark.parametrize("operation", [lambda i: i**2, lambda i: 1/i])
@pytest.mark.parametrize("matrix_components", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_element_wise(matrix_components, operation):
    matrix = Matrix(matrix_components)
    matrix = matrix.element_wise(operation)
    assert Matrix([
            [operation(matrix.xx), operation(matrix.xy), operation(matrix.xz)],
            [operation(matrix.yx), operation(matrix.yy), operation(matrix.yz)],
            [operation(matrix.zx), operation(matrix.zy), operation(matrix.zz)]
    ])

@pytest.mark.parametrize("A_c", [example_matrix_1, example_matrix_2, example_matrix_3])
@pytest.mark.parametrize("B_c", [example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_dot(A_c, B_c):
    A, B = Matrix(A_c), np.array(B_c)
    assert A @ B == np.dot(A, B)

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_str(components):
    matrix = Matrix(components)
    assert str(matrix) == (
        f"[{components[0][0]}, {components[0][1]}, {components[0][2]}]\n" + 
        f"[{components[1][0]}, {components[1][1]}, {components[1][2]}]\n" +
        f"[{components[2][0]}, {components[2][1]}, {components[2][2]}]]"
    )

@pytest.mark.parametrize([example_matrix_1, example_matrix_2, example_matrix_3])
def test_matrix_repr(components):
    matrix = Matrix(components)
    assert repr(matrix) == (
        "Matrix(" +
        f"[{components[0][0]}, {components[0][1]}, {components[0][2]}]," + 
        f"[{components[1][0]}, {components[1][1]}, {components[1][2]}]," +
        f"[{components[2][0]}, {components[2][1]}, {components[2][2]}]])"
    )

def test_matrix_identity():
    assert Matrix.identity() == np.eye(3)

def test_matrix_zeros():
    assert Matrix.zeros() == np.zeros((3, 3))


