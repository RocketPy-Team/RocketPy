import numpy as np
import pytest

from rocketpy.mathutils import Matrix

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
