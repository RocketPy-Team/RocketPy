import numpy as np
import pytest

from rocketpy.mathutils import Vector

test_vector_1 = [1, 2, 3]
test_vector_2 = [-np.pi, 1, np.e]
test_vector_3 = [3 * 1j, -2j, 0j]
test_vectors = [test_vector_1, test_vector_2, test_vector_3]


@pytest.mark.parametrize("vector_components", test_vectors)
def test_vector_x_y_z(vector_components):
    vector = Vector(vector_components)
    assert vector.x == vector_components[0]
    assert vector.y == vector_components[1]
    assert vector.z == vector_components[2]
