"""Utilities related to inertia tensor transformations.

This module centralizes dynamic helpers for applying the parallel axis
 theorem (PAT). It lives inside ``rocketpy.mathutils`` so that functionality
 depending on :class:`rocketpy.mathutils.function.Function` does not leak into
 generic utility modules such as ``rocketpy.tools``.
"""

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Vector


def _pat_dynamic_helper(com_inertia_moment, mass, distance_vec_3d, axes_term_lambda):
    """Apply the PAT to inertia moments, supporting static and dynamic inputs."""

    is_dynamic = (
        isinstance(com_inertia_moment, Function)
        or isinstance(mass, Function)
        or isinstance(distance_vec_3d, Function)
    )

    def get_val(arg, t):
        return arg(t) if isinstance(arg, Function) else arg

    if not is_dynamic:
        d_vec = Vector(distance_vec_3d)
        mass_term = mass * axes_term_lambda(d_vec)
        return com_inertia_moment + mass_term

    def new_source(t):
        d_vec_t = get_val(distance_vec_3d, t)
        mass_t = get_val(mass, t)
        inertia_t = get_val(com_inertia_moment, t)
        mass_term = mass_t * axes_term_lambda(d_vec_t)
        return inertia_t + mass_term

    return Function(new_source, inputs="t", outputs="Inertia (kg*m^2)")


def _pat_dynamic_product_helper(
    com_inertia_product, mass, distance_vec_3d, product_term_lambda
):
    """Apply the PAT to inertia products, supporting static and dynamic inputs."""

    is_dynamic = (
        isinstance(com_inertia_product, Function)
        or isinstance(mass, Function)
        or isinstance(distance_vec_3d, Function)
    )

    def get_val(arg, t):
        return arg(t) if isinstance(arg, Function) else arg

    if not is_dynamic:
        d_vec = Vector(distance_vec_3d)
        mass_term = mass * product_term_lambda(d_vec)
        return com_inertia_product + mass_term

    def new_source(t):
        d_vec_t = get_val(distance_vec_3d, t)
        mass_t = get_val(mass, t)
        inertia_t = get_val(com_inertia_product, t)
        mass_term = mass_t * product_term_lambda(d_vec_t)
        return inertia_t + mass_term

    return Function(new_source, inputs="t", outputs="Inertia (kg*m^2)")


# --- Public functions for the Parallel Axis Theorem ---


def parallel_axis_theorem_I11(com_inertia_moment, mass, distance_vec_3d):
    """Apply PAT to the I11 inertia term.

    Parameters
    ----------
    com_inertia_moment : float or Function
        Inertia moment relative to the component center of mass.
    mass : float or Function
        Mass of the component. If a Function, it must map time to mass.
    distance_vec_3d : array-like or Function
        Displacement vector from the component COM to the reference COM.

    Returns
    -------
    float or Function
        Updated I11 value referenced to the new axis.
    """

    return _pat_dynamic_helper(
        com_inertia_moment, mass, distance_vec_3d, lambda d_vec: d_vec.y**2 + d_vec.z**2
    )


def parallel_axis_theorem_I22(com_inertia_moment, mass, distance_vec_3d):
    """Apply PAT to the I22 inertia term.

    Parameters
    ----------
    com_inertia_moment : float or Function
        Inertia moment relative to the component center of mass.
    mass : float or Function
        Mass of the component. If a Function, it must map time to mass.
    distance_vec_3d : array-like or Function
        Displacement vector from the component COM to the reference COM.

    Returns
    -------
    float or Function
        Updated I22 value referenced to the new axis.
    """

    return _pat_dynamic_helper(
        com_inertia_moment, mass, distance_vec_3d, lambda d_vec: d_vec.x**2 + d_vec.z**2
    )


def parallel_axis_theorem_I33(com_inertia_moment, mass, distance_vec_3d):
    """Apply PAT to the I33 inertia term.

    Parameters
    ----------
    com_inertia_moment : float or Function
        Inertia moment relative to the component center of mass.
    mass : float or Function
        Mass of the component. If a Function, it must map time to mass.
    distance_vec_3d : array-like or Function
        Displacement vector from the component COM to the reference COM.

    Returns
    -------
    float or Function
        Updated I33 value referenced to the new axis.
    """

    return _pat_dynamic_helper(
        com_inertia_moment, mass, distance_vec_3d, lambda d_vec: d_vec.x**2 + d_vec.y**2
    )


def parallel_axis_theorem_I12(com_inertia_product, mass, distance_vec_3d):
    """Apply PAT to the I12 inertia product.

    Parameters
    ----------
    com_inertia_product : float or Function
        Product of inertia relative to the component center of mass.
    mass : float or Function
        Mass of the component. If a Function, it must map time to mass.
    distance_vec_3d : array-like or Function
        Displacement vector from the component COM to the reference COM.

    Returns
    -------
    float or Function
        Updated I12 value referenced to the new axis.
    """

    return _pat_dynamic_product_helper(
        com_inertia_product, mass, distance_vec_3d, lambda d_vec: d_vec.x * d_vec.y
    )


def parallel_axis_theorem_I13(com_inertia_product, mass, distance_vec_3d):
    """Apply PAT to the I13 inertia product.

    Parameters
    ----------
    com_inertia_product : float or Function
        Product of inertia relative to the component center of mass.
    mass : float or Function
        Mass of the component. If a Function, it must map time to mass.
    distance_vec_3d : array-like or Function
        Displacement vector from the component COM to the reference COM.

    Returns
    -------
    float or Function
        Updated I13 value referenced to the new axis.
    """

    return _pat_dynamic_product_helper(
        com_inertia_product, mass, distance_vec_3d, lambda d_vec: d_vec.x * d_vec.z
    )


def parallel_axis_theorem_I23(com_inertia_product, mass, distance_vec_3d):
    """Apply PAT to the I23 inertia product.

    Parameters
    ----------
    com_inertia_product : float or Function
        Product of inertia relative to the component center of mass.
    mass : float or Function
        Mass of the component. If a Function, it must map time to mass.
    distance_vec_3d : array-like or Function
        Displacement vector from the component COM to the reference COM.

    Returns
    -------
    float or Function
        Updated I23 value referenced to the new axis.
    """

    return _pat_dynamic_product_helper(
        com_inertia_product, mass, distance_vec_3d, lambda d_vec: d_vec.y * d_vec.z
    )


__all__ = [
    "parallel_axis_theorem_I11",
    "parallel_axis_theorem_I22",
    "parallel_axis_theorem_I33",
    "parallel_axis_theorem_I12",
    "parallel_axis_theorem_I13",
    "parallel_axis_theorem_I23",
]
