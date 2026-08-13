Tube Fin Aerodynamics
=====================

RocketPy models a tube-fin set as a symmetric ring of uncanted ring airfoils.
The implementation follows the preliminary tube-fin model in OpenRocket's
`TubeFinSetCalc <https://github.com/openrocket/openrocket/blob/unstable/core/src/main/java/info/openrocket/core/aerodynamics/barrowman/TubeFinSetCalc.java>`_.
The normal-force derivative is based on Ribner's analysis of a ring airfoil in
nonaxial flow [1]_.

Geometry
--------

Let :math:`n` be the number of tubes, :math:`R` the rocket-body radius,
:math:`r_i` the tube inner radius, :math:`r_o` the tube outer radius, and
:math:`L` the tube length. The tubes are distributed evenly around the rocket.
The current implementation requires each tube to touch its two neighbors:

.. math::

   r_o = R \frac{\sin(\pi / n)}{1 - \sin(\pi / n)}.

This constraint also places each tube against the rocket body. Configurations
with fewer than three tubes, gaps between tubes, or overlapping tubes are
rejected.

Normal Force
------------

The ring-airfoil aspect ratio and its modified form are

.. math::

   AR = \frac{2 r_i}{L}, \qquad AR' = \frac{2 AR}{\pi}.

For a rocket reference area :math:`A_{ref} = \pi R^2`, the normal-force
coefficient derivative of the complete tube set is

.. math::

   C_{N_\alpha} =
   \frac{n}{A_{ref}}
   2 \left(\frac{AR'}{1 + AR'}\right) \pi^2 r_i L.

RocketPy applies this derivative symmetrically for positive and negative
angles of attack and caps the magnitude at 20 degrees:

.. math::

   C_N(\alpha) = C_{N_\alpha}
   \operatorname{clip}\left(\alpha, -20^\circ, 20^\circ\right).

Center of Pressure
------------------

For Mach numbers up to 0.5, the center of pressure is placed at the quarter
chord:

.. math::

   x_{CP} = \frac{L}{4}.

The position is measured from the tube leading edge. OpenRocket moves this
position with Mach number above Mach 0.5; RocketPy does not yet implement that
correction. Simulations that exceed Mach 0.5 should use aerodynamic data from a
higher-fidelity source instead of this fixed-CP model.

Model Limits
------------

The tube-fin surface contributes normal force and the corresponding pitch and
yaw moments about its center of pressure. It does not calculate:

- friction or pressure drag from the tubes;
- roll forcing or damping;
- side-force or yaw behavior for asymmetric tube layouts;
- tube cant; or
- aerodynamic corrections for separated or overlapping tubes.

Represent tube-fin drag in the rocket's power-on and power-off drag curves.
Use :class:`rocketpy.GenericSurface` when measured, wind-tunnel, or CFD
coefficients are available outside the limits above.

References
----------

.. [1] Ribner, H. S. "The Ring Airfoil in Nonaxial Flow." *Journal of the
   Aeronautical Sciences*, 14(9), 529--530, 1947.
