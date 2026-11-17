Tube Fin Aerodynamics
=====================

RocketPy models a tube-fin set as a symmetric ring of uncanted ring wings, each
treated as if it were alone in the flow. The model started from the tube-fin
calculation of OpenRocket
(`TubeFinSetCalc <https://github.com/openrocket/openrocket/blob/unstable/core/src/main/java/info/openrocket/core/aerodynamics/barrowman/TubeFinSetCalc.java>`_).
The normal force and the center of pressure are different from OpenRocket's:
they follow the wind tunnel measurements of ring wings in NACA TN 4117 [2]_.

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

The aspect ratio of one tube is its inner diameter divided by its length:

.. math::

   A = \frac{2 r_i}{L}.

Normal Force
------------

Ribner [1]_ showed that a ring wing lifts twice as much as an elliptic flat
wing that spans its diameter and has the length of the ring as root chord.
That wing has the area and the aspect ratio

.. math::

   S_e = \frac{\pi r_i L}{2}, \qquad A_e = \frac{(2 r_i)^2}{S_e} = \frac{4 A}{\pi}.

Its lift slope is taken from Helmbold's equation [3]_, which holds for both
low and high aspect ratios:

.. math::

   a_e = \frac{2 \pi A_e}{2 + \sqrt{A_e^2 + 4}}.

For a rocket reference area :math:`A_{ref} = \pi R^2`, the normal-force
coefficient derivative of the complete tube set is

.. math::

   C_{N_\alpha} = \frac{n}{A_{ref}} \, 2 \, a_e \, S_e.

The normal force grows in proportion to the angle of attack,
:math:`C_N = C_{N_\alpha} \alpha`.

For a long tube this tends to the slender-body value, a normal force of
:math:`4 \pi r_i^2` per unit of dynamic pressure and angle of attack. For a
wide ring it tends to the lifting-line value, :math:`2 \pi^2 r_i L`.

The table compares the lift slope of one ring wing, per degree and based on
the area :math:`2 r_i L`, with the measurements of NACA TN 4117 (figure 11).
The last column is the lifting-line formula of [1]_, which OpenRocket uses and
which only holds for wide rings.

.. list-table::
   :header-rows: 1
   :widths: 16 20 22 22 20

   * - :math:`A`
     - Measured
     - RocketPy
     - Difference
     - Lifting line
   * - 1/3
     - 0.018
     - 0.0181
     - +0.4%
     - 0.0302
   * - 2/3
     - 0.034
     - 0.0350
     - +3.1%
     - 0.0513
   * - 1.0
     - 0.049
     - 0.0502
     - +2.4%
     - 0.0670
   * - 1.5
     - 0.069
     - 0.0690
     - +0.1%
     - 0.0841
   * - 3.0
     - 0.102
     - 0.1042
     - +2.2%
     - 0.1131

Center of Pressure
------------------

The center of pressure of a ring wing depends on its aspect ratio. RocketPy
uses the positions measured in NACA TN 4117 (figure 8), given from the tube
leading edge as a fraction of the tube length:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - :math:`A`
     - :math:`x_{CP} / L`
   * - 1/3
     - -0.11
   * - 2/3
     - 0.14
   * - 1.0
     - 0.20
   * - 1.5
     - 0.25
   * - 3.0
     - 0.35

Values in between are interpolated linearly. Below :math:`A = 1/3` and above
:math:`A = 3` the nearest measured value is used. A negative value means the
center of pressure is ahead of the leading edge, as it is on a slender body.

Roll Damping
------------

When the rocket rolls at the rate :math:`\omega`, the center of each tube
moves sideways at :math:`\omega (R + r_o)`. The tube sees this as an angle of
attack :math:`\omega (R + r_o) / V`, with :math:`V` the airspeed, and the
normal force it produces acts against the roll. For the complete set the roll
moment is

.. math::

   M_{roll} = -\frac{1}{2} \rho V A_{ref} C_{N_\alpha} (R + r_o)^2 \omega.

This is the same assumption OpenRocket makes. It has not been compared with
measurements.

Mach Number
-----------

The measurements behind the normal force and the center of pressure were made
at low speed. RocketPy uses both unchanged at every Mach number.

For the normal force this is a small approximation in subsonic flight. The
Prandtl-Glauert rule, which RocketPy applies to the fins, would raise the lift
slope of a tube with :math:`A = 1` by 2.0% at Mach 0.5 and by 5.6% at Mach 0.8.
For :math:`A = 1/3` the increase is 0.7% at Mach 0.8. The longer the tube, the
closer it is to a slender body, whose normal force does not depend on the Mach
number.

Above about Mach 0.8 the model should not be relied on. The flow through the
tubes and through the gaps between the tubes and the body changes in ways this
model does not describe.

OpenRocket also keeps the normal force independent of the Mach number. For the
center of pressure it uses the quarter of the tube length up to Mach 0.5 and
the rule of its flat fins above that. RocketPy does not, because that rule was
not derived for ring wings.

Model Limits
------------

- Each tube is treated as an isolated ring wing. The effect of the tubes on
  each other, and of the rocket body on the tubes, is not accounted for. The
  measurements used here were made on single ring wings with no body.
- There is no stall. In the measurements the normal force is proportional to
  the angle of attack up to about 16 to 20 degrees and levels off beyond that.
- Friction and pressure drag of the tubes are not calculated. Represent them
  in the rocket's power-on and power-off drag curves.
- Tube cant, separated tubes and overlapping tubes are not supported.

Use :class:`rocketpy.GenericSurface` when measured, wind-tunnel, or CFD
coefficients are available outside the limits above.

References
----------

.. [1] Ribner, H. S. "The Ring Airfoil in Nonaxial Flow." *Journal of the
   Aeronautical Sciences*, 14(9), 529--530, 1947.
   https://doi.org/10.2514/8.1437
.. [2] Fletcher, H. S. "Experimental Investigation of Lift, Drag, and Pitching
   Moment of Five Annular Airfoils." NACA Technical Note 4117, 1957.
   https://ntrs.nasa.gov/citations/19930084906
.. [3] Helmbold, H. B. "Der unverwundene Ellipsenflügel als tragende Fläche."
   *Jahrbuch 1942 der Deutschen Luftfahrtforschung*, I111--I113, 1942.
