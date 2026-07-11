.. _aero_cp_stability:

==========================================================
Aerodynamics: Coefficients, Centers and Stability
==========================================================

:Author: RocketPy Team
:Date: June 2026

Introduction
============

This document describes how RocketPy models aerodynamic forces and moments and
how the rocket's stability quantities — the **aerodynamic center**, the
**center of pressure**, the **static** and **stability margins**, and the
**dynamic-stability** parameters — are derived from them.

The model is built as a strict set of layers, each derived only from the one
below it:

#. **Surface coefficients** — every aerodynamic surface exposes the six
   dimensionless aerodynamic coefficients.
#. **Rocket aggregate** — the surfaces are summed into the rocket's total force
   and moment.
#. **Stability references** — the *aerodynamic center* (linear) and the
   *center of pressure* (nonlinear).
#. **Margins** — the linear (aerodynamic-center) and realized (center-of-
   pressure) stability margins.
#. **Dynamic stability** — the linearized attitude oscillator.

Since the aerodynamic-surface refactor, :class:`rocketpy.GenericSurface` is the
**root of the aerodynamic-surface hierarchy**: nose cones, fin sets, individual
fins, tails/transitions and air brakes are all described by the same coefficient
set and computed through a single coefficient-based force-and-moment model. The
geometric (Barrowman) surfaces translate their geometry into those same
coefficients (see :ref:`barrowman_mapping`), so the rocket knows the full
aerodynamic coefficient set for every surface.

.. note::
   The legacy ``AeroSurface`` base class is deprecated. It is retained only as a
   compatibility shim: :class:`rocketpy.GenericSurface` is registered as a
   virtual subclass, so ``isinstance(surface, AeroSurface)`` still returns
   ``True``.

Layer 0 — The aerodynamic coefficient model
============================================

A generic aerodynamic surface is defined by six dimensionless coefficients,
each a function of a set of independent variables:

- force coefficients: lift :math:`C_L`, side force :math:`C_Q`, drag :math:`C_D`;
- moment coefficients: pitch :math:`C_m`, yaw :math:`C_n`, roll :math:`C_l`.

The standard independent variables are the angle of attack :math:`\alpha`, the
sideslip angle :math:`\beta`, the Mach number :math:`M`, the Reynolds number
:math:`Re`, and the body angular rates (pitch :math:`q`, yaw :math:`r`, roll
:math:`p`):

.. math::

   C_i = C_i(\alpha,\ \beta,\ M,\ Re,\ q,\ r,\ p)

Subclasses may append extra axes — control deflections for
:class:`rocketpy.ControllableGenericSurface`, or the unsteady terms
:math:`\dot\alpha,\ \dot\beta` when ``unsteady_aero=True``.

Forces and moments of a surface
-------------------------------

At each step the surface receives the freestream velocity in the body frame.
Reversing it into the standard aerodynamic frame, the incidence angles are

.. math::

   \alpha = \operatorname{atan2}(-v_y,\ -v_z), \qquad
   \beta  = \operatorname{atan2}(-v_x,\ -v_z)

With the dynamic pressure times reference area
:math:`\bar q A = \tfrac{1}{2}\rho V^2 A_\text{ref}`, the aerodynamic force
:math:`(Q, -L, -D)` is rotated from the aerodynamic frame into the body frame,
giving :math:`\mathbf{R}=(R_1, R_2, R_3)`, and the moment about the rocket's
center of dry mass is

.. math::
   :label: moment_transport

   \mathbf{M} = \bar q A L_\text{ref}\,(C_m, C_n, C_l)
              + \mathbf{r}_\text{cp} \times \mathbf{R}

The first term is the couple carried by the moment coefficients; the second
transports the resultant force from its application point
:math:`\mathbf{r}_\text{cp}` to the center of dry mass. This is implemented in
:meth:`rocketpy.GenericSurface.compute_forces_and_moments`.

Layer 1 — Rocket aggregate
==========================

The simulation, and every stability quantity below, sums the surfaces into the
rocket's total body-frame force and moment about the center of dry mass. The
nonlinear aggregate at a given state is
:meth:`rocketpy.Rocket._aerodynamic_forces_and_moments`; the dimensionless
totals are exposed by :meth:`rocketpy.Rocket.aerodynamic_coefficients` (total
normal-force coefficient :math:`C_N` and pitch-moment coefficient :math:`C_m`).
The **linear** aggregate — the normal-force-curve slope and the
slope-weighted positions — is built by
:meth:`rocketpy.Rocket.evaluate_center_of_pressure` (see Layer 2).

Layer 2 — Aerodynamic center vs. center of pressure
===================================================

These two are the heart of the model and are frequently confused. They are the
same physics in two regimes.

Aerodynamic center (linear)
---------------------------

The **aerodynamic center** (AC) is the *linearized*, small-incidence
(:math:`\alpha=\beta=0`) location about which the pitching moment is independent
of angle of attack:

.. math::
   :label: ac

   x_\text{AC}(M) = x_\text{ref}
       - \frac{\partial C_m/\partial\alpha}{\partial C_N/\partial\alpha}\,L_\text{ref}

It is well-conditioned, a function of Mach alone, and is the classical reference
that the static margin is built on. At the rocket level it is the
normal-force-slope-weighted average of the component locations,

.. math::
   :label: rocket_ac

   x_\text{AC,rocket}(M) =
     \frac{\sum_i k_i\, C_{N,\alpha,i}(M)\,\big(p_i - c\, z_{\text{cp},i}(M)\big)}
          {\sum_i k_i\, C_{N,\alpha,i}(M)}

with the area-correction factor :math:`k_i = A_{\text{ref},i}/A_\text{rocket}`,
:math:`p_i` the surface position and :math:`c=\pm 1` the coordinate-system
orientation. Because the weight is the normal-force slope, a zero-lift surface
(e.g. a pure-drag element) drops out cleanly. This is computed by
:meth:`rocketpy.Rocket.evaluate_center_of_pressure` and stored as
``Rocket.aerodynamic_center``.

.. note::
   ``Rocket.cp_position`` is an **alias** for ``Rocket.aerodynamic_center``. The
   historical "center of pressure" attribute was always the aerodynamic center;
   the alias is kept for backward compatibility and convenience.

Center of pressure (nonlinear)
------------------------------

The **center of pressure** (CP) is the point at which the *actual* resultant
aerodynamic force acts with no residual moment, at a finite angle of
attack/sideslip:

.. math::
   :label: cp

   x_\text{CP}(\alpha,\beta,M,Re) =
     x_\text{cdm} + c\,\frac{M_2 R_1 - M_1 R_2}{R_1^2 + R_2^2}

evaluated from the Layer-1 aggregate (:math:`M = r\times F`). Equivalently, per
plane, :math:`x_\text{CP} = x_\text{cdm} + c\,L_\text{ref}\,C_m/C_L` (pitch) and
:math:`+\,c\,L_\text{ref}\,C_n/C_Q` (yaw). Unlike the AC, the CP **moves with
incidence**.

The CP is a **genuinely partial quantity**: it is a :math:`0/0` limit at zero
incidence (the normal force vanishes), undefined there, and converges to the AC
as :math:`\alpha,\beta \to 0`. Because it plays no role in the equations of
motion and the stability margins are correctly built on the AC (a slope, see
Layer 3), RocketPy does **not** expose it as a dedicated method — that would
force an arbitrary regularization of a real singularity. When the
force-application CP is genuinely wanted (e.g. comparing against wind-tunnel or
CFD CP-vs-:math:`\alpha` data), it is reconstructed on demand from the aggregate
coefficients :meth:`rocketpy.Rocket.aerodynamic_coefficients_full` using the
relation above, with the caller deciding how to treat the zero-incidence limit.

Pitch and yaw planes
--------------------

Because :class:`rocketpy.GenericSurface` allows **non-axisymmetric** rockets, the
*linear* AC is computed independently for the two planes:

- pitch (``aerodynamic_center``) from :math:`\partial C_L/\partial\alpha` and
  :math:`C_m`;
- yaw (``aerodynamic_center_yaw``) from the side-force slope and :math:`C_n`.

They coincide for an axisymmetric rocket; ``Rocket.is_axisymmetric`` reports
whether they agree (to caliber tolerance) and
:meth:`rocketpy.Rocket.evaluate_center_of_pressure` warns when they do not, since
the scalar ``static_margin``/``stability_margin`` then describe the pitch plane
only, and the ``*_yaw`` counterparts expose the yaw plane.

Layer 3 — Static and stability margins
======================================

A margin is the longitudinal center-of-mass-to-stability-reference distance in
calibers (rocket diameters). With the center of mass :math:`z_\text{cm}(t)`, the
rocket radius :math:`R` and the orientation factor :math:`c`, there are **two
co-equal families**:

**Linear (aerodynamic-center) margins.** Built on the AC; well-conditioned and
never spiking. The conventional design parameters:

.. math::
   :label: static_margin

   \text{static margin}(t) = c\,\frac{z_\text{cm}(t) - x_\text{AC}(0)}{2R},
   \qquad
   \text{stability margin}(M, t) = c\,\frac{z_\text{cm}(t) - x_\text{AC}(M)}{2R}

The static margin (:meth:`rocketpy.Rocket.evaluate_static_margin`) is the
incompressible (:math:`M=0`) limit, a function of time; the stability margin
(:meth:`rocketpy.Rocket.evaluate_stability_margin`) is a function of Mach and
time. The ``*_yaw`` counterparts use ``aerodynamic_center_yaw``.

At the :class:`rocketpy.Flight` level, ``Flight.stability_margin`` (and
``stability_margin_yaw``) evaluates the linear margin along the realized Mach and
time — smooth, conventional, and the source of ``initial_stability_margin`` /
``out_of_rail_stability_margin`` / ``min_stability_margin`` /
``max_stability_margin``.

A positive margin (stability reference behind the center of mass) is the classic
passive-stability condition.

.. note::
   **Nonlinear (large-incidence) static stability.** For tabulated
   :class:`rocketpy.GenericSurface` coefficients that are nonlinear in
   :math:`\alpha`, the stability reference -- the *local neutral point*, the AC
   re-linearized at the flown incidence -- migrates with angle of attack,

   .. math::

      x_\text{NP}(\alpha,\beta,M) = x_\text{cdm}
        + c\,L_\text{ref}\,\frac{\partial C_m/\partial\alpha}
                                {\partial C_L/\partial\alpha},

   which (unlike the singular force-application CP :math:`-C_m/C_N`) is well
   conditioned at every incidence and isolates the stability-relevant part of
   the CP travel. It is reconstructed on demand from
   :meth:`rocketpy.Rocket.aerodynamic_coefficients_full` by a central finite
   difference in :math:`\alpha`. For linear Barrowman aerodynamics it reduces to
   the :math:`\alpha=0` AC, so the linear margin already captures it; only
   nonlinear tabulated coefficients make it move. Large-incidence stability is
   usually read more meaningfully from the dynamic-stability coefficients
   (Layer 4).

Layer 4 — Dynamic stability
===========================

A static margin only gives the *sign* of the restoring moment. The actual
attitude response is the linearized pitch (or yaw) oscillator

.. math::

   I_L\,\ddot\theta + C_2\,\dot\theta + C_1\,\theta = 0

with the **corrective moment coefficient** (restoring moment per radian),

.. math::

   C_1 = \bar q\, A_\text{ref}\, C_{N,\alpha}\, (z_\text{cm} - x_\text{AC}),

the **damping moment coefficient** (aerodynamic plus jet damping),

.. math::

   C_2 = \tfrac{1}{2}\rho V A_\text{ref} \sum_i k_i\,C_{N,\alpha,i}\,(x_i - z_\text{cm})^2
       \;+\; \dot m\,(x_\text{nozzle} - z_\text{cm})^2,

and the lateral moment of inertia about the instantaneous center of mass
:math:`I_L`. From these,

.. math::

   \omega_n = \sqrt{C_1/I_L}, \qquad \zeta = \frac{C_2}{2\sqrt{C_1\,I_L}}.

These are exposed on :class:`rocketpy.Flight` as
``corrective_moment_coefficient``, ``damping_moment_coefficient``,
``pitch_natural_frequency``, ``pitch_damping_ratio`` and the ``yaw_*``
counterparts. :math:`\zeta < 1` is an underdamped (oscillatory) response;
RocketPy also exposes the empirical FFT ``attitude_frequency_response`` as a
cross-check.

.. note::
   **Roll has no natural frequency.** A conventional rocket has no aerodynamic
   roll-restoring moment, so roll is *neutrally stable* (a first-order system:
   fin-cant forcing balanced by roll damping, spinning up to a steady rate).
   The roll-pitch/yaw coupling of concern is **roll resonance** ("roll
   lock-in"): when the roll rate crosses the pitch/yaw natural frequency, the
   spin couples into the attitude oscillation and the amplitude can diverge.
   ``Flight.plots.dynamic_stability_data`` therefore overlays the roll rate (as
   a frequency) on the natural-frequency plot — the crossings are the points to
   watch.

Quick reference
===============

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Attribute
     - Variables
     - Meaning
   * - ``Rocket.aerodynamic_center`` (``_yaw``)
     - :math:`M`
     - Linear (small-incidence) center of pressure; static-margin reference.
       ``cp_position`` is an alias.
   * - ``Rocket.aerodynamic_coefficients(α, β, M, Re)``
     - :math:`\alpha,\beta,M,Re`
     - Total :math:`C_N`, :math:`C_m` about the center of dry mass.
   * - ``Rocket.aerodynamic_coefficients_full(α, β, M, Re)``
     - :math:`\alpha,\beta,M,Re`
     - Six signed coefficients; reconstruct the nonlinear CP as
       :math:`x_\text{cdm} + c\,L_\text{ref}\,C_m/C_L`.
   * - ``Rocket.static_margin`` (``_yaw``)
     - :math:`t`
     - Linear margin at :math:`M=0` (calibers).
   * - ``Rocket.stability_margin`` (``_yaw``)
     - :math:`M, t`
     - Linear margin vs Mach and time (calibers).
   * - ``Flight.stability_margin`` (``_yaw``)
     - :math:`t`
     - Linear margin along the realized Mach(t) — smooth.
   * - ``Flight.{pitch,yaw}_natural_frequency``
     - :math:`t`
     - Attitude oscillation natural frequency :math:`\omega_n`.
   * - ``Flight.{pitch,yaw}_damping_ratio``
     - :math:`t`
     - Attitude oscillation damping ratio :math:`\zeta`.
   * - ``Flight.{corrective,damping}_moment_coefficient``
     - :math:`t`
     - Oscillator coefficients :math:`C_1`, :math:`C_2`.

Visualizing stability
=====================

- ``Rocket.plots.stability_margin`` — linear margin vs Mach and time (surface).
- ``Rocket.plots.aerodynamic_coefficients`` — :math:`C_N`, :math:`C_m` vs
  :math:`\alpha`; ``drag_curves`` for :math:`C_D` vs Mach.
- ``Flight.plots.stability_and_control_data`` — linear margin (pitch and yaw) vs
  time, plus the FFT frequency response.
- ``Flight.plots.dynamic_stability_data`` — natural frequency and damping ratio
  vs time (pitch and yaw).

For non-axisymmetric rockets, ``Rocket.plots.all`` / ``Rocket.all_info`` also
draw both the pitch (``xz``) and yaw (``yz``) planes and the yaw-plane margins.

.. _barrowman_mapping:

Mapping Barrowman surfaces to coefficients
==========================================

The geometric surfaces expose a lift-curve slope :math:`C_{N,\alpha}(M)`
(``clalpha``), a geometric cp :math:`z_\text{cp}` and — for fins — roll
forcing/damping. These are translated into the linear coefficient model:

.. math::

   C_{L,\alpha} = C_{N,\alpha}, \qquad
   C_{Q,\beta}  = -C_{N,\alpha}

.. math::

   C_{m,\alpha} = -C_{N,\alpha}\,\frac{z_\text{cp}}{L_\text{ref}}, \qquad
   C_{n,\beta}  = +C_{N,\alpha}\,\frac{z_\text{cp}}{L_\text{ref}}

For an **individual fin** at angular position :math:`\phi`, the lift only resists
incidence in its own plane, so its slope is projected onto the two planes —
:math:`\sin^2\phi` to the pitch plane and :math:`\cos^2\phi` to the yaw plane.
An evenly spaced set of :math:`n` fins sums to :math:`n/2` in each plane,
reproducing the axisymmetric fin-set result; a one-plane layout (e.g. canards at
:math:`0^\circ/180^\circ`) makes the pitch- and yaw-plane aerodynamic centers
differ.

For fin sets, the cant-angle roll forcing and roll damping add

.. math::

   C_{l,0} = C_{lf,\delta}(M)\,\delta, \qquad
   C_{l,p} = C_{ld,\omega}(M)

where :math:`\delta` is the cant angle. With this mapping the geometric surfaces
reproduce the Barrowman lift and roll behavior while flowing through the same
generic coefficient path as every other surface.

.. note::
   The independent :math:`\alpha,\ \beta` decomposition of the linear model
   coincides with the classical single-plane Barrowman projection to first
   order and diverges only at large combined angle of attack, where the
   underlying linear coefficients are themselves no longer valid; that regime is
   captured by tabulated :class:`rocketpy.GenericSurface` coefficients, from
   which the local neutral point and the nonlinear CP can be reconstructed via
   :meth:`rocketpy.Rocket.aerodynamic_coefficients_full`.

References
==========

The Barrowman method and its coefficients are described in [Barrowman]_ and
[Niskanen]_. The dynamic-stability oscillator (corrective and damping moment
coefficients, natural frequency and damping ratio) follows [Niskanen]_. See also
the :ref:`individual_fins` and roll-moment technical documents for the fin
derivations.
