.. _tipoff:

===========================================
Tip-off: the 3-DOF Single Rail Button Phase
===========================================

Introduction
------------

Between the moment the *upper* rail button leaves the launch rail and the moment
the *lower* button follows it, the rocket is still guided --- but only at one
point. It slides along the rail while free to pitch and yaw about that remaining
button. This short interval is what the literature calls **tip-off**, and it sets
the attitude and angular rate with which the rocket begins free flight.

This document derives the equations of motion used by
:meth:`rocketpy.Flight.udot_rail2`, the flight phase that models this interval.
It is enabled with ``Flight(..., use_udot_rail2=True)``; when disabled (the
default) the simulation transitions straight from the 1-DOF rail phase to the
generalized 6-DOF equations, exactly as it did before this phase existed.

The three flight phases around rail departure are, in order of the distance
``d`` travelled from the launch point:

.. math::

   \begin{aligned}
   \texttt{udot_rail1} \quad & \text{for } d < \ell_1
     && \text{(both buttons engaged, 1 DOF)} \\
   \texttt{udot_rail2} \quad & \text{for } \ell_1 \le d < \ell_2
     && \text{(upper button gone, lower engaged, 3 DOF)} \\
   \texttt{u_dot_generalized} \quad & \text{for } d \ge \ell_2
     && \text{(free flight, 6 DOF)}
   \end{aligned}

where :math:`\ell_1` and :math:`\ell_2` are the ``effective_1rl`` and
``effective_2rl`` attributes of :class:`rocketpy.Flight` --- the distances at
which the upper and the lower button reach the end of the rail. Their difference
is the button-to-button distance, so the phase has zero length for a rocket with
a single rail button and is skipped in that case.

Frames and conventions
----------------------

The solver integrates the same 13-element state vector used by the other
right-hand sides,

.. math::

   \mathbf{u} = [\,x,\ y,\ z,\ v_x,\ v_y,\ v_z,\ e_0,\ e_1,\ e_2,\ e_3,\
   \omega_1,\ \omega_2,\ \omega_3\,]

with :math:`\mathbf{r} = [x, y, z]` the inertial position of the **center of dry
mass** (CDM, the tracked point), :math:`\mathbf{v}` its inertial velocity,
:math:`\mathbf{e}` the attitude quaternion and :math:`\boldsymbol{\omega}` the
angular velocity in the **body** frame. The matrix
:math:`\mathbf{K} = \texttt{Matrix.transformation}(\mathbf{e})` rotates body
components into inertial ones, and :math:`\hat{\mathbf{z}}_b = [0, 0, 1]` is the
body roll (symmetry) axis.

The relevant mass geometry at time :math:`t`, in the body frame, is the total
mass :math:`m`, the CDM-to-center-of-mass offset :math:`\mathbf{r}_{CM}`, and the
inertia tensor about the CDM, :math:`\mathbf{I}`. Shifting the latter to the
instantaneous center of mass gives

.. math::

   \mathbf{I}_{CM} = \mathbf{I} - m\left(|\mathbf{r}_{CM}|^2 \mathbb{1}
   - \mathbf{r}_{CM}\mathbf{r}_{CM}^{\mathsf T}\right).

**Rail geometry.** The rail is a line fixed in the inertial frame, set by the
launch inclination and heading. Its unit vector is the ``attitude_unit``
attribute of :class:`rocketpy.Flight`; the constrained body point is the lower
rail button, at the fixed body position :math:`\mathbf{r}_{B}` relative to the
CDM.

The constraint
--------------

A free rigid body has six degrees of freedom. The single engaged button removes
three of them:

#. **The button stays on the rail line.** Its position may only vary along the
   rail, so the component of its acceleration perpendicular to the rail
   vanishes. That is **two** scalar constraints.
#. **Roll is suppressed** by the button in its rail slot:
   :math:`\dot{\boldsymbol{\omega}} \cdot \hat{\mathbf{z}}_b = 0`. That is
   **one** more.

Three constraints leave :math:`6 - 3 = 3` degrees of freedom: translation along
the rail, pitch and yaw --- the 3 DOF this phase is named for.

The forces that enforce them are the unknowns of the problem:

* a **normal reaction** at the button, perpendicular to the rail because a
  frictionless slot can neither pull nor push along it. Writing an orthonormal
  body triad :math:`\{\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{n}}_b\}`
  around the body-frame rail direction
  :math:`\hat{\mathbf{n}}_b = \mathbf{K}^{\mathsf T}\hat{\mathbf{n}}`, it has two
  components: :math:`\mathbf{N}_b = \lambda_1 \hat{\mathbf{e}}_1 + \lambda_2
  \hat{\mathbf{e}}_2`;
* a **roll reaction moment** :math:`\mu \hat{\mathbf{z}}_b` about the body axis.

Three unknowns, three constraints: a :math:`3\times3` linear system, solved once
per evaluation of the right-hand side.

The initial conditions are consistent with the constraint. The phase is entered
from ``udot_rail1``, where the rocket has no angular velocity and its velocity is
along the rail, so the button's perpendicular *velocity* is already zero.
Enforcing zero perpendicular *acceleration* therefore keeps it on the rail.

Augmented dynamics
------------------

The generalized equations of motion assemble a total force
:math:`\mathbf{T}_{20}` and a total moment about the CDM :math:`\mathbf{T}_{21}`,
both in the body frame, and solve

.. math::

   \dot{\boldsymbol{\omega}}_{\text{free}} = \mathbf{I}_{CM}^{-1}
   \left(\mathbf{T}_{21} + \mathbf{T}_{20} \times \mathbf{r}_{CM}\right),
   \qquad
   \mathbf{a}_{\text{free}} = \frac{\mathbf{T}_{20}}{m}
   - \mathbf{r}_{CM} \times \dot{\boldsymbol{\omega}}_{\text{free}}.

Both totals are *sums of external contributions*, which is what makes the
constraint clean to add: the reaction wrench simply enters the sums,

.. math::

   \mathbf{T}_{20}' = \mathbf{T}_{20} + \mathbf{N}_b,
   \qquad
   \mathbf{T}_{21}' = \mathbf{T}_{21} + \mathbf{r}_{B} \times \mathbf{N}_b
   + \mu \hat{\mathbf{z}}_b,

after which the same two lines apply. Since the solve is linear in the totals,
the result splits into the free solution plus a response to the unknowns. Using
:math:`\mathbf{r}_B \times \mathbf{N} + \mathbf{N} \times \mathbf{r}_{CM}
= (\mathbf{r}_B - \mathbf{r}_{CM}) \times \mathbf{N}` and writing
:math:`\mathbf{d} = \mathbf{r}_{B} - \mathbf{r}_{CM}` for the center-of-mass-to-button
vector,

.. math::

   \Delta\dot{\boldsymbol{\omega}} = \mathbf{I}_{CM}^{-1}
   \left(\mathbf{d} \times \mathbf{N}_b + \mu \hat{\mathbf{z}}_b\right),
   \qquad
   \Delta\mathbf{a} = \frac{\mathbf{N}_b}{m}
   - \mathbf{r}_{CM} \times \Delta\dot{\boldsymbol{\omega}}.

The button is body-fixed, so its acceleration in body components is

.. math::

   \mathbf{A} = \mathbf{A}_{\text{free}} + \Delta\mathbf{a}
   + \Delta\dot{\boldsymbol{\omega}} \times \mathbf{r}_{B},
   \qquad
   \mathbf{A}_{\text{free}} = \mathbf{K}^{\mathsf T}\mathbf{a}_{\text{free}}
   + \dot{\boldsymbol{\omega}}_{\text{free}} \times \mathbf{r}_{B}
   + \boldsymbol{\omega} \times
   \left(\boldsymbol{\omega} \times \mathbf{r}_{B}\right).

The linear solve
----------------

Collect the unknowns in :math:`\boldsymbol{\chi} = [\lambda_1, \lambda_2, \mu]`.
The three constraints read

.. math::

   \mathbf{A} \cdot \hat{\mathbf{e}}_1 = 0,
   \qquad
   \mathbf{A} \cdot \hat{\mathbf{e}}_2 = 0,
   \qquad
   \dot{\boldsymbol{\omega}} \cdot \hat{\mathbf{z}}_b = 0,

and because :math:`\mathbf{A}` and :math:`\dot{\boldsymbol{\omega}}` are linear
in :math:`\boldsymbol{\chi}`, they form the system
:math:`\mathbf{J}\boldsymbol{\chi} = -\mathbf{g}_{\text{free}}` with

.. math::

   \mathbf{g}_{\text{free}} =
   \begin{bmatrix}
   \mathbf{A}_{\text{free}} \cdot \hat{\mathbf{e}}_1 \\
   \mathbf{A}_{\text{free}} \cdot \hat{\mathbf{e}}_2 \\
   \dot{\boldsymbol{\omega}}_{\text{free}} \cdot \hat{\mathbf{z}}_b
   \end{bmatrix}.

Each column of :math:`\mathbf{J}` is obtained by evaluating the response above at
one of the three unit inputs :math:`(\mathbf{N}_b, \mu) =
(\hat{\mathbf{e}}_1, 0)`, :math:`(\hat{\mathbf{e}}_2, 0)`,
:math:`(\mathbf{0}, 1)`. Solving for :math:`\boldsymbol{\chi}` and substituting
back gives the constrained accelerations, which override the free ones in the
returned derivative:

.. math::

   \dot{\boldsymbol{\omega}} = \dot{\boldsymbol{\omega}}_{\text{free}}
   + \Delta\dot{\boldsymbol{\omega}},
   \qquad
   \mathbf{a}_{CDM} = \mathbf{a}_{\text{free}}
   + \mathbf{K}\,\Delta\mathbf{a}.

The position and quaternion derivatives are the ordinary kinematic ones. In
particular :math:`\dot{\mathbf{r}} = \mathbf{v}`: the velocity is **not**
projected onto the rail. The constraint acts at the acceleration level on the
*button*, and the CDM legitimately acquires a small perpendicular velocity as the
rocket pitches about that button.

Modelling assumptions and edge cases
------------------------------------

* **Roll axis.** The constraint suppresses roll about the *body* axis. The rocket
  travels only the button-to-button distance during this phase, so the tip-off
  angle is small and the body axis stays close to the rail direction; the
  difference between body roll and rail-axis roll is of that order. This is
  consistent with ``udot_rail1``, which freezes rotation entirely.
* **Radial button offset.** The button is modelled on the rocket axis. The roll
  constraint is enforced explicitly by :math:`\mu`, so what is lost is only the
  (small) roll coupling through the button's radial standoff.
* **Zero-length phase.** A rocket with a single rail button has
  :math:`\ell_1 = \ell_2`, and the phase is skipped.
* **Singular system.** Should the geometry make :math:`\mathbf{J}` singular, the
  step falls back to the unconstrained dynamics and warns.

Expected behaviour
------------------

The phase reproduces the two effects tip-off is modelled for. With no wind, the
center of mass sits ahead of the button that the rocket now pivots about, so
gravity pitches the nose down by a fraction of a degree. With a crosswind, the
aerodynamic moment turns the rocket into the wind before it is fully free ---
the weathercock effect --- and the rocket therefore leaves the rail with a small
angular rate rather than none.

References
----------

The tip-off phase and its effect on the initial conditions of free flight are
treated in the launcher dynamics literature: [Chou]_ studies a vehicle moving
along an inclined guideway with dynamic interactions, and [Hosken]_ analyses
tip-off effects in rail launchers.
