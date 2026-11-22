.. _tipoff:

===========================================
Tip-off: the 3-DOF Single Rail Button Phase
===========================================

Introduction
------------

Between the moment the upper rail button leaves the launch rail and the moment
the lower button follows it, the rocket is guided at one point only. It slides
along the rail while free to pitch and yaw about the remaining button. This
short interval is called **tip-off**, and it sets the attitude and angular rate
with which the rocket begins free flight.

This document derives the equations of motion used by
:meth:`rocketpy.Flight.udot_rail2`, the flight phase that models this interval.
The phase runs whenever the rocket has two rail buttons.

The phase adds a constraint wrench (a force and a moment), built from the full
inertia tensor, to the generalized 6-DOF solution. It therefore needs
``simulation_mode="6 DOF"`` and ``equations_of_motion="standard"``. The other
formulations do not have the state this correction needs: the 3 DOF equations
model no attitude, and ``solid_propulsion`` uses a different, axisymmetric set.
Flights that use either of them skip the phase and go straight from the 1-DOF
rail phase to free flight, and so does a flight with a point-mass motor, which
forces ``simulation_mode`` to ``"3 DOF"``.

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
``effective_2rl`` attributes of :class:`rocketpy.Flight`, the distances at
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

with :math:`\mathbf{r} = [x, y, z]` the inertial position of the center of dry
mass (CDM, the tracked point), :math:`\mathbf{v}` its inertial velocity,
:math:`\mathbf{e}` the attitude quaternion and :math:`\boldsymbol{\omega}` the
angular velocity in the body frame. The matrix
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

The rail is a line fixed in the inertial frame, set by the launch inclination
and heading. Its unit vector is the ``attitude_unit`` attribute of
:class:`rocketpy.Flight`. The constrained body point is the lower rail button,
at the fixed body position :math:`\mathbf{r}_{B}` relative to the CDM.

The constraint
--------------

A free rigid body has six degrees of freedom. The single engaged button removes
three of them:

#. The button stays on the rail line. Its position may only vary along the
   rail, so the component of its acceleration perpendicular to the rail is
   zero. That is two scalar constraints.
#. The button in its rail slot prevents roll:
   :math:`\dot{\boldsymbol{\omega}} \cdot \hat{\mathbf{z}}_b = 0`. That is one
   more.

Three constraints leave :math:`6 - 3 = 3` degrees of freedom: translation along
the rail, pitch and yaw.

The forces that enforce them are the unknowns of the problem:

* a normal reaction at the button, perpendicular to the rail because a
  frictionless slot can neither pull nor push along it. Writing an orthonormal
  body triad :math:`\{\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{n}}_b\}`
  around the body-frame rail direction
  :math:`\hat{\mathbf{n}}_b = \mathbf{K}^{\mathsf T}\hat{\mathbf{n}}`, it has two
  components: :math:`\mathbf{N}_b = \lambda_1 \hat{\mathbf{e}}_1 + \lambda_2
  \hat{\mathbf{e}}_2`;
* a roll reaction moment :math:`\mu \hat{\mathbf{z}}_b` about the body axis.

That is three unknowns for three constraints, a :math:`3\times3` linear system
solved once per evaluation of the right-hand side.

The initial conditions are consistent with the constraint. The phase is entered
from ``udot_rail1``, where the rocket has no angular velocity and its velocity is
along the rail, so the button's velocity perpendicular to the rail is already
zero. Enforcing zero perpendicular acceleration therefore keeps it on the rail.

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
   + \mathbf{r}_{CM} \times \dot{\boldsymbol{\omega}}_{\text{free}}.

Both totals are sums of external contributions, so the reaction wrench is added
to them directly,

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
   + \mathbf{r}_{CM} \times \Delta\dot{\boldsymbol{\omega}}.

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
particular :math:`\dot{\mathbf{r}} = \mathbf{v}`: the velocity is not projected
onto the rail. The constraint acts on the acceleration of the button, and the
CDM acquires a small perpendicular velocity as the rocket pitches about that
button.

Modelling assumptions and edge cases
------------------------------------

* Roll axis: the constraint prevents roll about the body axis, not about the
  rail. The rocket travels only the button-to-button distance during this phase,
  so the tip-off angle is small and the body axis stays close to the rail
  direction. The difference between body roll and rail-axis roll is of that
  order. ``udot_rail1`` allows no rotation at all, so the two phases agree here.
* Radial button offset: the button is modelled on the rocket axis. The roll
  constraint is enforced by :math:`\mu`, so the only thing lost is the small
  roll coupling through the button's radial standoff.
* Rail friction: the rail is frictionless, as it is in ``udot_rail1``.
* Contact with the rail: only the button touches the rail. Nothing stops the
  rocket body or the fins from turning into it, so the model is meant for the
  small rotations of a normal launch. A rocket that leaves the rail very slowly
  (a thrust barely above its weight on a strongly inclined rail) can turn by
  tens of degrees inside the window, which no real rail would allow.
* Zero-length phase: a rocket with a single rail button has
  :math:`\ell_1 = \ell_2`, and the phase is skipped.

Expected behaviour
------------------

The phase reproduces the two effects tip-off is modelled for. With no wind, the
center of mass sits ahead of the button that the rocket now pivots about, so
gravity pitches the nose down by a fraction of a degree. With a crosswind, the
aerodynamic moment turns the rocket into the wind before it is fully free (the
weathercock effect), so the rocket leaves the rail with a small angular rate
instead of none.

Reading the results
-------------------

The window is recorded on the :class:`rocketpy.Flight` object, so a dispersion
study does not have to recover it from the raw solution:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Meaning
   * - ``out_of_rail_time``
     - Time the upper button leaves the rail, starting the window.
   * - ``tip_off_end_time``
     - Time the lower button leaves the rail, ending the window. Equal to
       ``out_of_rail_time`` when the phase does not run.
   * - ``tip_off_end_velocity``
     - Speed at that moment, when the rocket is fully off the rail.
   * - ``tip_off_end_state``
     - Full state vector handed to the free-flight phase.
   * - ``tip_off_duration``
     - Length of the window, zero when the phase does not run.
   * - ``rail_button2_normal_force``, ``rail_button2_shear_force``
     - Force between the lower button and the rail. Inside the window this is
       the force :math:`\mathbf{N}_b` solved above, with the opposite sign
       (the force of the rocket on the rail), and the upper button force is
       zero.

While both buttons are on the rail, the rail button forces count the
aerodynamic loads only. Inside the window the lower button force also carries
the weight of the rocket, so it rises when the window starts. The maximum lower
button forces (``max_rail_button2_normal_force`` and
``max_rail_button2_shear_force``) include the window.

``Flight.info()`` prints these under a *Tip-Off State* heading whenever the
phase ran, and the attitude plots shade the window so the change in attitude
during tip-off can be read from them.

Verification
------------

The phase is tested against an independent model of the same problem: a rigid
body of constant mass, under thrust and gravity only, whose lower button slides
along the rail. That model is written in two coordinates, the position
:math:`s` of the button along the rail and the angle :math:`\varphi` of the
rocket axis from the rail. With :math:`l` the distance from the button to the
center of mass, :math:`I` the inertia about the center of mass, :math:`T` the
thrust and :math:`\theta` the rail inclination,

.. math::

   \begin{aligned}
   \left(I + m l^2 \cos^2\varphi\right) \ddot{\varphi} &= l \cos\varphi
   \left(m l \dot{\varphi}^2 \sin\varphi + T \sin\varphi
   - m g \cos\theta\right), \\
   \ddot{s} &= \frac{T \cos\varphi}{m} - g \sin\theta
   + l \ddot{\varphi} \sin\varphi + l \dot{\varphi}^2 \cos\varphi.
   \end{aligned}

For a rocket with no aerodynamic surfaces, no drag and almost no propellant,
the pitch angle and the pitch rate at the end of the window agree with this
model to about :math:`10^{-7}`, at rail inclinations of 85° and 60°. The
leading term, :math:`\ddot{\varphi} \approx -m g l \cos\theta / (I + m l^2)`,
is also the one in equation (2) of [Peng]_, which treats the same phase in one
plane and adds rail friction. The force on the lower button is checked against
the same model, in which the rail pushes the rocket with
:math:`m g \cos\theta - T \sin\varphi + m l \left(\ddot{\varphi} \cos\varphi
- \dot{\varphi}^2 \sin\varphi\right)`.

The response to wind is not covered by such a closed model. It rests on the
aerodynamic forces and moments of the generalized equations of motion, which
this phase uses unchanged.

References
----------

[Chou]_ studies a vehicle moving along an inclined guideway with dynamic
interactions, [Hosken]_ analyses tip-off effects in rail launchers, and
[Peng]_ gives a planar model of the single-button phase of a sounding rocket.
