.. _aero_cp_stability:

================================
Center of Pressure and Stability
================================

Introduction
============

Stability is a central consideration in rocket design. Two rules of thumb are
widely repeated in amateur and student rocketry: the center of pressure must be
behind the center of gravity, and a static margin of one to two calibers is
generally desirable. This document explains the reasoning behind those rules
and distinguishes between three related but distinct quantities: **static
margin**, **stability margin**, and **dynamic stability**.

The presentation proceeds from physical principles to the exact definitions and
formulas used by RocketPy, illustrated throughout with the **Calisto** reference
rocket, introduced in the :ref:`firstsimulation` guide.

.. contents:: On this page
   :local:
   :depth: 2

.. jupyter-execute::
   :hide-code:
   :hide-output:

   # The Calisto reference rocket from the First Simulation guide, reused for
   # every worked example below. See the firstsimulation guide for the full
   # build. IMPORTANT: adjust the data paths below to match your own system.
   from rocketpy import Environment, SolidMotor, Rocket, Flight

   env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
   env.set_atmospheric_model(type="standard_atmosphere")

   Pro75M1670 = SolidMotor(
       thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
       dry_mass=1.815,
       dry_inertia=(0.125, 0.125, 0.002),
       nozzle_radius=33 / 1000,
       grain_number=5,
       grain_density=1815,
       grain_outer_radius=33 / 1000,
       grain_initial_inner_radius=15 / 1000,
       grain_initial_height=120 / 1000,
       grain_separation=5 / 1000,
       grains_center_of_mass_position=0.397,
       center_of_dry_mass_position=0.317,
       nozzle_position=0,
       burn_time=3.9,
       throat_radius=11 / 1000,
       coordinate_system_orientation="nozzle_to_combustion_chamber",
   )

   rocket = Rocket(
       radius=127 / 2000,
       mass=14.426,
       inertia=(6.321, 6.321, 0.034),
       power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
       power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
       center_of_mass_without_motor=0,
       coordinate_system_orientation="tail_to_nose",
   )
   rocket.add_motor(Pro75M1670, position=-1.255)
   rocket.set_rail_buttons(
       upper_button_position=0.0818,
       lower_button_position=-0.618,
       angular_position=45,
   )
   rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)
   rocket.add_trapezoidal_fins(
       n=4,
       root_chord=0.120,
       tip_chord=0.060,
       span=0.110,
       cant_angle=0.0,
       position=-1.04956,
       airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
   )
   rocket.add_tail(
       top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
   )

   test_flight = Flight(
       rocket=rocket, environment=env, rail_length=5.2, inclination=85, heading=0
   )


Part 1: Physical principles
============================

Static stability
-----------------

.. admonition:: Static stability
   :class: note

   A rocket is **statically stable** if, when a disturbance pushes its nose
   away from the flight direction, the aerodynamic forces generate a
   restoring moment that returns it toward that direction.

   An **unstable** rocket exhibits the opposite behavior: a disturbance is
   never corrected, and the rocket tumbles.

Whether a restoring moment exists depends on the relative position of two
points along the rocket's axis: the **center of mass** and the **center of
pressure**:

- **Center of mass (CM), also called center of gravity (CG).** The point about
  which the rocket's mass is balanced, and about which the rocket rotates in
  flight. In RocketPy this quantity is ``Rocket.center_of_mass``, a function of
  time since propellant consumption shifts it.

- **Center of pressure (CP).** The point at which the net aerodynamic force can
  be considered to act. At a small angle to the airflow, a sideways ("normal")
  force is generated, and the CP is its effective point of application. In
  RocketPy this quantity is ``Rocket.cp_position``, an alias of
  ``Rocket.aerodynamic_center``.

The relative position of these two points determines stability:

.. admonition:: The stability rule
   :class: important

   If the **center of pressure is located behind the center of mass** (toward
   the tail), the rocket is stable.

   If the center of pressure is located ahead of the center of mass the rocket
   is unstable.

.. figure:: ../../static/rocket/stable-unstable.png
   :align: center
   :width: 80%

   The aerodynamic force acting at the CP creates a torque about the CM.
   When the CP is aft of the CM the torque is restoring (left); when the CP
   is forward of the CM the torque grows the disturbance (right).

.. Role of the fins
.. -----------------

.. Fins are the primary means of shifting the CP toward the tail. As large
.. lifting surfaces positioned far aft, they move the rocket's aggregate center
.. of pressure behind the center of mass, establishing the restoring lever arm.
.. Nose cones and outward-flaring transitions tend to move the CP forward and are
.. destabilizing in isolation; the fins must more than compensate for this
.. effect.

.. Increasing fin size, moving the fins aft, or adding mass to the nose (which
.. moves the CM forward) are the standard corrective measures for an unstable
.. design, since each increases the distance by which the CP trails the CM.


Part 2: Static margin
=====================

Definition
----------

.. admonition:: Static margin
   :class: note

   The **static margin** is the distance between the **CM** and the **CP**,
   expressed in **calibers**, where one caliber equals the body diameter.

   .. math::
      :label: static_margin_intro

      \text{static margin} =
          \frac{(\text{CP position}) - (\text{CM position})}{\text{diameter}}
          \;\;[\text{calibers}]

   - A **positive** static margin indicates that the CP is behind the CM, and
     therefore that the rocket is stable. A margin of two calibers indicates
     that the two points are separated by two body diameters.
   - A **negative** static margin indicates that the CP is ahead of the CM, and
     therefore that the rocket is unstable.

   The diameter is used as the reference length because it is the natural
   length scale of the aerodynamics: the normal force generated by a body
   scales with its cross-sectional area. This convention is widely used in
   rocketry.

Here is a plot of the static margin of the Calisto rocket over time. The 
variation of the static margin here is due to the forward shift of the center of
mass as the motor burns and propellant is consumed. The right-hand axis
expresses the same margin as a percentage of body length (discussed in
:ref:`percent_of_length`):

.. jupyter-execute::

   rocket.plots.static_margin()

Recommended margin range
------------------------

The following ranges are widely used design rules of thumb, not precise or
authoritative thresholds:

- **Below approximately 1 caliber:** marginal. Manufacturing tolerances, added
  nose-cone mass, or a shifted CG can be enough to make the rocket unstable.
- **Approximately 1 to 2 calibers:** the standard target range for most
  rockets, and the most common recommendation in hobby and student rocketry.
- **Approximately 2 to 3 calibers:** another common target range for high-power
  rockets, providing extra margin for the CG shift that occurs as the motor
  burns.
- **Above approximately 4 calibers (over-stable):** also undesirable, for the
  reasons described below.

**These figures are rules of thumb, not objective truths**: the transition
between ranges is gradual and depends on the specific rocket, so a margin
just outside one of them does not mean the rocket will not fly safely. Falling
within a target range also does not by itself guarantee good flight behavior.

.. admonition:: Excessive static margin
   :class: warning

   An over-stable rocket has a strong restoring moment, so in a crosswind it
   turns into the wind and drifts further downwind ("weathercocking"). Aim for
   enough margin to keep the rocket reliably stable, rather than the largest
   margin achievable. The flight studies in :ref:`stability_in_flight` examine
   how much this actually affects a real flight, and the
   :ref:`practical studies <mass_vs_turn>` show that the altitude usually
   blamed on over-stability is really the cost of the added nose weight.

.. _percent_of_length:

Margin as a percentage of length
--------------------------------

An alternative convention expresses the margin as a **percentage of the
rocket's overall length** rather than in calibers:

.. math::

   \text{margin (\% of length)} =
       \frac{(\text{CP position}) - (\text{CM position})}{\text{body length}}
       \times 100

.. figure:: ../../static/rocket/cal-per-length.png
   :align: center
   :width: 80%

   The same CM-to-CP distance expressed against two reference lengths: the
   body diameter (one caliber) and the overall body length. Both numbers
   describe the identical physical gap.

This measures the identical physical distance as the caliber margin,
normalized by the total length instead of the diameter, as illustrated
above. RocketPy exposes the overall length as
:attr:`rocketpy.Rocket.length`, defined as the axial span from the nose tip
to the aft-most point of the rocket, whether that is an aerodynamic surface
or the motor nozzle if it extends further aft.

For Calisto (from :ref:`firstsimulation`), with a length of **2.53 m** and a 
fineness ratio of approximately 20, the two conventions yield:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Condition
     - Calibers
     - % of length
   * - Lift-off (``t = 0``)
     - ``2.20 c``
     - ``11.0 %``
   * - Burnout (``t = 3.9 s``)
     - ``3.11 c``
     - ``15.6 %``

As a rule of thumb, a percent-of-length margin of roughly **8 to 15%** is a
commonly cited target, with the lower bound the firmer of the two. For a
typical rocket with a fineness ratio near 10, one caliber is about 10% of the
length, so this range corresponds to the familiar 1 to 2 calibers. The two
conventions only diverge for unusually short or slender rockets, which is
precisely where the percentage form is meant to help. Like the caliber ranges,
these are **informal guidelines rather than authoritative thresholds**.

.. admonition:: Why express margin as a percent of length?
   :class: note

   The percent-of-length convention fixes a **gap in the caliber measure**: the
   caliber says nothing about the rocket's overall length. Two rockets with
   a static margin of 2 calibers, one short and one long and slender, don't
   necessarily behave the same in flight.
   
   The slender rocket has
   **substantially greater** rotational inertia (which scales with length
   squared) and a longer aerodynamic damping arm. Expressing the margin as
   a fraction of length scales the required
   physical margin with rocket size. This is a rough correction for
   slenderness.


.. _stability_margin_part:

Part 3: Stability margin
========================

The **stability margin** is the same center-of-mass-to-center-of-pressure 
distance, in the same calibers, but with the center of pressure taken at the
rocket's **actual flight condition**, its Mach number and angle of attack, 
rather than at rest.

Writing :math:`z_\text{cm}(t)` for the center of mass, :math:`2R` for the body
diameter (one caliber), the two margins have the same form and differ only in 
the center-of-pressure reference they subtract:

.. math::

   \text{static margin}(t) = c\,\frac{z_\text{cm}(t) - x_\text{AC}(0)}{2R},
   \qquad
   \text{stability margin}(\alpha, M, t)
       = c\,\frac{z_\text{cm}(t) - x_\text{NP}(\alpha, M)}{2R}.

- The static margin uses the **aerodynamic center** :math:`x_\text{AC}(0)`: the
  center of pressure linearized about zero angle of attack and evaluated at zero
  airspeed (:math:`M = 0`). It is a fixed reference, so the static margin varies
  only through the center of mass, that is, with time.
- The stability margin uses the **neutral point** :math:`x_\text{NP}(\alpha, M)`,
  the local center of pressure at the actual Mach number and angle of attack.
  Because that reference moves with the flow, the stability margin depends on
  angle of attack and Mach number as well as time.

The static margin and the stability margin are the same underlying
quantity, evaluated under different
conditions:

.. list-table::
   :header-rows: 1
   :widths: 24 24 26

   * -
     - **Static margin**
     - **Stability margin**
   * - Depends on
     - Time only
     - Angle of attack, Mach number **and** time
   * - Evaluated at
     - Zero incidence, zero airspeed (``M = 0``)
     - *Any* angle of attack and Mach number (an aerodynamic map)
   * - Answers
     - Stability at rest
     - Stability at any chosen flow state
   * - RocketPy
     - ``Rocket.static_margin`` (function of ``t``)
     - ``Rocket.stability_margin`` (function of ``alpha, M, t``)

The margin varies for up to three independent reasons:

**1. Center-of-mass displacement (time).** As the motor consumes propellant the
CM moves. This changes the CM-to-CP distance.

**2. Center-of-pressure displacement (Mach number).** Aerodynamic surfaces
effectivenss (e.g. fins) varies with speed, causing the CP
to shift with Mach number.

**3. Center-of-pressure displacement (angle of attack).** For a rocket
carrying a surface that generates lift nonlinearly with incidence, the center
of pressure also migrates with the angle of attack. RocketPy's pre-set
geometric surfaces (``NoseCone``, ``TrapezoidalFins``,
``EllipticalFins``, ``FreeFormFins``, ``Tail``) are all linear in
incidence, so a rocket built only from them never sees this effect. It shows up
when a surface is added as a :class:`rocketpy.GenericSurface` with a nonlinear
coefficient, for example a Galejs body-lift term.

The flight stability margin
----------------------------

``Flight.stability_margin`` samples the rocket's ``stability_margin`` map at
the angle of attack, Mach number and time realized during the simulated flight:

.. jupyter-execute::

   test_flight.prints.stability_margin()

``Flight.plots.stability_margin_data()`` plots this curve for the pitch and yaw
planes, with a percent-of-length axis on the right:

.. jupyter-execute::

   test_flight.plots.stability_margin_data()

.. tip::

   The **out-of-rail stability margin** is frequently the most significant
   single value: it is the margin at the instant of rail departure, when the
   rocket is slowest and most susceptible to wind disturbance. A recommended
   design practice is to ensure this value is suficcient. The out-of-rail
   instant is examined in detail in :ref:`stability_in_flight`.


.. _part_dynamic:

Part 4: Dynamic stability
=========================

The static margin has a fundamental limitation: it indicates only the *sign*
of the restoring moment, not the rocket's actual dynamic behavior. A positive
margin guarantees the existence of a restoring tendency but provides no
information regarding the rate of return, the presence of overshoot, or
whether resulting oscillations decay or persist. These characteristics are
described by **dynamic stability**.

.. jupyter-execute::
   :hide-code:
   :hide-output:

   import numpy as np
   import matplotlib.pyplot as plt

   def build_rocket(center_of_mass_without_motor=0.0, lateral_inertia=6.321,
                    mass=14.426, motor=Pro75M1670):
       """Calisto with four adjustable parameters. Shifting the dry center of
       mass toward the nose (positive) raises the static margin; toward the tail
       (negative) lowers it. The lateral moment of inertia is adjustable
       separately, which changes the dynamic response without changing the
       static margin. The dry mass (in kilograms) and the motor can also be
       swapped, for the mass and thrust studies later in this document.
       Everything else, including the aerodynamics, is fixed.
       """
       rocket = Rocket(
           radius=127 / 2000,
           mass=mass,
           inertia=(lateral_inertia, lateral_inertia, 0.034),
           power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
           power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
           center_of_mass_without_motor=center_of_mass_without_motor,
           coordinate_system_orientation="tail_to_nose",
       )
       rocket.add_motor(motor, position=-1.255)
       rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)
       rocket.add_trapezoidal_fins(
           n=4, root_chord=0.120, tip_chord=0.060, span=0.110,
           cant_angle=0.0, position=-1.04956,
           airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
       )
       rocket.add_tail(
           top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
       )
       return rocket

   def windy_site(wind_speed):
       """A standard atmosphere with a constant eastward crosswind, in m/s."""
       site = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
       site.set_atmospheric_model(
           type="custom_atmosphere", wind_u=wind_speed, wind_v=0
       )
       return site

The attitude oscillator
-----------------------

A stable rocket disturbed by angle :math:`\theta` behaves as a damped
spring-mass oscillator:

.. math::

   I_L\,\ddot\theta + C_2\,\dot\theta + C_1\,\theta = 0

with three governing parameters:

- :math:`C_1`, the **corrective (restoring) moment coefficient**, analogous
  to a spring constant. It is proportional to the static margin:
  :math:`C_1 = \bar q\, A\, C_{N,\alpha}\, (z_\text{cm} - x_\text{cp})`, where
  :math:`\bar q` is the dynamic pressure. A larger margin or higher airspeed
  produces a stiffer restoring moment.
- :math:`C_2`, the **damping moment coefficient**, analogous to a damping
  coefficient. It arises from aerodynamic resistance of the fins to rotation,
  together with **jet damping** resulting from mass ejection through the
  nozzle.
- :math:`I_L`, the **lateral moment of inertia** about the center of mass,
  representing the rotational inertia opposing angular acceleration.

These parameters determine the two quantities that characterize the dynamic
response:

.. math::

   \omega_n = \sqrt{\frac{C_1}{I_L}}
   \quad(\text{natural frequency}),
   \qquad
   \zeta = \frac{C_2}{2\sqrt{C_1\,I_L}}
   \quad(\text{damping ratio}).

- The **natural frequency** :math:`\omega_n` sets the oscillation rate, in
  rad/s (divide by :math:`2\pi` for Hz), and increases with static margin and
  airspeed.
- The **damping ratio** :math:`\zeta` sets how quickly the oscillation
  decays. Rockets are normally *underdamped* (:math:`\zeta < 1`), oscillating
  with the amplitude shrinking over several cycles.

.. figure:: ../../static/rocket/damped-oscillation.png
   :align: center
   :width: 90%

   Attitude response to a single disturbance. The spacing between peaks is set
   by the natural frequency; the rate at which the peaks shrink toward zero, the
   dashed envelope, is set by the damping ratio. A larger damping ratio decays
   faster for the same natural frequency.

RocketPy exposes each of these quantities on the ``Flight`` object:
``corrective_moment_coefficient`` (:math:`C_1`),
``damping_moment_coefficient`` (:math:`C_2`),
``pitch_natural_frequency``, ``pitch_damping_ratio``, and the corresponding
``yaw_*`` quantities. ``Flight.prints.dynamic_stability()`` summarizes them at
the key ascent instants, rail departure and burnout, together with the roll
rate at burnout:

.. jupyter-execute::

   test_flight.prints.dynamic_stability()

``Flight.plots.dynamic_stability_data()`` plots natural frequency and damping
ratio as functions of time, with roll rate overlaid.

.. jupyter-execute::

   test_flight.plots.dynamic_stability_data()

.. _stability_in_flight:

Part 5: Stability in a real flight
==================================

Parts 1 to 4 defined each stability quantity on a single nominal flight. This
part watches those quantities at work: the wind disturbance at rail exit, the
margin and damping correcting it, and how stable the rocket stays across the
spread of conditions a real launch day brings.

.. admonition:: The out-of-rail instant
   :class: note  

   Rail departure is the most critical moment of a rocket's flight. The rail
   buttons hold its orientation until the last one clears the rail. From then
   on, aerodynamics takes over, right as the rocket is at its slowest and
   least able to resist a disturbance.

Stall is not modeled
--------------------

**RocketPy does not model stall.** On a real fin, lift grows with angle of
attack only up to a point. Past roughly 10 to 15 degrees the flow
**separates** from the surface, the lift collapses, and the drag climbs
sharply: the fin *stalls*. A stalled fin **stops correcting**, and the rocket
can tumble out of controlled flight.

RocketPy's Barrowman fins never do this. A fin takes its lift *slope* at zero
angle of attack and applies that **same slope at every angle**, so its normal
force keeps growing without limit:

.. jupyter-execute::
   :hide-code:

   # Left: the measured NACA 0012 lift curve (the file stores angle in radians).
   # Right: the RocketPy fin built from that same airfoil. RocketPy keeps only
   # the curve's slope at 0 degrees, so the fin's coefficient never stalls.
   airfoil = np.loadtxt("../data/airfoils/NACA0012-radians.txt", delimiter=",")
   aoa_deg, cl = np.degrees(airfoil[:, 0]), airfoil[:, 1]

   fin = build_rocket().fins[0]
   clalpha = fin.clalpha(0.1)                 # normal-force slope at low Mach
   aoa = np.linspace(0, 15, 200)
   cn_fin = clalpha * np.radians(aoa)
   peak = np.argmax(cl[aoa_deg <= 20])        # airfoil peak, just before stall

   fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8))

   shown = aoa_deg <= 15
   axL.plot(aoa_deg[shown], cl[shown], "-o", color="#c0392b", lw=2, ms=4)
   axL.annotate("stall", xy=(aoa_deg[peak], cl[peak]),
                xytext=(aoa_deg[peak] + 1.5, cl[peak] + 0.03),
                fontsize=11, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
   axL.set_title("Real airfoil (NACA 0012 data)", fontweight="bold")
   axL.set_ylabel(r"lift coefficient  $C_L$")

   axR.plot(aoa, cn_fin, color="#2980b9", lw=2.4)
   axR.text(0.05, 0.9, "linear: never stalls", transform=axR.transAxes,
            fontsize=11, fontweight="bold", color="#2980b9")
   axR.set_title("RocketPy fin (Barrowman)", fontweight="bold")
   axR.set_ylabel(r"normal-force coefficient  $C_N$")

   for ax in (axL, axR):
       ax.set_xlabel("angle of attack (deg)")
       ax.set_xlim(0, 15)
       ax.set_ylim(0, 0.9)
       ax.grid(alpha=0.25)
       ax.spines[["top", "right"]].set_visible(False)
   fig.tight_layout()
   plt.show()

The airfoil on the left carries the whole story of the flow: lift rises,
**peaks near 9 degrees, then falls off a cliff** as the flow separates. The
RocketPy fin on the right, built from that very airfoil, **keeps the same
initial slope forever**.

So **a high computed rail-exit angle of attack marks a failure, not a
survivable condition.** The simulation shows the rocket swinging back into
line even past the angle where a real fin would have stalled. Keep the
out-of-rail velocity high relative to the wind and the rocket stays below the
stall range; the recovery the simulation shows past it **would not happen in
reality**. In the sweeps below, read a large computed angle of attack as a
warning sign, not a number the simulation can be trusted to reproduce.

To actually simulate stall, or any other measured nonlinear aerodynamics,
provide the coefficients directly with a generic surface (see
:ref:`genericsurfaces`).

The angle of attack at rail exit
--------------------------------

At rail departure the body points along the rail. But the air it meets is the
vector sum of the rocket's own velocity and the wind. That gives an angle of
attack of approximately

.. math::

   \alpha_\text{exit} \approx \arctan\!\left(\frac{V_\text{wind}}{V_\text{exit}}\right),

where :math:`V_\text{wind}` is the crosswind and :math:`V_\text{exit}` is the
out-of-rail velocity. Only their *ratio* matters: a stronger wind and a
slower exit velocity push the angle up the same way. The flight below, the
nominal Calisto in an 8 m/s crosswind, checks the estimate against a real
simulation:

.. jupyter-execute::

   import numpy as np

   windy_flight = Flight(
       rocket=build_rocket(), environment=windy_site(5),
       rail_length=5.2, inclination=85, heading=0, terminate_on_apogee=True,
   )

   v_exit = windy_flight.out_of_rail_velocity
   aoa_exit = windy_flight.angle_of_attack(windy_flight.out_of_rail_time)
   estimate = np.degrees(np.arctan(5 / v_exit))

   print(f"out-of-rail velocity : {v_exit:5.1f} m/s")
   print(f"rail-exit AoA        : {aoa_exit:5.2f} deg   (simulated)")
   print(f"arctan(wind / V_exit): {estimate:5.2f} deg   (geometric estimate)")

The simulated angle tracks the arctangent estimate closely. This is the
disturbance set the instant the rail lets go, and the rest of the flight has
to correct it. Keeping it well below the stall range of the fins is the
first stability requirement of any launch.

The disturbance, corrected
--------------------------

The damped oscillation from :ref:`part_dynamic` can be represented in a flight
simulation. Tracking the angle of
attack after rail exit shows the rocket swinging back toward the relative
wind, and the swing dying away:

.. jupyter-execute::

   t0 = windy_flight.out_of_rail_time
   t = np.linspace(t0, t0 + 4, 400)
   aoa = [windy_flight.angle_of_attack(ti) for ti in t]

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(t, aoa)
   ax.axvline(t0, color="0.6", ls="--", lw=1, label="rail exit")
   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Angle of attack (deg)")
   ax.set_title("Response to the rail-exit disturbance (8 m/s crosswind)")
   ax.legend()
   ax.grid(True)
   plt.show()

Every stability quantity from earlier parts shows up here. The rocket returns
toward zero at all because its **stability margin**
(:ref:`stability_margin_part`) is positive. The center of pressure sits
behind the center of mass, so the aerodynamic force restores rather than
diverges. The *rate* of the wobble is the **natural frequency**. The *speed*
it settles at is the **damping ratio** (:ref:`part_dynamic`).
``windy_flight.prints.dynamic_stability()`` reports both for this flight. A
positive margin only guarantees the curve trends back to zero. It says
nothing about how fast or how smoothly, which is exactly the distinction
Part 4 draws. Here the angle of attack swings through several cycles before it
settles, a sign that this rocket is only lightly damped. It recovers either
way, but for a cleaner flight, one that settles after an overshoot or two, it
could do with more damping.

Stability across a launch day
-----------------------------

**A single nominal flight is not what a launch actually delivers.** The wind
shifts from minute to minute. The finished mass differs from the design
value. The center of mass is never exactly where the drawing puts it. That
spread of conditions moves two quantities that matter at rail exit: the
stability margin and the angle of attack.

RocketPy's Monte Carlo tooling measures exactly that. The ``Stochastic*``
classes wrap the nominal environment, rocket, motor and flight, and attach a
spread to each uncertain input. :class:`rocketpy.MonteCarlo` then runs the
flight many times, each with a fresh draw, and saves the results. Here the
balance (dry mass and center of mass), the motor's total impulse and burn
time, and the crosswind are dispersed:

.. code-block:: python

   from rocketpy import MonteCarlo, NoseCone, TrapezoidalFins, Tail
   from rocketpy.stochastic import (
       StochasticEnvironment,
       StochasticRocket,
       StochasticSolidMotor,
       StochasticFlight,
       StochasticNoseCone,
       StochasticTrapezoidalFins,
       StochasticTail,
   )

   # Environment: 3 m/s mean crosswind, scaled by a normal factor so the wind
   # spans roughly 3 +/- 2.5 m/s from flight to flight.
   stochastic_env = StochasticEnvironment(
       environment=windy_site(3),
       wind_velocity_x_factor=(1.0, 0.42, "normal"),
   )

   # Motor: total impulse and burn time vary together, the way two motors from
   # the same production lot differ. Roughly +/- 3%, a typical manufacturing
   # tolerance; reshapes the whole thrust curve to match each draw.
   stochastic_motor = StochasticSolidMotor(
       solid_motor=Pro75M1670,
       total_impulse=(6026, 180, "normal"),  # newton-seconds
       burn_out_time=(3.9, 0.12, "normal"),  # seconds
   )

   # Rocket: same Calisto, but the dry mass and the balance point vary too.
   # The surfaces are added back with no spread (fixed geometry).
   stochastic_rocket = StochasticRocket(
       rocket=rocket,
       mass=(14.426, 0.4, "normal"),                        # dry mass, kg
       center_of_mass_without_motor=(0.0, 0.03, "normal"),  # balance, m
   )
   stochastic_rocket.add_motor(stochastic_motor, position=(-1.255, 0))
   fixed_surface = {
       NoseCone: (StochasticNoseCone, stochastic_rocket.add_nose),
       TrapezoidalFins: (StochasticTrapezoidalFins,
                         stochastic_rocket.add_trapezoidal_fins),
       Tail: (StochasticTail, stochastic_rocket.add_tail),
   }
   # Re-add each surface with no spread; a (value, 0) position means "fixed".
   for surface, position in rocket.aerodynamic_surfaces:
       stochastic_surface, add = fixed_surface[type(surface)]
       add(stochastic_surface(surface), (position.z, 0))

   # Flight: fixed rail and launch angles; stop each run at apogee to save time.
   base_flight = Flight(
       rocket=rocket, environment=windy_site(5),
       rail_length=5.2, inclination=85, heading=0, terminate_on_apogee=True,
   )
   stochastic_flight = StochasticFlight(flight=base_flight, terminate_on_apogee=True)

   # A data_collector callback receives each finished flight and returns a value,
   # here the rail-exit angle of attack (not one of the standard exports).
   analysis = MonteCarlo(
       filename="../data/monte_carlo/stability_dispersion",
       environment=stochastic_env,
       rocket=stochastic_rocket,
       flight=stochastic_flight,
       data_collector={
           "rail_exit_aoa": lambda f: f.angle_of_attack(f.out_of_rail_time),
       },
   )
   analysis.simulate(number_of_simulations=100, include_function_data=False)

   analysis.set_results()
   margins = np.array(analysis.results["out_of_rail_stability_margin"])
   aoa_exits = np.array(analysis.results["rail_exit_aoa"])

.. jupyter-execute::
   :hide-code:
   :hide-output:

   # The docs build does not run the Monte Carlo above; it loads the committed
   # output below instead. Regenerate it by running the code above (which writes
   # to the same path) after changing any distribution.
   import json

   results_file = "../data/monte_carlo/stability_dispersion.outputs.txt"
   with open(results_file, encoding="utf-8") as f:
       records = [json.loads(line) for line in f]
   margins = np.array([r["out_of_rail_stability_margin"] for r in records])
   aoa_exits = np.array([r["rail_exit_aoa"] for r in records])

The two quantities are now distributions over a simulated launch day:

.. jupyter-execute::

   fig, (axm, axa) = plt.subplots(1, 2, figsize=(10, 4))
   axm.hist(margins, bins=20, color="#4c72b0")
   axm.axvspan(2.0, 3.0, color="green", alpha=0.12, label="2-3 cal target")
   axm.set_xlabel("Out-of-rail stability margin (cal)")
   axm.set_ylabel("Simulations")
   axm.legend()
   axa.hist(aoa_exits, bins=20, color="#c44e52")
   axa.axvline(10, color="k", ls="--", label="stall onset (~10 deg)")
   axa.set_xlabel("Rail-exit angle of attack (deg)")
   axa.legend()
   fig.suptitle(f"Launch-day dispersion ({margins.size} simulations)")
   fig.tight_layout()
   plt.show()

   print(f"out-of-rail margin : {np.percentile(margins, 5):.2f} to "
         f"{np.percentile(margins, 95):.2f} cal (5th-95th percentile)")
   print(f"rail-exit AoA      : 95th pct {np.percentile(aoa_exits, 95):.1f} deg, "
         f"worst {aoa_exits.max():.1f} deg")
   print(f"flights above stall: {100 * np.mean(aoa_exits > 10):.0f}%")

This is a concrete deliverable of a stability analysis. Not a single
margin, but a *distribution* read against the design thresholds. If a meaningful
fraction of flights cross either line, the rail, the mass or the margin needs 
another look before flying.

.. seealso::

   A full dispersion study varies every input, not just five, and runs the
   flights in parallel. ``analysis.simulate(..., parallel=True)`` does that,
   and the ``Stochastic*`` classes cover parachutes and rail buttons too. See
   :ref:`stochastic_usage` for the class walkthrough, and :ref:`MRS` for
   weighting a finished sample toward measured launch-day conditions.

Part 6: How much does the static margin matter?
===============================================

The dispersion above shows a Calisto-class rocket that is comfortably stable,
and yet a great deal of design effort in rocketry goes into chasing static
margin. The simulation lets us weigh that margin against the other things a
builder can change, and see **how much it really decides**, following
Thomas Fetter's flight-data study *How Far Does a Rocket Turn Into the
Wind?* (NARCON-2024).

A rocket launched straight up into a crosswind turns as it climbs, and by the
time the motor burns out its flight path has tilted some degrees away from
vertical. This tilt is the *turn*, and because the launch was vertical it is
**entirely the rocket's response to the wind**, the weathercocking that
stability is meant to hold in check. Sweeping each design parameter on its own across a
realistic range, with the others left at their nominal values, shows how much
each one moves the turn:

.. jupyter-execute::

   def turn_at_burnout(flight):
       """Flight-path tilt away from vertical at motor burnout, in degrees."""
       return 90 - flight.path_angle(flight.rocket.motor.burn_out_time)

   def vertical_flight(rocket, wind, rail=5.2):
       return Flight(
           rocket=rocket, environment=windy_site(wind),
           rail_length=rail, inclination=90, heading=0, terminate_on_apogee=True,
       )

   def sweep(values, make_flight):
       flights = [make_flight(v) for v in values]
       return np.array([turn_at_burnout(f) for f in flights]), flights

   # Each lever swept alone across a realistic range; others nominal, 5 m/s wind.
   winds = np.linspace(0, 14, 9)              # crosswind, m/s
   rails = np.linspace(1.2, 9.0, 8)           # rail length sets the exit velocity
   cgs = np.linspace(-0.25, 0.9, 10)          # sets the static margin
   masses = np.linspace(9, 30, 8)             # dry mass, kg

   turn_wind, _ = sweep(winds, lambda w: vertical_flight(build_rocket(), w))
   turn_rail, rail_f = sweep(rails, lambda r: vertical_flight(build_rocket(), 5, rail=r))
   turn_cg, cg_f = sweep(cgs, lambda c: vertical_flight(build_rocket(c), 5))
   turn_mass, _ = sweep(masses, lambda m: vertical_flight(build_rocket(mass=m), 5))

   exit_v = np.array([f.out_of_rail_velocity for f in rail_f])
   margins = np.array([f.rocket.static_margin(0) for f in cg_f])

   panels = [
       (winds,   turn_wind, "crosswind (m/s)",     "wind speed",                 "#c0392b"),
       (exit_v,  turn_rail, "exit velocity (m/s)", "exit velocity (rail length)", "#e67e22"),
       (margins, turn_cg,   "static margin (cal)", "static margin",              "#2980b9"),
       (masses,  turn_mass, "dry mass (kg)",       "mass",                       "#27ae60"),
   ]
   ymax = max(y.max() for _, y, *_ in panels)

   fig, axs = plt.subplots(2, 2, figsize=(9.5, 7), sharey=True)
   for ax, (x, y, xlabel, title, color) in zip(axs.flat, panels):
       ax.fill_between(x, 0, y, color=color, alpha=0.12)
       ax.plot(x, y, "-o", color=color, lw=2.4, ms=6, mfc=color, mec="white", mew=0.8)
       ax.annotate(f"swing {y[-1] - y[0]:+.1f}°",   # signed: low end -> high end
                   xy=(0.04, 0.92), xycoords="axes fraction", ha="left", va="top",
                   fontsize=11, fontweight="bold", color=color)
       ax.set_title(title, fontweight="bold")
       ax.set_xlabel(xlabel)
       ax.grid(True, alpha=0.3)
       ax.set_ylim(0, ymax * 1.12)
   axs[0, 0].set_ylabel("turn at burnout (deg)")
   axs[1, 0].set_ylabel("turn at burnout (deg)")
   fig.suptitle("What moves the turn into the wind?  (each lever alone; others nominal, "
                "5 m/s wind)", fontsize=12, fontweight="bold")
   fig.tight_layout()
   plt.show()

Each panel is labeled with its *swing*, meaning how many degrees the turn
changes as that parameter goes from the low end of its range to the high end.

The wind dominates, mass comes next, and a higher exit velocity has a
moderate effect the other way, lowering the turn. 

**The static margin is the weakest of the four**: across a change in margin
the turn barely moves, and it levels off at high margins, holding steady well
past the over-stable range. For a Calisto-class rocket the margin is simply
not what decides how far it weathercocks. A rocket that turns hard into the
wind is easy to **misjudge as over- or super-stable**.

What static margin it does instead is *correction*.
Keeping the center of pressure behind the center of mass is what lets a
disturbance correct itself at all (:ref:`stability_margin_part` and
:ref:`part_dynamic`), and the rail-exit angle of attack still has to stay
below the stall range. Beyond that, **a larger margin does little for a
flight like this one**.

Part 7: Pitch and yaw planes
============================

A rocket with evenly spaced fins, like Calisto, is **axisymmetric**. Its
geometry does not change under rotation about the body axis, so its
stability is the same in every plane, and a single margin describes it
fully. For these rockets, ``Rocket.is_axisymmetric`` returns ``True``, and
the rest of this section does not apply.

Some configurations are **not** axisymmetric: canards on a single axis,
off-center payloads, fins arranged asymmetrically. For these, stability
differs between the **pitch** plane and the **yaw** plane, and RocketPy
computes each one independently:

- pitch: ``aerodynamic_center``, ``static_margin``, ``stability_margin``;
- yaw: ``aerodynamic_center_yaw``, ``static_margin_yaw``, ``stability_margin_yaw``.

For an axisymmetric rocket, the two planes coincide. When they do not,
RocketPy issues a warning, because the unqualified ``static_margin`` then
describes the pitch plane only. In that case, the plotting and print
methods report both planes.

Helper code
===========

Every worked example on this page is built on the same Calisto reference
rocket and a handful of small helper functions. Most of that code runs behind
the scenes so the examples above can stay focused on one idea at a time. It is
gathered here so it can be read and reused. The data-file paths are written
relative to the RocketPy ``docs`` directory; adjust them to your own setup.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from rocketpy import Environment, SolidMotor, Rocket, Flight

   # The Calisto reference rocket from the First Simulation guide, reused for
   # every worked example on this page.
   env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
   env.set_atmospheric_model(type="standard_atmosphere")

   Pro75M1670 = SolidMotor(
       thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
       dry_mass=1.815,
       dry_inertia=(0.125, 0.125, 0.002),
       nozzle_radius=33 / 1000,
       grain_number=5,
       grain_density=1815,
       grain_outer_radius=33 / 1000,
       grain_initial_inner_radius=15 / 1000,
       grain_initial_height=120 / 1000,
       grain_separation=5 / 1000,
       grains_center_of_mass_position=0.397,
       center_of_dry_mass_position=0.317,
       nozzle_position=0,
       burn_time=3.9,
       throat_radius=11 / 1000,
       coordinate_system_orientation="nozzle_to_combustion_chamber",
   )

   rocket = Rocket(
       radius=127 / 2000,
       mass=14.426,
       inertia=(6.321, 6.321, 0.034),
       power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
       power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
       center_of_mass_without_motor=0,
       coordinate_system_orientation="tail_to_nose",
   )
   rocket.add_motor(Pro75M1670, position=-1.255)
   rocket.set_rail_buttons(
       upper_button_position=0.0818,
       lower_button_position=-0.618,
       angular_position=45,
   )
   rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)
   rocket.add_trapezoidal_fins(
       n=4,
       root_chord=0.120,
       tip_chord=0.060,
       span=0.110,
       cant_angle=0.0,
       position=-1.04956,
       airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
   )
   rocket.add_tail(
       top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
   )

   test_flight = Flight(
       rocket=rocket, environment=env, rail_length=5.2, inclination=85, heading=0
   )

The design studies build fresh rockets and windy environments through two
small factories:

.. code-block:: python

   def build_rocket(center_of_mass_without_motor=0.0, lateral_inertia=6.321,
                    mass=14.426, motor=Pro75M1670):
       """Calisto with four adjustable parameters. Shifting the dry center of
       mass toward the nose (positive) raises the static margin; toward the tail
       (negative) lowers it. The lateral moment of inertia is adjustable
       separately, which changes the dynamic response without changing the
       static margin. The dry mass (in kilograms) and the motor can also be
       swapped, for the mass and thrust studies later in this document.
       Everything else, including the aerodynamics, is fixed.
       """
       rocket = Rocket(
           radius=127 / 2000,
           mass=mass,
           inertia=(lateral_inertia, lateral_inertia, 0.034),
           power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
           power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
           center_of_mass_without_motor=center_of_mass_without_motor,
           coordinate_system_orientation="tail_to_nose",
       )
       rocket.add_motor(motor, position=-1.255)
       rocket.add_nose(length=0.55829, kind="vonKarman", position=1.278)
       rocket.add_trapezoidal_fins(
           n=4, root_chord=0.120, tip_chord=0.060, span=0.110,
           cant_angle=0.0, position=-1.04956,
           airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
       )
       rocket.add_tail(
           top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
       )
       return rocket

   def windy_site(wind_speed):
       """A standard atmosphere with a constant eastward crosswind, in m/s."""
       site = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
       site.set_atmospheric_model(
           type="custom_atmosphere", wind_u=wind_speed, wind_v=0
       )
       return site

The three helpers that measure the turn and sweep one parameter at a time
(``turn_at_burnout``, ``vertical_flight`` and ``sweep``) are shown inline where
they are used, in Part 6 above.

