Missile DATCOM Aerodynamics
===========================

When construction or finish is unspecified, assume **carbon-fibre construction
with a vinyl finish**. Explicit material details, measured surface properties
and supplied geometry take precedence. This is a construction assumption,
not a measured roughness or fin-section specification: it does not determine
one universal roughness height, thickness, bevel or leading-edge radius.
Document any additional assumptions and use the calibration methods below
to set or fit native aerodynamic inputs.

Build the rocket, motor and tanks normally, using :class:`rocketpy.Rocket`,
``add_motor``, ``add_nose`` and ``add_tail``. Select whole-vehicle DATCOM
aerodynamics by adding one of these fin sets:

* ``add_datcom_trapezoidal_fins``: trapezoidal planform with a symmetric HEX
  section, a flat middle 90% of the chord and 5% edge bevels.
* ``add_datcom_arc_fins``: trapezoidal planform with a symmetric circular-arc
  airfoil section.
* ``add_datcom_hex_fins``: trapezoidal planform with a configurable symmetric
  hexagonal or diamond airfoil section.

ARC and HEX describe the airfoil cross section, not the planform. Root chord,
tip chord, exposed span, sweep and placement follow ``add_trapezoidal_fins``.
For example, after defining ``motor`` and ``environment``::

    from rocketpy import Flight, Rocket

    rocket = Rocket(
        radius=0.0635,
        mass=14.426,                    # excludes the entire motor, kg
        inertia=(6.321, 6.321, 0.034),   # excludes the motor, kg m²
        center_of_mass_without_motor=0,
        power_on_drag=0,                # replaced by DATCOM
        power_off_drag=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.373)
    rocket.add_nose(length=0.55, kind="von karman", position=1.2)
    fins = rocket.add_datcom_hex_fins(
        n=4,
        root_chord=0.25,
        tip_chord=0.10,
        span=0.10,
        position=-0.9,
        sweep_length=0.10,
        path="/absolute/path/to/native/datcom",
        thickness_to_chord=(0.02, 0.03),
        max_thickness_position=0.25,
        flat_length=0.5,
        leading_edge_radius=0,
        datcom_options={
            "mach": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0],
            "alpha": [-20, -10, -5, -2, -1, -0.5, -0.25, -0.1, -0.05, 0,
                      0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
            "altitude": 0,
            "timeout": 60,
        },
    )
    rocket.add_tail(
        top_radius=rocket.radius, bottom_radius=0.045,
        length=0.10, position=-1.2, kind="ogive",
    )
    flight = Flight(rocket, environment, rail_length=5)

The nose, fins and tail retain their original geometry for drawing and export.
Flight uses one DATCOM model for all aerodynamic forces and moments, replacing
their component models and the rocket drag inputs. Geometry is read from the
assembled objects, including surfaces and motor added after the DATCOM fins.
Adding a :class:`rocketpy.DatcomFins` instance with ``add_surfaces`` also selects
DATCOM aerodynamics.

Engineering Sensitivity Runs
----------------------------

With DATCOM fins, ``Flight`` runs the nominal simulation and two additional
isolated simulations through apogee by default. Disable the extra simulations
with ``Flight(..., datcom_uncertainty=False)``. Ordinary rockets retain their
existing aerodynamic equations, simulation and printed output.

The current engineering rule is:

* Static wind-axis drag CD varies by **at least ±20%**, at every Mach and AoA,
  for both powered and coast aerodynamics.
* Within Mach 0.8–3.02 and alpha -6 to +6 degrees, use a larger native-relative
  axial-force discrepancy from Wu et al. where available. The transferred CA
  discrepancy is a **CD-error proxy**, not a measured universal CD uncertainty.
  Outside that envelope, use the 20% excursion without extrapolating the
  canard study's larger errors.
* CP's distance from the nose varies by **±2%**. Static pitching/yawing moments
  are transferred consistently about the original case CG reference.
* Perpendicular aerodynamic force and native DAMP derivatives are retained.
  The earlier independent CN and CM-alpha error scaling is no longer used.

The design-accuracy reference is J. T. White (1995), *An assessment of missile
DATCOM prediction accuracy relative to generic Body+Wing+Tail missile pitch
aerodynamics*, 13th Applied Aerodynamics Conference, AIAA 95-1893,
`doi:10.2514/6.1995-1893 <https://doi.org/10.2514/6.1995-1893>`_. The filename
of the supplied PDF says 2012, but the paper is from 1995. White tests revision
6/93 on 15 configurations, more than 900 cases, Mach 1.5–5.0, AoA 0–24° and
roll orientations 0°/45°, without control deflections. His cited preliminary
accuracy criteria concern **normal force** and CP relative to **body length**.
They do not establish universal ±20% CD or ±2% nose-to-CP confidence bounds.
Using the requested percentages throughout flight is an explicit engineering
assumption, including below the paper's supersonic coverage. The ambiguous
entries in his CP summary table are not used as standard deviations.

The larger drag discrepancies adapt Table 2 of Hanzhou Wu, Min Gao, Weidong
Song, Zhao Jie and Yi Wang (2019), *Accuracy Analysis of Aerodynamic Calculation
of Two-dimensional Ballistic correction Projectile based on Missile Datcom*,
IOP Conference Series: Materials Science and Engineering **612**, 032096,
`doi:10.1088/1757-899X/612/3/032096
<https://doi.org/10.1088/1757-899X/612/3/032096>`_, under
`CC BY 3.0 <https://creativecommons.org/licenses/by/3.0/>`_. It studies one
canard projectile, including deflected canards. The signed published error
uses the measurement denominator:
``e = (DATCOM - measured) / measured``. With ``e`` as a fraction, convert to
``u_Wu = abs(e / (1 + e))`` and linearly interpolate these magnitudes with
Mach/alpha inside the paper's envelope. The applied CD excursion is
``u_CD = max(0.20, u_Wu)`` there, and ``0.20`` elsewhere. The original raw
CN/CM-alpha tables remain in the source for provenance, not as active bounds.

At zero alpha and beta this directly multiplies the static CD curve by
``1 + u_CD`` or ``1 - u_CD``. At finite incidence, the added force lies along
the relative airflow, preserving the perpendicular force. CA is not blindly
substituted for CD. Body-axis components are recalculated for Flight; the
static drag component follows the same multiplier. Native damping-load
increments remain unchanged. Geometry, calibration inputs, native tables and
ordinary RocketPy algebraic equations are not overwritten.

CP uses a distance measured aft from the nose, in meters:
``x_CP_new = x_CP_native * (1 ± 0.02)``. Consequently the change in stability
margin is ``delta_margin = ±0.02 * x_CP_native / diameter``, in calibers,
not ±2% of the nominal stability margin. The native small-angle CP is computed
at each Mach and case reference. Added drag acts there, the new static force
is transferred to the shifted CP, and the native residual moment couple is
retained. Both rocket axial coordinate orientations and wet/dry case CG
references are respected. Undefined native CP remains unavailable.

``upper`` means more drag and CP farther aft from the nose; ``lower`` means
less drag and CP farther forward. These are two correlated scenarios. They
do not guarantee the greatest/smallest apogee or stability margin among all
independent combinations of drag and CP errors. No bias correction is applied
to the nominal run. The report still gives the fraction of ascent time outside
Wu's Mach/total-incidence coverage, identifying the scope of that source.

Inspect the calculated results without running simulations during printing::

    flight.prints.apogee_conditions()
    flight.prints.stability_margin()
    uncertainty_m = flight.datcom_apogee_uncertainty
    report = flight.datcom_uncertainty
    print(report.policy)
    print(report.cases)  # nominal / upper / lower; SI units, stability in calibers
    print(report.uncertainty("max_speed"))  # m/s, ascent only
    upper = report.flights["upper"]
    lower = report.flights["lower"]

Printed ``apogee ± datcom_uncertainty`` is the **largest absolute departure
from nominal**, in meters, for ASL and AGL. It summarizes potentially asymmetric
scenario results; it is not a standard deviation, confidence interval or
percentile. The charts use the actual asymmetric range across all three
apogees. Peak ascent speed is inertial speed; apogee freestream speed is
relative to the wind. Each scenario uses its own apogee and rail-exit time.
Incomplete or failed ascents are unavailable, not zero-width uncertainty.

Replay rockets and environments are copied before nominal integration. They
reuse native tables and atmosphere schedules, stop at apogee and do not launch
new DATCOM runs for unchanged decks. They do not refit the atmosphere schedule
or adaptive grid: check each replay's envelope. Callbacks/controllers execute
again; their external side effects remain the caller's responsibility. Python
and NumPy global random states are restored after replay. This covers only the
stated aerodynamic sensitivity, not wind, mass, thrust, recovery or measurement
uncertainty. ``datcom_uncertainty="upper"`` or ``"lower"`` runs an individual
case. Replay Flight loads/stability use its local scenario model; the shared
Rocket's reporting curves remain nominal.

Body Profile Selection
----------------------

After adding a DATCOM fin set, select the actual body geometry::

    rocket.add_nose(length=0.81, position=0, kind="von karman")
    rocket.add_tail(top_radius=0.0895, bottom_radius=0.0835,
                    length=0.125, position=5.137, kind="ogive")

``NoseCone`` and ``add_nose`` use their existing ``kind`` argument.
Supported DATCOM nose profiles are conical, tangent-ogive, von Karman,
L-V Haack and power series. Other existing RocketPy nose kinds retain
their ordinary component behavior.

``Tail`` and ``add_tail`` accept ``kind="conical"`` (the backward-compatible
default) or ``kind="ogive"``. The ogive is a circular arc tangent to the upstream
cylinder, joining the given radii over the given axial length in meters.
Ogive geometry, drawing and DATCOM ``TAFT`` follow that profile, including
fin attachment to its local radius. Ogive tails have no component aerodynamic
model; the whole-vehicle DATCOM tables supply their loads and CP. Add a DATCOM
fin set before calling ``add_tail(kind="ogive")``; ordinary rockets raise
``ValueError``. The kind is selected at construction and is read-only.
Conical tails use the original :class:`rocketpy.Tail` without any changes to
its equations or serialization. Profile selection survives serialization.

The default DATCOM alpha grid has 19 points over ±20 degrees, with 13 points
within ±2 degrees. Explicit ``datcom_options["alpha"]`` overrides it. Grid
convergence and the actual flight-incidence envelope still need checking.

Executable and Tables
---------------------

Supply an executable for the current operating system and processor, with
execute permission where required. RocketPy does not distribute DATCOM or
automatically run a Windows executable on another platform.
The integration targets Asteria's revision 3/99 input/output convention.
Its body incidence angles are the projected angles
``alpha=atan(v_down/v_forward)`` and ``beta=atan(v_right/v_forward)``.
The native regression suite checks this using an axisymmetric-body force
ratio. Verify angle definitions before substituting another DATCOM revision.

Flight automatically prepares powered and coast databases in two executable
invocations. Each runs in a separate temporary
directory containing a copy of the executable and a generated ``for005.dat``.
RocketPy parses ``for006.dat`` and removes the directory on success or failure.
Every deck includes ``DAMP``; missing or invalid damping tables raise an error.
No ``PROTUB`` or plume-interaction calculations are included.
Fixed-sideslip cases share each invocation when a beta grid is requested;
they do not require additional executable copies or persistent output files.

By default the powered case omits ``DEXIT`` to exclude all base drag, preserving
an explicit zero-powered-base-drag assumption. For a motor whose nozzle is
smaller than the rocket base, this also removes the exposed annular base drag.
To model that known annulus instead, configure::

    datcom_options={"powered_base_drag": "nozzle"}

This writes ``DEXIT=2*motor.nozzle_radius`` for the powered case. The nozzle
must have positive radius and fit inside the aft body. ``"full"`` writes
``DEXIT=0.`` for a full-base diagnostic; ``"none"`` is the default.
If the motor's radius is an unverified placeholder, supply the measured nozzle
exit diameter independently, in meters::

    datcom_options={"nozzle_exit_diameter": 0.03}

An explicit diameter selects nozzle mode unless ``powered_base_drag`` is also
specified; combining it with ``"none"`` or ``"full"`` raises an error. This
changes only the aerodynamic input, leaving the motor's geometry and thrust
calculations unchanged. Motor case diameter is not nozzle exit diameter.
The coast case always uses ``DEXIT=0.`` for full base drag. ``BASE`` stays false:
that flag controls jet/plume interaction, not the annular base-area calculation.
Nozzle mode therefore requires no plume-flow inputs. Powered/coast selection
switches at ignition and burnout independently of propellant fraction.
The flown-rocket comparison example uses nozzle mode; AST-HER retains its
explicit zero-powered-base-drag assumption.

Call ``rocket.prepare_datcom()`` explicitly to inspect drag or stability before
creating a Flight. It returns the complete aerodynamic model. Identical decks
and executable metadata reuse the in-memory result. A geometry, reference-CG,
flight-condition or executable change rebuilds both cases. Re-add a modified
motor to the rocket after changing its tanks, following the normal mass-property
workflow.

Native approximation and fallback messages are retained in
``rocket.prepare_datcom().tables[False].diagnostics`` for coast, and
``tables[True].diagnostics`` for powered flight. Some messages require native
``PART`` output to be printed. The manual's section 4.2.2 explains that
``CREST CRITICAL MACH NUMBER EXCEEDED`` selects approximate airfoil properties;
it does not by itself mean the complete coefficient tables are invalid. These
messages deserve inspection even when the executable finishes successfully.
When parsing ``PART`` output, damping must come from the final total-vehicle
tables, after the component tables (section 4.2.3). The parser selects these
totals. A native regression independently reproduces the manual's two-fin-set
worked example (Figures 19--21), including its published CN, CM and CA.

Section Inputs and Flight Conditions
------------------------------------

``thickness_to_chord`` is the total symmetric section thickness divided by
chord. ``max_thickness_position`` locates the beginning of the maximum-thickness
region of HEX sections; ``flat_length`` gives its length. Both are fractions
of chord. Set ``flat_length=0`` for a diamond section. ARC uses thickness and
leading-edge radius. Section inputs accept a scalar or one value per span
station; ``leading_edge_radius`` is in meters. A zero tip chord is supported.

Use ``upper_thickness_to_chord`` and ``lower_thickness_to_chord`` for separate
upper/lower dimensions. ``lower_max_thickness_position`` and
``lower_flat_length`` default to their upper-surface equivalents. Thickness
may be zero on one surface, provided the total thickness is positive.

The trapezoidal helper also accepts ``section_kind="NACA", naca="0012"``.
Four-digit sections can omit the series prefix. Other supported designations
include ``"5-23012"``, ``"6-64-005"`` and the manual's modified-series formats.
The fin-set number is inserted into the NACA control card after sorting.
For custom sections, use ``section_kind="USER"`` with
``airfoil_coordinates={"x": [...], "upper": [...], "lower": [...]}``, or
``x``, ``mean`` and ``thickness`` arrays. Supply 3 to 50 increasing chord
stations from zero to one. All coordinate values are fractions of chord.

This DATCOM version includes camber effects on normal force and pitch moment
only for NACA and USER sections at Mach 0.8 and below. HEX/ARC camber affects
drag calculations only.
The airfoil section module assumes sharp trailing edges. According to the
manual's section 4.2.2, non-sharp trailing edges are treated as an ideal airfoil
approximation. A measured blunt plate supplied as USER coordinates therefore
still needs validation; supplying its contour does not remove that native
method limitation.

Multi-station Planforms and Panel Placement
-------------------------------------------

``add_datcom_fins`` creates a :class:`rocketpy.DatcomFreeFormFins` object from
2 to 10 exposed ``span_stations``, ``chords`` and ``leading_edges``. Span
stations begin at zero on the body; leading edges begin at zero relative to
``position``. Chords and distances are in meters. For example::

    fins = rocket.add_datcom_fins(
        n=4, position=-0.9, path="/path/to/datcom",
        span_stations=[0, 0.05, 0.10],
        chords=[0.25, 0.20, 0],
        leading_edges=[0, 0.04, 0.10],
        section_kind="HEX", thickness_to_chord=[0.02, 0.025, 0.03],
    )

All stations share one airfoil section type, with proportions allowed to vary
by station. The example has a triangular tip. NACA and USER designate a common
normalized section for the entire fin set.

``panel_angles`` specifies each panel's azimuth using DATCOM PHIF: degrees
clockwise from the top vertical when viewed forward from behind the rocket.
``dihedral`` specifies each panel's constant GAM in degrees; this tilts the
span direction away from its radial ray. ``incidence`` specifies fixed panel
angles in DATCOM's convention, where positive incidence produces a negative
roll increment. Alternatively use RocketPy's ``cant_angle`` convention, with
the opposite roll-forcing sign. Do not specify both. Incidence and dihedral
accept a scalar or one value per panel. Time-varying actuators and trailing-edge
flaps are not part of this interface.

``auto_body_radius=True`` uses DATCOM's ``SSPAN(1)=0`` convention to mount the
root at the interpolated body radius at root mid-chord. This supports roots
on the selected boattail profile as well as the cylindrical body. After preparation,
rocket drawings use that attachment radius and the actual panel placement.
Fixed-incidence drawings use the default half-root-chord hinge and zero skew.

``datcom_options`` accepts:

* ``mach``: 2 to 20 positive, increasing Mach values. Without a reference
  flight, the default is a provisional 20-point grid; see adaptive coverage below.
* ``alpha``: 2 to 20 increasing angles in degrees, including negative angles,
  exactly one zero and positive angles.
* ``beta``: ``[0]`` or an increasing signed sideslip grid containing zero, in
  degrees. The default is ``[0]`` for the symmetric configuration and
  ``[-5, 0, 5]`` for nonstandard panel placements, dihedral, incidence, or
  fin sets with fewer than three panels.
* ``altitude``: meters above sea level, either a scalar or one value per Mach.
  These specify a Mach/altitude schedule, not an independent altitude grid.
  Separate schedules can be provided as
  ``{"power_on": [...], "power_off": [...]}``.
* ``timeout``: maximum execution time in seconds for each case.
* ``reynolds``: optional positive Reynolds number per meter, either a scalar,
  one value per Mach, or separate ``power_on``/``power_off`` schedules. It
  selects native ``MACH``/``REN`` conditions instead of ``MACH``/``ALT``.
  The API uses inverse meters, not Reynolds number based on diameter or chord.
* ``body_length``: optional cylindrical section length in meters, excluding the
  nose and boattail. With a tail, its position determines this length and an
  explicit value must agree. Without a tail, the nozzle position supplies the
  inferred aft end unless ``body_length`` is specified.

Atmosphere from a preliminary flight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``altitude`` alone, DATCOM derives atmospheric properties using its 1962
Standard Atmosphere; altitude defaults to sea level. It does not automatically
read a RocketPy Environment. To use the actual launch atmosphere, first run an
ordinary RocketPy model with the intended environment, then configure the
DATCOM rocket from that flight::

    atmosphere = datcom_rocket.set_datcom_atmosphere(ordinary_flight)
    datcom_flight = Flight(rocket=datcom_rocket, environment=env, **launch_options)

    # Optional explicit refinement using the resulting DATCOM trajectory:
    atmosphere = datcom_rocket.set_datcom_atmosphere(datcom_flight)
    refined_flight = Flight(rocket=datcom_rocket, environment=env, **launch_options)

Use separate rocket instances when retaining ordinary and DATCOM results.
Both must have the same motor burn interval. The method does not create flights
or run the executable; the next Flight regenerates native tables.

Adaptive coverage uses all **20 Mach slots**, always including **0.1 and 1.0**.
Fixed points from 0.8 to 1.2 concentrate resolution near Mach 1; the remaining
points adapt to the reached regime. The upper bound includes 15% headroom plus
0.02 Mach and is at least 1.2. Mach zero uses the positive 0.01 endpoint. Values
are spaced by at least 0.01 to survive the native two-decimal output format.
One shared grid covers the union of powered and coast flight conditions, with
separate Reynolds schedules for the two native cases.

Alpha adapts too: 19 signed points stay concentrated near zero, retaining at
least ±20 degrees and extending beyond observed incidence by 15% plus 2 degrees.
Only samples with dynamic pressure at least ``minimum_dynamic_pressure=100`` Pa
set the extent; no loaded samples retain ±20 degrees. This excludes ambiguous
pad/near-apogee flow directions from grid selection. Set a lower threshold if
those low-load conditions matter. The returned ``coverage`` report records the
scope, excluded sample count and separate phase Mach/loaded-alpha maxima.

``samples=1000`` supplies uniform samples, augmented by actual solver states
and burn boundaries. The default covers the rail, powered ascent and coast to
apogee; descent and parachute flight are excluded. Adaptive envelopes
approaching 90 degrees are rejected: the current forward-flight load mapping
cannot faithfully represent tail-first flow. Extending a grid does not validate DATCOM's high-incidence methodology.

Explicit ``mach=[...]`` and ``alpha=[...]`` arguments fix custom grids instead,
for controlled studies or tunnel conditions; custom Mach grids may omit the
adaptive anchors. Every call without those arguments recalculates both grids
from the new reference flight. The example reproduction pipeline applies this
again after its initial DATCOM simulation. No measured apogee selects grid points.

At each sampled altitude in meters ASL, the reference Environment supplies
actual density, speed of sound and dynamic viscosity. Reynolds number per meter
is ``rho * Mach * speed_of_sound / dynamic_viscosity``. Native ``REN`` replaces
``ALT``, bypassing DATCOM's standard-atmosphere Reynolds calculation. Pressure
(Pa), temperature (K), altitude and Reynolds schedules are returned for provenance;
pressure and temperature are not separately sent to DATCOM. Flight forces still
use the simulated flight's actual density and relative airspeed.

Separate powered and unpowered ascent schedules assign one atmospheric state
per Mach. Duplicate sampled Mach values retain the earliest sample. Outside
the reached Mach range, atmospheric state is clamped to its endpoint, but speed
is recomputed at the requested Mach. This avoids clamping Reynolds number to a
different speed. Multiple crossings, descent and separated stages require a
richer database than this approximation. No convergence is assumed: repeat
explicitly with the DATCOM trajectory if needed, comparing apogee and schedules.
Existing ordinary RocketPy equations are unchanged.

All DATCOM fin sets must use the same executable and options. At most four fin
sets with 1 to 8 panels each are supported. Sets are automatically numbered
from nose to tail. Overlapping axial planform envelopes are rejected;
the root chord center must lie on the body's axial extent. A trailing edge
may extend aft of the base, as verified with the native 3/99 executable.
Separation below four exposed semispans triggers a warning about DATCOM's
upstream vortex-roll-up approximation. Existing ordinary
trapezoidal fin sets are included with a default symmetric HEX section of 2%
thickness and 25% edge bevels. Other aerodynamic surface types are rejected
rather than omitted. Exactly one unblunted conical, tangent-ogive, von Karman,
L-V Haack or power-series nose is required. An optional conical or tangent-ogive boattail must
connect to the cylindrical body. Off-axis geometry is unsupported.

Explicit Calibration
--------------------

After adding DATCOM fins, attach calibration to the whole rocket::

    rocket.add_datcom_calibration(
        roughness_height=30.48e-6,  # equivalent sand roughness, meters
        leading_edge_radius=0.0005,  # optional radius, meters
    )
    rocket.prepare_datcom()

``roughness_height`` is an equivalent sand roughness height, not the measured
arithmetic mean roughness (Ra). Carbon-fibre construction with a vinyl wrap
does not determine this value: seams, finish, exposed edges and other losses
can differ between vehicles. A fit to apogee is an effective calibration and
does not identify the real surface finish. No material-specific value is
silently selected.

The carbon-fibre/vinyl assumption applies only where more specific information
is missing. It does not supersede supplied masses, inertias, motor data,
geometry, measured coefficients or calibration values. Fitted values represent
effective aerodynamic losses and must not be described as measured material
properties. In the four-rocket apogee example, physically bounded leading-edge
radius sweeps did not reach the targets; separate roughness fits did. A shared
GENESIS roughness did not fit all four rockets, and Andromeda's much larger
fitted roughness indicates that the fit can absorb other missing losses.

The conversion uses dimensionless DATCOM ``RHR``, with equivalent sand
roughness equal to three times RHR in microinches. For example,
30.48 micrometers produces ``RHR=400``. This avoids a verified discrepancy:
the local 3/99 executable treats ``ROUGH`` as inches even with ``DIM M``,
whereas the 5/97 Missile DATCOM guide Table 4 specifies centimeters for metric
``ROUGH``. The RHR/roughness conversion was checked against that executable;
other native versions should be independently verified.

``leading_edge_radius`` overrides ``LER`` at every station of every supported
fin set without modifying the stored fin definitions. ``None`` retains the
individual fin radii. NACA sections define their own edge shape and reject an
override. Use physically compatible values and choose fit bounds from the
actual fin section rather than from the material label alone.

Calling ``add_datcom_calibration`` replaces the previous settings; omitted
roughness defaults to zero and omitted radius to ``None``. Call without
arguments to reset. ``rocket.datcom_calibration`` returns a copy of the current
settings. Changes invalidate prepared native tables, which the next
``prepare_datcom`` or Flight rebuilds. Rocket serialization preserves the
settings. The method requires DATCOM fins and does not fit values automatically.
No multiplier or correction is applied to RocketPy's ordinary aerodynamics;
all calibrated aerodynamic properties still come from native DATCOM.

Wind-Tunnel Coefficient Fitting
-------------------------------

Mach, alpha and measured CD are sufficient for a roughness fit::

    report = rocket.fit_datcom_calibration("wind_tunnel.csv")
    rocket.prepare_datcom()

For example, a CSV header can be ``mach,alpha,CD``. A column mapping with
numeric vectors is also supported. Headers are case-insensitive. Alpha and
beta are degrees and default to zero when omitted. The default reference area
is the rocket frontal area; supply ``reference_area`` in square meters when
the measurements use another normalization. The fit uses static coefficients
at zero body rates and zero incidence rate.

Optional measured columns are ``CL``, ``CA``, ``CN``, ``CY``, ``CM``, ``CLN``
and ``CLL``, together with ``beta``, ``power_on`` and
``reynolds_per_meter``. Blank coefficient cells are allowed if each row has at
least one finite measurement. Unknown columns are rejected. Body coefficients
use DATCOM's axes and signs. ``CL`` is pitch-plane lift. By default ``CD``
opposes the full flow, including sideslip; ``drag_axis="stability"`` selects
the pitch-plane drag projection. Native reporting conventions can differ at
nonzero beta, so match the experiment's convention explicitly.

Moment measurements require ``reference_length`` in meters and
``moment_reference`` in the rocket's axial position coordinates. Native moments
are transferred to that axial reference before comparison. An optional
``lateral_reference_length`` specifies the yaw/roll normalization length;
otherwise it equals ``reference_length``. Off-axis moment references are not
supported. Positive ``weights={"CD": ..., "CM": ...}`` supplies coefficient-wise
least-squares weights; for a known uncertainty sigma, use ``1/sigma**2``.

The default fitted parameter is ``roughness_height``, bounded to 0--300
micrometers. To fit an edge radius, supply explicit physically justified
bounds in meters::

    report = rocket.fit_datcom_calibration(
        "wind_tunnel.csv",
        parameters=("roughness_height", "leading_edge_radius"),
        bounds={
            "roughness_height": (0, 100e-6),
            "leading_edge_radius": (0, 0.001),
        },
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        moment_reference=rocket.center_of_dry_mass_position,
        max_evaluations=200,
    )

``model_scale`` is the tunnel-model length divided by the full-rocket length,
defaulting to 1. Reference areas and lengths are supplied for the model and
converted to full-scale equivalents. The default area is the scaled model
frontal area. ``moment_reference`` always remains in full-rocket axial
coordinates. Reported fitted dimensions are full-scale equivalents, not
measured model-surface properties.

``reynolds_per_meter`` supplies tunnel conditions, with one positive value per
Mach and power state. It is converted to full-scale Reynolds number per meter
by multiplying by ``model_scale``. These conditions affect only the fitting
probes; Flight keeps the original rocket options. Without this column, the
probes use the existing DATCOM atmospheric conditions. Nonzero measured beta
requires a beta table; measurements outside the native Mach/angle envelope
are rejected. ``power_on=False`` is the default, or use a per-row column
containing 0 or 1 to match the measured base/nozzle condition.

Successful fits apply their settings through ``add_datcom_calibration``.
The JSON-compatible report contains initial and final coefficient RMSE,
weighted RMSE, residual rows, references, fitted inputs, bounds, boundary
flags and native evaluation count. Failed data validation, execution or
optimization leaves calibration and the prepared model unchanged. The fit
does not replace any aerodynamic coefficient; damping remains native DATCOM.
Static measurements do not identify forced-oscillation damping derivatives,
and this interface does not ingest measured dynamic derivatives.

A drag-only curve may not uniquely identify two fitted parameters. A bounded
numerical fit is not guaranteed to be a unique or global solution; use
additional channels and independent test conditions to assess predictive
accuracy. Matching the fitted measurements, or a single apogee, is not an
independent validation. Real measured coefficients and their reference conditions
are required for wind-tunnel validation.

Accuracy of Apogee Predictions
------------------------------

Ordinary RocketPy can predict apogee more accurately than RocketPy using
DATCOM when its supplied drag curve better represents the real vehicle.
In the four untuned comparisons, ordinary RocketPy was closer to the
notebook-reported flight altitude for Cavour, GENESIS and Andromeda; DATCOM
was closer for EREBUS. This does not establish a general ranking. A successful
fit to those same apogees or coefficients is calibration; predictive accuracy
must be assessed using independent data. Asteria / AST-HER has not flown and
its results are simulation predictions, not flight validation.

The :doc:`recorded-flight comparison </examples/datcom/rocket_comparison>`
compares altitude, upward vertical velocity, CD and stability margin between
ordinary RocketPy, DATCOM and measured records where available, including
percentage errors. Apogee-fitted constant CD is explicitly inferred calibration,
not measured CD. Missing measurements and unsupported source configurations
remain visible; archived outputs are labelled. No inferred CD replaces the
native DATCOM predictions shown in that comparison.

CG, Damping and Interpolation
-----------------------------

RocketPy derives the filled CG from the dry rocket, dry motor, and the motor's
propellant masses and positions, including liquid tanks. The input
``center_of_mass_without_motor`` still excludes the entire motor; it is not
the assembled empty-tank CG. DATCOM uses the filled CG at ignition for the
powered run and the assembled empty CG for coast. Physical moments are
transferred to the instantaneous CG throughout the burn. Coefficients are not
blended by fuel fraction, which would mix base drag with CG migration.

Static coefficients and all eleven DAMP derivatives interpolate linearly in
Mach, signed angle of attack, and sideslip when a beta grid is available.
With only ``beta=[0]``, sideslip uses the first-order lateral derivatives.
The full beta database bypasses that expansion to avoid counting it twice.
The reported ``CD`` is DATCOM's stability-axis drag, which differs from
wind-axis drag at nonzero sideslip. Flight applies the body-axis force
coefficients directly, including during the constrained rail phase.
The parser retains DATCOM's per-degree units; the load model
converts body rates in rad/s to the required nondimensional rates. ``CNAD`` and
``CMAD`` use a simultaneous incidence-rate/acceleration solve within each ODE
evaluation, including local rotation and wind shear. Adaptive solver call
history does not determine incidence rate.

Mach and angle are clamped to the generated envelope. The returned model records
``outside_mach_envelope``, ``outside_alpha_envelope`` and, for full beta grids,
``outside_beta_envelope`` excursions. Low-speed
damping is disabled below 1 m/s to avoid dividing by vanishing airspeed.
Reverse axial drag opposes tail-first motion, but the bounded continuation
outside the small-angle envelope is not validated descent aerodynamics.
The 3 DOF solver uses the force coefficients and retains its usual simplified
attitude behavior; use 6 DOF to simulate rotational damping.

``cp_position``, ``static_margin`` and ``stability_margin`` use the coast
small-angle DATCOM derivatives after preparation. Before preparation they
are undefined (NaN), rather than displaying component-model results. CP is
also undefined when the normal-force slope vanishes.

The parser is adapted from the Asteria-Flight-Dynamics DATCOM implementation.
Deck definitions follow the Missile DATCOM user's guide, Tables 5, 10 and 11.

The executed :doc:`AST-HER comparison </examples/datcom/asteria_comparison>`
uses the saved Asteria configuration and atmosphere. It compares ordinary
RocketPy using the supplied Asteria drag curve with native DATCOM. AST-HER has
not flown: percentages are model-to-model differences, not measured-flight
errors. The ordinary conical-tail approximation and native ogive tail are
explicit. Mass and CG histories are checked to match.


Interpreting disagreement with source flight examples
-----------------------------------------------------

Ordinary RocketPy's whole-vehicle drag curves are supplied independently of its
nose and fin normal-force models. Replacing them with native DATCOM therefore
changes the drag model even with identical mass and planform. The
:doc:`comparison notebook </examples/datcom/rocket_comparison>` keeps supplied
CD, native CD, inferred apogee-fitted CD and measured values separate. One
constant fitted to apogee does not uniquely identify a physical drag curve.

In particular, a constant thickness/chord ratio describes a thickness taper.
For a measured constant-thickness plate with nonzero root and tip chords, supply
separate ratios, for example ``thickness_to_chord=(t / root_chord, t / tip_chord)``
with all lengths in meters. Match bevels, section contour and edge radius to
hardware. Smooth-surface roughness zero is an assumption, not a measured vinyl
finish. Carbon fibre and vinyl do not determine these missing numerical inputs.
Use measured geometry or the calibration API; an apogee-fitted roughness is an
effective loss calibration, not an identified material property.

The manual audit also found unverified nozzle inputs in the recorded-rocket
examples: GENESIS supplies an 84.708 mm exit diameter through its motor radius,
while the thrust-file header lists a 75 mm motor case; Bella Lui similarly
supplies 88.9 mm for a listed 54 mm case. These dimensions need hardware
verification, rather than automatic halving or substitution of case diameter.
For GENESIS, even the full-powered-base-drag sensitivity run reduces apogee
only from 3297 m to 3223 m, versus the recorded 2916 m. Nozzle-area uncertainty
alone does not explain that discrepancy. The existing comparison decks omit
``PART``, so correcting component-versus-total damping parsing does not change
their saved results.

Equivalent metric/English decks also expose a small native transonic
sensitivity: for the controlled Andromeda check, CA at Mach 0.9 is 0.396
versus 0.404, with matching values at Mach 0.3 and 0.6. The difference comes
from body pressure/wave drag in the native partial output. The audit retains
this numerical discrepancy; an independent transonic reference is needed to
decide which prediction is more accurate.
