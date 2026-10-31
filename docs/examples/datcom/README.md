# DATCOM comparisons

**Work in progress:** DATCOM currently produces poor flight predictions for several comparison rockets. The discrepancies are unresolved and are being investigated; these results should not be treated as validated predictions.

DATCOM flights calculate two engineering sensitivity ascents around the nominal
result. The current rule is **at least ±20% static wind-axis CD at every Mach
and AoA**, with **±2% nose-to-CP distance**. Within Wu et al.'s Mach 0.8–3.02 /
alpha ±6° envelope, use a larger native-relative axial-force discrepancy as a
CD-error proxy. Outside that envelope use the 20% excursion. At zero incidence
this scales the whole CD-versus-Mach curve; at finite incidence the added force
follows the relative airflow and preserves perpendicular force. Static pitch
and yaw moments are transferred consistently; native DAMP terms are retained.
The earlier independent CN and CM-alpha scaling is no longer active.

The design-accuracy reference is J. T. White (1995), *An assessment of missile
DATCOM prediction accuracy relative to generic Body+Wing+Tail missile pitch
aerodynamics*, 13th Applied Aerodynamics Conference, AIAA 95-1893,
[doi:10.2514/6.1995-1893](https://doi.org/10.2514/6.1995-1893).
The supplied filename says 2012, but the paper is from 1995. It assesses
revision 6/93 on 15 configurations and over 900 cases at Mach 1.5–5.0,
AoA 0–24°, roll 0°/45°, without control deflections. White's cited preliminary
criteria concern **normal force** and CP relative to **body length**; he does
not establish universal ±20% CD or ±2% nose-to-CP confidence bounds. The two
percentages used here are the requested engineering assumptions, extended
throughout flight. His normal-force bias/scatter and ambiguous CP statistics
are not applied as a nominal correction or a statistical distribution.

The larger drag discrepancies adapt Table 2 of Hanzhou Wu, Min Gao, Weidong
Song, Zhao Jie and Yi Wang (2019), *Accuracy Analysis of Aerodynamic Calculation
of Two-dimensional Ballistic correction Projectile based on Missile Datcom*,
**IOP Conf. Ser.: Mater. Sci. Eng. 612**, 032096,
[doi:10.1088/1757-899X/612/3/032096](https://doi.org/10.1088/1757-899X/612/3/032096),
under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
For signed measurement-relative error `e`, the native-relative magnitude is
`abs(e / (1 + e))`, with `e` expressed as a fraction. The drag excursion is
`max(0.20, converted_Wu_error)` inside the paper's envelope and `0.20` outside.
CA discrepancy is transferred to CD as an explicit assumption. The raw CN and
CM-alpha tables remain for provenance but no longer set sensitivity bounds.

CP is measured from the nose: `x_CP_new = x_CP_native * (1 ± 0.02)`.
The stability-margin change is `±0.02 * x_CP_native / diameter`, in calibers,
not 2% of the original margin. Coordinate origins and powered/coast CG
references remain consistent. Upper/lower mean more/less drag with CP
farther aft/forward; they are two correlated scenarios, not an exhaustive
search over independent combinations. The nominal prediction is uncorrected.

Printing shows three apogees, peak ascent speeds, apogee freestream speeds
and stability margins. Apogee **± uncertainty** is the largest absolute
scenario departure in meters, not a confidence interval or percentile.
`flight.datcom_uncertainty.cases` exposes the values and
`flight.datcom_apogee_uncertainty` exposes the scalar. Disable replays using
`Flight(..., datcom_uncertainty=False)`. See the
[full method and limitations](../../user/rocket/datcom.rst).

The regenerated [sensitivity results](comparison_data/uncertainty_sensitivity.csv)
cover all 11 currently supported recorded-rocket DATCOM models, with the
[machine-readable report](comparison_data/uncertainty_sensitivity.json) and
[example printed output](comparison_data/genesis_uncertainty.txt). Reproduce
all of them with
`python docs/examples/datcom/comparison_data/uncertainty_sensitivity.py`.

The [apogee chart](comparison_data/apogee_uncertainty_comparison.png) compares
RocketPy's supplied-CD prediction, nominal DATCOM with **asymmetric scenario
whiskers**, and recorded apogee as a dashed line. Green/red rank nominal
accuracy. A [compact three-rocket view](comparison_data/apogee_uncertainty_comparison_focus.png)
and [stability chart](comparison_data/stability_uncertainty_comparison.png)
are also exported, with [CSV values](comparison_data/apogee_uncertainty_comparison.csv).
All charts and tables are in [the comparison notebook](rocket_comparison.ipynb);
the original 16-rocket chart still identifies unavailable cases separately.
No recorded stability margin is available for the 11 paired DATCOM models,
so their stability whiskers compare simulations only. Rebuild charts without
running DATCOM using
`python docs/examples/datcom/comparison_data/plot_apogee_uncertainty.py`.

CD-versus-Mach charts are exported for all 11 rockets, separately for
[powered flight](comparison_data/cd_powered_comparison.png) and
[coast](comparison_data/cd_coast_comparison.png), with a
[compact three-rocket view](comparison_data/cd_comparison_focus.png).
Blue is RocketPy's supplied drag curve; orange is nominal DATCOM; the orange
band uses the active 20%-minimum CD rule at **alpha = beta = 0°**. The dotted
black line is the **apogee-fitted constant CD**, inferred using ordinary
RocketPy; it is not measured CD and is not unique. No measured CD records
are supplied, so these charts do not rank drag-prediction accuracy. Grey
background marks Mach values above the reference ascent peak, including
unflown transonic points. Curves retain their saved Mach/atmosphere schedules;
the 2% CP shift does not directly change zero-incidence static CD.
The [AST-HER chart](comparison_data/asteria_cd_uncertainty_comparison.png)
compares the same two models without a fitted or measured CD: **AST-HER has
not flown**. Its saved RocketPy curve lacks the Mach 0.01 sample; that point
remains a gap rather than an invented value. Both existing notebooks include
their corresponding CD charts.
PNG/SVG exports and the complete
[rocket curve data](comparison_data/cd_uncertainty_comparison.csv) and
[AST-HER curve data](comparison_data/asteria_cd_uncertainty_comparison.csv)
can be regenerated without running DATCOM:
`python docs/examples/datcom/comparison_data/plot_cd_comparison.py`.

Recorded apogees fall inside the scenario ranges for **4 of 11 rockets**. RocketPy's supplied-CD nominal prediction is closer for **8 of 11**. Inclusion is a diagnostic, not evidence of a calibrated confidence level.

| Rocket | Recorded AGL (m) | RocketPy (m) | DATCOM nominal (m) | DATCOM scenario range (m) |
| --- | ---: | ---: | ---: | ---: |
| GENESIS | 2916 | 3045 | 3297 | 3068–3584 |
| Astra | 3250 | 3043 | 3696 | 3437–4032 |
| Andromeda | 3443 | 3616 | 4171 | 3794–4832 |

These bars do not cover every recorded apogee. Geometry assumptions, drag modelling and other uncertainties remain unresolved.

Only two example notebooks are included:

- [Recorded rockets](rocket_comparison.ipynb): DATCOM, ordinary RocketPy and
  recorded altitude, upward vertical velocity, CD and stability margin where
  available, with signed percentage errors. Apogee-fitted constant CD is
  explicitly inferred calibration, not a measurement or native correction.
  Apogee bars cover all 16 rockets: green is the closer paired prediction and
  red is the farther one, with recorded apogee as the dashed reference.
- [AST-HER / Asteria](asteria_comparison.ipynb): the same configuration, tanks
  and atmosphere using ordinary RocketPy's supplied Asteria drag curve versus
  native DATCOM. **AST-HER has not flown.** Percentages are model differences.

Missing measurements are shown as unavailable. Scalar CD comparisons use Mach
0.6 and zero incidence; static margin uses loaded ignition and Mach zero
(DATCOM uses its lowest positive Mach). NDRT's measured stability conditions
are unspecified. Velocity uses recorded vertical/altimeter channels, never a
numerical derivative presented as a measurement. Fitted CD uses one constant
for powered and coast drag in the ordinary source model; it is not unique.

Source models and raw flight records stay in their original repository folders.
`comparison_data/` contains reproduction scripts, source hashes, reference
notes, QA masks, fitted residuals, values and model histories. Unspecified
construction assumes carbon fibre with vinyl, without inventing a measured
roughness or fin section. The recorded-rocket DATCOM models retain their
explicit smooth 2% HEX / 5% bevel / sharp-edge assumptions. AST-HER uses its
saved Asteria section inputs. All native runs include DAMP and use adaptive
20-point Mach grids and signed alpha coverage from rail through apogee.

Reproduce from the repository environment:

```sh
python docs/examples/datcom/comparison_data/reproduce_rockets.py
python docs/examples/datcom/comparison_data/prepare_comparisons.py
```

Set `DATCOM_EXECUTABLE` to a native executable when the sibling Asteria copy is
unavailable. The latter command fits effective CD against the recorded apogees;
those are calibration results, not independent validation. Notebook execution
reads the saved comparison outputs and requires no new native runs.

The [apogee bar chart](comparison_data/apogee_comparison.png) is also exported
as [SVG](comparison_data/apogee_comparison.svg), with values in
`comparison_data/apogee_comparison.csv`. Rebuild only these exports with
`python docs/examples/datcom/comparison_data/plot_apogee_comparison.py`.
Labels show signed percentage error. P50/P75/P90/P95 in
`comparison_data/apogee_error_percentiles.csv` are empirical absolute-error
percentiles across the same paired rockets, not per-flight uncertainty.
Missing DATCOM results remain unavailable, and archived RocketPy values are
marked. Colour ranking is withheld when the competing prediction is absent.

The supplied CD group includes RASAero, constants, curves of unspecified
provenance and Juno's explicitly CFD-scaled curve. It is not universally CFD.
RocketPy has no separate built-in geometry-to-CD predictor. The example loader
now preserves legacy post-construction `power_on_drag`/`power_off_drag` edits
in the seven-input solver coefficient; this corrects stale curve use for
Juno and Bella Lui without changing any RocketPy equations. The two affected
ordinary/DATCOM comparisons and the associated metrics have been regenerated.

The [manual audit](comparison_data/datcom_usage_audit.json) records an
independent reproduction of the guide's worked example, `PART` invariance for
all 11 native comparison models, source nozzle dimensions and base-drag
sensitivity flights. Reproduce it with:

```sh
python docs/examples/datcom/comparison_data/audit_datcom_usage.py
```

The audit found and corrected component damping being selected instead of
total-vehicle damping when parsing `PART` output. These comparison decks do
not request `PART`, so their existing predictions are unaffected. GENESIS and
Bella Lui have source nozzle exit diameters larger than their thrust files'
listed motor case diameters; those inputs need hardware verification.
GENESIS remains 10.5% above recorded apogee even with full base drag during
the burn (3,223 m versus 2,916 m). This bounds the nozzle-area contribution;
it does not establish a measured nozzle correction. Fin section, bevel,
edge radius and surface finish remain assumptions. The guide also describes
approximate airfoil properties after its crest-critical-Mach warning, which
all 11 native grids produce. Parsed native diagnostics are now retained.
Some warnings concern unflown grid points. The guide also assumes sharp
trailing edges in its airfoil section solver and approximates blunt sections
as ideal airfoils. Neither a carbon-fibre label nor USER coordinates establish
that these approximations describe the real fin edges.

Explicit trailing-edge sweep and running Mach 0.6 alone versus in a larger
grid produce identical CA/CN in all 11 controlled checks. Equivalent metre
and inch decks show a small native transonic unit sensitivity: Andromeda's
CA at Mach 0.9 is 0.396 versus 0.404, localized to body pressure/wave drag in
the partial output. Its Mach 0.3/0.6 values agree. The audit records this
remaining numerical discrepancy; it does not silently select the result
closer to the recorded flight.
Running the affected full-flight models with equivalent inch decks changes
apogee by -12.06 m for Andromeda, -0.78 m for Astra, -0.01 m for GENESIS and
+0.09 m for Valetudo. These changes are much smaller than their flight errors.

The [fin-parameter audit](comparison_data/fin_parameter_audit.json) tests
additional section assumptions without replacing the baseline comparisons.
Reproduce its 33 coefficient probes and 15 flight runs with:

```sh
python docs/examples/datcom/comparison_data/audit_fin_parameters.py
```

Trailing-edge thickness is a substantial missing hardware input. In the
controlled GENESIS run at Mach 0.6, changing a sharp USER section to a section
with a full-thickness trailing edge raises zero-incidence CD from 0.472 to
0.586. Native partial output attributes the increase to trailing-edge drag:
0.0285 per fin, or 0.114 for four fins. Doubling the HEX total thickness/chord
from 2% to 4% gives only 0.474 at that speed; 0.5/1 mm leading-edge-radius
probes give 0.472. Thickness and bevels have larger transonic effects.

| Rocket | Recorded apogee AGL (m) | Baseline HEX (m) | Sharp USER control (m) | Blunt USER hypothesis (m) |
| --- | ---: | ---: | ---: | ---: |
| GENESIS | 2916 | 3297 (+13.1%) | 3325 (+14.0%) | 3025 (+3.7%) |
| Astra | 3250 | 3696 (+13.7%) | 3773 (+16.1%) | 3378 (+3.9%) |
| Andromeda | 3443 | 4171 (+21.1%) | 4293 (+24.7%) | 3974 (+15.4%) |

The USER controls distinguish edge changes from differences between native
HEX and USER methods. The blunt hypothesis retains 2% thickness/chord,
which tapers in physical thickness: for GENESIS it is 4 mm at the root and
2.2 mm at the tip. It is not a measured constant-thickness plate. A separate
4 mm plate / 5 mm bevel HEX probe gives GENESIS 3284 m, so that change alone
does not close the gap. All flight variants stay within the sampled Mach,
alpha and beta grids at dynamic pressure above 100 Pa.

Ordinary RocketPy's fin inputs describe planform and lift; whole-vehicle drag
comes from the separately supplied drag curve. Its optional `airfoil` input
is a lift curve, not cross-section coordinates. DATCOM additionally needs
thickness, bevels and edge geometry. `ZUPPER`/`ZLOWER` describe each half of
the section, while our `thickness_to_chord` is their sum. `LMAX`/`LFLAT` are
chord fractions, `LER` is a dimensional radius, and `SSPAN` is normally from
the centreline; our `SSPAN(1)=0` convention correctly requests exposed-span
automatic body attachment. These conversions were checked. The hardware
section values remain unknown, and no fit or default change is applied.

## Published Missile DATCOM examples versus this integration

The [reference-example audit](comparison_data/reference_example_audit.json)
compares the 1997 guide's worked body/two-fin-set case (Figures 19–21,
printed pages 65–67) with geometry rebuilt through RocketPy's public API,
and checks the documented
[Ansys MissileDatCom wrapper example](https://ansyshelp.ansys.com/public/Views/Secured/ModelCenter/v252/en/ModelCenter_Rem_Exec/ModelCenter_Rem_Exec/mcre_functions/wrappers_components/FileWrapper/examples/ref_fileWrapper_example_missiledatcom.html).
Run it with:

```sh
python docs/examples/datcom/comparison_data/audit_reference_examples.py
```

[The generated manual-case deck](comparison_data/manual_generated_for005.dat)
is the unmodified output of the geometry generator. The audit adds `SOSE`
to a copy for the matching-method comparison; no production defaults or
recorded-rocket predictions change.

| Item | Published examples | Our implementation and result |
| --- | --- | --- |
| Geometry and units | Guide: inch dimensions, centreline semispans, trailing-edge sweep and REN per foot | API: metres, exposed span, explicit leading-edge coordinates and REN per metre. With the same body method, all eight published CN/CM/CA rows match at printed precision. |
| Body method | The guide selects `SOSE`; the Ansys demonstration selects `HYPER` | The generator currently leaves the native default and exposes no method selector. For the guide case at Mach 2.36, default CA at zero incidence is 0.460 versus 0.466 with SOSE; CM at 4 degrees is -1.462 versus -1.577. |
| Fin thickness arrays | Ansys specifies only `ZUPPER=0.04`; native effective total thickness is 0.066 | Explicit root and tip values of 0.04 give 0.080 total thickness. Our scalar section arguments intentionally repeat at every station, and total thickness is divided between upper/lower surfaces. A single low-level DATCOM array value does not imply that repetition. |
| Reference quantities | Ansys uses SREF=346.4 square inches and LREF=21 inches for a 20-inch body | We use frontal area and diameter consistently in both native coefficients and physical loads. Different coefficient normalization must be converted before comparison. The Ansys zero-angle CA changes from 0.611 to 0.673 when using frontal area. |
| Automatic trim | Ansys uses `$TRIM`, which changes panel deflections | Our passive flight model uses specified fixed incidence. In the local reproduction at 10 degrees, trimmed CD is 0.736 and untrimmed CD is 0.711. Trim output is a different table format and is rejected by our passive-table parser. |
| Surface/atmosphere | Ansys specifies roughness, turbulent flow and sea-level altitude | We default to smooth/turbulent flow and can use actual-atmosphere REN schedules. The example's finish and altitude are inputs for another configuration, not calibration values for our rockets. |
| Multiple cases | `SAVE` carries selected input namelists into the next case | We repeat complete fixed-beta cases without `SAVE`, avoiding inherited geometry or flow settings. Both are supported native input patterns. |

An additional native numerical sensitivity appears in derivatives. Adding a
negative-alpha row to the guide's original 0–28 degree schedule changes CNA
at zero from 0.2802 to 0.2948 per degree and changes some DAMP derivatives,
even though the positive-angle CN/CM/CA rows remain unchanged. Therefore,
damping comparisons must also match the alpha schedule. After matching
method and grids, the generated and independent reference damping tables
agree within 0.001 at printed precision. An optional native regression now
checks this complete geometry conversion, rather than only the hand-written
reference deck and parser.

The missing method selector matters for reproducing supersonic examples.
However, SOSE produced no CA/CN/CM change in controlled Mach 0.3–1.2 probes
for GENESIS, Andromeda or Astra. All 11 paired native flight peaks are below
Mach 1.04, so this finding does not resolve their apogee discrepancies.
Fin section/edge geometry and surface finish in those models remain assumed.
Reproducing these examples verifies input conversion and native execution;
it does not independently validate full-flight accuracy against hardware.
