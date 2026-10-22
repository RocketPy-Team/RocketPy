# DATCOM comparisons

**Work in progress:** DATCOM currently produces poor flight predictions for several comparison rockets. The discrepancies are unresolved and are being investigated; these results should not be treated as validated predictions.

Only two example notebooks are included:

- [Recorded rockets](rocket_comparison.ipynb): DATCOM, ordinary RocketPy and
  recorded altitude, upward vertical velocity, CD and stability margin where
  available, with signed percentage errors. Apogee-fitted constant CD is
  explicitly inferred calibration, not a measurement or native correction.
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
