# Asteria AST-HER DATCOM example inputs

**Flight status: Asteria / AST-HER has not flown.** All Asteria apogees and trajectories in this example are simulated predictions, not flight measurements or flight validation.

Snapshot from the local `Asteria-Flight-Dynamics` repository on 2026-10-21.
`configuration.json` merges its current input and visualization JSON files;
the provenance section records source paths and SHA-256 hashes of data files.

The configured `input/thrust-curves/opt-ThrustCurve.csv` does not exist, so
`thrust_curve.csv` copies its existing `Optimized_ThrustCurve.csv`. The saved
atmosphere is for 2026-10-20 15:30 UTC. No network request is necessary.

See `docs/examples/datcom/asteria_comparison.ipynb` for the executed comparison
of ordinary RocketPy with native DATCOM. Both retain these motor/tank mass and
inertia inputs and the saved atmosphere. Asteria's supplied ordinary drag curve
is snapshotted in `docs/examples/datcom/comparison_data/asteria_drag_curve.csv`
with its hash recorded. The configured ogive tail is native-DATCOM only;
ordinary RocketPy uses an explicitly noted conical-tail approximation.

The native executable is user supplied. DAMP, powered nozzle-annulus/coast
full-base drag and adaptive Mach/alpha/Reynolds schedules are used. No Asteria
hold-down clamp, PROTUB or recovery model is included. AST-HER has not flown;
these are simulation comparisons, not measured-flight validation.
