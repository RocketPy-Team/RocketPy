---
# Scout Report — 2023-10-24

## ✅ SELECTED ISSUE: 1
*(Fill in 1, 2, or 3 after reviewing the briefs below)*

---

## Issue 1 — BUG: Wind Heading Profile Plots are not that good (https://github.com/RocketPy-Team/RocketPy/issues/253)

### What's broken
When wind direction wraps around from 360° to 0°, the matplotlib plot draws a connecting line across the entire width of the graph. This creates a visually confusing plot that misrepresents the actual wind heading interpolation between those altitudes. The line should break or wrap around cleanly instead of crossing the plot.

### Files to touch
- `rocketpy/plots/environment_plots.py`
- `tests/integration/environment/test_environment.py`

### Implementation approach
1. In `rocketpy/plots/environment_plots.py`, locate the `__wind` method where wind speed and direction are plotted.
2. Extract the wind direction values into a numpy array: `directions = np.array([self.environment.wind_direction(i) for i in self.grid])`.
3. Identify points where the absolute difference between consecutive values exceeds a threshold (e.g., 180°).
4. Insert `np.nan` into both the `directions` array and the corresponding `grid` (altitude) array at these jump points to break the matplotlib line, or apply a masked array to hide the connecting segments.
5. Update the `axup.plot()` call to use these modified arrays instead of the list comprehension.

### Acceptance criteria
- The wind direction plot no longer shows horizontal lines connecting 360° back to 0° across the graph.
- The plotted data points remain accurate and are not shifted.
- All existing environment tests still pass.

### Guardrails
- Do not modify the underlying `Environment` or `Function` classes or how wind heading is calculated mathematically.
- Do not add new external dependencies to handle the plotting logic.

### Difficulty
2 — The issue is contained entirely within the plotting utility and relies on a standard matplotlib workaround for cyclic data.

---

## Issue 2 — ENH: Save Monte Carlo outputs to .csv and .json formats (https://github.com/RocketPy-Team/RocketPy/issues/242)

### What's broken
Currently, the `MonteCarlo` class saves simulation outputs (results, inputs, and errors) primarily to `.txt` files. This makes post-processing difficult for users who want to analyze their data in external tools like Excel, MATLAB, or generic Python scripts without loading RocketPy. The user should be able to choose the output format easily to export their dispersion results.

### Files to touch
- `rocketpy/simulation/monte_carlo.py`
- `tests/integration/simulation/test_monte_carlo.py`

### Implementation approach
1. In `rocketpy/simulation/monte_carlo.py`, add new methods to the `MonteCarlo` class: `export_results_to_csv(self, filename)` and `export_results_to_json(self, filename)`.
2. Inside `export_results_to_csv`, use Python's built-in `csv` module to open the file and write headers (keys of `self.results`) and rows corresponding to each simulation iteration.
3. Inside `export_results_to_json`, use Python's built-in `json` module to dump the `self.results` dictionary directly to the file.
4. Optionally, create a unified `export_results(self, filename, format="csv")` method that calls the appropriate specific method based on the `format` argument.
5. Ensure that custom `data_collector` variables are included in the export (they should already be in `self.results`).

### Acceptance criteria
- The `MonteCarlo` class has methods to export data to both `.csv` and `.json` formats.
- The exported files contain the correct data stored in the `self.results` dictionary, structured appropriately.
- Unit tests verify that the files are created and contain the expected data.
- Built-in `csv` and `json` libraries are used without adding heavy dependencies like `pandas`.

### Guardrails
- Do not modify how the Monte Carlo simulation itself runs or how it generates `self.results`.
- Avoid using heavy dependencies like `pandas`; stick to Python's built-in libraries.

### Difficulty
2 — Straightforward implementation using standard Python libraries, requiring basic dictionary and list manipulation to format the outputs correctly.

---

## Issue 3 — ENH: Implement Parachute Opening Shock Force Estimation (https://github.com/RocketPy-Team/RocketPy/issues/161)

### What's broken
RocketPy currently cannot estimate the transient opening shock force that occurs when a parachute inflates. This leaves users unable to size their recovery hardware properly, as the steady-state drag force is significantly lower than the peak shock force.

### Files to touch
- `rocketpy/rocket/parachute.py`
- `tests/fixtures/parachutes/parachute_fixtures.py`

### Implementation approach
1. Update the `Parachute.__init__` signature in `rocketpy/rocket/parachute.py` to accept an `opening_shock_coefficient` parameter, defaulting to a standard value like `1.5` or `1.6`.
2. Store this parameter as an instance variable (`self.opening_shock_coefficient = opening_shock_coefficient`).
3. Add a new method `calculate_opening_shock(self, density, velocity)` to the `Parachute` class.
4. Implement the Knacke formula: `F_o = self.opening_shock_coefficient * 0.5 * density * velocity**2 * self.cd_s`. (Note: this assumes the infinite mass opening factor $X_1$ is combined with $C_x$ as mentioned in the issue).
5. Return the calculated force $F_o$.

### Acceptance criteria
- The `Parachute` class accepts and stores `opening_shock_coefficient`.
- The new `calculate_opening_shock` method correctly calculates the peak shock force based on the provided inputs and the parachute's $C_d S$.
- A unit test verifies the calculation against a known manual calculation or textbook example.

### Guardrails
- Do not integrate this transient force into the main `Flight` equations of motion yet, as the issue requests it primarily for post-processing estimation.
- Do not modify the existing steady-state drag calculations.

### Difficulty
1 — Requires adding a single instance variable and a straightforward mathematical method to an existing class, plus a basic unit test.
---
