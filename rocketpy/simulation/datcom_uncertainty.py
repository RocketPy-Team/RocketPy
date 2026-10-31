"""Isolated ascent replays for paper-based DATCOM sensitivity reporting."""

import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from rocketpy.rocket.datcom_uncertainty import DatcomUncertaintySurface


@dataclass
class DatcomUncertaintyReport:
    """Three deterministic ascent scenarios, not confidence limits.

    ``cases`` maps nominal/upper/lower to SI results (stability in calibers).
    ``flights`` holds the two additional Flights for plots. Missing apogees
    or failed replays are explicit; an incomplete envelope has no scalar ±.
    """

    cases: dict = field(default_factory=dict)
    flights: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    source: str = "https://doi.org/10.1088/1757-899X/612/3/032096"
    policy: str = "CD: at least ±20%, larger Wu CA discrepancies inside its envelope; nose-to-CP distance: ±2%. Engineering scenarios, not confidence intervals."
    design_reference: str = "https://doi.org/10.2514/6.1995-1893"

    def uncertainty(self, metric):
        """Largest absolute excursion from nominal, or None if incomplete."""
        values = [
            self.cases.get(case, {}).get(metric)
            for case in ("nominal", "upper", "lower")
        ]
        if any(value is None or not np.isfinite(value) for value in values):
            return None
        return max(abs(value - values[0]) for value in values[1:])


def _ascent_results(flight):
    """Sample each trajectory only from its initial state to its own apogee."""

    def finite(value):
        return float(value) if np.isfinite(value) else None

    reached_apogee = len(flight.apogee_state) > 1
    end = flight.apogee_time if reached_apogee else flight.t_final
    times = np.asarray(flight.time)
    times = np.unique(np.r_[times[times <= end], end])
    speed = np.asarray(flight.speed(times))
    margin = np.asarray(flight.stability_margin(times))
    # Uniform time samples make the out-of-paper coverage fraction independent
    # of ODE adaptive step density. This is coverage, not a confidence measure.
    coverage_times = np.linspace(flight.t_initial, end, 1001)
    mach = np.asarray(flight.mach_number(coverage_times))
    alpha = np.asarray(flight.angle_of_attack(coverage_times))
    outside = (mach < 0.8) | (mach > 3.02) | (np.abs(alpha) > 6)
    return {
        "apogee": float(flight.apogee) if reached_apogee else None,
        "apogee_agl": float(flight.apogee - flight.env.elevation)
        if reached_apogee
        else None,
        "max_speed": float(np.max(speed)),
        "apogee_speed": float(flight.free_stream_speed(end))
        if reached_apogee
        else None,
        "initial_stability_margin": finite(margin[0]),
        "out_of_rail_stability_margin": finite(flight.out_of_rail_stability_margin),
        "apogee_stability_margin": finite(margin[-1]) if reached_apogee else None,
        "min_stability_margin": finite(np.min(margin)),
        "max_stability_margin": finite(np.max(margin)),
        "outside_paper_fraction": float(np.mean(outside)),
    }


class DatcomUncertaintyMixin:
    """Keep scenario models and replay state local to each Flight."""

    def _initialize_datcom_uncertainty(self, setting, parameters):
        """Snapshot pristine inputs before controllers or parachutes mutate them."""
        if not (setting is True or setting is False or setting in ("upper", "lower")):
            raise ValueError(
                "datcom_uncertainty must be True, False, 'upper' or 'lower'"
            )
        self.datcom_uncertainty = None
        self._datcom_uncertainty_setting = setting
        self.datcom_uncertainty_status = None
        self._datcom_load_model = self.rocket._datcom_model
        if self._datcom_load_model is None or setting is False:
            return
        if setting in ("upper", "lower"):
            self._datcom_load_model = DatcomUncertaintySurface.from_model(
                self._datcom_load_model,
                1 if setting == "upper" else -1,
                nose_position=next(
                    position.z
                    for surface, position, _ in self.rocket.aerodynamic_surfaces
                    if surface in self.rocket.nosecones
                ),
            )
            return
        inputs = {key: value for key, value in parameters.items() if key != "self"}
        if hasattr(inputs["initial_solution"], "solution"):
            inputs["initial_solution"] = inputs["initial_solution"].solution[-1]
        self._datcom_uncertainty_inputs = deepcopy(inputs)
        self._datcom_uncertainty_rng = (np.random.get_state(), random.getstate())

    def _complete_datcom_uncertainty(self):
        """Replay upper/lower static-coefficient scenarios without recursive runs."""
        inputs = self.__dict__.pop("_datcom_uncertainty_inputs", None)
        if inputs is None:
            return
        initial_rng = self.__dict__.pop("_datcom_uncertainty_rng")
        if len(self.apogee_state) == 1:
            self.datcom_uncertainty_status = (
                "Nominal flight did not reach apogee; DATCOM uncertainty unavailable."
            )
            return
        report = DatcomUncertaintyReport(cases={"nominal": _ascent_results(self)})
        current_rng = (np.random.get_state(), random.getstate())
        try:
            for case in ("upper", "lower"):
                np.random.set_state(initial_rng[0])
                random.setstate(initial_rng[1])
                options = deepcopy(inputs)
                options.update(
                    datcom_uncertainty=case,
                    terminate_on_apogee=True,
                    verbose=False,
                    name=f"{self.name} DATCOM {case}",
                )
                try:
                    flight = type(self)(**options)
                    report.flights[case] = flight
                    report.cases[case] = _ascent_results(flight)
                    if len(flight.apogee_state) == 1:
                        report.errors[case] = "Flight did not reach apogee."
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    report.errors[case] = str(error)
                if case in report.errors:
                    warnings.warn(
                        f"DATCOM {case} sensitivity run incomplete: {report.errors[case]}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            np.random.set_state(current_rng[0])
            random.setstate(current_rng[1])
        self.datcom_uncertainty = report

    @property
    def datcom_apogee_uncertainty(self):
        """Maximum scenario apogee excursion in meters; None if unavailable."""
        return (
            None
            if self.datcom_uncertainty is None
            else self.datcom_uncertainty.uncertainty("apogee")
        )

    def _datcom_stability_margin(self, mach, time):
        """Scenario CP/CG distance in calibers; ordinary rocket behavior unchanged."""
        model = getattr(self, "_datcom_load_model", self.rocket._datcom_model)
        if model is self.rocket._datcom_model:
            return self.rocket.stability_margin(mach, time)
        return (
            (self.rocket.center_of_mass(time) - model.small_angle_cp(mach))
            * self.rocket._csys
            / (2 * self.rocket.radius)
        )
