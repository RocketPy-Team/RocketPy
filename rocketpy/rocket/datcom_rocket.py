"""DATCOM-specific Rocket assembly and preparation, isolated from core models."""

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.aero_surface.datcom_fins import DatcomFins, DatcomFreeFormFins


class DatcomRocketMixin:
    """Optional DATCOM behavior shared by Rocket and its subclasses."""

    def set_datcom_atmosphere(
        self,
        flight,
        *,
        mach=None,
        alpha=None,
        samples=1000,
        minimum_dynamic_pressure=100,
    ):
        """Adapt native Mach/alpha grids and Reynolds schedules to a flight.

        The default Mach grid uses all 20 native slots, always includes 0.1
        and 1.0, concentrates points near Mach 1, and extends above the reference
        peak by 15% plus 0.02. Both powered and coast samples determine coverage.
        Mach zero uses the positive 0.01 endpoint. Explicit ``mach`` retains a
        custom 2..20-point grid instead. The initial no-flight grid is provisional.

        Adaptive alpha has 19 signed points concentrated around zero, retaining
        at least ±20 degrees and extending beyond observed incidence by 15% plus
        2 degrees. Only samples at ``minimum_dynamic_pressure`` (Pa, default
        100) or greater determine its extent; low-load pad/apogee directions do
        not force extreme grids. With no loaded samples it retains ±20 degrees.
        Explicit ``alpha`` fixes a custom native grid (degrees). Tail-first
        adaptive envelopes are rejected because the current load model assumes
        forward flight. The returned coverage report identifies excluded samples.

        ``flight`` may be an ordinary or previous DATCOM Flight with the same
        motor burn interval. Sampling includes ``samples`` uniform points,
        solver states and burn boundaries, from rail to apogee. Descent is
        excluded; the current load model assumes forward-flight aerodynamics.

        At sampled altitude (meters ASL), the reference Environment supplies
        density, sound speed and dynamic viscosity. Native Reynolds per meter
        is rho*a*Mach/mu; REN replaces ALT, bypassing the standard atmosphere.
        Return Mach, alpha, coverage, Reynolds, altitude, pressure (Pa) and
        temperature (K) schedules for provenance. No Flight or native run is
        started here; the next Flight regenerates the tables. Call again with
        that result to refine both the envelope and atmospheric schedules.

        Each phase still uses one atmospheric state per Mach. Repeated Mach
        retains the earliest sample; unreached Mach clamps atmospheric state
        while recomputing speed. This does not resolve multiple ascent/descent
        crossings at different altitudes or guarantee iterative convergence.
        Ordinary equations are unchanged; failed validation preserves settings.
        """
        from rocketpy.rocket.datcom import _flight_conditions
        from rocketpy.rocket.datcom_conditions import (
            adaptive_alpha_grid,
            adaptive_mach_grid,
        )

        fins = [
            s
            for s, _, _ in self.aerodynamic_surfaces
            if isinstance(s, (DatcomFins, DatcomFreeFormFins))
        ]
        if not fins:
            raise ValueError("Add DATCOM fins before setting their atmosphere")
        if not isinstance(samples, (int, np.integer)) or samples < 3:
            raise ValueError("samples must be an integer of at least 3")
        if not np.allclose(self.motor.burn_time, flight.rocket.motor.burn_time):
            raise ValueError("Reference flight must use the same motor burn interval")
        if not np.isfinite(minimum_dynamic_pressure) or minimum_dynamic_pressure < 0:
            raise ValueError("minimum_dynamic_pressure must be finite and nonnegative")
        end = float(flight.apogee_time)
        if not np.isfinite(end) or end <= 0:
            raise ValueError("Reference flight must contain a positive flight end time")
        time = np.linspace(0, end, samples)
        # Include actual solver states and phase boundaries, not just uniform
        # samples: short burns and rail/burnout transitions must not disappear.
        extra = list(self.motor.burn_time)
        if hasattr(flight, "solution"):
            extra.extend(np.asarray(flight.solution)[:, 0])
        time = np.unique(np.concatenate((time, np.asarray(extra))))
        time = time[(time >= 0) & (time <= end)]
        trajectory_mach = np.asarray(flight.mach_number(time), float)
        altitude = np.asarray(flight.z(time), float)
        env = flight.env
        dynamic_pressure = (
            0.5
            * np.asarray(env.density(altitude))
            * (trajectory_mach * env.speed_of_sound(altitude)) ** 2
        )
        loaded = np.isfinite(dynamic_pressure) & (
            dynamic_pressure >= minimum_dynamic_pressure
        )
        options = dict(fins[0].datcom_options)
        options["mach"] = (
            adaptive_mach_grid(float(np.max(trajectory_mach))) if mach is None else mach
        )
        if alpha is None:
            observed_alpha = np.asarray(flight.partial_angle_of_attack(time), float)
            options["alpha"] = adaptive_alpha_grid(
                observed_alpha[loaded] if np.any(loaded) else [0.0]
            )
        else:
            observed_alpha = None
            options["alpha"] = alpha
        # The old ALT schedule belongs to the old Mach grid. REN replaces it.
        options.pop("altitude", None)
        grid, alpha_grid, _, _ = _flight_conditions(options)
        start, stop = self.motor.burn_time
        powered = (time >= start) & (time < stop)
        report = {
            "mach": grid.tolist(),
            "alpha": alpha_grid.tolist(),
            "coverage": {
                "scope": "rail through apogee",
                "minimum_dynamic_pressure_pa": float(minimum_dynamic_pressure),
                "excluded_low_q_samples": int(np.count_nonzero(~loaded)),
                "phases": {},
            },
            "reynolds": {},
            "altitude_m_asl": {},
            "pressure_pa": {},
            "temperature_k": {},
        }
        for state, label in [(True, "power_on"), (False, "power_off")]:
            indices = np.flatnonzero(
                (powered == state)
                & np.isfinite(trajectory_mach)
                & np.isfinite(altitude)
            )
            if not len(indices):
                raise ValueError(f"Reference ascent has no {label} samples")
            phase_loaded = indices[loaded[indices]]
            phase = {
                "mach_min": float(np.min(trajectory_mach[indices])),
                "mach_max": float(np.max(trajectory_mach[indices])),
                "loaded_alpha_max_abs_deg": (
                    float(np.max(np.abs(observed_alpha[phase_loaded])))
                    if observed_alpha is not None and len(phase_loaded)
                    else None
                ),
            }
            report["coverage"]["phases"][label] = phase
            order = indices[np.argsort(trajectory_mach[indices], kind="stable")]
            unique, first = np.unique(trajectory_mach[order], return_index=True)
            z = altitude[order[first]]
            env = flight.env
            # Interpolate Re/M, then apply target Mach: clamped endpoints retain
            # atmospheric state rather than incorrectly retaining endpoint speed.
            factor = (
                np.asarray(env.density(z))
                * np.asarray(env.speed_of_sound(z))
                / np.asarray(env.dynamic_viscosity(z))
            )
            ren = grid * np.interp(grid, unique, factor)
            if not np.all(np.isfinite(ren) & (ren > 0)):
                raise ValueError(
                    "Environment must supply finite positive Reynolds numbers"
                )
            report["reynolds"][label] = ren.tolist()
            report["altitude_m_asl"][label] = np.interp(grid, unique, z).tolist()
            for key, function in [
                ("pressure_pa", env.pressure),
                ("temperature_k", env.temperature),
            ]:
                report[key][label] = np.interp(grid, unique, function(z)).tolist()
        options.update(mach=grid.tolist(), reynolds=report["reynolds"])
        options.pop("altitude", None)
        self._invalidate_datcom()
        for surface in fins:
            surface.datcom_options = dict(options)
        self.evaluate_center_of_pressure()
        self.evaluate_stability_margin()
        self._invalidate_static_margin()
        return report

    @property
    def datcom_calibration(self):
        """Return a copy of the explicit whole-rocket calibration inputs."""
        return dict(
            getattr(
                self,
                "_datcom_calibration",
                {"roughness_height": 0.0, "leading_edge_radius": None},
            )
        )

    def add_datcom_calibration(self, *, roughness_height=0.0, leading_edge_radius=None):
        """Set native DATCOM surface/edge calibration, replacing previous values.

        Parameters
        ----------
        roughness_height : float, optional
            Nonnegative equivalent sand roughness height in meters, not Ra.
            Converted to DATCOM RHR using k = 3 RHR microinches, avoiding the
            version-dependent metric ROUGH convention. Default is smooth.
        leading_edge_radius : float or None, optional
            Nonnegative radius in meters, overriding LER at every station of
            every supported fin set. None retains each fin's own radius.
            NACA sections define their own edge geometry and reject overrides.

        Returns
        -------
        dict
            Copy of the stored calibration. Call without arguments to reset.

        Notes
        -----
        Add DATCOM fins first. This method sets inputs; it does not fit them or
        alter mass, planform or ordinary RocketPy aerodynamics. A fitted
        roughness is an effective calibration, not a measured material finish.
        Changes invalidate the prepared tables; Flight regenerates them.
        """
        if not self._has_datcom_fins():
            raise ValueError(
                "DATCOM calibration requires a DATCOM fin set; add it first"
            )
        values = {}
        for name, value in (
            ("roughness_height", roughness_height),
            ("leading_edge_radius", leading_edge_radius),
        ):
            if name == "leading_edge_radius" and value is None:
                values[name] = None
                continue
            if np.ndim(value) != 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} must be a finite nonnegative scalar"
                ) from error
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(number)
                or number < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative scalar")
            values[name] = number
        if leading_edge_radius is not None and any(
            getattr(s, "section_type", None) == "NACA"
            for s, _, _ in self.aerodynamic_surfaces
        ):
            raise ValueError("NACA sections define their own leading-edge radius")
        if values != self.datcom_calibration:
            self._invalidate_datcom()
            self._datcom_calibration = values
            self.evaluate_center_of_pressure()
            self.evaluate_stability_margin()
            self._invalidate_static_margin()
        return self.datcom_calibration

    def fit_datcom_calibration(
        self,
        data,
        *,
        reference_area=None,
        parameters=("roughness_height",),
        bounds=None,
        reference_length=None,
        moment_reference=None,
        lateral_reference_length=None,
        model_scale=1.0,
        power_on=False,
        weights=None,
        drag_axis="wind",
        max_evaluations=100,
    ):
        """Fit native calibration inputs to static wind-tunnel coefficients.

        ``data`` is a CSV path or a column mapping. Required columns are
        ``mach`` and at least one of CD, CL, CA, CN, CY, CM, CLN, CLL. Optional
        ``alpha``/``beta`` are degrees (default zero); ``power_on`` is 0 or 1;
        ``reynolds_per_meter`` specifies tunnel Reynolds number per meter.

        ``reference_area`` is the measurement area in m², defaulting to the
        rocket frontal area. Moment measurements
        also require ``reference_length`` (m) and ``moment_reference`` (m in
        this rocket's axial position convention). An optional lateral length
        scales yaw/roll coefficients; it defaults to reference_length.
        ``model_scale`` is tunnel-model size divided by full-rocket size.
        Reference areas/lengths and tunnel Reynolds numbers are converted to
        full-scale equivalents; moment_reference stays in rocket coordinates.

        Fit roughness_height and/or leading_edge_radius, with bounds in meters.
        Roughness defaults to 0..300 micrometers. Fitting edge radius requires
        explicit physically justified bounds. ``weights`` maps coefficient
        names to positive least-squares weights. CD is wind-axis drag by
        default; drag_axis='stability' selects the pitch-plane convention.

        Returns a JSON-compatible report with residuals, initial/final RMSE,
        fitted inputs, bounds and native evaluation count. Successful fits
        are applied through add_datcom_calibration. Failed fits preserve the
        existing settings and prepared model. No measured coefficient or
        derivative replaces the native DATCOM tables.
        """
        from rocketpy.rocket.datcom_calibration import fit_datcom_calibration

        return fit_datcom_calibration(
            self,
            data,
            reference_area=reference_area,
            parameters=parameters,
            bounds=bounds,
            reference_length=reference_length,
            moment_reference=moment_reference,
            lateral_reference_length=lateral_reference_length,
            model_scale=model_scale,
            power_on=power_on,
            weights=weights,
            drag_axis=drag_axis,
            max_evaluations=max_evaluations,
        )

    def _create_datcom_or_conical_tail(self, top, bottom, length, radius, name, kind):
        from rocketpy.rocket.aero_surface.tail import Tail

        key = kind.replace(" ", "").replace("-", "").lower()
        if key in ("cone", "conical"):
            return Tail(top, bottom, length, radius, name)
        if key not in ("ogive", "tangent", "tangentogive"):
            raise ValueError("Tail kind must be 'conical' or 'ogive'")
        if not self._has_datcom_fins():
            raise ValueError("Ogive tails require a DATCOM fin set; add it first")
        from rocketpy.rocket.aero_surface.datcom_tail import DatcomTail

        return DatcomTail(top, bottom, length, radius, name)

    def _validate_datcom_tail(self, surface):
        from rocketpy.rocket.aero_surface.datcom_tail import DatcomTail

        if isinstance(surface, DatcomTail) and not self._has_datcom_fins():
            raise ValueError("Ogive tails require a DATCOM fin set; add it first")

    def _set_undefined_datcom_static_margin(self):
        """Keep an undefined CP out of the ordinary finite-data spline fit."""
        if self._has_datcom_fins() and not np.isfinite(self.cp_position(0)):
            self._static_margin.set_source(lambda time: float("nan"))
            self._static_margin_dirty = False
            return True
        return False

    def _evaluate_datcom_center_of_pressure(self):
        if self._has_datcom_fins():
            model = self._datcom_model
            if model is None:
                self.cp_position.set_source(lambda mach: float("nan"))
                self.total_lift_coeff_der.set_source(lambda mach: float("nan"))
            else:
                self.cp_position.set_source(lambda mach: model.small_angle_cp(mach))
                table = model.tables[False]
                zero = int(np.flatnonzero(table[5] == 0)[0])
                self.total_lift_coeff_der.set_source(
                    lambda mach: float(
                        np.interp(mach, table[4], table.derivatives["CNA"][:, zero])
                    )
                    * 180
                    / np.pi
                )
            return True

        return False

    def _has_datcom_fins(self):
        return any(
            isinstance(s, (DatcomFins, DatcomFreeFormFins))
            for s, _, _ in self.aerodynamic_surfaces
        )

    @property
    def _flight_aerodynamic_surfaces(self):
        """Active flight models; original geometry remains available for drawing."""
        if self._datcom_model is not None:
            return [
                (
                    self._datcom_model,
                    Vector([0, 0, self.center_of_dry_mass_position]),
                    1,
                )
            ]
        return self.aerodynamic_surfaces

    def prepare_datcom(self):
        """Generate powered/coast DATCOM tables from the current assembled rocket.

        Flight calls this automatically. Call it explicitly before inspecting
        DATCOM drag or stability plots. Identical geometry, CG references,
        conditions and executable reuse the in-memory model. Changes rebuild
        both cases. No executable or generated output is kept in the repository.

        Returns
        -------
        DatcomSurface or None
            Whole-vehicle aerodynamic model, or None without DATCOM fins.
        """
        if not self._has_datcom_fins():
            for surface, _, _ in self.aerodynamic_surfaces:
                self._validate_datcom_tail(surface)
            self._invalidate_datcom()
            return None

        from pathlib import Path  # pylint: disable=import-outside-toplevel

        from rocketpy.rocket.datcom import (  # pylint: disable=import-outside-toplevel
            DatcomSurface,
            datcom_decks,
            run_datcom,
        )

        decks, references, timeout, path = datcom_decks(self)
        stat = Path(path).stat()
        signature = (
            tuple(decks.items()),
            tuple(references.items()),
            path,
            stat.st_size,
            stat.st_mtime_ns,
            self.motor.burn_start_time,
            self.motor.burn_out_time,
        )
        if signature == self._datcom_signature:
            return self._datcom_model
        # Publish only after both runs succeed, so failed cases cannot leave a
        # half-populated aerodynamic database attached to the rocket.
        tables = {
            powered: run_datcom(path, deck, timeout) for powered, deck in decks.items()
        }
        model = DatcomSurface(self, tables, references)
        self._invalidate_datcom()
        self._datcom_model = model
        self._datcom_signature = signature
        names = [
            "power_on_drag_7d",
            "power_off_drag_7d",
            "power_on_drag_by_mach",
            "power_off_drag_by_mach",
            "power_on_drag",
            "power_off_drag",
        ]
        self._datcom_original_drag = {name: getattr(self, name) for name in names}
        for powered, prefix in ((True, "power_on"), (False, "power_off")):
            # Function counts callable arguments, so bind the powered selector
            # through a closure factory rather than an eighth default argument.
            def make_coefficient(state):
                return (
                    lambda alpha, beta, mach, reynolds, q, r, p: model.drag_coefficient(
                        state, alpha, beta, mach, reynolds, q, r, p
                    )
                )

            drag = Function(
                make_coefficient(powered),
                inputs=["alpha", "beta", "mach", "reynolds", "q", "r", "p"],
            )

            def make_curve(state):
                return lambda mach: model.drag_coefficient(
                    state, 0, 0, mach, 0, 0, 0, 0
                )

            curve = Function(
                make_curve(powered), inputs="Mach", outputs="Drag Coefficient"
            )
            setattr(self, f"{prefix}_drag_7d", drag)
            setattr(self, f"{prefix}_drag_by_mach", curve)
            setattr(self, f"{prefix}_drag", curve)
        self.surfaces_cp_to_cdm[model] = Vector([0, 0, 0])
        self.evaluate_center_of_pressure()
        self.evaluate_stability_margin()
        self._invalidate_static_margin()
        return model

    def _invalidate_datcom(self):
        """Discard whole-vehicle tables when assembly changes."""
        if self._datcom_model is not None:
            self.surfaces_cp_to_cdm.pop(self._datcom_model, None)
        if self._datcom_original_drag is not None:
            for name, value in self._datcom_original_drag.items():
                setattr(self, name, value)
        self._datcom_model = None
        self._datcom_signature = None
        self._datcom_original_drag = None

    def add_datcom_trapezoidal_fins(
        self,
        n,
        root_chord,
        tip_chord,
        span,
        position,
        *,
        path,
        thickness_to_chord=0.02,
        leading_edge_radius=0,
        sweep_length=None,
        sweep_angle=None,
        name="DATCOM Trapezoidal Fins",
        datcom_options=None,
        **section_options,
    ):
        """Add a trapezoidal planform with a beveled HEX section by default.

        Uses the same geometry/position convention as add_trapezoidal_fins.
        The section has a flat middle 90% of each chord and 5% edge bevels.
        ``thickness_to_chord`` is full thickness/chord, scalar or root/tip pair.
        ``path`` is the native DATCOM executable; ``datcom_options`` accepts
        mach, alpha (degrees), altitude (meters), timeout (seconds), and optional
        cylindrical body_length (meters). All existing nose, tail and supported
        fins are included in the DATCOM model and their Barrowman loads disabled.
        DAMP is mandatory. No protuberance or plume calculations are requested.
        See :class:`rocketpy.DatcomFins` for section inputs and limitations.
        """
        return self._add_datcom_fins(
            n,
            root_chord,
            tip_chord,
            span,
            position,
            path=path,
            thickness_to_chord=thickness_to_chord,
            max_thickness_position=section_options.pop("max_thickness_position", 0.05),
            flat_length=section_options.pop("flat_length", 0.9),
            leading_edge_radius=leading_edge_radius,
            sweep_length=sweep_length,
            sweep_angle=sweep_angle,
            name=name,
            datcom_options=datcom_options,
            **section_options,
        )

    def add_datcom_arc_fins(
        self,
        n,
        root_chord,
        tip_chord,
        span,
        position,
        *,
        path,
        thickness_to_chord=0.02,
        leading_edge_radius=0,
        sweep_length=None,
        sweep_angle=None,
        name="DATCOM Arc Fins",
        datcom_options=None,
        **section_options,
    ):
        """Add trapezoidal-planform fins with a circular ARC section.

        Geometry and DATCOM settings follow add_datcom_trapezoidal_fins.
        ARC describes the airfoil cross section, not a curved planform.
        """
        return self._add_datcom_fins(
            n,
            root_chord,
            tip_chord,
            span,
            position,
            path=path,
            section_type="ARC",
            thickness_to_chord=thickness_to_chord,
            leading_edge_radius=leading_edge_radius,
            sweep_length=sweep_length,
            sweep_angle=sweep_angle,
            name=name,
            datcom_options=datcom_options,
            **section_options,
        )

    def add_datcom_hex_fins(
        self,
        n,
        root_chord,
        tip_chord,
        span,
        position,
        *,
        path,
        thickness_to_chord=0.02,
        max_thickness_position=0.25,
        flat_length=0.5,
        leading_edge_radius=0,
        sweep_length=None,
        sweep_angle=None,
        name="DATCOM Hex Fins",
        datcom_options=None,
        **section_options,
    ):
        """Add trapezoidal-planform fins with a HEX airfoil section.

        Geometry and DATCOM settings follow add_datcom_trapezoidal_fins.
        ``max_thickness_position`` locates the start of maximum thickness;
        ``flat_length`` is its extent, both as scalar or root/tip chord fractions.
        A zero flat_length gives a diamond section. HEX describes the airfoil
        cross section, not a six-sided planform.
        """
        return self._add_datcom_fins(
            n,
            root_chord,
            tip_chord,
            span,
            position,
            path=path,
            thickness_to_chord=thickness_to_chord,
            max_thickness_position=max_thickness_position,
            flat_length=flat_length,
            leading_edge_radius=leading_edge_radius,
            sweep_length=sweep_length,
            sweep_angle=sweep_angle,
            name=name,
            datcom_options=datcom_options,
            **section_options,
        )

    def add_datcom_fins(
        self,
        n,
        position,
        *,
        span_stations,
        chords,
        leading_edges,
        path,
        name="DATCOM Free Form Fins",
        **section_options,
    ):
        """Add a 2..10-station DATCOM planform with any supported airfoil section.

        span_stations are exposed distances from the body surface in meters.
        leading_edges are aft distances from position; their first value is zero.
        Chords are meters and the final chord can be zero for a triangular tip.
        Section and panel-placement options follow DatcomFins.
        """
        fins = DatcomFreeFormFins(
            n,
            span_stations,
            chords,
            leading_edges,
            self.radius,
            path=path,
            name=name,
            **section_options,
        )
        self.add_surfaces(fins, position)
        return fins

    def _add_datcom_fins(self, n, root_chord, tip_chord, span, position, **kwargs):
        fins = DatcomFins(n, root_chord, tip_chord, span, self.radius, **kwargs)
        self.add_surfaces(fins, position)
        return fins
