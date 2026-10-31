"""Missile DATCOM fin planforms, airfoil sections and panel placement.

Planform distances are meters. Airfoil coordinates and section dimensions are
fractions of chord. PHIF, GAM and DEFLCT inputs use DATCOM's degree conventions.
"""

import re
from pathlib import Path

import numpy as np

from .fins.free_form_fins import FreeFormFins
from .fins.trapezoidal_fins import TrapezoidalFins


def _station_values(value, name, count):
    """Normalize a scalar or one finite value per station/panel."""
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, count)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite scalar or have {count} values")
    return tuple(values.tolist())


def _station_pair(value, name):
    """Backward-compatible root/tip section normalization."""
    return _station_values(value, name, 2)


def _planform(stations, chords, leading_edges):
    """Validate a DATCOM spanwise planform and create its closed outline."""
    stations = np.asarray(stations, float)
    if (
        stations.ndim != 1
        or not 2 <= len(stations) <= 10
        or not np.isfinite(stations).all()
        or stations[0] != 0
        or np.any(np.diff(stations) <= 0)
    ):
        raise ValueError("span_stations needs 2..10 increasing values starting at zero")
    chords = np.asarray(_station_values(chords, "chords", len(stations)))
    edges = np.asarray(_station_values(leading_edges, "leading_edges", len(stations)))
    if np.any(chords[:-1] <= 0) or chords[-1] < 0:
        raise ValueError(
            "Chords must be positive except for an optional zero tip chord"
        )
    if edges[0] != 0:
        raise ValueError(
            "The root leading_edges value must be zero (relative to position)"
        )
    points = list(zip(edges, stations)) + list(
        zip((edges + chords)[::-1], stations[::-1])
    )
    return tuple(stations), tuple(chords), tuple(edges), points


class _DatcomFinSettings:
    """Shared section metadata for existing RocketPy planform classes."""

    def _configure_datcom(
        self,
        n,
        count,
        *,
        path,
        section_type="HEX",
        thickness_to_chord=0.02,
        upper_thickness_to_chord=None,
        lower_thickness_to_chord=None,
        max_thickness_position=0.25,
        flat_length=0.5,
        lower_max_thickness_position=None,
        lower_flat_length=None,
        leading_edge_radius=0,
        naca=None,
        airfoil_coordinates=None,
        panel_angles=None,
        dihedral=0,
        incidence=None,
        cant_angle=0,
        auto_body_radius=True,
        datcom_options=None,
    ):
        if not isinstance(n, (int, np.integer)) or not 1 <= n <= 8:
            raise ValueError("DATCOM fin sets require an integer n between 1 and 8")
        section_type = section_type.upper()
        if section_type not in ("HEX", "ARC", "NACA", "USER"):
            raise ValueError("section_type must be HEX, ARC, NACA or USER")
        total = _station_values(thickness_to_chord, "thickness_to_chord", count)
        upper = _station_values(
            np.asarray(total) / 2
            if upper_thickness_to_chord is None
            else upper_thickness_to_chord,
            "upper_thickness_to_chord",
            count,
        )
        lower = _station_values(
            upper if lower_thickness_to_chord is None else lower_thickness_to_chord,
            "lower_thickness_to_chord",
            count,
        )
        maximum = _station_values(
            max_thickness_position, "max_thickness_position", count
        )
        flat = _station_values(flat_length, "flat_length", count)
        lower_maximum = _station_values(
            maximum
            if lower_max_thickness_position is None
            else lower_max_thickness_position,
            "lower_max_thickness_position",
            count,
        )
        lower_flat = _station_values(
            flat if lower_flat_length is None else lower_flat_length,
            "lower_flat_length",
            count,
        )
        radius = _station_values(leading_edge_radius, "leading_edge_radius", count)
        if any(r < 0 for r in radius):
            raise ValueError("leading_edge_radius cannot be negative")
        if section_type in ("HEX", "ARC"):
            if any(
                top < 0 or bottom < 0 or not 0 < top + bottom < 1
                for top, bottom in zip(upper, lower)
            ):
                raise ValueError(
                    "Section upper/lower thickness must be nonnegative with total between 0 and 1"
                )
            if section_type == "HEX":
                for maxima, flats in ((maximum, flat), (lower_maximum, lower_flat)):
                    if any(
                        not (0 < m < 1 and 0 <= f < 1 and m + f < 1)
                        for m, f in zip(maxima, flats)
                    ):
                        raise ValueError(
                            "HEX requires 0 < maximum, 0 <= flat, maximum + flat < 1"
                        )
        if section_type == "NACA":
            if not isinstance(naca, str):
                raise ValueError(
                    "NACA sections require naca, e.g. '0012' or '6-64-005'"
                )
            if re.fullmatch(r"\d{4}(?:\.\d+)?", naca):
                naca = "4-" + naca
            if not re.fullmatch(r"[1456S]-[0-9.AS-]+(?: A=0\.\d+)?", naca):
                raise ValueError(
                    "Invalid NACA designation; omit the NACA and fin-set prefix"
                )
        elif naca is not None:
            raise ValueError("naca requires section_type='NACA'")
        coordinates = None
        if section_type == "USER":
            coordinates = self._validate_airfoil_coordinates(airfoil_coordinates)
        elif airfoil_coordinates is not None:
            raise ValueError("airfoil_coordinates requires section_type='USER'")
        angles = _station_values(
            np.arange(n) * 360 / n if panel_angles is None else panel_angles,
            "panel_angles",
            n,
        )
        angles = tuple(float(a % 360) for a in angles)
        if len(set(round(a, 8) for a in angles)) != n:
            raise ValueError("panel_angles must not repeat the same azimuth")
        gam = _station_values(dihedral, "dihedral", n)
        if any(abs(a) >= 90 for a in gam):
            raise ValueError("dihedral must lie strictly between -90 and 90 degrees")
        if not np.isfinite(cant_angle):
            raise ValueError("cant_angle must be finite")
        if incidence is not None and cant_angle != 0:
            raise ValueError(
                "Use either DATCOM incidence or RocketPy cant_angle, not both"
            )
        delta = (
            None if incidence is None else _station_values(incidence, "incidence", n)
        )
        if any(abs(a) >= 90 for a in (delta or (-cant_angle,))):
            raise ValueError(
                "incidence/cant must lie strictly between -90 and 90 degrees"
            )
        self.datcom_path = str(Path(path).expanduser().resolve())
        self.datcom_options = dict(datcom_options or {})
        self.section_type = section_type
        self.thickness_to_chord = tuple(
            top + bottom for top, bottom in zip(upper, lower)
        )
        self.upper_thickness_to_chord = upper
        self.lower_thickness_to_chord = lower
        self.max_thickness_position = maximum
        self.flat_length = flat
        self.lower_max_thickness_position = lower_maximum
        self.lower_flat_length = lower_flat
        self.leading_edge_radius = radius
        self.naca = naca
        self.airfoil_coordinates = coordinates
        self.panel_angles = angles
        self.dihedral = gam
        self.incidence = delta
        self.auto_body_radius = bool(auto_body_radius)

    @staticmethod
    def _validate_airfoil_coordinates(coordinates):
        if not isinstance(coordinates, dict):
            raise ValueError("USER sections require an airfoil_coordinates dictionary")
        keys = set(coordinates)
        if keys not in ({"x", "upper", "lower"}, {"x", "mean", "thickness"}):
            raise ValueError("USER needs x/upper/lower or x/mean/thickness")
        result = {k: np.asarray(v, float) for k, v in coordinates.items()}
        x = result["x"]
        if (
            x.ndim != 1
            or not 3 <= len(x) <= 50
            or x[0] != 0
            or x[-1] != 1
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError(
                "USER x needs 3..50 increasing chord fractions from 0 to 1"
            )
        if any(v.shape != x.shape or not np.isfinite(v).all() for v in result.values()):
            raise ValueError("USER coordinates must have matching finite arrays")
        thickness = result.get(
            "thickness", result.get("upper", 0) - result.get("lower", 0)
        )
        if np.any(thickness < 0) or not np.any(thickness > 0):
            raise ValueError("USER section thickness must be nonnegative and nonzero")
        return {k: tuple(v.tolist()) for k, v in result.items()}

    @property
    def datcom_incidence(self):
        """DATCOM positive incidence gives negative roll; RocketPy cant is opposite."""
        return (
            self.incidence
            if self.incidence is not None
            else (-self.cant_angle,) * self.n
        )

    def _datcom_settings_dict(self):
        names = (
            "section_type",
            "thickness_to_chord",
            "upper_thickness_to_chord",
            "lower_thickness_to_chord",
            "max_thickness_position",
            "flat_length",
            "lower_max_thickness_position",
            "lower_flat_length",
            "leading_edge_radius",
            "naca",
            "airfoil_coordinates",
            "panel_angles",
            "dihedral",
            "incidence",
            "cant_angle",
            "auto_body_radius",
            "datcom_options",
        )
        return {
            "path": self.datcom_path,
            **{name: getattr(self, name) for name in names},
        }

    def draw_on_rocket(self, ax, position, drawn_surfaces, vis_args, plane, csys):
        """Project actual PHIF/GAM/DEFLCT placement into a Rocket drawing."""
        from rocketpy.plots.datcom_plots import draw_datcom_fins

        return draw_datcom_fins(
            self, ax, position, drawn_surfaces, vis_args, plane, csys
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class DatcomFins(_DatcomFinSettings, TrapezoidalFins):
    """A trapezoidal or triangular planform with a HEX, ARC, NACA or USER section.

    Dimensions and position follow TrapezoidalFins. Section inputs accept scalar
    or root/tip values; thicknesses are chord fractions and radius is meters.
    NACA uses a designation such as ``0012`` or ``6-64-005``. USER accepts
    ``airfoil_coordinates={"x": ..., "upper": ..., "lower": ...}`` or mean and
    thickness arrays instead of upper/lower. PHIF panel_angles and GAM dihedral
    use degrees. Fixed incidence follows DATCOM's negative-roll convention;
    cant_angle follows RocketPy's positive-roll convention. All execution and
    geometry inputs survive serialization. See the DATCOM user guide page.
    """

    def __init__(
        self,
        n,
        root_chord,
        tip_chord,
        span,
        rocket_radius,
        *,
        path,
        sweep_length=None,
        sweep_angle=None,
        name="DATCOM Fins",
        **section_options,
    ):
        for value, label in (
            (root_chord, "root_chord"),
            (span, "span"),
            (rocket_radius, "rocket_radius"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"DATCOM {label} must be positive and finite")
        if not np.isfinite(tip_chord) or tip_chord < 0:
            raise ValueError("DATCOM tip_chord must be finite and nonnegative")
        self._configure_datcom(n, 2, path=path, **section_options)
        super().__init__(
            n=n,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=span,
            rocket_radius=rocket_radius,
            sweep_length=sweep_length,
            sweep_angle=sweep_angle,
            cant_angle=section_options.get("cant_angle", 0),
            name=name,
        )

    @property
    def datcom_span_stations(self):
        return (0.0, self.span)

    @property
    def datcom_chords(self):
        return (self.root_chord, self.tip_chord)

    @property
    def datcom_leading_edges(self):
        return (0.0, self.sweep_length)

    def evaluate_shape(self):
        _, _, _, points = _planform(
            self.datcom_span_stations, self.datcom_chords, self.datcom_leading_edges
        )
        self.shape_vec = [np.array([p[i] for p in points]) for i in (0, 1)]
        return self.shape_vec

    def to_dict(self, **kwargs):
        return dict(
            n=self.n,
            root_chord=self.root_chord,
            tip_chord=self.tip_chord,
            span=self.span,
            rocket_radius=self.rocket_radius,
            sweep_length=self.sweep_length,
            name=self.name,
            **self._datcom_settings_dict(),
        )


class DatcomFreeFormFins(_DatcomFinSettings, FreeFormFins):
    """Piecewise-linear planform with 2..10 exposed span stations.

    Span stations start at zero at the body surface. Chords and leading_edges
    are meters; leading_edges start at zero relative to the root position.
    All cross sections belong to one type, with proportions varying by station.
    Section options and panel placement are identical to DatcomFins.
    """

    def __init__(
        self,
        n,
        span_stations,
        chords,
        leading_edges,
        rocket_radius,
        *,
        path,
        name="DATCOM Free Form Fins",
        **section_options,
    ):
        stations, chords, edges, points = _planform(
            span_stations, chords, leading_edges
        )
        self._datcom_stations = stations
        self._datcom_chords = chords
        self._datcom_edges = edges
        if not np.isfinite(rocket_radius) or rocket_radius <= 0:
            raise ValueError("rocket_radius must be positive and finite")
        self._configure_datcom(n, len(stations), path=path, **section_options)
        super().__init__(
            n,
            points,
            rocket_radius,
            cant_angle=section_options.get("cant_angle", 0),
            name=name,
        )

    @property
    def datcom_span_stations(self):
        return self._datcom_stations

    @property
    def datcom_chords(self):
        return self._datcom_chords

    @property
    def datcom_leading_edges(self):
        return self._datcom_edges

    def to_dict(self, **kwargs):
        return dict(
            n=self.n,
            span_stations=self.datcom_span_stations,
            chords=self.datcom_chords,
            leading_edges=self.datcom_leading_edges,
            rocket_radius=self.rocket_radius,
            name=self.name,
            **self._datcom_settings_dict(),
        )
