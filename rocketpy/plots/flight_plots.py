# pylint: disable=too-many-lines

import logging
import os
import time
from collections.abc import Mapping, Sequence
from functools import cached_property
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from ..tools import import_optional_dependency
from .plot_helpers import show_or_save_plot

logger = logging.getLogger(__name__)


class _FlightPlots:
    """Class that holds plot methods for Flight class.

    Attributes
    ----------
    _FlightPlots.flight : Flight
        Flight object that will be used for the plots.

    _FlightPlots.first_event_time : float
        Time of first event.

    _FlightPlots.first_event_time_index : int
        Time index of first event.
    """

    def __init__(self, flight):
        """Initializes _FlightPlots class.

        Parameters
        ----------
        flight : Flight
            Instance of the Flight class

        Returns
        -------
        None
        """
        self.flight = flight

    @cached_property
    def first_event_time(self):
        """Time of the first flight event."""
        if len(self.flight.parachute_events) > 0:
            return (
                self.flight.parachute_events[0][0]
                + self.flight.parachute_events[0][1].lag
            )
        else:
            return self.flight.t_final

    @cached_property
    def first_event_time_index(self):
        """Time index of the first flight event."""
        if len(self.flight.parachute_events) > 0:
            return np.nonzero(self.flight.x[:, 0] == self.first_event_time)[0][0]
        else:
            return -1

    def trajectory_3d(self, *, filename=None):  # pylint: disable=too-many-statements
        """Plot a 3D graph of the trajectory

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        max_z = max(self.flight.altitude[:, 1])
        min_z = min(self.flight.altitude[:, 1])
        max_x = max(self.flight.x[:, 1])
        min_x = min(self.flight.x[:, 1])
        max_y = max(self.flight.y[:, 1])
        min_y = min(self.flight.y[:, 1])
        min_xy = min(min_x, min_y)
        max_xy = max(max_x, max_y)

        # avoids errors when x_lim and y_lim are the same
        if abs(min_z - max_z) < 1e-5:
            max_z += 1
        if abs(min_xy - max_xy) < 1e-5:
            max_xy += 1

        _ = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111, projection="3d")
        ax1.plot(
            self.flight.x[:, 1], self.flight.y[:, 1], zs=min_z, zdir="z", linestyle="--"
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.altitude[:, 1],
            zs=min_y,
            zdir="y",
            linestyle="--",
        )
        ax1.plot(
            self.flight.y[:, 1],
            self.flight.altitude[:, 1],
            zs=min_x,
            zdir="x",
            linestyle="--",
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.y[:, 1],
            self.flight.altitude[:, 1],
            linewidth="2",
        )
        ax1.scatter(
            self.flight.x(0),
            self.flight.y(0),
            self.flight.z(0) - self.flight.env.elevation,
            color="black",
        )
        ax1.scatter(
            self.flight.x(self.flight.t_final),
            self.flight.y(self.flight.t_final),
            self.flight.z(self.flight.t_final) - self.flight.env.elevation,
            color="red",
            marker="X",
        )
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Z - Altitude Above Ground Level (m)")
        ax1.set_title("Flight Trajectory")
        ax1.set_xlim(min_xy, max_xy)
        ax1.set_ylim(min_xy, max_xy)
        ax1.set_zlim(min_z, max_z)
        ax1.view_init(15, 45)
        ax1.set_box_aspect(None, zoom=0.95)  # 95% for label adjustment
        show_or_save_plot(filename)

    def _resolve_animation_model_path(self, file_name):
        """Resolve model path, defaulting to the built-in STL when omitted."""
        if file_name is not None:
            return file_name

        return str(
            resources.files("rocketpy.plots").joinpath("assets/default_rocket.stl")
        )

    def _validate_animation_inputs(self, file_name, start, stop, time_step):
        """Validate shared input parameters for 3D animation methods."""
        if time_step <= 0:
            raise ValueError(
                f"Invalid time_step: {time_step}. It must be greater than 0."
            )

        if stop is None:
            stop = self.flight.t_final

        if (
            start < 0
            or stop < 0
            or start > self.flight.t_final
            or stop > self.flight.t_final
            or start >= stop
        ):
            raise ValueError(
                f"Invalid animation time range: start={start}, stop={stop}. "
                f"Both must be within [0, {self.flight.t_final}] and start < stop."
            )

        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                f"Could not find the 3D model file: '{file_name}'. "
                "Provide a valid .stl file path."
            )

        return stop

    @staticmethod
    def _rotation_matrix_from_quaternion(q0, q1, q2, q3):
        """Return the body-to-inertial homogeneous rotation matrix."""
        quaternion = np.asarray([q0, q1, q2, q3], dtype=float)
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            return np.eye(4)

        q0, q1, q2, q3 = quaternion / norm
        rotation = np.array(
            [
                [
                    1 - 2 * (q2 * q2 + q3 * q3),
                    2 * (q1 * q2 - q0 * q3),
                    2 * (q1 * q3 + q0 * q2),
                ],
                [
                    2 * (q1 * q2 + q0 * q3),
                    1 - 2 * (q1 * q1 + q3 * q3),
                    2 * (q2 * q3 - q0 * q1),
                ],
                [
                    2 * (q1 * q3 - q0 * q2),
                    2 * (q2 * q3 + q0 * q1),
                    1 - 2 * (q1 * q1 + q2 * q2),
                ],
            ]
        )
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        return transformation

    def _animation_position(self, time_value):
        """Return the rocket position in the East-North-Up AGL frame, in m."""
        return np.array(
            [
                self.flight.x(time_value),
                self.flight.y(time_value),
                self.flight.z(time_value) - self.flight.env.elevation,
            ]
        )

    def _animation_velocity(self, time_value):
        """Return inertial East-North-Up velocity at ``time_value``, in m/s."""
        return np.array(
            [
                self.flight.vx(time_value),
                self.flight.vy(time_value),
                self.flight.vz(time_value),
            ]
        )

    def _animation_wind(self, time_value):
        """Return the wind velocity in the East-North-Up frame, in m/s."""
        return np.array(
            [
                self.flight.wind_velocity_x(time_value),
                self.flight.wind_velocity_y(time_value),
                0.0,
            ]
        )

    @staticmethod
    def _safe_unit_vector(vector, fallback=(0.0, 0.0, 1.0)):
        """Normalize a vector, returning a finite fallback for zero magnitude."""
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            return np.asarray(fallback, dtype=float)
        return vector / norm

    def _animation_transformation(self, time_value, position=None):
        """Return the body-to-inertial transform at ``time_value``."""
        transformation = self._rotation_matrix_from_quaternion(
            self.flight.e0(time_value),
            self.flight.e1(time_value),
            self.flight.e2(time_value),
            self.flight.e3(time_value),
        )
        if position is not None:
            transformation[:3, 3] = position
        return transformation

    @classmethod
    def _direction_arrow(cls, pyvista, direction, scale, start=(0, 0, 0)):
        """Create a slender, constant-length arrow for a vector direction."""
        return pyvista.Arrow(
            start=start,
            direction=cls._safe_unit_vector(direction),
            scale=scale,
            shaft_radius=0.018,
            tip_radius=0.055,
            tip_length=0.18,
            shaft_resolution=16,
            tip_resolution=20,
        )

    @staticmethod
    def _animation_color_scheme():
        """Return the color scheme shared by both PyVista animations.

        Keep animation colors in this single dictionary so the complete visual
        scheme can be adjusted without searching through either scene builder.
        """
        colors = {
            # Background gradient
            "day_bottom": "#8FB3C9",
            "day_top": "#C6DCE8",
            "night_bottom": "#0A0F14",
            "night_top": "#17222C",
            "space_bottom": "#020611",
            "space_top": "#09172A",
            # Scientific overlays and UI
            "panel_background": "#f7f7f759",
            "panel_border": "#45535E",
            "panel_text": "#192229",
            "label_text": "#1B242A",
            "axes": "#536B7A",
            "control_on": "#5E8E76",
            "control_off": "#59636C",
            "control_background": "#C8D0D6",
            "slider_tube": (0.29, 0.34, 0.38),
            "slider_handle": (0.67, 0.72, 0.76),
            "slider_selected": (0.38, 0.56, 0.64),
            "chart_cursor": "#D55E00",
            "chart_altitude": "#0072B2",
            "chart_speed": "#009E73",
            "chart_acceleration": "#D55E00",
            "scalar_cmap": "viridis",
            # Trajectory scene
            "ground": "#D8DED8",
            "ground_grid": "#89968F",
            "simulated_path": "#5B6573",
            "flown_path": "#009E73",
            "velocity": "#E69F00",
            "wind": "#CC79A7",
            "rocket": "#D1D6DA",
            "rocket_legend": "#596873",
            "ground_projection": "#56B4E9",
            "marker_outline": "#17202A",
            "event_start": "#0072B2",
            "event_burnout": "#E69F00",
            "event_apogee": "#F0E442",
            "event_parachute_trigger": "#CC79A7",
            "event_parachute_open": "#56B4E9",
            "event_end": "#D55E00",
            # Attitude reference scene
            "reference_grid": "#83919B",
            "horizon": "#AAB5BD",
            "body_x": "#B96565",
            "body_y": "#6D9B7D",
            "body_z": "#668BAE",
            "center_of_mass": "#F0E442",
            "center_of_pressure": "#D55E00",
        }
        return colors

    def _animation_event_markers(self, start, stop, colors):
        """Return significant flight event times, labels and display colors."""
        events = [(start, "Start", colors["event_start"])]

        burn_out_time = self.flight.rocket.motor.burn_out_time
        if start < burn_out_time < stop:
            events.append((burn_out_time, "Motor burnout", colors["event_burnout"]))

        if start < self.flight.apogee_time < stop:
            events.append((self.flight.apogee_time, "Apogee", colors["event_apogee"]))

        for trigger_time, parachute in self.flight.parachute_events:
            if start < trigger_time < stop:
                events.append(
                    (
                        trigger_time,
                        f"{parachute.name} trigger",
                        colors["event_parachute_trigger"],
                    )
                )
            deployment_time = trigger_time + parachute.lag
            if start < deployment_time < stop:
                events.append(
                    (
                        deployment_time,
                        f"{parachute.name} open",
                        colors["event_parachute_open"],
                    )
                )

        events.append((stop, "End", colors["event_end"]))
        return sorted(events, key=lambda event: event[0])

    @staticmethod
    def _polyline(pyvista, points, *, closed=False):
        """Build a connected ``PolyData`` line, optionally closed."""
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        if len(points) < 2:
            return pyvista.PolyData(points)

        if closed:
            connectivity = np.concatenate(
                ([len(points) + 1], np.arange(len(points)), [0])
            )
        else:
            connectivity = np.concatenate(([len(points)], np.arange(len(points))))
        return pyvista.PolyData(points, lines=connectivity)

    def _animation_scalar(self, time_value, color_by):
        """Return a trajectory coloring scalar at ``time_value``."""
        evaluators = {
            "speed": lambda: float(self.flight.speed(time_value)),
            "mach": lambda: float(self.flight.mach_number(time_value)),
            "dynamic_pressure": lambda: float(self.flight.dynamic_pressure(time_value)),
            "acceleration": lambda: float(self.flight.acceleration(time_value)),
            "altitude": lambda: float(self._animation_position(time_value)[2]),
        }
        return evaluators[color_by]()

    @staticmethod
    def _animation_scalar_metadata(color_by):
        """Return display label and SI unit for a trajectory scalar."""
        return {
            "speed": ("Speed", "m/s"),
            "mach": ("Mach number", "-"),
            "dynamic_pressure": ("Dynamic pressure", "Pa"),
            "acceleration": ("Acceleration", "m/s²"),
            "altitude": ("Altitude AGL", "m"),
        }[color_by]

    @classmethod
    def _polyline_with_scalars(cls, pyvista, points, scalars, scalar_name):
        """Build a polyline carrying one point scalar array."""
        mesh = cls._polyline(pyvista, points)
        mesh.point_data[scalar_name] = np.asarray(scalars, dtype=float)
        return mesh

    @staticmethod
    def _dashed_polyline(  # pylint: disable=too-many-statements
        pyvista, points, *, scalars=None, scalar_name=None, dash_count=32
    ):
        """Build an arc-length-spaced dashed line with optional point scalars."""
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        scalar_values = None if scalars is None else np.asarray(scalars, dtype=float)
        if len(points) < 2:
            mesh = pyvista.PolyData(points)
            if scalar_values is not None:
                mesh.point_data[scalar_name] = scalar_values
            return mesh

        cumulative_distance = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )
        distinct = np.concatenate(
            ([True], np.diff(cumulative_distance) > np.finfo(float).eps)
        )
        points = points[distinct]
        cumulative_distance = cumulative_distance[distinct]
        if scalar_values is not None:
            scalar_values = scalar_values[distinct]
        if len(points) < 2 or cumulative_distance[-1] <= np.finfo(float).eps:
            mesh = pyvista.PolyData(points)
            if scalar_values is not None:
                mesh.point_data[scalar_name] = scalar_values
            return mesh

        dash_count = max(1, min(int(dash_count), len(points) - 1))
        dash_unit = cumulative_distance[-1] / (2 * dash_count - 1)
        dash_distances = np.column_stack(
            (
                2 * np.arange(dash_count) * dash_unit,
                (2 * np.arange(dash_count) + 1) * dash_unit,
            )
        ).ravel()
        dashed_points = np.column_stack(
            [
                np.interp(dash_distances, cumulative_distance, points[:, axis])
                for axis in range(3)
            ]
        )
        starts = 2 * np.arange(dash_count)
        connectivity = np.column_stack(
            (np.full(dash_count, 2), starts, starts + 1)
        ).ravel()
        mesh = pyvista.PolyData(dashed_points, lines=connectivity)
        if scalar_values is not None:
            mesh.point_data[scalar_name] = np.interp(
                dash_distances, cumulative_distance, scalar_values
            )
        return mesh

    def _animation_kinematic_series(self, times):
        """Return altitude, speed and acceleration histories in SI units."""
        return [
            (
                "Altitude AGL (m)",
                np.array([self._animation_position(t)[2] for t in times]),
                "chart_altitude",
            ),
            (
                "Speed (m/s)",
                np.array([self.flight.speed(t) for t in times]),
                "chart_speed",
            ),
            (
                "Acceleration (m/s²)",
                np.array([self.flight.acceleration(t) for t in times]),
                "chart_acceleration",
            ),
        ]

    def _animation_attitude_series(self, times):
        """Return aerodynamic-angle, Euler-angle and body-rate histories."""
        return [
            (
                "Aerodynamic angles (deg)",
                [
                    (
                        "Angle of attack",
                        [self.flight.angle_of_attack(t) for t in times],
                    ),
                    ("Sideslip", [self.flight.angle_of_sideslip(t) for t in times]),
                ],
            ),
            (
                "3-1-3 Euler angles (deg)",
                [
                    ("Precession ψ", [self.flight.psi(t) for t in times]),
                    ("Nutation θ", [self.flight.theta(t) for t in times]),
                    ("Spin φ", [self.flight.phi(t) for t in times]),
                ],
            ),
            (
                "Body angular rates (deg/s)",
                [
                    ("Pitch ω1", np.degrees([self.flight.w1(t) for t in times])),
                    ("Yaw ω2", np.degrees([self.flight.w2(t) for t in times])),
                    ("Roll ω3", np.degrees([self.flight.w3(t) for t in times])),
                ],
            ),
        ]

    @staticmethod
    def _add_animation_charts(  # pylint: disable=too-many-statements
        pyvista,
        plotter,
        times,
        chart_series,
        colors,
        *,
        attitude=False,
        compact=False,
    ):
        """Add compact PyVista history charts and return their time cursors."""
        cursors = []
        line_colors = (
            colors["body_x"],
            colors["body_y"],
            colors["body_z"],
        )
        chart_width = 0.247 if compact else 0.312
        chart_x = 0.743 if compact and attitude else 0.01
        locations = ((chart_x, 0.12), (chart_x, 0.38), (chart_x, 0.64))
        size = (chart_width, 0.243)

        for index, series in enumerate(chart_series):
            chart = pyvista.Chart2D(size=size, loc=locations[index])
            chart.title = series[0]
            chart.background_color = colors["panel_background"]
            chart.border_color = colors["panel_border"]
            chart.x_axis.label = "Time (s)"
            chart.x_axis.label_size = 12
            chart.y_axis.label_size = 12
            chart.x_axis.tick_label_size = 11
            chart.y_axis.tick_label_size = 11

            if attitude:
                values = []
                for line_index, (label, line_values) in enumerate(series[1]):
                    line_values = np.asarray(line_values, dtype=float)
                    values.append(line_values)
                    chart.line(
                        times,
                        line_values,
                        color=line_colors[line_index],
                        width=1.5,
                        label=label,
                    )
                all_values = np.concatenate(values)
            else:
                all_values = np.asarray(series[1], dtype=float)
                chart.line(
                    times,
                    all_values,
                    color=colors[series[2]],
                    width=1.7,
                )

            finite_values = all_values[np.isfinite(all_values)]
            if finite_values.size:
                value_min, value_max = np.min(finite_values), np.max(finite_values)
            else:
                value_min, value_max = 0.0, 1.0
            if np.isclose(value_min, value_max):
                value_min -= 0.5
                value_max += 0.5
            cursor = chart.line(
                [times[0], times[0]],
                [value_min, value_max],
                color=colors["chart_cursor"],
                width=1.2,
            )
            cursors.append((cursor, value_min, value_max))
            plotter.add_chart(chart)
        return cursors

    @staticmethod
    def _update_animation_chart_cursors(cursors, time_value):
        """Move all chart cursors to the selected flight time."""
        for cursor, value_min, value_max in cursors:
            cursor.update([time_value, time_value], [value_min, value_max])

    def _ground_bounds_from_spec(self, spec, fallback_bounds):
        """Convert explicit ENU or latitude/longitude image bounds to ENU."""
        bounds = spec.get("bounds")
        if bounds is None:
            return fallback_bounds
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("ground image bounds must contain four finite values.")
        west, east, south, north = map(float, bounds)
        if not west < east or not south < north:
            raise ValueError(
                "ground image bounds must satisfy west < east and south < north."
            )

        coordinates = spec.get("coordinates", "enu").lower()
        if coordinates == "enu":
            return west, east, south, north
        if coordinates != "latlon":
            raise ValueError("ground image coordinates must be 'enu' or 'latlon'.")

        latitude = float(self.flight.env.latitude)
        longitude = float(self.flight.env.longitude)
        earth_radius = 6_371_000.0
        east_bounds = (
            earth_radius
            * np.cos(np.radians(latitude))
            * np.radians(np.array([west, east]) - longitude)
        )
        north_bounds = earth_radius * np.radians(np.array([south, north]) - latitude)
        return (*east_bounds, *north_bounds)

    @staticmethod
    def _interpolated_camera_path(camera_path, fraction):
        """Interpolate a sequence of PyVista camera positions."""
        if len(camera_path) < 2:
            raise ValueError("camera_path must contain at least two camera positions.")
        scaled = np.clip(fraction, 0, 1) * (len(camera_path) - 1)
        lower = min(int(np.floor(scaled)), len(camera_path) - 2)
        blend = scaled - lower
        camera = []
        for first, second in zip(
            camera_path[lower], camera_path[lower + 1], strict=True
        ):
            camera.append(
                tuple((1 - blend) * np.asarray(first) + blend * np.asarray(second))
            )
        return camera

    @classmethod
    def _update_animation_camera(
        cls,
        plotter,
        mode,
        position,
        rotation,
        scene_span,
        time_value,
        start,
        stop,
        camera_path,
    ):
        """Update the camera from a preset mode or deterministic path."""
        fraction = 0 if stop == start else (time_value - start) / (stop - start)
        if camera_path is not None:
            if callable(camera_path):
                plotter.camera_position = camera_path(time_value)
            else:
                plotter.camera_position = cls._interpolated_camera_path(
                    camera_path, fraction
                )
            return
        if mode == "static":
            return

        span = max(float(scene_span), 1.0)
        if mode == "follow":
            offset = np.array([0.65, -0.85, 0.45]) * span
            camera_position = position + offset
            view_up = (0, 0, 1)
        elif mode == "ground":
            camera_position = position + np.array([0, -0.8 * span, 0.18 * span])
            camera_position[2] = max(camera_position[2], 0.08 * span)
            view_up = (0, 0, 1)
        else:  # body-fixed
            offset = rotation[:3, :3] @ (np.array([0.7, -0.9, 0.35]) * span)
            camera_position = position + offset
            view_up = tuple(rotation[:3, :3] @ np.array([0, 0, 1]))
        plotter.camera_position = [tuple(camera_position), tuple(position), view_up]

    def _rocket_axial_display_coordinate(self, value, display_length):
        """Map a rocket axial coordinate onto the centered display model."""
        coordinates = [
            float(position.z)
            for _surface, position in self.flight.rocket.aerodynamic_surfaces
        ]
        coordinates.extend(
            [
                float(self.flight.rocket.center_of_dry_mass_position),
            ]
        )
        coordinate_min = min(coordinates)
        coordinate_max = max(coordinates)
        physical_span = max(
            coordinate_max - coordinate_min,
            2 * float(self.flight.rocket.radius),
            np.finfo(float).eps,
        )
        coordinate_center = 0.5 * (coordinate_min + coordinate_max)
        orientation_sign = (
            1
            if self.flight.rocket.coordinate_system_orientation == "tail_to_nose"
            else -1
        )
        return (
            orientation_sign
            * (float(value) - coordinate_center)
            * (0.8 * display_length / physical_span)
        )

    def _animation_phase(self, time_value):
        """Return a concise phase label for the telemetry overlay."""
        if time_value <= self.flight.rocket.motor.burn_out_time:
            return "POWERED ASCENT"
        if time_value <= self.flight.apogee_time:
            return "COAST"

        deployed = any(
            event_time + parachute.lag <= time_value
            for event_time, parachute in self.flight.parachute_events
        )
        return "PARACHUTE DESCENT" if deployed else "DESCENT"

    def _trajectory_telemetry(self, time_value):
        """Format the trajectory animation's live telemetry panel."""
        position = self._animation_position(time_value)
        velocity = self._animation_velocity(time_value)
        wind = self._animation_wind(time_value)
        ground_range = np.linalg.norm(position[:2] - self._animation_position(0)[:2])
        lines = [
            self._animation_phase(time_value),
            f"T+ {time_value:7.2f} s",
            f"ALTITUDE   {position[2]:8.1f} m AGL",
            f"SPEED      {np.linalg.norm(velocity):8.1f} m/s",
            f"VERTICAL   {velocity[2]:+8.1f} m/s",
            f"MACH       {self.flight.mach_number(time_value):8.2f}",
            f"WIND       {np.linalg.norm(wind):8.1f} m/s",
            f"RANGE      {ground_range:8.1f} m",
        ]
        width = max(map(len, lines))
        return "\n".join(line.ljust(width) for line in lines)

    def _rotation_telemetry(self, time_value, rotation, include_stability=False):
        """Format the attitude animation's live telemetry panel."""
        body_axis = rotation[:3, 2]
        tilt = np.degrees(np.arccos(np.clip(body_axis[2], -1, 1)))
        heading = np.degrees(np.arctan2(body_axis[0], body_axis[1])) % 360
        velocity = self._animation_velocity(time_value)
        wind = self._animation_wind(time_value)
        angular_rates = np.degrees(
            [
                self.flight.w1(time_value),
                self.flight.w2(time_value),
                self.flight.w3(time_value),
            ]
        )
        telemetry = (
            f"ATTITUDE  T+ {time_value:7.2f} s\n"
            f"TILT {tilt:7.2f} deg   HDG {heading:7.2f} deg\n"
            f"SPEED {np.linalg.norm(velocity):7.1f} m/s  "
            f"WIND {np.linalg.norm(wind):7.1f} m/s\n"
            f"RATES P/Y/R {angular_rates[0]:+6.1f} / "
            f"{angular_rates[1]:+6.1f} / {angular_rates[2]:+6.1f} deg/s"
        )
        if include_stability:
            center_of_mass = self.flight.rocket.center_of_mass(time_value)
            center_of_pressure = self.flight.rocket.cp_position(
                self.flight.mach_number(time_value)
            )
            telemetry += (
                f"\nCM {center_of_mass:+7.3f} m   CP {center_of_pressure:+7.3f} m"
                f"\nSTATIC MARGIN {self.flight.rocket.static_margin(time_value):+6.2f} cal"
            )
        return telemetry

    def _animation_background_palette(self, background_color=None, colors=None):
        """Return launch and near-space background color pairs."""
        colors = colors or self._animation_color_scheme()
        if background_color is not None:
            launch_color = np.asarray(to_rgb(background_color))
            launch_bottom = launch_color
            launch_top = launch_color
        else:
            local_date = getattr(self.flight.env, "local_date", None)
            is_daylight = local_date is None or 6 <= local_date.hour < 20
            if is_daylight:
                launch_bottom = np.asarray(to_rgb(colors["day_bottom"]))
                launch_top = np.asarray(to_rgb(colors["day_top"]))
            else:
                launch_bottom = np.asarray(to_rgb(colors["night_bottom"]))
                launch_top = np.asarray(to_rgb(colors["night_top"]))

        return {
            "launch_bottom": launch_bottom,
            "launch_top": launch_top,
            "space_bottom": np.asarray(to_rgb(colors["space_bottom"])),
            "space_top": np.asarray(to_rgb(colors["space_top"])),
        }

    @staticmethod
    def _animation_background_at_altitude(palette, altitude_agl):
        """Linearly blend the launch palette into near-space by 50 km AGL."""
        blend = float(np.clip(altitude_agl / 50_000, 0, 1))
        bottom = (1 - blend) * palette["launch_bottom"] + blend * palette[
            "space_bottom"
        ]
        top = (1 - blend) * palette["launch_top"] + blend * palette["space_top"]
        return tuple(bottom), tuple(top)

    @classmethod
    def _set_animation_background(cls, plotter, palette, altitude_agl):
        """Update the scene background for the current altitude."""
        bottom, top = cls._animation_background_at_altitude(palette, altitude_agl)
        plotter.set_background(bottom, top=top)

    @staticmethod
    def _style_telemetry_actor(actor, colors):
        """Give telemetry a compact Matplotlib-like annotation box."""
        text_property = actor.GetTextProperty()
        text_property.background_color = colors["panel_background"]
        text_property.background_opacity = 0.88
        text_property.show_frame = True
        text_property.frame_color = colors["panel_border"]
        text_property.frame_width = 1

    @staticmethod
    def _style_legend_actor(actor, colors):
        """Give a PyVista legend compact scientific-plot styling."""
        text_property = actor.GetEntryTextProperty()
        text_property.SetFontSize(7)
        text_property.SetColor(0.10, 0.14, 0.17)
        actor.GetBoxProperty().SetColor(*to_rgb(colors["panel_border"]))
        actor.SetPadding(2)

    @classmethod
    def _style_animation_plotter(
        cls, plotter, palette, colors, *, show_kinematic_plots=False
    ):
        """Apply RocketPy's animation scene style."""
        cls._set_animation_background(plotter, palette, 0)
        # SSAA scales VTK's 2D chart layer independently from the 3D scene,
        # shifting normalized chart positions by roughly half a viewport.
        plotter.enable_anti_aliasing("msaa")
        plotter.add_axes(
            xlabel="E — East",
            ylabel="N — North",
            zlabel="U — Up",
            color=colors["axes"],
            line_width=1,
            viewport=(
                (0.63, 0.09, 0.76, 0.22)
                if show_kinematic_plots
                else (0.20, 0.09, 0.33, 0.22)
            ),
        )

    @staticmethod
    def _style_animation_slider(widget, colors):
        """Apply a compact neutral style to a native PyVista slider."""
        representation = widget.GetRepresentation()
        representation.SetSliderLength(0.025)
        representation.SetSliderWidth(0.012)
        representation.SetTubeWidth(0.004)
        representation.SetEndCapLength(0.006)
        representation.SetEndCapWidth(0.012)
        representation.GetTubeProperty().SetColor(*colors["slider_tube"])
        representation.GetCapProperty().SetColor(*colors["slider_tube"])
        representation.GetSliderProperty().SetColor(*colors["slider_handle"])
        representation.GetSelectedProperty().SetColor(*colors["slider_selected"])

    @staticmethod
    def _animation_options(  # pylint: disable=too-many-statements
        kwargs,
    ):
        """Remove and validate RocketPy-specific options from Plotter kwargs."""
        options = {
            "background_color": kwargs.pop("background_color", None),
            "playback_controls": kwargs.pop("playback_controls", True),
            "show_subrocket_point": kwargs.pop("show_subrocket_point", True),
            "ground_image": kwargs.pop("ground_image", None),
            "ground_image_bounds": kwargs.pop("ground_image_bounds", None),
            "ground_image_coordinates": kwargs.pop("ground_image_coordinates", "enu"),
            "ground_image_flip_y": kwargs.pop("ground_image_flip_y", False),
            "backend": kwargs.pop("backend", "auto"),
            "force_external": kwargs.pop("force_external", False),
            "shadows": kwargs.pop("shadows", False),
            "trajectory_line_width": kwargs.pop("trajectory_line_width", 4),
            "color_by": kwargs.pop("color_by", "speed"),
            "show_kinematic_plots": kwargs.pop("show_kinematic_plots", False),
            "camera_mode": kwargs.pop("camera_mode", "static"),
            "camera_path": kwargs.pop("camera_path", None),
            "show_attitude_plots": kwargs.pop("show_attitude_plots", False),
            "show_cp_cm": kwargs.pop("show_cp_cm", False),
            "export_file": kwargs.pop("export_file", None),
            "export_fps": kwargs.pop("export_fps", 30),
            "export_resolution": kwargs.pop("export_resolution", None),
            "transparent_background": kwargs.pop("transparent_background", False),
            "color_scheme": kwargs.pop("color_scheme", None),
        }
        valid_backends = {"auto", "none", "trame", "client"}
        if options["backend"] not in valid_backends:
            raise ValueError(
                f"Invalid backend: {options['backend']!r}. Expected one of "
                f"{sorted(valid_backends)}."
            )
        line_width = options["trajectory_line_width"]
        if isinstance(line_width, bool) or not isinstance(line_width, (int, float)):
            raise TypeError("trajectory_line_width must be a positive number.")
        if line_width <= 0:
            raise ValueError("trajectory_line_width must be greater than 0.")
        options["trajectory_line_width"] = float(line_width)

        color_by = options["color_by"]
        if color_by is False or color_by is None:
            options["color_by"] = None
        elif not isinstance(color_by, str) or color_by.lower() not in {
            "speed",
            "mach",
            "dynamic_pressure",
            "acceleration",
            "altitude",
        }:
            raise ValueError(
                "color_by must be one of 'speed', 'mach', 'dynamic_pressure', "
                "'acceleration', 'altitude', False or None."
            )
        else:
            options["color_by"] = color_by.lower()

        camera_mode = options["camera_mode"]
        if camera_mode is True:
            camera_mode = "follow"
        elif camera_mode is False or camera_mode is None:
            camera_mode = "static"
        if not isinstance(camera_mode, str) or camera_mode.lower() not in {
            "static",
            "follow",
            "ground",
            "body",
        }:
            raise ValueError(
                "camera_mode must be 'static', 'follow', 'ground', 'body', "
                "True or False."
            )
        options["camera_mode"] = camera_mode.lower()

        camera_path = options["camera_path"]
        if (
            camera_path is not None
            and not callable(camera_path)
            and (
                isinstance(camera_path, (str, bytes))
                or not isinstance(camera_path, Sequence)
            )
        ):
            raise TypeError(
                "camera_path must be callable or a camera-position sequence."
            )

        export_fps = options["export_fps"]
        if isinstance(export_fps, bool) or not isinstance(export_fps, (int, float)):
            raise TypeError("export_fps must be a positive number.")
        if export_fps <= 0:
            raise ValueError("export_fps must be greater than 0.")
        options["export_fps"] = float(export_fps)

        resolution = options["export_resolution"]
        if resolution is not None:
            if (
                not isinstance(resolution, Sequence)
                or len(resolution) != 2
                or any(
                    isinstance(value, bool) or int(value) <= 0 for value in resolution
                )
            ):
                raise ValueError(
                    "export_resolution must contain two positive integer values."
                )
            options["export_resolution"] = tuple(map(int, resolution))

        export_file = options["export_file"]
        if export_file is not None:
            extension = os.path.splitext(os.fspath(export_file))[1].lower()
            if extension not in {".gif", ".mp4"}:
                raise ValueError("export_file must end in '.gif' or '.mp4'.")
            if options["force_external"]:
                raise ValueError("force_external cannot be used with export_file.")
            if options["transparent_background"] and extension == ".mp4":
                raise ValueError(
                    "transparent_background is supported for GIF export, not MP4."
                )
            if (
                extension == ".mp4"
                and resolution is not None
                and any(value % 2 for value in options["export_resolution"])
            ):
                raise ValueError("MP4 export_resolution values must be even.")

        color_scheme = options["color_scheme"]
        if color_scheme is not None and not isinstance(color_scheme, Mapping):
            raise TypeError("color_scheme must be a mapping of palette keys to colors.")
        return options

    @classmethod
    def _resolved_animation_colors(cls, override):
        """Merge a user color override into the centralized default scheme."""
        colors = cls._animation_color_scheme()
        if override is None:
            return colors
        unknown = set(override) - set(colors)
        if unknown:
            raise ValueError(f"Unknown color_scheme keys: {sorted(unknown)}.")
        colors.update(override)
        return colors

    @staticmethod
    def _run_animation(  # pylint: disable=too-many-statements,too-many-locals
        plotter,
        update_frame,
        start,
        stop,
        time_step,
        playback_speed,
        *,
        colors,
        playback_controls=True,
        backend="auto",
        force_external=False,
        export_file=None,
        export_fps=30,
        transparent_background=False,
    ):
        """Add playback controls and start PyVista's timer-driven event loop."""
        if playback_speed <= 0:
            raise ValueError(
                f"Invalid playback_speed: {playback_speed}. It must be greater than 0."
            )

        if export_file is not None:
            plotter.image_transparent_background = bool(transparent_background)
            extension = os.path.splitext(os.fspath(export_file))[1].lower()
            transparent_gif = extension == ".gif" and transparent_background
            if extension == ".gif" and not transparent_gif:
                plotter.open_gif(os.fspath(export_file), fps=export_fps)
            elif extension == ".mp4":
                plotter.open_movie(
                    os.fspath(export_file),
                    framerate=int(round(export_fps)),
                    macro_block_size=1,
                )
            duration = (stop - start) / playback_speed
            frame_count = max(int(round(duration * export_fps)) + 1, 2)
            transparent_frames = []
            try:
                for time_value in np.linspace(start, stop, frame_count):
                    update_frame(float(time_value))
                    if transparent_gif:
                        transparent_frames.append(
                            np.asarray(
                                plotter.screenshot(
                                    transparent_background=True,
                                    return_img=True,
                                )
                            )
                        )
                    else:
                        plotter.write_frame()
            finally:
                plotter.close()
            if transparent_gif:
                image_module = import_optional_dependency("PIL.Image")
                palette_frames = []
                for frame in transparent_frames:
                    rgba_frame = image_module.fromarray(frame, mode="RGBA")
                    alpha = rgba_frame.getchannel("A")
                    palette_frame = rgba_frame.convert("RGB").convert(
                        "P", palette=image_module.Palette.ADAPTIVE, colors=255
                    )
                    palette_frame.paste(255, mask=alpha.point(lambda value: value == 0))
                    palette_frames.append(palette_frame)
                palette_frames[0].save(
                    os.fspath(export_file),
                    save_all=True,
                    append_images=palette_frames[1:],
                    duration=round(1000 / export_fps),
                    loop=0,
                    disposal=2,
                    transparency=255,
                )
            return os.fspath(export_file)

        speed_values = sorted({0.5, 1.0, 2.0, 3.0, float(playback_speed)})
        speed_labels = [f"{speed:g}x" for speed in speed_values]
        state = {
            "time": float(start),
            "speed": float(playback_speed),
            "playing": True,
            "last_tick": time.perf_counter(),
            "accumulator": 0.0,
        }
        controls = {}

        def set_time(value):
            state["time"] = float(np.clip(value, start, stop))
            state["last_tick"] = time.perf_counter()
            state["accumulator"] = 0.0
            update_frame(state["time"])

        if playback_controls:
            controls["timeline"] = plotter.add_slider_widget(
                set_time,
                (start, stop),
                value=start,
                title="FLIGHT TIME (s)",
                pointa=(0.14, 0.055),
                pointb=(0.70, 0.055),
                color=colors["axes"],
                title_color=colors["axes"],
                interaction_event="always",
                style="modern",
                fmt="%6.2f",
                slider_width=0.018,
                tube_width=0.006,
            )
            _FlightPlots._style_animation_slider(controls["timeline"], colors)

            def set_speed(label):
                state["speed"] = speed_values[speed_labels.index(label)]
                state["last_tick"] = time.perf_counter()

            controls["speed"] = plotter.add_text_slider_widget(
                set_speed,
                speed_labels,
                value=speed_values.index(float(playback_speed)),
                pointa=(0.77, 0.055),
                pointb=(0.95, 0.055),
                color=colors["axes"],
                interaction_event="end",
                style="modern",
            )
            _FlightPlots._style_animation_slider(controls["speed"], colors)

            def set_playing(is_playing):
                if is_playing and state["time"] >= stop:
                    set_time(start)
                    controls["timeline"].GetRepresentation().SetValue(start)
                state["playing"] = bool(is_playing)
                state["last_tick"] = time.perf_counter()

            controls["play"] = plotter.add_checkbox_button_widget(
                set_playing,
                value=True,
                position=(22, 24),
                size=28,
                border_size=2,
                color_on=colors["control_on"],
                color_off=colors["control_off"],
                background_color=colors["control_background"],
            )
            plotter.add_text(
                "PLAY / PAUSE",
                position=(60, 30),
                font_size=10,
                color=colors["axes"],
            )
        else:
            update_frame(start)

        def advance(_step):
            now = time.perf_counter()
            elapsed = now - state["last_tick"]
            state["last_tick"] = now
            if not state["playing"]:
                return

            state["accumulator"] += elapsed * state["speed"]
            if state["accumulator"] < time_step:
                return

            next_time = min(state["time"] + state["accumulator"], stop)
            state["accumulator"] = 0.0
            state["time"] = next_time
            if playback_controls:
                controls["timeline"].GetRepresentation().SetValue(next_time)
            update_frame(next_time)
            if next_time >= stop:
                state["playing"] = False
                if playback_controls:
                    controls["play"].GetRepresentation().SetState(0)

        plotter.add_timer_event(
            max_steps=np.iinfo(np.int32).max,
            duration=16,
            callback=advance,
        )
        try:
            show_kwargs = {"auto_close": False}
            selected_backend = "none" if force_external else backend
            if selected_backend != "auto":
                show_kwargs["jupyter_backend"] = selected_backend
            plotter.show(**show_kwargs)
        finally:
            plotter.close()
        return None

    def animate_trajectory(  # pylint: disable=too-many-statements,too-many-locals
        self,
        file_name=None,
        start=0,
        stop=None,
        time_step=0.1,
        playback_speed=1.0,
        **kwargs,
    ):
        """Animate the 6-DOF trajectory and attitude using PyVista.

        Parameters
        ----------
        file_name : str | None, optional
            Path to a 3D model file representing the rocket, usually ``.stl``.
            If None, RocketPy uses a built-in default STL model.
            Default is None.
        start : int, float, optional
            Animation start time in seconds. Default is 0.
        stop : int, float | None, optional
            Animation end time in seconds. If None, uses ``flight.t_final``.
            Default is None.
        time_step : float, optional
            Animation frame step in seconds. Must be greater than 0.
            Default is 0.1.
        playback_speed : float, optional
            Ratio of simulation time to wall-clock playback time. For example,
            ``2`` plays at twice real time. Must be greater than 0. Default is 1.
        **kwargs : dict, optional
            RocketPy animation options and additional keyword arguments passed
            to :class:`pyvista.Plotter`. See Notes.

        Other Parameters
        ----------------
        background_color : color-like | None, optional
            Launch background override. None selects a daylight or night
            palette. Default is None.
        playback_controls : bool, optional
            Display play/pause, timeline and playback-speed controls. Default
            is True.
        show_subrocket_point : bool, optional
            Display the rocket's vertical projection on the ground plane.
            Default is True.
        ground_image : path-like | pyvista.Texture | mapping | None, optional
            Ground texture. A mapping may define ``image``, ``bounds``,
            ``coordinates`` (``"enu"`` or ``"latlon"``), and ``flip_y`` for
            geographic placement. Default is None.
        color_by : str | bool | None, optional
            Trajectory point scalar, one of ``"speed"``, ``"mach"``,
            ``"dynamic_pressure"``, ``"acceleration"``, ``"altitude"``,
            ``False`` or ``None``. Default is "speed".
        show_kinematic_plots : bool, optional
            Show altitude, speed and acceleration histories. Default is False.
        camera_mode : {"static", "follow", "ground", "body"}, optional
            Camera tracking preset. Default is "static".
        camera_path : callable | sequence | None, optional
            Custom camera function or interpolated camera positions. Default
            is None.
        backend : {"auto", "none", "trame", "client"}, optional
            Visualization backend. Default is "auto".
        force_external : bool, optional
            Force rendering in an external window. Default is False.
        shadows : bool, optional
            Enable PyVista scene shadows. Scientific overlays remain unlit so
            their colors stay camera-independent. Default is False.
        trajectory_line_width : float, optional
            Width of the flown trajectory; related path widths scale from this
            value. Default is 4.
        export_file : path-like | None, optional
            Deterministic ``.gif`` or ``.mp4`` output. Default is None.
        export_fps : float, optional
            Export frame rate. Default is 30.
        export_resolution : tuple[int, int] | None, optional
            Export width and height in pixels. Default is None.
        transparent_background : bool, optional
            Enable GIF alpha transparency. Default is False.
        color_scheme : mapping | None, optional
            Overrides merged into the default animation color dictionary.
            Default is None.

        Notes
        -----
        Coordinates use the inertial East-North-Up frame and metres. Altitude
        is above ground level. Wind arrows point in the direction the air is
        moving. The rocket is display-scaled so it remains visible. Native
        controls provide play/pause, time scrubbing and playback-speed
        selection.

        RocketPy options accepted through ``kwargs`` are ``background_color``,
        ``playback_controls``, ``show_subrocket_point``, ``ground_image``,
        ``color_by``, charts, camera, export, backend and styling options.
        """
        pyvista = import_optional_dependency("pyvista")
        options = self._animation_options(kwargs)
        colors = self._resolved_animation_colors(options["color_scheme"])
        file_name = self._resolve_animation_model_path(file_name)
        stop = self._validate_animation_inputs(file_name, start, stop, time_step)
        if playback_speed <= 0:
            raise ValueError(
                f"Invalid playback_speed: {playback_speed}. It must be greater than 0."
            )
        frame_times = np.append(np.arange(start, stop, time_step), stop)
        path_times = np.linspace(start, stop, min(max(len(frame_times), 120), 800))
        path_points = np.array([self._animation_position(t) for t in path_times])
        background_palette = self._animation_background_palette(
            options["background_color"], colors
        )

        kwargs.setdefault("window_size", options["export_resolution"] or (1280, 800))
        if options["export_file"] is not None:
            kwargs["notebook"] = False
            kwargs["off_screen"] = True
        if options["force_external"]:
            kwargs["notebook"] = False
            kwargs["off_screen"] = False
        plotter = pyvista.Plotter(**kwargs)
        self._style_animation_plotter(
            plotter,
            background_palette,
            colors,
            show_kinematic_plots=options["show_kinematic_plots"],
        )
        if options["shadows"]:
            plotter.enable_shadows()

        base_rocket = pyvista.read(file_name)
        base_rocket.translate(-np.asarray(base_rocket.center), inplace=True)
        scene_span = max(np.ptp(path_points, axis=0).max(), 50.0)
        display_length = max(base_rocket.length, scene_span * 0.025)
        base_rocket.scale(display_length / base_rocket.length, inplace=True)

        initial_position = self._animation_position(start)
        rocket = base_rocket.transform(
            self._animation_transformation(start, initial_position), inplace=False
        )
        color_by = options["color_by"]
        scalar_name = None
        scalar_values = None
        scalar_clim = None
        if color_by is not None:
            scalar_label, scalar_unit = self._animation_scalar_metadata(color_by)
            scalar_name = f"{scalar_label} ({scalar_unit})"
            scalar_values = np.array(
                [self._animation_scalar(t, color_by) for t in path_times]
            )
            finite_scalars = scalar_values[np.isfinite(scalar_values)]
            scalar_clim = (
                (float(np.min(finite_scalars)), float(np.max(finite_scalars)))
                if finite_scalars.size
                else (0.0, 1.0)
            )
            if np.isclose(*scalar_clim):
                scalar_clim = (scalar_clim[0] - 0.5, scalar_clim[1] + 0.5)
            simulated_path = self._dashed_polyline(
                pyvista,
                path_points,
                scalars=scalar_values,
                scalar_name=scalar_name,
            )
            flown_path = self._polyline_with_scalars(
                pyvista,
                [initial_position],
                [self._animation_scalar(start, color_by)],
                scalar_name,
            )
        else:
            simulated_path = self._dashed_polyline(pyvista, path_points)
            flown_path = self._polyline(pyvista, [initial_position])
        velocity = self._animation_velocity(start)
        velocity_arrow = self._direction_arrow(
            pyvista,
            velocity,
            display_length * 1.8,
            start=initial_position,
        )
        wind = self._animation_wind(start)
        wind_arrow = self._direction_arrow(
            pyvista,
            wind,
            display_length * 1.55,
            start=initial_position,
        )

        horizontal_span = max(np.ptp(path_points[:, :2], axis=0).max() * 1.25, 50)
        ground_center = np.mean(path_points[:, :2], axis=0)
        fallback_bounds = (
            ground_center[0] - horizontal_span / 2,
            ground_center[0] + horizontal_span / 2,
            ground_center[1] - horizontal_span / 2,
            ground_center[1] + horizontal_span / 2,
        )
        image = options["ground_image"]
        image_spec = {
            "image": image,
            "bounds": options["ground_image_bounds"],
            "coordinates": options["ground_image_coordinates"],
            "flip_y": options["ground_image_flip_y"],
        }
        if isinstance(image, Mapping):
            image_spec.update(image)
            image = image_spec.get("image")
            if image is None:
                raise ValueError("ground_image mapping must define an 'image' value.")
        ground_bounds = self._ground_bounds_from_spec(image_spec, fallback_bounds)
        west, east, south, north = ground_bounds
        ground = pyvista.Plane(
            center=((west + east) / 2, (south + north) / 2, 0),
            direction=(0, 0, 1),
            i_size=east - west,
            j_size=north - south,
            i_resolution=20,
            j_resolution=20,
        )
        if image is None:
            plotter.add_mesh(
                ground,
                color=colors["ground"],
                opacity=0.55,
                show_edges=True,
                edge_color=colors["ground_grid"],
                line_width=1,
                lighting=False,
            )
        else:
            texture = image
            if isinstance(texture, (str, os.PathLike)):
                texture = pyvista.read_texture(os.fspath(texture))
            if image_spec.get("flip_y", False):
                texture = texture.flip_y()
            ground.texture_map_to_plane(use_bounds=True, inplace=True)
            plotter.add_mesh(
                ground,
                texture=texture,
                opacity=0.92,
                lighting=False,
            )
        simulated_options = {
            "opacity": 0.72,
            "line_width": max(1, options["trajectory_line_width"] * 0.45),
            "lighting": False,
            "label": "Simulated path",
        }
        flown_options = {
            "line_width": options["trajectory_line_width"],
            "lighting": False,
            "label": "Flown path",
        }
        simulated_options["color"] = colors["simulated_path"]
        if color_by is None:
            flown_options["color"] = colors["flown_path"]
        else:
            scalar_bar_args = {
                "title": scalar_name,
                "position_x": 0.76,
                "position_y": 0.14,
                "width": 0.2,
                "height": 0.065,
                "title_font_size": 10,
                "label_font_size": 9,
                "color": colors["panel_text"],
            }
            flown_options.update(
                scalars=scalar_name,
                cmap=colors["scalar_cmap"],
                clim=scalar_clim,
                show_scalar_bar=True,
                scalar_bar_args=scalar_bar_args,
            )
        plotter.add_mesh(simulated_path, **simulated_options)
        plotter.add_mesh(flown_path, **flown_options)
        velocity_actor = plotter.add_mesh(
            velocity_arrow,
            color=colors["velocity"],
            lighting=False,
            label="Velocity direction",
        )
        velocity_actor.SetVisibility(bool(np.linalg.norm(velocity) > 1e-12))
        wind_actor = plotter.add_mesh(
            wind_arrow,
            color=colors["wind"],
            lighting=False,
            label="Wind velocity (toward)",
        )
        wind_actor.SetVisibility(bool(np.linalg.norm(wind) > 1e-12))
        plotter.add_mesh(
            rocket,
            color=colors["rocket"],
            smooth_shading=True,
            specular=0.18,
            specular_power=18,
            label="Rocket (not to scale)",
        )

        subrocket_point = None
        if options["show_subrocket_point"]:
            subrocket_point = pyvista.PolyData(
                [initial_position[0], initial_position[1], 0]
            )
            plotter.add_mesh(
                subrocket_point,
                style="points",
                color=colors["marker_outline"],
                point_size=17,
                render_points_as_spheres=True,
                lighting=False,
            )
            plotter.add_mesh(
                subrocket_point,
                style="points",
                color=colors["ground_projection"],
                point_size=10,
                render_points_as_spheres=True,
                lighting=False,
                label="Ground projection",
            )

        marker_events = self._animation_event_markers(start, stop, colors)
        marker_points = np.array(
            [self._animation_position(event_time) for event_time, _, _ in marker_events]
        )
        marker_labels = [label for _, label, _ in marker_events]
        for point, (_, _label, color) in zip(marker_points, marker_events, strict=True):
            plotter.add_points(
                point[np.newaxis, :],
                color=colors["marker_outline"],
                point_size=16,
                render_points_as_spheres=True,
                lighting=False,
            )
            plotter.add_points(
                point[np.newaxis, :],
                color=color,
                point_size=9,
                render_points_as_spheres=True,
                lighting=False,
            )
        plotter.add_point_labels(
            marker_points,
            marker_labels,
            font_size=10,
            text_color=colors["label_text"],
            shape_color=colors["panel_background"],
            shape_opacity=0.9,
            point_size=0,
            always_visible=True,
        )

        telemetry_position = (
            "upper_right" if options["show_kinematic_plots"] else "upper_left"
        )
        legend_position = (
            "lower right" if options["show_kinematic_plots"] else "upper right"
        )
        telemetry = plotter.add_text(
            self._trajectory_telemetry(start),
            position=telemetry_position,
            font="courier",
            font_size=9,
            color=colors["panel_text"],
            shadow=False,
        )
        self._style_telemetry_actor(telemetry, colors)
        legend = plotter.add_legend(
            labels=[
                ["Simulated path", colors["simulated_path"]],
                ["Flown path", colors["flown_path"]],
                ["Velocity direction", colors["velocity"]],
                ["Wind velocity (toward)", colors["wind"]],
                *(
                    [["Ground projection", colors["ground_projection"]]]
                    if options["show_subrocket_point"]
                    else []
                ),
                ["Rocket (not to scale)", colors["rocket_legend"]],
            ],
            bcolor=colors["panel_background"],
            border=True,
            background_opacity=0.88,
            size=(0.145, 0.115),
            loc=legend_position,
        )
        self._style_legend_actor(legend, colors)
        if options["show_kinematic_plots"]:
            # Keep the scene key directly above the scalar bar and speed
            # selector instead of occupying chart space at the left.
            legend.SetPosition(0.815, 0.225)
        chart_cursors = []
        if options["show_kinematic_plots"]:
            chart_cursors = self._add_animation_charts(
                pyvista,
                plotter,
                path_times,
                self._animation_kinematic_series(path_times),
                colors,
            )
        plotter.show_bounds(
            ztitle="Altitude AGL (m)",
            color=colors["axes"],
            show_xaxis=False,
            show_yaxis=False,
            show_xlabels=False,
            show_ylabels=False,
            n_zlabels=5,
            grid="back",
            location="outer",
        )
        plotter.view_isometric()
        plotter.set_viewup((0, 0, 1))
        plotter.reset_camera()

        def update_frame(time_value):
            position = self._animation_position(time_value)
            self._set_animation_background(
                plotter, background_palette, max(position[2], 0)
            )
            transformed_rocket = base_rocket.transform(
                self._animation_transformation(time_value, position), inplace=False
            )
            rocket.copy_from(transformed_rocket)
            if subrocket_point is not None:
                subrocket_point.copy_from(
                    pyvista.PolyData([position[0], position[1], 0])
                )

            flown_points = path_points[path_times < time_value]
            flown_points = np.vstack((flown_points, position))
            if color_by is None:
                updated_flown_path = self._polyline(pyvista, flown_points)
            else:
                flown_times = np.append(path_times[path_times < time_value], time_value)
                flown_scalars = [
                    self._animation_scalar(t, color_by) for t in flown_times
                ]
                updated_flown_path = self._polyline_with_scalars(
                    pyvista, flown_points, flown_scalars, scalar_name
                )
            flown_path.copy_from(updated_flown_path)
            current_velocity = self._animation_velocity(time_value)
            velocity_arrow.copy_from(
                self._direction_arrow(
                    pyvista,
                    current_velocity,
                    display_length * 1.8,
                    start=position,
                )
            )
            velocity_actor.SetVisibility(bool(np.linalg.norm(current_velocity) > 1e-12))
            current_wind = self._animation_wind(time_value)
            wind_arrow.copy_from(
                self._direction_arrow(
                    pyvista,
                    current_wind,
                    display_length * 1.55,
                    start=position,
                )
            )
            wind_actor.SetVisibility(bool(np.linalg.norm(current_wind) > 1e-12))
            telemetry.set_text(
                telemetry_position, self._trajectory_telemetry(time_value)
            )
            self._update_animation_chart_cursors(chart_cursors, time_value)
            self._update_animation_camera(
                plotter,
                options["camera_mode"],
                position,
                self._animation_transformation(time_value),
                scene_span,
                time_value,
                start,
                stop,
                options["camera_path"],
            )

        return self._run_animation(
            plotter,
            update_frame,
            start,
            stop,
            time_step,
            playback_speed,
            colors=colors,
            playback_controls=options["playback_controls"],
            backend=options["backend"],
            force_external=options["force_external"],
            export_file=options["export_file"],
            export_fps=options["export_fps"],
            transparent_background=options["transparent_background"],
        )

    def animate_rotate(  # pylint: disable=too-many-statements,too-many-locals
        self,
        file_name=None,
        start=0,
        stop=None,
        time_step=0.1,
        playback_speed=1.0,
        **kwargs,
    ):
        """Animate rocket attitude in an inertial reference scene using PyVista.

        Parameters
        ----------
        file_name : str | None, optional
            Path to a 3D model file representing the rocket, usually ``.stl``.
            If None, RocketPy uses a built-in default STL model.
            Default is None.
        start : int, float, optional
            Animation start time in seconds. Default is 0.
        stop : int, float | None, optional
            Animation end time in seconds. If None, uses ``flight.t_final``.
            Default is None.
        time_step : float, optional
            Animation frame step in seconds. Must be greater than 0.
            Default is 0.1.
        playback_speed : float, optional
            Ratio of simulation time to wall-clock playback time. For example,
            ``2`` plays at twice real time. Must be greater than 0. Default is 1.
        **kwargs : dict, optional
            RocketPy animation options and additional keyword arguments passed
            to :class:`pyvista.Plotter`. See Notes.

        Other Parameters
        ----------------
        background_color : color-like | None, optional
            Launch background override. None selects a daylight or night
            palette. Default is None.
        playback_controls : bool, optional
            Display play/pause, timeline and playback-speed controls. Default
            is True.
        backend : {"auto", "none", "trame", "client"}, optional
            Visualization backend. Default is "auto".
        force_external : bool, optional
            Force rendering in an external window. Default is False.
        shadows : bool, optional
            Enable PyVista scene shadows. Body and direction overlays remain
            unlit so their colors stay camera-independent. Default is False.
        show_kinematic_plots : bool, optional
            Show altitude, speed and acceleration histories. Default is False.
        show_attitude_plots : bool, optional
            Show aerodynamic angles, 3-1-3 Euler angles and body angular-rate
            histories. Default is False.
        show_cp_cm : bool, optional
            Show dynamic center-of-mass and center-of-pressure markers and
            telemetry. Default is False.
        camera_mode : {"static", "follow", "ground", "body"}, optional
            Camera tracking preset. Default is "static".
        camera_path : callable | sequence | None, optional
            Custom camera function or interpolated camera positions. Default
            is None.
        export_file : path-like | None, optional
            Deterministic ``.gif`` or ``.mp4`` output. Default is None.
        export_fps : float, optional
            Export frame rate. Default is 30.
        export_resolution : tuple[int, int] | None, optional
            Export width and height in pixels. Default is None.
        transparent_background : bool, optional
            Enable GIF alpha transparency. Default is False.
        color_scheme : mapping | None, optional
            Overrides merged into the default animation color dictionary.
            Default is None.

        Notes
        -----
        The fixed reference frame is East-North-Up. Body X, Y and Z correspond
        to pitch, yaw and roll axes respectively. Angular rates are displayed
        in degrees per second. Velocity and wind arrows show inertial directions
        at the selected time; the wind arrow points toward air motion. Native
        controls provide play/pause, time scrubbing and playback-speed selection.

        RocketPy options accepted through ``kwargs`` are ``background_color``,
        ``playback_controls``, charts, stability markers, camera, export,
        backend and styling options.
        Trajectory-only options are accepted and ignored so shared option
        dictionaries can be used with both animation methods.
        """
        pyvista = import_optional_dependency("pyvista")
        options = self._animation_options(kwargs)
        colors = self._resolved_animation_colors(options["color_scheme"])
        file_name = self._resolve_animation_model_path(file_name)
        stop = self._validate_animation_inputs(file_name, start, stop, time_step)
        if playback_speed <= 0:
            raise ValueError(
                f"Invalid playback_speed: {playback_speed}. It must be greater than 0."
            )
        sample_count = min(max(int(np.ceil((stop - start) / time_step)) + 1, 120), 800)
        history_times = np.linspace(start, stop, sample_count)
        background_palette = self._animation_background_palette(
            options["background_color"], colors
        )

        kwargs.setdefault("window_size", options["export_resolution"] or (1100, 800))
        if options["export_file"] is not None:
            kwargs["notebook"] = False
            kwargs["off_screen"] = True
        if options["force_external"]:
            kwargs["notebook"] = False
            kwargs["off_screen"] = False
        plotter = pyvista.Plotter(**kwargs)
        self._style_animation_plotter(
            plotter,
            background_palette,
            colors,
            show_kinematic_plots=options["show_kinematic_plots"],
        )
        if options["shadows"]:
            plotter.enable_shadows()

        base_rocket = pyvista.read(file_name)
        base_rocket.translate(-np.asarray(base_rocket.center), inplace=True)
        rocket = base_rocket.transform(
            self._animation_transformation(start), inplace=False
        )
        reference_radius = base_rocket.length * 0.8
        arrow_scale = base_rocket.length * 0.64
        rotation = self._animation_transformation(start)
        body_arrows = [
            self._direction_arrow(pyvista, rotation[:3, index], arrow_scale)
            for index in range(3)
        ]
        velocity = self._animation_velocity(start)
        velocity_arrow = self._direction_arrow(pyvista, velocity, arrow_scale * 0.92)
        wind = self._animation_wind(start)
        wind_arrow = self._direction_arrow(pyvista, wind, arrow_scale * 0.82)

        plotter.add_mesh(
            pyvista.Sphere(
                radius=reference_radius, theta_resolution=36, phi_resolution=18
            ),
            style="wireframe",
            color=colors["reference_grid"],
            opacity=0.11,
            line_width=1,
            lighting=False,
        )
        theta = np.linspace(0, 2 * np.pi, 121)[:-1]
        horizon_points = np.column_stack(
            (
                reference_radius * np.cos(theta),
                reference_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
        plotter.add_mesh(
            self._polyline(pyvista, horizon_points, closed=True),
            color=colors["horizon"],
            opacity=0.92,
            line_width=3,
            lighting=False,
        )
        plotter.add_mesh(
            rocket,
            color=colors["rocket"],
            smooth_shading=True,
            specular=0.18,
            specular_power=18,
        )
        axis_colors = (colors["body_x"], colors["body_y"], colors["body_z"])
        axis_labels = ("Body X — pitch", "Body Y — yaw", "Body Z — roll")
        for arrow, color, label in zip(
            body_arrows, axis_colors, axis_labels, strict=True
        ):
            plotter.add_mesh(
                arrow,
                color=color,
                lighting=False,
                label=label,
            )
        velocity_actor = plotter.add_mesh(
            velocity_arrow,
            color=colors["velocity"],
            lighting=False,
            label="Velocity direction",
        )
        velocity_actor.SetVisibility(bool(np.linalg.norm(velocity) > 1e-12))
        wind_actor = plotter.add_mesh(
            wind_arrow,
            color=colors["wind"],
            lighting=False,
            label="Wind velocity (toward)",
        )
        wind_actor.SetVisibility(bool(np.linalg.norm(wind) > 1e-12))

        center_of_mass_marker = None
        center_of_pressure_marker = None
        center_of_mass_connector = None
        center_of_pressure_connector = None
        if options["show_cp_cm"]:
            marker_radius = base_rocket.length * 0.035
            callout_offset = reference_radius * 0.30
            center_of_mass = self.flight.rocket.center_of_mass(start)
            center_of_pressure = self.flight.rocket.cp_position(
                self.flight.mach_number(start)
            )
            cm_station = rotation[:3, :3] @ np.array(
                [
                    0,
                    0,
                    self._rocket_axial_display_coordinate(
                        center_of_mass, base_rocket.length
                    ),
                ]
            )
            cp_station = rotation[:3, :3] @ np.array(
                [
                    0,
                    0,
                    self._rocket_axial_display_coordinate(
                        center_of_pressure, base_rocket.length
                    ),
                ]
            )
            cm_position = cm_station + rotation[:3, 1] * callout_offset
            cp_position = cp_station - rotation[:3, 1] * callout_offset
            center_of_mass_marker = pyvista.Sphere(
                radius=marker_radius, center=cm_position
            )
            center_of_pressure_marker = pyvista.Sphere(
                radius=marker_radius, center=cp_position
            )
            plotter.add_mesh(
                center_of_mass_marker,
                color=colors["center_of_mass"],
                lighting=False,
                label="Center of mass",
            )
            plotter.add_mesh(
                center_of_pressure_marker,
                color=colors["center_of_pressure"],
                lighting=False,
                label="Center of pressure",
            )
            center_of_mass_connector = self._polyline(
                pyvista, [cm_station, cm_position]
            )
            center_of_pressure_connector = self._polyline(
                pyvista, [cp_station, cp_position]
            )
            plotter.add_mesh(
                center_of_mass_connector,
                color=colors["center_of_mass"],
                line_width=3,
                lighting=False,
            )
            plotter.add_mesh(
                center_of_pressure_connector,
                color=colors["center_of_pressure"],
                line_width=3,
                lighting=False,
            )

        telemetry_position = "upper_left"
        legend_position = (
            "upper center"
            if options["show_kinematic_plots"] and options["show_attitude_plots"]
            else "upper right"
        )

        telemetry = plotter.add_text(
            self._rotation_telemetry(
                start, rotation, include_stability=options["show_cp_cm"]
            ),
            position=telemetry_position,
            font="courier",
            font_size=8,
            color=colors["panel_text"],
            shadow=False,
        )
        self._style_telemetry_actor(telemetry, colors)
        legend = plotter.add_legend(
            labels=[
                ["Body X — pitch", colors["body_x"]],
                ["Body Y — yaw", colors["body_y"]],
                ["Body Z — roll", colors["body_z"]],
                ["Velocity direction", colors["velocity"]],
                ["Wind velocity (toward)", colors["wind"]],
                *(
                    [
                        ["Center of mass", colors["center_of_mass"]],
                        ["Center of pressure", colors["center_of_pressure"]],
                    ]
                    if options["show_cp_cm"]
                    else []
                ),
            ],
            bcolor=colors["panel_background"],
            border=True,
            background_opacity=0.88,
            size=(0.145, 0.14 if options["show_cp_cm"] else 0.105),
            loc=legend_position,
        )
        self._style_legend_actor(legend, colors)
        kinematic_cursors = []
        attitude_cursors = []
        dual_chart_columns = bool(
            options["show_kinematic_plots"] and options["show_attitude_plots"]
        )
        if options["show_kinematic_plots"]:
            kinematic_cursors = self._add_animation_charts(
                pyvista,
                plotter,
                history_times,
                self._animation_kinematic_series(history_times),
                colors,
                compact=dual_chart_columns,
            )
        if options["show_attitude_plots"]:
            attitude_cursors = self._add_animation_charts(
                pyvista,
                plotter,
                history_times,
                self._animation_attitude_series(history_times),
                colors,
                attitude=True,
                compact=dual_chart_columns,
            )
        plotter.view_isometric()
        plotter.set_viewup((0, 0, 1))
        plotter.reset_camera()

        def update_frame(time_value):
            current_rotation = self._animation_transformation(time_value)
            position = self._animation_position(time_value)
            self._set_animation_background(
                plotter, background_palette, max(position[2], 0)
            )
            rocket.copy_from(base_rocket.transform(current_rotation, inplace=False))
            for index, arrow in enumerate(body_arrows):
                arrow.copy_from(
                    self._direction_arrow(
                        pyvista, current_rotation[:3, index], arrow_scale
                    )
                )
            current_velocity = self._animation_velocity(time_value)
            velocity_arrow.copy_from(
                self._direction_arrow(pyvista, current_velocity, arrow_scale * 0.92)
            )
            velocity_actor.SetVisibility(bool(np.linalg.norm(current_velocity) > 1e-12))
            current_wind = self._animation_wind(time_value)
            wind_arrow.copy_from(
                self._direction_arrow(pyvista, current_wind, arrow_scale * 0.82)
            )
            wind_actor.SetVisibility(bool(np.linalg.norm(current_wind) > 1e-12))
            if center_of_mass_marker is not None:
                center_of_mass = self.flight.rocket.center_of_mass(time_value)
                center_of_pressure = self.flight.rocket.cp_position(
                    self.flight.mach_number(time_value)
                )
                cm_station = current_rotation[:3, :3] @ np.array(
                    [
                        0,
                        0,
                        self._rocket_axial_display_coordinate(
                            center_of_mass, base_rocket.length
                        ),
                    ]
                )
                cp_station = current_rotation[:3, :3] @ np.array(
                    [
                        0,
                        0,
                        self._rocket_axial_display_coordinate(
                            center_of_pressure, base_rocket.length
                        ),
                    ]
                )
                cm_position = cm_station + current_rotation[:3, 1] * callout_offset
                cp_position = cp_station - current_rotation[:3, 1] * callout_offset
                center_of_mass_marker.copy_from(
                    pyvista.Sphere(radius=marker_radius, center=cm_position)
                )
                center_of_pressure_marker.copy_from(
                    pyvista.Sphere(radius=marker_radius, center=cp_position)
                )
                center_of_mass_connector.copy_from(
                    self._polyline(pyvista, [cm_station, cm_position])
                )
                center_of_pressure_connector.copy_from(
                    self._polyline(pyvista, [cp_station, cp_position])
                )
            telemetry.set_text(
                telemetry_position,
                self._rotation_telemetry(
                    time_value,
                    current_rotation,
                    include_stability=options["show_cp_cm"],
                ),
            )
            self._update_animation_chart_cursors(
                [*kinematic_cursors, *attitude_cursors], time_value
            )
            self._update_animation_camera(
                plotter,
                options["camera_mode"],
                np.zeros(3),
                current_rotation,
                reference_radius * 2.5,
                time_value,
                start,
                stop,
                options["camera_path"],
            )

        return self._run_animation(
            plotter,
            update_frame,
            start,
            stop,
            time_step,
            playback_speed,
            colors=colors,
            playback_controls=options["playback_controls"],
            backend=options["backend"],
            force_external=options["force_external"],
            export_file=options["export_file"],
            export_fps=options["export_fps"],
            transparent_background=options["transparent_background"],
        )

    def linear_kinematics_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Kinematics graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(414)
        ax1.plot(self.flight.vx[:, 0], self.flight.vx[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_title("Velocity X | Acceleration X")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity X (m/s)", color="#ff7f0e")
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(self.flight.ax[:, 0], self.flight.ax[:, 1], color="#1f77b4")
        ax1up.set_ylabel("Acceleration X (m/s²)", color="#1f77b4")
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(413)
        ax2.plot(self.flight.vy[:, 0], self.flight.vy[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, self.flight.t_final)
        ax2.set_title("Velocity Y | Acceleration Y")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity Y (m/s)", color="#ff7f0e")
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(self.flight.ay[:, 0], self.flight.ay[:, 1], color="#1f77b4")
        ax2up.set_ylabel("Acceleration Y (m/s²)", color="#1f77b4")
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(412)
        ax3.plot(self.flight.vz[:, 0], self.flight.vz[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, self.flight.t_final)
        ax3.set_title("Velocity Z | Acceleration Z")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity Z (m/s)", color="#ff7f0e")
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(self.flight.az[:, 0], self.flight.az[:, 1], color="#1f77b4")
        ax3up.set_ylabel("Acceleration Z (m/s²)", color="#1f77b4")
        ax3up.tick_params("y", colors="#1f77b4")

        ax4 = plt.subplot(411)
        ax4.plot(self.flight.speed[:, 0], self.flight.speed[:, 1], color="#ff7f0e")
        ax4.set_xlim(0, self.flight.t_final)
        ax4.set_title("Velocity Magnitude | Acceleration Magnitude")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity (m/s)", color="#ff7f0e")
        ax4.tick_params("y", colors="#ff7f0e")
        ax4.grid(True)

        ax4up = ax4.twinx()
        ax4up.plot(
            self.flight.acceleration[:, 0],
            self.flight.acceleration[:, 1],
            color="#1f77b4",
        )
        ax4up.set_ylabel("Acceleration (m/s²)", color="#1f77b4")
        ax4up.tick_params("y", colors="#1f77b4")

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def attitude_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Angular position graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        # Angular position plots
        _ = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.flight.e0[:, 0], self.flight.e0[:, 1], label="$e_0$")
        ax1.plot(self.flight.e1[:, 0], self.flight.e1[:, 1], label="$e_1$")
        ax1.plot(self.flight.e2[:, 0], self.flight.e2[:, 1], label="$e_2$")
        ax1.plot(self.flight.e3[:, 0], self.flight.e3[:, 1], label="$e_3$")
        ax1.set_xlim(0, self.first_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Euler Parameters")
        ax1.set_title("Euler Parameters")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(412)
        ax2.plot(self.flight.psi[:, 0], self.flight.psi[:, 1])
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ψ (°)")
        ax2.set_title("Euler Precession Angle")
        ax2.grid(True)

        ax3 = plt.subplot(413)
        ax3.plot(self.flight.theta[:, 0], self.flight.theta[:, 1], label="θ - Nutation")
        ax3.set_xlim(0, self.first_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("θ (°)")
        ax3.set_title("Euler Nutation Angle")
        ax3.grid(True)

        ax4 = plt.subplot(414)
        ax4.plot(self.flight.phi[:, 0], self.flight.phi[:, 1], label="φ - Spin")
        ax4.set_xlim(0, self.first_event_time)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("φ (°)")
        ax4.set_title("Euler Spin Angle")
        ax4.grid(True)

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def flight_path_angle_data(self, *, filename=None):
        """Prints out Flight path and Rocket Attitude angle graphs available
        about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(
            self.flight.path_angle[:, 0],
            self.flight.path_angle[:, 1],
            label="Flight Path Angle",
        )
        ax1.plot(
            self.flight.attitude_angle[:, 0],
            self.flight.attitude_angle[:, 1],
            label="Rocket Attitude Angle",
        )
        ax1.set_xlim(0, self.first_event_time)
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.set_title("Flight Path and Attitude Angle")

        ax2 = plt.subplot(212)
        ax2.plot(
            self.flight.lateral_attitude_angle[:, 0],
            self.flight.lateral_attitude_angle[:, 1],
        )
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Lateral Attitude Angle (°)")
        ax2.set_title("Lateral Attitude Angle")
        ax2.grid(True)

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def angular_kinematics_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Angular velocity and acceleration graphs available
        about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(311)
        ax1.plot(self.flight.w1[:, 0], self.flight.w1[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.first_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(r"Angular Velocity - ${\omega_1}$ (rad/s)", color="#ff7f0e")
        ax1.set_title(
            r"Angular Velocity ${\omega_1}$ | Angular Acceleration ${\alpha_1}$"
        )
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(self.flight.alpha1[:, 0], self.flight.alpha1[:, 1], color="#1f77b4")
        ax1up.set_ylabel(
            r"Angular Acceleration - ${\alpha_1}$ (rad/s²)", color="#1f77b4"
        )
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(312)
        ax2.plot(self.flight.w2[:, 0], self.flight.w2[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(r"Angular Velocity - ${\omega_2}$ (rad/s)", color="#ff7f0e")
        ax2.set_title(
            r"Angular Velocity ${\omega_2}$ | Angular Acceleration ${\alpha_2}$"
        )
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(self.flight.alpha2[:, 0], self.flight.alpha2[:, 1], color="#1f77b4")
        ax2up.set_ylabel(
            r"Angular Acceleration - ${\alpha_2}$ (rad/s²)", color="#1f77b4"
        )
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(313)
        ax3.plot(self.flight.w3[:, 0], self.flight.w3[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, self.first_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel(r"Angular Velocity - ${\omega_3}$ (rad/s)", color="#ff7f0e")
        ax3.set_title(
            r"Angular Velocity ${\omega_3}$ | Angular Acceleration ${\alpha_3}$"
        )
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(self.flight.alpha3[:, 0], self.flight.alpha3[:, 1], color="#1f77b4")
        ax3up.set_ylabel(
            r"Angular Acceleration - ${\alpha_3}$ (rad/s²)", color="#1f77b4"
        )
        ax3up.tick_params("y", colors="#1f77b4")

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def rail_buttons_bending_moments(self, *, filename=None):
        """Prints out Rail Buttons Bending Moments graphs.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        if len(self.flight.rocket.rail_buttons) == 0:
            logger.warning(
                "No rail buttons were defined. Skipping rail button bending moment plots."
            )
        elif self.flight.out_of_rail_time_index == 0:
            logger.warning(
                "No rail phase was found. Skipping rail button bending moment plots."
            )
        else:
            # Check if button_height is defined
            rail_buttons_tuple = self.flight.rocket.rail_buttons[0]
            if rail_buttons_tuple.component.button_height is None:
                logger.warning(
                    "Rail button height not defined. Skipping bending moment plots."
                )
            else:
                plt.figure(figsize=(9, 3))

                ax1 = plt.subplot(111)
                ax1.plot(
                    self.flight.rail_button1_bending_moment[
                        : self.flight.out_of_rail_time_index, 0
                    ],
                    self.flight.rail_button1_bending_moment[
                        : self.flight.out_of_rail_time_index, 1
                    ],
                    label="Upper Rail Button",
                )
                ax1.plot(
                    self.flight.rail_button2_bending_moment[
                        : self.flight.out_of_rail_time_index, 0
                    ],
                    self.flight.rail_button2_bending_moment[
                        : self.flight.out_of_rail_time_index, 1
                    ],
                    label="Lower Rail Button",
                )
                ax1.set_xlim(
                    0,
                    (
                        self.flight.out_of_rail_time
                        if self.flight.out_of_rail_time > 0
                        else self.flight.tFinal
                    ),
                )
                ax1.legend()
                ax1.grid(True)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Bending Moment (N·m)")
                ax1.set_title("Rail Button Bending Moments")

                show_or_save_plot(filename)

    def rail_buttons_forces(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Rail Buttons Forces graphs available about the Flight.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        if len(self.flight.rocket.rail_buttons) == 0:
            logger.warning("No rail buttons were defined. Skipping rail button plots.")
        elif self.flight.out_of_rail_time_index == 0:
            logger.warning("No rail phase was found. Skipping rail button plots.")
        else:
            plt.figure(figsize=(9, 6))

            ax1 = plt.subplot(211)
            ax1.plot(
                self.flight.rail_button1_normal_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button1_normal_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Upper Rail Button",
            )
            ax1.plot(
                self.flight.rail_button2_normal_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button2_normal_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Lower Rail Button",
            )
            ax1.set_xlim(
                0,
                (
                    self.flight.out_of_rail_time
                    if self.flight.out_of_rail_time > 0
                    else self.flight.tFinal
                ),
            )
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlabel(self.flight.rail_button1_normal_force.get_inputs()[0])
            ax1.set_ylabel(self.flight.rail_button1_normal_force.get_outputs()[0])
            ax1.set_title("Rail Buttons Normal Force")

            ax2 = plt.subplot(212)
            ax2.plot(
                self.flight.rail_button1_shear_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button1_shear_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Upper Rail Button",
            )
            ax2.plot(
                self.flight.rail_button2_shear_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button2_shear_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Lower Rail Button",
            )
            ax2.set_xlim(
                0,
                (
                    self.flight.out_of_rail_time
                    if self.flight.out_of_rail_time > 0
                    else self.flight.tFinal
                ),
            )
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlabel(self.flight.rail_button1_shear_force.__inputs__[0])
            ax2.set_ylabel(self.flight.rail_button1_shear_force.__outputs__[0])
            ax2.set_title("Rail Buttons Shear Force")

            plt.subplots_adjust(hspace=0.5)
            show_or_save_plot(filename)

    def aerodynamic_forces(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Forces and Moments graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(
            self.flight.aerodynamic_lift[: self.first_event_time_index, 0],
            self.flight.aerodynamic_lift[: self.first_event_time_index, 1],
            label="Resultant",
        )
        ax1.plot(
            self.flight.R1[: self.first_event_time_index, 0],
            self.flight.R1[: self.first_event_time_index, 1],
            label="R1",
        )
        ax1.plot(
            self.flight.R2[: self.first_event_time_index, 0],
            self.flight.R2[: self.first_event_time_index, 1],
            label="R2",
        )
        ax1.set_xlim(0, self.first_event_time)
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Lift Force (N)")
        ax1.set_title("Aerodynamic Lift Resultant Force")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            self.flight.aerodynamic_drag[: self.first_event_time_index, 0],
            self.flight.aerodynamic_drag[: self.first_event_time_index, 1],
        )
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Drag Force (N)")
        ax2.set_title("Aerodynamic Drag Force")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.aerodynamic_bending_moment[: self.first_event_time_index, 0],
            self.flight.aerodynamic_bending_moment[: self.first_event_time_index, 1],
            label="Resultant",
        )
        ax3.plot(
            self.flight.M1[: self.first_event_time_index, 0],
            self.flight.M1[: self.first_event_time_index, 1],
            label="M1",
        )
        ax3.plot(
            self.flight.M2[: self.first_event_time_index, 0],
            self.flight.M2[: self.first_event_time_index, 1],
            label="M2",
        )
        ax3.set_xlim(0, self.first_event_time)
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Bending Moment (N m)")
        ax3.set_title("Aerodynamic Bending Resultant Moment")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(
            self.flight.aerodynamic_spin_moment[: self.first_event_time_index, 0],
            self.flight.aerodynamic_spin_moment[: self.first_event_time_index, 1],
        )
        ax4.set_xlim(0, self.first_event_time)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Spin Moment (N m)")
        ax4.set_title("Aerodynamic Spin Moment")
        ax4.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def energy_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Energy components graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(411)
        ax1.plot(
            self.flight.kinetic_energy[:, 0],
            self.flight.kinetic_energy[:, 1],
            label="Kinetic Energy",
        )
        ax1.plot(
            self.flight.rotational_energy[:, 0],
            self.flight.rotational_energy[:, 1],
            label="Rotational Energy",
        )
        ax1.plot(
            self.flight.translational_energy[:, 0],
            self.flight.translational_energy[:, 1],
            label="Translational Energy",
        )
        ax1.set_xlim(
            0,
            (
                self.flight.apogee_time
                if self.flight.apogee_time != 0.0
                else self.flight.t_final
            ),
        )
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax1.set_title("Kinetic Energy Components")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Energy (J)")

        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            self.flight.total_energy[:, 0],
            self.flight.total_energy[:, 1],
            label="Total Energy",
        )
        ax2.plot(
            self.flight.kinetic_energy[:, 0],
            self.flight.kinetic_energy[:, 1],
            label="Kinetic Energy",
        )
        ax2.plot(
            self.flight.potential_energy[:, 0],
            self.flight.potential_energy[:, 1],
            label="Potential Energy",
        )
        ax2.set_xlim(
            0,
            (
                self.flight.apogee_time
                if self.flight.apogee_time != 0.0
                else self.flight.t_final
            ),
        )
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Total Mechanical Energy Components")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (J)")
        ax2.legend()
        ax2.grid()

        ax3 = plt.subplot(413)
        # Handle both array-based and callable-based Functions
        thrust_power = self.flight.thrust_power
        if not thrust_power.is_array_source():
            # For callable sources, discretize based on speed
            thrust_power = thrust_power.set_discrete_based_on_model(
                self.flight.speed, mutate_self=False
            )
        ax3.plot(
            thrust_power[:, 0],
            thrust_power[:, 1],
            label="|Thrust Power|",
        )
        ax3.set_xlim(0, self.flight.rocket.motor.burn_out_time)
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Thrust Absolute Power")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Power (W)")
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(414)
        # Handle both array-based and callable-based Functions
        drag_power = self.flight.drag_power
        if not drag_power.is_array_source():
            # For callable sources, discretize based on speed
            drag_power = drag_power.set_discrete_based_on_model(
                self.flight.speed, mutate_self=False
            )
        ax4.plot(
            drag_power[:, 0],
            -drag_power[:, 1],
            label="|Drag Power|",
        )
        ax4.set_xlim(
            0,
            (
                self.flight.apogee_time
                if self.flight.apogee_time != 0.0
                else self.flight.t_final
            ),
        )
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax4.set_title("Drag Absolute Power")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Power (W)")
        ax4.legend()
        ax4.grid()

        plt.subplots_adjust(hspace=1)
        show_or_save_plot(filename)

    @staticmethod
    def __signed_angle_ylim(angle_function, t_start, t_end, margin=5):
        """Return ``(ymin, ymax)`` limits for a signed angle plotted between
        ``t_start`` and ``t_end``. The range is based only on the samples inside
        that time window so the full (positive and negative) oscillation is
        visible, unlike a fixed floor at zero which would clip it.
        """
        data = angle_function[:, :]
        mask = (data[:, 0] >= t_start) & (data[:, 0] <= t_end)
        values = data[mask, 1] if np.any(mask) else data[:, 1]
        return values.min() - margin, values.max() + margin

    def fluid_mechanics_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out a summary of the Fluid Mechanics graphs available about
        the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 16))

        ax1 = plt.subplot(611)
        ax1.plot(self.flight.mach_number[:, 0], self.flight.mach_number[:, 1])
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()

        ax2 = plt.subplot(612)
        ax2.plot(self.flight.reynolds_number[:, 0], self.flight.reynolds_number[:, 1])
        ax2.set_xlim(0, self.flight.t_final)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()

        ax3 = plt.subplot(613)
        ax3.plot(
            self.flight.dynamic_pressure[:, 0],
            self.flight.dynamic_pressure[:, 1],
            label="Dynamic Pressure",
        )
        ax3.plot(
            self.flight.total_pressure[:, 0],
            self.flight.total_pressure[:, 1],
            label="Total Pressure",
        )
        ax3.plot(
            self.flight.pressure[:, 0],
            self.flight.pressure[:, 1],
            label="Static Pressure",
        )
        ax3.set_xlim(0, self.flight.t_final)
        ax3.legend()
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Total and Dynamic Pressure")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (Pa)")
        ax3.grid()

        ax4 = plt.subplot(614)
        ax4.plot(self.flight.angle_of_attack[:, 0], self.flight.angle_of_attack[:, 1])
        ax4.set_title("Angle of Attack")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle of Attack (°)")
        ax4.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        ax4.set_ylim(0, self.flight.angle_of_attack(self.flight.out_of_rail_time) + 15)
        ax4.grid()

        ax5 = plt.subplot(615)
        ax5.plot(
            self.flight.partial_angle_of_attack[:, 0],
            self.flight.partial_angle_of_attack[:, 1],
        )
        ax5.set_title("Partial Angle of Attack")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Partial Angle of Attack (°)")
        ax5.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        # Partial angle of attack is a signed angle oscillating around zero, so
        # scale to the data in the plotted window instead of flooring at 0
        # (which would clip the negative half of the oscillation).
        ax5.set_ylim(
            *self.__signed_angle_ylim(
                self.flight.partial_angle_of_attack,
                self.flight.out_of_rail_time,
                self.first_event_time,
            )
        )
        ax5.grid()

        ax6 = plt.subplot(616)
        ax6.plot(
            self.flight.angle_of_sideslip[:, 0], self.flight.angle_of_sideslip[:, 1]
        )
        ax6.set_title("Angle of Sideslip")
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Angle of Sideslip (°)")
        ax6.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        # Sideslip is also signed; keep the full oscillation visible.
        ax6.set_ylim(
            *self.__signed_angle_ylim(
                self.flight.angle_of_sideslip,
                self.flight.out_of_rail_time,
                self.first_event_time,
            )
        )
        ax6.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def stability_and_control_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out Rocket Stability and Control parameters graphs available
        about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(self.flight.stability_margin[:, 0], self.flight.stability_margin[:, 1])
        ax1.set_xlim(0, self.flight.stability_margin[:, 0][-1])
        ax1.set_title("Stability Margin")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Stability Margin (c)")
        ax1.set_xlim(0, self.first_event_time)
        ax1.axvline(
            x=self.flight.out_of_rail_time,
            color="r",
            linestyle="--",
            label="Out of Rail Time",
        )
        ax1.axvline(
            x=self.flight.rocket.motor.burn_out_time,
            color="g",
            linestyle="--",
            label="Burn Out Time",
        )

        ax1.axvline(
            x=self.flight.apogee_time,
            color="m",
            linestyle="--",
            label="Apogee Time",
        )
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(212)
        x_axis = np.arange(0, 5, 0.01)
        max_attitude = self.flight.attitude_frequency_response.max
        max_attitude = max_attitude if max_attitude != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.attitude_frequency_response(x_axis) / max_attitude,
            label="Attitude Angle",
        )
        max_omega1 = self.flight.omega1_frequency_response.max
        max_omega1 = max_omega1 if max_omega1 != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.omega1_frequency_response(x_axis) / max_omega1,
            label=r"$\omega_1$",
        )
        max_omega2 = self.flight.omega2_frequency_response.max
        max_omega2 = max_omega2 if max_omega2 != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.omega2_frequency_response(x_axis) / max_omega2,
            label=r"$\omega_2$",
        )
        max_omega3 = self.flight.omega3_frequency_response.max
        max_omega3 = max_omega3 if max_omega3 != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.omega3_frequency_response(x_axis) / max_omega3,
            label=r"$\omega_3$",
        )
        ax2.set_title("Frequency Response")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude Magnitude Normalized")
        ax2.set_xlim(0, 5)
        ax2.legend()
        ax2.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def pressure_rocket_altitude(self, *, filename=None):
        """Plots out pressure at rocket's altitude.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        # self.flight.pressure()

        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(self.flight.pressure[:, 0], self.flight.pressure[:, 1])
        ax1.set_title("Pressure at Rocket's Altitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pressure (Pa)")
        ax1.set_xlim(0, self.flight.t_final)
        ax1.grid()

        show_or_save_plot(filename)

    def pressure_signals(self):
        """Plots out all Parachute Trigger Pressure Signals.
        This function can be called also for plot pressure data for flights
        without Parachutes, in this case the Pressure Signals will be simply
        the pressure provided by the atmosphericModel, at Flight z positions.
        This means that no noise will be considered if at least one parachute
        has not been added.

        This function aims to help the engineer to visually check if there
        are anomalies with the Flight Simulation.

        Returns
        -------
        None
        """

        if len(self.flight.parachute_events) == 0:
            logger.warning("Rocket has no parachutes. No parachute plots available.")
            return

        for parachute in self.flight.rocket.parachutes:
            clean = parachute.clean_pressure_signal_function
            noisy = parachute.noisy_pressure_signal_function
            # Nothing was recorded (e.g. parachute never triggered)
            if not isinstance(clean.source, np.ndarray) or clean.source.ndim != 2:
                continue
            time_signal = clean.source[:, 0]

            plt.figure(figsize=(9, 4))
            plt.plot(
                time_signal, clean(time_signal), label="Without noise", linewidth=1.5
            )
            plt.plot(
                time_signal,
                noisy(time_signal),
                label="With noise",
                alpha=0.7,
                linewidth=0.8,
            )
            plt.title(f"Parachute trigger pressure signal: {parachute.name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Pressure (Pa)")
            plt.legend()
            plt.grid(True)
            show_or_save_plot()

    def all(self):  # pylint: disable=too-many-statements
        """Prints out all plots available about the Flight.

        Returns
        -------
        None
        """

        print("\n\nTrajectory 3d Plot\n")
        self.trajectory_3d()

        print("\n\nTrajectory Kinematic Plots\n")
        self.linear_kinematics_data()

        print("\n\nAngular Position Plots\n")
        self.flight_path_angle_data()

        print("\n\nPath, Attitude and Lateral Attitude Angle Plots\n")
        self.attitude_data()

        print("\n\nTrajectory Angular Velocity and Acceleration Plots\n")
        self.angular_kinematics_data()

        print("\n\nAerodynamic Forces Plots\n")
        self.aerodynamic_forces()

        print("\n\nRail Buttons Bending Moments Plots\n")
        self.rail_buttons_bending_moments()

        print("\n\nRail Buttons Forces Plots\n")
        self.rail_buttons_forces()

        print("\n\nTrajectory Energy Plots\n")
        self.energy_data()

        print("\n\nTrajectory Fluid Mechanics Plots\n")
        self.fluid_mechanics_data()

        print("\n\nTrajectory Stability and Control Plots\n")
        self.stability_and_control_data()

        print("\n\nRocket and Parachute Pressure Plots\n")
        self.pressure_rocket_altitude()
        self.pressure_signals()
