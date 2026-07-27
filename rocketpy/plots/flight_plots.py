# pylint: disable=too-many-lines
import logging
import os
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from functools import cached_property
from importlib import resources
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator, MultipleLocator

from ..mathutils.reference_frame import ReferenceFrame
from ..tools import import_optional_dependency
from .plot_helpers import show_or_save_plot


class _FlightPlots:
    """Class that holds plot methods for Flight class.

    Attributes
    ----------
    _FlightPlots.flight : Flight
        Flight object that will be used for the plots.

    _FlightPlots.first_parachute_event_time : float
        Time of first event.

    _FlightPlots.first_parachute_event_time_index : int
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

    _LOW_ALTITUDE_ALTITUDE_LIMIT = 80_000.0

    @cached_property
    def is_high_altitude_flight(self):
        """Whether ``all`` should add an Earth-centred full-flight view."""
        return (
            self.flight._maximum_geodetic_altitude >= self._LOW_ALTITUDE_ALTITUDE_LIMIT
        )

    @cached_property
    def low_altitude_end_time(self):
        """End of the launch-local presentation, capped at 80 km."""
        if not self.is_high_altitude_flight:
            return float(self.flight.t_final)
        crossing = self.flight._first_geodetic_altitude_crossing(
            self._LOW_ALTITUDE_ALTITUDE_LIMIT
        )
        return float(self.flight.time[0] if crossing is None else crossing)

    @property
    def has_low_altitude_segment(self):
        """Whether a non-empty launch-local segment is available."""
        has_launch_origin = (
            hasattr(self.flight, "_launch_site_fixed")
            or self.flight.reference_frame == ReferenceFrame.FLAT_EARTH
        )
        return has_launch_origin and self.low_altitude_end_time > float(
            self.flight.time[0]
        )

    @staticmethod
    def _clip_values(times, values, end_time):
        """Clip sampled values and interpolate a final row at ``end_time``."""
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        mask = times <= end_time
        clipped_times = times[mask]
        clipped_values = values[mask]
        if (
            len(clipped_times) > 0
            and clipped_times[-1] < end_time
            and end_time < times[-1]
        ):
            upper = int(np.searchsorted(times, end_time, side="right"))
            lower = upper - 1
            fraction = (end_time - times[lower]) / (times[upper] - times[lower])
            endpoint = values[lower] + fraction * (values[upper] - values[lower])
            clipped_times = np.append(clipped_times, end_time)
            clipped_values = np.concatenate(
                (clipped_values, np.asarray(endpoint)[None, ...]), axis=0
            )
        return clipped_times, clipped_values

    @cached_property
    def low_altitude_positions(self):
        """Launch-local position samples through the presentation ceiling."""
        return self._clip_values(
            self.flight.time,
            self.flight.position_local(),
            self.low_altitude_end_time,
        )

    def _low_altitude_position_at(self, time_value):
        """Interpolate launch-local position at an event time."""
        times, positions = self.low_altitude_positions
        return np.array(
            [
                np.interp(time_value, times, positions[:, component])
                for component in range(3)
            ]
        )

    def _low_altitude_series(self, source, end_time=None):
        """Return a two-dimensional Function/array clipped for local plots."""
        array = np.asarray(source[:, :], dtype=float)
        times, values = self._clip_values(
            array[:, 0],
            array[:, 1:],
            self.low_altitude_end_time if end_time is None else end_time,
        )
        return np.column_stack((times, values))

    @cached_property
    def first_parachute_event_time(self):
        """Time of the first flight event."""
        if len(self.flight.parachute_events) > 0:
            return min(
                float(self.flight.parachute_events[0][0]),
                self.low_altitude_end_time,
            )
        return self.low_altitude_end_time

    @cached_property
    def first_parachute_event_time_index(self):
        """Time index of the first flight event."""
        return int(
            np.searchsorted(
                self.flight.time,
                self.first_parachute_event_time,
                side="right",
            )
        )

    # Consistent red used for the rocket trajectory line across all plots.
    _TRAJECTORY_COLOR = "#e63946"

    # Burnout vertical/drop-line color -- kept separate from the orange dot marker so
    # the dashed line stays readable against typical orange and blue plot lines.
    _BURNOUT_LINE_COLOR = "#4a4a4a"
    _EVENT_LINE_WIDTH = 1.2

    # Shared color scheme — mirrors trajectory_3d exactly.
    _RESERVED_COLORS = {
        "Impact": "#ff1f1f",
        "Apogee": "#46daff",
        "Burnout": "#ff8121",
        "Out Of Rail": "#8b0000",
    }
    _COLOR_CYCLE = [
        "#7de07a",
        "#f781bf",
        "#a65628",
        "#ff7f00",
        "#ffff33",
        "#377eb8",
        "#984ea3",
        "#66c2a5",
    ]

    def _collect_events(self):
        """Return list of (time, label, marker, color, size) sorted by time."""
        events = []
        parachute_color_map = {}

        try:
            t_burn = self.flight.rocket.motor.burn_out_time
            events.append(
                (t_burn, "Burnout", "o", self._RESERVED_COLORS["Burnout"], 40)
            )
        except AttributeError:
            pass

        one_time_events = [
            ev
            for ev in getattr(self.flight, "events", [])
            if getattr(ev, "trigger_only_once", False)
        ]
        for ev in one_time_events:
            if not getattr(ev, "triggered_times", None):
                continue
            t_ev = ev.triggered_times[0]
            name = getattr(ev, "name", "") or ""
            if name == "Apogee":
                events.append(
                    (t_ev, "Apogee", "o", self._RESERVED_COLORS["Apogee"], 40)
                )
            elif name == "Out Of Rail":
                events.append(
                    (t_ev, "Out Of Rail", "^", self._RESERVED_COLORS["Out Of Rail"], 30)
                )
            elif "Parachute" in name:
                if name not in parachute_color_map:
                    parachute_color_map[name] = self._COLOR_CYCLE[
                        len(parachute_color_map) % len(self._COLOR_CYCLE)
                    ]
                events.append((t_ev, name, "s", parachute_color_map[name], 50))
            elif name == "Impact":
                events.append(
                    (t_ev, "Landing", "x", self._RESERVED_COLORS["Impact"], 60)
                )
            else:
                events.append((t_ev, name or None, "o", "#66c2a5", 40))

        events.sort(key=lambda e: e[0])
        return events

    def _sorted_legend(self, ax):
        """Show the legend with entries in event-time order.

        Non-event entries (e.g. "Trajectory", "Launch") sort before events.
        Uses ax.legend() first to capture all artists reliably (including 3D
        scatter collections that get_legend_handles_labels() may miss), then
        re-applies sorted ordering.
        """
        event_times = {ev[1]: ev[0] for ev in self._collect_events()}
        _, available_labels = ax.get_legend_handles_labels()
        if not available_labels:
            return
        leg = ax.legend()
        if leg is None:
            return
        handles = getattr(leg, "legend_handles", None) or getattr(
            leg, "legendHandles", []
        )
        labels = [t.get_text() for t in leg.get_texts()]
        combined = sorted(zip(labels, handles), key=lambda p: event_times.get(p[0], -1))
        if combined:
            labels_s, handles_s = zip(*combined)
            ax.legend(list(handles_s), list(labels_s))

    def _add_event_markers(self, ax, legend=True):
        """Add a vertical dashed line for each trigger-once event within xlim.

        Burnout uses a distinct dark color and thinner line for legibility against
        typical orange/blue plot lines. Out Of Rail and Landing are excluded.
        """
        xlim = ax.get_xlim()
        for t_ev, label, _marker, color, _size in self._collect_events():
            if label in ("Out Of Rail", "Landing"):
                continue
            if not xlim[0] <= t_ev <= xlim[1]:
                continue
            if label == "Burnout":
                ax.axvline(
                    x=t_ev,
                    color=self._BURNOUT_LINE_COLOR,
                    linestyle="--",
                    linewidth=self._EVENT_LINE_WIDTH,
                    alpha=1.0,
                    label=label,
                )
            else:
                ax.axvline(
                    x=t_ev,
                    color=color,
                    linestyle="--",
                    linewidth=self._EVENT_LINE_WIDTH,
                    alpha=1.0,
                    label=label,
                )
        if legend:
            self._sorted_legend(ax)

    def _add_event_markers_dropline(self, ax, legend=True, y_bottom=None, labels=None):
        """Event markers on the plotted curve with drop-lines from the y-axis bottom.

        For each trigger-once event (excluding Out Of Rail and Landing), draws an
        unlabelled dashed vertical line from the axis bottom to the curve value at
        that time, and a labelled scatter marker on the curve itself.  Apogee is
        drawn last so it renders on top of coincident markers.

        Parameters
        ----------
        y_bottom : float or None
            Y coordinate for the bottom of drop-lines.  When None (default) the
            bottom is derived from the minimum of the visible plotted data.
        labels : set or None
            If given, only events whose label is in this set are drawn.
        """
        lines = [ln for ln in ax.lines if len(ln.get_xdata()) > 1]
        if not lines:
            return
        xdata = np.asarray(lines[0].get_xdata(), dtype=float)
        ydata = np.asarray(lines[0].get_ydata(), dtype=float)
        xlim = ax.get_xlim()

        if y_bottom is None:
            vis = ydata[(xdata >= xlim[0]) & (xdata <= xlim[1])]
            if vis.size:
                span = max(float(vis.max() - vis.min()), 1e-6)
                y_bottom = float(vis.min()) - 0.05 * span
            else:
                y_bottom = ax.get_ylim()[0]
            ax.set_ylim(bottom=y_bottom)

        deferred_apogee = None
        for t_ev, label, marker, color, size in self._collect_events():
            if label in ("Out Of Rail", "Landing"):
                continue
            if labels is not None and label not in labels:
                continue
            if not xlim[0] <= t_ev <= xlim[1]:
                continue
            y_ev = float(np.interp(t_ev, xdata, ydata))
            line_color = self._BURNOUT_LINE_COLOR if label == "Burnout" else color
            lw = (
                self._EVENT_LINE_WIDTH if label == "Burnout" else self._EVENT_LINE_WIDTH
            )
            ax.vlines(
                t_ev,
                y_bottom,
                y_ev,
                colors=line_color,
                linestyles="--",
                linewidth=lw,
                alpha=1.0,
            )
            if label == "Apogee":
                deferred_apogee = (t_ev, y_ev, label, marker, color, size)
                continue
            s2d = size if marker == "s" else size * 0.5
            kw = {
                "marker": marker,
                "color": color,
                "s": s2d,
                "label": label,
                "zorder": 10,
            }
            if marker != "x":
                kw["edgecolors"] = "black"
                kw["linewidths"] = 0.8
            else:
                kw["linewidths"] = 1.5
            ax.scatter(t_ev, y_ev, **kw)
        if deferred_apogee is not None:
            t_ev, y_ev, label, marker, color, size = deferred_apogee
            kw = {
                "marker": marker,
                "color": color,
                "s": size * 0.5,
                "label": label,
                "zorder": 20,
            }
            kw["edgecolors"] = "black"
            kw["linewidths"] = 0.8
            ax.scatter(t_ev, y_ev, **kw)
        if legend:
            self._sorted_legend(ax)

    def trajectory_3d(
        self,
        *,
        filename=None,
        show_events=True,
        reserved_colors=None,
        event_palette=None,
    ):  # pylint: disable=too-many-statements
        """Plot a 3D graph of the trajectory

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        show_events : bool, optional
            Whether to display event markers (Impact, Apogee, Burnout, Parachutes, etc.).
            By default True.
        reserved_colors : dict | None, optional
            Custom color mapping for reserved events. Keys are event names
            ("Impact", "Apogee", "Burnout", "Out Of Rail"), values are hex colors.
            By default None, which uses a saturated palette.
        event_palette : list | None, optional
            Custom color palette for parachute and other event markers.
            By default None, which uses a saturated palette excluding reserved colors.

        Returns
        -------
        None
        """
        _, positions = self.low_altitude_positions
        east, north, up = positions.T
        max_z = max(up)
        min_z = min(up)
        max_x = max(east)
        min_x = min(east)
        max_y = max(north)
        min_y = min(north)
        min_xy = min(min_x, min_y)
        max_xy = max(max_x, max_y)

        # avoids errors when x_lim and y_lim are the same
        if abs(min_z - max_z) < 1e-5:
            max_z += 1
        if abs(min_xy - max_xy) < 1e-5:
            max_xy += 1

        _ = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111, projection="3d")
        ax1.plot(east, north, zs=min_z, zdir="z", linestyle="--")
        ax1.plot(
            east,
            up,
            zs=min_y,
            zdir="y",
            linestyle="--",
        )
        ax1.plot(
            north,
            up,
            zs=min_x,
            zdir="x",
            linestyle="--",
        )
        ax1.plot(
            east,
            north,
            up,
            color=self._TRAJECTORY_COLOR,
            linewidth="2",
            zorder=2,
        )
        ax1.scatter(
            east[0],
            north[0],
            up[0],
            s=20,
            facecolors="#ffd400",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Launch",
            depthshade=False,
        )
        # Plot single-trigger events (events configured with trigger_only_once)
        if show_events:  # pylint: disable=too-many-nested-blocks
            try:
                # Set defaults for color palettes if not provided
                if reserved_colors is None:
                    reserved_colors = {
                        "Impact": "#ff1f1f",
                        "Apogee": "#46daff",
                        "Burnout": "#ff8121",
                        "Out Of Rail": "#8b0000",
                    }

                if event_palette is None:
                    event_palette = [
                        "#7de07a",
                        "#f781bf",
                        "#a65628",
                        "#ff7f00",
                        "#ffff33",
                        "#377eb8",
                        "#984ea3",
                        "#66c2a5",
                    ]

                # Remove reserved colors from the palette to avoid duplication
                available_colors = [
                    c for c in event_palette if c not in reserved_colors.values()
                ]
                parachute_color_map = {}
                default_event_color = "#66c2a5"

                marker_size = 20

                # burnout marker (motor burn out time)
                try:
                    t_burn = self.flight.rocket.motor.burn_out_time
                    if t_burn > self.low_altitude_end_time:
                        raise AttributeError
                    x_b, y_b, z_b = self._low_altitude_position_at(t_burn)
                    ax1.scatter(
                        x_b,
                        y_b,
                        z_b,
                        color=reserved_colors.get("Burnout", "#ff6f00"),
                        s=marker_size,
                        label="Burnout",
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=5,
                    )
                except AttributeError:
                    # ignore if burn time unavailable
                    pass

                one_time_events = [
                    ev
                    for ev in getattr(self.flight, "events", [])
                    if getattr(ev, "trigger_only_once", False)
                ]
                # Collect apogee data and draw it last so it paints over any
                # coincident parachute square (3D painter's algorithm uses draw order).
                deferred_apogee = None
                for ev in one_time_events:
                    if getattr(ev, "triggered_times", None):
                        t_ev = ev.triggered_times[0]
                        if t_ev > self.low_altitude_end_time:
                            continue
                        x_ev, y_ev, z_ev = self._low_altitude_position_at(t_ev)
                        name = getattr(ev, "name", "") or ""
                        if name == "Apogee":
                            deferred_apogee = (x_ev, y_ev, z_ev)
                        elif name == "Out Of Rail":
                            pass
                        elif "Parachute" in name:
                            # assign a unique saturated color per parachute name
                            if name not in parachute_color_map:
                                parachute_color_map[name] = available_colors[
                                    len(parachute_color_map) % len(available_colors)
                                ]
                            ax1.scatter(
                                x_ev,
                                y_ev,
                                z_ev,
                                marker="s",
                                label=name,
                                s=45,
                                color=parachute_color_map[name],
                                edgecolors="black",
                                linewidths=0.8,
                                zorder=5,
                                depthshade=False,
                            )
                        elif name == "Impact":
                            ax1.scatter(
                                x_ev,
                                y_ev,
                                z_ev,
                                color=reserved_colors["Impact"],
                                marker="x",
                                label="Landing",
                                s=70,
                                linewidths=2.0,
                                zorder=5,
                            )
                        else:
                            ax1.scatter(
                                x_ev,
                                y_ev,
                                z_ev,
                                s=marker_size,
                                label=name or None,
                                color=default_event_color,
                                edgecolors="black",
                                linewidths=0.8,
                                zorder=5,
                            )

                # Draw apogee last so it renders on top of any coincident marker.
                # Use a very high zorder to win the depth-sort tiebreaker in 3D.
                if deferred_apogee is not None:
                    ax1.scatter(
                        *deferred_apogee,
                        color=reserved_colors["Apogee"],
                        label="Apogee",
                        s=marker_size,
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=100,
                        depthshade=False,
                    )

                self._sorted_legend(ax1)
            except Exception:  # pylint: disable=broad-exception-caught
                # plotting of events should never break the main plot
                pass
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Up from Launch Site (m)")
        ax1.set_title("Launch-Local Flight Trajectory")
        ax1.set_xlim(min_xy, max_xy)
        ax1.set_ylim(min_xy, max_xy)
        ax1.set_zlim(min_z, max_z)
        ax1.view_init(15, 45)
        ax1.set_box_aspect(None, zoom=0.95)  # 95% for label adjustment
        show_or_save_plot(filename)

    def orbit_3d(self, frame="gcrf", *, filename=None, backend="matplotlib"):
        """Plot an Earth-centered trajectory and reference ellipsoid.

        Parameters
        ----------
        frame : {"gcrf", "itrf"}, optional
            Frame used for the trajectory. Default is GCRF.
        filename : str, optional
            Save the figure instead of displaying it.
        backend : {"matplotlib", "plotly"}, optional
            Backend used for plotting. Default is "matplotlib".

        Returns
        -------
        plotly.graph_objects.Figure | None
            Plotly figure if backend is "plotly", else None.
        """
        if str(backend).lower() == "plotly":
            return self.plot_3d_trajectory(frame=frame, filename=filename)

        frame = self.flight.reference_frame.coerce(frame)
        positions = self.flight.position(frame)
        radius = self.flight.datum.semi_major_axis
        azimuth, polar = np.mgrid[0 : 2 * np.pi : 80j, 0 : np.pi : 40j]
        earth_x = radius * np.cos(azimuth) * np.sin(polar)
        earth_y = radius * np.sin(azimuth) * np.sin(polar)
        earth_z = radius * np.cos(polar)

        figure = plt.figure(figsize=(9, 9))
        axes = figure.add_subplot(111, projection="3d")
        axes.plot_surface(
            earth_x,
            earth_y,
            earth_z,
            color="#4C78A8",
            alpha=0.35,
            linewidth=0,
        )
        axes.plot(*positions.T, color="#E45756", linewidth=1.5)
        axes.scatter(*positions[0], color="black", label="Initial state")
        axes.scatter(*positions[-1], color="#E45756", marker="X", label="Final state")
        extent = max(radius, float(np.max(np.abs(positions)))) * 1.05
        axes.set_xlim(-extent, extent)
        axes.set_ylim(-extent, extent)
        axes.set_zlim(-extent, extent)
        axes.set_box_aspect((1, 1, 1))
        axes.set_xlabel(f"X {frame.value.upper()} (m)")
        axes.set_ylabel(f"Y {frame.value.upper()} (m)")
        axes.set_zlabel(f"Z {frame.value.upper()} (m)")
        axes.set_title("Earth-Centered Flight Trajectory")
        axes.legend()
        show_or_save_plot(filename)
        return None

    def earth_centered_state(self, *, filename=None):
        """Plot the full GCRF position, velocity and acceleration histories."""
        if self.flight.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError("Earth-centred state plots require a GCRF Flight.")
        time_values = self.flight.time
        position = self.flight.position(ReferenceFrame.GCRF)
        velocity = self.flight.velocity(ReferenceFrame.GCRF)
        acceleration = np.column_stack(
            (self.flight.ax[:, 1], self.flight.ay[:, 1], self.flight.az[:, 1])
        )
        figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        labels = ("X", "Y", "Z")
        for index, label in enumerate(labels):
            axes[0].plot(time_values, position[:, index], label=label)
            axes[1].plot(time_values, velocity[:, index], label=label)
            axes[2].plot(time_values, acceleration[:, index], label=label)
        axes[0].set_ylabel("Position (m)")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[2].set_ylabel("Acceleration (m/s²)")
        axes[2].set_xlabel("Time (s)")
        axes[0].set_title("GCRF Cartesian State")
        for axis in axes:
            axis.legend()
            axis.grid(True)
        figure.tight_layout()
        show_or_save_plot(filename)

    def geodetic_coordinates(self, *, filename=None):
        """Plot full-flight geodetic latitude, longitude and altitude.

        Longitude is rendered as points so an antimeridian crossing does not
        produce a misleading line across the plot.
        """
        if self.flight.reference_frame != ReferenceFrame.GCRF:
            raise AttributeError("Geodetic plots require a GCRF Flight.")
        time_values = self.flight.time
        figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(time_values, self.flight.latitude[:, 1])
        axes[0].set_ylabel("Latitude (deg)")
        axes[1].scatter(
            time_values,
            self.flight.longitude[:, 1],
            s=2,
            color="tab:green",
        )
        axes[1].set_ylabel("Longitude (deg)")
        axes[2].plot(
            time_values,
            self.flight.altitude[:, 1] / 1000,
            color="tab:red",
        )
        axes[2].set_ylabel("Altitude (km)")
        axes[2].set_xlabel("Time (s)")
        axes[0].set_title("Geodetic Coordinates")
        for axis in axes:
            axis.grid(True)
        figure.tight_layout()
        show_or_save_plot(filename)

    def plot_3d_trajectory(self, frame="gcrf", *, filename=None):
        """Plot 3D Earth and satellite trajectory using Plotly.

        Parameters
        ----------
        frame : {"gcrf", "itrf"}, optional
            Frame used for the trajectory. Default is GCRF.
        filename : str, optional
            Save HTML or static image output to filename.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated Plotly figure object.
        """
        import_optional_dependency("plotly")
        import plotly.graph_objects as go

        frame = self.flight.reference_frame.coerce(frame)
        positions = self.flight.position(frame)
        radius = self.flight.datum.semi_major_axis

        phi, theta = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
        x_earth = radius * np.cos(phi) * np.sin(theta)
        y_earth = radius * np.sin(phi) * np.sin(theta)
        z_earth = radius * np.cos(theta)

        fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=x_earth,
                y=y_earth,
                z=z_earth,
                colorscale="Blues",
                showscale=False,
                opacity=0.7,
                name="Earth",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines",
                line={"color": "red", "width": 4},
                name="Trajectory",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[positions[0, 2]],
                mode="markers",
                marker={"color": "green", "size": 8},
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[positions[-1, 0]],
                y=[positions[-1, 1]],
                z=[positions[-1, 2]],
                mode="markers",
                marker={"color": "orange", "size": 8},
                name="End",
            )
        )

        fig.update_layout(
            title="3D Earth and Satellite Trajectory",
            scene={
                "xaxis": {"title": f"X {frame.value.upper()} (m)"},
                "yaxis": {"title": f"Y {frame.value.upper()} (m)"},
                "zaxis": {"title": f"Z {frame.value.upper()} (m)"},
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 30},
        )

        if filename:
            if str(filename).endswith(".html"):
                fig.write_html(filename)
            else:
                fig.write_image(filename)
        elif "pytest" not in sys.modules:
            fig.show()

        return fig

    def _orbital_ground_track(self, *, filename=None):
        """Plot geodetic longitude versus latitude for a GCRF Flight."""
        figure, axes = plt.subplots(figsize=(10, 5))
        axes.plot(self.flight.longitude[:, 1], self.flight.latitude[:, 1])
        axes.scatter(
            self.flight.longitude[0, 1],
            self.flight.latitude[0, 1],
            color="black",
            label="Initial state",
        )
        axes.set_xlim(-180, 180)
        axes.set_ylim(-90, 90)
        axes.set_xlabel("Longitude (°)")
        axes.set_ylabel("Latitude (°)")
        axes.set_title("Ground Track")
        axes.grid(True)
        axes.legend()
        figure.tight_layout()
        show_or_save_plot(filename)
        return None

    def animate_orbit_3d(  # pylint: disable=too-many-statements,too-many-locals
        self,
        frame="gcrf",
        interval=50,
        *,
        start=None,
        stop=None,
        filename=None,
        backend="auto",
        **kwargs,
    ):
        """Animate an Earth-centered trajectory using PyVista or Plotly.

        Parameters
        ----------
        frame : {"gcrf", "itrf"}, optional
            Frame used for the trajectory. Default is GCRF.
        interval : int, optional
            Frame update interval in milliseconds. Default is 50.
        start : float | None, optional
            Animation start time in seconds. Default is None.
        stop : float | None, optional
            Animation end time in seconds. Default is None.
        filename : str | None, optional
            Output path for saving animation. Default is None.
        backend : {"auto", "pyvista", "plotly", "matplotlib", "none", "trame", "client"}, optional
            Visualization backend. Default is "auto".
        **kwargs : dict, optional
            Additional options passed to PyVista or animation renderer, such as:
            use_sun : bool, default True
                Enable Sun lighting.
            show_subsatellite_point : bool, default True
                Show sub-satellite projection point.
            show_skybox : bool, default True
                Display space skybox.
            force_external : bool, default False
                Render in an external window.
            smooth : bool, default False
                Enable trajectory smoothing.

        Returns
        -------
        pyvista.Plotter | plotly.graph_objects.Figure | FuncAnimation
            Animation or plotter object depending on selected backend.
        """
        backend_lower = str(backend).lower()
        if backend_lower == "matplotlib":
            return self._animate_orbit_3d_matplotlib(
                frame=frame,
                interval=interval,
                start=start,
                stop=stop,
                filename=filename,
            )

        if backend_lower == "plotly":
            return self._animate_orbit_3d_plotly(
                frame=frame,
                interval=interval,
                start=start,
                stop=stop,
                filename=filename,
            )

        import_optional_dependency("pyvista")
        import pyvista as pv
        from pyvista import examples

        from ..mathutils.orbital_elements import OrbitalElements

        frame_obj = self.flight.reference_frame.coerce(frame)
        ref_name = frame_obj.value.lower()

        force_external = kwargs.get("force_external", False)
        smooth = kwargs.get("smooth", False)
        use_sun = kwargs.get("use_sun", True)
        show_subsatellite_point = kwargs.get("show_subsatellite_point", True)
        show_skybox = kwargs.get("show_skybox", True)

        times = self.flight.time
        t_start = self.flight.t_initial if start is None else float(start)
        t_stop = self.flight.t_final if stop is None else float(stop)
        mask = (times >= t_start) & (times <= t_stop)
        if not np.any(mask):
            raise ValueError("The requested animation interval contains no states.")

        times = times[mask]
        positions = self.flight.position(frame_obj)[mask]
        velocities = self.flight.velocity(frame_obj)[mask]
        radius = self.flight.datum.semi_major_axis
        mu = self.flight.datum.gravitational_parameter
        start_epoch = self.flight.start_epoch

        def _in_notebook():
            try:
                from IPython import get_ipython

                shell = get_ipython()
                return shell is not None and "IPKernelApp" in shell.config
            except Exception:
                return False

        in_notebook = _in_notebook()
        if force_external:
            pv_backend = "none"
        elif backend_lower in ("auto", "pyvista"):
            pv_backend = "trame" if in_notebook else "none"
            if pv_backend == "trame" and smooth:
                pv_backend = "client"
        else:
            pv_backend = backend_lower

        is_external = pv_backend == "none"
        pv.set_jupyter_backend(pv_backend)
        pv.global_theme.allow_empty_mesh = True

        plotter = pv.Plotter(
            title="RocketPy 3D Orbit Visualizer",
            lighting="none",
            notebook=not is_external,
        )

        if show_skybox:
            try:
                cubemap = examples.download_cubemap_space_4k()
                plotter.add_actor(cubemap.to_skybox())
                plotter.set_environment_texture(cubemap, is_srgb=True)
            except Exception:
                pass

        if use_sun:
            sun_light = pv.Light(
                position=(1, 0, 0), focal_point=(0, 0, 0), positional=False
            )
            plotter.add_light(sun_light)
        else:
            plotter.add_light(pv.Light(light_type="headlight"))

        try:
            earth_mesh = examples.planets.load_earth(radius=radius)
            earth_mesh.rotate_z(180, inplace=True)
            earth_tex = examples.load_globe_texture()
            earth_actor = plotter.add_mesh(
                earth_mesh, texture=earth_tex, smooth_shading=True
            )
        except Exception:
            earth_mesh = pv.Sphere(
                radius=radius, theta_resolution=60, phi_resolution=60
            )
            earth_actor = plotter.add_mesh(
                earth_mesh, color="#4C78A8", opacity=0.8, smooth_shading=True
            )

        sat_point = pv.PolyData([positions[0]])
        trail = pv.PolyData()
        orbit = pv.PolyData()

        plotter.add_mesh(
            sat_point, color="cyan", point_size=12, render_points_as_spheres=True
        )
        plotter.add_mesh(trail, color="red", line_width=3)
        plotter.add_mesh(orbit, color="green", line_width=2, opacity=0.6)

        if show_subsatellite_point:
            subsat_point = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
            plotter.add_mesh(
                subsat_point,
                color="yellow",
                point_size=10,
                render_points_as_spheres=True,
            )

        stride = max(1, len(times) // 500)
        n_orbit_pts = 100 if is_external else 40

        def update_scene(time_val):
            idx = int(np.abs(times - time_val).argmin())
            r_current = positions[idx]
            sat_point.points = np.array([r_current])

            if idx > 1:
                pts = positions[: idx + 1]
                trail.points = pts
                trail.lines = np.hstack([[len(pts)], np.arange(len(pts))])
            else:
                trail.points = np.empty((0, 3))

            if show_subsatellite_point:
                r_norm = np.linalg.norm(r_current)
                if r_norm > 0:
                    subsat = (radius / r_norm) * r_current
                    subsat_point.points = np.array([subsat])

            current_epoch = start_epoch + float(times[idx])

            if use_sun:
                try:
                    sun_pos = self.flight.env.sun.position(current_epoch)
                    if ref_name == "itrf":
                        sun_pos, _, _ = self.flight.env.transform_kinematics(
                            sun_pos, [0, 0, 0], [0, 0, 0], "gcrf", "itrf", current_epoch
                        )
                    sun_light.position = sun_pos
                except Exception:
                    pass

            if ref_name == "gcrf":
                era = current_epoch.earth_rotation_angle
                cos_a = np.cos(-era)
                sin_a = np.sin(-era)
                u_rot = np.array(
                    [
                        [cos_a, sin_a, 0.0],
                        [-sin_a, cos_a, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                transform = np.eye(4)
                transform[:3, :3] = u_rot
                earth_actor.user_matrix = transform

            try:
                r_gcrf = positions[idx]
                v_gcrf = velocities[idx]
                if ref_name == "itrf":
                    r_gcrf, v_gcrf, _ = self.flight.env.transform_kinematics(
                        r_gcrf, v_gcrf, [0, 0, 0], "itrf", "gcrf", current_epoch
                    )
                elements = OrbitalElements.from_state(r_gcrf, v_gcrf, mu)
                nu = np.linspace(0, 2 * np.pi, n_orbit_pts)
                p = elements.semi_latus_rectum
                if p > 0 and elements.eccentricity < 1.0:
                    r_peri = (p / (1.0 + elements.eccentricity * np.cos(nu)))[
                        :, None
                    ] * np.column_stack([np.cos(nu), np.sin(nu), np.zeros_like(nu)])
                    cos_raan, sin_raan = np.cos(elements.raan), np.sin(elements.raan)
                    cos_arg, sin_arg = (
                        np.cos(elements.argument_of_periapsis),
                        np.sin(elements.argument_of_periapsis),
                    )
                    cos_inc, sin_inc = (
                        np.cos(elements.inclination),
                        np.sin(elements.inclination),
                    )
                    rot = np.array(
                        [
                            [
                                cos_raan * cos_arg - sin_raan * sin_arg * cos_inc,
                                -cos_raan * sin_arg - sin_raan * cos_arg * cos_inc,
                                sin_raan * sin_inc,
                            ],
                            [
                                sin_raan * cos_arg + cos_raan * sin_arg * cos_inc,
                                -sin_raan * sin_arg + cos_raan * cos_arg * cos_inc,
                                -cos_raan * sin_inc,
                            ],
                            [sin_arg * sin_inc, cos_arg * sin_inc, cos_inc],
                        ]
                    )
                    pts_gcrf = r_peri @ rot.T
                    if ref_name == "itrf":
                        pts_frame = np.array(
                            [
                                self.flight.env.transform_kinematics(
                                    pt,
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    "gcrf",
                                    "itrf",
                                    current_epoch,
                                )[0]
                                for pt in pts_gcrf
                            ]
                        )
                    else:
                        pts_frame = pts_gcrf
                    orbit.points = pts_frame
                    orbit.lines = np.hstack(
                        [[len(pts_frame)], np.arange(len(pts_frame))]
                    )
            except Exception:
                pass

            if is_external:
                plotter.render()

        slider = plotter.add_slider_widget(
            callback=lambda v: update_scene(v),
            rng=[times[0], times[-1]],
            value=times[0],
            title="Time (s)",
            pointa=(0.2, 0.05),
            pointb=(0.9, 0.05),
            color="gray",
        )

        if is_external:
            play_state = {"playing": False, "anim_idx": 0}

            def toggle_play(state):
                play_state["playing"] = state

            plotter.add_checkbox_button_widget(
                toggle_play, value=False, position=(20, 20), size=30
            )
            plotter.add_text("Play", position=(60, 25), font_size=10)

            def step(_):
                if play_state["playing"]:
                    play_state["anim_idx"] = (play_state["anim_idx"] + stride) % len(
                        times
                    )
                    t_val = times[play_state["anim_idx"]]
                    slider.GetRepresentation().SetValue(t_val)
                    update_scene(t_val)

            plotter.add_timer_event(
                max_steps=10_000_000, duration=interval, callback=step
            )

        update_scene(times[0])

        if filename:
            if str(filename).endswith(".gif"):
                plotter.open_gif(filename)
                for t_val in times[::stride]:
                    update_scene(t_val)
                    plotter.write_frame()
                plotter.close()
            elif "pytest" not in sys.modules:
                plotter.show(auto_close=True)
        elif "pytest" not in sys.modules:
            plotter.show(auto_close=True)

        return plotter

    def _animate_orbit_3d_plotly(self, frame, interval, start, stop, filename):
        import_optional_dependency("plotly")
        import plotly.graph_objects as go

        frame_obj = self.flight.reference_frame.coerce(frame)
        positions = self.flight.position(frame_obj)
        times = self.flight.time
        t_start = self.flight.t_initial if start is None else float(start)
        t_stop = self.flight.t_final if stop is None else float(stop)
        mask = (times >= t_start) & (times <= t_stop)
        if not np.any(mask):
            raise ValueError("The requested animation interval contains no states.")

        positions = positions[mask]
        times = times[mask]

        radius = self.flight.datum.semi_major_axis
        phi, theta = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
        x_earth = radius * np.cos(phi) * np.sin(theta)
        y_earth = radius * np.sin(phi) * np.sin(theta)
        z_earth = radius * np.cos(theta)

        earth_trace = go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale="Blues",
            showscale=False,
            opacity=0.7,
            name="Earth",
        )
        trail_trace = go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode="lines",
            line={"color": "red", "width": 4},
            name="Trajectory",
        )
        sat_trace = go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode="markers",
            marker={"color": "cyan", "size": 8},
            name="Satellite",
        )

        step = max(1, len(positions) // 150)
        indices = list(range(0, len(positions), step))
        if indices[-1] != len(positions) - 1:
            indices.append(len(positions) - 1)

        frames = []
        slider_steps = []
        for idx in indices:
            t_val = times[idx]
            pts = positions[: idx + 1]
            frame_data = [
                earth_trace,
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line={"color": "red", "width": 4},
                    name="Trajectory",
                ),
                go.Scatter3d(
                    x=[positions[idx, 0]],
                    y=[positions[idx, 1]],
                    z=[positions[idx, 2]],
                    mode="markers",
                    marker={"color": "cyan", "size": 8},
                    name="Satellite",
                ),
            ]
            frames.append(go.Frame(data=frame_data, name=f"t_{idx}"))
            slider_steps.append(
                {
                    "args": [
                        [f"t_{idx}"],
                        {
                            "frame": {"duration": interval, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": f"{t_val:.1f}s",
                    "method": "animate",
                }
            )

        fig = go.Figure(
            data=[earth_trace, trail_trace, sat_trace],
            frames=frames,
            layout=go.Layout(
                title="3D Earth and Satellite Trajectory Animation",
                scene={
                    "xaxis": {"title": f"X {frame_obj.value.upper()} (m)"},
                    "yaxis": {"title": f"Y {frame_obj.value.upper()} (m)"},
                    "zaxis": {"title": f"Z {frame_obj.value.upper()} (m)"},
                    "aspectmode": "data",
                },
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {
                                            "duration": interval,
                                            "redraw": True,
                                        },
                                        "fromcurrent": True,
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {
                                        "frame": {
                                            "duration": 0,
                                            "redraw": False,
                                        },
                                        "mode": "immediate",
                                    },
                                ],
                            },
                        ],
                    }
                ],
                sliders=[{"steps": slider_steps, "currentvalue": {"prefix": "Time: "}}],
            ),
        )

        if filename:
            if str(filename).endswith(".html"):
                fig.write_html(filename)
            else:
                fig.write_image(filename)
        elif "pytest" not in sys.modules:
            fig.show()

        return fig

    def _animate_orbit_3d_matplotlib(self, frame, interval, start, stop, filename):
        from matplotlib.animation import FuncAnimation

        frame_obj = self.flight.reference_frame.coerce(frame)
        positions = self.flight.position(frame_obj)
        times = self.flight.time
        t_start = self.flight.t_initial if start is None else float(start)
        t_stop = self.flight.t_final if stop is None else float(stop)
        mask = (times >= t_start) & (times <= t_stop)
        if not np.any(mask):
            raise ValueError("The requested animation interval contains no states.")
        positions = positions[mask]
        times = times[mask]
        radius = self.flight.datum.semi_major_axis
        azimuth, polar = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 25j]
        figure = plt.figure(figsize=(8, 8))
        axes = figure.add_subplot(111, projection="3d")
        axes.plot_surface(
            radius * np.cos(azimuth) * np.sin(polar),
            radius * np.sin(azimuth) * np.sin(polar),
            radius * np.cos(polar),
            color="#4C78A8",
            alpha=0.3,
            linewidth=0,
        )
        extent = max(radius, float(np.max(np.abs(positions)))) * 1.05
        axes.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
        axes.set_box_aspect((1, 1, 1))
        axes.set_title("Earth-Centered Flight Animation")
        (trail,) = axes.plot([], [], [], color="#E45756")
        (vehicle,) = axes.plot([], [], [], marker="o", color="black")
        timestamp = axes.text2D(0.02, 0.95, "", transform=axes.transAxes)

        def update(index):
            current = positions[: index + 1]
            trail.set_data_3d(current[:, 0], current[:, 1], current[:, 2])
            vehicle.set_data_3d(
                [positions[index, 0]], [positions[index, 1]], [positions[index, 2]]
            )
            timestamp.set_text(f"t = {times[index]:.1f} s")
            return trail, vehicle, timestamp

        animation = FuncAnimation(
            figure,
            update,
            frames=len(positions),
            interval=interval,
            blit=False,
        )
        if filename is not None:
            animation.save(filename)
            plt.close(figure)
        return animation

    def animate_orbit_2d(
        self,
        interval=50,
        *,
        start=None,
        stop=None,
        filename=None,
    ):
        """Animate a 2D orbital trajectory and osculating elements in Matplotlib.

        Parameters
        ----------
        interval : int, optional
            Frame update interval in milliseconds. Default is 50.
        start : float | None, optional
            Animation start time in seconds. Default is None.
        stop : float | None, optional
            Animation end time in seconds. Default is None.
        filename : str | None, optional
            Save animation to file (e.g. GIF or MP4). Default is None.

        Returns
        -------
        FuncAnimation
            Matplotlib animation object.
        """
        from matplotlib.animation import FuncAnimation

        from ..mathutils.orbital_elements import OrbitalElements

        positions = self.flight.position("gcrf")
        velocities = self.flight.velocity("gcrf")
        times = self.flight.time
        t_start = self.flight.t_initial if start is None else float(start)
        t_stop = self.flight.t_final if stop is None else float(stop)
        mask = (times >= t_start) & (times <= t_stop)
        if not np.any(mask):
            raise ValueError("The requested animation interval contains no states.")
        positions = positions[mask]
        velocities = velocities[mask]
        times = times[mask]

        radius = self.flight.datum.semi_major_axis
        mu = self.flight.datum.gravitational_parameter

        figure, axes = plt.subplots(figsize=(9, 9))
        max_r = max(radius, float(np.max(np.abs(positions[:, :2]))))
        axes.set_xlim(-1.5 * max_r, 1.5 * max_r)
        axes.set_ylim(-1.5 * max_r, 1.5 * max_r)
        axes.set_aspect("equal")
        axes.grid(True)
        axes.set_xlabel("X GCRF (m)")
        axes.set_ylabel("Y GCRF (m)")
        axes.set_title("Satellite Trajectory Animation 2D")

        earth = plt.Circle((0, 0), radius, color="blue", alpha=0.3, label="Earth")
        axes.add_patch(earth)

        (trajectory,) = axes.plot([], [], "r-", label="Trajectory")
        (satellite,) = axes.plot([], [], "ko", markersize=6, label="Satellite")
        (orbit_plot,) = axes.plot([], [], "g--", label="Instant Orbit")

        text = axes.text(
            0.02,
            0.95,
            "",
            transform=axes.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.5},
        )
        axes.legend(loc="upper right")

        def update(frame_idx):
            x, y = positions[frame_idx, 0], positions[frame_idx, 1]
            trajectory.set_data(
                positions[: frame_idx + 1, 0], positions[: frame_idx + 1, 1]
            )
            satellite.set_data([x], [y])

            try:
                elements = OrbitalElements.from_state(
                    positions[frame_idx], velocities[frame_idx], mu
                )
                nu = np.linspace(0, 2 * np.pi, 100)
                p = elements.semi_latus_rectum
                if p > 0 and elements.eccentricity < 1.0:
                    r_peri = (p / (1.0 + elements.eccentricity * np.cos(nu)))[
                        :, None
                    ] * np.column_stack([np.cos(nu), np.sin(nu), np.zeros_like(nu)])
                    cos_raan, sin_raan = np.cos(elements.raan), np.sin(elements.raan)
                    cos_arg, sin_arg = (
                        np.cos(elements.argument_of_periapsis),
                        np.sin(elements.argument_of_periapsis),
                    )
                    cos_inc, sin_inc = (
                        np.cos(elements.inclination),
                        np.sin(elements.inclination),
                    )
                    rot = np.array(
                        [
                            [
                                cos_raan * cos_arg - sin_raan * sin_arg * cos_inc,
                                -cos_raan * sin_arg - sin_raan * cos_arg * cos_inc,
                                sin_raan * sin_inc,
                            ],
                            [
                                sin_raan * cos_arg + cos_raan * sin_arg * cos_inc,
                                -sin_raan * sin_arg + cos_raan * cos_arg * cos_inc,
                                -cos_raan * sin_inc,
                            ],
                            [sin_arg * sin_inc, cos_arg * sin_inc, cos_inc],
                        ]
                    )
                    pts_gcrf = r_peri @ rot.T
                    orbit_plot.set_data(pts_gcrf[:, 0], pts_gcrf[:, 1])

                text.set_text(
                    f"Time: {times[frame_idx]:.1f} s\n"
                    f"Semi-latus rectum: {elements.semi_latus_rectum / 1e3:.1f} km\n"
                    f"Eccentricity: {elements.eccentricity:.4f}\n"
                    f"Inclination: {np.degrees(elements.inclination):.2f}°\n"
                    f"RAAN: {np.degrees(elements.raan):.2f}°\n"
                    f"Argument of periapsis: {np.degrees(elements.argument_of_periapsis):.2f}°"
                )
            except Exception:
                pass

            return trajectory, satellite, text, orbit_plot

        animation = FuncAnimation(
            figure,
            update,
            frames=len(positions),
            interval=interval,
            blit=False,
        )
        if filename is not None:
            animation.save(filename)
            plt.close(figure)
        return animation

    def orbital_elements(self, *, filename=None):
        """Plot the principal osculating orbital elements versus time."""
        orbit = self.flight.orbit
        figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
        series = (
            (orbit.semi_major_axis, "Semi-Major Axis (m)"),
            (orbit.eccentricity, "Eccentricity"),
            (orbit.inclination, "Inclination (rad)"),
            (orbit.raan, "RAAN (rad)"),
            (orbit.argument_of_periapsis, "Argument of Periapsis (rad)"),
            (orbit.true_anomaly, "True Anomaly (rad)"),
        )
        for axis, (function, label) in zip(axes.flat, series):
            axis.plot(function[:, 0], function[:, 1])
            axis.set_ylabel(label)
            axis.grid(True)
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        figure.suptitle("Osculating Orbital Elements")
        figure.tight_layout()
        show_or_save_plot(filename)

    def orbital_acceleration_components(self, *, filename=None):
        """Plot the magnitude of each modeled orbital acceleration."""
        figure, axes = plt.subplots(figsize=(10, 6))
        for name, function in self.flight.orbital_accelerations.items():
            values = function[:, 1:4]
            axes.plot(function[:, 0], np.linalg.norm(values, axis=1), label=name)
        axes.set_yscale("log")
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Acceleration Magnitude (m/s²)")
        axes.set_title("Orbital Acceleration Contributions")
        axes.grid(True)
        axes.legend()
        figure.tight_layout()
        show_or_save_plot(filename)

    def orbital_accelerations_rtn(self, *, filename=None):
        """Plot radial, transverse, and normal perturbation accelerations."""
        models = self.flight.orbital_accelerations_rtn
        figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        labels = ("Radial", "Transverse", "Normal")
        for name, function in models.items():
            for index, axis in enumerate(axes, start=1):
                axis.plot(function[:, 0], function[:, index], label=name)
        for axis, label in zip(axes, labels):
            axis.set_ylabel(f"{label} (m/s²)")
            axis.grid(True)
        axes[0].legend()
        axes[-1].set_xlabel("Time (s)")
        figure.suptitle("Orbital Accelerations in RTN")
        figure.tight_layout()
        show_or_save_plot(filename)

    def orbital_energy(self, *, filename=None):
        """Plot specific mechanical energy and angular-momentum magnitude."""
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        energy = self.flight.specific_orbital_energy
        momentum = self.flight.specific_angular_momentum.magnitude
        axes[0].plot(energy[:, 0], energy[:, 1])
        axes[0].set_ylabel("Specific Energy (J/kg)")
        axes[1].plot(momentum[:, 0], momentum[:, 1])
        axes[1].set_ylabel("Specific Angular Momentum (m²/s)")
        axes[1].set_xlabel("Time (s)")
        for axis in axes:
            axis.grid(True)
        figure.suptitle("Orbital Invariants")
        figure.tight_layout()
        show_or_save_plot(filename)

    def ground_station_pass(
        self,
        latitude,
        longitude,
        *,
        altitude=0.0,
        minimum_elevation=0.0,
        filename=None,
    ):
        """Plot elevation and range from a user-defined ground station."""
        observation = self.flight.ground_station_observation(
            latitude,
            longitude,
            altitude=altitude,
            minimum_elevation=minimum_elevation,
        )
        figure, range_axis = plt.subplots(figsize=(10, 6))
        elevation_axis = range_axis.twinx()
        range_axis.plot(
            observation["range"][:, 0],
            observation["range"][:, 1] / 1000.0,
            label="Range",
            color="#4C78A8",
        )
        elevation_axis.plot(
            observation["elevation"][:, 0],
            observation["elevation"][:, 1],
            label="Elevation",
            color="#E45756",
        )
        elevation_axis.axhline(minimum_elevation, color="black", linestyle="--")
        range_axis.set_xlabel("Time (s)")
        range_axis.set_ylabel("Range (km)")
        elevation_axis.set_ylabel("Elevation (°)")
        range_axis.grid(True)
        figure.suptitle("Ground Station Pass")
        figure.tight_layout()
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
        if getattr(self.flight.reference_frame, "value", None) == "gcrf":
            export_file = kwargs.pop("export_file", None)
            return self.animate_orbit_3d(
                interval=1000.0 * time_step / playback_speed,
                start=start,
                stop=stop,
                filename=export_file,
            )

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
        velocity = self.flight.velocity_local()
        acceleration = self.flight.acceleration_local()
        time_values, velocity = self._clip_values(
            self.flight.time, velocity, self.low_altitude_end_time
        )
        _, acceleration = self._clip_values(
            self.flight.time, acceleration, self.low_altitude_end_time
        )
        speed = np.linalg.norm(velocity, axis=1)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)

        plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(time_values, speed, color="#ff7f0e")
        ax1.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax1.set_title("Velocity Magnitude | Acceleration Magnitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (m/s)", color="#ff7f0e")
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)
        self._add_event_markers(ax1)

        ax1up = ax1.twinx()
        ax1up.plot(time_values, acceleration_magnitude, color="#1f77b4")
        ax1up.set_ylabel("Acceleration (m/s²)", color="#1f77b4")
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(412)
        ax2.plot(time_values, velocity[:, 2], color="#ff7f0e")
        ax2.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax2.set_title("Velocity Up | Acceleration Up")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity Up (m/s)", color="#ff7f0e")
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)
        self._add_event_markers(ax2, legend=False)

        ax2up = ax2.twinx()
        ax2up.plot(time_values, acceleration[:, 2], color="#1f77b4")
        ax2up.set_ylabel("Acceleration Up (m/s²)", color="#1f77b4")
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(413)
        ax3.plot(time_values, velocity[:, 1], color="#ff7f0e")
        ax3.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax3.set_title("Velocity North | Acceleration North")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity North (m/s)", color="#ff7f0e")
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)
        self._add_event_markers(ax3, legend=False)

        ax3up = ax3.twinx()
        ax3up.plot(time_values, acceleration[:, 1], color="#1f77b4")
        ax3up.set_ylabel("Acceleration North (m/s²)", color="#1f77b4")
        ax3up.tick_params("y", colors="#1f77b4")

        ax4 = plt.subplot(414)
        ax4.plot(time_values, velocity[:, 0], color="#ff7f0e")
        ax4.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax4.set_title("Velocity East | Acceleration East")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity East (m/s)", color="#ff7f0e")
        ax4.tick_params("y", colors="#ff7f0e")
        ax4.grid(True)
        self._add_event_markers(ax4, legend=False)

        ax4up = ax4.twinx()
        ax4up.plot(time_values, acceleration[:, 0], color="#1f77b4")
        ax4up.set_ylabel("Acceleration East (m/s²)", color="#1f77b4")
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

        quaternions = self.flight.attitude_local_quaternions()
        euler = self.flight.attitude_local_euler_angles()
        time_values, quaternions = self._clip_values(
            self.flight.time, quaternions, self.first_parachute_event_time
        )
        _, euler = self._clip_values(
            self.flight.time, euler, self.first_parachute_event_time
        )

        # Angular position plots
        _ = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        for index in range(4):
            ax1.plot(time_values, quaternions[:, index], label=f"$e_{index}$")
        ax1.set_xlim(0, self.first_parachute_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Euler Parameters")
        ax1.set_title("Euler Parameters")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(412)
        ax2.plot(time_values, euler[:, 0])
        ax2.set_xlim(0, self.first_parachute_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ψ (°)")
        ax2.set_title("Euler Precession Angle")
        ax2.grid(True)

        ax3 = plt.subplot(413)
        ax3.plot(time_values, euler[:, 1], label="θ - Nutation")
        ax3.set_xlim(0, self.first_parachute_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("θ (°)")
        ax3.set_title("Euler Nutation Angle")
        ax3.grid(True)

        ax4 = plt.subplot(414)
        ax4.plot(time_values, euler[:, 2], label="φ - Spin")
        ax4.set_xlim(0, self.first_parachute_event_time)
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
        local_velocity = self.flight.velocity_local()
        body_axis = self.flight.attitude_local()[:, :, 2]
        horizontal_velocity = np.linalg.norm(local_velocity[:, :2], axis=1)
        path_angle = np.degrees(np.arctan2(local_velocity[:, 2], horizontal_velocity))
        attitude_angle = np.degrees(
            np.arctan2(body_axis[:, 2], np.linalg.norm(body_axis[:, :2], axis=1))
        )
        lateral_angle = np.degrees(
            np.arctan2(
                body_axis[:, 0] * np.cos(np.radians(self.flight.heading))
                - body_axis[:, 1] * np.sin(np.radians(self.flight.heading)),
                np.sqrt(
                    body_axis[:, 2] ** 2
                    + (
                        body_axis[:, 0] * np.sin(np.radians(self.flight.heading))
                        + body_axis[:, 1] * np.cos(np.radians(self.flight.heading))
                    )
                    ** 2
                ),
            )
        )
        time_values, angles = self._clip_values(
            self.flight.time,
            np.column_stack((path_angle, attitude_angle, lateral_angle)),
            self.first_parachute_event_time,
        )

        plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(time_values, angles[:, 0], label="Flight Path Angle")
        ax1.plot(time_values, angles[:, 1], label="Rocket Attitude Angle")
        ax1.set_xlim(0, self.first_parachute_event_time)
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.set_title("Flight Path and Attitude Angle")

        ax2 = plt.subplot(212)
        ax2.plot(time_values, angles[:, 2])
        ax2.set_xlim(0, self.first_parachute_event_time)
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
        end_time = self.first_parachute_event_time
        angular_velocity = [
            self._low_altitude_series(function, end_time)
            for function in (self.flight.w1, self.flight.w2, self.flight.w3)
        ]
        angular_acceleration = [
            self._low_altitude_series(function, end_time)
            for function in (self.flight.alpha1, self.flight.alpha2, self.flight.alpha3)
        ]
        plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(311)
        ax1.plot(
            angular_velocity[0][:, 0],
            angular_velocity[0][:, 1],
            color="#ff7f0e",
        )
        ax1.set_xlim(0, self.first_parachute_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(r"Angular Velocity - ${\omega_1}$ (rad/s)", color="#ff7f0e")
        ax1.set_title(
            r"Angular Velocity ${\omega_1}$ | Angular Acceleration ${\alpha_1}$"
        )
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(
            angular_acceleration[0][:, 0],
            angular_acceleration[0][:, 1],
            color="#1f77b4",
        )
        ax1up.set_ylabel(
            r"Angular Acceleration - ${\alpha_1}$ (rad/s²)", color="#1f77b4"
        )
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(312)
        ax2.plot(
            angular_velocity[1][:, 0],
            angular_velocity[1][:, 1],
            color="#ff7f0e",
        )
        ax2.set_xlim(0, self.first_parachute_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(r"Angular Velocity - ${\omega_2}$ (rad/s)", color="#ff7f0e")
        ax2.set_title(
            r"Angular Velocity ${\omega_2}$ | Angular Acceleration ${\alpha_2}$"
        )
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(
            angular_acceleration[1][:, 0],
            angular_acceleration[1][:, 1],
            color="#1f77b4",
        )
        ax2up.set_ylabel(
            r"Angular Acceleration - ${\alpha_2}$ (rad/s²)", color="#1f77b4"
        )
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(313)
        ax3.plot(
            angular_velocity[2][:, 0],
            angular_velocity[2][:, 1],
            color="#ff7f0e",
        )
        ax3.set_xlim(0, self.first_parachute_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel(r"Angular Velocity - ${\omega_3}$ (rad/s)", color="#ff7f0e")
        ax3.set_title(
            r"Angular Velocity ${\omega_3}$ | Angular Acceleration ${\alpha_3}$"
        )
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(
            angular_acceleration[2][:, 0],
            angular_acceleration[2][:, 1],
            color="#1f77b4",
        )
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
            print(
                "No rail buttons were defined. Skipping rail button bending moment plots."
            )
        elif self.flight.out_of_rail_time_index == 0:
            print("No rail phase was found. Skipping rail button bending moment plots.")
        else:
            # Check if button_height is defined
            rail_buttons_tuple = self.flight.rocket.rail_buttons[0]
            if rail_buttons_tuple.component.button_height is None:
                print("Rail button height not defined. Skipping bending moment plots.")
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
                        else self.low_altitude_end_time
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
            print("No rail buttons were defined. Skipping rail button plots.")
        elif self.flight.out_of_rail_time_index == 0:
            print("No rail phase was found. Skipping rail button plots.")
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
                    else self.low_altitude_end_time
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
                    else self.low_altitude_end_time
                ),
            )
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlabel(self.flight.rail_button1_shear_force.get_inputs()[0])
            ax2.set_ylabel(self.flight.rail_button1_shear_force.get_outputs()[0])
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
        plt.figure(figsize=(9, 18))

        ax1 = plt.subplot(611)
        ax1.plot(
            self.flight.aerodynamic_normal_force[
                : self.first_parachute_event_time_index, 0
            ],
            self.flight.aerodynamic_normal_force[
                : self.first_parachute_event_time_index, 1
            ],
            label="Resultant",
        )
        ax1.plot(
            self.flight.R1[: self.first_parachute_event_time_index, 0],
            self.flight.R1[: self.first_parachute_event_time_index, 1],
            label="R1",
        )
        ax1.plot(
            self.flight.R2[: self.first_parachute_event_time_index, 0],
            self.flight.R2[: self.first_parachute_event_time_index, 1],
            label="R2",
        )
        ax1.set_xlim(0, self.first_parachute_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normal Force (N)")
        ax1.set_title("Aerodynamic Normal Force (Body Frame)")
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(612)
        ax2.plot(
            self.flight.aerodynamic_axial_force[
                : self.first_parachute_event_time_index, 0
            ],
            self.flight.aerodynamic_axial_force[
                : self.first_parachute_event_time_index, 1
            ],
        )
        ax2.set_xlim(0, self.first_parachute_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Axial Force (N)")
        ax2.set_title("Aerodynamic Axial Force (Body Frame)")
        ax2.grid()

        ax3 = plt.subplot(613)
        ax3.plot(
            self.flight.aerodynamic_lift[: self.first_parachute_event_time_index, 0],
            self.flight.aerodynamic_lift[: self.first_parachute_event_time_index, 1],
        )
        ax3.set_xlim(0, self.first_parachute_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Lift Force (N)")
        ax3.set_title("Aerodynamic Lift Force (Aerodynamic Frame)")
        ax3.grid()

        ax4 = plt.subplot(614)
        ax4.plot(
            self.flight.aerodynamic_drag[: self.first_parachute_event_time_index, 0],
            self.flight.aerodynamic_drag[: self.first_parachute_event_time_index, 1],
        )
        ax4.set_xlim(0, self.first_parachute_event_time)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Drag Force (N)")
        ax4.set_title("Aerodynamic Drag Force (Aerodynamic Frame)")
        ax4.grid()

        ax5 = plt.subplot(615)
        ax5.plot(
            self.flight.aerodynamic_bending_moment[
                : self.first_parachute_event_time_index, 0
            ],
            self.flight.aerodynamic_bending_moment[
                : self.first_parachute_event_time_index, 1
            ],
            label="Resultant",
        )
        ax5.plot(
            self.flight.M1[: self.first_parachute_event_time_index, 0],
            self.flight.M1[: self.first_parachute_event_time_index, 1],
            label="M1",
        )
        ax5.plot(
            self.flight.M2[: self.first_parachute_event_time_index, 0],
            self.flight.M2[: self.first_parachute_event_time_index, 1],
            label="M2",
        )
        ax5.set_xlim(0, self.first_parachute_event_time)
        ax5.legend()
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Bending Moment (N m)")
        ax5.set_title("Aerodynamic Bending Resultant Moment")
        ax5.grid()

        ax6 = plt.subplot(616)
        ax6.plot(
            self.flight.aerodynamic_spin_moment[
                : self.first_parachute_event_time_index, 0
            ],
            self.flight.aerodynamic_spin_moment[
                : self.first_parachute_event_time_index, 1
            ],
        )
        ax6.set_xlim(0, self.first_parachute_event_time)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Spin Moment (N m)")
        ax6.set_title("Aerodynamic Spin Moment")
        ax6.grid()

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

        kinetic_energy = self._low_altitude_series(self.flight.kinetic_energy)
        rotational_energy = self._low_altitude_series(self.flight.rotational_energy)
        translational_energy = self._low_altitude_series(
            self.flight.translational_energy
        )
        total_energy = self._low_altitude_series(self.flight.total_energy)
        potential_energy = self._low_altitude_series(self.flight.potential_energy)

        plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(411)
        ax1.plot(
            kinetic_energy[:, 0],
            kinetic_energy[:, 1],
            label="Kinetic Energy",
        )
        ax1.plot(
            rotational_energy[:, 0],
            rotational_energy[:, 1],
            label="Rotational Energy",
        )
        ax1.plot(
            translational_energy[:, 0],
            translational_energy[:, 1],
            label="Translational Energy",
        )
        ax1.set_xlim(
            self.flight.time[0],
            min(
                self.low_altitude_end_time,
                (
                    self.flight.apogee_time
                    if self.flight.apogee_time != 0.0
                    else self.low_altitude_end_time
                ),
            ),
        )
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax1.set_title("Kinetic Energy Components")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Energy (J)")

        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            total_energy[:, 0],
            total_energy[:, 1],
            label="Total Energy",
        )
        ax2.plot(
            kinetic_energy[:, 0],
            kinetic_energy[:, 1],
            label="Kinetic Energy",
        )
        ax2.plot(
            potential_energy[:, 0],
            potential_energy[:, 1],
            label="Potential Energy",
        )
        ax2.set_xlim(
            self.flight.time[0],
            min(
                self.low_altitude_end_time,
                (
                    self.flight.apogee_time
                    if self.flight.apogee_time != 0.0
                    else self.low_altitude_end_time
                ),
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
        if callable(thrust_power.source):
            # For callable sources, discretize based on speed
            thrust_power = thrust_power.set_discrete_based_on_model(
                self.flight.speed, mutate_self=False
            )
        thrust_power = self._low_altitude_series(
            thrust_power,
            min(self.flight.rocket.motor.burn_out_time, self.low_altitude_end_time),
        )
        ax3.plot(
            thrust_power[:, 0],
            thrust_power[:, 1],
            label="|Thrust Power|",
        )
        ax3.set_xlim(
            self.flight.time[0],
            min(self.flight.rocket.motor.burn_out_time, self.low_altitude_end_time),
        )
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Thrust Absolute Power")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Power (W)")
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(414)
        # Handle both array-based and callable-based Functions
        drag_power = self.flight.drag_power
        if callable(drag_power.source):
            # For callable sources, discretize based on speed
            drag_power = drag_power.set_discrete_based_on_model(
                self.flight.speed, mutate_self=False
            )
        drag_power = self._low_altitude_series(drag_power)
        ax4.plot(
            drag_power[:, 0],
            -drag_power[:, 1],
            label="|Drag Power|",
        )
        ax4.set_xlim(
            self.flight.time[0],
            min(
                self.low_altitude_end_time,
                (
                    self.flight.apogee_time
                    if self.flight.apogee_time != 0.0
                    else self.low_altitude_end_time
                ),
            ),
        )
        ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax4.set_title("Drag Absolute Power")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Power (W)")
        ax4.legend()
        ax4.grid()

        plt.subplots_adjust(hspace=1)
        show_or_save_plot(filename)

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
        mach = self._low_altitude_series(self.flight.mach_number)
        reynolds = self._low_altitude_series(self.flight.reynolds_number)
        dynamic_pressure = self._low_altitude_series(self.flight.dynamic_pressure)
        total_pressure = self._low_altitude_series(self.flight.total_pressure)
        pressure = self._low_altitude_series(self.flight.pressure)

        plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(311)
        ax1.plot(mach[:, 0], mach[:, 1])
        ax1.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()
        self._add_event_markers(ax1)

        ax2 = plt.subplot(312)
        ax2.plot(reynolds[:, 0], reynolds[:, 1])
        ax2.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()
        self._add_event_markers(ax2, legend=False)

        ax3 = plt.subplot(313)
        ax3.plot(
            dynamic_pressure[:, 0],
            dynamic_pressure[:, 1],
            label="Dynamic Pressure",
        )
        ax3.plot(
            total_pressure[:, 0],
            total_pressure[:, 1],
            label="Total Pressure",
        )
        ax3.plot(
            pressure[:, 0],
            pressure[:, 1],
            label="Static Pressure",
        )
        ax3.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax3.legend()
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Total and Dynamic Pressure")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (Pa)")
        ax3.grid()
        self._add_event_markers(ax3, legend=False)

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

        asymmetric = not self.flight.rocket.is_axisymmetric
        stability_margin = self._low_altitude_series(
            self.flight.stability_margin,
            self.first_parachute_event_time,
        )
        ax1 = plt.subplot(211)
        ax1.plot(
            stability_margin[:, 0],
            stability_margin[:, 1],
            label="Linear pitch" if asymmetric else "Linear (aerodynamic center)",
        )
        if asymmetric:
            stability_margin_yaw = self._low_altitude_series(
                self.flight.stability_margin_yaw,
                self.first_parachute_event_time,
            )
            ax1.plot(
                stability_margin_yaw[:, 0],
                stability_margin_yaw[:, 1],
                label="Linear yaw",
            )
        ax1.set_title("Stability Margin")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Stability Margin (c)")
        ax1.set_xlim(0, self.first_parachute_event_time)
        ax1.legend()
        ax1.grid()
        self._add_event_markers_dropline(ax1, labels={"Burnout"})

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

    def dynamic_stability_data(self, *, filename=None):
        """Plots the rocket's dynamic-stability quantities over the flight: the
        pitch (and, for non-axisymmetric rockets, yaw) natural frequency and
        damping ratio of the linearized attitude oscillation.

        The roll rate is overlaid on the natural-frequency plot (as a frequency).
        Roll is neutrally stable -- it has no restoring moment and therefore no
        natural frequency of its own -- but **roll resonance** ("roll lock-in")
        occurs where the roll rate crosses the pitch/yaw natural frequency, the
        roll-pitch/yaw coupling driving the attitude oscillation. Those crossings
        are the points to watch.

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
        asymmetric = not self.flight.rocket.is_axisymmetric
        upper = self.first_parachute_event_time

        plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        freq = self._low_altitude_series(self.flight.pitch_natural_frequency, upper)
        ax1.plot(freq[:, 0], freq[:, 1] / (2 * np.pi), label="Pitch natural freq.")
        if asymmetric:
            yaw_freq = self._low_altitude_series(
                self.flight.yaw_natural_frequency, upper
            )
            ax1.plot(
                yaw_freq[:, 0],
                yaw_freq[:, 1] / (2 * np.pi),
                "--",
                label="Yaw natural freq.",
            )
        # Roll rate as a frequency: where it crosses the natural frequency the
        # rocket is in roll resonance (roll-pitch/yaw coupling).
        roll_rate = self._low_altitude_series(self.flight.w3, upper)
        ax1.plot(
            roll_rate[:, 0],
            np.abs(roll_rate[:, 1]) / (2 * np.pi),
            ":",
            color="tab:red",
            label="Roll rate (resonance if crossing)",
        )
        ax1.set_title("Natural Frequency & Roll Rate")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlim(0, upper)
        ax1.legend()
        ax1.grid()
        self._add_event_markers_dropline(ax1, labels={"Burnout"})

        ax2 = plt.subplot(212)
        ratio = self._low_altitude_series(self.flight.pitch_damping_ratio, upper)
        ax2.plot(ratio[:, 0], ratio[:, 1], label="Pitch")
        if asymmetric:
            yaw_ratio = self._low_altitude_series(self.flight.yaw_damping_ratio, upper)
            ax2.plot(yaw_ratio[:, 0], yaw_ratio[:, 1], "--", label="Yaw")
        ax2.axhline(1.0, color="gray", linestyle=":", label="Critical (ζ=1)")
        ax2.set_title("Damping Ratio")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Damping Ratio (ζ)")
        ax2.set_xlim(0, upper)
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
        """Deprecated. Pressure signal plots have been removed.

        Use a Sensor (e.g. a Barometer) with built-in noise and access its
        recorded measurements via ``flight.sensor_data`` instead.
        """
        warnings.warn(
            "pressure_signals() is deprecated and will be removed in v1.13. "
            "Use a Barometer Sensor with built-in noise and access its data "
            "via flight.sensor_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def sensor_data(self, *, filename=None):
        """Plots the measured data of every sensor attached to the flight.

        Delegates to each sensor's own ``plots`` object. A sensor added to the
        rocket multiple times appears once in the output.

        Parameters
        ----------
        filename : str | None, optional
            The path the plots should be saved to. By default None, in which
            case the plots will be shown instead of saved. When given, a
            per-sensor suffix is appended to keep one file per sensor.
        """
        if not self.flight.sensors:
            print("No sensors were registered in this flight.")
            return

        seen = []
        for sensor in self.flight.sensors:
            if sensor in seen:  # a multiply-added sensor is listed more than once
                continue
            seen.append(sensor)
            measured_data = self.flight.sensor_data[sensor]
            print(f"\n\n{sensor.name} Sensor Data\n")
            if filename is None:
                sensor.plots.all(data=measured_data)
            else:
                path = Path(filename)
                sensor_filename = str(
                    path.with_name(f"{path.stem}_{sensor.name}{path.suffix}")
                )
                sensor.plots.time_series(filename=sensor_filename, data=measured_data)

    def altitude_data(self, *, filename=None):
        """Plots altitude AGL vs time with event markers.

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
        plt.figure(figsize=(9, 4))

        if self.flight.reference_frame == ReferenceFrame.GCRF:
            source = self.flight.altitude[:, :]
            title = "Geodetic Height"
            ylabel = "Geodetic Height (m)"
        else:
            source = np.column_stack(
                (
                    self.flight.z[:, 0],
                    self.flight.z[:, 1] - self.flight.env.elevation,
                )
            )
            title = "Altitude Above Ground Level"
            ylabel = "Altitude AGL (m)"
        z_times, z_agl = self._clip_values(
            source[:, 0], source[:, 1], self.low_altitude_end_time
        )

        ax1 = plt.subplot(111)
        ax1.plot(z_times, z_agl, color=self._TRAJECTORY_COLOR)
        ax1.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax1.set_ylim(bottom=min(0, float(np.min(z_agl))))
        ax1.set_title(title)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(ylabel)
        ax1.grid(True)

        # Event markers: dot on the curve + dashed line from y=0 (line not in legend).
        # Apogee is drawn last so it renders on top of coincident markers.
        xlim = ax1.get_xlim()
        deferred_apogee = None
        for t_ev, label, marker, color, size in self._collect_events():
            if label in ("Out Of Rail", "Landing"):
                continue
            if not xlim[0] <= t_ev <= xlim[1]:
                continue
            alt_ev = float(np.interp(t_ev, z_times, z_agl))
            line_color = self._BURNOUT_LINE_COLOR if label == "Burnout" else color
            lw = (
                self._EVENT_LINE_WIDTH if label == "Burnout" else self._EVENT_LINE_WIDTH
            )
            ax1.vlines(
                t_ev,
                0,
                alt_ev,
                colors=line_color,
                linestyles="--",
                linewidth=lw,
                alpha=1.0,
            )
            if label == "Apogee":
                deferred_apogee = (t_ev, label, marker, color, size, alt_ev)
                continue
            s2d = size if marker == "s" else size * 0.5
            kw = {
                "marker": marker,
                "color": color,
                "s": s2d,
                "label": label,
                "zorder": 10,
            }
            if marker != "x":
                kw["edgecolors"] = "black"
                kw["linewidths"] = 0.8
            else:
                kw["linewidths"] = 1.5
            ax1.scatter(t_ev, alt_ev, **kw)
        if deferred_apogee is not None:
            t_ev, label, marker, color, size, alt_ev = deferred_apogee
            ax1.scatter(
                t_ev,
                alt_ev,
                marker=marker,
                color=color,
                s=size * 0.5,
                label=label,
                zorder=20,
                edgecolors="black",
                linewidths=0.8,
            )
        self._sorted_legend(ax1)

        plt.tight_layout()
        show_or_save_plot(filename)

    def ground_track(self, *, filename=None, local=False):
        """Plots the 2D ground track (East vs North displacement from launch).

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
        if self.flight.reference_frame == ReferenceFrame.GCRF and not local:
            return self._orbital_ground_track(filename=filename)

        plt.figure(figsize=(6, 6))
        _, positions = self.low_altitude_positions
        east, north = positions[:, 0], positions[:, 1]

        ax1 = plt.subplot(111)
        ax1.plot(
            east,
            north,
            color=self._TRAJECTORY_COLOR,
            label="_nolegend_",
            zorder=1,
        )
        # Launch point (t=0 is not a trigger-once event, so add it explicitly)
        ax1.scatter(
            [east[0]],
            [north[0]],
            color="#ffd400",
            edgecolors="black",
            linewidths=1.2,
            s=40,
            zorder=5,
            label="Launch",
        )
        # Events at their ground-track position (Out Of Rail omitted; Apogee drawn last)
        deferred_apogee = None
        for t_ev, label, marker, color, size in self._collect_events():
            if label == "Out Of Rail" or t_ev > self.low_altitude_end_time:
                continue
            if label == "Apogee":
                deferred_apogee = (t_ev, label, marker, color, size)
                continue
            if marker == "s":
                s2d = size
            elif marker == "x":
                s2d = size * 0.9
            else:
                s2d = size * 0.5
            kw = {
                "marker": marker,
                "color": color,
                "s": s2d,
                "label": label,
                "zorder": 10,
            }
            if marker != "x":
                kw["edgecolors"] = "black"
                kw["linewidths"] = 0.8
            else:
                kw["linewidths"] = 1.5
            event_position = self._low_altitude_position_at(t_ev)
            ax1.scatter([event_position[0]], [event_position[1]], **kw)
        if deferred_apogee is not None:
            t_ev, label, marker, color, size = deferred_apogee
            event_position = self._low_altitude_position_at(t_ev)
            ax1.scatter(
                [event_position[0]],
                [event_position[1]],
                marker=marker,
                color=color,
                s=size * 0.5,
                label=label,
                edgecolors="black",
                linewidths=0.8,
                zorder=20,
            )
        ax1.set_title("Launch-Local Ground Track")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        self._sorted_legend(ax1)
        ax1.grid(True)
        # Compute symmetric equal-range limits so the axes fill the square figure
        x_data = east
        y_data = north
        x_center = (float(x_data.max()) + float(x_data.min())) / 2
        y_center = (float(y_data.max()) + float(y_data.min())) / 2
        half = (
            max(
                float(x_data.max()) - float(x_data.min()),
                float(y_data.max()) - float(y_data.min()),
            )
            / 2
            * 1.1
            + 1
        )
        ax1.set_xlim(x_center - half, x_center + half)
        ax1.set_ylim(y_center - half, y_center + half)
        ax1.set_aspect("equal", adjustable="box")
        # Use the same tick spacing on both axes so the square grid is uniform
        tick_values = MaxNLocator(nbins=6).tick_values(-half, half)
        step = tick_values[1] - tick_values[0]
        ax1.xaxis.set_major_locator(MultipleLocator(step))
        ax1.yaxis.set_major_locator(MultipleLocator(step))

        plt.tight_layout()
        show_or_save_plot(filename)

    def drift_bearing_data(self, *, filename=None):
        """Plots drift (m) and bearing (°) from launch vs time.

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
        drift = self._low_altitude_series(self.flight.drift)
        bearing = self._low_altitude_series(self.flight.bearing)
        plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(
            drift[:, 0],
            drift[:, 1],
            color=self._TRAJECTORY_COLOR,
        )
        ax1.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax1.set_ylim(bottom=0)
        ax1.set_title("Drift from Launch")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Drift (m)")
        ax1.grid(True)
        self._add_event_markers_dropline(ax1, y_bottom=0)

        ax2 = plt.subplot(212)
        ax2.plot(
            bearing[:, 0],
            bearing[:, 1],
            color=self._TRAJECTORY_COLOR,
        )
        ax2.set_xlim(self.flight.time[0], self.low_altitude_end_time)
        ax2.set_title("Bearing from Launch")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Bearing (°)")
        ax2.grid(True)
        self._add_event_markers_dropline(ax2, legend=False)

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def angle_of_attack_data(self, *, filename=None):
        """Plots angle of attack, partial angle of attack, and angle of sideslip.

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
        t_lower = self.flight.out_of_rail_time
        t_upper = (
            self.flight.apogee_time
            if self.flight.apogee_time != 0
            else self.low_altitude_end_time
        )
        t_upper = min(t_upper, self.low_altitude_end_time)

        def _ylim_in_range(arr):
            mask = (arr[:, 0] >= t_lower) & (arr[:, 0] <= t_upper)
            vals = arr[mask, 1]
            if len(vals) == 0:
                return 10.0
            # Use 95th percentile so the runaway rise near apogee (v→0)
            # does not dominate the y-scale; multiply by 1.5 to keep headroom.
            top = float(np.percentile(vals, 95)) * 1.5
            return max(top, 1.0)

        def _ylim_signed(arr):
            # Symmetric y-limits for signed quantities (partial angle of attack,
            # sideslip): these are arctan2-based and routinely go negative, so a
            # 0 lower bound would clip half the signal. Scale by the 95th
            # percentile of the magnitude to ignore the runaway rise near apogee.
            mask = (arr[:, 0] >= t_lower) & (arr[:, 0] <= t_upper)
            vals = arr[mask, 1]
            if len(vals) == 0:
                return (-10.0, 10.0)
            top = float(np.percentile(np.abs(vals), 95)) * 1.5
            top = max(top, 1.0)
            return (-top, top)

        plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(311)
        ax1.plot(self.flight.angle_of_attack[:, 0], self.flight.angle_of_attack[:, 1])
        ax1.set_xlim(t_lower, t_upper)
        ax1.set_ylim(0, _ylim_in_range(self.flight.angle_of_attack[:, :]))
        ax1.set_title("Angle of Attack")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle of Attack (°)")
        ax1.grid()

        ax2 = plt.subplot(312)
        ax2.plot(
            self.flight.partial_angle_of_attack[:, 0],
            self.flight.partial_angle_of_attack[:, 1],
        )
        ax2.set_xlim(t_lower, t_upper)
        ax2.set_ylim(*_ylim_signed(self.flight.partial_angle_of_attack[:, :]))
        ax2.axhline(0, color="0.6", linewidth=0.8)
        ax2.set_title("Partial Angle of Attack")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Partial Angle of Attack (°)")
        ax2.grid()

        ax3 = plt.subplot(313)
        ax3.plot(
            self.flight.angle_of_sideslip[:, 0], self.flight.angle_of_sideslip[:, 1]
        )
        ax3.set_xlim(t_lower, t_upper)
        ax3.set_ylim(*_ylim_signed(self.flight.angle_of_sideslip[:, :]))
        ax3.axhline(0, color="0.6", linewidth=0.8)
        ax3.set_title("Angle of Sideslip")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle of Sideslip (°)")
        ax3.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def all(self):  # pylint: disable=too-many-statements
        """Plot a trajectory-aware summary of the Flight.

        Flights remaining below 80 km use the established low_altitude view.
        For a launch that crosses 80 km, that view is capped at the first
        geodetic-height crossing and is followed by full-flight Earth-centred
        plots. An arbitrary GCRF initial state without a launch origin receives
        only the Earth-centred view.

        Returns
        -------
        None
        """

        if self.has_low_altitude_segment:
            if self.is_high_altitude_flight:
                print("\n\nLaunch-Site View (limited to first 80 km geodetic height)\n")
            print("\n\nTrajectory 3D Plot\n")
            self.trajectory_3d()

            print("\n\nAltitude Data\n")
            self.altitude_data()

            print("\n\nLaunch-Local Ground Track\n")
            self.ground_track(local=True)

            print("\n\nDrift and Bearing Data\n")
            self.drift_bearing_data()

            print("\n\nTrajectory Kinematic Plots\n")
            self.linear_kinematics_data()

            print("\n\nTrajectory Angular Velocity and Acceleration Plots\n")
            self.angular_kinematics_data()

            print("\n\nAngle of Attack Plots\n")
            self.angle_of_attack_data()

            print("\n\nAngular Position Plots\n")
            self.flight_path_angle_data()

            print("\n\nPath, Attitude and Lateral Attitude Angle Plots\n")
            self.attitude_data()

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
            self.dynamic_stability_data()

            if self.flight.sensors:
                print("\n\nSensor Data Plots\n")
                self.sensor_data()

        show_earth_centered = self.flight.reference_frame == ReferenceFrame.GCRF and (
            self.is_high_altitude_flight or not self.has_low_altitude_segment
        )
        if show_earth_centered:
            print("\n\nFull Earth-Centered Flight / Orbit View\n")
            print("\n\nGCRF State History\n")
            self.earth_centered_state()
            print("\n\nGeodetic Coordinate History\n")
            self.geodetic_coordinates()
            print("\n\nOrbit 3D Plot\n")
            self.orbit_3d()
            print("\n\nFull Ground Track Plot\n")
            self.ground_track()
            print("\n\nOsculating Orbital Elements\n")
            self.orbital_elements()
            print("\n\nOrbital Acceleration Contributions\n")
            self.orbital_acceleration_components()
            print("\n\nOrbital Accelerations in RTN\n")
            self.orbital_accelerations_rtn()
            print("\n\nOrbital Invariants\n")
            self.orbital_energy()
