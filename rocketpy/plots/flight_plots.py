from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

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

    @cached_property
    def first_parachute_event_time(self):
        """Time of the first flight event."""
        if len(self.flight.parachute_events) > 0:
            return self.flight.parachute_events[0][0]
        else:
            return self.flight.t_final

    @cached_property
    def first_parachute_event_time_index(self):
        """Time index of the first flight event."""
        if len(self.flight.parachute_events) > 0:
            return int(
                np.argmin(np.abs(self.flight.x[:, 0] - self.first_parachute_event_time))
            )
        else:
            return -1

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
        "#7de07a", "#f781bf", "#a65628", "#ff7f00",
        "#ffff33", "#377eb8", "#984ea3", "#66c2a5",
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
        except Exception:
            pass

        one_time_events = [
            ev for ev in getattr(self.flight, "events", [])
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
        leg = ax.legend()
        if leg is None:
            return
        handles = getattr(leg, "legend_handles", None) or getattr(
            leg, "legendHandles", []
        )
        labels = [t.get_text() for t in leg.get_texts()]
        combined = sorted(
            zip(labels, handles), key=lambda p: event_times.get(p[0], -1)
        )
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
            if not (xlim[0] <= t_ev <= xlim[1]):
                continue
            if label == "Burnout":
                ax.axvline(x=t_ev, color=self._BURNOUT_LINE_COLOR, linestyle="--",
                           linewidth=self._EVENT_LINE_WIDTH, alpha=1.0, label=label)
            else:
                ax.axvline(x=t_ev, color=color, linestyle="--",
                           linewidth=self._EVENT_LINE_WIDTH, alpha=1.0, label=label)
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
            if not (xlim[0] <= t_ev <= xlim[1]):
                continue
            y_ev = float(np.interp(t_ev, xdata, ydata))
            line_color = self._BURNOUT_LINE_COLOR if label == "Burnout" else color
            lw = self._EVENT_LINE_WIDTH if label == "Burnout" else self._EVENT_LINE_WIDTH
            ax.vlines(t_ev, y_bottom, y_ev, colors=line_color, linestyles="--", linewidth=lw, alpha=1.0)
            if label == "Apogee":
                deferred_apogee = (t_ev, y_ev, label, marker, color, size)
                continue
            s2d = size if marker == "s" else size * 0.5
            kw = dict(marker=marker, color=color, s=s2d, label=label, zorder=10)
            if marker != "x":
                kw["edgecolors"] = "black"
                kw["linewidths"] = 0.8
            else:
                kw["linewidths"] = 1.5
            ax.scatter(t_ev, y_ev, **kw)
        if deferred_apogee is not None:
            t_ev, y_ev, label, marker, color, size = deferred_apogee
            kw = dict(marker=marker, color=color, s=size * 0.5, label=label, zorder=20)
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
            color=self._TRAJECTORY_COLOR,
            linewidth="2",
            zorder=2,
        )
        ax1.scatter(
            self.flight.x(0),
            self.flight.y(0),
            self.flight.z(0) - self.flight.env.elevation,
            s=20,
            facecolors="#ffd400",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Launch",
            depthshade=False,
        )
        # Plot single-trigger events (events configured with trigger_only_once)
        if show_events:
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
                    x_b = self.flight.x(t_burn)
                    y_b = self.flight.y(t_burn)
                    z_b = self.flight.z(t_burn) - self.flight.env.elevation
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
                except Exception:
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
                        x_ev = self.flight.x(t_ev)
                        y_ev = self.flight.y(t_ev)
                        z_ev = self.flight.z(t_ev) - self.flight.env.elevation
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
            except Exception:
                # plotting of events should never break the main plot
                pass
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

        ax1 = plt.subplot(411)
        ax1.plot(self.flight.speed[:, 0], self.flight.speed[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_title("Velocity Magnitude | Acceleration Magnitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (m/s)", color="#ff7f0e")
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)
        self._add_event_markers(ax1)

        ax1up = ax1.twinx()
        ax1up.plot(
            self.flight.acceleration[:, 0],
            self.flight.acceleration[:, 1],
            color="#1f77b4",
        )
        ax1up.set_ylabel("Acceleration (m/s²)", color="#1f77b4")
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(412)
        ax2.plot(self.flight.vz[:, 0], self.flight.vz[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, self.flight.t_final)
        ax2.set_title("Velocity Z | Acceleration Z")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity Z (m/s)", color="#ff7f0e")
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)
        self._add_event_markers(ax2, legend=False)

        ax2up = ax2.twinx()
        ax2up.plot(self.flight.az[:, 0], self.flight.az[:, 1], color="#1f77b4")
        ax2up.set_ylabel("Acceleration Z (m/s²)", color="#1f77b4")
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(413)
        ax3.plot(self.flight.vy[:, 0], self.flight.vy[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, self.flight.t_final)
        ax3.set_title("Velocity Y | Acceleration Y")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity Y (m/s)", color="#ff7f0e")
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)
        self._add_event_markers(ax3, legend=False)

        ax3up = ax3.twinx()
        ax3up.plot(self.flight.ay[:, 0], self.flight.ay[:, 1], color="#1f77b4")
        ax3up.set_ylabel("Acceleration Y (m/s²)", color="#1f77b4")
        ax3up.tick_params("y", colors="#1f77b4")

        ax4 = plt.subplot(414)
        ax4.plot(self.flight.vx[:, 0], self.flight.vx[:, 1], color="#ff7f0e")
        ax4.set_xlim(0, self.flight.t_final)
        ax4.set_title("Velocity X | Acceleration X")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity X (m/s)", color="#ff7f0e")
        ax4.tick_params("y", colors="#ff7f0e")
        ax4.grid(True)
        self._add_event_markers(ax4, legend=False)

        ax4up = ax4.twinx()
        ax4up.plot(self.flight.ax[:, 0], self.flight.ax[:, 1], color="#1f77b4")
        ax4up.set_ylabel("Acceleration X (m/s²)", color="#1f77b4")
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
        ax1.set_xlim(0, self.first_parachute_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Euler Parameters")
        ax1.set_title("Euler Parameters")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(412)
        ax2.plot(self.flight.psi[:, 0], self.flight.psi[:, 1])
        ax2.set_xlim(0, self.first_parachute_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ψ (°)")
        ax2.set_title("Euler Precession Angle")
        ax2.grid(True)

        ax3 = plt.subplot(413)
        ax3.plot(self.flight.theta[:, 0], self.flight.theta[:, 1], label="θ - Nutation")
        ax3.set_xlim(0, self.first_parachute_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("θ (°)")
        ax3.set_title("Euler Nutation Angle")
        ax3.grid(True)

        ax4 = plt.subplot(414)
        ax4.plot(self.flight.phi[:, 0], self.flight.phi[:, 1], label="φ - Spin")
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
        ax1.set_xlim(0, self.first_parachute_event_time)
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
        plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(311)
        ax1.plot(self.flight.w1[:, 0], self.flight.w1[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.first_parachute_event_time)
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
        ax2.set_xlim(0, self.first_parachute_event_time)
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
        ax3.set_xlim(0, self.first_parachute_event_time)
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
                        else self.flight.t_final
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
                    else self.flight.t_final
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
                    else self.flight.t_final
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
        plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(
            self.flight.aerodynamic_lift[: self.first_parachute_event_time_index, 0],
            self.flight.aerodynamic_lift[: self.first_parachute_event_time_index, 1],
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
        ax1.set_ylabel("Lift Force (N)")
        ax1.set_title("Aerodynamic Lift Resultant Force")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            self.flight.aerodynamic_drag[: self.first_parachute_event_time_index, 0],
            self.flight.aerodynamic_drag[: self.first_parachute_event_time_index, 1],
        )
        ax2.set_xlim(0, self.first_parachute_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Drag Force (N)")
        ax2.set_title("Aerodynamic Drag Force")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.aerodynamic_bending_moment[
                : self.first_parachute_event_time_index, 0
            ],
            self.flight.aerodynamic_bending_moment[
                : self.first_parachute_event_time_index, 1
            ],
            label="Resultant",
        )
        ax3.plot(
            self.flight.M1[: self.first_parachute_event_time_index, 0],
            self.flight.M1[: self.first_parachute_event_time_index, 1],
            label="M1",
        )
        ax3.plot(
            self.flight.M2[: self.first_parachute_event_time_index, 0],
            self.flight.M2[: self.first_parachute_event_time_index, 1],
            label="M2",
        )
        ax3.set_xlim(0, self.first_parachute_event_time)
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Bending Moment (N m)")
        ax3.set_title("Aerodynamic Bending Resultant Moment")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(
            self.flight.aerodynamic_spin_moment[
                : self.first_parachute_event_time_index, 0
            ],
            self.flight.aerodynamic_spin_moment[
                : self.first_parachute_event_time_index, 1
            ],
        )
        ax4.set_xlim(0, self.first_parachute_event_time)
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
        if callable(thrust_power.source):
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
        if callable(drag_power.source):
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
        plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(311)
        ax1.plot(self.flight.mach_number[:, 0], self.flight.mach_number[:, 1])
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()
        self._add_event_markers(ax1)

        ax2 = plt.subplot(312)
        ax2.plot(self.flight.reynolds_number[:, 0], self.flight.reynolds_number[:, 1])
        ax2.set_xlim(0, self.flight.t_final)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()
        self._add_event_markers(ax2, legend=False)

        ax3 = plt.subplot(313)
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

        ax1 = plt.subplot(211)
        ax1.plot(self.flight.stability_margin[:, 0], self.flight.stability_margin[:, 1])
        ax1.set_title("Stability Margin")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Stability Margin (c)")
        ax1.set_xlim(0, self.first_parachute_event_time)
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
        import warnings

        warnings.warn(
            "pressure_signals() is deprecated and will be removed in v1.13. "
            "Use a Barometer Sensor with built-in noise and access its data "
            "via flight.sensor_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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

        z_times = self.flight.z[:, 0]
        z_agl = self.flight.z[:, 1] - self.flight.env.elevation

        ax1 = plt.subplot(111)
        ax1.plot(z_times, z_agl, color=self._TRAJECTORY_COLOR)
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_ylim(bottom=0)
        ax1.set_title("Altitude Above Ground Level")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Altitude AGL (m)")
        ax1.grid(True)

        # Event markers: dot on the curve + dashed line from y=0 (line not in legend).
        # Apogee is drawn last so it renders on top of coincident markers.
        xlim = ax1.get_xlim()
        deferred_apogee = None
        for t_ev, label, marker, color, size in self._collect_events():
            if label in ("Out Of Rail", "Landing"):
                continue
            if not (xlim[0] <= t_ev <= xlim[1]):
                continue
            alt_ev = float(np.interp(t_ev, z_times, z_agl))
            line_color = self._BURNOUT_LINE_COLOR if label == "Burnout" else color
            lw = self._EVENT_LINE_WIDTH if label == "Burnout" else self._EVENT_LINE_WIDTH
            ax1.vlines(t_ev, 0, alt_ev, colors=line_color, linestyles="--", linewidth=lw, alpha=1.0)
            if label == "Apogee":
                deferred_apogee = (t_ev, label, marker, color, size, alt_ev)
                continue
            s2d = size if marker == "s" else size * 0.5
            kw = dict(marker=marker, color=color, s=s2d, label=label, zorder=10)
            if marker != "x":
                kw["edgecolors"] = "black"
                kw["linewidths"] = 0.8
            else:
                kw["linewidths"] = 1.5
            ax1.scatter(t_ev, alt_ev, **kw)
        if deferred_apogee is not None:
            t_ev, label, marker, color, size, alt_ev = deferred_apogee
            ax1.scatter(t_ev, alt_ev, marker=marker, color=color, s=size * 0.5,
                        label=label, zorder=20, edgecolors="black", linewidths=0.8)
        self._sorted_legend(ax1)

        plt.tight_layout()
        show_or_save_plot(filename)

    def ground_track(self, *, filename=None):
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
        plt.figure(figsize=(7, 7))

        ax1 = plt.subplot(111)
        ax1.plot(self.flight.x[:, 1], self.flight.y[:, 1], color=self._TRAJECTORY_COLOR, label="Trajectory", zorder=1)
        # Launch point (t=0 is not a trigger-once event, so add it explicitly)
        ax1.scatter(
            [self.flight.x(0)],
            [self.flight.y(0)],
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
            if label == "Out Of Rail":
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
            kw = dict(marker=marker, color=color, s=s2d, label=label, zorder=10)
            if marker != "x":
                kw["edgecolors"] = "black"
                kw["linewidths"] = 0.8
            else:
                kw["linewidths"] = 1.5
            ax1.scatter([self.flight.x(t_ev)], [self.flight.y(t_ev)], **kw)
        if deferred_apogee is not None:
            t_ev, label, marker, color, size = deferred_apogee
            ax1.scatter(
                [self.flight.x(t_ev)], [self.flight.y(t_ev)],
                marker=marker, color=color, s=size * 0.5, label=label,
                edgecolors="black", linewidths=0.8, zorder=20,
            )
        ax1.set_title("Ground Track")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        self._sorted_legend(ax1)
        ax1.grid(True)
        # Compute symmetric equal-range limits so the axes fill the square figure
        x_data = self.flight.x[:, 1]
        y_data = self.flight.y[:, 1]
        x_center = (float(x_data.max()) + float(x_data.min())) / 2
        y_center = (float(y_data.max()) + float(y_data.min())) / 2
        half = max(
            float(x_data.max()) - float(x_data.min()),
            float(y_data.max()) - float(y_data.min()),
        ) / 2 * 1.1 + 1
        ax1.set_xlim(x_center - half, x_center + half)
        ax1.set_ylim(y_center - half, y_center + half)
        ax1.set_aspect("equal", adjustable="box")

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
        plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(self.flight.drift[:, 0], self.flight.drift[:, 1], color=self._TRAJECTORY_COLOR)
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_ylim(bottom=0)
        ax1.set_title("Drift from Launch")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Drift (m)")
        ax1.grid(True)
        self._add_event_markers_dropline(ax1, y_bottom=0)

        ax2 = plt.subplot(212)
        ax2.plot(self.flight.bearing[:, 0], self.flight.bearing[:, 1], color=self._TRAJECTORY_COLOR)
        ax2.set_xlim(0, self.flight.t_final)
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
            else self.flight.t_final
        )

        def _ylim_in_range(arr):
            mask = (arr[:, 0] >= t_lower) & (arr[:, 0] <= t_upper)
            vals = arr[mask, 1]
            if len(vals) == 0:
                return 10.0
            # Use 95th percentile so the runaway rise near apogee (v→0)
            # does not dominate the y-scale; multiply by 1.5 to keep headroom.
            top = float(np.percentile(vals, 95)) * 1.5
            return max(top, 1.0)

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
        ax2.set_ylim(0, _ylim_in_range(self.flight.partial_angle_of_attack[:, :]))
        ax2.set_title("Partial Angle of Attack")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Partial Angle of Attack (°)")
        ax2.grid()

        ax3 = plt.subplot(313)
        ax3.plot(
            self.flight.angle_of_sideslip[:, 0], self.flight.angle_of_sideslip[:, 1]
        )
        ax3.set_xlim(t_lower, t_upper)
        ax3.set_ylim(0, _ylim_in_range(self.flight.angle_of_sideslip[:, :]))
        ax3.set_title("Angle of Sideslip")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle of Sideslip (°)")
        ax3.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def all(self):  # pylint: disable=too-many-statements
        """Prints out all plots available about the Flight.

        Returns
        -------
        None
        """

        print("\n\nTrajectory 3d Plot\n")
        self.trajectory_3d()

        print("\n\nAltitude Data\n")
        self.altitude_data()

        print("\n\nGround Track\n")
        self.ground_track()

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

        print("\n\nPath, Attitude and Lateral Attitude Angle plots\n")
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