"""Mission – orchestrates one Flight per vehicle configuration."""

from copy import deepcopy

import matplotlib.pyplot as plt

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.plots.compare.compare_flights import CompareFlights
from rocketpy.plots.plot_helpers import show_or_save_plot
from rocketpy.rocket.multistage import MultiStageRocket, Stage
from rocketpy.simulation.flight import Flight

# Colors for plot_timeline()'s event markers, keyed by the event name's
# prefix before ":" (or the whole name, for liftoff/rail_departure,
# which carry no body name).
_EVENT_COLORS = {
    "liftoff": "black",
    "rail_departure": "gray",
    "ignition": "green",
    "burnout": "darkorange",
    "separation": "royalblue",
    "ejection": "magenta",
    "apogee": "purple",
    "impact": "red",
}

# Marker shapes for plot_trajectory_events()'s 3D event points, same
# keys as _EVENT_COLORS - distinct shape *and* color per event type
# reads clearly even before the legend is checked.
_EVENT_MARKERS = {
    "liftoff": "^",
    "rail_departure": "^",
    "ignition": ">",
    "burnout": "s",
    "separation": "D",
    "ejection": "v",
    "apogee": "*",
    "impact": "X",
}

# Motor attributes that are Functions of the motor's own local time and
# must be re-anchored when a stage ignites later than its own local t=0.
# Constants (nozzle_position, dry inertia, center_of_dry_mass_position,
# ...) don't vary with time and are left untouched.
_MOTOR_TIME_FUNCTIONS = (
    "thrust",
    "vacuum_thrust",
    "exhaust_velocity",
    "total_mass",
    "propellant_mass",
    "total_mass_flow_rate",
    "center_of_mass",
    "center_of_propellant_mass",
    "I_11",
    "I_22",
    "I_33",
    "I_12",
    "I_13",
    "I_23",
    "propellant_I_11",
    "propellant_I_22",
    "propellant_I_33",
    "propellant_I_12",
    "propellant_I_13",
    "propellant_I_23",
)


class Mission:
    """Simulate a complete mission: a vehicle that splits into multiple
    bodies, each simulated to impact (or until max_time).

    Walks the vehicle's separation/ejection timing, runs one Flight per
    vehicle configuration with correct state handoff between them, and
    groups the resulting Flight objects per physical body along with a
    global event timeline.

    Any number of stages, and any number of deployables (each riding a
    chosen stage, ejecting independently), are supported, in any
    combination - e.g. a deployable riding the second of three stages,
    ejecting after the first stage separates but before the second does.

    Separation and ignition timing are deterministic, computed ahead of
    time from motor burn_time and the delays given on each Stage - there
    is no generic mid-flight trigger/event solver (Flight only exposes
    max_time and terminate_on_apogee for early termination). A Stage's
    ``separation`` is therefore a plain float: the delay, in seconds,
    after that stage's own motor burns out. Deployable ejection is
    either ``None`` (never) or the string ``"apogee"`` (via Flight's own
    terminate_on_apogee). Event-triggered separation/ignition/ejection
    beyond that is not supported yet, nor is more than one deployable
    ejecting from the same configuration simultaneously.

    Parameters
    ----------
    vehicle : MultiStageRocket or Rocket
        The vehicle to fly. A plain Rocket is sugar for a single-stage
        MultiStageRocket with nothing separable (one Flight).
    environment : Environment
    rail_length, inclination, heading : same vocabulary as Flight.
    max_time : float
        Mission-wide limit in seconds; every body's flight ends by then.
    (remaining solver parameters are passed through to each Flight)

    Attributes
    ----------
    flights : dict
        Body name -> list of Flight, in time order. A Flight appears
        under every body that was aboard it.
    timeline : list of (float, str)
        (time, event_name) tuples sorted by time. Canonical names:
        "ignition:<stage>", "liftoff", "rail_departure",
        "separation:<stage>", "ejection:<deployable>", "impact:<body>".
    """

    def __init__(
        self,
        vehicle,
        environment,
        rail_length,
        inclination=80.0,
        heading=90.0,
        name="Mission",
        max_time=600,
        rtol=1e-6,
        atol=None,
        time_overshoot=True,
        ode_solver="LSODA",
        verbose=False,
    ):
        self.vehicle = (
            vehicle
            if isinstance(vehicle, MultiStageRocket)
            else MultiStageRocket(stages=[vehicle])
        )
        self.environment = environment
        self.rail_length = rail_length
        self.inclination = inclination
        self.heading = heading
        self.name = name
        self.max_time = max_time
        self.rtol = rtol
        self.atol = atol
        self.time_overshoot = time_overshoot
        self.ode_solver = ode_solver
        self.verbose = verbose

        self.flights = {}
        self.timeline = []
        self._all_flights = []
        self._recorded_burnouts = set()
        self._simulate()

    @property
    def all_flights(self):
        """Every Flight object, in execution order, each appearing once
        (a Flight shared by several bodies, e.g. the full stack, is not
        repeated). Feeds :meth:`trajectories_3d` and :meth:`positions`
        internally, and is also usable directly with any other
        :class:`~rocketpy.plots.compare.compare_flights.CompareFlights`
        plot those two don't wrap, e.g.
        ``CompareFlights(mission.all_flights).velocities()``.
        """
        return list(self._all_flights)

    def trajectories_3d(self, figsize=(7, 7), legend=None, filename=None):
        """Plain 3D trajectory plot for every flight in this mission, with
        no event markers - the mission-level equivalent of a single
        ``Flight``'s own ``flight.plots.trajectory_3d()``, so a caller
        doesn't need to reach for :class:`CompareFlights` directly for
        this. See :meth:`plot_trajectory_events` for the same plot with
        every timeline event marked.

        Parameters
        ----------
        figsize : tuple, optional
            Passed through to CompareFlights.trajectories_3d(). Default
            (7, 7).
        legend : bool | None, optional
            Passed through to CompareFlights.trajectories_3d(). Default
            None (shows the legend).
        filename : str | None, optional
            Path to save the plot to. Default None, which shows it
            instead.
        """
        CompareFlights(self.all_flights).trajectories_3d(
            figsize=figsize, legend=legend, filename=filename
        )

    def positions(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """x/y/z vs time, side by side, for every flight in this mission -
        sugar for ``CompareFlights(mission.all_flights).positions()``, the
        same way :meth:`trajectories_3d` wraps
        ``CompareFlights(...).trajectories_3d()``.

        Parameters
        ----------
        figsize : tuple, optional
            Passed through to CompareFlights.positions(). Default (7, 10).
        x_lim : tuple, optional
            Passed through to CompareFlights.positions(). Default None.
        y_lim : tuple, optional
            Passed through to CompareFlights.positions(). Default None.
        legend : bool, optional
            Passed through to CompareFlights.positions(). Default True.
        filename : str | None, optional
            Path to save the plot to. Default None, which shows it
            instead.
        """
        CompareFlights(self.all_flights).positions(
            figsize=figsize, x_lim=x_lim, y_lim=y_lim, legend=legend, filename=filename
        )

    def _event_label_positions(self):
        """(time, name, color, label_x, stagger_level) per timeline
        event, for :meth:`plot_timeline`.

        Deterministic-delay/apogee timing tends to bunch several events
        close together (e.g. every stage's own ignition/burnout right
        around a separation) - events within a small fraction of the
        mission's own time span of each other are grouped into a
        cluster and their LABELS (not the marker lines themselves,
        which stay at each event's exact time) are fanned out across a
        small horizontal window and staggered vertically, so a dense
        cluster doesn't render as unreadable stacked text.
        """
        if not self.timeline:
            return []
        times = [t for t, _ in self.timeline]
        time_span = (max(times) - min(times)) or 1.0
        cluster_gap = time_span * 0.03
        fan_width = time_span * 0.06

        clusters = []
        current = []
        last_time = None
        for entry in self.timeline:
            t, _ = entry
            if last_time is not None and t - last_time > cluster_gap:
                clusters.append(current)
                current = []
            current.append(entry)
            last_time = t
        if current:
            clusters.append(current)

        positions = []
        level = 0
        for cluster in clusters:
            size = len(cluster)
            for i, (t, name) in enumerate(cluster):
                color = _EVENT_COLORS.get(name.split(":")[0], "black")
                spread = (i / (size - 1) - 0.5) if size > 1 else 0.0
                label_x = t + fan_width * spread
                positions.append((t, name, color, label_x, level % 10))
                level += 1
        return positions

    def plot_timeline(self, filename=None):
        """Plot altitude vs time for every flight in this mission, with
        every timeline event (ignition, burnout, separation, ejection,
        apogee, impact, ...) marked and labeled at its own time - a
        "mission profile" chart.

        Reads only ``self.timeline``'s ``(time, name)`` tuples and each
        flight's own solution - nothing here assumes how a timeline
        entry was produced, so this keeps working unchanged if
        ``self.timeline`` is ever built from real ``Event`` objects
        instead of Mission's own deterministic bookkeeping (see the
        "Advance Multistage" plan's Gap 3 - event-triggered separation/
        ignition/ejection beyond deterministic delays and apogee).

        Parameters
        ----------
        filename : str | None, optional
            Path to save the plot to. Default None, which shows it
            instead.
        """
        _, ax = plt.subplots(figsize=(12, 6))

        for flight in self.all_flights:
            times = [state[0] for state in flight.solution]
            altitudes = [state[3] for state in flight.solution]
            ax.plot(times, altitudes, label=flight.name, linewidth=1.5)

        ymin, ymax = ax.get_ylim()
        label_span = (ymax - ymin) or 1.0
        for time, name, color, label_x, level in self._event_label_positions():
            ax.axvline(time, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
            label_y = ymax - label_span * 0.035 * (1 + level)
            ax.text(
                label_x,
                label_y,
                name,
                rotation=90,
                ha="center",
                va="top",
                color=color,
                fontsize=7,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Altitude (m)")
        ax.set_title(f"{self.name}: Flight Profile")
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, title="Flights"
        )
        plt.tight_layout()

        show_or_save_plot(filename)

    def _flight_covering_time(self, time, tolerance=1e-6):
        """The flight (from :attr:`all_flights`, in execution order)
        whose own solution spans ``time``, or ``None`` if none does.

        Position is continuous across every handoff (by construction -
        see ``_handoff_state``), so at an exact boundary instant (e.g.
        a separation time, simultaneously the end of the parent flight
        and the start of a child's) either flight is an equally correct
        answer; this returns whichever comes first in execution order.
        """
        for flight in self.all_flights:
            start_time = flight.solution[0][0]
            end_time = flight.solution[-1][0]
            if start_time - tolerance <= time <= end_time + tolerance:
                return flight
        return None

    def plot_trajectory_events(self, filename=None):
        """3D trajectory for every flight in this mission, with every
        timeline event marked as a colored, shaped point at its own
        position - not a line, which has no natural per-instant meaning
        in 3D the way a vertical line does on an altitude-vs-time axis.

        Reads only ``self.timeline``'s ``(time, name)`` tuples and each
        flight's own ``x``/``y``/``z`` - see :meth:`plot_timeline` for
        why that keeps this forward compatible with a future
        `Event`-object-backed timeline.

        Parameters
        ----------
        filename : str | None, optional
            Path to save the plot to. Default None, which shows it
            instead.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        for flight in self.all_flights:
            xs = [state[1] for state in flight.solution]
            ys = [state[2] for state in flight.solution]
            zs = [state[3] for state in flight.solution]
            ax.plot(xs, ys, zs, label=flight.name, linewidth=1.5)

        seen_event_types = set()
        for time, name in self.timeline:
            flight = self._flight_covering_time(time)
            if flight is None:
                continue
            event_type = name.split(":")[0]
            color = _EVENT_COLORS.get(event_type, "black")
            marker = _EVENT_MARKERS.get(event_type, "o")
            size = 160 if event_type == "apogee" else 70
            ax.scatter(
                [flight.x(time)],
                [flight.y(time)],
                [flight.z(time)],
                color=color,
                marker=marker,
                s=size,
                edgecolor="black",
                linewidth=0.5,
                zorder=10,
                label=None if event_type in seen_event_types else event_type,
            )
            seen_event_types.add(event_type)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{self.name}: Trajectory with Events")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.tight_layout()

        show_or_save_plot(filename)

    def _simulate(self):
        if not self.vehicle.stages:
            raise ValueError("Vehicle has no stages.")
        for stage in self.vehicle.stages:
            if stage.ignition is not None:
                raise NotImplementedError(
                    "Event-triggered ignition is not supported yet; use "
                    "ignition_delay (a deterministic float delay) instead."
                )
        for deployable in self.vehicle.deployables:
            if deployable.free_rocket is None:
                raise NotImplementedError(
                    "Deployable ejection currently requires free_rocket; "
                    "building a free-flight Rocket from add_surface()-added "
                    "surfaces is not yet implemented."
                )
            if deployable.ejection not in (None, "apogee"):
                raise NotImplementedError(
                    "Deployable.ejection currently only supports None "
                    "(never) or 'apogee'."
                )

        self.timeline.append((0.0, f"ignition:{self.vehicle.stages[0].name}"))
        self.timeline.append((0.0, "liftoff"))
        self._walk(
            tuple(self.vehicle.stages),
            tuple(self.vehicle.deployables),
            initial_solution=None,
            is_root=True,
        )
        self.timeline.sort(key=lambda entry: entry[0])

    def _walk(
        self, active_stages, carried_deployables, initial_solution, is_root, rocket=None
    ):
        """Run one configuration to its next event (or natural
        completion) and recurse into whatever children result.

        ``rocket`` lets a caller pass an already-composed Rocket (e.g.
        one whose bottom stage's motor was just time-shifted by
        _shift_motor_ignition) instead of recomposing it here.
        """
        if rocket is None:
            rocket = self.vehicle.flight_rocket(active_stages, carried_deployables)
        bottom = active_stages[0]

        if bottom.name not in self._recorded_burnouts:
            self._recorded_burnouts.add(bottom.name)
            self.timeline.append((bottom.burn_out_time, f"burnout:{bottom.name}"))

        pending_separation = len(active_stages) > 1 and bottom.separation is not None
        apogee_deployables = tuple(
            d for d in carried_deployables if d.ejection == "apogee"
        )
        if len(apogee_deployables) > 1:
            raise NotImplementedError(
                "More than one deployable ejecting at apogee from the same "
                "configuration simultaneously is not yet supported."
            )

        if pending_separation:
            self._walk_separation(
                active_stages, carried_deployables, rocket, initial_solution, is_root
            )
        elif apogee_deployables:
            self._walk_ejection(
                active_stages,
                carried_deployables,
                apogee_deployables[0],
                rocket,
                initial_solution,
                is_root,
            )
        else:
            self._walk_terminal(
                active_stages, carried_deployables, rocket, initial_solution, is_root
            )

    def _walk_separation(
        self, active_stages, carried_deployables, rocket, initial_solution, is_root
    ):
        """Run to the bottom stage's separation, then recurse into the
        departing stage (falling away on its own) and the remaining
        stack (continuing, possibly with its new bottom stage's motor
        ignition time-shifted).
        """
        bottom = active_stages[0]
        separation_time = bottom.burn_out_time + bottom.separation
        name = _configuration_name(active_stages, carried_deployables)

        flight = self._run_flight(
            rocket,
            name=name,
            initial_solution=initial_solution,
            max_time=separation_time,
        )
        self._register_flight((*active_stages, *carried_deployables), flight)
        if is_root:
            self.timeline.append((flight.out_of_rail_time, "rail_departure"))
        self.timeline.append((separation_time, f"separation:{bottom.name}"))
        ending_state = flight.solution[-1]

        split = self._split_at_separation(active_stages, carried_deployables, bottom)
        (
            departing_deployables,
            remaining_stages,
            remaining_deployables,
            departing_rocket,
            departing_delta_v,
            remaining_delta_v,
        ) = split

        departing_initial = self._handoff_state(
            ending_state, rocket, departing_rocket, departing_delta_v
        )
        self._walk((bottom,), departing_deployables, departing_initial, is_root=False)

        ignition_time = separation_time + remaining_stages[0].ignition_delay
        self.timeline.append((ignition_time, f"ignition:{remaining_stages[0].name}"))
        shifted_stages = self._shift_motor_ignition(remaining_stages, ignition_time)
        ignited_rocket = self.vehicle.flight_rocket(
            shifted_stages, remaining_deployables
        )
        remaining_initial = self._handoff_state(
            ending_state, rocket, ignited_rocket, remaining_delta_v
        )
        self._walk(
            shifted_stages,
            remaining_deployables,
            remaining_initial,
            is_root=False,
            rocket=ignited_rocket,
        )

    def _split_at_separation(self, active_stages, carried_deployables, bottom):
        """Everything _walk_separation needs about the two children of a
        separation: which deployables go with which body, the departing
        body's own composed Rocket, and the momentum-conserving
        separation_delta_v split between the two (the departing stage is
        spent - dry_mass; the remaining stack hasn't ignited its new
        bottom stage yet - full total_mass at that stage's own t=0).
        """
        departing_deployables = tuple(
            d for d in carried_deployables if d.stage is bottom
        )
        remaining_deployables = tuple(
            d for d in carried_deployables if d not in departing_deployables
        )
        remaining_stages = active_stages[1:]

        departing_rocket = self.vehicle.flight_rocket((bottom,), departing_deployables)
        remaining_rocket = self.vehicle.flight_rocket(
            remaining_stages, remaining_deployables
        )
        departing_delta_v, remaining_delta_v = self._momentum_split(
            departing_rocket.dry_mass,
            remaining_rocket.total_mass(0),
            bottom.separation_delta_v,
        )
        return (
            departing_deployables,
            remaining_stages,
            remaining_deployables,
            departing_rocket,
            departing_delta_v,
            remaining_delta_v,
        )

    def _walk_ejection(
        self,
        active_stages,
        carried_deployables,
        deployable,
        rocket,
        initial_solution,
        is_root,
    ):
        """Run to apogee, then recurse into the carrier (continuing
        without the deployable) and the deployable (on its own
        free_rocket).

        A handoff can leave a body already at or past its own apogee -
        e.g. a departing stage given a separation_delta_v that further
        slows an already-near-peak trajectory. Flight's own apogee
        root-finding assumes a flight starts out ascending; handing it
        an already-descending initial_solution with
        terminate_on_apogee=True crashes deep inside Flight's flight-
        phase bookkeeping instead of raising a clear error. That case
        skips running a Flight altogether and treats the handoff
        instant itself as the ejection moment - there is no real time
        spent "carrying the deployable while past apogee" to represent.
        """
        name = _configuration_name(active_stages, carried_deployables)
        already_past_apogee = initial_solution is not None and initial_solution[6] <= 0
        if already_past_apogee:
            ending_state = initial_solution
            apogee_time = initial_solution[0]
        else:
            flight = self._run_flight(
                rocket,
                name=name,
                initial_solution=initial_solution,
                terminate_on_apogee=True,
            )
            self._register_flight((*active_stages, *carried_deployables), flight)
            if is_root:
                self.timeline.append((flight.out_of_rail_time, "rail_departure"))
            ending_state = flight.solution[-1]
            apogee_time = flight.apogee_time
        self.timeline.append((apogee_time, f"ejection:{deployable.name}"))

        remaining_deployables = tuple(
            d for d in carried_deployables if d is not deployable
        )
        carrier_rocket = self.vehicle.flight_rocket(
            active_stages, remaining_deployables
        )
        carrier_delta_v, deployable_delta_v = self._momentum_split(
            carrier_rocket.total_mass(apogee_time),
            deployable.free_rocket.total_mass(0),
            deployable.separation_delta_v,
        )

        carrier_initial = self._handoff_state(
            ending_state, rocket, carrier_rocket, carrier_delta_v
        )
        self._walk(
            active_stages,
            remaining_deployables,
            carrier_initial,
            is_root=False,
            rocket=carrier_rocket,
        )

        deployable_initial = self._handoff_state(
            ending_state, rocket, deployable.free_rocket, deployable_delta_v
        )
        deployable_flight = self._run_flight(
            deployable.free_rocket,
            name=deployable.name,
            initial_solution=deployable_initial,
        )
        self._register_flight((deployable,), deployable_flight)
        self.timeline.append((deployable_flight.t_final, f"impact:{deployable.name}"))

    def _walk_terminal(
        self, active_stages, carried_deployables, rocket, initial_solution, is_root
    ):
        """No more separations or ejections pending: fly to impact (or
        max_time) and stop recursing.
        """
        name = _configuration_name(active_stages, carried_deployables)
        flight = self._run_flight(rocket, name=name, initial_solution=initial_solution)
        bodies = (*active_stages, *carried_deployables)
        self._register_flight(bodies, flight)
        if is_root:
            self.timeline.append((flight.out_of_rail_time, "rail_departure"))
        for body in bodies:
            self.timeline.append((flight.t_final, f"impact:{body.name}"))

    def _register_flight(self, bodies, flight):
        for body in bodies:
            self.flights.setdefault(body.name, []).append(flight)
        # Flight.apogee_time defaults to 0 (not some "not found" sentinel)
        # when the flight ends - via separation, or max_time - before it
        # ever reaches a genuine local-altitude-maximum: this phase's
        # vehicle was still ascending the whole time, and its true
        # apogee happens later, in a subsequent configuration's own
        # flight (which will get its own, correctly-timed apogee entry).
        # Whether THIS flight actually reached one is settled by its own
        # physics, not by comparing apogee_time against its time bounds
        # (an ejection flight runs with terminate_on_apogee=True and so
        # ends EXACTLY at its own apogee, by design - apogee_time then
        # sits right at this flight's own end, which a bounds check
        # would wrongly treat the same as "never reached"): a flight
        # that's already at or past its peak by the time it ends
        # (vz <= 0) did reach a real apogee within it; one still
        # ascending at the end (vz > 0) did not.
        if flight.solution[-1][6] <= 0:
            self.timeline.append((flight.apogee_time, f"apogee:{flight.name}"))

    def _run_flight(
        self,
        rocket,
        name,
        initial_solution=None,
        max_time=None,
        terminate_on_apogee=False,
    ):
        """Run one Flight in absolute mission time.

        The single choke point through which every Flight Mission creates
        is constructed, so this is also where all_flights collects them,
        in execution order, each exactly once. ``name`` distinguishes
        each Flight in plots such as CompareFlights, which otherwise
        labels every line "Flight" (Flight's own default).
        """
        flight = Flight(
            rocket=rocket,
            environment=self.environment,
            rail_length=self.rail_length,
            inclination=self.inclination,
            heading=self.heading,
            initial_solution=initial_solution,
            terminate_on_apogee=terminate_on_apogee,
            max_time=max_time if max_time is not None else self.max_time,
            rtol=self.rtol,
            atol=self.atol,
            time_overshoot=self.time_overshoot,
            ode_solver=self.ode_solver,
            verbose=self.verbose,
            name=name,
        )
        self._all_flights.append(flight)
        return flight

    @staticmethod
    def _momentum_split(mass_a, mass_b, delta_v):
        """Momentum-conserving split of a relative separation_delta_v
        between two children of masses mass_a and mass_b: returns
        (delta_v_a, delta_v_b) such that delta_v_b - delta_v_a == delta_v
        and mass_a * delta_v_a + mass_b * delta_v_b == 0 (momentum is
        conserved about the common pre-separation velocity).
        """
        total_mass = mass_a + mass_b
        delta_v_a = -(mass_b / total_mass) * delta_v
        delta_v_b = (mass_a / total_mass) * delta_v
        return delta_v_a, delta_v_b

    @staticmethod
    def _handoff_state(state, parent_rocket, child_rocket, delta_v):
        """Transform a flight's ending state into a child's
        initial_solution.

        Flight's state vector tracks the CDM of its own rocket
        configuration, so this converts between parent and child CDM:
        position gets the body-frame offset rotated into the inertial
        frame; velocity additionally picks up omega x offset and this
        child's share of separation_delta_v (both along the stack axis,
        in the body frame). Quaternion, angular velocity and time are
        unchanged - same body frame, same instant, absolute mission time.
        """
        t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3 = state
        position = Vector([x, y, z])
        velocity = Vector([vx, vy, vz])
        omega = Vector([w1, w2, w3])
        rotation = Matrix.transformation((e0, e1, e2, e3))

        offset = (
            child_rocket.center_of_dry_mass_position
            - parent_rocket.center_of_dry_mass_position
        )
        d = Vector([0, 0, offset])

        child_position = position + rotation @ d
        body_frame_velocity_delta = omega.cross(d) + Vector([0, 0, delta_v])
        child_velocity = velocity + rotation @ body_frame_velocity_delta

        return [
            t,
            *child_position,
            *child_velocity,
            e0,
            e1,
            e2,
            e3,
            w1,
            w2,
            w3,
        ]

    @staticmethod
    def _shift_motor_ignition(active_stages, ignition_time):
        """``active_stages`` with its bottom stage's motor time-shifted
        so it ignites at ``ignition_time`` (absolute mission time)
        instead of its own local t=0.

        Returns a new active_stages tuple - the bottom stage replaced by
        a shifted copy, everything else unchanged. Callers must use this
        returned tuple for the rest of that configuration's life (not
        the original active_stages), or a later recomposition (e.g. a
        deployable ejecting afterwards) would silently lose the shift.
        """
        bottom = active_stages[0]
        motor = deepcopy(bottom.rocket.motor)
        for attr_name in _MOTOR_TIME_FUNCTIONS:
            original = getattr(motor, attr_name)
            setattr(
                motor, attr_name, Function(Mission._shifted(original, ignition_time))
            )
        motor.burn_time = (
            motor.burn_time[0] + ignition_time,
            motor.burn_time[1] + ignition_time,
        )
        motor.burn_start_time += ignition_time
        motor.burn_out_time += ignition_time

        shifted_rocket = deepcopy(bottom.rocket)
        shifted_rocket.add_motor(motor, bottom.rocket.motor_position)
        shifted_stage = Stage(
            name=bottom.name,
            rocket=shifted_rocket,
            separation=bottom.separation,
            separation_delta_v=bottom.separation_delta_v,
            ignition=bottom.ignition,
            ignition_delay=bottom.ignition_delay,
            length=bottom.length,
        )
        return (shifted_stage,) + active_stages[1:]

    @staticmethod
    def _shifted(function, offset):
        return lambda t: function(t - offset)


def _configuration_name(active_stages, carried_deployables):
    return "+".join(body.name for body in (*active_stages, *carried_deployables))
