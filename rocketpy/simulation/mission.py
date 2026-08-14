"""Mission – orchestrates one Flight per vehicle configuration."""

from copy import deepcopy

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket.multistage import MultiStageRocket, Stage
from rocketpy.simulation.flight import Flight

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

    Walks the vehicle's separation/ignition timing, runs one Flight per
    vehicle configuration with correct state handoff between them, and
    groups the resulting Flight objects per physical body along with a
    global event timeline.

    Implemented so far: a single-stage vehicle with no deployables
    (degenerates to one Flight), and a two-stage vehicle with no
    deployables. Deployable ejection and more than two stages are later
    commits.

    Separation and ignition timing are deterministic, computed ahead of
    time from motor burn_time and the delays given on each Stage - there
    is no generic mid-flight trigger/event solver (Flight only exposes
    max_time and terminate_on_apogee for early termination). A Stage's
    ``separation`` is therefore a plain float: the delay, in seconds,
    after that stage's own motor burns out. ``ignition`` (an
    event-triggered alternative to ``ignition_delay``) is not supported
    yet.

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
        Body name -> list of Flight, in time order. The full stack's
        flight appears under every body that was aboard it.
    timeline : list of (float, str)
        (time, event_name) tuples sorted by time. Canonical names so
        far: "ignition:<stage>", "liftoff", "rail_departure",
        "separation:<stage>", "impact:<stage>".
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
        self._simulate()

    def _simulate(self):
        if self.vehicle.deployables:
            raise NotImplementedError(
                "Mission does not support deployables yet."
            )
        if len(self.vehicle.stages) == 1:
            self._simulate_single_stage()
        elif len(self.vehicle.stages) == 2:
            self._simulate_two_stage()
        else:
            raise NotImplementedError(
                "Mission currently supports at most two stages."
            )
        self.timeline.sort(key=lambda entry: entry[0])

    def _simulate_single_stage(self):
        stage = self.vehicle.stages[0]
        rocket = self.vehicle.flight_rocket(active_stages=(stage,))

        self.timeline.append((0.0, f"ignition:{stage.name}"))
        self.timeline.append((0.0, "liftoff"))

        flight = self._run_flight(rocket)

        self.timeline.append((flight.out_of_rail_time, "rail_departure"))
        self.timeline.append((flight.t_final, f"impact:{stage.name}"))

        self.flights[stage.name] = [flight]

    def _simulate_two_stage(self):
        booster, sustainer = self.vehicle.stages
        if booster.separation is None:
            raise ValueError(
                "booster.separation must be set (delay in seconds after "
                "burnout) for a two-stage Mission."
            )
        if sustainer.ignition is not None:
            raise NotImplementedError(
                "Event-triggered ignition is not supported yet; use "
                "ignition_delay (a deterministic float delay) instead."
            )

        stack_rocket, stack_flight, separation_time = self._run_stack_phase(
            booster, sustainer
        )
        self.flights[booster.name] = [stack_flight]
        self.flights[sustainer.name] = [stack_flight]

        ending_state = stack_flight.solution[-1]
        booster_delta_v, sustainer_delta_v = self._split_separation_delta_v(
            booster, sustainer
        )
        self._run_booster_phase(
            booster, stack_rocket, ending_state, booster_delta_v
        )
        self._run_sustainer_phase(
            sustainer, stack_rocket, ending_state, sustainer_delta_v, separation_time
        )

    def _run_stack_phase(self, booster, sustainer):
        """Full stack, booster firing, from the rail to separation."""
        stack_rocket = self.vehicle.flight_rocket(active_stages=(booster, sustainer))
        separation_time = booster.burn_out_time + booster.separation

        self.timeline.append((0.0, f"ignition:{booster.name}"))
        self.timeline.append((0.0, "liftoff"))

        stack_flight = self._run_flight(stack_rocket, max_time=separation_time)
        self.timeline.append((stack_flight.out_of_rail_time, "rail_departure"))
        self.timeline.append((separation_time, f"separation:{booster.name}"))

        return stack_rocket, stack_flight, separation_time

    def _run_booster_phase(self, booster, stack_rocket, ending_state, delta_v):
        """Spent booster, falling away on its own from the separation
        state onward.
        """
        booster_rocket = self.vehicle.flight_rocket(active_stages=(booster,))
        initial_solution = self._handoff_state(
            ending_state, stack_rocket, booster_rocket, delta_v
        )
        booster_flight = self._run_flight(
            booster_rocket, initial_solution=initial_solution
        )
        self.timeline.append((booster_flight.t_final, f"impact:{booster.name}"))
        self.flights[booster.name].append(booster_flight)

    def _run_sustainer_phase(
        self, sustainer, stack_rocket, ending_state, delta_v, separation_time
    ):
        """Sustainer, igniting after ignition_delay and continuing on its
        own from the separation state onward.
        """
        ignition_time = separation_time + sustainer.ignition_delay
        self.timeline.append((ignition_time, f"ignition:{sustainer.name}"))

        sustainer_rocket = self._shift_motor_ignition(sustainer, ignition_time)
        initial_solution = self._handoff_state(
            ending_state, stack_rocket, sustainer_rocket, delta_v
        )
        sustainer_flight = self._run_flight(
            sustainer_rocket, initial_solution=initial_solution
        )
        self.timeline.append((sustainer_flight.t_final, f"impact:{sustainer.name}"))
        self.flights[sustainer.name].append(sustainer_flight)

    def _run_flight(self, rocket, initial_solution=None, max_time=None):
        """Run one Flight in absolute mission time."""
        return Flight(
            rocket=rocket,
            environment=self.environment,
            rail_length=self.rail_length,
            inclination=self.inclination,
            heading=self.heading,
            initial_solution=initial_solution,
            max_time=max_time if max_time is not None else self.max_time,
            rtol=self.rtol,
            atol=self.atol,
            time_overshoot=self.time_overshoot,
            ode_solver=self.ode_solver,
            verbose=self.verbose,
        )

    @staticmethod
    def _split_separation_delta_v(booster, sustainer):
        """Momentum-conserving split of booster.separation_delta_v between
        the two children at the separation instant: the booster is spent
        (dry_mass), the sustainer hasn't ignited yet (full total_mass at
        its own t=0).
        """
        booster_mass_after = booster.rocket.dry_mass
        sustainer_mass_after = sustainer.rocket.total_mass(0)
        total_mass_after = booster_mass_after + sustainer_mass_after
        delta_v = booster.separation_delta_v
        booster_delta_v = -(sustainer_mass_after / total_mass_after) * delta_v
        sustainer_delta_v = (booster_mass_after / total_mass_after) * delta_v
        return booster_delta_v, sustainer_delta_v

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

    def _shift_motor_ignition(self, stage, ignition_time):
        """Rocket for ``stage`` flying alone, with its motor's own time
        origin shifted so it ignites at ``ignition_time`` (absolute
        mission time) instead of its own local t=0.
        """
        motor = deepcopy(stage.rocket.motor)
        for attr_name in _MOTOR_TIME_FUNCTIONS:
            original = getattr(motor, attr_name)
            setattr(motor, attr_name, Function(self._shifted(original, ignition_time)))
        motor.burn_time = (
            motor.burn_time[0] + ignition_time,
            motor.burn_time[1] + ignition_time,
        )
        motor.burn_start_time += ignition_time
        motor.burn_out_time += ignition_time

        shifted_rocket = deepcopy(stage.rocket)
        shifted_rocket.add_motor(motor, stage.rocket.motor_position)
        shifted_stage = Stage(name=stage.name, rocket=shifted_rocket)
        return self.vehicle.flight_rocket(active_stages=(shifted_stage,))

    @staticmethod
    def _shifted(function, offset):
        return lambda t: function(t - offset)
