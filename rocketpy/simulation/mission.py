"""Mission – orchestrates one Flight per vehicle configuration."""

from rocketpy.rocket.multistage import MultiStageRocket
from rocketpy.simulation.flight import Flight


class Mission:
    """Simulate a complete mission: a vehicle that splits into multiple
    bodies, each simulated to impact (or until max_time).

    Walks the vehicle's separation/ejection events, runs one Flight per
    vehicle configuration with correct state handoff between them, and
    groups the resulting Flight objects per physical body along with a
    global event timeline.

    A single-stage vehicle with no deployables degenerates to a thin
    wrapper around one Flight - the only case implemented so far.
    Multi-stage orchestration (separation, ignition delays, state
    handoff) and deployable ejection are later commits.

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
        Body name -> list of Flight, in time order. Currently always
        exactly one body with exactly one Flight.
    timeline : list of (float, str)
        (time, event_name) tuples sorted by time. Canonical names so
        far: "ignition:<stage>", "liftoff", "impact:<stage>".
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
        """Run the mission.

        Only the degenerate case is implemented so far: a single stage,
        no deployables aboard, flown once from the rail to impact (or
        max_time). Anything else raises NotImplementedError until
        multi-stage orchestration lands.
        """
        if len(self.vehicle.stages) != 1 or self.vehicle.deployables:
            raise NotImplementedError(
                "Mission currently only supports a single-stage vehicle "
                "with no deployables; multi-stage orchestration is not "
                "yet implemented."
            )
        stage = self.vehicle.stages[0]
        rocket = self.vehicle.flight_rocket(active_stages=(stage,))

        self.timeline.append((0.0, f"ignition:{stage.name}"))
        self.timeline.append((0.0, "liftoff"))

        flight = Flight(
            rocket=rocket,
            environment=self.environment,
            rail_length=self.rail_length,
            inclination=self.inclination,
            heading=self.heading,
            max_time=self.max_time,
            rtol=self.rtol,
            atol=self.atol,
            time_overshoot=self.time_overshoot,
            ode_solver=self.ode_solver,
            verbose=self.verbose,
        )

        self.timeline.append((flight.out_of_rail_time, "rail_departure"))
        self.timeline.append((flight.t_final, f"impact:{stage.name}"))
        self.timeline.sort(key=lambda entry: entry[0])

        self.flights[stage.name] = [flight]
