"""Event-oriented concatenation of Flight phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocketpy.mathutils.flight_state import FlightState
from rocketpy.simulation.events.event import Event
from rocketpy.simulation.flight import Flight


@dataclass
class _Leg:
    flight: Flight
    start: float
    end: float
    offset: float


class Mission:
    """A sequence of Flights joined at explicit times or triggered events.

    This is useful for deployables and stage changes without splitting launch
    and orbital propagation into different simulation classes. Each leg keeps
    its own vehicle, Earth/Environment and Space domains while
    ``solution_array`` exposes one continuous mission timeline.
    """

    def __init__(self, flight: Flight):
        if not isinstance(flight, Flight):
            raise TypeError("Mission requires an initial Flight.")
        self.flights = [flight]
        self._legs = [_Leg(flight, flight.t_initial, flight.t_final, 0.0)]

    @staticmethod
    def _resolve_time(flight, at):
        if at is None or at == "end":
            return flight.t_final
        if isinstance(at, (int, float)):
            return float(at)
        if isinstance(at, Event):
            if not at.triggered_times:
                raise ValueError(f"Event {at.name!r} did not trigger in this Flight.")
            return float(at.triggered_times[0])
        if callable(at):
            return float(at(flight))
        if isinstance(at, str):
            if at.lower() == "apogee":
                return float(flight.apogee_time)
            matches = [
                event for event in flight.events if event.name.lower() == at.lower()
            ]
            if not matches:
                raise ValueError(f"No Flight event named {at!r}.")
            return Mission._resolve_time(flight, matches[0])
        raise TypeError("at must be a time, Event, event name, callable, or 'end'.")

    def concatenate(
        self,
        vehicle,
        *,
        at="end",
        environment=None,
        space=None,
        duration=None,
        forces=None,
        **flight_kwargs,
    ) -> Flight:
        """Append a phase beginning at a previous phase event or time.

        Passing a different vehicle naturally represents staging or deployment.
        The returned Flight is also appended to ``flights``.
        """
        previous_leg = self._legs[-1]
        previous = previous_leg.flight
        join_time = self._resolve_time(previous, at)
        if not previous.t_initial <= join_time <= previous.t_final:
            raise ValueError("Concatenation time is outside the previous Flight.")
        previous_leg.end = join_time
        state = previous.state_at_time(join_time)
        mission_elapsed = previous_leg.offset + (join_time - previous_leg.start)
        state = FlightState.cartesian(
            epoch=state.epoch,
            position=state.position,
            velocity=state.velocity,
            quaternion=state.quaternion,
            angular_velocity=state.angular_velocity,
            frame=state.frame,
            elapsed_time=mission_elapsed,
        )
        next_flight = Flight.from_state(
            vehicle,
            previous.env if environment is None else environment,
            state,
            space=previous.space if space is None else space,
            duration=duration,
            forces=forces,
            **flight_kwargs,
        )
        self.flights.append(next_flight)
        self._legs.append(
            _Leg(
                next_flight, next_flight.t_initial, next_flight.t_final, mission_elapsed
            )
        )
        return next_flight

    @property
    def solution_array(self):
        """Concatenated state history with a continuous mission time column."""
        arrays = []
        for index, leg in enumerate(self._legs):
            values = leg.flight.solution_array
            values = values[
                (values[:, 0] >= leg.start) & (values[:, 0] <= leg.end)
            ].copy()
            values[:, 0] = leg.offset + values[:, 0] - leg.start
            if index and len(values):
                values = values[1:]
            arrays.append(values)
        nonempty = [values for values in arrays if len(values)]
        return np.vstack(nonempty) if nonempty else np.empty((0, 14))

    @property
    def time(self):
        return self.solution_array[:, 0]

    def state_at_time(self, time):
        """Return a typed state on the continuous mission timeline."""
        time = float(time)
        for leg in self._legs:
            leg_end = leg.offset + leg.end - leg.start
            if leg.offset <= time <= leg_end:
                local_time = leg.start + time - leg.offset
                state = leg.flight.state_at_time(local_time)
                return FlightState.cartesian(
                    epoch=state.epoch,
                    position=state.position,
                    velocity=state.velocity,
                    quaternion=state.quaternion,
                    angular_velocity=state.angular_velocity,
                    frame=state.frame,
                    elapsed_time=time,
                )
        raise ValueError("Requested time is outside the Mission timeline.")

    @property
    def final_state(self):
        """Typed state at the end of the final concatenated leg."""
        return self.state_at_time(self.time[-1])
