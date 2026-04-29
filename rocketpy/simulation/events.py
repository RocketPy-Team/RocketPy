import inspect
from typing import get_type_hints


class Event:
    """A class representing an event in the simulation."""

    # TODO: should "sensors" arg of the trigger function be a dictionary instead
    #  of a list? It would be more intuitive to access the sensors by name
    def __init__(self, trigger, action, name, event_context=None):
        """Initializes an Event object.

        Parameters
        ----------
        trigger : function
            A function that must return a boolean value. The event will be
            triggered when this function returns True. The function should be
            defined with the following signature: trigger(**kwargs), where
            kwargs is a dictionary containing the keys:

                - `"time"` (float): The current simulation time in seconds.
                - `"state"` (list): The state vector of the simulation, structured
                  as `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.
                - `"state_dot"` (list): The time derivative of the state vector,
                  structured as `[vx, vy, vz, ax, ay, az, e0_dot, e1_dot, e2_dot, e3_dot, wx_dot, wy_dot, wz_dot]`.
                - `"sampling_rate"` (float or None): The sampling rate of the
                  event, in seconds. If None, the event will be checked for
                  triggering at every time step of the simulation. If a float
                  value is provided, the event will only be checked for
                  triggering at that specific time interval.
                - `"sensors"` (list): A list of sensors that are attached to the
                  rocket. The most recent measurements of the sensors are provided
                  with the ``sensor.measurement`` attribute. The sensors are
                  listed in the same order as they are added to the rocket.
                - `"environment"` (Environment): The current environment object
                  assigned to the simulation.
                - `"rocket"` (Rocket): The current rocket object assigned to the
                  simulation.
                - `"phase"` (FlightPhase): The current flight phase object.
                - `"phase_index"` (int): The index of the current flight phase.
                - `"node_index"` (int): The index of the current node in the
                  current flight phase.
                - Any additional custom key-value pairs provided via the
                  `event_context` parameter (see below).

        action : function
            A function that will be executed when the event is triggered. The
            function should be defined with the following signature:
            action(**kwargs), where kwargs is a dictionary containing the same
            keys as the trigger function. The action function can also modify
            the state of the simulation by returning a dictionary with the keys:
                - `"state"` (list): A new state vector to replace the current state
                  vector. The structure of the state vector is the same as the
                  one provided in the trigger function.
                - `"disable_event"` (bool): If True, the event will not be
                  checked for triggering again after being triggered, making
                  it a one-time event. Defaults to True.
                - `"new_events"` (list): A list of new Event objects to be added
                  to the simulation when the event is triggered. This can be
                  used to create events that spawn new events when they are
                  triggered, such as a parachute deployment event that spawns
                  a new event to check for the parachute deployment after a
                  certain time delay.
                - `"remove_events"` (list): A list of Event objects to be
                  removed from the simulation when the event is triggered. This
                  can be used to create events that remove other events when
                  they are triggered, such as a parachute deployment event that
                  removes the apogee event when it is triggered.
                - Any other key-value pairs defined in `event_context` will
                  also be included. These allow you to maintain custom state or
                  counters across multiple trigger and action calls. Use cases
                  include: tracking the number of times an event has been triggered
                  (e.g., `{"trigger_count": 0}`), recording the time of the last
                  trigger (e.g., `{"last_trigger_time": None}`), or any other
                  custom data your trigger/action functions need to share state.

                  Example: If you initialize the event with
                  `event_context={"trigger_count": 0}`, your trigger and action
                  functions will receive `trigger_count=0` in their kwargs dict.
                  You can then update this value in the action function by
                  including it in the returned dictionary (e.g.,
                  `{"trigger_count": 1}`), and it will be passed to subsequent
                  trigger/action calls.

        name : str
            A name for the event, used for identification purposes.
        event_context : dict, optional
            A dictionary of custom key-value pairs that will be passed to the
            trigger and action functions. This allows you to initialize and
            maintain custom state that persists across multiple trigger/action
            calls. For example, `event_context={"trigger_count": 0,
            "last_trigger_time": None}` can be used to track event state.
            When the action function returns a dictionary with updated values
            (e.g., `{"trigger_count": 1}`), those values persist and are
            passed to subsequent calls. Defaults to an empty dictionary if not
            provided.
        """
        self.name = name
        self.trigger = self.__verify_trigger(trigger)
        self.action = self.__verify_action(action)
        self.event_context = event_context if event_context is not None else {}
        # TODO: implement tracking for whether this event is currently enabled
        # or disabled. The disable_event flag from the action return value should
        # control whether this event continues to be checked for triggering.

    def __verify_trigger(self, trigger):
        """Verifies that the trigger function is valid.

        Parameters
        ----------
        trigger : function
            The trigger function to be verified.

        Returns
        -------
        function
            The verified trigger function.

        Raises
        ------
        ValueError
            If the trigger function does not have the correct signature or does not return a boolean value
            (at least if not declared or annotated).
        """
        # verify if the trigger function accepts only **kwargs arguments
        s = inspect.signature(trigger)
        if any(p.kind != inspect.Parameter.VAR_KEYWORD for p in s.parameters.values()):
            raise ValueError(
                f"The Trigger function of the {self.name} event must accept only keyword arguments. def {trigger.__name__}(**kwargs) -> bool:"
            )
        # Verify if the return type annotation is bool.
        # Since is not possible to know for sure if the user is actually returning a bool value,
        # we enforce bool annotation to motivate users to actually return bool values.
        return_annotation = get_type_hints(trigger).get("return", None)
        if return_annotation is not bool:
            raise ValueError(
                f"The Trigger function of the {self.name} event must return a boolean value and must be annotated with '-> bool' for type checking.\n"
                f"def {trigger.__name__}(**kwargs) -> bool:"
            )
        return trigger

    def __verify_action(self, action):
        """Verifies that the action function is valid.

        Parameters
        ----------
        action : function
            The action function to be verified.

        Returns
        -------
        function
            The verified action function.

        Raises
        ------
        ValueError
            If the action function does not have the correct signature or does not return a valid type.
        """
        # verify if the action function accepts only **kwargs arguments
        s = inspect.signature(action)
        if any(p.kind != inspect.Parameter.VAR_KEYWORD for p in s.parameters.values()):
            raise ValueError(
                f"The Action function of the {self.name} event must accept only keyword arguments. def {action.__name__}(**kwargs) -> None or dict:"
            )
        # verify if the return type annotation is None or dict
        # Since is not possible to know for sure if the user is actually returning None or a dict,
        # we enforce None or dict annotation to motivate users to actually return None or dict.
        return_annotation = get_type_hints(action).get("return", None)
        if return_annotation is not None and return_annotation is not (
            type(None) or dict
        ):
            raise ValueError(
                f"The Action function of the {self.name} event must return None or a dictionary and must be annotated with '-> None' or '-> dict' for type checking.\n"
                f"def {action.__name__}(**kwargs) -> None or dict:"
            )
        return action

    def __repr__(self):
        # TODO: Implement a more informative string representation of the Event object.
        pass

    def __str__(self):
        # TODO: Implement a more informative string representation of the Event object.
        pass

    def __call__(self, *args, **kwds):
        # TODO: Implement the main event logic:
        # 1. Construct kwargs dict with:
        #    - "time": current simulation time
        #    - "state": current state vector [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
        #    - "state_dot": time derivative of state
        #    - "sampling_rate": event sampling rate (None for every step)
        #    - "sensors": list of sensor objects with latest measurements
        #    - "environment": current Environment object
        #    - "rocket": current Rocket object
        #    - "phase": current FlightPhase object
        #    - "phase_index": index of current flight phase
        #    - "node_index": index of current node in phase
        #    - All key-value pairs from self.event_context
        # 2. Call self.trigger(**kwargs) and check return value is boolean
        # 3. If trigger returns True:
        #    a. Call self.action(**kwargs)
        #    b. If action returns a dict, process return values:
        #       - "state": update simulation state if provided
        #       - "disable_event": set internal flag to disable this event from
        #         being triggered again (default True)
        #       - "new_events": add new Event objects to the simulation
        #       - "remove_events": remove Event objects from the simulation
        #       - Any other keys: update self.event_context for next trigger call
        # 4. Log trigger result and current simulation time for debugging
        # TODO: handle sampling_rate: if not None, only check trigger at
        # specified time intervals, not at every time step
        pass


# TODO: add a parameter to the Event class that specify whether the event should
# be triggered only once, or if it can be triggered multiple times. Also, add a
# way to stop the event from continuously triggering on command inside the action
# function, such as a "disable" method that can be called inside the action
# function to prevent the event from being triggered again.

# TODO: add a parameter to the Event class that specify whether the event should
# be a discrete event, meaning that it should only be checked for triggering at
# specific time intervals (e.g. every 0.1 seconds) instead of at every time step
# of the simulation. This would be useful for parachute events. This should be
# done by adding a "sampling_rate" parameter to the Event class, that is none by
# default (meaning that the event is checked at every time step), but if it is
# set to a float value, the event will only be checked for triggering at that
# specific time interval. The flight class should be able to differentiate
# between the discrete and continuous events (we will handle this later)


# FOR STANO:
# TODO: Implement Event orchestration at the Flight/Simulation level:
# - Flight or an event manager class should maintain a list of active events
# - At each simulation step, iterate through enabled events and call them with
#   the current simulation state (time, state, state_dot, sensors, etc.)
# - Collect return values from events and apply state changes, add/remove events,
#   and update event_context values for subsequent calls
# - Respect the disable_event flag and sampling_rate to control when events
#   are checked for triggering
# - Handle the sampling_rate logic: only check events at the specified intervals,
#   not at every simulation time step
