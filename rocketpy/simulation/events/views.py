"""The views an event or controller function reads the flight through:
``context.controls`` (the control inputs, written by controllers),
``context.outputs`` (the values events share with each other) and
``context.sensors`` (the latest sensor readings, read-only). A controllable,
an event or a sensor is found by its name as given,
``context.controls["Air Brakes"]``, or by that name in snake_case as an
attribute, ``context.controls.air_brakes``, and its values are then read by
attribute, such as ``context.controls["Air Brakes"].deployment_level`` or
``context.sensors.accelerometer.az``. After the flight the same values are
``flight.controls``, ``flight.event_outputs``, ``flight.sensor_history`` and
``flight.records``.
"""

from rocketpy.tools import snake_case_key, unique_names


class ChannelsView:
    """Groups of named values, found by name or by attribute.

    There is one group per controllable, event or sensor, and each value in a
    group is a channel: ``view["Canards"].fin_1`` or ``view.canards.fin_1``.
    This is the base of ``context.controls``, ``context.outputs`` and
    ``context.sensors``.

    A subclass gives:

    - the groups and their channels by key, the name in snake_case
      (:meth:`_groups`);
    - the names as given (:meth:`_names`);
    - how a channel is read (:meth:`_read`) and, if it can be, written
      (:meth:`_write`);
    - the words of the error messages, which list what exists so a misspelled
      name is easy to fix.
    """

    __slots__ = ()

    # Error messages, filled with the names involved
    _unknown_name = "There is nothing named {name!r}. The names are: {names}."
    _unknown_key = "There is nothing named '{key}'. The names are: {keys}."
    _unknown_channel = (
        "'{key}' has no channel '{channel}'. Its channels are: {channels}."
    )
    _key_written = "Write a channel, such as `{key}.<channel> = ...`."
    _read_only = "These values are read-only."

    def _groups(self):
        """Return each key with the tuple of its channels, in order."""
        raise NotImplementedError

    def _names(self):
        """Return each name as given with its key, in order."""
        raise NotImplementedError

    def _read(self, key, channel):
        """Return the value of one existing channel."""
        raise NotImplementedError

    def _write(self, key, channel, value):
        """Write one existing channel."""
        raise AttributeError(self._read_only)

    def _check(self, key, channel):
        """Raise an ``AttributeError`` with a list of what exists if
        ``channel`` is not a channel of ``key``."""
        channels = self._groups()[key]
        if channel not in channels:
            raise AttributeError(
                self._unknown_channel.format(
                    key=key, channel=channel, channels=", ".join(channels)
                )
            )

    def _key_of(self, name):
        """Return the key of the group ``view[name]`` gives."""
        names = self._names()
        if isinstance(name, str) and name in names:
            return names[name]
        raise KeyError(
            self._unknown_name.format(
                name=name, names=", ".join(map(repr, names)) or "none"
            )
        )

    def __getitem__(self, name):
        return _KeyView(self, self._key_of(name))

    def __setitem__(self, name, value):
        raise TypeError(self._key_written.format(key=self._key_of(name)))

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())

    def __contains__(self, name):
        return isinstance(name, str) and name in self._names()

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        groups = self._groups()
        if key not in groups:
            raise AttributeError(
                self._unknown_key.format(key=key, keys=", ".join(groups) or "none")
            )
        return _KeyView(self, key)

    def __setattr__(self, key, value):
        raise AttributeError(self._key_written.format(key=key))

    def __dir__(self):
        return list(self._groups())

    def __repr__(self):
        groups = {name: repr(self[name]) for name in self}
        return f"{type(self).__name__}({groups})"


class _KeyView:
    """The channels of one controllable, event or sensor of a
    :class:`ChannelsView`, read and written by attribute,
    ``context.controls.canards.fin_1``, or by name,
    ``context.controls.canards["fin_1"]``."""

    __slots__ = ("_view", "_key")

    def __init__(self, view, key):
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "_key", key)

    def _get(self, channel):
        self._view._check(self._key, channel)
        return self._view._read(self._key, channel)

    def __getattr__(self, channel):
        if channel.startswith("_"):
            raise AttributeError(channel)
        return self._get(channel)

    def __setattr__(self, channel, value):
        self._view._check(self._key, channel)
        self._view._write(self._key, channel, value)

    # The same by name, for a name built in a loop: ``controls.canards[fin]``
    def __getitem__(self, channel):
        # Not through the attribute, so a channel named as a method of this
        # class, ``update``, is still read here
        try:
            return self._get(channel)
        except AttributeError as error:
            raise KeyError(str(error)) from None

    def __setitem__(self, channel, value):
        try:
            setattr(self, channel, value)
        except AttributeError as error:
            raise KeyError(str(error)) from None

    def update(self, values):
        """Write several channels at once from a dictionary of name to value,
        such as the one :meth:`rocketpy.ControlFins.deflections` returns:
        ``context.controls.canards.update(canards.deflections(roll=0.05))``."""
        for channel, value in values.items():
            self[channel] = value

    def __iter__(self):
        return iter(self._view._groups()[self._key])

    def __dir__(self):
        return list(self._view._groups()[self._key])

    def __repr__(self):
        values = {channel: self[channel] for channel in self}
        return f"{self._key}{values}"


class ControlsView(ChannelsView):
    """The flight's control inputs as a controller sees them: ``context.controls``
    in a sampled controller or event, and ``controls`` in a continuous law.

    A controllable is found by its name in snake_case as an attribute, by its
    name as given, or by the controllable itself, as returned by
    ``rocket.add_air_brakes`` or given to ``rocket.add_surfaces``. Its
    controls are then read and written by attribute, or by name when the name
    is built in a loop. These are all the same:

    .. code-block:: python

        controls.canards.fin_1 = 0.05
        controls["Canards"].fin_1 = 0.05
        controls[canards].fin_1 = 0.05
        controls.canards["fin_1"] = 0.05

    Looping over it gives the names of the controllables, and ``"Canards" in
    controls`` tells whether there is one. Writing a control that does not
    exist raises an error that lists the ones that do.

    Parameters
    ----------
    controls : dict
        The control inputs, ``"<key>.<control>"`` to value. Written in place.
        A name without a dot is not a control input and is not shown: the
        dictionary a continuous law reads also holds the states of the
        controllables, by their names.
    controllables : sequence, optional
        The rocket's controllables, which give the names. A key with no
        controllable among them is found by the key itself. Default: none.
    """

    __slots__ = ("_controls", "_controllables", "_cache")

    _unknown_name = (
        "No controllable of the rocket is named {name!r}. The controllables "
        "with controls are: {names}."
    )
    _unknown_key = (
        "No controllable of the rocket has controls named '{key}'. The "
        "controllables with controls are: {keys}."
    )
    _unknown_channel = (
        "'{key}' has no control named '{channel}'. Its controls are: {channels}."
    )
    _key_written = (
        "Write a control of a controllable, such as "
        "`controls.{key}.<control> = ...`, not the controllable itself."
    )

    def __init__(self, controls, controllables=()):
        # The flight's own dictionary: a write here is a write to the flight
        object.__setattr__(self, "_controls", controls)
        object.__setattr__(self, "_controllables", controllables)
        object.__setattr__(self, "_cache", None)

    def _point_at(self, controls):
        """Read and write ``controls`` from now on: another dictionary of the
        same control inputs. The equations of motion keep one view for their
        continuous laws and point it at the values of each evaluation, which
        is much faster than building a view every time."""
        object.__setattr__(self, "_controls", controls)

    def _groups(self):
        # The control inputs of a flight never change, so this is built once
        if self._cache is None:
            groups = {}
            for name in self._controls:
                key, dot, channel = name.partition(".")
                if dot:
                    groups.setdefault(key, []).append(channel)
            object.__setattr__(self, "_cache", groups)
        return self._cache

    def _names(self):
        by_key = {snake_case_key(c.name): c.name for c in self._controllables}
        return {by_key.get(key, key): key for key in self._groups()}

    def _key_of(self, name):
        if isinstance(name, str) or not hasattr(name, "controls"):
            return super()._key_of(name)
        # A controllable, recognised by shape so this module does not import
        # rocketpy.control
        key = snake_case_key(name.name)
        if key in self._groups():
            return key
        return super()._key_of(name.name)

    def _read(self, key, channel):
        return self._controls[f"{key}.{channel}"]

    def _write(self, key, channel, value):
        self._controls[f"{key}.{channel}"] = value


class OutputsView(ChannelsView):
    """The values the events of a flight share with each other, as an event
    reads and writes them: ``context.outputs``.

    An event that declared outputs is found by its name in snake_case as an
    attribute, by its name as given, or by the event itself. Its outputs are
    then read and written by attribute, or by name. These are all the same:

    .. code-block:: python

        outputs.guidance.pitch_setpoint
        outputs["Guidance"].pitch_setpoint
        outputs[guidance].pitch_setpoint
        outputs.guidance["pitch_setpoint"]

    Every event can read every output. Only the event that declared an output
    can write it. Looping over the view gives the names of the events with
    outputs.

    Parameters
    ----------
    events : dict
        The events with outputs, by their name in snake_case (see
        :func:`output_events`).
    writer : Event, optional
        The event being run, the only one whose outputs can be written.
        Default ``None``, a view that only reads.
    time : float, optional
        The time a write is stored with, in seconds.
    """

    __slots__ = ("_events", "_writer", "_time")

    _unknown_name = (
        "No event with outputs is named {name!r}. The events with outputs are: {names}."
    )
    _unknown_key = (
        "No event has outputs under the name '{key}'. The events with outputs "
        "are: {keys}."
    )
    _unknown_channel = (
        "'{key}' has no output named '{channel}'. Its outputs are: {channels}."
    )
    _key_written = (
        "Write an output of the event, such as `outputs.{key}.<output> = ...`, "
        "not the event itself."
    )

    def __init__(self, events, writer=None, time=None):
        object.__setattr__(self, "_events", events)
        object.__setattr__(self, "_writer", writer)
        object.__setattr__(self, "_time", time)

    def _groups(self):
        return {key: event.outputs for key, event in self._events.items()}

    def _names(self):
        return {event.name: key for key, event in self._events.items()}

    def _key_of(self, name):
        if not isinstance(name, str) and hasattr(name, "outputs"):
            # The event itself
            key = snake_case_key(name.name)
            if self._events.get(key) is name:
                return key
            name = name.name
        return super()._key_of(name)

    def _read(self, key, channel):
        return self._events[key]._outputs[channel]

    def _write(self, key, channel, value):
        event = self._events[key]
        if event is not self._writer:
            raise AttributeError(
                f"The output '{key}.{channel}' is written only by '{event.name}', "
                "the event that declared it. Other events read it."
            )
        event._write_output(self._time, channel, value)


def output_events(events):
    """Return the events of ``events`` that declared outputs, by their name in
    snake_case, which is how ``context.outputs`` finds them.

    Raises
    ------
    ValueError
        If two events with outputs have names that give the same key, since
        their outputs could not be told apart.
    """
    found = {}
    for event in events:
        if not event.outputs:
            continue
        key = snake_case_key(event.name)
        if key in found:
            raise ValueError(
                f"The events '{found[key].name}' and '{event.name}' both have "
                f"outputs, and their outputs would both be named "
                f"'{key}.<output>'. Give them different names."
            )
        found[key] = event
    return found


def sensor_positions(sensor_events):
    """Return each sensor position of a rocket by its key, as ``{key: (sensor,
    event)}``, and give each event the name its readings go by
    (``event._position_name``).

    The name is the sensor's own, and the key is that name in snake_case:
    ``"Nose IMU"`` and ``nose_imu``. A name used again, by another sensor or
    by the same sensor added in another position, is numbered from 2 in the
    order they were added: ``"Nose IMU 2"`` and ``nose_imu_2`` (see
    :func:`rocketpy.tools.unique_names`). Adding a sensor never renames the
    ones already there, so a controller reading them keeps working.

    Parameters
    ----------
    sensor_events : sequence
        The rocket's ``(sensor, event)`` pairs, in the order they were added.

    Returns
    -------
    dict
    """
    names = unique_names([sensor.name for sensor, _ in sensor_events])
    positions = {}
    for name, (sensor, event) in zip(names, sensor_events):
        event._position_name = name
        positions[snake_case_key(name)] = (sensor, event)
    return positions


class SensorsView(ChannelsView):
    """The latest reading of every channel of the rocket's sensors, as an
    event reads them: ``context.sensors``.

    A sensor is found by its name as given, or by that name in snake_case as
    an attribute. Its channels are then read by name, such as ``ax``, ``wz``,
    ``pressure`` or ``latitude``. These are the same:

    .. code-block:: python

        sensors["Accelerometer"].az
        sensors.accelerometer.az

    A name used again, by another sensor or by the same sensor in another
    position, is numbered from 2 in the order they were added:
    ``"Accelerometer"`` and ``"Accelerometer 2"``, read as ``accelerometer``
    and ``accelerometer_2``, each with its own readings. Looping over the view
    gives the names. Every sensor reads once on the pad when the flight
    starts, so there is always a reading. The readings cannot be written.

    Parameters
    ----------
    sensors : dict
        Key to ``(sensor, event)`` for each sensor position (see
        :func:`sensor_positions`).
    solution : Solution
        The flight's solution, which holds the readings.
    """

    __slots__ = ("_sensors", "_solution")

    _unknown_name = "The rocket has no sensor named {name!r}. Its sensors are: {names}."
    _unknown_key = "The rocket has no sensor named '{key}'. Its sensors are: {keys}."
    _unknown_channel = (
        "The sensor '{key}' has no channel named '{channel}'. Its channels are: "
        "{channels}."
    )
    _key_written = "Sensor readings are read-only."
    _read_only = "Sensor readings are read-only."

    def __init__(self, sensors, solution):
        object.__setattr__(self, "_sensors", sensors)
        object.__setattr__(self, "_solution", solution)

    def _groups(self):
        return {
            key: tuple(label for label, _ in sensor.channels)
            for key, (sensor, _) in self._sensors.items()
        }

    def _names(self):
        return {event._position_name: key for key, (_, event) in self._sensors.items()}

    def _read(self, key, channel):
        # The last value the solution holds, or None before the first reading
        latest = self._solution._latest_reading(
            self._sensors[key][1]._position_name, channel
        )
        return None if latest is None else latest[1]
