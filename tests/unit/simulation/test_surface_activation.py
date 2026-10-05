from types import SimpleNamespace

import pytest

from rocketpy import GenericSurface
from rocketpy.simulation.events import Event
from rocketpy.simulation.events.commands import Commands
from rocketpy.simulation.events.event_execution import apply_event_commands


def _surface(name, **kwargs):
    return GenericSurface(1.0, 1.0, {}, name=name, **kwargs)


def _flight(*surfaces, burn_out_time=3.0):
    """A stand-in flight of a rocket holding the surfaces: only what
    activation reads, so no real Flight is built."""
    rocket = SimpleNamespace(
        aerodynamic_surfaces=[(surface, None) for surface in surfaces],
        motor=SimpleNamespace(burn_out_time=burn_out_time),
    )
    return SimpleNamespace(rocket=rocket, _surface_switches={})


def _switch(flight, time, **commands):
    """Run an event that switches surfaces at ``time``, through the usual
    command path. ``commands`` are ``on=[...]`` and ``off=[...]``."""
    event = Event(callback=lambda context: None, name="Switch")
    for surface in commands.get("off", ()):
        event.commands.deactivate_surface(surface)
    for surface in commands.get("on", ()):
        event.commands.activate_surface(surface)
    apply_event_commands(
        flight,
        event,
        event.commands,
        phase=SimpleNamespace(),
        phase_index=0,
        node_index=0,
        command_time=time,
    )


def test_a_surface_starting_switched_off_is_never_active():
    deployable = _surface("deployable", active=False)
    flight = _flight(deployable)

    assert deployable.is_active(0.0, flight) is False
    assert deployable.is_active(10.0, flight) is False


def test_switching_takes_effect_from_its_time_on():
    fixed = _surface("fixed")
    removable = _surface("removable")
    deployable = _surface("deployable", active=False)
    flight = _flight(fixed, removable, deployable)

    _switch(flight, 5.0, off=[removable], on=[deployable])

    assert fixed.is_active(8.0, flight) is True
    assert removable.is_active(5.0, flight) is False
    assert deployable.is_active(5.0, flight) is True


def test_earlier_times_keep_what_was_active_then():
    """The flight's outputs are worked out after it ends, at every stored time."""
    removable = _surface("removable")
    flight = _flight(removable)

    _switch(flight, 5.0, off=[removable])
    _switch(flight, 9.0, on=[removable])

    assert removable.is_active(4.999, flight) is True
    assert removable.is_active(5.0, flight) is False
    assert removable.is_active(8.999, flight) is False
    assert removable.is_active(9.0, flight) is True


def test_a_switched_on_surface_still_follows_its_motor_phase():
    base = _surface("base drag", active_during="power_off", active=False)
    flight = _flight(base, burn_out_time=3.0)

    _switch(flight, 1.0, on=[base])

    assert base.is_active(2.0, flight) is False  # on, but the motor still burns
    assert base.is_active(3.0, flight) is True


def test_switching_does_not_change_the_surface_itself():
    """Another flight of the same rocket starts from the surface's own setting."""
    removable = _surface("removable")
    first = _flight(removable)
    _switch(first, 5.0, off=[removable])

    assert removable.active is True
    assert removable.is_active(10.0, _flight(removable)) is True


def test_switching_a_surface_the_rocket_does_not_have_is_an_error():
    flight = _flight(_surface("fixed"))

    with pytest.raises(ValueError, match="stray.*not one of the rocket"):
        _switch(flight, 1.0, on=[_surface("stray")])


def test_surface_commands_are_queued_and_mark_a_trajectory_change():
    commands = Commands()
    removable, deployable = _surface("removable"), _surface("deployable")
    assert commands.changes_trajectory is False

    commands.deactivate_surface(removable)
    commands.activate_surface(deployable)

    assert commands.surface_switches == [(removable, False), (deployable, True)]
    # the forces change, so the solver has to restart from the switch
    assert commands.changes_trajectory is True

    commands.reset()

    assert commands.surface_switches == []
    assert commands.changes_trajectory is False
