.. _sil-parachute-ejection:

Software-in-the-Loop Parachute Ejection
=======================================

RocketPy can run a **Software-in-the-Loop (SIL)** recovery system by calling
your ejection-detection algorithm from the parachute trigger (and, when useful,
from a discrete :doc:`controller <controllers>`). During ``Flight``, RocketPy
samples pressure, height and the state vector at the parachute
``sampling_rate`` and asks your callable whether to fire the ejection charge.

This page shows a minimal SIL pattern: a small stateful detector that mimics a
flight-computer recovery algorithm, wired as a parachute ``trigger``. For a
full research-grade example that wraps a compiled C/C++ recovery stack
(``SisRec``) inside Monte Carlo runs, see the
`Valetudo Monte Carlo analysis <https://github.com/RocketPy-Team/RocketPaper/tree/main/Valetudo_Monte_Carlo>`_.

.. seealso::

   - :doc:`Parachute Triggers (Acceleration-Based) <parachute_triggers>`:
     trigger signatures, ``u_dot``, and sensor-aware callables.
   - :doc:`Controllers <controllers>`: discrete sampling of in-flight logic.
   - :ref:`Parachute Trigger Details <triggerdetails>`: callable
     ``(pressure, height, state)`` contract on ``Rocket.add_parachute``.

Why SIL for recovery
--------------------

Real recovery firmware rarely uses the ideal ``"apogee"`` or altitude string
triggers. It filters barometer (and often IMU) samples, keeps an internal
flight phase, and decides when to fire drogue or main charges. In SIL you:

1. Keep that detection logic in a callable (pure Python, or a thin wrapper
   around C/C++/Rust via SWIG, ctypes, cffi, or pybind11).
2. Attach the callable as ``trigger=...`` on ``Rocket.add_parachute``.
3. Let ``Flight`` drive the loop at a fixed ``sampling_rate`` (Hz), with
   optional ``noise`` on the pressure channel and ``lag`` for charge-to-open
   delay.

RocketPy then evaluates aerodynamics with the canopy after the lag, so you can
compare trigger times, inflation velocities and landing footprints against the
same algorithm that flies on the vehicle.

Simplified ejection detector
----------------------------

The following class is a **teaching stand-in** for a barometric apogee
detector. It is not Valetudo's ``SisRec``; it only illustrates the stateful
pattern: feed noisy pressure each sample, return a discrete flight state, and
map that state to a boolean trigger.

.. code-block:: python

    class SimpleBarometricEjectionDetector:
        """Minimal pressure-based apogee detector for SIL demos.

        States:
            0: armed / ascent
            1: apogee candidate (pressure rising while still high)
            2: drogue fire command
        """

        def __init__(self, min_ascent_samples=20, pressure_eps=20.0):
            self.min_ascent_samples = min_ascent_samples
            self.pressure_eps = pressure_eps  # Pa
            self.reset()

        def reset(self):
            self.samples = 0
            self.min_pressure = None
            self.state = 0

        def update(self, pressure_pa):
            """Ingest one pressure sample [Pa]; return internal state."""
            self.samples += 1
            if self.min_pressure is None or pressure_pa < self.min_pressure:
                self.min_pressure = pressure_pa
                if self.state == 1:
                    self.state = 0
                return self.state

            # Pressure rising relative to the running minimum → descending.
            if (
                self.samples >= self.min_ascent_samples
                and pressure_pa > self.min_pressure + self.pressure_eps
            ):
                self.state = 1 if self.state == 0 else 2
            return self.state


Wire the detector as a parachute trigger
----------------------------------------

A parachute trigger receives freestream pressure (with the parachute
``noise`` model applied), height AGL, and the state vector. Return ``True``
exactly when your algorithm commands the charge.

.. code-block:: python

    from rocketpy import Environment, SolidMotor, Rocket, Flight

    env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
    env.set_atmospheric_model(type="standard_atmosphere")

    # Build motor + rocket as usual (Calisto / your vehicle).
    # motor = SolidMotor(...)
    # rocket = Rocket(...)
    # rocket.add_motor(motor, position=...)
    # rocket.add_nose(...); rocket.add_trapezoidal_fins(...); ...

    drogue_detector = SimpleBarometricEjectionDetector(
        min_ascent_samples=25,
        pressure_eps=30.0,
    )

    def drogue_sil_trigger(pressure, height, state_vector):
        # Optional guards: ignore rail / early flight.
        if height < 50.0:
            return False
        state = drogue_detector.update(pressure)
        return state == 2  # fire when detector reaches "drogue command"

    rocket.add_parachute(
        name="Drogue",
        cd_s=1.0,
        trigger=drogue_sil_trigger,
        sampling_rate=100,  # Hz; match your flight computer loop rate
        lag=1.5,  # seconds between fire command and full open
        noise=(0, 8.3, 0.5),  # (mean, std, time-correlation) on pressure [Pa]
    )


    def main_sil_trigger(pressure, height, state_vector):
        # Main: descending and below a deployment altitude.
        vz = state_vector[5]
        return vz < -1.0 and height < 800.0

    rocket.add_parachute(
        name="Main",
        cd_s=10.0,
        trigger=main_sil_trigger,
        sampling_rate=100,
        lag=0.5,
        noise=(0, 8.3, 0.5),
    )

    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        # Stop the integrator on trigger sample times when using discrete rates:
        time_overshoot=False,
    )

    print(flight.parachute_events)
    flight.info()

.. tip::

   Reset detector state (``drogue_detector.reset()``) before every new
   ``Flight`` if you reuse the same detector instance across Monte Carlo
   samples or parameter sweeps.

.. note::

   Set ``time_overshoot=False`` when the SIL loop must see samples at exactly
   ``1 / sampling_rate``. The same rule applies to discrete
   :doc:`controllers <controllers>`.

Dual path: trigger vs controller callback
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mechanism
     - Use when
   * - Parachute ``trigger``
     - The algorithm's output is "fire this canopy now". This is the usual SIL
       recovery path.
   * - Controller function (:doc:`controllers`)
     - You need a fixed-rate loop that updates actuators (air brakes, canards)
       or logs observed variables alongside recovery logic.
   * - Acceleration / sensor triggers (:doc:`parachute_triggers`)
     - The algorithm needs ``u_dot`` or attached IMU/barometer sensor objects
       instead of (or in addition to) the noisy pressure channel.

You can combine them: a controller may update shared state that a parachute
trigger reads, or a 5-argument trigger can read ``sensors`` and ``u_dot``
directly. Prefer a single source of truth for the fire decision so Monte Carlo
and hardware stay aligned.

Wrapping compiled recovery firmware
-----------------------------------

Issue #524's bonus path is hardware-faithful binaries, not a RocketPy API
change. Practical options:

1. **Compile the flight algorithm** to a shared library (``.so`` / ``.dylib`` /
   ``.dll``) with the same interface your avionics uses (for example
   ``update(pressure) -> state``).
2. **Expose it to Python** with SWIG (as Valetudo's ``SisRec`` does), ctypes,
   cffi, or pybind11.
3. **Call it from the trigger** exactly like the pure-Python detector above.

Sketch of a SWIG/ctypes-style wrapper (API names are illustrative):

.. code-block:: python

    # After building your recovery library and its Python wrapper:
    # import SisRec  # Valetudo-style SWIG module
    #
    # detector = SisRec.SisRecSt(main_pressure_ratio, mu)
    # detector.initializeBuffers(p0)
    # detector.enable()
    #
    # def drogue_trigger(pressure, height, state_vector):
    #     # SisRec historically expected pressure in bar-like units;
    #     # convert to match your firmware's input convention.
    #     return detector.update(pressure / 1e5) == detector.detectDrogue

Keep unit conversions and enable/reset semantics identical to the flight
computer. The `Valetudo Monte Carlo folder
<https://github.com/RocketPy-Team/RocketPaper/tree/main/Valetudo_Monte_Carlo>`_
shows this pattern end-to-end with ``SisRec.py`` / ``_SisRec.so`` driving
drogue deployment inside thousands of ``Flight`` runs.

Toward Hardware-in-the-Loop (optional)
--------------------------------------

SIL stops at "same software, simulated sensors." A **Hardware-in-the-Loop
(HIL)** step would stream RocketPy's pressure/IMU time history to the real
flight computer (serial, CAN, or a board-level harness) and feed the board's
fire discrete back into the simulation. RocketPy does not ship a HIL bridge
today; teams usually:

- export ``Flight`` solution / sensor histories, or
- call a thin I/O shim from a discrete controller / trigger that talks to the
  device under test.

Treat HIL as an integration project on top of the SIL trigger contract above,
not as a built-in ``Flight`` mode.

See also
--------

- `Valetudo Monte Carlo (RocketPaper) <https://github.com/RocketPy-Team/RocketPaper/tree/main/Valetudo_Monte_Carlo>`_
- :doc:`parachute_triggers`
- :doc:`controllers`
- :doc:`stochastic`
- :class:`rocketpy.Parachute`
- :class:`rocketpy.Flight`
