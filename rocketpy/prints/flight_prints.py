import logging

logger = logging.getLogger(__name__)

"""rocketpy/prints/flight_prints.py

This module contains the _FlightPrints class, which is responsible for printing
flight information in a user-friendly manner.

Notes
-----
- This module does not have any external dependencies (avoid importing libraries).
- We assume that all flight information is valid, no validation checks is run.
- Avoid calculating values here, only print the values from the Flight class.
- The _FlightPrints is a private class, it is subjected to change without notice.
"""


class _FlightPrints:
    """Class that holds prints methods for Flight class.

    Attributes
    ----------
    _FlightPrints.flight : Flight
        Flight object that will be used for the prints.
    """

    def __init__(
        self,
        flight,
    ):
        """Initializes _FlightPrints class

        Parameters
        ----------
        flight: Flight
            Instance of the Flight class.

        Returns
        -------
        None
        """
        self.flight = flight

    def initial_conditions(self):
        """Prints initial conditions data available about the flight, including
        position, velocity, attitude, euler angles, angular velocity, and
        stability margin.

        Returns
        -------
        None
        """
        logger.info("\nInitial Conditions\n")

        t_init = self.flight.time[0]

        logger.info(f"Initial time: {t_init:.3f} s")
        logger.info(
            f"Position - x: {self.flight.x(t_init):.2f} m | "
            f"y: {self.flight.y(t_init):.2f} m | "
            f"z: {self.flight.z(t_init):.2f} m"
        )
        logger.info(
            f"Velocity - Vx: {self.flight.vx(t_init):.2f} m/s | "
            f"Vy: {self.flight.vy(t_init):.2f} m/s | "
            f"Vz: {self.flight.vz(t_init):.2f} m/s"
        )
        logger.info(
            f"Attitude (quaternions) - e0: {self.flight.e0(t_init):.3f} | "
            f"e1: {self.flight.e1(t_init):.3f} | "
            f"e2: {self.flight.e2(t_init):.3f} | "
            f"e3: {self.flight.e3(t_init):.3f}"
        )
        logger.info(
            f"Euler Angles - Spin φ : {self.flight.phi(t_init):.2f}° | "
            f"Nutation θ: {self.flight.theta(t_init):.2f}° | "
            f"Precession ψ: {self.flight.psi(t_init):.2f}°"
        )
        logger.info(
            f"Angular Velocity - ω1: {self.flight.w1(t_init):.2f} rad/s | "
            f"ω2: {self.flight.w2(t_init):.2f} rad/s | "
            f"ω3: {self.flight.w3(t_init):.2f} rad/s"
        )
        logger.info(f"Initial Stability Margin: {self.flight.initial_stability_margin:.3f} c")

    def numerical_integration_settings(self):
        """Prints out the numerical integration settings available about the
        flight, this includes the maximum allowed flight time, maximum allowed
        time step, and other settings.

        Returns
        -------
        None
        """
        logger.info("\nNumerical Integration Settings\n")
        logger.info(f"Maximum Allowed Flight Time: {self.flight.max_time:.2f} s")
        logger.info(f"Maximum Allowed Time Step: {self.flight.max_time_step:.2f} s")
        logger.info(f"Minimum Allowed Time Step: {self.flight.min_time_step:.2e} s")
        logger.info(f"Relative Error Tolerance: {self.flight.rtol}")
        logger.info(f"Absolute Error Tolerance: {self.flight.atol}")
        logger.info(f"Allow Event Overshoot: {self.flight.time_overshoot}")
        logger.info(f"Terminate Simulation on Apogee: {self.flight.terminate_on_apogee}")
        logger.info(f"Number of Time Steps Used: {len(self.flight.time_steps)}")
        logger.info(
            "Number of Derivative Functions Evaluation: "
            f"{sum(self.flight.function_evaluations_per_time_step)}"
        )
        avg_func_evals_per_step = sum(
            self.flight.function_evaluations_per_time_step
        ) / len(self.flight.time_steps)
        logger.info(
            f"Average Function Evaluations per Time Step: {avg_func_evals_per_step:.3f}"
        )

    def surface_wind_conditions(self):
        """Prints out the Surface Wind Conditions for the flight.

        Returns
        -------
        None
        """
        logger.info("\nSurface Wind Conditions\n")
        logger.info(f"Frontal Surface Wind Speed: {self.flight.frontal_surface_wind:.2f} m/s")
        logger.info(f"Lateral Surface Wind Speed: {self.flight.lateral_surface_wind:.2f} m/s")

    def launch_rail_conditions(self):
        """Prints out the Launch Rail Conditions available about the flight,
        including the length, inclination, and heading of the launch rail.

        Returns
        -------
        None
        """
        logger.info("\nLaunch Rail\n")
        logger.info(f"Launch Rail Length: {self.flight.rail_length} m")
        logger.info(f"Launch Rail Inclination: {self.flight.inclination:.2f}°")
        logger.info(f"Launch Rail Heading: {self.flight.heading:.2f}°")

    def out_of_rail_conditions(self):
        """Prints out the Out of Rail Conditions available about the flight,
        including the time, velocity, stability margin, angle of attack, thrust
        to weight ratio, and Reynolds number.

        Returns
        -------
        None
        """
        logger.info("\nRail Departure State\n")
        logger.info(f"Rail Departure Time: {self.flight.out_of_rail_time:.3f} s")
        logger.info(f"Rail Departure Velocity: {self.flight.out_of_rail_velocity:.3f} m/s")
        logger.info(
            "Rail Departure Stability Margin: "
            f"{self.flight.out_of_rail_stability_margin:.3f} c"
        )
        logger.info(
            "Rail Departure Angle of Attack: "
            f"{self.flight.angle_of_attack(self.flight.out_of_rail_time):.3f}°"
        )
        logger.info(
            "Rail Departure Thrust-Weight Ratio: "
            f"{self.flight.rocket.thrust_to_weight(self.flight.out_of_rail_time):.3f}"
        )
        logger.info(
            "Rail Departure Reynolds Number: "
            f"{self.flight.reynolds_number(self.flight.out_of_rail_time):.3e}"
        )

    def burn_out_conditions(self):
        """Prints out the Burn Out Conditions available about the flight,
        including the burn out time, altitude, speed, freestream speed, Mach
        number, and kinetic energy.

        Returns
        -------
        None
        """
        logger.info("\nBurn out State\n")
        logger.info(f"Burn out time: {self.flight.rocket.motor.burn_out_time:.3f} s")
        logger.info(
            "Altitude at burn out: "
            f"{self.flight.z(self.flight.rocket.motor.burn_out_time):.3f} m (ASL) | "
            f"{self.flight.altitude(self.flight.rocket.motor.burn_out_time):.3f} "
            "m (AGL)"
        )
        logger.info(
            "Rocket speed at burn out: "
            f"{self.flight.speed(self.flight.rocket.motor.burn_out_time):.3f} m/s"
        )

        stream_velocity = self.flight.free_stream_speed(
            self.flight.rocket.motor.burn_out_time
        )
        logger.info(f"Freestream velocity at burn out: {stream_velocity:.3f} m/s")

        logger.info(
            "Mach Number at burn out: "
            f"{self.flight.mach_number(self.flight.rocket.motor.burn_out_time):.3f}"
        )
        logger.info(
            "Kinetic energy at burn out: "
            f"{self.flight.kinetic_energy(self.flight.rocket.motor.burn_out_time):.3e} "
            "J"
        )

    def apogee_conditions(self):
        """Prints out the Apogee Conditions available about the flight,
        including the apogee time, altitude, freestream speed, latitude, and
        longitude.

        Returns
        -------
        None
        """
        logger.info("\nApogee State\n")
        logger.info(f"Apogee Time: {self.flight.apogee_time:.3f} s")
        logger.info(
            f"Apogee Altitude: {self.flight.apogee:.3f} m (ASL) | "
            f"{self.flight.altitude(self.flight.apogee_time):.3f} m (AGL)"
        )
        logger.info(f"Apogee Freestream Speed: {self.flight.apogee_freestream_speed:.3f} m/s")
        logger.info(f"Apogee X position: {self.flight.x(self.flight.apogee_time):.3f} m")
        logger.info(f"Apogee Y position: {self.flight.y(self.flight.apogee_time):.3f} m")
        logger.info(f"Apogee latitude: {self.flight.latitude(self.flight.apogee_time):.7f}°")
        logger.info(
            f"Apogee longitude: {self.flight.longitude(self.flight.apogee_time):.7f}°"
        )

    def events_registered(self):
        """Prints out the Events Registered available about the flight.

        Returns
        -------
        None
        """
        logger.info("\nParachute Events\n")
        if len(self.flight.parachute_events) == 0:
            logger.info("No Parachute Events Were Triggered.")
        for event in self.flight.parachute_events:
            trigger_time = event[0]
            parachute = event[1]
            open_time = trigger_time + parachute.lag
            speed = self.flight.free_stream_speed(open_time)
            altitude = self.flight.z(open_time)
            name = parachute.name.title()
            logger.info(f"Parachute: {name}")
            logger.info(f"\tEjection time: {trigger_time:.3f} s")
            logger.info(f"\tInflation time: {open_time:.3f} s")
            logger.info(f"\tFreestream speed at inflation: {speed:.3f} m/s")
            logger.info(
                f"\tAltitude at inflation: {altitude:.3f} m (ASL) | "
                f"{self.flight.altitude(open_time):.3f} m (AGL)"
            )

    def impact_conditions(self):
        """Prints out the Impact Conditions available about the flight.

        Returns
        -------
        None
        """
        if len(self.flight.impact_state) != 0:
            logger.info("\nImpact Conditions\n")
            logger.info(f"Time of impact: {self.flight.t_final:.3f} s")
            logger.info(f"X impact: {self.flight.x_impact:.3f} m")
            logger.info(f"Y impact: {self.flight.y_impact:.3f} m")
            logger.info(
                f"Altitude impact: {self.flight.z(self.flight.t_final):.3f} m (ASL) | "
                f"{self.flight.altitude(self.flight.t_final):.3f} m (AGL) "
            )
            logger.info(f"Latitude: {self.flight.latitude(self.flight.t_final):.7f}°")
            logger.info(f"Longitude: {self.flight.longitude(self.flight.t_final):.7f}°")
            logger.info(f"Vertical velocity at impact: {self.flight.impact_velocity:.3f} m/s")
            num_parachute_events = sum(
                1
                for event in self.flight.parachute_events
                if event[0] < self.flight.t_final
            )
            logger.info(
                f"Number of parachutes triggered until impact: {num_parachute_events}"
            )
        elif self.flight.terminate_on_apogee is False:
            logger.info("End of Simulation")
            t_final = self.flight.time[-1]
            logger.info(f"Time: {t_final:.3f} s")
            logger.info(
                f"Altitude: {self.flight.z(t_final)} m (ASL) | "
                f"{self.flight.altitude(t_final):.3f} m (AGL)"
            )
            logger.info(f"Latitude: {self.flight.latitude(t_final):.7f}°")
            logger.info(f"Longitude: {self.flight.longitude(t_final):.7f}°")

    def maximum_values(self):
        """Prints out the Maximum Values available about the flight.

        Returns
        -------
        None
        """
        logger.info("\nMaximum Values\n")
        logger.info(
            f"Maximum Speed: {self.flight.max_speed:.3f} m/s "
            f"at {self.flight.max_speed_time:.2f} s"
        )
        logger.info(
            f"Maximum Mach Number: {self.flight.max_mach_number:.3f} Mach "
            f"at {self.flight.max_mach_number_time:.2f} s"
        )
        logger.info(
            f"Maximum Reynolds Number: {self.flight.max_reynolds_number:.3e} "
            f"at {self.flight.max_reynolds_number_time:.2f} s"
        )
        logger.info(
            f"Maximum Dynamic Pressure: {self.flight.max_dynamic_pressure:.3e} Pa "
            f"at {self.flight.max_dynamic_pressure_time:.2f} s"
        )
        logger.info(
            "Maximum Acceleration During Motor Burn: "
            f"{self.flight.max_acceleration_power_on:.3f} m/s² "
            f"at {self.flight.max_acceleration_power_on_time:.2f} s"
        )
        logger.info(
            "Maximum Gs During Motor Burn: "
            f"{self.flight.max_acceleration_power_on / self.flight.env.standard_g:.3f} "
            f"g at {self.flight.max_acceleration_power_on_time:.2f} s"
        )
        logger.info(
            "Maximum Acceleration After Motor Burn: "
            f"{self.flight.max_acceleration_power_off:.3f} m/s² "
            f"at {self.flight.max_acceleration_power_off_time:.2f} s"
        )
        logger.info(
            "Maximum Gs After Motor Burn: "
            f"{self.flight.max_acceleration_power_off / self.flight.env.standard_g:.3f}"
            f" Gs at {self.flight.max_acceleration_power_off_time:.2f} s"
        )
        logger.info(
            f"Maximum Stability Margin: {self.flight.max_stability_margin:.3f} c "
            f"at {self.flight.max_stability_margin_time:.2f} s"
        )

        if (
            len(self.flight.rocket.rail_buttons) == 0
            or self.flight.out_of_rail_time_index == 0
        ):
            pass
        else:
            logger.info(
                "Maximum Upper Rail Button Normal Force: "
                f"{self.flight.max_rail_button1_normal_force:.3f} N"
            )
            logger.info(
                "Maximum Upper Rail Button Shear Force: "
                f"{self.flight.max_rail_button1_shear_force:.3f} N"
            )
            logger.info(
                "Maximum Lower Rail Button Normal Force: "
                f"{self.flight.max_rail_button2_normal_force:.3f} N"
            )
            logger.info(
                "Maximum Lower Rail Button Shear Force: "
                f"{self.flight.max_rail_button2_shear_force:.3f} N"
            )

    def rail_button_bending_moments(self):
        """Prints rail button bending moment data.

        Returns
        -------
        None
        """
        if (
            len(self.flight.rocket.rail_buttons) == 0
            or self.flight.out_of_rail_time_index == 0
        ):
            return

        # Check if button_height is defined
        rail_buttons_tuple = self.flight.rocket.rail_buttons[0]
        if rail_buttons_tuple.component.button_height is None:
            return

        logger.info("\nRail Button Bending Moments\n")
        logger.info(
            "Maximum Upper Rail Button Bending Moment: "
            f"{self.flight.max_rail_button1_bending_moment:.3f} N·m"
        )
        logger.info(
            "Maximum Lower Rail Button Bending Moment: "
            f"{self.flight.max_rail_button2_bending_moment:.3f} N·m"
        )

    def stability_margin(self):
        """Prints out the stability margins of the flight at different times.

        This method prints the following: Initial Stability Margin, Out of Rail
        Stability Margin, Maximum Stability Margin, and Minimum Stability Margin

        Each stability margin is printed along with the time it occurred.

        Notes
        -----
        The stability margin is typically measured in calibers (c), where 1
        caliber is the diameter of the rocket.
        """
        logger.info("\nStability Margin\n")
        logger.info(
            f"Initial Stability Margin: {self.flight.initial_stability_margin:.3f} c "
            f"at {self.flight.time[0]:.2f} s"
        )
        logger.info(
            "Out of Rail Stability Margin: "
            f"{self.flight.out_of_rail_stability_margin:.3f} c "
            f"at {self.flight.out_of_rail_time:.2f} s"
        )
        logger.info(
            f"Maximum Stability Margin: {self.flight.max_stability_margin:.3f} c "
            f"at {self.flight.max_stability_margin_time:.2f} s"
        )
        logger.info(
            f"Minimum Stability Margin: {self.flight.min_stability_margin:.3f} c "
            f"at {self.flight.min_stability_margin_time:.2f} s"
        )

    def all(self):
        """Prints out all data available about the Flight. This method invokes
        all other print methods in the class.

        Returns
        -------
        None
        """

        self.initial_conditions()
        logger.info("")

        self.surface_wind_conditions()
        logger.info("")

        self.launch_rail_conditions()
        logger.info("")

        self.out_of_rail_conditions()
        logger.info("")

        self.burn_out_conditions()
        logger.info("")

        self.apogee_conditions()
        logger.info("")

        self.events_registered()
        logger.info("")

        self.impact_conditions()
        logger.info("")

        self.stability_margin()
        logger.info("")

        self.maximum_values()
        logger.info("")

        self.rail_button_bending_moments()
        logger.info("")

        self.numerical_integration_settings()
        logger.info("")
