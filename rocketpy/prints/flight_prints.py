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
        print("\nInitial Conditions\n")

        t_init = self.flight.time[0]

        print(f"Initial time: {t_init:.3f} s")
        print(
            f"Position - x: {self.flight.x(t_init):.2f} m | "
            f"y: {self.flight.y(t_init):.2f} m | "
            f"z: {self.flight.z(t_init):.2f} m"
        )
        print(
            f"Velocity - Vx: {self.flight.vx(t_init):.2f} m/s | "
            f"Vy: {self.flight.vy(t_init):.2f} m/s | "
            f"Vz: {self.flight.vz(t_init):.2f} m/s"
        )
        print(
            f"Attitude (quaternions) - e0: {self.flight.e0(t_init):.3f} | "
            f"e1: {self.flight.e1(t_init):.3f} | "
            f"e2: {self.flight.e2(t_init):.3f} | "
            f"e3: {self.flight.e3(t_init):.3f}"
        )
        print(
            f"Euler Angles - Spin φ : {self.flight.phi(t_init):.2f}° | "
            f"Nutation θ: {self.flight.theta(t_init):.2f}° | "
            f"Precession ψ: {self.flight.psi(t_init):.2f}°"
        )
        print(
            f"Angular Velocity - ω1: {self.flight.w1(t_init):.2f} rad/s | "
            f"ω2: {self.flight.w2(t_init):.2f} rad/s | "
            f"ω3: {self.flight.w3(t_init):.2f} rad/s"
        )
        print(f"Initial Stability Margin: {self.flight.initial_stability_margin:.3f} c")

    def numerical_integration_settings(self):
        """Prints out the numerical integration settings available about the
        flight, this includes the maximum allowed flight time, maximum allowed
        time step, and other settings.

        Returns
        -------
        None
        """
        print("\nNumerical Integration Settings\n")
        print(f"Maximum Allowed Flight Time: {self.flight.max_time:.2f} s")
        print(f"Maximum Allowed Time Step: {self.flight.max_time_step:.2f} s")
        print(f"Minimum Allowed Time Step: {self.flight.min_time_step:.2e} s")
        print(f"Relative Error Tolerance: {self.flight.rtol}")
        print(f"Absolute Error Tolerance: {self.flight.atol}")
        print(f"Allow Event Overshoot: {self.flight.time_overshoot}")
        print(f"Terminate Simulation on Apogee: {self.flight.terminate_on_apogee}")
        print(f"Number of Time Steps Used: {len(self.flight.time_steps)}")
        print(
            "Number of Derivative Functions Evaluation: "
            f"{sum(self.flight.function_evaluations_per_time_step)}"
        )
        avg_func_evals_per_step = sum(
            self.flight.function_evaluations_per_time_step
        ) / len(self.flight.time_steps)
        print(
            f"Average Function Evaluations per Time Step: {avg_func_evals_per_step:.3f}"
        )

    def surface_wind_conditions(self):
        """Prints out the Surface Wind Conditions for the flight.

        Returns
        -------
        None
        """
        print("\nSurface Wind Conditions\n")
        print(f"Frontal Surface Wind Speed: {self.flight.frontal_surface_wind:.2f} m/s")
        print(f"Lateral Surface Wind Speed: {self.flight.lateral_surface_wind:.2f} m/s")

    def launch_rail_conditions(self):
        """Prints out the Launch Rail Conditions available about the flight,
        including the length, inclination, and heading of the launch rail.

        Returns
        -------
        None
        """
        print("\nLaunch Rail\n")
        print(f"Launch Rail Length: {self.flight.rail_length} m")
        print(f"Launch Rail Inclination: {self.flight.inclination:.2f}°")
        print(f"Launch Rail Heading: {self.flight.heading:.2f}°")

    def out_of_rail_conditions(self):
        """Prints out the Out of Rail Conditions available about the flight,
        including the time, velocity, stability margin, angle of attack, thrust
        to weight ratio, and Reynolds number.

        Returns
        -------
        None
        """
        print("\nRail Departure State\n")
        print(f"Rail Departure Time: {self.flight.out_of_rail_time:.3f} s")
        print(f"Rail Departure Velocity: {self.flight.out_of_rail_velocity:.3f} m/s")
        print(
            "Rail Departure Stability Margin: "
            f"{self.flight.out_of_rail_stability_margin:.3f} c"
        )
        print(
            "Rail Departure Angle of Attack: "
            f"{self.flight.angle_of_attack(self.flight.out_of_rail_time):.3f}°"
        )
        print(
            "Rail Departure Thrust-Weight Ratio: "
            f"{self.flight.rocket.thrust_to_weight(self.flight.out_of_rail_time):.3f}"
        )
        print(
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
        print("\nBurn out State\n")
        print(f"Burn out time: {self.flight.rocket.motor.burn_out_time:.3f} s")
        print(
            "Altitude at burn out: "
            f"{self.flight.z(self.flight.rocket.motor.burn_out_time):.3f} m (ASL) | "
            f"{self.flight.altitude(self.flight.rocket.motor.burn_out_time):.3f} "
            "m (AGL)"
        )
        print(
            "Rocket speed at burn out: "
            f"{self.flight.speed(self.flight.rocket.motor.burn_out_time):.3f} m/s"
        )

        stream_velocity = self.flight.free_stream_speed(
            self.flight.rocket.motor.burn_out_time
        )
        print(f"Freestream velocity at burn out: {stream_velocity:.3f} m/s")

        print(
            "Mach Number at burn out: "
            f"{self.flight.mach_number(self.flight.rocket.motor.burn_out_time):.3f}"
        )
        print(
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
        print("\nApogee State\n")
        print(f"Apogee Time: {self.flight.apogee_time:.3f} s")
        print(
            f"Apogee Altitude: {self.flight.apogee:.3f} m (ASL) | "
            f"{self.flight.altitude(self.flight.apogee_time):.3f} m (AGL)"
        )
        print(f"Apogee Freestream Speed: {self.flight.apogee_freestream_speed:.3f} m/s")
        print(f"Apogee X position: {self.flight.x(self.flight.apogee_time):.3f} m")
        print(f"Apogee Y position: {self.flight.y(self.flight.apogee_time):.3f} m")
        print(f"Apogee latitude: {self.flight.latitude(self.flight.apogee_time):.7f}°")
        print(
            f"Apogee longitude: {self.flight.longitude(self.flight.apogee_time):.7f}°"
        )

    def events_registered(self):
        """Prints out the Events Registered available about the flight.

        Returns
        -------
        None
        """
        print("\nParachute Events\n")
        if len(self.flight.parachute_events) == 0:
            print("No Parachute Events Were Triggered.")
        for event in self.flight.parachute_events:
            trigger_time = event[0]
            parachute = event[1]
            open_time = trigger_time + parachute.lag
            speed = self.flight.free_stream_speed(open_time)
            altitude = self.flight.z(open_time)
            name = parachute.name.title()
            print(f"Parachute: {name}")
            print(f"\tEjection time: {trigger_time:.3f} s")
            print(f"\tInflation time: {open_time:.3f} s")
            print(f"\tFreestream speed at inflation: {speed:.3f} m/s")
            print(
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
            print("\nImpact Conditions\n")
            print(f"Time of impact: {self.flight.t_final:.3f} s")
            print(f"X impact: {self.flight.x_impact:.3f} m")
            print(f"Y impact: {self.flight.y_impact:.3f} m")
            print(
                f"Altitude impact: {self.flight.z(self.flight.t_final):.3f} m (ASL) | "
                f"{self.flight.altitude(self.flight.t_final):.3f} m (AGL) "
            )
            print(f"Latitude: {self.flight.latitude(self.flight.t_final):.7f}°")
            print(f"Longitude: {self.flight.longitude(self.flight.t_final):.7f}°")
            print(f"Vertical velocity at impact: {self.flight.impact_velocity:.3f} m/s")
            num_parachute_events = sum(
                1
                for event in self.flight.parachute_events
                if event[0] < self.flight.t_final
            )
            print(
                f"Number of parachutes triggered until impact: {num_parachute_events}"
            )
        elif self.flight.terminate_on_apogee is False:
            print("End of Simulation")
            t_final = self.flight.time[-1]
            print(f"Time: {t_final:.3f} s")
            print(
                f"Altitude: {self.flight.z(t_final)} m (ASL) | "
                f"{self.flight.altitude(t_final):.3f} m (AGL)"
            )
            print(f"Latitude: {self.flight.latitude(t_final):.7f}°")
            print(f"Longitude: {self.flight.longitude(t_final):.7f}°")

    def maximum_values(self):
        """Prints out the Maximum Values available about the flight.

        Returns
        -------
        None
        """
        print("\nMaximum Values\n")
        print(
            f"Maximum Speed: {self.flight.max_speed:.3f} m/s "
            f"at {self.flight.max_speed_time:.2f} s"
        )
        print(
            f"Maximum Mach Number: {self.flight.max_mach_number:.3f} Mach "
            f"at {self.flight.max_mach_number_time:.2f} s"
        )
        print(
            f"Maximum Reynolds Number: {self.flight.max_reynolds_number:.3e} "
            f"at {self.flight.max_reynolds_number_time:.2f} s"
        )
        print(
            f"Maximum Dynamic Pressure: {self.flight.max_dynamic_pressure:.3e} Pa "
            f"at {self.flight.max_dynamic_pressure_time:.2f} s"
        )
        print(
            "Maximum Acceleration During Motor Burn: "
            f"{self.flight.max_acceleration_power_on:.3f} m/s² "
            f"at {self.flight.max_acceleration_power_on_time:.2f} s"
        )
        print(
            "Maximum Gs During Motor Burn: "
            f"{self.flight.max_acceleration_power_on / self.flight.env.standard_g:.3f} "
            f"g at {self.flight.max_acceleration_power_on_time:.2f} s"
        )
        print(
            "Maximum Acceleration After Motor Burn: "
            f"{self.flight.max_acceleration_power_off:.3f} m/s² "
            f"at {self.flight.max_acceleration_power_off_time:.2f} s"
        )
        print(
            "Maximum Gs After Motor Burn: "
            f"{self.flight.max_acceleration_power_off / self.flight.env.standard_g:.3f}"
            f" Gs at {self.flight.max_acceleration_power_off_time:.2f} s"
        )
        print(
            f"Maximum Stability Margin: {self.flight.max_stability_margin:.3f} c "
            f"at {self.flight.max_stability_margin_time:.2f} s"
        )

        if (
            len(self.flight.rocket.rail_buttons) == 0
            or self.flight.out_of_rail_time_index == 0
        ):
            pass
        else:
            print(
                "Maximum Upper Rail Button Normal Force: "
                f"{self.flight.max_rail_button1_normal_force:.3f} N"
            )
            print(
                "Maximum Upper Rail Button Shear Force: "
                f"{self.flight.max_rail_button1_shear_force:.3f} N"
            )
            print(
                "Maximum Lower Rail Button Normal Force: "
                f"{self.flight.max_rail_button2_normal_force:.3f} N"
            )
            print(
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

        print("\nRail Button Bending Moments\n")
        print(
            "Maximum Upper Rail Button Bending Moment: "
            f"{self.flight.max_rail_button1_bending_moment:.3f} N·m"
        )
        print(
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
        print("\nStability Margin\n")
        print(
            f"Initial Stability Margin: {self.flight.initial_stability_margin:.3f} c "
            f"at {self.flight.time[0]:.2f} s"
        )
        print(
            "Out of Rail Stability Margin: "
            f"{self.flight.out_of_rail_stability_margin:.3f} c "
            f"at {self.flight.out_of_rail_time:.2f} s"
        )
        print(
            f"Maximum Stability Margin: {self.flight.max_stability_margin:.3f} c "
            f"at {self.flight.max_stability_margin_time:.2f} s"
        )
        print(
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
        print()

        self.surface_wind_conditions()
        print()

        self.launch_rail_conditions()
        print()

        self.out_of_rail_conditions()
        print()

        self.burn_out_conditions()
        print()

        self.apogee_conditions()
        print()

        self.events_registered()
        print()

        self.impact_conditions()
        print()

        self.stability_margin()
        print()

        self.maximum_values()
        print()

        self.rail_button_bending_moments()
        print()

        self.numerical_integration_settings()
        print()
