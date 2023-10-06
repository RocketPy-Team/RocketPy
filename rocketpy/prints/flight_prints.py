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
        return None

    def initial_conditions(self):
        """Prints all initial conditions data available about the Flight.

        Returns
        -------
        None
        """

        print("\nInitial Conditions\n")

        # Post-process results
        if self.flight.post_processed is False:
            self.flight.post_process()
        print(
            "Position - x: {:.2f} m | y: {:.2f} m | z: {:.2f} m".format(
                self.flight.x(0), self.flight.y(0), self.flight.z(0)
            )
        )
        print(
            "Velocity - Vx: {:.2f} m/s | Vy: {:.2f} m/s | Vz: {:.2f} m/s".format(
                self.flight.vx(0), self.flight.vy(0), self.flight.vz(0)
            )
        )
        print(
            "Attitude - e0: {:.3f} | e1: {:.3f} | e2: {:.3f} | e3: {:.3f}".format(
                self.flight.e0(0),
                self.flight.e1(0),
                self.flight.e2(0),
                self.flight.e3(0),
            )
        )
        print(
            "Euler Angles - Spin φ : {:.2f}° | Nutation θ: {:.2f}° | Precession ψ: {:.2f}°".format(
                self.flight.phi(0), self.flight.theta(0), self.flight.psi(0)
            )
        )
        print(
            "Angular Velocity - ω1: {:.2f} rad/s | ω2: {:.2f} rad/s| ω3: {:.2f} rad/s".format(
                self.flight.w1(0), self.flight.w2(0), self.flight.w3(0)
            )
        )

        return None

    def numerical_integration_settings(self):
        """Prints out the Numerical Integration settings available about the
        flight.

        Returns
        -------
        None
        """

        print("\nNumerical Integration Settings\n")
        print("Maximum Allowed Flight Time: {:f} s".format(self.flight.max_time))
        print("Maximum Allowed Time Step: {:f} s".format(self.flight.max_time_step))
        print("Minimum Allowed Time Step: {:e} s".format(self.flight.min_time_step))
        print("Relative Error Tolerance: ", self.flight.rtol)
        print("Absolute Error Tolerance: ", self.flight.atol)
        print("Allow Event Overshoot: ", self.flight.time_overshoot)
        print("Terminate Simulation on Apogee: ", self.flight.terminate_on_apogee)
        print("Number of Time Steps Used: ", len(self.flight.time_steps))
        print(
            "Number of Derivative Functions Evaluation: ",
            sum(self.flight.function_evaluations_per_time_step),
        )
        print(
            "Average Function Evaluations per Time Step: {:3f}".format(
                sum(self.flight.function_evaluations_per_time_step)
                / len(self.flight.time_steps)
            )
        )

        return None

    def surface_wind_conditions(self):
        """Prints out the Surface Wind Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.post_processed is False:
            self.flight.post_process()
        print("\nSurface Wind Conditions\n")
        print(
            "Frontal Surface Wind Speed: {:.2f} m/s".format(
                self.flight.frontal_surface_wind
            )
        )
        print(
            "Lateral Surface Wind Speed: {:.2f} m/s".format(
                self.flight.lateral_surface_wind
            )
        )

        return None

    def launch_rail_conditions(self):
        """Prints out the Launch Rail Conditions available about the flight.

        Returns
        -------
        None
        """

        print("\nLaunch Rail\n")
        print("Launch Rail Length:", self.flight.rail_length, " m")
        print("Launch Rail Inclination: {:.2f}°".format(self.flight.inclination))
        print("Launch Rail Heading: {:.2f}°".format(self.flight.heading))
        return None

    def out_of_rail_conditions(self):
        """Prints out the Out of Rail Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.post_processed is False:
            self.flight.post_process()
        print("\nRail Departure State\n")
        print("Rail Departure Time: {:.3f} s".format(self.flight.out_of_rail_time))
        print(
            "Rail Departure Velocity: {:.3f} m/s".format(
                self.flight.out_of_rail_velocity
            )
        )
        print(
            "Rail Departure Stability Margin: {:.3f} c".format(
                self.flight.stability_margin(self.flight.out_of_rail_time)
            )
        )
        print(
            "Rail Departure Angle of Attack: {:.3f}°".format(
                self.flight.angle_of_attack(self.flight.out_of_rail_time)
            )
        )
        print(
            "Rail Departure Thrust-Weight Ratio: {:.3f}".format(
                self.flight.rocket.thrust_to_weight(self.flight.out_of_rail_time)
            )
        )
        print(
            "Rail Departure Reynolds Number: {:.3e}".format(
                self.flight.reynolds_number(self.flight.out_of_rail_time)
            )
        )

        return None

    def burn_out_conditions(self):
        """Prints out the Burn Out Conditions available about the flight.

        Returns
        -------
        None
        """
        print("\nBurn out State\n")
        print("Burn out time: {:.3f} s".format(self.flight.rocket.motor.burn_out_time))
        print(
            "Altitude at burn out: {:.3f} m (AGL)".format(
                self.flight.z(self.flight.rocket.motor.burn_out_time)
                - self.flight.env.elevation
            )
        )
        print(
            "Rocket velocity at burn out: {:.3f} m/s".format(
                self.flight.speed(self.flight.rocket.motor.burn_out_time)
            )
        )
        print(
            "Freestream velocity at burn out: {:.3f} m/s".format(
                (
                    self.flight.stream_velocity_x(
                        self.flight.rocket.motor.burn_out_time
                    )
                    ** 2
                    + self.flight.stream_velocity_y(
                        self.flight.rocket.motor.burn_out_time
                    )
                    ** 2
                    + self.flight.stream_velocity_z(
                        self.flight.rocket.motor.burn_out_time
                    )
                    ** 2
                )
                ** 0.5
            )
        )
        print(
            "Mach Number at burn out: {:.3f}".format(
                self.flight.mach_number(self.flight.rocket.motor.burn_out_time)
            )
        )
        print(
            "Kinetic energy at burn out: {:.3e} J".format(
                self.flight.kinetic_energy(self.flight.rocket.motor.burn_out_time)
            )
        )

        return None

    def apogee_conditions(self):
        """Prints out the Apogee Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.post_processed is False:
            self.flight.post_process()
        print("\nApogee State\n")
        print(
            "Apogee Altitude: {:.3f} m (ASL) | {:.3f} m (AGL)".format(
                self.flight.apogee, self.flight.apogee - self.flight.env.elevation
            )
        )
        print("Apogee Time: {:.3f} s".format(self.flight.apogee_time))
        print(
            "Apogee Freestream Speed: {:.3f} m/s".format(
                self.flight.apogee_freestream_speed
            )
        )

        return None

    def events_registered(self):
        """Prints out the Events Registered available about the flight.

        Returns
        -------
        None
        """
        if self.flight.post_processed is False:
            self.flight.post_process()
        print("\nParachute Events\n")
        if len(self.flight.parachute_events) == 0:
            print("No Parachute Events Were Triggered.")
        for event in self.flight.parachute_events:
            trigger_time = event[0]
            parachute = event[1]
            open_time = trigger_time + parachute.lag
            velocity = self.flight.free_stream_speed(open_time)
            altitude = self.flight.z(open_time)
            name = parachute.name.title()
            print(name + " Ejection Triggered at: {:.3f} s".format(trigger_time))
            print(name + " Parachute Inflated at: {:.3f} s".format(open_time))
            print(
                name
                + " Parachute Inflated with Freestream Speed of: {:.3f} m/s".format(
                    velocity
                )
            )
            print(
                name
                + " Parachute Inflated at Height of: {:.3f} m (AGL)".format(
                    altitude - self.flight.env.elevation
                )
            )
        return None

    def impact_conditions(self):
        """Prints out the Impact Conditions available about the flight.

        Returns
        -------
        None
        """
        if self.flight.post_processed is False:
            self.flight.post_process()
        if len(self.flight.impact_state) != 0:
            print("\nImpact Conditions\n")
            print("X Impact: {:.3f} m".format(self.flight.x_impact))
            print("Y Impact: {:.3f} m".format(self.flight.y_impact))
            print("Latitude: {:.7f}°".format(self.flight.latitude(self.flight.t_final)))
            print(
                "Longitude: {:.7f}°".format(self.flight.longitude(self.flight.t_final))
            )
            print("Time of Impact: {:.3f} s".format(self.flight.t_final))
            print("Velocity at Impact: {:.3f} m/s".format(self.flight.impact_velocity))
        elif self.flight.terminate_on_apogee is False:
            print("End of Simulation")
            t_final = self.flight.solution[-1][0]
            print("Time: {:.3f} s".format(t_final))
            print("Altitude: {:.3f} m".format(self.flight.solution[-1][3]))
            print("Latitude: {:.3f}°".format(self.flight.latitude(t_final)))
            print("Longitude: {:.3f}°".format(self.flight.longitude(t_final)))

        return None

    def maximum_values(self):
        """Prints out the Maximum Values available about the flight.

        Returns
        -------
        None
        """
        print("\nMaximum Values\n")
        print(
            "Maximum Speed: {:.3f} m/s at {:.2f} s".format(
                self.flight.max_speed, self.flight.max_speed_time
            )
        )
        print(
            "Maximum Mach Number: {:.3f} Mach at {:.2f} s".format(
                self.flight.max_mach_number, self.flight.max_mach_number_time
            )
        )
        print(
            "Maximum Reynolds Number: {:.3e} at {:.2f} s".format(
                self.flight.max_reynolds_number, self.flight.max_reynolds_number_time
            )
        )
        print(
            "Maximum Dynamic Pressure: {:.3e} Pa at {:.2f} s".format(
                self.flight.max_dynamic_pressure, self.flight.max_dynamic_pressure_time
            )
        )
        print(
            "Maximum Acceleration During Motor Burn: {:.3f} m/s² at {:.2f} s".format(
                self.flight.max_acceleration_power_on,
                self.flight.max_acceleration_power_on_time,
            )
        )
        print(
            "Maximum Gs During Motor Burn: {:.3f} g at {:.2f} s".format(
                self.flight.max_acceleration_power_on / self.flight.env.standard_g,
                self.flight.max_acceleration_power_on_time,
            )
        )
        print(
            "Maximum Acceleration After Motor Burn: {:.3f} m/s² at {:.2f} s".format(
                self.flight.max_acceleration_power_off,
                self.flight.max_acceleration_power_off_time,
            )
        )
        print(
            "Maximum Gs After Motor Burn: {:.3f} g at {:.2f} s".format(
                self.flight.max_acceleration_power_off / self.flight.env.standard_g,
                self.flight.max_acceleration_power_off_time,
            )
        )
        print(
            "Maximum Stability Margin: {:.3f} c at {:.2f} s".format(
                self.flight.max_stability_margin, self.flight.max_stability_margin_time
            )
        )

        if (
            len(self.flight.rocket.rail_buttons) == 0
            or self.flight.out_of_rail_time_index == 0
        ):
            pass
        else:
            print(
                "Maximum Upper Rail Button Normal Force: {:.3f} N".format(
                    self.flight.max_rail_button1_normal_force
                )
            )
            print(
                "Maximum Upper Rail Button Shear Force: {:.3f} N".format(
                    self.flight.max_rail_button1_shear_force
                )
            )
            print(
                "Maximum Lower Rail Button Normal Force: {:.3f} N".format(
                    self.flight.max_rail_button2_normal_force
                )
            )
            print(
                "Maximum Lower Rail Button Shear Force: {:.3f} N".format(
                    self.flight.max_rail_button2_shear_force
                )
            )
        return None

    def stability_margin(self):
        """Prints out the maximum and minimum stability margin available
        about the flight."""
        print("\nStability Margin\n")
        print(
            "Maximum Stability Margin: {:.3f} c at {:.2f} s".format(
                self.flight.max_stability_margin, self.flight.max_stability_margin_time
            )
        )
        print(
            "Minimum Stability Margin: {:.3f} c at {:.2f} s".format(
                self.flight.min_stability_margin, self.flight.min_stability_margin_time
            )
        )
        return None

    def all(self):
        """Prints out all data available about the Flight.

        Returns
        -------
        None
        """

        # Print initial conditions
        self.initial_conditions()
        print()

        # Print surface wind conditions
        self.surface_wind_conditions()
        print()

        # Print launch rail orientation
        self.launch_rail_conditions()
        print()

        # Print out of rail conditions
        self.out_of_rail_conditions()
        print()

        # Print burn out conditions
        self.burn_out_conditions()
        print()

        # Print apogee conditions
        self.apogee_conditions()
        print()

        # Print events registered
        self.events_registered()
        print()

        # Print impact conditions
        self.impact_conditions()
        print()

        # Print stability margin
        self.stability_margin()
        print()

        # Print maximum values
        self.maximum_values()
        print()

        # Print Numerical Integration Information
        self.numerical_integration_settings()
        print()

        return None
