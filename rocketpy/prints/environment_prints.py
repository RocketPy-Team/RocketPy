class _EnvironmentPrints:
    """Class that holds prints methods for Environment class.

    Attributes
    ----------
    _EnvironmentPrints.environment : environment
        Environment object that will be used for the prints.

    """

    def __init__(
        self,
        environment,
    ):
        """Initializes _EnvironmentPrints class

        Parameters
        ----------
        environment: Environment
            Instance of the Environment class.

        Returns
        -------
        None
        """
        self.environment = environment
        return None

    def gravity_details(self):
        """Prints gravity details.

        Returns
        -------
        None
        """
        elevation = self.environment.elevation
        max_expected_height = self.environment.max_expected_height
        surface_gravity = self.environment.gravity([elevation])
        ceiling_gravity = self.environment.gravity([max_expected_height])
        print("\nGravity Details\n")
        print(f"Acceleration of gravity at surface level: {surface_gravity:9.4f} m/s²")
        print(
            f"Acceleration of gravity at {max_expected_height/1000:7.3f} km (ASL): {ceiling_gravity:.4f} m/s²"
        )
        return None

    def launch_site_details(self):
        """Prints launch site details.

        Returns
        -------
        None
        """
        print("\nLaunch Site Details\n")
        time_format = "%Y-%m-%d %H:%M:%S"
        if (
            self.environment.datetime_date != None
            and "UTC" not in self.environment.timezone
        ):
            print(
                "Launch Date:",
                self.environment.datetime_date.strftime(time_format),
                "UTC |",
                self.environment.local_date.strftime(time_format),
                self.environment.timezone,
            )
        elif self.environment.datetime_date != None:
            print(
                "Launch Date:",
                self.environment.datetime_date.strftime(time_format),
                "UTC",
            )
        if self.environment.latitude != None and self.environment.longitude != None:
            print("Launch Site Latitude: {:.5f}°".format(self.environment.latitude))
            print("Launch Site Longitude: {:.5f}°".format(self.environment.longitude))
        print("Reference Datum: " + self.environment.datum)
        print(
            "Launch Site UTM coordinates: {:.2f} ".format(self.environment.initial_east)
            + self.environment.initial_ew
            + "    {:.2f} ".format(self.environment.initial_north)
            + self.environment.initial_hemisphere
        )
        print(
            "Launch Site UTM zone:",
            str(self.environment.initial_utm_zone)
            + self.environment.initial_utm_letter,
        )
        print(
            "Launch Site Surface Elevation: {:.1f} m".format(self.environment.elevation)
        )

        return None

    def atmospheric_model_details(self):
        """Prints atmospheric model details.

        Returns
        -------
        None
        """
        print("\nAtmospheric Model Details\n")
        model_type = self.environment.atmospheric_model_type
        print("Atmospheric Model Type:", model_type)
        print(
            model_type
            + " Maximum Height: {:.3f} km".format(
                self.environment.max_expected_height / 1000
            )
        )
        if model_type in ["Forecast", "Reanalysis", "Ensemble"]:
            # Determine time period
            initDate = self.environment.atmospheric_model_init_date
            endDate = self.environment.atmospheric_model_end_date
            interval = self.environment.atmospheric_model_interval
            print(model_type + " Time Period: From ", initDate, " to ", endDate, " UTC")
            print(model_type + " Hour Interval:", interval, " hrs")
            # Determine latitude and longitude range
            initLat = self.environment.atmospheric_model_init_lat
            endLat = self.environment.atmospheric_model_end_lat
            initLon = self.environment.atmospheric_model_init_lon
            endLon = self.environment.atmospheric_model_end_lon
            print(model_type + " Latitude Range: From ", initLat, "° To ", endLat, "°")
            print(model_type + " Longitude Range: From ", initLon, "° To ", endLon, "°")
        if model_type == "Ensemble":
            print("Number of Ensemble Members:", self.environment.num_ensemble_members)
            print(
                "Selected Ensemble Member:",
                self.environment.ensemble_member,
                " (Starts from 0)",
            )

        return None

    def atmospheric_conditions(self):
        """Prints atmospheric conditions.

        Returns
        -------
        None
        """
        print("\nSurface Atmospheric Conditions\n")
        print(
            "Surface Wind Speed: {:.2f} m/s".format(
                self.environment.wind_speed(self.environment.elevation)
            )
        )
        print(
            "Surface Wind Direction: {:.2f}°".format(
                self.environment.wind_direction(self.environment.elevation)
            )
        )
        print(
            "Surface Wind Heading: {:.2f}°".format(
                self.environment.wind_heading(self.environment.elevation)
            )
        )
        print(
            "Surface Pressure: {:.2f} hPa".format(
                self.environment.pressure(self.environment.elevation) / 100
            )
        )
        print(
            "Surface Temperature: {:.2f} K".format(
                self.environment.temperature(self.environment.elevation)
            )
        )
        print(
            "Surface Air Density: {:.3f} kg/m³".format(
                self.environment.density(self.environment.elevation)
            )
        )
        print(
            "Surface Speed of Sound: {:.2f} m/s".format(
                self.environment.speed_of_sound(self.environment.elevation)
            )
        )

        return None

    def print_earth_details(self):
        """
        Function to print information about the Earth Model used in the
        Environment Class

        """
        print("\nEarth Model Details\n")
        earth_radius = self.environment.earth_radius
        semi_major_axis = self.environment.ellipsoid.semi_major_axis
        flattening = self.environment.ellipsoid.flattening
        semi_minor_axis = semi_major_axis * (1 - flattening)
        print(f"Earth Radius at Launch site: {earth_radius/1000:.2f} km")
        print(f"Semi-major Axis: {semi_major_axis/1000:.2f} km")
        print(f"Semi-minor Axis: {semi_minor_axis/1000:.2f} km")
        print(f"Flattening: {flattening:.4f}\n")

        return None

    def all(self):
        """Prints all print methods about the Environment.

        Returns
        -------
        None
        """

        # Print gravity details
        self.gravity_details()
        print()

        # Print launch site details
        self.launch_site_details()
        print()

        # Print atmospheric model details
        self.atmospheric_model_details()
        print()

        # Print atmospheric conditions
        self.atmospheric_conditions()
        print()

        self.print_earth_details()

        return None
