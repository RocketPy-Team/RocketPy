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
            f"Acceleration of gravity at {max_expected_height / 1000:7.3f} "
            f"km (ASL): {ceiling_gravity:.4f} m/s²\n"
        )

    def launch_site_details(self):
        """Prints launch site details.

        Returns
        -------
        None
        """
        print("\nLaunch Site Details\n")
        time_format = "%Y-%m-%d %H:%M:%S"
        if (
            self.environment.datetime_date is not None
            and "UTC" not in self.environment.timezone
        ):
            print(
                "Launch Date:",
                self.environment.datetime_date.strftime(time_format),
                "UTC |",
                self.environment.local_date.strftime(time_format),
                self.environment.timezone,
            )
        elif self.environment.datetime_date is not None:
            print(
                "Launch Date:",
                self.environment.datetime_date.strftime(time_format),
                "UTC",
            )
        if (
            self.environment.latitude is not None
            and self.environment.longitude is not None
        ):
            print(f"Launch Site Latitude: {self.environment.latitude:.5f}°")
            print(f"Launch Site Longitude: {self.environment.longitude:.5f}°")
        print(f"Reference Datum: {self.environment.datum}")
        if self.environment.initial_east:
            print(
                f"Launch Site UTM coordinates: {self.environment.initial_east:.2f} "
                f"{self.environment.initial_ew}    {self.environment.initial_north:.2f} "
                f"{self.environment.initial_hemisphere}"
            )
            print(
                f"Launch Site UTM zone: {self.environment.initial_utm_zone}"
                f"{self.environment.initial_utm_letter}"
            )
        print(f"Launch Site Surface Elevation: {self.environment.elevation:.1f} m\n")

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
            f"{model_type} Maximum Height: "
            f"{self.environment.max_expected_height / 1000:.3f} km"
        )
        if model_type in ["Forecast", "Reanalysis", "Ensemble"]:
            # Determine time period
            init_date = self.environment.atmospheric_model_init_date
            end_date = self.environment.atmospheric_model_end_date
            interval = self.environment.atmospheric_model_interval
            print(f"{model_type} Time Period: from {init_date} to {end_date} utc")
            print(f"{model_type} Hour Interval: {interval} hrs")
            # Determine latitude and longitude range
            init_lat = self.environment.atmospheric_model_init_lat
            end_lat = self.environment.atmospheric_model_end_lat
            init_lon = self.environment.atmospheric_model_init_lon
            end_lon = self.environment.atmospheric_model_end_lon
            print(f"{model_type} Latitude Range: From {init_lat}° to {end_lat}°")
            print(f"{model_type} Longitude Range: From {init_lon}° to {end_lon}°")
        if model_type == "Ensemble":
            print(
                f"Number of Ensemble Members: {self.environment.num_ensemble_members}"
            )
            print(
                f"Selected Ensemble Member: {self.environment.ensemble_member} "
                "(Starts from 0)\n"
            )

    def atmospheric_conditions(self):
        """Prints atmospheric conditions.

        Returns
        -------
        None
        """
        print("\nSurface Atmospheric Conditions\n")
        wind_speed = self.environment.wind_speed(self.environment.elevation)
        wind_direction = self.environment.wind_direction(self.environment.elevation)
        wind_heading = self.environment.wind_heading(self.environment.elevation)
        pressure = self.environment.pressure(self.environment.elevation) / 100
        temperature = self.environment.temperature(self.environment.elevation)
        air_density = self.environment.density(self.environment.elevation)
        speed_of_sound = self.environment.speed_of_sound(self.environment.elevation)
        print(f"Surface Wind Speed: {wind_speed:.2f} m/s")
        print(f"Surface Wind Direction: {wind_direction:.2f}°")
        print(f"Surface Wind Heading: {wind_heading:.2f}°")
        print(f"Surface Pressure: {pressure:.2f} hPa")
        print(f"Surface Temperature: {temperature:.2f} K")
        print(f"Surface Air Density: {air_density:.3f} kg/m³")
        print(f"Surface Speed of Sound: {speed_of_sound:.2f} m/s\n")

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
        print(f"Earth Radius at Launch site: {earth_radius / 1000:.2f} km")
        print(f"Semi-major Axis: {semi_major_axis / 1000:.2f} km")
        print(f"Semi-minor Axis: {semi_minor_axis / 1000:.2f} km")
        print(f"Flattening: {flattening:.4f}\n")

    def all(self):
        """Prints all print methods about the Environment.

        Returns
        -------
        None
        """
        self.gravity_details()
        self.launch_site_details()
        self.atmospheric_model_details()
        self.atmospheric_conditions()
        self.print_earth_details()
