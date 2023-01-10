__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


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

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nGravity Details\n")
        print("Acceleration of Gravity: " + str(self.environment.g) + " m/s²")

        return None

    def launch_site_details(self):
        """Prints launch site details.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nLaunch Site Details\n")
        print("Launch Rail Length:", self.environment.rL, " m")
        time_format = "%Y-%m-%d %H:%M:%S"
        if self.environment.date != None and "UTC" not in self.environment.timeZone:
            print(
                "Launch Date:",
                self.environment.date.strftime(time_format),
                "UTC |",
                self.environment.localDate.strftime(time_format),
                self.environment.timeZone,
            )
        elif self.environment.date != None:
            print("Launch Date:", self.environment.date.strftime(time_format), "UTC")
        if self.environment.lat != None and self.environment.lon != None:
            print("Launch Site Latitude: {:.5f}°".format(self.environment.lat))
            print("Launch Site Longitude: {:.5f}°".format(self.environment.lon))
        print("Reference Datum: " + self.environment.datum)
        print(
            "Launch Site UTM coordinates: {:.2f} ".format(self.environment.initialEast)
            + self.environment.initialEW
            + "    {:.2f} ".format(self.environment.initialNorth)
            + self.environment.initialHemisphere
        )
        print(
            "Launch Site UTM zone:",
            str(self.environment.initialUtmZone) + self.environment.initialUtmLetter,
        )
        print(
            "Launch Site Surface Elevation: {:.1f} m".format(self.environment.elevation)
        )

        return None

    def atmospheric_model_details(self):
        """Prints atmospheric model details.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nAtmospheric Model Details\n")
        modelType = self.environment.atmosphericModelType
        print("Atmospheric Model Type:", modelType)
        print(
            modelType
            + " Maximum Height: {:.3f} km".format(
                self.environment.maxExpectedHeight / 1000
            )
        )
        if modelType in ["Forecast", "Reanalysis", "Ensemble"]:
            # Determine time period
            initDate = self.environment.atmosphericModelInitDate
            endDate = self.environment.atmosphericModelEndDate
            interval = self.environment.atmosphericModelInterval
            print(modelType + " Time Period: From ", initDate, " to ", endDate, " UTC")
            print(modelType + " Hour Interval:", interval, " hrs")
            # Determine latitude and longitude range
            initLat = self.environment.atmosphericModelInitLat
            endLat = self.environment.atmosphericModelEndLat
            initLon = self.environment.atmosphericModelInitLon
            endLon = self.environment.atmosphericModelEndLon
            print(modelType + " Latitude Range: From ", initLat, "° To ", endLat, "°")
            print(modelType + " Longitude Range: From ", initLon, "° To ", endLon, "°")
        if modelType == "Ensemble":
            print("Number of Ensemble Members:", self.environment.numEnsembleMembers)
            print(
                "Selected Ensemble Member:",
                self.environment.ensembleMember,
                " (Starts from 0)",
            )

        return None

    def atmospheric_conditions(self):
        """Prints atmospheric conditions.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nSurface Atmospheric Conditions\n")
        print(
            "Surface Wind Speed: {:.2f} m/s".format(
                self.environment.windSpeed(self.environment.elevation)
            )
        )
        print(
            "Surface Wind Direction: {:.2f}°".format(
                self.environment.windDirection(self.environment.elevation)
            )
        )
        print(
            "Surface Wind Heading: {:.2f}°".format(
                self.environment.windHeading(self.environment.elevation)
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
                self.environment.speedOfSound(self.environment.elevation)
            )
        )

        return None

    def printEarthDetails(self):
        """[UNDER CONSTRUCTION]
        Function to print information about the Earth Model used in the
        Environment Class

        """
        # Print launch site details
        # print("Launch Site Details")
        # print("Launch Site Latitude: {:.5f}°".format(self.environment.lat))
        # print("Launch Site Longitude: {:.5f}°".format(self.environment.lon))
        # print("Reference Datum: " + self.environment.datum)
        # print("Launch Site UTM coordinates: {:.2f} ".format(self.environment.initialEast)
        #    + self.environment.initialEW + "    {:.2f} ".format(self.environment.initialNorth) + self.environment.initialHemisphere
        # )
        # print("Launch Site UTM zone number:", self.environment.initialUtmZone)
        # print("Launch Site Surface Elevation: {:.1f} m".format(self.environment.elevation))
        print(
            "Earth Radius at Launch site: {:.1f} m".format(self.environment.earthRadius)
        )
        print("Gravity acceleration at launch site: Still not implemented :(")

        return None

    def all(self):
        """Prints all print methods about the Environment.

        Parameters
        ----------
        None

        Return
        ------
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

        return None
