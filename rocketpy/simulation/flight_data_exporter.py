"""
Exports a rocketpy.Flight object's data to external files.
"""

import json

import numpy as np
import simplekml


class FlightDataExporter:
    """Export data from a rocketpy.Flight object to various formats."""

    def __init__(self, flight, name="Flight Data"):
        """
        Parameters
        ----------
        flight : rocketpy.simulation.flight.Flight
            The Flight instance to export from.
        name : str, optional
            A label for this exporter instance.
        """
        self.name = name
        self._flight = flight

    def __repr__(self):
        return f"FlightDataExporter(name='{self.name}', flight='{type(self._flight).__name__}')"

    def export_pressures(self, file_name, time_step):
        """Exports the pressure experienced by the rocket during the flight to
        an external file, the '.csv' format is recommended, as the columns will
        be separated by commas. It can handle flights with or without
        parachutes, although it is not possible to get a noisy pressure signal
        if no parachute is added.

        If a parachute is added, the file will contain 3 columns: time in
        seconds, clean pressure in Pascals and noisy pressure in Pascals.
        For flights without parachutes, the third column will be discarded

        This function was created especially for the 'Projeto Jupiter'
        Electronics Subsystems team and aims to help in configuring
        micro-controllers.

        Parameters
        ----------
        file_name : string
            The final file name,
        time_step : float
            Time step desired for the final file

        Return
        ------
        None
        """
        f = self._flight
        time_points = np.arange(0, f.t_final, time_step)
        # pylint: disable=W1514, E1121
        with open(file_name, "w") as file:
            if len(f.rocket.parachutes) == 0:
                print("No parachutes in the rocket, saving static pressure.")
                for t in time_points:
                    file.write(f"{t:f}, {f.pressure.get_value_opt(t):.5f}\n")
            else:
                for parachute in f.rocket.parachutes:
                    for t in time_points:
                        p_cl = parachute.clean_pressure_signal_function.get_value_opt(t)
                        p_ns = parachute.noisy_pressure_signal_function.get_value_opt(t)
                        file.write(f"{t:f}, {p_cl:.5f}, {p_ns:.5f}\n")
                    # We need to save only 1 parachute data
                    break

    def export_data(self, file_name, *variables, time_step=None):
        """Exports flight data to a comma separated value file (.csv).

        Data is exported in columns, with the first column representing time
        steps. The first line of the file is a header line, specifying the
        meaning of each column and its units.

        Parameters
        ----------
        file_name : string
            The file name or path of the exported file. Example: flight_data.csv
            Do not use forbidden characters, such as / in Linux/Unix and
            `<, >, :, ", /, \\, | ?, *` in Windows.
        variables : strings, optional
            Names of the data variables which shall be exported. Must be Flight
            class attributes which are instances of the Function class. Usage
            example: test_flight.export_data('test.csv', 'z', 'angle_of_attack',
            'mach_number').
        time_step : float, optional
            Time step desired for the data. If None, all integration time steps
            will be exported. Otherwise, linear interpolation is carried out to
            calculate values at the desired time steps. Example: 0.001.
        """
        f = self._flight

        # Fast evaluation for the most basic scenario
        if time_step is None and len(variables) == 0:
            np.savetxt(
                file_name,
                f.solution,
                fmt="%.6f",
                delimiter=",",
                header=""
                "Time (s),"
                "X (m),"
                "Y (m),"
                "Z (m),"
                "E0,"
                "E1,"
                "E2,"
                "E3,"
                "W1 (rad/s),"
                "W2 (rad/s),"
                "W3 (rad/s)",
            )
            return

        # Not so fast evaluation for general case
        if variables is None:
            variables = [
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "e0",
                "e1",
                "e2",
                "e3",
                "w1",
                "w2",
                "w3",
            ]

        if time_step is None:
            time_points = f.time
        else:
            time_points = np.arange(f.t_initial, f.t_final, time_step)

        exported_matrix = [time_points]
        exported_header = "Time (s)"

        # Loop through variables, get points and names (for the header)
        for variable in variables:
            if variable in f.__dict__:
                variable_function = f.__dict__[variable]
            # Deal with decorated Flight methods
            else:
                try:
                    variable_function = getattr(f, variable)
                except AttributeError as exc:
                    raise AttributeError(
                        f"Variable '{variable}' not found in Flight class"
                    ) from exc
            variable_points = variable_function(time_points)
            exported_matrix += [variable_points]
            exported_header += f", {variable_function.__outputs__[0]}"

        exported_matrix = np.array(exported_matrix).T  # Fix matrix orientation

        np.savetxt(
            file_name,
            exported_matrix,
            fmt="%.6f",
            delimiter=",",
            header=exported_header,
            encoding="utf-8",
        )

    def export_sensor_data(self, file_name, sensor=None):
        """Exports sensors data to a file. The file format can be either .csv or
        .json.

        Parameters
        ----------
        file_name : str
            The file name or path of the exported file. Example: flight_data.csv
            Do not use forbidden characters, such as / in Linux/Unix and
            `<, >, :, ", /, \\, | ?, *` in Windows.
        sensor : Sensor, string, optional
            The sensor to export data from. Can be given as a Sensor object or
            as a string with the sensor name. If None, all sensors data will be
            exported. Default is None.
        """
        f = self._flight

        if sensor is None:
            data_dict = {}
            for used_sensor, measured_data in f.sensor_data.items():
                data_dict[used_sensor.name] = measured_data
        else:
            # export data of only that sensor
            data_dict = {}

            if not isinstance(sensor, str):
                data_dict[sensor.name] = f.sensor_data[sensor]
            else:  # sensor is a string
                matching_sensors = [s for s in f.sensor_data if s.name == sensor]

                if len(matching_sensors) > 1:
                    data_dict[sensor] = []
                    for s in matching_sensors:
                        data_dict[s.name].append(f.sensor_data[s])
                elif len(matching_sensors) == 1:
                    data_dict[sensor] = f.sensor_data[matching_sensors[0]]
                else:
                    raise ValueError("Sensor not found in the Flight.sensor_data.")

        with open(file_name, "w") as file:
            json.dump(data_dict, file)
        print("Sensor data exported to: ", file_name)

    def export_kml(
        self,
        file_name="trajectory.kml",
        time_step=None,
        extrude=True,
        color="641400F0",
        altitude_mode="absolute",
    ):
        """Exports flight data to a .kml file, which can be opened with Google
        Earth to display the rocket's trajectory.

        Parameters
        ----------
        file_name : string
            The file name or path of the exported file. Example: flight_data.csv
        time_step : float, optional
            Time step desired for the data. If None, all integration time steps
            will be exported. Otherwise, linear interpolation is carried out to
            calculate values at the desired time steps. Example: 0.001.
        extrude: bool, optional
            To be used if you want to project the path over ground by using an
            extruded polygon. In case False only the linestring containing the
            flight path will be created. Default is True.
        color : str, optional
            Color of your trajectory path, need to be used in specific kml
            format. Refer to http://www.zonums.com/gmaps/kml_color/ for more
            info.
        altitude_mode: str
            Select elevation values format to be used on the kml file. Use
            'relativetoground' if you want use Above Ground Level elevation, or
            'absolute' if you want to parse elevation using Above Sea Level.
            Default is 'relativetoground'. Only works properly if the ground
            level is flat. Change to 'absolute' if the terrain is to irregular
            or contains mountains.
        """
        f = self._flight

        # Define time points vector
        if time_step is None:
            time_points = f.time
        else:
            time_points = np.arange(f.t_initial, f.t_final + time_step, time_step)

        kml = simplekml.Kml(open=1)
        trajectory = kml.newlinestring(name="Rocket Trajectory - Powered by RocketPy")

        if altitude_mode == "relativetoground":
            # In this mode the elevation data will be the Above Ground Level
            # elevation. Only works properly if the ground level is similar to
            # a plane, i.e. it might not work well if the terrain has mountains
            coords = [
                (
                    f.longitude.get_value_opt(t),
                    f.latitude.get_value_opt(t),
                    f.altitude.get_value_opt(t),
                )
                for t in time_points
            ]
            trajectory.coords = coords
            trajectory.altitudemode = simplekml.AltitudeMode.relativetoground
        else:  # altitude_mode == 'absolute'
            # In this case the elevation data will be the Above Sea Level elevation
            # Ensure you use the correct value on self.env.elevation, otherwise
            # the trajectory path can be offset from ground
            coords = [
                (
                    f.longitude.get_value_opt(t),
                    f.latitude.get_value_opt(t),
                    f.z.get_value_opt(t),
                )
                for t in time_points
            ]
            trajectory.coords = coords
            trajectory.altitudemode = simplekml.AltitudeMode.absolute

        # Modify style of trajectory linestring
        trajectory.style.linestyle.color = color
        trajectory.style.polystyle.color = color
        if extrude:
            trajectory.extrude = 1

        # Save the KML
        kml.save(file_name)
        print("File ", file_name, " saved with success!")
