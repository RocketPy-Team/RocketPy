import numpy as np

from rocketpy.tools import from_hex_decode, to_hex_encode

from ...mathutils.function import Function
from .base_parachute import BaseParachute


class HemisphericalParachute(BaseParachute):
    """Implements a hemispherical parachute.

    Attributes
    ----------
    Parachute.name : string
        Parachute name, such as drogue and main. Has no impact in
        simulation, as it is only used to display data in a more
        organized matter.
    Parachute.parachute_type : string
        Parachute type, such as hemispherical and parafoil.
    Parachute.cd_s : float
        Drag coefficient times reference area for parachute. It has units of
        area and must be given in squared meters.
    Parachute.trigger : callable, float, str
        This parameter defines the trigger condition for the parachute ejection
        system. It can be one of the following:

        - A callable function that takes three arguments:
          1. Freestream pressure in pascals.
          2. Height in meters above ground level.
          3. The state vector of the simulation, which is defined as:

             `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.

          4. A list of sensors that are attached to the rocket. The most recent
             measurements of the sensors are provided with the
             ``sensor.measurement`` attribute. The sensors are listed in the same
             order as they are added to the rocket.

          The function should return ``True`` if the parachute ejection system
          should be triggered and False otherwise. The function will be called
          according to the specified sampling rate.

        - A float value, representing an absolute height in meters. In this
          case, the parachute will be ejected when the rocket reaches this height
          above ground level.

        - The string "apogee" which triggers the parachute at apogee, i.e.,
          when the rocket reaches its highest point and starts descending.


    Parachute.triggerfunc : function
        Trigger function created from the trigger used to evaluate the trigger
        condition for the parachute ejection system. It is a callable function
        that takes three arguments: Freestream pressure in Pa, Height above
        ground level in meters, and the state vector of the simulation. The
        returns ``True`` if the parachute ejection system should be triggered
        and ``False`` otherwise.

        .. note:

            The function will be called according to the sampling rate specified.

    Parachute.sampling_rate : float
        Sampling rate, in Hz, for the trigger function.
    Parachute.lag : float
        Time, in seconds, between the parachute ejection system is triggered
        and the parachute is fully opened.
    Parachute.noise : tuple, list
        List in the format (mean, standard deviation, time-correlation).
        The values are used to add noise to the pressure signal which is passed
        to the trigger function. Default value is (0, 0, 0). Units are in Pa.
    Parachute.noise_bias : float
        Mean value of the noise added to the pressure signal, which is
        passed to the trigger function. Unit is in Pa.
    Parachute.noise_deviation : float
        Standard deviation of the noise added to the pressure signal,
        which is passed to the trigger function. Unit is in Pa.
    Parachute.noise_corr : tuple, list
        Tuple with the correlation between noise and time.
    Parachute.noise_signal : list of tuple
        List of (t, noise signal) corresponding to signal passed to
        trigger function. Completed after running a simulation.
    Parachute.noisy_pressure_signal : list of tuple
        List of (t, noisy pressure signal) that is passed to the
        trigger function. Completed after running a simulation.
    Parachute.clean_pressure_signal : list of tuple
        List of (t, clean pressure signal) corresponding to signal passed to
        trigger function. Completed after running a simulation.
    Parachute.noise_signal_function : Function
        Function of noiseSignal.
    Parachute.noisy_pressure_signal_function : Function
        Function of noisy_pressure_signal.
    Parachute.clean_pressure_signal_function : Function
        Function of clean_pressure_signal.
    Parachute.drag_coefficient : float
        Drag coefficient of the inflated canopy shape, used only when
        ``radius`` is not provided to estimate the parachute radius from
        ``cd_s``: ``R = sqrt(cd_s / (drag_coefficient * pi))``. Typical
        values: 1.4 for hemispherical canopies (default), 0.75 for flat
        circular canopies, 1.5 for extended-skirt canopies.
    Parachute.radius : float
        Length of the non-unique semi-axis (radius) of the inflated hemispherical
        parachute in meters. If not provided at construction time, it is
        estimated from ``cd_s`` and ``drag_coefficient``.
    Parachute.height : float
        Length of the unique semi-axis (height) of the inflated hemispherical
        parachute in meters.
    Parachute.porosity : float
        Geometric porosity of the canopy (ratio of open area to total canopy
        area), in [0, 1]. Affects only the added-mass scaling during descent;
        it does not change ``cd_s`` (drag). The default value of 0.0432 is
        chosen so that the resulting ``added_mass_coefficient`` equals
        approximately 1.0 ("neutral" added-mass behavior).
    Parachute.added_mass_coefficient : float
        Coefficient used to calculate the added-mass due to dragged air. It is
        calculated from the porosity of the parachute.
    """

    def __init__(
        self,
        name,
        cd_s,
        trigger,
        sampling_rate,
        lag=0,
        noise=(0, 0, 0),
        radius=None,
        height=None,
        porosity=0.0432,
        drag_coefficient=1.4,
    ):
        """Initializes Parachute class.

        Parameters
        ----------
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        cd_s : float
            Drag coefficient times reference area of the parachute.
        trigger : callable, float, str
            Defines the trigger condition for the parachute ejection system. It
            can be one of the following:

            - A callable function that takes three arguments: \

                1. Freestream pressure in pascals.
                2. Height in meters above ground level.
                3. The state vector of the simulation, which is defined as: \

                    .. code-block:: python

                        u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

                .. note::

                    The function should return ``True`` if the parachute \
                    ejection system should be triggered and ``False`` otherwise.
            - A float value, representing an absolute height in meters. In this \
                case, the parachute will be ejected when the rocket reaches this \
                height above ground level.
            - The string "apogee" which triggers the parachute at apogee, i.e., \
                when the rocket reaches its highest point and starts descending.

            .. note::

                The function will be called according to the sampling rate specified.
        sampling_rate : float
            Sampling rate in which the parachute trigger will be checked at.
            It is used to simulate the refresh rate of onboard sensors such
            as barometers. Default value is 100. Value must be given in hertz.
        lag : float, optional
            Time between the parachute ejection system is triggered and the
            parachute is fully opened. During this time, the simulation will
            consider the rocket as flying without a parachute. Default value
            is 0. Must be given in seconds.
        noise : tuple, list, optional
            List in the format (mean, standard deviation, time-correlation).
            The values are used to add noise to the pressure signal which is
            passed to the trigger function. Default value is ``(0, 0, 0)``.
            Units are in Pa.
        radius : float, optional
            Length of the non-unique semi-axis (radius) of the inflated
            hemispherical parachute. If not provided, it is estimated from
            ``cd_s`` and ``drag_coefficient`` using:
            ``radius = sqrt(cd_s / (drag_coefficient * pi))``.
            Units are in meters.
        height : float, optional
            Length of the unique semi-axis (height) of the inflated hemispherical
            parachute. Default value is the radius of the parachute.
            Units are in meters.
        porosity : float, optional
            Geometric porosity of the canopy (ratio of open area to total
            canopy area), in [0, 1]. Affects only the added-mass scaling
            during descent; it does not change ``cd_s`` (drag). The default
            value of 0.0432 is chosen so that the resulting
            ``added_mass_coefficient`` equals approximately 1.0 ("neutral"
            added-mass behavior).
        drag_coefficient : float, optional
            Drag coefficient of the inflated canopy shape, used only when
            ``radius`` is not provided. It relates the aerodynamic ``cd_s``
            to the physical canopy area via
            ``cd_s = drag_coefficient * pi * radius**2``. Typical values:

            - **1.4** — hemispherical canopy (default, NASA SP-8066)
            - **0.75** — flat circular canopy
            - **1.5** — extended-skirt canopy

            Has no effect when ``radius`` is explicitly provided.
        """

        parachute_type = "hemispherical"
        super().__init__(
            name=name,
            parachute_type=parachute_type,
            trigger=trigger,
            sampling_rate=sampling_rate,
            lag=lag,
            noise=noise,
        )
        self.cd_s = cd_s
        self.trigger = trigger
        self.drag_coefficient = drag_coefficient
        self.porosity = porosity

        # Initialize derived attributes
        self.radius = self.__resolve_radius(radius, cd_s, drag_coefficient)
        self.height = self.__resolve_height(height, self.radius)
        self.added_mass_coefficient = self.__compute_added_mass_coefficient(
            self.porosity
        )

    def __resolve_radius(self, radius, cd_s, drag_coefficient):
        """Resolves parachute radius from input or aerodynamic relation."""
        if radius is not None:
            return radius

        # cd_s = Cd * S = Cd * pi * R^2  =>  R = sqrt(cd_s / (Cd * pi))
        return np.sqrt(cd_s / (drag_coefficient * np.pi))

    def __resolve_height(self, height, radius):
        """Resolves parachute height defaulting to radius when not provided."""
        return height or radius

    def __compute_added_mass_coefficient(self, porosity):
        """Computes the added-mass coefficient from canopy porosity."""
        return 1.068 * (
            1 - 1.465 * porosity - 0.25975 * porosity**2 + 1.2626 * porosity**3
        )

    def add_information_to_flight(self, flight_obj, additional_info):
        """Adds parachute information to flight"""
        drag = additional_info["drag"]
        t = additional_info["t"]
        if self.name not in flight_obj.parachutes_info.keys():
            flight_obj.parachutes_info[self.name] = {"drag": [], "t": []}
            flight_obj.parachutes_info[self.name]["drag"].append(drag)
            flight_obj.parachutes_info[self.name]["t"].append(t)
        else:
            # LSODA did not accept last solution, we replace it
            if t == flight_obj.parachutes_info[self.name]["t"][-1]:
                flight_obj.parachutes_info[self.name]["drag"][-1] = drag
                flight_obj.parachutes_info[self.name]["t"][-1] = t
            else:
                flight_obj.parachutes_info[self.name]["drag"].append(drag)
                flight_obj.parachutes_info[self.name]["t"].append(t)

    # pylint: disable=too-many-locals, too-many-statements
    def u_dot(self, t, u, flight_information, post_processing=False):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying under parachute. Each parachute type has


        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        flight_information : dictionary
            A dictionary containing additional information used in
            the parachute equations of motion. Examples are
            Environment and Rocket data
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack. Default is False.

        Return
        ------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].

        """
        # Get relevant state data
        z, vx, vy, vz = u[2:6]

        env = flight_information["env"]
        rocket = flight_information["rocket"]

        # Get atmospheric data
        rho = env.density.get_value_opt(z)
        wind_velocity_x = env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = env.wind_velocity_y.get_value_opt(z)

        # Get the mass of the rocket
        mp = rocket.dry_mass

        # to = 1.2
        # eta = 1
        # Rdot = (6 * R * (1 - eta) / (1.2**6)) * (
        #     (1 - eta) * t**5 + eta * (to**3) * (t**2)
        # )
        # Rdot = 0

        # tf = 8 * nominal diameter / velocity at line stretch

        # Calculate added mass
        ma = (
            self.added_mass_coefficient
            * rho
            * (2 / 3)
            * np.pi
            * self.radius**2
            * self.height
        )

        # Calculate freestream speed
        freestream_x = vx - wind_velocity_x
        freestream_y = vy - wind_velocity_y
        freestream_z = vz
        free_stream_speed = (freestream_x**2 + freestream_y**2 + freestream_z**2) ** 0.5

        # Determine drag force
        pseudo_drag = -0.5 * rho * self.cd_s * free_stream_speed
        # pseudo_drag = pseudo_drag - ka * rho * 4 * np.pi * (R**2) * Rdot
        Dx = pseudo_drag * freestream_x  # add eta efficiency for wake
        Dy = pseudo_drag * freestream_y
        Dz = pseudo_drag * freestream_z
        total_drag = np.sqrt(Dx**2 + Dy**2 + Dz**2)
        ax = Dx / (mp + ma)
        ay = Dy / (mp + ma)
        az = (Dz - mp * env.gravity.get_value_opt(z)) / (mp + ma)

        # Add coriolis acceleration
        _, w_earth_y, w_earth_z = env.earth_rotation_vector
        ax -= 2 * (vz * w_earth_y - vy * w_earth_z)
        ay -= 2 * (vx * w_earth_z)
        az -= 2 * (-vx * w_earth_y)

        additional_info = {
            "t": t,
            "drag": total_drag,
        }
        output = {
            "state": [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0],
            "additional_info": additional_info,
        }

        if post_processing:
            data_dict = {
                "state": [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0],
                "post_processing_information": [
                    t,
                    ax,
                    ay,
                    az,
                    0,
                    0,
                    0,
                    Dx,
                    Dy,
                    Dz,
                    0,
                    0,
                    0,
                    0,
                ],
            }
            return data_dict
        return output

    # serialization methods
    def to_dict(self, **kwargs):
        allow_pickle = kwargs.get("allow_pickle", True)
        trigger = self.trigger

        if callable(self.trigger) and not isinstance(self.trigger, Function):
            if allow_pickle:
                trigger = to_hex_encode(trigger)
            else:
                trigger = trigger.__name__

        data = {
            "name": self.name,
            "parachute_type": self.parachute_type,
            "cd_s": self.cd_s,
            "trigger": trigger,
            "sampling_rate": self.sampling_rate,
            "lag": self.lag,
            "noise": self.noise,
            "radius": self.radius,
            "drag_coefficient": self.drag_coefficient,
            "height": self.height,
            "porosity": self.porosity,
        }

        if kwargs.get("include_outputs", False):
            data["noise_signal"] = self.noise_signal
            data["noise_function"] = (
                to_hex_encode(self.noise_function)
                if allow_pickle
                else self.noise_function.__name__
            )
            data["noisy_pressure_signal"] = self.noisy_pressure_signal
            data["clean_pressure_signal"] = self.clean_pressure_signal

        return data

    @classmethod
    def from_dict(cls, data):
        trigger = data["trigger"]

        try:
            trigger = from_hex_decode(trigger)
        except (TypeError, ValueError):
            pass

        parachute = cls(
            name=data["name"],
            cd_s=data["cd_s"],
            trigger=trigger,
            sampling_rate=data["sampling_rate"],
            lag=data["lag"],
            noise=data["noise"],
            radius=data.get("radius", None),
            drag_coefficient=data.get("drag_coefficient", 1.4),
            height=data.get("height", None),
            porosity=data.get("porosity", 0.0432),
        )

        return parachute
