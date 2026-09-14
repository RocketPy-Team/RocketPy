"""Defines the StochasticFlight class."""

from rocketpy.simulation import Flight

from .stochastic_model import StochasticModel


class StochasticFlight(StochasticModel):
    """A Stochastic Flight class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`Flight <rocketpy.simulation.Flight>`

    Attributes
    ----------
    flight : Flight
        The Flight object to be used as a base for the Stochastic flight.
    rail_length : int, float, tuple, list
        The rail length of the flight.
    inclination : int, float, tuple, list
        The inclination of the launch rail.
    heading : int, float, tuple, list
        The heading of the launch rail.
    initial_solution : tuple, list
        The initial solution of the flight. This is a tuple of 14 elements that
        represent the initial conditions of the flight. This attribute can not
        be randomized.
    terminate_on_apogee : bool
        Whether or not the flight should terminate on apogee. This attribute
        can not be randomized.
    time_overshoot : bool
        If False, the simulation will run at the time step defined by the controller
        sampling rate. Be aware that this will make the simulation run much slower.
    max_time : int, float
        The maximum time of the flight simulation. If the flight simulation
        reaches this time, it will terminate. This attribute can not be randomized.
    """

    def __init__(
        self,
        flight,
        rail_length=None,
        inclination=None,
        heading=None,
        initial_solution=None,
        terminate_on_apogee=None,
        time_overshoot=None,
        max_time=None,
    ):
        """Initializes the Stochastic Flight class.

        See Also
        --------
        :ref:`stochastic_model` and :class:`Flight <rocketpy.simulation.Flight>`

        Parameters
        ----------
        flight : Flight
            The Flight object to be used as a base for the Stochastic flight.
        rail_length : int, float, tuple, list, optional
            The rail length of the flight.
        inclination : int, float, tuple, list, optional
            The inclination of the launch rail.
        heading : int, float, tuple, list, optional
            The heading of the launch rail.
        initial_solution : tuple, list, optional
            The initial solution of the flight. This is a tuple of 14 elements
            that represent the initial conditions of the flight. This attribute
            can not be randomized.
        terminate_on_apogee : bool, optional
            Whether or not the flight should terminate on apogee. This attribute
            can not be randomized.
        time_overshoot : bool
            If False, the simulation will run at the time step defined by the controller
            sampling rate. Be aware that this will make the simulation run much slower.
        max_time : int, float
            The maximum time of the flight simulation. If the flight simulation
            reaches this time, it will terminate. This attribute can not be randomized.
        """
        if terminate_on_apogee is not None:
            if not isinstance(terminate_on_apogee, bool):
                raise AssertionError("`terminate_on_apogee` must be a boolean")
        if time_overshoot is not None:
            if not isinstance(time_overshoot, bool):
                raise TypeError("`time_overshoot` must be a boolean")
        if max_time is not None:
            if not isinstance(max_time, (int, float)):
                raise TypeError("`max_time` must be a number")
        super().__init__(
            flight,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
        )

        self._validate_initial_solution(initial_solution)
        self.initial_solution = initial_solution
        self.terminate_on_apogee = terminate_on_apogee
        if max_time is None:
            self.max_time = flight.max_time
        else:
            self.max_time = max_time
        if time_overshoot is None:
            self.time_overshoot = flight.time_overshoot
        else:
            self.time_overshoot = time_overshoot

    def _validate_initial_solution(self, initial_solution):
        if initial_solution is not None:
            if isinstance(initial_solution, (tuple, list)):
                if not len(initial_solution) == 14:
                    raise AssertionError(
                        "`initial_solution` must be a 14 element tuple, the "
                        "elements are:\n t_initial, x_init, y_init, z_init, "
                        "vx_init, vy_init, vz_init, e0_init, e1_init, e2_init, "
                        "e3_init, w1Init, w2Init, w3Init"
                    )
                if not all(isinstance(i, (int, float)) for i in initial_solution):
                    raise AssertionError(
                        "`initial_solution` must be a tuple of numbers"
                    )
            else:
                raise TypeError("`initial_solution` must be a tuple of numbers")

    def _sample_flight_inputs(self):
        """Sample rail_length, inclination, and heading in a single draw.

        Returns
        -------
        dict
            Mapping with keys ``rail_length``, ``inclination``, and ``heading``.
            Also updates ``last_rnd_dict``.
        """
        return next(self.dict_generator())

    def _randomize_rail_length(self):
        """Randomizes the rail length of the flight."""
        return self._sample_flight_inputs()["rail_length"]

    def _randomize_inclination(self):
        """Randomizes the inclination of the flight."""
        return self._sample_flight_inputs()["inclination"]

    def _randomize_heading(self):
        """Randomizes the heading of the flight."""
        return self._sample_flight_inputs()["heading"]

    def create_object(self):
        """Creates and returns a Flight object from the randomly generated input
        arguments.

        Returns
        -------
        flight : Flight
            Flight object with the randomly generated input arguments.
        """
        generated_dict = self._sample_flight_inputs()
        return Flight(
            rocket=self.obj.rocket,
            environment=self.obj.env,
            rail_length=generated_dict["rail_length"],
            inclination=generated_dict["inclination"],
            heading=generated_dict["heading"],
            initial_solution=self.initial_solution,
            terminate_on_apogee=self.terminate_on_apogee,
            max_time=self.max_time,
            max_time_step=self.obj.max_time_step,
            min_time_step=self.obj.min_time_step,
            rtol=self.obj.rtol,
            atol=self.obj.atol,
            time_overshoot=self.time_overshoot,
            name=self.obj.name,
            equations_of_motion=self.obj.equations_of_motion,
            ode_solver=self.obj.ode_solver,
            simulation_mode=self.obj.simulation_mode,
        )
