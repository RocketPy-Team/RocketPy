from rocketpy.simulation import Flight

from .stochastic_model import StochasticModel


class StochasticFlight(StochasticModel):
    """A Stochastic Flight class that inherits from StochasticModel. This
    class is used to receive a Flight object and information about the
    dispersion of its parameters and generate a random flight object based on
    the provided information.

    Attributes
    ----------
    flight : Flight
        The Flight object to be used as a base for the Stochastic flight.
    rail_length : int, float, tuple, list, optional
        The rail length of the flight. Follows the standard input format of
        Stochastic Models.
    inclination : int, float, tuple, list, optional
        The inclination of the flight. Follows the standard input format of
        Stochastic Models.
    heading : int, float, tuple, list, optional
        The heading of the flight. Follows the standard input format of
        Stochastic Models.
    initial_solution : tuple, list, optional
        The initial solution of the flight. This is a tuple of 14 elements that
        represent the initial conditions of the flight. This attribute can not
        be randomized.
    terminate_on_apogee : bool, optional
        Whether or not the flight should terminate on apogee. This attribute
        can not be randomized.
    """

    def __init__(
        self,
        flight,
        rail_length=None,
        inclination=None,
        heading=None,
        initial_solution=None,
        terminate_on_apogee=None,
    ):
        """Initializes the Stochastic Flight class.

        See Also
        --------
        This should link to somewhere that explains how inputs works in
        Stochastic models.

        Parameters
        ----------
        flight : Flight
            The Flight object to be used as a base for the Stochastic flight.
        rail_length : int, float, tuple, list, optional
            The rail length of the flight. Follows the standard input format of
            Stochastic Models.
        inclination : int, float, tuple, list, optional
            The inclination of the flight. Follows the standard input format of
            Stochastic Models.
        heading : int, float, tuple, list, optional
            The heading of the flight. Follows the standard input format of
            Stochastic Models.
        initial_solution : tuple, list, optional
            The initial solution of the flight. This is a tuple of 14 elements
            that represent the initial conditions of the flight. This attribute
            can not be randomized.
        terminate_on_apogee : bool, optional
            Whether or not the flight should terminate on apogee. This attribute
            can not be randomized.
        """
        if terminate_on_apogee is not None:
            assert isinstance(
                terminate_on_apogee, bool
            ), "`terminate_on_apogee` must be a boolean"
        super().__init__(
            flight,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
        )

        self.initial_solution = initial_solution
        self.terminate_on_apogee = terminate_on_apogee

    def _validate_initial_solution(self, initial_solution):
        if initial_solution is not None:
            if isinstance(initial_solution, (tuple, list)):
                assert len(initial_solution) == 14, (
                    "`initial_solution` must be a 14 element tuple, the "
                    "elements are:\n t_initial, x_init, y_init, z_init, "
                    "vx_init, vy_init, vz_init, e0_init, e1_init, e2_init, "
                    "e3_init, w1Init, w2Init, w3Init"
                )
                assert all(
                    [isinstance(i, (int, float)) for i in initial_solution]
                ), "`initial_solution` must be a tuple of numbers"
            else:
                raise TypeError("`initial_solution` must be a tuple of numbers")

    # TODO: these call dict_generator a lot of times unecessaryly
    def _randomize_rail_length(self):
        """Randomizes the rail length of the flight. Follows the standard input
        format of Stochastic Models.
        """
        generated_dict = next(self.dict_generator())
        return generated_dict["rail_length"]

    def _randomize_inclination(self):
        """Randomizes the inclination of the flight. Follows the standard input
        format of Stochastic Models.
        """
        generated_dict = next(self.dict_generator())
        return generated_dict["inclination"]

    def _randomize_heading(self):
        """Randomizes the heading of the flight. Follows the standard input
        format of Stochastic Models.
        """
        generated_dict = next(self.dict_generator())
        return generated_dict["heading"]

    def create_object(self):
        """Creates and returns a Flight object from the randomly generated input
        arguments.

        Returns
        -------
        flight : Flight
            Flight object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())
        obj = Flight(
            environment=self.object.env,
            rocket=self.object.rocket,
            inclination=generated_dict["inclination"],
            heading=generated_dict["heading"],
            initialSolution=self.initial_solution,
            terminateOnApogee=self.terminate_on_apogee,
        )
        return obj
