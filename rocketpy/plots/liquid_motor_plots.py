__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _LiquidMotorPlots:
    """Class that holds plot methods for LiquidMotor class.

    Attributes
    ----------
    _LiquidMotorPlots.liquid_motor : LiquidMotor
        LiquidMotor object that will be used for the plots.

    """

    def __init__(self, liquid_motor):
        """Initializes _MotorClass class.

        Parameters
        ----------
        liquid_motor : LiquidMotor
            Instance of the LiquidMotor class

        Returns
        -------
        None
        """

        self.liquid_motor = liquid_motor

        return None

    def thrust(self, lower_limit=None, upper_limit=None):
        """Plots thrust of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.thrust.plot(lower=lower_limit, upper=upper_limit)

        return None

    def total_mass(self, lower_limit=None, upper_limit=None):
        """Plots total_mass of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.total_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def mass_flow_rate(self, lower_limit=None, upper_limit=None):
        """Plots mass_flow_rate of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.mass_flow_rate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def exhaust_velocity(self, lower_limit=None, upper_limit=None):
        """Plots exhaust_velocity of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.exhaust_velocity.plot(lower=lower_limit, upper=upper_limit)

        return None

    def center_of_mass(self, lower_limit=None, upper_limit=None):
        """Plots center_of_mass of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.center_of_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_11(self, lower_limit=None, upper_limit=None):
        """Plots I_11 of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.I_11.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_22(self, lower_limit=None, upper_limit=None):
        """Plots I_22 of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.I_22.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_33(self, lower_limit=None, upper_limit=None):
        """Plots I_33 of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.I_33.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_12(self, lower_limit=None, upper_limit=None):
        """Plots I_12 of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.I_12.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_13(self, lower_limit=None, upper_limit=None):
        """Plots I_13 of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.I_13.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_23(self, lower_limit=None, upper_limit=None):
        """Plots I_23 of the liquid_motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is None, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is None, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """

        self.liquid_motor.I_23.plot(lower=lower_limit, upper=upper_limit)

        return None

    def all(self):
        """Prints out all graphs available about the LiquidMotor. It simply calls
        all the other plotter methods in this class.

        Parameters
        ----------
        None
        Return
        ------
        None
        """

        self.thrust(*self.liquid_motor.burn_time)
        self.total_mass(*self.liquid_motor.burn_time)
        self.mass_flow_rate(*self.liquid_motor.burn_time)
        self.exhaust_velocity(*self.liquid_motor.burn_time)
        self.center_of_mass(*self.liquid_motor.burn_time)
        self.I_11(*self.liquid_motor.burn_time)
        self.I_22(*self.liquid_motor.burn_time)
        self.I_33(*self.liquid_motor.burn_time)
        self.I_12(*self.liquid_motor.burn_time)
        self.I_13(*self.liquid_motor.burn_time)
        self.I_23(*self.liquid_motor.burn_time)

        return None
