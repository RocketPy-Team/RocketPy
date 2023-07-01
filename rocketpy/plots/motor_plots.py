__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _MotorPlots:
    """Class that holds plot methods for Motor class.

    Attributes
    ----------
    _MotorPlots.motor : Motor
        Motor object that will be used for the plots.

    """

    def __init__(self, motor):
        """Initializes _MotorClass class.

        Parameters
        ----------
        motor : Motor
            Instance of the Motor class

        Returns
        -------
        None
        """

        self.motor = motor

        return None

    def thrust(self, lower_limit=None, upper_limit=None):
        """Plots thrust of the motor as a function of time.

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

        self.motor.thrust.plot(lower=lower_limit, upper=upper_limit)

        return None

    def total_mass(self, lower_limit=None, upper_limit=None):
        """Plots total_mass of the motor as a function of time.

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

        self.motor.total_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def center_of_mass(self, lower_limit=None, upper_limit=None):
        """Plots center_of_mass of the motor as a function of time.

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

        self.motor.center_of_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_11(self, lower_limit=None, upper_limit=None):
        """Plots I_11 of the motor as a function of time.

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

        self.motor.I_11.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_22(self, lower_limit=None, upper_limit=None):
        """Plots I_22 of the motor as a function of time.

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

        self.motor.I_22.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_33(self, lower_limit=None, upper_limit=None):
        """Plots I_33 of the motor as a function of time.

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

        self.motor.I_33.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_12(self, lower_limit=None, upper_limit=None):
        """Plots I_12 of the motor as a function of time.

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

        self.motor.I_12.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_13(self, lower_limit=None, upper_limit=None):
        """Plots I_13 of the motor as a function of time.

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

        self.motor.I_13.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_23(self, lower_limit=None, upper_limit=None):
        """Plots I_23 of the motor as a function of time.

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

        self.motor.I_23.plot(lower=lower_limit, upper=upper_limit)

        return None

    def all(self):
        """Prints out all graphs available about the Motor. It simply calls
        all the other plotter methods in this class.

        Parameters
        ----------
        None
        Return
        ------
        None
        """

        # Show plots
        self.thrust()
        self.total_mass()
        self.center_of_mass()
        self.I_11()
        self.I_22()
        self.I_33()
        self.I_12()
        self.I_13()
        self.I_23()

        return None
