__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _SolidMotorPlots:
    """Class that holds plot methods for SolidMotor class.

    Attributes
    ----------
    _SolidMotorPlots.solid_motor : SolidMotor
        SolidMotor object that will be used for the plots.

    """

    def __init__(self, solid_motor):
        """Initializes _MotorClass class.

        Parameters
        ----------
        solid_motor : SolidMotor
            Instance of the SolidMotor class

        Returns
        -------
        None
        """

        self.solid_motor = solid_motor

        return None

    def thrust(self, lower_limit=None, upper_limit=None):
        """Plots thrust of the solid_motor as a function of time.

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

        self.solid_motor.thrust.plot(lower=lower_limit, upper=upper_limit)

        return None

    def total_mass(self, lower_limit=None, upper_limit=None):
        """Plots total_mass of the solid_motor as a function of time.

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

        self.solid_motor.total_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def mass_flow_rate(self, lower_limit=None, upper_limit=None):
        """Plots mass_flow_rate of the solid_motor as a function of time.

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

        self.solid_motor.mass_flow_rate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def exhaust_velocity(self, lower_limit=None, upper_limit=None):
        """Plots exhaust_velocity of the solid_motor as a function of time.

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

        self.solid_motor.exhaust_velocity.plot(lower=lower_limit, upper=upper_limit)

        return None

    def grain_inner_radius(self, lower_limit=None, upper_limit=None):
        """Plots grain_inner_radius of the solid_motor as a function of time.

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

        self.solid_motor.grain_inner_radius.plot(lower=lower_limit, upper=upper_limit)

        return None

    def grain_height(self, lower_limit=None, upper_limit=None):
        """Plots grain_height of the solid_motor as a function of time.

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

        self.solid_motor.grain_height.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burn_rate(self, lower_limit=None, upper_limit=None):
        """Plots burn_rate of the solid_motor as a function of time.

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

        self.solid_motor.burn_rate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burn_area(self, lower_limit=None, upper_limit=None):
        """Plots burn_area of the solid_motor as a function of time.

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

        self.solid_motor.burn_area.plot(lower=lower_limit, upper=upper_limit)

        return None

    def Kn(self, lower_limit=None, upper_limit=None):
        """Plots Kn of the solid_motor as a function of time.

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

        self.solid_motor.Kn.plot(lower=lower_limit, upper=upper_limit)

        return None

    def center_of_mass(self, lower_limit=None, upper_limit=None):
        """Plots center_of_mass of the solid_motor as a function of time.

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

        self.solid_motor.center_of_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_11(self, lower_limit=None, upper_limit=None):
        """Plots I_11 of the solid_motor as a function of time.

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

        self.solid_motor.I_11.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_22(self, lower_limit=None, upper_limit=None):
        """Plots I_22 of the solid_motor as a function of time.

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

        self.solid_motor.I_22.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_33(self, lower_limit=None, upper_limit=None):
        """Plots I_33 of the solid_motor as a function of time.

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

        self.solid_motor.I_33.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_12(self, lower_limit=None, upper_limit=None):
        """Plots I_12 of the solid_motor as a function of time.

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

        self.solid_motor.I_12.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_13(self, lower_limit=None, upper_limit=None):
        """Plots I_13 of the solid_motor as a function of time.

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

        self.solid_motor.I_13.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_23(self, lower_limit=None, upper_limit=None):
        """Plots I_23 of the solid_motor as a function of time.

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

        self.solid_motor.I_23.plot(lower=lower_limit, upper=upper_limit)

        return None

    def all(self):
        """Prints out all graphs available about the SolidMotor. It simply calls
        all the other plotter methods in this class.

        Parameters
        ----------
        None
        Return
        ------
        None
        """

        self.thrust(*self.solid_motor.burn_time)
        self.total_mass(*self.solid_motor.burn_time)
        self.mass_flow_rate(*self.solid_motor.burn_time)
        self.exhaust_velocity(*self.solid_motor.burn_time)
        self.grain_inner_radius(*self.solid_motor.burn_time)
        self.grain_height(*self.solid_motor.burn_time)
        self.burn_rate(self.solid_motor.burn_time[0], self.solid_motor.grainBurnOut)
        self.burn_area(*self.solid_motor.burn_time)
        self.Kn()
        self.center_of_mass(*self.solid_motor.burn_time)
        self.I_11(*self.solid_motor.burn_time)
        self.I_22(*self.solid_motor.burn_time)
        self.I_33(*self.solid_motor.burn_time)
        self.I_12(*self.solid_motor.burn_time)
        self.I_13(*self.solid_motor.burn_time)
        self.I_23(*self.solid_motor.burn_time)

        return None
