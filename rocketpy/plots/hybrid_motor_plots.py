__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _HybridMotorPlots:
    """Class that holds plot methods for HybridMotor class.

    Attributes
    ----------
    _HybridMotorPlots.hybrid_motor : HybridMotor
        HybridMotor object that will be used for the plots.

    """

    def __init__(self, hybrid_motor):
        """Initializes _MotorClass class.

        Parameters
        ----------
        hybrid_motor : HybridMotor
            Instance of the HybridMotor class

        Returns
        -------
        None
        """

        self.hybrid_motor = hybrid_motor

        return None

    def thrust(self, lower_limit=None, upper_limit=None):
        """Plots thrust of the hybrid_motor as a function of time.

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

        self.hybrid_motor.thrust.plot(lower=lower_limit, upper=upper_limit)

        return None

    def total_mass(self, lower_limit=None, upper_limit=None):
        """Plots total_mass of the hybrid_motor as a function of time.

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

        self.hybrid_motor.total_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def mass_flow_rate(self, lower_limit=None, upper_limit=None):
        """Plots mass_flow_rate of the hybrid_motor as a function of time.

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

        self.hybrid_motor.mass_flow_rate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def exhaust_velocity(self, lower_limit=None, upper_limit=None):
        """Plots exhaust_velocity of the hybrid_motor as a function of time.

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

        self.hybrid_motor.exhaust_velocity.plot(lower=lower_limit, upper=upper_limit)

        return None

    def grain_inner_radius(self, lower_limit=None, upper_limit=None):
        """Plots grain_inner_radius of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.grain_inner_radius.plot(
            lower=lower_limit, upper=upper_limit
        )

        return None

    def grain_height(self, lower_limit=None, upper_limit=None):
        """Plots grain_height of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.grain_height.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burn_rate(self, lower_limit=None, upper_limit=None):
        """Plots burn_rate of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.burn_rate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burn_area(self, lower_limit=None, upper_limit=None):
        """Plots burn_area of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.burn_area.plot(lower=lower_limit, upper=upper_limit)

        return None

    def Kn(self, lower_limit=None, upper_limit=None):
        """Plots Kn of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.Kn.plot(lower=lower_limit, upper=upper_limit)

        return None

    def center_of_mass(self, lower_limit=None, upper_limit=None):
        """Plots center_of_mass of the hybrid_motor as a function of time.

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

        self.hybrid_motor.center_of_mass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_11(self, lower_limit=None, upper_limit=None):
        """Plots I_11 of the hybrid_motor as a function of time.

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

        self.hybrid_motor.I_11.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_22(self, lower_limit=None, upper_limit=None):
        """Plots I_22 of the hybrid_motor as a function of time.

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

        self.hybrid_motor.I_22.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_33(self, lower_limit=None, upper_limit=None):
        """Plots I_33 of the hybrid_motor as a function of time.

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

        self.hybrid_motor.I_33.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_12(self, lower_limit=None, upper_limit=None):
        """Plots I_12 of the hybrid_motor as a function of time.

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

        self.hybrid_motor.I_12.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_13(self, lower_limit=None, upper_limit=None):
        """Plots I_13 of the hybrid_motor as a function of time.

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

        self.hybrid_motor.I_13.plot(lower=lower_limit, upper=upper_limit)

        return None

    def I_23(self, lower_limit=None, upper_limit=None):
        """Plots I_23 of the hybrid_motor as a function of time.

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

        self.hybrid_motor.I_23.plot(lower=lower_limit, upper=upper_limit)

        return None

    def all(self):
        """Prints out all graphs available about the HybridMotor. It simply calls
        all the other plotter methods in this class.

        Parameters
        ----------
        None
        Return
        ------
        None
        """

        self.thrust(*self.hybrid_motor.burn_time)
        self.total_mass(*self.hybrid_motor.burn_time)
        self.mass_flow_rate(*self.hybrid_motor.burn_time)
        self.exhaust_velocity(*self.hybrid_motor.burn_time)
        self.grain_inner_radius(*self.hybrid_motor.burn_time)
        self.grain_height(*self.hybrid_motor.burn_time)
        self.burn_rate(
            self.hybrid_motor.burn_time[0], self.hybrid_motor.solid.grainBurnOut
        )
        self.burn_area(*self.hybrid_motor.burn_time)
        self.Kn()
        self.center_of_mass(*self.hybrid_motor.burn_time)
        self.I_11(*self.hybrid_motor.burn_time)
        self.I_22(*self.hybrid_motor.burn_time)
        self.I_33(*self.hybrid_motor.burn_time)
        self.I_12(*self.hybrid_motor.burn_time)
        self.I_13(*self.hybrid_motor.burn_time)
        self.I_23(*self.hybrid_motor.burn_time)

        return None
