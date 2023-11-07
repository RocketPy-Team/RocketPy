from .motor_plots import _MotorPlots


class _SolidMotorPlots(_MotorPlots):
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

        super().__init__(solid_motor)

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

        self.motor.grain_inner_radius.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.grain_height.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.burn_rate.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.burn_area.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.Kn.plot(lower=lower_limit, upper=upper_limit)

    def all(self):
        """Prints out all graphs available about the SolidMotor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """

        self.thrust(*self.motor.burn_time)
        self.total_mass(*self.motor.burn_time)
        self.center_of_mass(*self.motor.burn_time)
        self.mass_flow_rate(*self.motor.burn_time)
        self.exhaust_velocity(*self.motor.burn_time)
        self.grain_inner_radius(*self.motor.burn_time)
        self.grain_height(*self.motor.burn_time)
        self.burn_rate(self.motor.burn_time[0], self.motor.grain_burn_out)
        self.burn_area(*self.motor.burn_time)
        self.Kn()
        self.I_11(*self.motor.burn_time)
        self.I_22(*self.motor.burn_time)
        self.I_33(*self.motor.burn_time)
        self.I_12(*self.motor.burn_time)
        self.I_13(*self.motor.burn_time)
        self.I_23(*self.motor.burn_time)
