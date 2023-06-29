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

    def totalMass(self, lower_limit=None, upper_limit=None):
        """Plots totalMass of the solid_motor as a function of time.

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

        self.solid_motor.totalMass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def massFlowRate(self, lower_limit=None, upper_limit=None):
        """Plots massFlowRate of the solid_motor as a function of time.

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

        self.solid_motor.massFlowRate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def exhaustVelocity(self, lower_limit=None, upper_limit=None):
        """Plots exhaustVelocity of the solid_motor as a function of time.

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

        self.solid_motor.exhaustVelocity.plot(lower=lower_limit, upper=upper_limit)

        return None

    def grainInnerRadius(self, lower_limit=None, upper_limit=None):
        """Plots grainInnerRadius of the solid_motor as a function of time.

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

        self.solid_motor.grainInnerRadius.plot(lower=lower_limit, upper=upper_limit)

        return None

    def grainHeight(self, lower_limit=None, upper_limit=None):
        """Plots grainHeight of the solid_motor as a function of time.

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

        self.solid_motor.grainHeight.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burnRate(self, lower_limit=None, upper_limit=None):
        """Plots burnRate of the solid_motor as a function of time.

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

        self.solid_motor.burnRate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burnArea(self, lower_limit=None, upper_limit=None):
        """Plots burnArea of the solid_motor as a function of time.

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

        self.solid_motor.burnArea.plot(lower=lower_limit, upper=upper_limit)

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

    def centerOfMass(self, lower_limit=None, upper_limit=None):
        """Plots centerOfMass of the solid_motor as a function of time.

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

        self.solid_motor.centerOfMass.plot(lower=lower_limit, upper=upper_limit)

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
        self.totalMass(*self.solid_motor.burn_time)
        self.massFlowRate(*self.solid_motor.burn_time)
        self.exhaustVelocity(*self.solid_motor.burn_time)
        self.grainInnerRadius(*self.solid_motor.burn_time)
        self.grainHeight(*self.solid_motor.burn_time)
        self.burnRate(self.solid_motor.burn_time[0], self.solid_motor.grainBurnOut)
        self.burnArea(*self.solid_motor.burn_time)
        self.Kn()
        self.centerOfMass(*self.solid_motor.burn_time)
        self.I_11(*self.solid_motor.burn_time)
        self.I_22(*self.solid_motor.burn_time)
        self.I_33(*self.solid_motor.burn_time)
        self.I_12(*self.solid_motor.burn_time)
        self.I_13(*self.solid_motor.burn_time)
        self.I_23(*self.solid_motor.burn_time)

        return None
