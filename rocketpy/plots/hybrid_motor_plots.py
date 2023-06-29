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

    def totalMass(self, lower_limit=None, upper_limit=None):
        """Plots totalMass of the hybrid_motor as a function of time.

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

        self.hybrid_motor.totalMass.plot(lower=lower_limit, upper=upper_limit)

        return None

    def massFlowRate(self, lower_limit=None, upper_limit=None):
        """Plots massFlowRate of the hybrid_motor as a function of time.

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

        self.hybrid_motor.massFlowRate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def exhaustVelocity(self, lower_limit=None, upper_limit=None):
        """Plots exhaustVelocity of the hybrid_motor as a function of time.

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

        self.hybrid_motor.exhaustVelocity.plot(lower=lower_limit, upper=upper_limit)

        return None

    def grainInnerRadius(self, lower_limit=None, upper_limit=None):
        """Plots grainInnerRadius of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.grainInnerRadius.plot(
            lower=lower_limit, upper=upper_limit
        )

        return None

    def grainHeight(self, lower_limit=None, upper_limit=None):
        """Plots grainHeight of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.grainHeight.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burnRate(self, lower_limit=None, upper_limit=None):
        """Plots burnRate of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.burnRate.plot(lower=lower_limit, upper=upper_limit)

        return None

    def burnArea(self, lower_limit=None, upper_limit=None):
        """Plots burnArea of the hybrid_motor as a function of time.

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

        self.hybrid_motor.solid.burnArea.plot(lower=lower_limit, upper=upper_limit)

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

    def centerOfMass(self, lower_limit=None, upper_limit=None):
        """Plots centerOfMass of the hybrid_motor as a function of time.

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

        self.hybrid_motor.centerOfMass.plot(lower=lower_limit, upper=upper_limit)

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

        self.thrust(*self.burn_time)
        self.totalMass(*self.burn_time)
        self.massFlowRate(*self.burn_time)
        self.exhaustVelocity(*self.burn_time)
        self.grainInnerRadius(*self.burn_time)
        self.grainHeight(*self.burn_time)
        self.burnRate(self.burn_time[0], self.grainBurnOut)
        self.burnArea(*self.burn_time)
        self.Kn()
        self.centerOfMass(*self.burn_time)
        self.I_11(*self.burn_time)
        self.I_22(*self.burn_time)
        self.I_33(*self.burn_time)
        self.I_12(*self.burn_time)
        self.I_13(*self.burn_time)
        self.I_23(*self.burn_time)

        return None
