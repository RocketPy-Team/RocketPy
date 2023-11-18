from matplotlib import pyplot as plt

from .motor_plots import _MotorPlots


class _HybridMotorPlots(_MotorPlots):
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
        super().__init__(hybrid_motor)

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

        self.motor.solid.grain_inner_radius.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.solid.grain_height.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.solid.burn_rate.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.solid.burn_area.plot(lower=lower_limit, upper=upper_limit)

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

        self.motor.solid.Kn.plot(lower=lower_limit, upper=upper_limit)

    def draw(self):
        """Draw a representation of the HybridMotor.

        Returns
        -------
        None
        """
        _, ax = plt.subplots(figsize=(8, 6), facecolor="#EEEEEE")

        tanks_and_centers = self._generate_positioned_tanks(csys=self.motor._csys)
        nozzle = self._generate_nozzle(
            translate=(self.motor.nozzle_position, 0), csys=self.motor._csys
        )
        chamber = self._generate_combustion_chamber(
            translate=(self.motor.grains_center_of_mass_position, 0)
        )
        grains = self._generate_grains(
            translate=(self.motor.grains_center_of_mass_position, 0)
        )
        outline = self._generate_motor_region(
            list_of_patches=[nozzle, chamber, *grains]
            + [tank for tank, _ in tanks_and_centers]
        )

        ax.add_patch(outline)
        ax.add_patch(chamber)
        for grain in grains:
            ax.add_patch(grain)
        for patch, center in tanks_and_centers:
            ax.add_patch(patch)
            ax.plot(center[0], center[1], marker="o", color="red", markersize=2)
        ax.add_patch(nozzle)

        ax.set_title("Hybrid Motor Representation")
        self._draw_center_of_mass(ax)
        self._set_plot_properties(ax)
        plt.show()

    def all(self):
        """Prints out all graphs available about the HybridMotor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.draw()
        self.thrust(*self.motor.burn_time)
        self.total_mass(*self.motor.burn_time)
        self.center_of_mass(*self.motor.burn_time)
        self.mass_flow_rate(*self.motor.burn_time)
        self.exhaust_velocity(*self.motor.burn_time)
        self.grain_inner_radius(*self.motor.burn_time)
        self.grain_height(*self.motor.burn_time)
        self.burn_rate(self.motor.burn_time[0], self.motor.solid.grain_burn_out)
        self.burn_area(*self.motor.burn_time)
        self.Kn()
        self.I_11(*self.motor.burn_time)
        self.I_22(*self.motor.burn_time)
        self.I_33(*self.motor.burn_time)
        self.I_12(*self.motor.burn_time)
        self.I_13(*self.motor.burn_time)
        self.I_23(*self.motor.burn_time)
