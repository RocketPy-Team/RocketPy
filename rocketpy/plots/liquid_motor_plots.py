import matplotlib.pyplot as plt

from .motor_plots import _MotorPlots


class _LiquidMotorPlots(_MotorPlots):
    """Class that holds plot methods for LiquidMotor class.

    Attributes
    ----------
    _LiquidMotorPlots.liquid_motor : LiquidMotor
        LiquidMotor object that will be used for the plots.

    """

    def draw(self):
        """Draw a representation of the LiquidMotor.

        Returns
        -------
        None
        """
        _, ax = plt.subplots(figsize=(8, 6), facecolor="#EEEEEE")

        tanks_and_centers = self._generate_positioned_tanks(csys=self.motor._csys)
        nozzle = self._generate_nozzle(
            translate=(self.motor.nozzle_position, 0), csys=self.motor._csys
        )
        outline = self._generate_motor_region(
            list_of_patches=[nozzle] + [tank for tank, _ in tanks_and_centers]
        )

        ax.add_patch(outline)
        for patch, center in tanks_and_centers:
            ax.add_patch(patch)
            ax.plot(center[0], center[1], marker="o", color="red", markersize=2)

        # add the nozzle
        ax.add_patch(nozzle)

        ax.set_title("Liquid Motor Representation")
        self._draw_center_of_mass(ax)
        self._set_plot_properties(ax)
        plt.show()

    def all(self):
        """Prints out all graphs available about the LiquidMotor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.draw()
        self.thrust(*self.motor.burn_time)
        self.mass_flow_rate(*self.motor.burn_time)
        self.exhaust_velocity(*self.motor.burn_time)
        self.total_mass(*self.motor.burn_time)
        self.propellant_mass(*self.motor.burn_time)
        self.center_of_mass(*self.motor.burn_time)
        self.inertia_tensor(*self.motor.burn_time)
