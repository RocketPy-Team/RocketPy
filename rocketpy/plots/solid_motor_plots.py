import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from rocketpy.plots import _generate_nozzle

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

    def draw(self):
        """Draws a simple 2D representation of the SolidMotor."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect("equal")

        _csys = self.solid_motor._csys

        chamber = self._generate_combustion_chamber(_csys=_csys)
        nozzle = _generate_nozzle(self.solid_motor)
        grains = self._generate_grains(_csys=_csys, translate=(0, 0))

        ax.add_patch(chamber)
        for grain in grains:
            ax.add_patch(grain)
        ax.add_patch(nozzle)

        # self._draw_nozzle(ax, nozzle_height, csys)
        # self._draw_combustion_chamber(ax, csys)
        # self._draw_grains(ax, csys)
        self._draw_center_of_mass(ax)

        self._set_plot_properties(ax)
        plt.show()
        return None

    def _generate_combustion_chamber(self, translate=(0, 0), csys=1):
        # csys = self.solid_motor.csys
        chamber_length = (
            abs(
                self.solid_motor.center_of_dry_mass_position
                - self.solid_motor.nozzle_position
            )
            * 2
        ) * csys
        x = np.array(
            [
                self.solid_motor.nozzle_position,
                self.solid_motor.nozzle_position,
                self.solid_motor.nozzle_position + chamber_length,
                self.solid_motor.nozzle_position + chamber_length,
            ]
        )
        y = np.array(
            [
                self.solid_motor.nozzle_radius,
                self.solid_motor.grain_outer_radius * 1.4,
                self.solid_motor.grain_outer_radius * 1.4,
                0,
            ]
        )
        # we need to draw the other half of the nozzle
        x = np.concatenate([x, x[::-1]])
        y = np.concatenate([y, -y[::-1]])
        # now we need to sum the  the translate
        x = x + translate[0]
        y = y + translate[1]

        patch = Polygon(
            np.column_stack([x, y]),
            label="Combustion Chamber",
            facecolor="lightslategray",
            edgecolor="black",
        )
        return patch

    def _generate_grains(self, translate=(0, 0), csys=1):
        patches = []
        n_total = self.solid_motor.grain_number
        separation = self.solid_motor.grain_separation
        height = self.solid_motor.grain_initial_height
        outer_radius = self.solid_motor.grain_outer_radius
        inner_radius = self.solid_motor.grain_initial_inner_radius

        cm_teo = (
            csys * ((n_total / 2) * (height + separation))
            + self.solid_motor.nozzle_position
        )
        cm_real = self.solid_motor.center_of_propellant_mass(0)

        init = abs(cm_teo - cm_real) * csys

        inner_y = np.array([0, inner_radius, inner_radius, 0])
        outer_y = np.array([inner_radius, outer_radius, outer_radius, inner_radius])
        inner_y = np.concatenate([inner_y, -inner_y[::-1]])
        outer_y = np.concatenate([outer_y, -outer_y[::-1]])
        inner_y = inner_y + translate[1]
        outer_y = outer_y + translate[1]
        for n in range(n_total):
            grain_start = init + csys * (separation / 2 + n * (height + separation))
            grain_end = grain_start + height * csys
            x = np.array([grain_start, grain_start, grain_end, grain_end])
            # draw the other half of the nozzle
            x = np.concatenate([x, x[::-1]])
            # sum the translate
            x = x + translate[0]
            patch = Polygon(
                np.column_stack([x, outer_y]),
                facecolor="olive",
                edgecolor="khaki",
            )
            patches.append(patch)

            patch = Polygon(
                np.column_stack([x, inner_y]),
                facecolor="khaki",
                edgecolor="olive",
            )
            if n == 0:
                patch.set_label("Grains")
            patches.append(patch)
        return patches

    def _draw_center_of_mass(self, ax):
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)  # symmetry line
        ax.plot(
            [self.solid_motor.grains_center_of_mass_position],
            [0],
            "ro",
            label="Grains Center of Mass",
        )
        ax.plot(
            [self.solid_motor.center_of_dry_mass_position],
            [0],
            "bo",
            label="Center of Dry Mass",
        )

    def _set_plot_properties(self, ax):
        ax.set_title("Solid Motor Representation")
        ax.set_ylabel("Radius (m)")
        ax.set_xlabel("Position (m)")
        # ax.grid(True)
        plt.ylim(
            -self.solid_motor.grain_outer_radius * 1.2 * 1.7,
            self.solid_motor.grain_outer_radius * 1.2 * 1.7,
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    def all(self):
        """Prints out all graphs available about the SolidMotor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.thrust(*self.motor.burn_time)
        self.mass_flow_rate(*self.motor.burn_time)
        self.exhaust_velocity(*self.motor.burn_time)
        self.total_mass(*self.motor.burn_time)
        self.propellant_mass(*self.motor.burn_time)
        self.center_of_mass(*self.motor.burn_time)
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
