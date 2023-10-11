import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from rocketpy.plots import _generate_nozzle


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

    def _generate_positioned_tanks(self, translate=(0, 0), csys=1):
        """Generates a list of patches that represent the tanks of the
        liquid_motor.

        Parameters
        ----------
        None

        Returns
        -------
        patches : list
            List of patches that represent the tanks of the liquid_motor.
        """
        colors = {
            0: ("black", "dimgray"),
            1: ("darkblue", "cornflowerblue"),
            2: ("darkgreen", "limegreen"),
            3: ("darkorange", "gold"),
            4: ("darkred", "tomato"),
            5: ("darkviolet", "violet"),
        }
        patches = []
        for idx, pos_tank in enumerate(self.liquid_motor.positioned_tanks):
            tank = pos_tank["tank"]
            position = pos_tank["position"]
            trans = (position + translate[0], translate[1])
            patch = tank.plots._generate_tank(trans, csys)
            patch.set_facecolor(colors[idx][1])
            patch.set_edgecolor(colors[idx][0])
            patches.append(patch)
        return patches

    def _draw_center_of_interests(self, ax, translate=(0, 0)):
        # center of dry mass position
        # center of wet mass time = 0
        # center of wet mass time = end
        return None

    def draw(self):
        fig, ax = plt.subplots(facecolor="#EEEEEE")

        patches = self._generate_positioned_tanks()
        for patch in patches:
            ax.add_patch(patch)

        # add the nozzle
        ax.add_patch(_generate_nozzle(self.liquid_motor, translate=(0, 0)))

        # find the maximum and minimum x and y values of the tanks
        x_min = y_min = np.inf
        x_max = y_max = -np.inf
        for patch in patches:
            x_min = min(x_min, patch.xy[:, 0].min())
            x_max = max(x_max, patch.xy[:, 0].max())
            y_min = min(y_min, patch.xy[:, 1].min())
            y_max = max(y_max, patch.xy[:, 1].max())

        ax.set_aspect("equal")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_ylim(y_min - 0.25, y_max + 0.25)
        ax.set_xlim(x_min - 0.10, x_max + 0.10)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Radius (m)")
        ax.set_title("Liquid Motor Geometry")
        plt.show()

    def all(self):
        """Prints out all graphs available about the LiquidMotor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
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
