# pylint: disable=too-many-statements
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .plot_helpers import show_or_save_plot


class _GenericSurfacePlots:
    """Base plots for a generic aerodynamic surface."""

    def __init__(self, aero_surface):
        """Initialize the class

        Parameters
        ----------
        aero_surface : rocketpy.GenericSurface
            Aerodynamic surface object to be plotted

        Returns
        -------
        None
        """
        self.aero_surface = aero_surface

    def draw(self, *, filename=None):
        """A plain generic surface has no geometry to draw."""

    # For each body-frame coefficient, the inputs worth sweeping, most relevant
    # first: its own plane's incidence angle, the other angle, then the angular
    # rate that damps its motion. A coefficient is swept against the first
    # input on its list that it depends on.
    _COEFFICIENT_SWEEP = {
        "cN": ("alpha", "beta", "pitch_rate"),
        "cY": ("beta", "alpha", "yaw_rate"),
        "cA": ("alpha", "beta"),
        "cm": ("alpha", "beta", "pitch_rate"),
        "cn": ("beta", "alpha", "yaw_rate"),
        "cl": ("roll_rate", "alpha", "beta"),
    }
    _AXIS_LABELS = {
        "alpha": "Angle of attack (°)",
        "beta": "Sideslip angle (°)",
        "pitch_rate": "Reduced pitch rate q·L/(2V)",
        "yaw_rate": "Reduced yaw rate r·L/(2V)",
        "roll_rate": "Reduced roll rate p·L/(2V)",
        "mach": "Mach number",
    }

    @staticmethod
    def _depends_on(coeff, index, mach):
        """Find the inputs a coefficient changes with.

        Read from ``depends_on`` when the coefficient has it. Otherwise each
        input is moved away from a reference state, and the coefficient depends
        on it if its value changes.

        Parameters
        ----------
        coeff : callable
            The coefficient, taking the surface's full list of inputs.
        index : dict
            Position of each input name in that list.
        mach : float
            Mach number of the reference state.

        Returns
        -------
        set of str
            Names of the inputs the coefficient depends on.
        """
        depends = getattr(coeff, "depends_on", None)
        if depends is not None:
            return set(depends)
        nudges = {
            "alpha": np.deg2rad(5),
            "beta": np.deg2rad(5),
            "mach": mach + 0.3,
            "reynolds": 1e6,
            "pitch_rate": 0.05,
            "yaw_rate": 0.05,
            "roll_rate": 0.05,
        }
        base = [0.0] * len(index)
        base[index["mach"]] = mach
        reference = coeff(*base)
        found = set()
        for name, value in nudges.items():
            if name not in index:
                continue
            args = list(base)
            args[index[name]] = value
            if not np.isclose(coeff(*args), reference):
                found.add(name)
        return found

    def coefficients(
        self,
        *,
        mach=(0.1, 0.3, 0.6, 0.9),
        angle_range_deg=15.0,
        rate_range=0.1,
        filename=None,
    ):
        """Plots the surface's body-frame aerodynamic coefficients.

        Each coefficient (``cN, cY, cA, cm, cn, cl``) that is not zero gets its
        own plot, against the input it depends on most: the angle of attack for
        the pitch-plane ones (``cN``, ``cA``, ``cm``), the sideslip angle for
        the yaw-plane ones (``cY``, ``cn``) and the roll rate for the roll
        moment ``cl``. A coefficient that does not use that input is plotted
        against the next one it does use (the other angle, then an angular
        rate), or against Mach if Mach is all it depends on. When a coefficient
        changes with Mach, one curve is drawn for each Mach number in ``mach``.
        Every input that is not plotted is held at zero.

        Parameters
        ----------
        mach : float or sequence of float, optional
            Mach number or numbers at which to draw each coefficient. Default
            ``(0.1, 0.3, 0.6, 0.9)``.
        angle_range_deg : float, optional
            The angles are plotted from ``-angle_range_deg`` to
            ``+angle_range_deg``, in degrees. Default 15.
        rate_range : float, optional
            The angular rates are plotted from ``-rate_range`` to
            ``+rate_range``. They are reduced rates, the rate times the
            reference length divided by twice the airspeed (for example
            ``p * L / (2 * V)`` for roll), so they have no units. Default 0.1.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        surface = self.aero_surface
        independent_vars = getattr(surface, "independent_vars", None)
        if independent_vars is None:
            return
        index = {name: i for i, name in enumerate(independent_vars)}
        if "mach" not in index:
            return
        machs = np.atleast_1d(np.asarray(mach, dtype=float))

        entries = []
        for name, candidates in self._COEFFICIENT_SWEEP.items():
            coeff = getattr(surface, name, None)
            if coeff is None or getattr(coeff, "is_zero", False):
                continue
            depends = self._depends_on(coeff, index, machs[0])
            var = next((v for v in candidates if v in depends and v in index), None)
            if var is None:
                var = "mach" if "mach" in depends else candidates[0]
            if var == "mach":
                x = np.linspace(0.0, machs.max(), 61)
                curve_machs = [None]
            else:
                limit = (
                    np.deg2rad(angle_range_deg)
                    if var in ("alpha", "beta")
                    else rate_range
                )
                x = np.linspace(-limit, limit, 61)
                curve_machs = machs if "mach" in depends else [None]
            curves = []
            for curve_mach in curve_machs:
                values = np.empty_like(x)
                for i, value in enumerate(x):
                    args = surface._reference_arguments()  # pylint: disable=protected-access
                    args[index["mach"]] = machs[0] if curve_mach is None else curve_mach
                    args[index[var]] = value
                    values[i] = coeff(*args)
                curves.append((curve_mach, values))
            if all(np.allclose(values, 0.0) for _, values in curves):
                continue
            entries.append((name, var, x, curves))

        if not entries:
            return

        fig, axes = plt.subplots(
            len(entries), 1, figsize=(7, 2.5 * len(entries)), squeeze=False
        )
        for ax, (name, var, x, curves) in zip(axes[:, 0], entries):
            x_plot = np.rad2deg(x) if var in ("alpha", "beta") else x
            for curve_mach, values in curves:
                label = None if len(curves) == 1 else f"Mach {curve_mach:g}"
                ax.plot(x_plot, values, label=label)
            if len(curves) == 1 and curves[0][0] is not None:
                ax.set_title(f"{name} at Mach {curves[0][0]:g}", fontsize=9)
            elif len(curves) > 1:
                ax.legend(fontsize=8)
            ax.set_xlabel(self._AXIS_LABELS[var])
            ax.set_ylabel(name)
            ax.grid(True)
        fig.suptitle(f"{surface.name} coefficients")
        plt.tight_layout()
        show_or_save_plot(filename)

    def all(self):
        """Plots the generic surface's aerodynamic coefficients."""
        self.coefficients()


class _LinearGenericSurfacePlots(_GenericSurfacePlots):
    """Plots for a linear generic surface; same plots as the generic base."""


class _BarrowmanSurfacePlots(_GenericSurfacePlots):
    """Plots shared by the geometry-defined (Barrowman) surfaces: adds the
    geometry drawing and the lift-coefficient surface plot."""

    def lift(self):
        """Plots the lift-curve slope (``clalpha``) of the aero surface as a
        function of Mach number.

        Returns
        -------
        None
        """
        self.aero_surface.clalpha()

    def all(self):
        """Plots the surface geometry, the lift coefficient and the
        aerodynamic coefficients."""
        self.draw()
        self.lift()
        self.coefficients()


class _NoseConePlots(_BarrowmanSurfacePlots):
    """Class that contains all nosecone plots. This class inherits from the
    _BarrowmanSurfacePlots class."""

    def draw(self, *, filename=None):
        """Draw the nosecone shape along with some important information,
        including the center line and the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Create the vectors X and Y with the points of the curve
        nosecone_x, nosecone_y = self.aero_surface.shape_vec

        # Figure creation and set up
        _, ax = plt.subplots()
        ax.set_xlim(-0.05, self.aero_surface.length * 1.02)  # Horizontal size
        ax.set_ylim(
            -self.aero_surface.base_radius * 1.05, self.aero_surface.base_radius * 1.05
        )  # Vertical size
        ax.set_aspect("equal")  # Makes the graduation be the same on both axis
        ax.set_facecolor("#EEEEEE")  # Background color
        ax.grid(True, linestyle="--", linewidth=0.5)

        cp_plot = (self.aero_surface.cpz, 0)
        # Plotting
        ax.plot(
            nosecone_x,
            nosecone_y,
            linestyle="-",
            color="#A60628",
        )  # Ogive's upper side
        ax.plot(
            nosecone_x,
            -nosecone_y,
            linestyle="-",
            color="#A60628",
        )  # Ogive's lower side
        ax.scatter(
            *cp_plot, label="Center Of Pressure", color="red", s=100, zorder=10
        )  # Center of pressure inner circle
        ax.scatter(
            *cp_plot, facecolors="none", edgecolors="red", s=500, zorder=10
        )  # Center of pressure outer circle
        # Center Line
        ax.plot(
            [0, nosecone_x[len(nosecone_x) - 1]],
            [0, 0],
            linestyle="--",
            color="#7A68A6",
            linewidth=1.5,
            label="Center Line",
        )
        # Vertical base line
        ax.plot(
            [
                nosecone_x[len(nosecone_x) - 1],
                nosecone_x[len(nosecone_x) - 1],
            ],
            [
                nosecone_y[len(nosecone_y) - 1],
                -nosecone_y[len(nosecone_y) - 1],
            ],
            linestyle="-",
            color="#A60628",
            linewidth=1.5,
        )

        # Labels and legend
        ax.set_xlabel("Length")
        ax.set_ylabel("Radius")
        ax.set_title(self.aero_surface.kind + " Nose Cone")
        ax.legend(bbox_to_anchor=(1, -0.2))
        show_or_save_plot(filename)


class _FinsPlots(_BarrowmanSurfacePlots):
    """Abstract class that contains all fin plots. This class inherits from the
    _BarrowmanSurfacePlots class."""

    def airfoil(self, *, filename=None):
        """Plots the airfoil information when the fin has an airfoil shape. If
        the fin does not have an airfoil shape, this method does nothing.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """

        if self.aero_surface.airfoil:
            print("Airfoil lift curve:")
            self.aero_surface.airfoil_cl.plot_1d(force_data=True, filename=filename)

    def roll(self, *, filename=None):
        """Plots the roll parameters of the fin set.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        print("Roll parameters:")
        self.aero_surface.roll_parameters[0](filename=filename)
        self.aero_surface.roll_parameters[1](filename=filename)

    def lift(self, *, filename=None):
        """Plots the lift coefficient of the aero surface as a function of Mach
        and the angle of attack. A 3D plot is expected. See the rocketpy.Function
        class for more information on how this plot is made. Also, this method
        plots the lift coefficient considering a single fin and the lift
        coefficient considering all fins.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        print("Lift coefficient derivative:")
        self.aero_surface.clalpha_single_fin(filename=filename)
        self.aero_surface.clalpha_multiple_fins(filename=filename)

    def all(self, *, filename=None):
        """Plots all available fin plots.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        self.draw(filename=filename)
        self.airfoil(filename=filename)
        self.roll(filename=filename)
        self.lift(filename=filename)
        self.coefficients(filename=filename)


class _FinPlots(_BarrowmanSurfacePlots):
    """Abstract class that contains all fin plots. This class inherits from the
    _BarrowmanSurfacePlots class."""

    def airfoil(self, *, filename=None):
        """Plots the airfoil information when the fin has an airfoil shape. If
        the fin does not have an airfoil shape, this method does nothing.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """

        if self.aero_surface.airfoil:
            print("Airfoil lift curve:")
            self.aero_surface.airfoil_cl.plot_1d(force_data=True, filename=filename)

    def roll(self, *, filename=None):
        """Plots the roll parameters of the fin set.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        print("Roll parameters:")
        self.aero_surface.roll_parameters[0](filename=filename)
        self.aero_surface.roll_parameters[1](filename=filename)

    def lift(self, *, filename=None):
        """Plots the lift coefficient of the aero surface as a function of Mach
        and the angle of attack. A 3D plot is expected. See the rocketpy.Function
        class for more information on how this plot is made. Also, this method
        plots the lift coefficient considering a single fin and the lift
        coefficient considering all fins.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        print("Lift coefficient derivative:")
        self.aero_surface.clalpha_single_fin(filename=filename)

    def all(self, *, filename=None):
        """Plots all available fin plots.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved.

        Returns
        -------
        None
        """
        self.draw(filename=filename)
        self.airfoil(filename=filename)
        self.roll(filename=filename)
        self.lift(filename=filename)
        self.coefficients(filename=filename)


class _TrapezoidalFinsPlots(_FinsPlots):
    """Class that contains all trapezoidal fin plots."""

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information, including
        the center line, the quarter line and the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Color cycle [#348ABD, #A60628, #7A68A6, #467821, #D55E00, #CC79A7,
        # #56B4E9, #009E73, #F0E442, #0072B2]
        # Fin
        leading_edge = plt.Line2D(
            (0, self.aero_surface.sweep_length),
            (0, self.aero_surface.span),
            color="#A60628",
        )
        tip = plt.Line2D(
            (
                self.aero_surface.sweep_length,
                self.aero_surface.sweep_length + self.aero_surface.tip_chord,
            ),
            (self.aero_surface.span, self.aero_surface.span),
            color="#A60628",
        )
        back_edge = plt.Line2D(
            (
                self.aero_surface.sweep_length + self.aero_surface.tip_chord,
                self.aero_surface.root_chord,
            ),
            (self.aero_surface.span, 0),
            color="#A60628",
        )
        root = plt.Line2D((self.aero_surface.root_chord, 0), (0, 0), color="#A60628")

        # Center and Quarter line
        center_line = plt.Line2D(
            (
                self.aero_surface.root_chord / 2,
                self.aero_surface.sweep_length + self.aero_surface.tip_chord / 2,
            ),
            (0, self.aero_surface.span),
            color="#7A68A6",
            alpha=0.35,
            linestyle="--",
            label="Center Line",
        )
        quarter_line = plt.Line2D(
            (
                self.aero_surface.root_chord / 4,
                self.aero_surface.sweep_length + self.aero_surface.tip_chord / 4,
            ),
            (0, self.aero_surface.span),
            color="#7A68A6",
            alpha=1,
            linestyle="--",
            label="Quarter Line",
        )

        # Center of pressure
        cp_point = [self.aero_surface.cpz, self.aero_surface.Yma]

        # Mean Aerodynamic Chord
        yma_start = (
            self.aero_surface.sweep_length
            * (self.aero_surface.root_chord + 2 * self.aero_surface.tip_chord)
            / (3 * (self.aero_surface.root_chord + self.aero_surface.tip_chord))
        )
        yma_end = (
            2 * self.aero_surface.root_chord**2
            + self.aero_surface.root_chord * self.aero_surface.sweep_length
            + 2 * self.aero_surface.root_chord * self.aero_surface.tip_chord
            + 2 * self.aero_surface.sweep_length * self.aero_surface.tip_chord
            + 2 * self.aero_surface.tip_chord**2
        ) / (3 * (self.aero_surface.root_chord + self.aero_surface.tip_chord))
        yma_line = plt.Line2D(
            (yma_start, yma_end),
            (self.aero_surface.Yma, self.aero_surface.Yma),
            color="#467821",
            linestyle="--",
            label="Mean Aerodynamic Chord",
        )

        # Plotting
        fig = plt.figure(figsize=(7, 4))
        with plt.style.context("bmh"):
            ax = fig.add_subplot(111)

        # Fin
        ax.add_line(leading_edge)
        ax.add_line(tip)
        ax.add_line(back_edge)
        ax.add_line(root)

        ax.add_line(center_line)
        ax.add_line(quarter_line)
        ax.add_line(yma_line)
        ax.scatter(*cp_point, label="Center of Pressure", color="red", s=100, zorder=10)
        ax.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        xlim = (
            self.aero_surface.root_chord
            if self.aero_surface.sweep_length + self.aero_surface.tip_chord
            < self.aero_surface.root_chord
            else self.aero_surface.sweep_length + self.aero_surface.tip_chord
        )
        ax.set_xlim(0, xlim * 1.1)
        ax.set_ylim(0, self.aero_surface.span * 1.1)
        ax.set_xlabel("Root chord (m)")
        ax.set_ylabel("Span (m)")
        ax.set_title("Trapezoidal Fin Cross Section")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.tight_layout()
        show_or_save_plot(filename)


class _TrapezoidalFinPlots(_FinPlots):
    """Class that contains all trapezoidal fin plots."""

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information, including
        the center line, the quarter line and the center of pressure position.

        Returns
        -------
        None
        """
        # Color cycle [#348ABD, #A60628, #7A68A6, #467821, #D55E00, #CC79A7,
        # #56B4E9, #009E73, #F0E442, #0072B2]
        # Fin
        leading_edge = plt.Line2D(
            (0, self.aero_surface.sweep_length),
            (0, self.aero_surface.span),
            color="#A60628",
        )
        tip = plt.Line2D(
            (
                self.aero_surface.sweep_length,
                self.aero_surface.sweep_length + self.aero_surface.tip_chord,
            ),
            (self.aero_surface.span, self.aero_surface.span),
            color="#A60628",
        )
        back_edge = plt.Line2D(
            (
                self.aero_surface.sweep_length + self.aero_surface.tip_chord,
                self.aero_surface.root_chord,
            ),
            (self.aero_surface.span, 0),
            color="#A60628",
        )
        root = plt.Line2D((self.aero_surface.root_chord, 0), (0, 0), color="#A60628")

        # Center and Quarter line
        center_line = plt.Line2D(
            (
                self.aero_surface.root_chord / 2,
                self.aero_surface.sweep_length + self.aero_surface.tip_chord / 2,
            ),
            (0, self.aero_surface.span),
            color="#7A68A6",
            alpha=0.35,
            linestyle="--",
            label="Center Line",
        )
        quarter_line = plt.Line2D(
            (
                self.aero_surface.root_chord / 4,
                self.aero_surface.sweep_length + self.aero_surface.tip_chord / 4,
            ),
            (0, self.aero_surface.span),
            color="#7A68A6",
            alpha=1,
            linestyle="--",
            label="Quarter Line",
        )

        # Center of pressure
        cp_point = [self.aero_surface.cpz, self.aero_surface.Yma]

        # Mean Aerodynamic Chord
        yma_start = (
            self.aero_surface.sweep_length
            * (self.aero_surface.root_chord + 2 * self.aero_surface.tip_chord)
            / (3 * (self.aero_surface.root_chord + self.aero_surface.tip_chord))
        )
        yma_end = (
            2 * self.aero_surface.root_chord**2
            + self.aero_surface.root_chord * self.aero_surface.sweep_length
            + 2 * self.aero_surface.root_chord * self.aero_surface.tip_chord
            + 2 * self.aero_surface.sweep_length * self.aero_surface.tip_chord
            + 2 * self.aero_surface.tip_chord**2
        ) / (3 * (self.aero_surface.root_chord + self.aero_surface.tip_chord))
        yma_line = plt.Line2D(
            (yma_start, yma_end),
            (self.aero_surface.Yma, self.aero_surface.Yma),
            color="#467821",
            linestyle="--",
            label="Mean Aerodynamic Chord",
        )

        # Plotting
        fig = plt.figure(figsize=(7, 4))
        with plt.style.context("bmh"):
            ax = fig.add_subplot(111)

        # Fin
        ax.add_line(leading_edge)
        ax.add_line(tip)
        ax.add_line(back_edge)
        ax.add_line(root)

        ax.add_line(center_line)
        ax.add_line(quarter_line)
        ax.add_line(yma_line)
        ax.scatter(*cp_point, label="Center of Pressure", color="red", s=100, zorder=10)
        ax.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        xlim = (
            self.aero_surface.root_chord
            if self.aero_surface.sweep_length + self.aero_surface.tip_chord
            < self.aero_surface.root_chord
            else self.aero_surface.sweep_length + self.aero_surface.tip_chord
        )
        ax.set_xlim(0, xlim * 1.1)
        ax.set_ylim(0, self.aero_surface.span * 1.1)
        ax.set_xlabel("Root chord (m)")
        ax.set_ylabel("Span (m)")
        ax.set_title("Trapezoidal Fin Cross Section")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.tight_layout()
        show_or_save_plot(filename)


class _EllipticalFinsPlots(_FinsPlots):
    """Class that contains all elliptical fin plots."""

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information.
        These being: the center line and the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Ellipse
        ellipse = Ellipse(
            (self.aero_surface.root_chord / 2, 0),
            self.aero_surface.root_chord,
            self.aero_surface.span * 2,
            fill=False,
            edgecolor="#A60628",
            linewidth=2,
        )

        # Mean Aerodynamic Chord # From Barrowman's theory
        yma_length = 8 * self.aero_surface.root_chord / (3 * np.pi)
        yma_start = (self.aero_surface.root_chord - yma_length) / 2
        yma_end = (
            self.aero_surface.root_chord
            - (self.aero_surface.root_chord - yma_length) / 2
        )
        yma_line = plt.Line2D(
            (yma_start, yma_end),
            (self.aero_surface.Yma, self.aero_surface.Yma),
            label="Mean Aerodynamic Chord",
            color="#467821",
        )

        # Center Line
        center_line = plt.Line2D(
            (self.aero_surface.root_chord / 2, self.aero_surface.root_chord / 2),
            (0, self.aero_surface.span),
            color="#7A68A6",
            alpha=0.35,
            linestyle="--",
            label="Center Line",
        )

        # Center of pressure
        cp_point = [self.aero_surface.cpz, self.aero_surface.Yma]

        # Plotting
        fig = plt.figure(figsize=(7, 4))
        with plt.style.context("bmh"):
            ax = fig.add_subplot(111)
        ax.add_patch(ellipse)
        ax.add_line(yma_line)
        ax.add_line(center_line)
        ax.scatter(*cp_point, label="Center of Pressure", color="red", s=100, zorder=10)
        ax.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        ax.set_xlim(0, self.aero_surface.root_chord)
        ax.set_ylim(0, self.aero_surface.span * 1.1)
        ax.set_xlabel("Root chord (m)")
        ax.set_ylabel("Span (m)")
        ax.set_title("Elliptical Fin Cross Section")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.tight_layout()
        show_or_save_plot(filename)


class _EllipticalFinPlots(_FinPlots):
    """Class that contains all elliptical fin plots."""

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information.
        These being: the center line and the center of pressure position.

        Returns
        -------
        None
        """
        # Ellipse
        ellipse = Ellipse(
            (self.aero_surface.root_chord / 2, 0),
            self.aero_surface.root_chord,
            self.aero_surface.span * 2,
            fill=False,
            edgecolor="#A60628",
            linewidth=2,
        )

        # Mean Aerodynamic Chord # From Barrowman's theory
        yma_length = 8 * self.aero_surface.root_chord / (3 * np.pi)
        yma_start = (self.aero_surface.root_chord - yma_length) / 2
        yma_end = (
            self.aero_surface.root_chord
            - (self.aero_surface.root_chord - yma_length) / 2
        )
        yma_line = plt.Line2D(
            (yma_start, yma_end),
            (self.aero_surface.Yma, self.aero_surface.Yma),
            label="Mean Aerodynamic Chord",
            color="#467821",
        )

        # Center Line
        center_line = plt.Line2D(
            (self.aero_surface.root_chord / 2, self.aero_surface.root_chord / 2),
            (0, self.aero_surface.span),
            color="#7A68A6",
            alpha=0.35,
            linestyle="--",
            label="Center Line",
        )

        # Center of pressure
        cp_point = [self.aero_surface.cpz, self.aero_surface.Yma]

        # Plotting
        fig = plt.figure(figsize=(7, 4))
        with plt.style.context("bmh"):
            ax = fig.add_subplot(111)
        ax.add_patch(ellipse)
        ax.add_line(yma_line)
        ax.add_line(center_line)
        ax.scatter(*cp_point, label="Center of Pressure", color="red", s=100, zorder=10)
        ax.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        ax.set_xlim(0, self.aero_surface.root_chord)
        ax.set_ylim(0, self.aero_surface.span * 1.1)
        ax.set_xlabel("Root chord (m)")
        ax.set_ylabel("Span (m)")
        ax.set_title("Elliptical Fin Cross Section")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.tight_layout()
        show_or_save_plot(filename)


class _FreeFormFinsPlots(_FinsPlots):
    """Class that contains all free form fin plots."""

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information.
        These being: the center line and the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Color cycle [#348ABD, #A60628, #7A68A6, #467821, #D55E00, #CC79A7,
        # #56B4E9, #009E73, #F0E442, #0072B2]

        # Center of pressure
        cp_point = [self.aero_surface.cpz, self.aero_surface.Yma]

        # Mean Aerodynamic Chord
        yma_line = plt.Line2D(
            (
                self.aero_surface.mac_lead,
                self.aero_surface.mac_lead + self.aero_surface.mac_length,
            ),
            (self.aero_surface.Yma, self.aero_surface.Yma),
            color="#467821",
            linestyle="--",
            label="Mean Aerodynamic Chord",
        )

        # Plotting
        fig = plt.figure(figsize=(7, 4))
        with plt.style.context("bmh"):
            ax = fig.add_subplot(111)

        # Fin
        ax.scatter(
            self.aero_surface.shape_vec[0],
            self.aero_surface.shape_vec[1],
            color="#A60628",
        )
        ax.plot(
            self.aero_surface.shape_vec[0],
            self.aero_surface.shape_vec[1],
            color="#A60628",
        )
        # line from the last point to the first point
        ax.plot(
            [self.aero_surface.shape_vec[0][-1], self.aero_surface.shape_vec[0][0]],
            [self.aero_surface.shape_vec[1][-1], self.aero_surface.shape_vec[1][0]],
            color="#A60628",
        )

        ax.add_line(yma_line)
        ax.scatter(*cp_point, label="Center of Pressure", color="red", s=100, zorder=10)
        ax.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        ax.set_xlabel("Root chord (m)")
        ax.set_ylabel("Span (m)")
        ax.set_title("Free Form Fin Cross Section")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.tight_layout()
        show_or_save_plot(filename)


class _FreeFormFinPlots(_FinPlots):
    """Class that contains all free form fin plots."""

    def draw(self, *, filename=None):
        """Draw the fin shape along with some important information.
        These being: the center line and the center of pressure position.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Color cycle [#348ABD, #A60628, #7A68A6, #467821, #D55E00, #CC79A7,
        # #56B4E9, #009E73, #F0E442, #0072B2]

        # Center of pressure
        cp_point = [self.aero_surface.cpz, self.aero_surface.Yma]

        # Mean Aerodynamic Chord
        yma_line = plt.Line2D(
            (
                self.aero_surface.mac_lead,
                self.aero_surface.mac_lead + self.aero_surface.mac_length,
            ),
            (self.aero_surface.Yma, self.aero_surface.Yma),
            color="#467821",
            linestyle="--",
            label="Mean Aerodynamic Chord",
        )

        # Plotting
        fig = plt.figure(figsize=(7, 4))
        with plt.style.context("bmh"):
            ax = fig.add_subplot(111)

        # Fin
        ax.scatter(
            self.aero_surface.shape_vec[0],
            self.aero_surface.shape_vec[1],
            color="#A60628",
        )
        ax.plot(
            self.aero_surface.shape_vec[0],
            self.aero_surface.shape_vec[1],
            color="#A60628",
        )
        # line from the last point to the first point
        ax.plot(
            [self.aero_surface.shape_vec[0][-1], self.aero_surface.shape_vec[0][0]],
            [self.aero_surface.shape_vec[1][-1], self.aero_surface.shape_vec[1][0]],
            color="#A60628",
        )

        ax.add_line(yma_line)
        ax.scatter(*cp_point, label="Center of Pressure", color="red", s=100, zorder=10)
        ax.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        ax.set_xlabel("Root chord (m)")
        ax.set_ylabel("Span (m)")
        ax.set_title("Free Form Fin Cross Section")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.tight_layout()
        show_or_save_plot(filename)


class _TailPlots(_BarrowmanSurfacePlots):
    """Class that contains all tail plots."""

    def draw(self, *, filename=None):
        # This will de done in the future
        pass


class _ControlFinsPlots(_GenericSurfacePlots):
    """Plots of a set of control fins: the fins' effectiveness, then the
    generic coefficients."""

    def draw(self, *, filename=None):
        """Draw the fin the set is copied from, or its first fin."""
        self.aero_surface.fins[0].plots.draw(filename=filename)

    def effectiveness(self, *, mach=(0.05, 0.95), alpha=0.0, beta=0.0, filename=None):
        """Plot how much the fins do per radian of each command of
        :meth:`rocketpy.ControlFins.deflections`, against Mach: the roll
        moment coefficient per radian of ``roll``, and the force coefficients
        toward ``+x`` and ``+y`` per radian of ``x`` and ``y``.

        Parameters
        ----------
        mach : tuple of float, optional
            ``(low, high)``: the Mach range plotted. Default ``(0.05, 0.95)``.
        alpha, beta : float, optional
            The angle of attack and the sideslip angle, in radians, the
            effectiveness is taken at. Default 0.
        filename : str | None, optional
            Where to save the figure. ``None`` (the default) shows it.
        """
        effectiveness = self.aero_surface.effectiveness(alpha=alpha, beta=beta)
        machs = np.linspace(mach[0], mach[1], 100)
        _, (roll_axis, force_axis) = plt.subplots(1, 2, figsize=(9, 4))
        roll_axis.plot(machs, [effectiveness["roll"](m) for m in machs])
        roll_axis.set_title("Roll moment per rad of the same deflection")
        roll_axis.set_xlabel("Mach Number")
        roll_axis.set_ylabel("Roll Moment Coefficient per rad")
        roll_axis.grid(True)
        for command, label in (("x", "toward +x"), ("y", "toward +y")):
            force_axis.plot(
                machs, [effectiveness[command](m) for m in machs], label=label
            )
        force_axis.set_title("Sideways force per rad of the x and y commands")
        force_axis.set_xlabel("Mach Number")
        force_axis.set_ylabel("Force Coefficient per rad")
        force_axis.grid(True)
        force_axis.legend()
        plt.suptitle(f"Effectiveness of {self.aero_surface.name}")
        plt.tight_layout()
        show_or_save_plot(filename)

    def hinge_moments(self, flight, *, filename=None):
        """Plot the torque the air put on each fin about its hinge line over a
        flight, in N m, one line per fin: the load the mechanism turning the
        fin had to hold. Positive pushes the fin toward a larger deflection.

        The fin model's center of pressure is fixed while the real one moves
        with speed and angle of attack, so with the hinge close to it trust
        the size of the torque, not its sign.

        Parameters
        ----------
        flight : Flight
            A flight of a rocket carrying this set of fins.
        filename : str | None, optional
            Where to save the figure. ``None`` (the default) shows it.
        """
        torques = self.aero_surface._hinge_moments_in(flight)
        _, axis = plt.subplots(figsize=(9, 4))
        for control, torque in torques.items():
            axis.plot(torque.x_array, torque.y_array, label=control)
        axis.set_title(f"Hinge moments of {self.aero_surface.name}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Hinge Moment (N m)")
        axis.grid(True)
        axis.legend()
        plt.tight_layout()
        show_or_save_plot(filename)

    def all(self):
        """Plots the first fin, the effectiveness and the coefficients."""
        self.draw()
        self.effectiveness()
        self.coefficients()


class _AirBrakesPlots(_GenericSurfacePlots):
    """Plots of air brakes: the drag coefficient curve, and the drag they made
    in a flight."""

    def _data_points(self):
        """The ``(level, mach, cd)`` rows the curve was given as, or ``None``
        for a curve given as a number or a function."""
        source = self.aero_surface.drag_coefficient.source
        if isinstance(source, np.ndarray) and source.ndim == 2 and source.shape[1] == 3:
            return source
        return None

    def drag_coefficient(self, *, levels=None, mach=None, filename=None):
        """Plot the drag coefficient against Mach, one line per deployment
        level.

        Parameters
        ----------
        levels : sequence of float, optional
            The deployment levels drawn. Default ``None``: the levels of the
            curve's data points, or 0.25, 0.5, 0.75 and 1 for a curve given as
            a number or a function.
        mach : tuple of float, optional
            ``(low, high)``: the Mach range drawn. Default ``None``: the range
            of the curve's data points, or 0.05 to 0.95.
        filename : str | None, optional
            Where to save the figure. ``None`` (the default) shows it.
        """
        points = self._data_points()
        if levels is None:
            levels = (
                [level for level in np.unique(points[:, 0]) if level > 0]
                if points is not None
                else (0.25, 0.5, 0.75, 1.0)
            )
        if mach is None:
            mach = (
                (points[:, 1].min(), points[:, 1].max())
                if points is not None
                else (0.05, 0.95)
            )
        machs = np.linspace(mach[0], mach[1], 100)
        curve = self.aero_surface.drag_coefficient
        _, axis = plt.subplots(figsize=(9, 4))
        for level in levels:
            axis.plot(machs, [curve(level, m) for m in machs], label=f"level {level:g}")
        axis.set_title(f"Drag coefficient of {self.aero_surface.name}")
        axis.set_xlabel("Mach Number")
        axis.set_ylabel("Drag Coefficient")
        axis.grid(True)
        axis.legend()
        plt.tight_layout()
        show_or_save_plot(filename)

    def drag_coefficient_curve(self):
        """The same as :meth:`drag_coefficient`, kept for older scripts."""
        self.drag_coefficient()

    def drag(self, flight, *, filename=None):
        """Plot the drag force the air brakes made over a flight, in N, next
        to the rocket's total drag.

        Parameters
        ----------
        flight : Flight
            A flight of a rocket carrying these air brakes.
        filename : str | None, optional
            Where to save the figure. ``None`` (the default) shows it.
        """
        drag = self.aero_surface._drag_in(flight)
        _, axis = plt.subplots(figsize=(9, 4))
        axis.plot(drag.x_array, drag.y_array, label=self.aero_surface.name)
        total = flight.aerodynamic_drag
        axis.plot(total.x_array, total.y_array, label="Whole rocket", alpha=0.6)
        axis.set_title(f"Drag of {self.aero_surface.name}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Drag (N)")
        axis.grid(True)
        axis.legend()
        plt.tight_layout()
        show_or_save_plot(filename)

    def all(self):
        """Plot the drag coefficient curve."""
        self.drag_coefficient()
