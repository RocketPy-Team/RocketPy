import urllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy
from PIL import UnidentifiedImageError

from ..tools import (
    convert_local_extent_to_wgs84,
    convert_mercator_extent_to_local,
    generate_monte_carlo_ellipses,
    import_optional_dependency,
)
from .plot_helpers import show_or_save_plot


class _MonteCarloPlots:
    """Class to plot the Monte Carlo analysis results."""

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo

    def _get_environment_coordinates(self):
        """Get origin coordinates and earth radius from the environment.

        Returns
        -------
        tuple[float, float, float]
            A tuple containing (origin_lat, origin_lon, earth_radius).

        Raises
        ------
        ValueError
            If MonteCarlo object doesn't have an environment attribute, or if
            environment doesn't have latitude and longitude attributes.
        """
        if not hasattr(self.monte_carlo, "environment"):
            raise ValueError(
                "MonteCarlo object must have an 'environment' attribute "
                "for automatically fetching the background map."
            )
        env = self.monte_carlo.environment
        if not hasattr(env, "latitude") or not hasattr(env, "longitude"):
            raise ValueError(
                "Environment must have 'latitude' and 'longitude' attributes "
                "for automatically fetching the background map."
            )

        # Handle both StochasticEnvironment (which stores as lists) and
        # Environment (which stores as scalars)
        origin_lat = env.latitude
        origin_lon = env.longitude
        if isinstance(origin_lat, (list, tuple)):
            origin_lat = origin_lat[0]
        if isinstance(origin_lon, (list, tuple)):
            origin_lon = origin_lon[0]

        # We enforce the standard WGS84 Earth radius (approx. 6,378,137 m) for
        # visualization purposes. Background map providers (e.g., via Contextily)
        # typically use Web Mercator (EPSG:3857), which assumes this standard radius.
        # Using a custom local radius here—even if used in the physics simulation—would
        # cause projection mismatches, resulting in the map being offset or scaled
        # incorrectly relative to the data points.
        earth_radius = 6378137.0

        return origin_lat, origin_lon, earth_radius

    def _resolve_map_provider(self, background, contextily):
        """Resolve the map provider string to a contextily provider object.

        Parameters
        ----------
        background : str
            Type of background map. Options: "satellite", "street", "terrain",
            or any contextily provider name (e.g., "CartoDB.Positron").
        contextily : module
            The contextily module.

        Returns
        -------
        object
            The resolved contextily provider object.

        Raises
        ------
        ValueError
            If the map provider string cannot be resolved in contextily.providers.
            This may occur if the provider name is invalid. Check the provider name
            or use one of the built-in options: 'satellite', 'street', or 'terrain'.
        """
        if background == "satellite":
            map_provider = "Esri.WorldImagery"
        elif background == "street":
            map_provider = "OpenStreetMap.Mapnik"
        elif background == "terrain":
            map_provider = "Esri.WorldTopoMap"
        else:
            map_provider = background

        # Attempt to resolve provider string (e.g., "Esri.WorldImagery") to object
        source_provider = map_provider
        if isinstance(map_provider, str):
            try:
                p = contextily.providers
                for key in map_provider.split("."):
                    p = p[key]
                source_provider = p
            except (KeyError, AttributeError) as e:
                raise ValueError(
                    f"Invalid map provider '{background}'. "
                    f"The provider '{map_provider}' could not be found in contextily.providers. "
                    f"Please check the provider name or use one of the built-in options: "
                    f"'satellite', 'street', or 'terrain'."
                ) from e

        return source_provider

    def _get_background_map(self, background, xlim, ylim):
        """
        Helper method to get the background map for the Monte Carlo analysis.

        Parameters
        ----------
        background : str, optional
            Type of background map to automatically download and display.
            Options: "satellite" (uses Esri.WorldImagery)
                     "street" (uses OpenStreetMap.Mapnik)
                     "terrain" (uses Esri.WorldTopoMap)
                     or any contextily provider name (e.g., "CartoDB.Positron").
        xlim : tuple
            Limits of the x-axis. Default is (-3000, 3000). Values in meters.
        ylim : tuple
            Limits of the y-axis. Default is (-3000, 3000). Values in meters.

        Returns
        -------
        bg : ndarray
            Image as a 3D array of RGB values
        extent : tuple
            Bounding box [minX, maxX, minY, maxY] of the returned image

        Raises
        ------
        ImportError
            If the contextily library is not installed.
        RuntimeError
            If unable to fetch the background map from the provider.
        """
        if background is None:
            return None, None

        contextily = import_optional_dependency("contextily")

        origin_lat, origin_lon, earth_radius = self._get_environment_coordinates()
        source_provider = self._resolve_map_provider(background, contextily)
        local_extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        west, south, east, north = convert_local_extent_to_wgs84(
            local_extent, origin_lat, origin_lon, earth_radius
        )

        try:
            bg, mercator_extent = contextily.bounds2img(
                west, south, east, north, source=source_provider, ll=True
            )
        except ValueError as e:
            raise ValueError(
                f"Input coordinates or zoom level are invalid.\n"
                f"  - Provided bounds: W={west:.6f}, S={south:.6f}, E={east:.6f}, N={north:.6f}\n"
                f"  - Provider: {source_provider}\n"
                f"  - Tip: Ensure West < East and South < North.\n"
                f"  - Tip: Ensure coordinates are within Web Mercator limits (approx +/-85 lat).\n"
                f"Original error: {str(e)}"
            ) from e

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            raise ConnectionError(
                f"Network error while fetching tiles from provider '{background}'.\n"
                f"  - Provider: {source_provider}\n"
                f"  - Status: Check your internet connection.\n"
                f"  - The tile server might be down or blocking requests (rate limited).\n"
                f"  - Original error: {str(e)}"
            ) from e

        except UnidentifiedImageError as e:
            raise RuntimeError(
                f"The provider '{background}' returned invalid image data.\n"
                f"  - Provider: {source_provider}\n"
                f"  - Cause: This often happens when the API requires a key/token that is missing or invalid.\n"
                f"  - Result: The server likely returned an HTML error page instead of a PNG/JPG."
                f"  - Original error: {str(e)}"
            ) from e

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while generating the map.\n"
                f"  - Bounds: {west:.6f}, {south:.6f}, {east:.6f}, {north:.6f}\n"
                f"  - Provider: {source_provider}\n"
                f"  - Error Detail: {str(e)}"
            ) from e
        local_extent = convert_mercator_extent_to_local(
            mercator_extent, origin_lat, origin_lon, earth_radius
        )

        return bg, local_extent

    # pylint: disable=too-many-statements
    def ellipses(
        self,
        image=None,
        background=None,
        actual_landing_point=None,
        perimeter_size=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
        save=False,
    ):
        """
        Plot the error ellipses for the apogee and impact points of the rocket.

        Parameters
        ----------
        image : str, optional
            Path to the background image, usually a map of the launch site.
            If both `image` and `background` are provided, `image` takes precedence.
        background : str, optional
            Type of background map to automatically download and display.
            Options: "satellite" (uses Esri.WorldImagery)
                     "street" (uses OpenStreetMap.Mapnik)
                     "terrain" (uses Esri.WorldTopoMap)
                     or any contextily provider name (e.g., "CartoDB.Positron").
            If both `image` and `background` are provided, `image` takes precedence.
        actual_landing_point : tuple, optional
            Actual landing point of the rocket in (x, y) meters.
        perimeter_size : int, optional
            Size of the perimeter to be plotted. Default is 3000.
        xlim : tuple, optional
            Limits of the x-axis. Default is (-3000, 3000). Values in meters.
        ylim : tuple, optional
            Limits of the y-axis. Default is (-3000, 3000). Values in meters.
        save : bool, optional
            Whether to save the plot as a file. Default is False. If True, the
            plot is saved and not displayed. If False, the plot is displayed.

        Returns
        -------
        None
        """

        # Import background map
        if image is not None:
            imageio = import_optional_dependency("imageio")
            try:
                img = imageio.imread(image)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "The image file was not found. Please check the path."
                ) from e

        bg, local_extent = None, None
        if image is None and background is not None:
            bg, local_extent = self._get_background_map(background, xlim, ylim)

        try:
            apogee_x = np.array(self.monte_carlo.results["apogee_x"])
            apogee_y = np.array(self.monte_carlo.results["apogee_y"])
        except KeyError:
            print("No apogee data found. Skipping apogee ellipses.")
            apogee_x = np.array([])
            apogee_y = np.array([])
        try:
            impact_x = np.array(self.monte_carlo.results["x_impact"])
            impact_y = np.array(self.monte_carlo.results["y_impact"])
        except KeyError:
            print("No impact data found. Skipping impact ellipses.")
            impact_x = np.array([])
            impact_y = np.array([])

        if len(apogee_x) == 0 and len(impact_x) == 0:
            raise ValueError("No apogee or impact data found. Cannot plot ellipses.")

        impact_ellipses, apogee_ellipses = generate_monte_carlo_ellipses(
            apogee_x,
            apogee_y,
            impact_x,
            impact_y,
        )

        # Create plot figure
        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.subplot(111)

        for ell in impact_ellipses:
            ax.add_artist(ell)
        for ell in apogee_ellipses:
            ax.add_artist(ell)

        # Draw points
        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        plt.scatter(
            apogee_x, apogee_y, s=5, marker="^", color="green", label="Simulated Apogee"
        )
        plt.scatter(
            impact_x,
            impact_y,
            s=5,
            marker="v",
            color="blue",
            label="Simulated Landing Point",
        )

        if actual_landing_point:
            plt.scatter(
                actual_landing_point[0],
                actual_landing_point[1],
                s=20,
                marker="X",
                color="red",
                label="Measured Landing Point",
            )

        plt.legend()

        ax.set_title("1$\\sigma$, 2$\\sigma$ and 3$\\sigma$ Monte Carlo Ellipses")
        north_south_offset = offset_copy(
            ax.transAxes, fig=plt.gcf(), x=-72, y=0, units="points"
        )
        east_west_offset = offset_copy(
            ax.transAxes, fig=plt.gcf(), x=0, y=-30, units="points"
        )
        ax.text(0, 0, "West", va="bottom", ha="center", transform=east_west_offset)
        ax.text(1, 0, "East", va="bottom", ha="center", transform=east_west_offset)
        ax.text(0, 0, "South", va="bottom", ha="left", transform=north_south_offset)
        ax.text(0, 1, "North", va="top", ha="left", transform=north_south_offset)
        ax.set_ylabel("Y (m)")
        ax.set_xlabel("X (m)")
        # Add background image to plot
        # TODO: In the future, integrate with other libraries to plot the map (e.g. cartopy, ee, etc.)
        # You can translate the basemap by changing dx and dy (in meters)
        dx = 0
        dy = 0
        if image is not None:
            plt.imshow(
                img,
                zorder=0,
                extent=[
                    -perimeter_size - dx,
                    perimeter_size - dx,
                    -perimeter_size - dy,
                    perimeter_size - dy,
                ],
            )

        elif bg is not None and local_extent is not None:
            plt.imshow(bg, extent=local_extent, zorder=0, interpolation="bilinear")

        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        # Set equal aspect ratio to ensure consistent display regardless of background
        ax.set_aspect("equal")

        if save:
            plt.savefig(
                f"{self.monte_carlo.filename}.png", bbox_inches="tight", pad_inches=0
            )
        else:
            plt.show()

    def all(self, keys=None, *, filename=None):
        """
        Plot the histograms of the Monte Carlo simulation results.

        Parameters
        ----------
        keys : str, list or tuple, optional
            The keys of the results to be plotted. If None, all results will be
            plotted. Default is None.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. If a filename is provided,
            each histogram will be saved with the key name appended to the
            filename (e.g., "plots/histogram_apogee.png" for key "apogee" with
            filename "plots/histogram.png"). Supported file endings are: eps,
            jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and
            webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        if keys is None:
            keys = self.monte_carlo.results.keys()
        elif isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, (list, tuple)):
            keys = list(set(keys).intersection(self.monte_carlo.results.keys()))
            if len(keys) == 0:
                raise ValueError(
                    "The specified 'keys' are not available in the results."
                )
        else:
            raise ValueError("The 'keys' argument must be a string, list, or tuple.")
        for key in keys:
            # Create figure with GridSpec
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

            # Create subplots using gridspec
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            # Plot boxplot
            # TODO: changes vert to orientation="horizontal" when support for Py3.9 ends
            ax1.boxplot(self.monte_carlo.results[key], vert=False)
            ax1.set_title(f"Box Plot of {key}")
            ax1.set_yticks([])

            # Plot histogram
            ax2.hist(self.monte_carlo.results[key])
            ax2.set_title(f"Histogram of {key}")
            ax2.set_ylabel("Number of Occurrences")
            ax1.set_xticks([])

            plt.tight_layout()

            # Generate the filename for this specific key if saving
            if filename is not None:
                file_path = Path(filename)
                key_filename = str(
                    file_path.parent / f"{file_path.stem}_{key}{file_path.suffix}"
                )
                show_or_save_plot(key_filename)
            else:
                show_or_save_plot(None)

    def plot_comparison(self, other_monte_carlo):
        """
        Plot the histograms of the Monte Carlo simulation results.

        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.

        Returns
        -------
        None
        """
        original_parameters_set = set(self.monte_carlo.processed_results.keys())
        other_parameters_set = set(other_monte_carlo.processed_results.keys())
        intersection_set = original_parameters_set.intersection(other_parameters_set)

        fill_colors = ["red", "blue"]
        for key in intersection_set:
            # Create figure with GridSpec
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

            # Create subplots using gridspec
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            # Plot boxplot
            bp = ax1.boxplot(
                [other_monte_carlo.results[key], self.monte_carlo.results[key]],
                vert=False,
                tick_labels=["Other", "Original"],
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], fill_colors):
                patch.set_facecolor(color)
            ax1.set_title(f"Box Plot of {key}")

            # Plot histogram
            ax2.hist(
                self.monte_carlo.results[key],
                alpha=0.5,
                color="blue",
                label="Original",
                density=True,
            )
            ax2.hist(
                other_monte_carlo.results[key],
                alpha=0.5,
                color="red",
                label="Other",
                density=True,
            )
            ax2.set_title(f"Histogram of {key}")
            ax2.set_ylabel("Density")

            plt.tight_layout()
            plt.legend()
            plt.show()

    # pylint: disable=too-many-statements
    def ellipses_comparison(
        self,
        other_monte_carlo,
        image=None,
        background=None,
        perimeter_size=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
        save=False,
    ):
        """
        Plot the error ellipses for the apogee and impact points of the rocket.

        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        image : str, optional
            Path to the background image, usually a map of the launch site.
        background : str, optional
            Type of background map to automatically download and display.
            Options: "satellite" (uses Esri.WorldImagery)
                     "street" (uses OpenStreetMap.Mapnik)
                     "terrain" (uses Esri.WorldTopoMap)
                     or any contextily provider name (e.g., "CartoDB.Positron").
            If both `image` and `background` are provided, `image` takes precedence.
        perimeter_size : int, optional
            Size of the perimeter to be plotted. Default is 3000.
        xlim : tuple, optional
            Limits of the x-axis. Default is (-3000, 3000). Values in meters.
        ylim : tuple, optional
            Limits of the y-axis. Default is (-3000, 3000). Values in meters.
        save : bool, optional
            Whether to save the plot as a file. Default is False. If True, the
            plot is saved and not displayed. If False, the plot is displayed.

        Returns
        -------
        None
        """

        # Import background map
        if image is not None:
            imageio = import_optional_dependency("imageio")
            try:
                img = imageio.imread(image)
            except FileNotFoundError as e:  # pragma no cover
                raise FileNotFoundError(
                    "The image file was not found. Please check the path."
                ) from e

        bg, local_extent = None, None
        if image is None and background is not None:
            bg, local_extent = self._get_background_map(background, xlim, ylim)

        try:
            original_apogee_x = np.array(self.monte_carlo.results["apogee_x"])
            original_apogee_y = np.array(self.monte_carlo.results["apogee_y"])
            other_apogee_x = np.array(other_monte_carlo.results["apogee_x"])
            other_apogee_y = np.array(other_monte_carlo.results["apogee_y"])
        except KeyError:
            print("No apogee data found. Skipping apogee ellipses.")
            original_apogee_x = np.array([])
            original_apogee_y = np.array([])
            other_apogee_x = np.array([])
            other_apogee_y = np.array([])
        try:
            original_impact_x = np.array(self.monte_carlo.results["x_impact"])
            original_impact_y = np.array(self.monte_carlo.results["y_impact"])
            other_impact_x = np.array(other_monte_carlo.results["x_impact"])
            other_impact_y = np.array(other_monte_carlo.results["y_impact"])
        except KeyError:
            print("No impact data found. Skipping impact ellipses.")
            original_impact_x = np.array([])
            original_impact_y = np.array([])
            other_impact_x = np.array([])
            other_impact_y = np.array([])

        if (
            len(original_apogee_x) == 0 and len(original_impact_x) == 0
        ):  # pragma no cover
            raise ValueError("No apogee or impact data found. Cannot plot ellipses.")

        original_impact_ellipses, original_apogee_ellipses = (
            generate_monte_carlo_ellipses(
                original_apogee_x,
                original_apogee_y,
                original_impact_x,
                original_impact_y,
                apogee_rgb=(0.0117647, 0.1490196, 0.9882352),
                impact_rgb=(0.9882352, 0.0117647, 0.6392156),
            )
        )

        other_impact_ellipses, other_apogee_ellipses = generate_monte_carlo_ellipses(
            other_apogee_x,
            other_apogee_y,
            other_impact_x,
            other_impact_y,
            apogee_rgb=(0.9882352, 0.8509803, 0.0117647),
            impact_rgb=(0.0117647, 0.9882352, 0.3607843),
        )

        # Create plot figure
        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.subplot(111)

        # Draw ellipses and points for original monte carlo
        for ell in original_impact_ellipses:
            ax.add_artist(ell)
        for ell in original_apogee_ellipses:
            ax.add_artist(ell)

        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        plt.scatter(
            original_apogee_x,
            original_apogee_y,
            s=5,
            marker="^",
            color="#0326FC",
            label="Original Apogee",
        )
        plt.scatter(
            original_impact_x,
            original_impact_y,
            s=5,
            marker="v",
            color="#FC03A3",
            label="Original Landing Point",
        )

        # Draw ellipses and points for other monte carlo
        for ell in other_impact_ellipses:
            ax.add_artist(ell)
        for ell in other_apogee_ellipses:
            ax.add_artist(ell)

        plt.scatter(
            other_apogee_x,
            other_apogee_y,
            s=5,
            marker="^",
            color="#FCD903",
            label="Other Apogee",
        )
        plt.scatter(
            other_impact_x,
            other_impact_y,
            s=5,
            marker="v",
            color="#03FC5C",
            label="Other Landing Point",
        )

        plt.legend()
        ax.set_title("1$\\sigma$, 2$\\sigma$ and 3$\\sigma$ Monte Carlo Ellipses")
        ax.set_ylabel("North (m)")
        ax.set_xlabel("East (m)")

        # Add background image to plot
        # TODO: In the future, integrate with other libraries to plot the map (e.g. cartopy, ee, etc.)
        # You can translate the basemap by changing dx and dy (in meters)
        dx = 0
        dy = 0
        if image is not None:
            plt.imshow(
                img,
                zorder=0,
                extent=[
                    -perimeter_size - dx,
                    perimeter_size - dx,
                    -perimeter_size - dy,
                    perimeter_size - dy,
                ],
            )
        elif bg is not None and local_extent is not None:
            plt.imshow(bg, extent=local_extent, zorder=0, interpolation="bilinear")

        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        ax.set_aspect("equal")

        if save:
            plt.savefig(
                f"{self.monte_carlo.filename}.png", bbox_inches="tight", pad_inches=0
            )
        else:
            plt.show()
