from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..tools import get_matplotlib_supported_file_endings

SAVEFIG_DPI = 300


def show_or_save_plot(filename=None):
    """Shows or saves the current matplotlib plot. If a filename is given, the
    plot will be saved, otherwise it will be shown.

    Parameters
    ----------
    filename : str | None, optional
        The path the plot should be saved to, by default None. Supported file
        endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz,
        tif, tiff and webp (these are the formats supported by matplotlib).
    """
    if filename is None:
        plt.show()
    else:
        file_ending = Path(filename).suffix
        supported_endings = get_matplotlib_supported_file_endings()
        if file_ending not in supported_endings:
            raise ValueError(
                f"Unsupported file ending '{file_ending}'."
                f"Supported file endings are: {supported_endings}."
            )

        # Before export, ensure the folder the file should go into exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(filename, dpi=SAVEFIG_DPI)
        plt.close()


def show_or_save_fig(fig: Figure, filename=None):
    """Shows or saves the given matplotlib Figure. If a filename is given, the
    figure will be saved, otherwise it will be shown.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be saved or shown.
    filename : str | None, optional
        The path the figure should be saved to, by default None. Supported file
        endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz,
        tif, tiff and webp (these are the formats supported by matplotlib).
    """
    if filename is None:
        fig.show()
    else:
        file_ending = Path(filename).suffix
        supported_endings = get_matplotlib_supported_file_endings()
        if file_ending not in supported_endings:
            raise ValueError(
                f"Unsupported file ending '{file_ending}'."
                f"Supported file endings are: {supported_endings}."
            )

        # Before export, ensure the folder the file should go into exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=SAVEFIG_DPI)


def show_or_save_animation(animation, filename=None, fps=30):
    """Shows or saves the given matplotlib animation. If a filename is given,
    the animation will be saved. Otherwise, it will be shown.

    Parameters
    ----------
    animation : matplotlib.animation.FuncAnimation
        The animation object to be saved or shown.
    filename : str | None, optional
        The path the animation should be saved to, by default None. Supported
        file ending is: gif.
    fps : int, optional
        Frames per second when saving the animation. Default is 30.
    """
    if filename is None:
        plt.show()
    else:
        file_ending = Path(filename).suffix
        supported_endings = [".gif"]

        if file_ending not in supported_endings:
            raise ValueError(
                f"Unsupported file ending '{file_ending}'. "
                f"Supported file endings are: {supported_endings}."
            )

        # Before export, ensure the folder the file should go into exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        animation.save(filename, fps=fps, writer="pillow")

        plt.close()
