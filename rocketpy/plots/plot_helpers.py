import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SAVEFIG_DPI = 300


def show_or_save_plot(filename=None):
    """Shows or saves the current matplotlib plot. If a filename is given, the plot will be saved, otherwise it will be shown.

    Parameters
    ----------
    filename : str | None, optional
        The path the plot should be saved to, by default None. Supported file endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp.
    """
    if filename == None:
        plt.show()
    else:
        # Warn if file ending is not supported
        file_ending = Path(filename).suffix
        if file_ending not in get_matplotlib_supported_file_endings():
            warnings.warn(
                f"Warning: Unsupported file ending '{file_ending}'!", UserWarning
            )


        # Before export, ensure the folder the file should go into exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Save the plot
        plt.savefig(filename, dpi=SAVEFIG_DPI)
        plt.close()


def show_or_save_fig(fig: Figure, filename=None):
    """Shows or saves the given matplotlib Figure. If a filename is given, the figure will be saved, otherwise it will be shown.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be saved or shown.
    filename : str | None, optional
        The path the figure should be saved to, by default None. Supported file endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp.
    """
    if filename == None:
        fig.show()
    else:
        # Warn if file ending is not supported
        file_ending = Path(filename).suffix
        if file_ending not in get_matplotlib_supported_file_endings():
            warnings.warn(
                f"Warning: Unsupported file ending '{file_ending}'!", UserWarning
            )

        # Before export, ensure the folder the file should go into exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Save the figure
        fig.savefig(filename, dpi=SAVEFIG_DPI)


def get_matplotlib_supported_file_endings():
    # Get matplotlib's supported file ending and return them (without descriptions, hence only keys)
    filetypes = plt.gcf().canvas.get_supported_filetypes().keys()

    # Ensure the dot is included in the filetype endings
    filetypes = ["." + filetype for filetype in filetypes]

    return filetypes
