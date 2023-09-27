import warnings
import matplotlib.pyplot as plt


class Compare:
    """A class to compare the results of multiple objects of the same type.

    Parameters
    ----------
    object_list : list
        A list of objects of the same type to be compared.

    Attributes
    ----------
    object_list : list
        A list of objects to be compared.

    """

    def __init__(self, object_list):
        """Initializes the Compare class.

        Parameters
        ----------
        object_list : list
            A list of objects objects to be compared.

        Returns
        -------
        None
        """

        # check if all items in object_list are the same type
        if not all(isinstance(obj, type(object_list[0])) for obj in object_list[1:]):
            warnings.warn(
                "Trying to compare objects of different classes. Make sure are "
                + "items in the list are of the same type."
            )

        self.object_list = object_list

        return None

    def create_comparison_figure(
        self,
        y_attributes,
        n_rows,
        n_cols,
        figsize,
        legend,
        title,
        x_labels,
        y_labels,
        x_lim,
        y_lim,
        x_attributes=None,
    ):
        """Creates a figure to compare the results of multiple objects of the
        same type.

        Parameters
        ----------
        y_attributes : list
            The attributes of the class to be plotted as the vertical
            coordinates of the data points. The attributes must be a list of
            strings. Each string must be a valid attribute of the object's
            class, i.e., should point to a attribute of the object's class that
            is a Function object or a numpy array. For example ["x", "y", "z"].
        n_rows : int
            The number of rows of the figure.
        n_cols : int
            The number of columns of the figure.
        figsize : tuple
            The standard matplotlib size of the figure, where the tuple means
            (width, height). For example (7, 10).
        legend : bool
            Whether to show the legend or not.
        title : str
            The title of the figure.
        x_labels : list
            A list of strings of the x labels of each subplot.
            For example ["Time (s)", "Time (s)", "Time (s)"].
        y_labels : list
            A list of strings of the y labels of each subplot.
            For example ["x (m)", "y (m)", "z (m)"].
        x_lim : tuple
            A tuple where the first item represents the x axis lower limit and
            second item, the x axis upper limit. If set to None, will be
            calculated automatically by matplotlib.
        y_lim : tuple
            A tuple where the first item represents the y axis lower limit and
            second item, the y axis upper limit. If set to None, will be
            calculated automatically by matplotlib.
        x_attributes : list
            The attributes of the class to be plotted as the horizontal
            coordinates of the data points. The attributes must be a list of
            strings. Each string must be a valid attribute of the object's
            class, i.e., should point to a attribute of the object's class that
            is a Function object or a numpy array.
            For example ["time", "time", "time"].

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object.
        """

        n_plots = len(y_attributes)

        # Create the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, y=1.02, x=0.5)

        # Create the subplots
        ax = []
        for i in range(n_plots):
            ax.append(plt.subplot(n_rows, n_cols, i + 1))

        # Adding the plots to each subplot
        if x_attributes:
            for object in self.object_list:
                for i in range(n_plots):
                    try:
                        ax[i].plot(
                            object.__getattribute__(x_attributes[i])[:, 1],
                            object.__getattribute__(y_attributes[i])[:, 1],
                            label=object.name,
                        )
                    except IndexError:
                        ax[i].plot(
                            object.__getattribute__(x_attributes[i]),
                            object.__getattribute__(y_attributes[i])[:, 1],
                            label=object.name,
                        )
                    except AttributeError:
                        raise AttributeError(
                            f"Invalid attribute {y_attributes[i]} or {x_attributes[i]}."
                        )
        else:
            # Adding the plots to each subplot
            for object in self.object_list:
                for i in range(n_plots):
                    try:
                        ax[i].plot(
                            object.__getattribute__(y_attributes[i])[:, 0],
                            object.__getattribute__(y_attributes[i])[:, 1],
                            label=object.name,
                        )
                    except AttributeError:
                        raise AttributeError(f"Invalid attribute {y_attributes[i]}.")

        for i, subplot in enumerate(ax):
            # Set the labels for the x and y axis
            subplot.set_xlabel(x_labels[i])
            subplot.set_ylabel(y_labels[i])

            # Set the limits for the x axis
            if x_lim:
                subplot.set_xlim(*x_lim)
            if y_lim:
                subplot.set_ylim(*y_lim)
            subplot.grid(True)  # Add a grid to the plot

        # Find the two closest integers to the square root of the number of object_list
        # to be used as the number of columns and rows of the legend
        n_cols_legend = int(round(len(self.object_list) ** 0.5))
        n_rows_legend = int(round(len(self.object_list) / n_cols_legend))

        # Set the legend
        if legend:  # Add a global legend to the figure
            fig.legend(
                *ax[0].get_legend_handles_labels(),
                loc="lower center",
                ncol=n_cols_legend,
                numpoints=1,
                frameon=True,
                bbox_to_anchor=(0.5, 1.05),
            )

        fig.tight_layout()

        return fig, ax
