import matplotlib.pyplot as plt
import numpy as np


class _SensitivityModelPlots:
    def __init__(self, model):
        self.model = model

    def __create_bar_plot(self, x, y, title, bar_colors):
        fig, axs = plt.subplots()
        fig.supxlabel("")
        fig.supylabel("Sensitivity (%)")
        axs.bar(x, y, color=bar_colors)
        axs.set_title(title)
        axs.tick_params(labelrotation=90)

    def __calculate_sensitivities(self, target_variable):
        x = self.model.parameters_names + ["LAE"]
        y = np.array(
            [
                100
                * self.model.target_variables_info[target_variable]["sensitivity"][
                    param
                ]
                for param in self.model.parameters_names
            ]
            + [100 * self.model.target_variables_info[target_variable]["LAE"]]
        )
        return x, y

    def bar_plot(self, target_variable="all"):
        """Creates a bar plot showing the sensitivity of the target_variable due
        to parameters

        Parameters
        ----------
        target_variable : str, optional
            Name of the target variable used to show sensitivity. It can also
            be "all", in which case a plot is created for each target variable
            in which the model was fitted. The default is "all".

        Returns
        -------
        None
        """
        self.model._raise_error_if_not_fitted()

        if (
            target_variable != "all"
            and target_variable not in self.model.target_variables_names
        ):
            raise ValueError(
                f"Target variable {target_variable} was not listed in initialization!"
            )

        bar_colors = self.model.n_parameters * ["blue"] + ["red"]

        if target_variable == "all":
            for current_target_variable in self.model.target_variables_names:
                x, y = self.__calculate_sensitivities(current_target_variable)
                # sort by sensitivity
                y, x, bar_colors = zip(*sorted(zip(y, x, bar_colors), reverse=True))
                self.__create_bar_plot(x, y, current_target_variable, bar_colors)
        else:
            x, y = self.__calculate_sensitivities(target_variable)
            # sort by sensitivity
            y, x, bar_colors = zip(*sorted(zip(y, x, bar_colors), reverse=True))
            self.__create_bar_plot(x, y, target_variable, bar_colors)

        plt.show()

    def all(self):
        self.bar_plot("all")
