import matplotlib.pyplot as plt
import numpy as np


class _SensitivityModelPlots:

    def __init__(self, model):
        self.model = model

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

        if (target_variable not in self.model.target_variables_names) and (
            target_variable != "all"
        ):
            raise ValueError(
                f"Target variable {target_variable} was not listed in \
                  initialization!"
            )

        # Parameters bars are blue colored
        # LAE bar is red colored
        bar_colors = self.model.n_parameters * ["blue"]
        bar_colors.append("red")

        if target_variable == "all":
            for i in range(self.model.n_target_variables):
                fig, axs = plt.subplots()
                fig.supxlabel("")
                fig.supylabel("Sensitivity (%)")
                x = self.model.parameters_names
                x.append("LAE")
                current_target_variable = self.model.target_variables_names[i]
                y = np.empty(self.model.n_parameters + 1)
                for j in range(self.model.n_parameters):
                    parameter = x[j]
                    y[j] = (
                        100
                        * self.model.target_variables_info[current_target_variable][
                            "sensitivity"
                        ][parameter]
                    )
                y[self.model.n_parameters] = (
                    100
                    * self.model.target_variables_info[current_target_variable]["LAE"]
                )
                axs.bar(x, y, color=bar_colors)
                axs.set_title(current_target_variable)
                axs.tick_params(labelrotation=90)
            plt.show()

            return

        fig, axs = plt.subplots()
        fig.supxlabel("")
        fig.supylabel("Sensitivity (%)")
        x = self.model.parameters_names
        x.append("LAE")
        y = np.empty(self.model.n_parameters + 1)
        for j in range(self.model.n_parameters):
            parameter = x[j]
            y[j] = (
                100
                * self.model.target_variables_info[target_variable]["sensitivity"][
                    parameter
                ]
            )
        y[self.model.n_parameters] = (
            100 * self.model.target_variables_info[target_variable]["LAE"]
        )
        axs.bar(x, y, color=bar_colors)
        axs.set_title(target_variable)
        axs.tick_params(labelrotation=90)
        plt.show()

    def all(self):
        self.bar_plot("all")
