.. _sensitivity_analysis:

Sensitivity Analysis
====================

You can use the results from a Monte Carlo simulation to perform sensitivity analysis.
We will first introduce the concepts of sensitivity analysis and then show how to use the `SensitivityModel` class.

It is highly recommended that you read about the Monte Carlo simulations.

Sources of Uncertainty
----------------------

The goal of any simulation software is to provide accurate estimates of the properties
of the simulated phenomena or process. For RocketPy, the goal is to accurately estimate 
rocket flight trajectories, so that the predicted trajectory closely resembles the observed 
trajectory.

To that end, we must understand the factors that increase variability in the predictions. 
From all sources of variation, there are four of major importance:

1. **Rocket Physics model**: consists of the physics models used in rocketry. It encompasses
which rocketry elements we can incorporate such as different types of motors, aerodynamic
surfaces, and other rocket components along with the mathematical equations used to describe them.

2. **Numerical approximations**: consists of how well we can solve the physics equations.
Analytic solutions are seldomly available, and therefore we must resort on numerical
approximations.

3. **Weather forecast**: consists of how well the environment is predicted. Accurate predictions 
are crucial for rocketry simulation as many components are influenced by it.

4. **Measurement uncertainty**: consists of measurement errors. Every instrument has a limited
precision, which causes us to simulate flights with parameters values that are not the true
values but should be somewhat close.

Accurate predictions requires analyzing carefully each source of variation, and this is
RocketPy's goal. The first two sources of variation are naturally handled in the simulator
itself as the library is enhanced with new rocketry components and computational methods.
Weather forecasting is also described extensively in RocketPy, where we allow the forecast
to be customized, come from different reference sources and even be an ensemble from forecasts.

The goal of sensitivity analysis is to analyze the variation due to measurement uncertainty.
Sensitivity analysis quantifies the magnitude of the effect that the variability in 
rocket parameters causes in variables of interest.

Framing the question
^^^^^^^^^^^^^^^^^^^^

Let us explore sensitivity analysis in more detail in a simplified yet practical example.
Consider that we will launch the Calisto Rocket and one of the goals is for its apogee
to reach at least 3650 meters above ground level. Will it reach this target apogee
under current specifications? To answer that question, we build Calisto in RocketPy, run
the simulations and get a predicted apogee of 3781 meters (AGL). Is this the final
answer to that question, then?

Well, the previous section just discussed that there is always uncertainty surrounding
that value. RocketPy, together with accurate modelling of the rocket and the environment,
takes care of the first three source of uncertainty. We need to deal, then, with
instrumental error. 

The code below defines a dictionary containing a description of the instrumental errors
for the parameters of the Rocket. They have been described in the following manner:
the keys of the first dictionary are the parameters names. Then, for each parameter,
we have a dictionary containing the *mean* of that parameter, referring to the nominal
value of that parameter, i.e. the measured value by the instrument, and the *sd*, which
is the standard deviation of the measurement.

.. jupyter-execute::

    analysis_parameters = {
        # Rocket
        "mass": {"mean": 14.426, "std": 0.5},
        "radius": {"mean": 127 / 2000, "std": 1 / 1000},
        # Motor
        "motors_dry_mass": {"mean": 1.815, "std": 1 / 100},
        "motors_grain_density": {"mean": 1815, "std": 50},
        "motors_total_impulse": {"mean": 6500, "std": 50},
        "motors_burn_out_time": {"mean": 3.9, "std": 0.2},
        "motors_nozzle_radius": {"mean": 33 / 1000, "std": 0.5 / 1000},
        "motors_grain_separation": {"mean": 5 / 1000, "std": 1 / 1000},
        "motors_grain_initial_height": {"mean": 120 / 1000, "std": 1 / 100},
        "motors_grain_initial_inner_radius": {"mean": 15 / 1000, "std": 0.375 / 1000},
        "motors_grain_outer_radius": {"mean": 33 / 1000, "std": 0.375 / 1000},
        # Parachutes
        "parachutes_cd_s": {"mean": 10, "std": 0.1},
        "parachutes_lag": {"mean": 1.5, "std": 0.1},
        # Flight
        "heading": {"mean": 53, "std": 2},
        "inclination": {"mean": 84.7, "std": 1},
    }

For simplicity, these are the only instrumental uncertainties that we will deal in this
example. The standard deviation is in the same unit as the mean. 

Notice how the uncertainty varies across different parameters. For instance,
the balance used to measure the mass of the Rocket had a standard deviation of
500 grams, which is not admissible in practice. Certainly, having such a large
uncertainty in the rocket mass will cause a large uncertainty in the apogee.

The question that sensitivity analysis will answer in this example is the 
following: what variables (rocket parameters) cause the most variability
in the predicted apogee? By answering this question, we will be able to
understand which parameters have to be measured more accurately so that
we are more certain about the apogee prediction.

We will show you how to perform sensitivity analysis and interpret its
results.

.. seealso::
    If you are unfamiliar with the Calisto Rocket, see :ref:`firstsimulation`

Importing Monte Carlo Data
--------------------------

Sensitivity analysis requires data from Monte Carlo simulations. We show, below,
the import process. Notice that we need to define the target variables of interest,
in this case the apogee, and the rocket parameters considered for the analysis,
which are given by the entries of the previous dictionary.

.. jupyter-execute::
    
    from rocketpy.tools import load_monte_carlo_data

    target_variables = ["apogee"]
    parameters = list(analysis_parameters.keys())

    parameters_matrix, target_variables_matrix = load_monte_carlo_data(
        input_filename="monte_carlo_analysis_outputs/sensitivity_analysis_data.inputs.txt",
        output_filename="monte_carlo_analysis_outputs/sensitivity_analysis_data.outputs.txt",
        parameters_list=parameters,
        target_variables_list=target_variables,
    )
    # The elevation (ASL) at the launch-site
    elevation = 1400
    # The apogee was saved as ASL, we need to remove the launch site elevation
    target_variables_matrix -= elevation


Creating and fitting a `SensitivityModel`
-----------------------------------------
We pass the parameters list and target variables list to the `SensitivityModel`
object in order to create it.


.. jupyter-execute::
    from rocketpy import SensitivityModel

    model = SensitivityModel(parameters, target_variables)

If we know the nominal values for the parameters and target variables in the
simulation, we can pass them using the methods `set_parameters_nominal` and
`set_target_variables_nominal`. If we do not pass it to the model, the fit method
estimates them from data. In this example, we will pass the nominal values only for the
parameters and let the method estimate the nominals for the target variables.

.. jupyter-execute::
    parameters_nominal_mean = [
        analysis_parameters[parameter_name]["mean"]
        for parameter_name in analysis_parameters.keys()
    ]
    parameters_nominal_sd = [
        analysis_parameters[parameter_name]["std"]
        for parameter_name in analysis_parameters.keys()
    ]
    model.set_parameters_nominal(parameters_nominal_mean, parameters_nominal_sd)

Finally, we fit the model by passing the parameters and target
variables matrices loaded previously.

.. jupyter-execute::
    model.fit(parameters_matrix, target_variables_matrix)

Results
-------
The results can be accessed through the `prints` and `plots` attributes, just 
like any other rocketpy object.

.. jupyter-execute::
    model.plots.bar_plot()


.. jupyter-execute::
    model.prints.all()

Interpreting the Results
------------------------

Final Considerations
--------------------