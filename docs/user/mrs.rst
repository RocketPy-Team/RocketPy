.. _MRS:

Multivariate Rejection Sampling
===============================

Multivariate Rejection Sampling allows you to quickly subsample the results of a
previous Monte Carlo simulation to obtain the results when one or more variables
have a different probability distribution without having to re-run the simulation.

We will show you how to use the  :class:`rocketpy.MultivariateRejectionSampler`
class to possibly save time. It is highly recommended that you read about Monte
Carlo simulations.

Motivation
----------

As discussed in :ref:`sensitivity-practical`, there are several sources of
uncertainty that can affect the flight of a rocket, notably the weather and
the measurements errors in design parameters. Still, it is desirable that the flight
accomplishes its goal, for instance reaching a certain apogee, as well as staying under
some safety restrictions, such as ensuring that the landing point is outside
of a given area.

Monte Carlo simulation is a technique that allows us to quantify the uncertainty and
give objective answers to those types of questions in terms of probabilities and
statistics. It relies on running several simulations under different conditions
specified by probability distributions provided by the user. Hence, depending on the
inputs and number of samples, running these Monte Carlo simulations might take a while.

Now, imagine that you ran and saved the Monte Carlo simulations. Later, you need new a
Monte Carlo simulation with different probability distributions that are somewhat close
to the original simulation. The first straightforward option is to just re-run the
monte carlo with the new arguments, but this might be time consuming. A second option
is to use a sub-sampler that leverages the existing simulation to produce a new sample
that conforms to the new probability distributions. The latter completely avoids
the necessity of re-running the simulations and is, therefore, much faster.

The Multivariate Rejection Sampler, or just MRS, is an algorithm that sub-samples the
original results based on weights proportional to the ratio of the new and old
probability distributions that have changed. The final result has a smaller sample size,
but their distribution matches the one newly specified without having to re-run
the simulation.

The time efficiency of the MRS is especially interesting in two scenarios: quick testing
and tight schedules. Imagine you have an initial design and ran a huge robust monte
carlo simulation but you are also interested in minor variations of the original
design. Instead of having to run an expensive monte carlo for each of these variations,
you can just re-sample the original accordingly. For tight schedules, it is not
unheard of cases where last minute changes have to be made to simulations. The MRS might
then allow you to quickly have monte carlo results for the new configuration when a
full simulation might just take more time than available.

Importing and using the MRS
---------------------------

We now show how to actually use the :class:`rocketpy.MultivariateRejectionSampler`
class. We begin by importing it along with other utilities

.. jupyter-execute::

    from rocketpy import MultivariateRejectionSampler, MonteCarlo
    import numpy as np
    from scipy.stats import norm

The reference monte carlo simulation used is the one from the
"monte_carlo_class_usage.ipynb" notebook with a 1000 samples. A
MultivariateRejectionSampler object is initialized by giving two file paths, one
for the prefix of the original monte carlo simulation, and one for the output of the
sub-samples. The code below defines these strings and initializes the MRS object


.. jupyter-execute::

    monte_carlo_filepath = (
        "notebooks/monte_carlo_analysis/monte_carlo_analysis_outputs/monte_carlo_class_example"
    )
    mrs_filepath = "notebooks/monte_carlo_analysis/monte_carlo_analysis_outputs/mrs"
    mrs = MultivariateRejectionSampler(
        monte_carlo_filepath=monte_carlo_filepath,
        mrs_filepath=mrs_filepath,
    )

Running a monte carlo simulation requires you to specify the distribution of
all parameters that have uncertainty. The MRS, however, only needs the previous and new
distributions of the parameters whose distribution changed. All other random parameters
in the original monte carlo simulation retain their original distribution.

In the original simulation, the mass of the rocket had a normal distribution with mean
:math:`15.426` and standard deviation of :math:`0.5`. Assume that the mean of this
distribution changed to :math:`15` and the standard deviation remained the same. To
run the mrs, we create a dictionary whose keys are the name of the parameter and the
value is a 2-tuple: the first entry contains the pdf of the old distribution, and the
second entry contains the pdf of the new distribution. The code below shows how to
create these distributions and the dictionary

.. jupyter-execute::

    old_mass_pdf = norm(15.426, 0.5).pdf
    new_mass_pdf = norm(15, 0.5).pdf
    distribution_dict = {
        "mass": (old_mass_pdf, new_mass_pdf),
    }

Finally, we execute the `sample` method, as shown below

.. jupyter-execute::
  :hide-code:
  :hide-output:

  np.random.seed(seed=42)

.. jupyter-execute::

    mrs.sample(distribution_dict=distribution_dict)

And that is it! The MRS has saved a file that has the same structure as the results of
a monte carlo simulation but now the mass has been sampled from the newly stated
distribution. To see that it is actually the case, let us import the results of the MRS
and check the mean and standard deviation of the mass. First, we import in the same
way we import the results from a monte carlo simulation


.. jupyter-execute::

    mrs_results = MonteCarlo(mrs_filepath, None, None, None)
    mrs_results.import_results()

Notice that the sample size is now smaller than 1000 samples. Albeit the sample size is
now random, we can check the expected number of samples by printing the
`expected_sample_size` attribute

.. jupyter-execute::

    print(mrs.expected_sample_size)

Now we check the mean and standard deviation of the mass.

.. jupyter-execute::

    mrs_mass_list = []
    for single_input_dict in mrs_results.inputs_log:
        mrs_mass_list.append(single_input_dict["mass"])

    print(f"MRS mass mean after resample: {np.mean(mrs_mass_list)}")
    print(f"MRS mass std after resample: {np.std(mrs_mass_list)}")

They are very close to the specified values.

Comparing Monte Carlo Results
-----------------------------

Alright, now that we have the results for this new configuration, how does it compare
to the original one? Our rocket has, on average, decreased its mass in about 400 grams
while maintaining all other aspects. How do you think, for example, that the distribution
of the apogee has changed? Let us find out.

First, we import the original results

.. jupyter-execute::

    original_results = MonteCarlo(monte_carlo_filepath, None, None, None)

Prints
^^^^^^

We use the `compare_info` method from the `MonteCarlo` class, passing along
the MRS monte carlo object as argument, to print a summary of the comparison

.. jupyter-execute::

    original_results.compare_info(mrs_results)

This summary resemble closely the printed information from one monte carlo simulation
alone, with the difference now that it has a new column, "Source", that alternates the
results between the original and the other simulation. To answer the question proposed
earlier, compare the mean and median of the apogee between both cases. Is it what you
expected?


Histogram and boxplots
^^^^^^^^^^^^^^^^^^^^^^

Besides printed comparison, we can also provide a comparison for the distributions in
the form of histograms and boxplots, using the `compare_plots` method


.. jupyter-execute::

    original_results.compare_plots(mrs_results)

Note that the histograms displays three colors. Two are from the sources, as depicted
in the legend, the third comes from the overlap of the two.

Ellipses
^^^^^^^^

Finally, we can compare the ellipses for the apogees and landing points using the
`compare_ellipses` method

.. jupyter-execute::

    original_results.compare_ellipses(mrs_results, ylim=(-4000, 3000))

Note we can pass along parameters used in the usual `ellipses` method of the
`MonteCarlo` class, in this case the `ylim` argument to expand the y-axis limits.


Calculating Confidence Intervals
--------------------------------

Beyond visual comparisons, you may want to calculate statistical confidence intervals
for specific attributes of the flight (e.g., apogee, max velocity) based on the
resampled data. Since the resulting data is loaded into a ``MonteCarlo`` object,
you can use its method to compute these intervals using the bootstrap method.

The following example shows how to calculate the 95% confidence interval for the
mean of the apogee using the ``mrs_results`` object created earlier:

.. jupyter-execute::

    # Calculate the 95% Confidence Interval for the mean apogee
    # We pass np.mean as the statistic to be evaluated
    apogee_ci = mrs_results.estimate_confidence_interval(
        attribute="apogee",
        statistic=np.mean,
        confidence_level=0.95,
        n_resamples=10000
    )

    print(f"95% Confidence Interval for Apogee Mean: {apogee_ci}")

You can use any statistic function that accepts an array of data, such as ``np.median``
or ``np.std``.




Time Comparison
---------------

Is the MRS really much faster than just re-running a Monte Carlo simulation?
Let us take a look at some numbers. All tests ran in a Dell G15 5530, with 16
13th Gen Intel® Core™ i5-13450HX CPUs, 16GB RAM, running Ubuntu 22.04. Each function
ran 10 times, and no parallelization was used.

To run the original monte carlo simulation with 1000 samples it took,
on average, about 644 seconds, that is, 10 minutes and 44 seconds. For the MRS described
here, it took, on average, 0.15 seconds, with an expected sample size of 117. To re-run
the monte carlo simulations with 117 samples it took, on average, 76.3 seconds. Hence,
the MRS was, on average, (76.3 / 0.15) ~ 500 times faster than re-running the monte
carlo simulations with the same sample size provided by the MRS.


Sampling from nested parameters
-------------------------------

To sample from parameters that are nested in inner levels, use a key name
formed by the name of the key of the outer level concatenated with a "_" and
the key of the inner level. For instance, to change the distribution
from the "total_impulse" parameter, which is nested within the "motors"
parameter dictionary, we have to use the key name "motors_total_impulse".


.. jupyter-execute::

    old_total_impulse_pdf = norm(6500, 1000).pdf
    new_total_impulse_pdf = norm(5500, 1000).pdf
    distribution_dict = {
        "motors_total_impulse": (old_total_impulse_pdf, new_total_impulse_pdf),
    }
    mrs.sample(distribution_dict=distribution_dict)

Finally, if there are multiple named nested structures, we need to use a key
name formed by outer level concatenated with "_", the structure name, "_" and the
inner parameter name. For instance, if we want to change the distribution of
the 'root_chord' of the 'Fins', we have to pass as key
'aerodynamic_surfaces_Fins_root_chord'

.. jupyter-execute::

    old_root_chord_pdf = norm(0.12, 0.001).pdf
    new_root_chord_pdf = norm(0.119, 0.002).pdf
    distribution_dict = {
        "aerodynamic_surfaces_Fins_root_chord": (
            old_root_chord_pdf, new_root_chord_pdf
        ),
    }
    mrs.sample(distribution_dict=distribution_dict)

A word of caution
-----------------

Albeit the MRS provides results way faster than running the simulation again, it
might reduce the sample size drastically. If several variables undergo
changes in their distribution and the more discrepant these are from the original
ones, the more pronounced will be this sample size reduction. If you need the monte
carlo simulations to have the same sample size as before or if the expected sample size
from the MRS is too low for you current application, then it might be better to
re-run the simulations.

References
----------

[1] CEOTTO, Giovani H., SCHMITT, Rodrigo N., ALVES, Guilherme F., et al. RocketPy: six
degree-of-freedom rocket trajectory simulator. Journal of Aerospace Engineering, 2021,
vol. 34, no 6, p. 04021093.

[2] CASELLA, George, ROBERT, Christian P., et WELLS, Martin T. Generalized accept-reject
sampling schemes. Lecture notes-monograph series, 2004, p. 342-347.