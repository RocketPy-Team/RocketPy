.. _custom_sampler:

Implementing custom sampler for Stochastic objects
==================================================

The :ref:`stochastic_usage` documentation teaches how to work with stochastic
objects, discussing the standard initializations, how to create objects and interpret
outputs. Our goal here is to show how to build a custom sampler that gives complete 
control of the distributions used.

Custom Sampler
--------------

Rocketpy provides a ``CustomSampler`` abstract class which works as the backbone of 
a custom sampler. We begin by first importing it and some other useful modules

.. jupyter-execute::

    from rocketpy import CustomSampler
    from matplotlib import pyplot as plt
    import numpy as np

In order to use it, we must create a new class that inherits from 
it and it **must** implement two methods: *sample* and *reset_seed*. The *sample* 
method has one argument, *n_samples*, and must return a list with *n_samples* 
entries, each of which is a sample of the desired variable. The *reset_seed* method 
has one argument, *seed*, which is used to reset the pseudorandom generators in order 
to avoid unwanted dependency across samples. This is especially important when the 
``MonteCarlo`` class is used in parallel mode. 

Below, we give an example of how to implement a mixture of two Gaussian 
distributions. 

.. jupyter-execute::

    class TwoGaussianMixture(CustomSampler):
        """Class to sample from a mixture of two Gaussian distributions
        """

        def __init__(self, means_tuple, sd_tuple, prob_tuple, seed = None):
            """ Creates a sampler for a mixture of two Gaussian distributions

            Parameters
            ----------
            means_tuple : 2-tuple
                2-Tuple that contains the mean of each normal distribution of the
                mixture
            sd_tuple : 2-tuple
                2-Tuple that contains the sd of each normal distribution of the
                mixture
            prob_tuple : 2-tuple
                2-Tuple that contains the probability of each normal distribution 
                of the mixture. Its entries should be non-negative and sum up to 1.
            """
            np.random.default_rng(seed)
            self.means_tuple = means_tuple
            self.sd_tuple = sd_tuple
            self.prob_tuple = prob_tuple

        def sample(self, n_samples = 1):
            """Samples from a mixture of two Gaussian

            Parameters
            ----------
            n_samples : int, optional
                Number of samples, by default 1

            Returns
            -------
            samples_list
                List containing n_samples samples
            """
            samples_list = [0] * n_samples
            mixture_id_list = np.random.binomial(1, self.prob_tuple[0], n_samples)
            for i, mixture_id in enumerate(mixture_id_list):
                if mixture_id:
                    samples_list[i] = np.random.normal(self.means_tuple[0], self.sd_tuple[0])
                else:
                    samples_list[i] = np.random.normal(self.means_tuple[1], self.sd_tuple[1])

            return samples_list

        def reset_seed(self, seed=None):
            """Resets all associated random number generators

            Parameters
            ----------
            seed : int, optional
                Seed for the random number generator.
            """
            np.random.default_rng(seed)

This is an example of a distribution that is not implemented in numpy. Note that it is
a general distribution, so we can use it for many different variables.

.. note::
    You can check more information about the mixture of Gaussian distributions 
    `here <https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model>`. 
    Intuitively, if you think of a single Gaussian as a bell curve distribution, 
    the mixture distribution resembles two bell curves superimposed, each with mode at their
    respective mean.

Example: Gaussian Mixture for Total Impulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use the new created sampler in a stochastic object, we first need
to build an object. In this example, we will consider a case where the distribution of
the total impulse is a mixture of two gaussian with mean parameters 
:math:`(6000, 7000)`, standard deviations :math:`(300, 100)` and mixture probabilities
:math:`(0.7, 0.3)`.

.. jupyter-execute::

    means_tuple = (6000, 7000)
    sd_tuple = (300, 100)
    prob_tuple = (0.7, 0.3)
    total_impulse_sampler = TwoGaussianMixture(means_tuple, sd_tuple, prob_tuple)

Finally, we can create ``StochasticSolidMotor`` object as we did in the example of
:ref:`stochastic_usage`, but we pass the sampler object instead for the *total_impulse*
argument

.. jupyter-execute::

    from rocketpy import SolidMotor, StochasticSolidMotor

    motor = SolidMotor(
        thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        burn_time=3.9,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    stochastic_motor = StochasticSolidMotor(
        solid_motor=motor,
        burn_start_time=(0, 0.1, "binomial"),
        grains_center_of_mass_position=0.001,
        grain_density=10,
        grain_separation=1 / 1000,
        grain_initial_height=1 / 1000,
        grain_initial_inner_radius=0.375 / 1000,
        grain_outer_radius=0.375 / 1000,
        throat_radius=0.5 / 1000,
        nozzle_radius=0.5 / 1000,
        nozzle_position=0.001,
        total_impulse=total_impulse_sampler, # total impulse using custom sampler! 
    )

    stochastic_motor.visualize_attributes()

Let's generate some random motors and check the distribution of the total impulse

.. jupyter-execute::

    total_impulse_samples = [
        stochastic_motor.create_object().total_impulse for _ in range(200)
    ]
    plt.hist(total_impulse_samples, density = True, bins = 30)

Introducing dependency between parameters
-----------------------------------------

Although probabilistic **independency between samples**, i.e. different flight runs, 
is desired for Monte Carlo simulations, it is often important to be able to introduce 
**dependency between parameters**. A clear example of this is wind speed: if we know
the wind speed in the x-axis, then our forecast model might tells us that the wind 
speed y-axis is more likely to be positive than negative, or vice-versa. These 
parameters are then correlated, and using samplers that model these correlations make
the Monte Carlo analysis more robust. 

When we use the default numpy samplers, the Monte Carlo analysis samples the 
parameters independently from each other. However, using custom samplers, we can
introduce dependency and correlation! It might be a bit tricky, but we will show how
it can be done. First, let us import the modules required

.. jupyter-execute::

    from rocketpy import Environment, StochasticEnvironment
    from datetime import datetime, timedelta

Assume we want to model the x and y axis wind speed as a Bivariate Gaussian with
parameters :math:`\mu = (1, 1)` and variance-covariance matrix 
:math:`\Sigma = \begin{bmatrix} 0.2 & 0.17 \\ 0.17 & 0.3 \end{bmatrix}`. This will 
make the correlation between the speeds be of :math:`0.7`. 

Now, in order to correlate the parameters using different custom samplers, 
**the key trick is to create a common generator that will be used by both.** The code 
below implements an example of such a generator

.. jupyter-execute::

    class BivariateGaussianGenerator:
        """Bivariate Gaussian generator used across custom samplers
        """
        def __init__(self, mean, cov, seed = None):
            """Initializes the generator

            Parameters
            ----------
            mean : tuple, list
                Tuple or list with mean of bivariate Gaussian
            cov : np.array
                Variance-Covariance matrix
            seed : int, optional
                Number to seed random generator, by default None
            """
            np.random.default_rng(seed)
            self.samples_list = []
            self.samples_generated = 0
            self.used_samples_x = 0
            self.used_samples_y = 0
            self.mean = mean
            self.cov = cov
            self.generate_samples(1000)

        def generate_samples(self, n_samples = 1):
            """Generate samples from bivariate Gaussian and append to sample list

            Parameters
            ----------
            n_samples : int, optional
                Number of samples to be generated, by default 1
            """
            samples_generated = np.random.multivariate_normal(self.mean, self.cov, n_samples)
            self.samples_generated += n_samples
            self.samples_list += list(samples_generated)

        def reset_seed(self, seed=None):
            np.random.default_rng(seed)

        def get_samples(self, n_samples, axis):
            if axis == "x":
                if self.samples_generated < self.used_samples_x:
                    self.generate_samples(n_samples)
                samples_list = [
                    sample[0] for sample in self.samples_list[self.used_samples_x:(self.used_samples_x + n_samples)]
                ]
            if axis == "y":
                if self.samples_generated < self.used_samples_y:
                    self.generate_samples(n_samples)
                samples_list = [
                    sample[1] for sample in self.samples_list[self.used_samples_y:(self.used_samples_y + n_samples)]
                ]
            self.update_info(n_samples, axis)
            return samples_list

        def update_info(self, n_samples, axis):
            """Updates the information of the used samples

            Parameters
            ----------
            n_samples : int
                Number of samples used
            axis : str
                Which axis was sampled
            """
            if axis == "x":
                self.used_samples_x += n_samples
            if axis == "y":
                self.used_samples_y += n_samples

This generator samples from the bivariate Gaussian and stores then in a *samples_list*
attribute. Then, the idea is to create two samplers for the wind speeds that will share 
an object of this class and their sampling methods only get samples from the stored
sample list.

.. jupyter-execute::

    class WindXSampler(CustomSampler):
        """Samples from X"""

        def __init__(self, bivariate_gaussian_generator):
            self.generator = bivariate_gaussian_generator

        def sample(self, n_samples=1):
            samples_list = self.generator.get_samples(n_samples, "x")
            return samples_list

        def reset_seed(self, seed=None):
            self.generator.reset_seed(seed)

    class WindYSampler(CustomSampler):
        """Samples from Y"""

        def __init__(self, bivariate_gaussian_generator):
            self.generator = bivariate_gaussian_generator

        def sample(self, n_samples=1):
            samples_list = self.generator.get_samples(n_samples, "y")
            return samples_list

        def reset_seed(self, seed=None):
            self.generator.reset_seed(seed)

Then, we create the objects

.. jupyter-execute::

    mean = [1, 2]
    cov_mat = [[0.2, 0.171], [0.171, 0.3]]
    bivariate_gaussian_generator = BivariateGaussianGenerator(mean, cov_mat)
    wind_x_sampler = WindXSampler(bivariate_gaussian_generator)
    wind_y_sampler = WindYSampler(bivariate_gaussian_generator)

With the sample objects created, we only need to create the stochastic objects and
pass them as argument

.. jupyter-execute::

    spaceport_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    spaceport_env.set_atmospheric_model("custom_atmosphere", wind_u = 1, wind_v = 1)
    spaceport_env.set_date(datetime.now() + timedelta(days=1))

    stochastic_environment = StochasticEnvironment(
        environment=spaceport_env,
        elevation=(1400, 10, "normal"),
        gravity=None,
        latitude=None,
        longitude=None,
        ensemble_member=None,
        wind_velocity_x_factor=wind_x_sampler,
        wind_velocity_y_factor=wind_y_sampler
    )

Finally, let us check that if there is a correlation between the wind speed in the
two axis

.. jupyter-execute::

    wind_velocity_x_samples = [0] * 200
    wind_velocity_y_samples = [0] * 200
    for i in range(200):
         stochastic_environment.create_object()
         wind_velocity_x_samples[i] = stochastic_environment.obj.wind_velocity_x(0)
         wind_velocity_y_samples[i] = stochastic_environment.obj.wind_velocity_y(0)

    np.corrcoef(wind_velocity_x_samples, wind_velocity_y_samples)
