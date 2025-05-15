"""This file contains fixtures of CustomSampler used in stochastic classes."""

import numpy as np
import pytest

from rocketpy import CustomSampler


@pytest.fixture
def elevation_sampler():
    """Fixture to create mixture of two gaussian sampler"""
    means_tuple = [1400, 1500]
    sd_tuple = [40, 50]
    prob_tuple = [0.4, 0.6]
    return TwoGaussianMixture(means_tuple, sd_tuple, prob_tuple)


class TwoGaussianMixture(CustomSampler):
    """Class to sample from a mixture of two Gaussian distributions"""

    def __init__(self, means_tuple, sd_tuple, prob_tuple, seed=None):
        """Creates a sampler for a mixture of two Gaussian distributions

        Parameters
        ----------
        means_tuple : 2-tuple
            2-Tuple that contains the mean of each normal distribution of the
            mixture
        sd_tuple : 2-tuple
            2-Tuple that contains the sd of each normal distribution of the
            mixture
        prob_tuple : 2-tuple
            2-Tuple that contains the probability of each normal distribution of the
            mixture. Its entries should be non-negative and sum up to 1.
        """
        np.random.default_rng(seed)
        self.means_tuple = means_tuple
        self.sd_tuple = sd_tuple
        self.prob_tuple = prob_tuple

    def sample(self, n_samples=1):
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
                samples_list[i] = np.random.normal(
                    self.means_tuple[0], self.sd_tuple[0]
                )
            else:
                samples_list[i] = np.random.normal(
                    self.means_tuple[1], self.sd_tuple[1]
                )

        return samples_list

    def reset_seed(self, seed=None):
        """Resets all associated random number generators

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.
        """
        np.random.default_rng(seed)
