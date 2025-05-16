"""
Provides an abstract class so that users can build custom samplers upon
"""

from abc import ABC, abstractmethod


class CustomSampler(ABC):
    """Abstract subclass for user defined samplers"""

    @abstractmethod
    def sample(self, n_samples=1):
        """Generates samples from the custom distribution

        Parameters
        ----------
        n_samples : int, optional
            Numbers of samples to be generated

        Returns
        -------
        samples_list : list
            A list with n_samples elements, each of which is a valid sample
        """

    @abstractmethod
    def reset_seed(self, seed=None):
        """Resets the seeds of all associated stochastic generators

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. The default is None

        Returns
        -------
        None
        """
