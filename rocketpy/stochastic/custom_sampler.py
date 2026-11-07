"""
Provides an abstract class so that users can build custom samplers upon
"""

from abc import ABC, abstractmethod


class CustomSampler(ABC):
    """Abstract subclass for user defined samplers"""

    @property
    def seed_group(self):
        """The generator this sampler shares, or ``self`` if it shares none.

        Samplers sharing a generator must all return it, so the group is seeded
        once rather than each member overwriting the previous seed. Return the
        same object every call: a rebuilt one has a new identity and forms a
        group of its own. A group belongs to one model; declared on two models,
        each seeds it and the last one wins.

        Returns
        -------
        object
            Matched by identity, not equality. Defaults to ``self``.
        """
        return self

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
