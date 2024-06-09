from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np


class SimulationCache:
    """
    A class to manage caching and reading of simulation input and output data from HDF5 files.

    Attributes
    ----------
    file_name : str
        Name of the base file to read simulation data from.
    batch_path : Path
        Path to the directory containing the batch of simulation files.
    sim_number : int
        Number of simulations available in the input file.

    Methods
    -------
    _get_sim_number():
        Determines the number of simulations in the input HDF5 file.
    read_inputs(var_name):
        Reads and returns the inputs for the specified variable name across all simulations.
    read_outputs(var_name):
        Reads and returns the outputs for the specified variable name across all simulations.
    """

    def __init__(self, file_name, batch_path):
        """
        Initializes the SimulationCache with the specified file name and batch path.

        Parameters
        ----------
        file_name : str
            Name of the base file to read simulation data from.
        batch_path : str or Path
            Path to the directory containing the batch of simulation files.
        """
        self.file_name = file_name
        self.batch_path = Path(batch_path)
        self.sim_number = self._get_sim_number()

    def _get_sim_number(self):
        """
        Determines the number of simulations in the input HDF5 file.

        Returns
        -------
        int
            Number of simulations available in the input file.
        """
        with h5py.File(self.batch_path / (self.file_name + ".inputs.h5"), 'r') as f:
            return len(f.keys())

    @lru_cache(maxsize=32)
    def read_inputs(self, var_name):
        """
        Reads and returns the inputs for the specified variable name across all simulations.

        Parameters
        ----------
        var_name : str
            The name of the variable to read inputs for.

        Returns
        -------
        numpy.ndarray
            A 3D array where each row corresponds to the inputs for the specified variable from one simulation.
        """
        data = []

        with h5py.File(self.batch_path / (self.file_name + ".inputs.h5"), 'r') as f:
            for i in range(self.sim_number):
                data.append(np.array(f[str(i)][var_name]))

        # Add NaNs at the end of the data and convert to a numpy array
        max_len = max([len(d) for d in data])
        data = [
            np.concatenate([d, np.full((max_len - len(d), d.shape[1]), np.nan)])
            for d in data
        ]

        return np.array(data)

    @lru_cache(maxsize=32)
    def read_outputs(self, var_name):
        """
        Reads and returns the outputs for the specified variable name across all simulations.

        Parameters
        ----------
        var_name : str
            The name of the variable to read outputs for.

        Returns
        -------
        numpy.ndarray
            A 3D array where each row corresponds to the outputs for the specified variable from one simulation.
        """
        data = []

        with h5py.File(self.batch_path / (self.file_name + ".outputs.h5"), 'r') as f:
            for i in range(self.sim_number):
                data.append(np.array(f[str(i)][var_name]))

        # Add NaNs at the end of the data and convert to a numpy array
        max_len = max([len(d) for d in data])
        data = [
            np.concatenate([d, np.full((max_len - len(d), d.shape[1]), np.nan)])
            for d in data
        ]

        return np.array(data)


if __name__ == '__main__':
    import easygui

    batch_path = easygui.diropenbox()
    cache = SimulationCache(
        'monte_carlo_class_example',
        batch_path,
    )
    # data = cache.read_outputs('ax')
    # print(data)
    # print(data.shape)
    data = cache.read_inputs('flight/heading')
    print(data)
    print(data.shape)
