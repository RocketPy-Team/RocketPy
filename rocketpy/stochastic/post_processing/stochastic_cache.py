from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np


class SimulationCache:
    def __init__(self, file_name, batch_path):
        self.file_name = file_name
        self.batch_path = Path(batch_path)
        self.sim_number = self._get_sim_number()

    def _get_sim_number(self):
        with h5py.File(self.batch_path / (self.file_name + ".inputs.h5"), 'r') as f:
            return len(f.keys())

    @lru_cache(maxsize=32)
    def read_inputs(self, var_name):
        data = []

        with h5py.File(self.batch_path / (self.file_name + ".inputs.h5"), 'r') as f:
            for i in range(self.sim_number):
                data.append(np.array(f[str(i)][var_name]))

        # Add nans at the end of the data and convert to numpy array
        max_len = max([len(d) for d in data])
        data = [np.concatenate([d, np.full(max_len - len(d), np.nan)]) for d in data]

        return np.array(data)

    @lru_cache(maxsize=32)
    def read_outputs(self, var_name):
        data = []

        with h5py.File(self.batch_path / (self.file_name + ".outputs.h5"), 'r') as f:
            for i in range(self.sim_number):
                data.append(np.array(f[str(i)][var_name]))

        # Add nans at the end of the data and convert to numpy array
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
    data = cache.read_outputs('ax')
    print(data)
    print(data.shape)
