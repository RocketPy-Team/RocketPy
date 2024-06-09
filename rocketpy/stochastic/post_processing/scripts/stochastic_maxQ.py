import matplotlib.pyplot as plt
import numpy as np

from rocketpy.stochastic.post_processing.stochastic_cache import SimulationCache


def compute_maxQ(cache, save, show):
    batch_path = cache.batch_path

    dyn_press = cache.read_outputs('dynamic_pressure') / 1000
    maxQarg = np.nanargmax(dyn_press[:, :, 1], axis=1)
    maxQ = dyn_press[np.arange(len(dyn_press)), maxQarg]

    fig, ax = plt.subplots()
    for i in range(len(dyn_press)):
        ax.plot(dyn_press[i, :, 0], dyn_press[i, :, 1], c='blue')
    ax.scatter(maxQ[:, 0], maxQ[:, 1], c='red', label='Max Q')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Dynamic Pressure [kPa]')
    ax.set_title('Max Q Distribution')
    ax.legend()
    ax.grid()

    if save:
        plt.savefig(batch_path / "Figures" / 'maxQ_distribution.png')

    if show:
        plt.show()


def run(cache, save, show):
    compute_maxQ(cache, save, show)


if __name__ == '__main__':
    import easygui

    # configuration
    file_name = 'monte_carlo_class_example'
    batch_path = easygui.diropenbox(title="Select the batch path")
    cache = SimulationCache(file_name, batch_path)
    run(cache, save=True, show=True)
