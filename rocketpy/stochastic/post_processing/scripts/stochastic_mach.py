from pathlib import Path

import matplotlib.pyplot as plt

from rocketpy.stochastic.post_processing.stochastic_cache import \
    SimulationCache


def compute_mach(file_name, batch_path, save, show):
    cache = SimulationCache(
        file_name,
        batch_path,
    )
    mach = cache.read_outputs('mach_number')

    fig, ax = plt.subplots()
    for i in range(len(mach)):
        ax.plot(mach[i, :, 0], mach[i, :, 1], c='blue')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mach [-]')
    ax.set_title('Mach number Distribution')
    ax.legend()
    ax.grid()

    if save:
        plt.savefig(batch_path / 'mach_distribution.png')

    if show:
        plt.show()
    plt.show()


def run(file_name, batch_path, save, show):
    compute_mach(file_name, batch_path, save, show)


if __name__ == '__main__':
    # import easygui

    batch_path = Path("mc_simulations/")
    file_name = 'monte_carlo_class_example'
    run(file_name, batch_path, save=True, show=True)
