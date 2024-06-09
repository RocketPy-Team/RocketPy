import matplotlib.pyplot as plt

from rocketpy.stochastic.post_processing.stochastic_cache import SimulationCache


def compute_mach(cache, save, show):
    batch_path = cache.batch_path

    mach = cache.read_outputs('mach_number')

    fig, ax = plt.subplots()
    for i in range(len(mach)):
        ax.plot(mach[i, :, 0], mach[i, :, 1], c='blue')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mach [-]')
    ax.set_title('Mach number Distribution')
    ax.grid()

    if save:
        plt.savefig(batch_path / "Figures" / 'mach_distribution.png')

    if show:
        plt.show()


def run(cache, save, show):
    compute_mach(cache, save, show)


if __name__ == '__main__':
    import easygui

    # configuration
    file_name = 'monte_carlo_class_example'
    batch_path = easygui.diropenbox(title="Select the batch path")
    cache = SimulationCache(file_name, batch_path)
    run(cache, save=True, show=True)
