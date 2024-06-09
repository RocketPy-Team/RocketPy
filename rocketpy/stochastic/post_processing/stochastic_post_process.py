# from pathlib import Path
import importlib
import pkgutil
from pathlib import Path

from rocketpy.stochastic.post_processing import scripts
from rocketpy.stochastic.post_processing.stochastic_cache import SimulationCache


def list_scripts(package):
    scripts = []
    for module in pkgutil.iter_modules(package.__path__):
        scripts.append(module)
    return scripts


def run_all_scripts(file_name, batch_path, save, show):
    # initialize path
    batch_path = Path(batch_path)

    figures_path = batch_path / "Figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # list all post_processing scripts
    scripts_list = list_scripts(scripts)

    # iniialize cache
    cache = SimulationCache(file_name, batch_path)

    # for each script, execute 'run' giving the cache as argument
    for module in scripts_list:
        print("Running script: ", module.name)

        # get the module
        # Import the module using the module name
        module = importlib.import_module(f"{scripts.__name__}.{module.name}")

        # Extract the function named 'run'
        run_function = getattr(module, 'run')

        run_function(cache, save, show)


if __name__ == "__main__":
    import easygui

    # configuration
    file_name = 'monte_carlo_class_example'
    batch_path = easygui.diropenbox(title="Select the batch path")
    save = True
    show = False

    run_all_scripts(file_name, Path(batch_path), save, show)
