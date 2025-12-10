<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/RocketPy_Logo_white.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/RocketPy_Logo_black.png">
  <img alt="RocketPy Logo" src="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/RocketPy_Logo_black.png">
</picture>

<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RocketPy-Team/rocketpy/blob/master/docs/notebooks/getting_started_colab.ipynb)
[![Documentation Status](https://readthedocs.org/projects/rocketpyalpha/badge/?version=latest)](https://docs.rocketpy.org/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/rocketpy?color=g)](https://pypi.org/project/rocketpy/)
![Conda Version](https://img.shields.io/conda/v/conda-forge/rocketpy?color=g)
[![codecov](https://codecov.io/gh/RocketPy-Team/RocketPy/graph/badge.svg?token=Ecc3bsHFeP)](https://codecov.io/gh/RocketPy-Team/RocketPy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributors](https://img.shields.io/github/contributors/RocketPy-Team/rocketpy)](https://github.com/RocketPy-Team/RocketPy/graphs/contributors)
[![Sponsor RocketPy](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/RocketPy-Team)
[![Chat on Discord](https://img.shields.io/discord/765037887016140840?logo=discord)](https://discord.gg/b6xYnNh)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=flat&logo=instagram&logoColor=white)](https://www.instagram.com/rocketpyteam)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/rocketpy)
[![DOI](https://img.shields.io/badge/DOI-10.1061%2F%28ASCE%29AS.1943--5525.0001331-blue.svg)](http://dx.doi.org/10.1061/%28ASCE%29AS.1943-5525.0001331)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/RocketPy-Team/RocketPy)

# RocketPy

RocketPy is the next-generation trajectory simulation solution for High-Power Rocketry. The code is written as a [Python](http://www.python.org) library and allows for a complete 6 degrees of freedom simulation of a rocket's flight trajectory, including high-fidelity variable mass effects as well as descent under parachutes. Weather conditions, such as wind profiles, can be imported from sophisticated datasets, allowing for realistic scenarios. Furthermore, the implementation facilitates complex simulations, such as multi-stage rockets, design and trajectory optimization and dispersion analysis.


## Main features

1. **Nonlinear 6 Degrees of Freedom Simulations**
   - Rigorous treatment of mass variation effects
   - Efficiently solved using LSODA with adjustable error tolerances
   - Highly optimized for fast performance

2. **Accurate Weather Modeling**
   - Supports International Standard Atmosphere (1976)
   - Custom atmospheric profiles and Soundings (Wyoming)
   - Weather forecasts, reanalysis, and ensembles for realistic scenarios

3. **Aerodynamic Models**
   - Optional Barrowman equations for lift coefficients
   - Easy import of drag coefficients from other sources (e.g., CFD simulations)

4. **Parachutes with External Trigger Functions**
   - Test the exact code that will fly
   - Sensor data augmentation with noise for comprehensive parachute simulations

5. **Solid, Hybrid, and Liquid Motors Models**
   - Burn rate and mass variation properties from the thrust curve
   - Define custom rocket tanks based on flux data
   - Support for CSV and ENG file formats

6. **Monte Carlo Simulations**
   - Conduct dispersion analysis and global sensitivity analysis

7. **Flexible and Modular**
   - Perform straightforward engineering analysis (e.g., apogee and lift-off speed as a function of mass)
   - Handle non-standard flights (e.g., parachute drop test from a helicopter)
   - Support multi-stage rockets and custom continuous/discrete control laws
   - Easily create new classes, such as other types of motors

8. **Integration with MATLAB®**
   - Effortlessly run RocketPy from MATLAB®
   - Convert RocketPy results to MATLAB® variables for further processing

These powerful features make RocketPy an indispensable tool for high-power rocket trajectory simulation, catering to enthusiasts, researchers, and engineers in the field of rocketry.

## Validation

RocketPy's features have been validated in our latest [research article published in the Journal of Aerospace Engineering](http://dx.doi.org/10.1061/%28ASCE%29AS.1943-5525.0001331).

The table below shows a comparison between experimental data and the output from RocketPy.
Flight data and rocket parameters used in this comparison were kindly provided by [EPFL Rocket Team](https://github.com/EPFLRocketTeam) and [Notre Dame Rocket Team](https://ndrocketry.weebly.com/).

|         Mission         |    Result Parameter    | RocketPy  | Measured  | Relative Error  |
|:-----------------------:|:-----------------------|:---------:|:---------:|:---------------:|
|   Bella Lui Kaltbrumn   | Apogee altitude (m)    |   461.03  |   458.97  |   **0.45 %**    |
|   Bella Lui Kaltbrumn   | Apogee time (s)        |    10.61  |    10.56  |   **0.47 %**    |
|   Bella Lui Kaltbrumn   | Maximum velocity (m/s) |    86.18  |    90.00  |   **-4.24 %**   |
|   NDRT launch vehicle   | Apogee altitude (m)    | 1,310.44  | 1,320.37  |   **-0.75 %**   |
|   NDRT launch vehicle   | Apogee time (s)        |    16.77  |    17.10  |   **-1.90 %**   |
|   NDRT launch vehicle   | Maximum velocity (m/s) |   172.86  |   168.95  |   **2.31 %**    |

Over years of development and testing, RocketPy has been validated across an expanding range of flight scenarios.
For more information on these validated flights, visit our [Flight Examples](https://docs.rocketpy.org/en/latest/examples/index.html) page in the documentation.

# Documentation

Check out documentation details using the links below:

- [User Guide](https://docs.rocketpy.org/en/latest/user/index.html)
- [Code Documentation](https://docs.rocketpy.org/en/latest/reference/index.html)
- [Development Guide](https://docs.rocketpy.org/en/latest/development/index.html)
- [Technical Documentation](https://docs.rocketpy.org/en/latest/technical/index.html)
- [Flight Examples](https://docs.rocketpy.org/en/latest/examples/index.html)

<br>

# Join Our Community!

RocketPy is growing fast! Many university groups and rocket hobbyists have already started using it. The number of stars and forks for this repository is skyrocketing. And this is all thanks to a great community of users, engineers, developers, marketing specialists, and everyone interested in helping.

If you want to be a part of this and make RocketPy your own, join our [Discord](https://discord.gg/b6xYnNh) server today!

<br>

# Previewing

You can preview RocketPy's main functionalities by browsing through a sample notebook in [Google Colab](https://colab.research.google.com/github/RocketPy-Team/rocketpy/blob/master/docs/notebooks/getting_started_colab.ipynb). No installation is required!

When you are ready to run RocketPy locally, you can read the *Getting Started* section!

<br>

# Getting Started

## Quick Installation

To install RocketPy's latest stable version from PyPI, just open up your terminal and run:

```shell
pip install rocketpy
```

For other installation options, visit our [Installation Docs](https://docs.rocketpy.org/en/latest/user/installation.html).
To learn more about RocketPy's requirements, visit our [Requirements Docs](https://docs.rocketpy.org/en/latest/user/requirements.html).

## Running Your First Simulation

In order to run your first rocket trajectory simulation using RocketPy, you can start a Jupyter Notebook and navigate to the `docs/notebooks` folder. Open `getting_started.ipynb` and you are ready to go. We recommend that you read the [First Simulation](https://docs.rocketpy.org/en/latest/user/first_simulation.html) page to get a complete description.

Otherwise, you may want to create your own script or your own notebook using RocketPy. To do this, let's see how to use RocketPy's four main classes:

- `Environment` - Keeps data related to weather.
- `Motor` - Subdivided into `SolidMotor`, `HybridMotor` and `LiquidMotor`. Keeps data related to rocket motors.
- `Rocket` - Keeps data related to a rocket.
- `Flight` - Runs the simulation and keeps the results.

The following image shows how the four main classes interact with each other:

![Diagram](https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/Fluxogram-Page-2.svg)

A typical workflow starts with importing these classes from RocketPy:

```python
from rocketpy import Environment, Rocket, SolidMotor, Flight
```

An optional step is to import datetime, which is used to define the date of the simulation:

```python
import datetime
```

Then create an Environment object. To learn more about it, you can use:

```python
help(Environment)
```

A sample code is:

```python
env = Environment(
    latitude=32.990254,
    longitude=-106.974998,
    elevation=1400,
)

tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env.set_date(
  (tomorrow.year, tomorrow.month, tomorrow.day, 12), timezone="America/Denver"
) # Tomorrow's date in year, month, day, hour UTC format

env.set_atmospheric_model(type='Forecast', file='GFS')
```

This can be followed up by starting a Solid Motor object. To get help on it, just use:

```python
help(SolidMotor)
```

A sample Motor object can be created by the following code:

```python
Pro75M1670 = SolidMotor(
    thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    center_of_dry_mass_position=0.317,
    grains_center_of_mass_position=0.397,
    burn_time=3.9,
    grain_number=5,
    grain_separation=0.005,
    grain_density=1815,
    grain_outer_radius=0.033,
    grain_initial_inner_radius=0.015,
    grain_initial_height=0.12,
    nozzle_radius=0.033,
    throat_radius=0.011,
    interpolation_method="linear",
    nozzle_position=0,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)
```

With a Solid Motor defined, you are ready to create your Rocket object. As you may have guessed, to get help on it, use:

```python
help(Rocket)
```

A sample code to create a Rocket is:

```python
calisto = Rocket(
    radius=0.0635,
    mass=14.426,  # without motor
    inertia=(6.321, 6.321, 0.034),
    power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
    power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

buttons = calisto.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.6182,
    angular_position=45,
)

calisto.add_motor(Pro75M1670, position=-1.255)

nose = calisto.add_nose(
    length=0.55829, kind="vonKarman", position=1.278
)

fins = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.040,
    span=0.100,
    sweep_length=None,
    cant_angle=0,
    position=-1.04956,
)

tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)
```

You may want to add parachutes to your rocket as well:

```python
main = calisto.add_parachute(
    name="main",
    cd_s=10.0,
    trigger=800,  # ejection altitude in meters
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
    radius=1.5,
    height=1.5,
    porosity=0.0432,
)

drogue = calisto.add_parachute(
    name="drogue",
    cd_s=1.0,
    trigger="apogee",  # ejection at apogee
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
    radius=1.5,
    height=1.5,
    porosity=0.0432,
)
```

Finally, you can create a Flight object to simulate your trajectory. To get help on the Flight class, use:

```python
help(Flight)
```

To actually create a Flight object, use:

```python
test_flight = Flight(
  rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0
)
```

Once the Flight object is created, your simulation is done! Use the following code to get a summary of the results:

```python
test_flight.info()
```

To see all available results, use:

```python
test_flight.all_info()
```

Here is just a quick taste of what RocketPy is able to calculate. There are hundreds of plots and data points computed by RocketPy to enhance your analyses.

![6-DOF Trajectory Plot](https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/rocketpy_example_trajectory.svg)

If you want to see the trajectory on Google Earth, RocketPy acn easily export a KML file for you:

```python
test_flight.export_kml(file_name="test_flight.kml")
```

<img alt="6-DOF Trajectory Plot" src="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/trajectory-earth.png" width="501">

# Authors and Contributors

This package was originally created by [Giovani Ceotto](https://github.com/giovaniceotto/) as part of his work at [Projeto Jupiter](https://github.com/Projeto-Jupiter/). [Rodrigo Schmitt](https://github.com/rodrigo-schmitt/) was one of the first contributors. Later, [Guilherme Fernandes](https://github.com/Gui-FernandesBR/) and [Lucas Azevedo](https://github.com/lucasfourier/) joined the team to work on the expansion and sustainability of this project.

Since then, the [RocketPy Team](https://github.com/orgs/RocketPy-Team/teams/rocketpy-team) has been growing fast and our contributors are what makes us special!

## Institutional Contributors

RocketPy extends its gratitude to the following institutions for their support and contributions:

<div>
    <a href="https://github.com/Projeto-Jupiter">
        <picture align=top>
            <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/institutional/projeto_jupiter_dark.png" height="150px">
            <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/institutional/projeto_jupiter_light.jpg" height="150px">
            <img align=top id="projeto-jupiter-image" alt="Projeto Jupiter Logo" src="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/institutional/projeto_jupiter_light.jpg" height="150px">
        </picture>
    </a>
    <a href="https://github.com/Space-Enterprise-at-Berkeley">
        <img align=top alt="Space Enterprise at Berkeley Logo" src="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/institutional/space_enterprise_at_berkeley.jpeg" height="150px">
    </a>
    <a href="https://www.instagram.com/faradayupv">
        <img align=top alt="Faraday Rocketry UPV Logo" src="https://raw.githubusercontent.com/RocketPy-Team/RocketPy/master/docs/static/institutional/faraday_team_logo.jpg" height="150px">
    </a>
</div>

## Individual Contributors

RocketPy is also indebted to a growing list of individual contributors who actively participate in its development. These include:

[![GitHub Contributors Image](https://contrib.rocks/image?repo=RocketPy-Team/RocketPy)](https://github.com/RocketPy-Team/RocketPy/contributors)

See a [detailed list of contributors](https://github.com/RocketPy-Team/RocketPy/contributors) who are actively working on RocketPy.

## Supporting RocketPy and Contributing

The easiest way to help RocketPy is to demonstrate your support by starring our repository!

[![starcharts stargazers over time](https://starchart.cc/rocketpy-team/rocketpy.svg)](https://starchart.cc/rocketpy-team/rocketpy)

You can also become a [sponsor](https://github.com/sponsors/RocketPy-Team) and help us financially to keep the project going.

If you are actively using RocketPy in one of your projects, reaching out to our core team via [Discord](https://discord.gg/b6xYnNh) and providing feedback can help improve RocketPy a lot!

And if you are interested in going one step further, please read the [development documentation](https://docs.rocketpy.org/en/latest/development/index.html) to learn more about how you can contribute to the development of this next-gen trajectory simulation solution for rocketry.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/RocketPy-Team/RocketPy/blob/master/LICENSE) file for details

## Release Notes

Want to know which bugs have been fixed and the new features of each version? Check out the [release notes](https://github.com/RocketPy-Team/RocketPy/releases).
