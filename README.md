[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giovaniceotto/rocketpy/blob/master/docs/notebooks/getting_started_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/rocketpy?color=g)](https://pypi.org/project/rocketpy/)
[![Downloads](https://pepy.tech/badge/rocketpyalpha)](https://pepy.tech/project/rocketpy)
[![Documentation Status](https://readthedocs.org/projects/rocketpyalpha/badge/?version=latest)](https://rocketpyalpha.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://app.travis-ci.com/Projeto-Jupiter/RocketPy.svg?branch=master)](https://app.travis-ci.com/Projeto-Jupiter/RocketPy)
[![DOI](https://img.shields.io/badge/DOI-10.1061%2F(ASCE)AS.1943--5525.0001331-blue.svg)](http://dx.doi.org/10.1061/(ASCE)AS.1943-5525.0001331)
[![Chat on Discord](https://img.shields.io/discord/765037887016140840?logo=discord)](https://discord.gg/b6xYnNh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/rocketpy)
[![Contributors](https://img.shields.io/github/contributors/Projeto-Jupiter/rocketpy)](https://github.com/Projeto-Jupiter/RocketPy/graphs/contributors)

# RocketPy
RocketPy is a trajectory simulation for High-Power Rocketry built by [Projeto Jupiter](https://www.facebook.com/ProjetoJupiter/). The code is written as a [Python](http://www.python.org) library and allows for a complete 6 degrees of freedom simulation of a rocket's flight trajectory, including high fidelity variable mass effects as well as descent under parachutes. Weather conditions, such as wind profile, can be imported from sophisticated datasets, allowing for realistic scenarios. Furthermore, the implementation facilitates complex simulations, such as multi-stage rockets, design and trajectory optimization and dispersion analysis.

### Main features

<details>
<summary>Nonlinear 6 degrees of freedom simulations</summary>
<ul>
  <li>Rigorous treatment of mass variation effects</li>
  <li>Solved using LSODA with adjustable error tolerances</li>
  <li>Highly optimized to run fast</li>
</ul>
</details>

<details>
<summary>Accurate weather modeling</summary>
<ul>
  <li>International Standard Atmosphere (1976)</li>
  <li>Custom atmospheric profiles</li>
  <li>Soundings (Wyoming, NOAARuc)</li>
  <li>Weather forecasts and reanalysis</li>
  <li>Weather ensembles</li>
</ul>
</details>

<details>
<summary>Aerodynamic models</summary>
<ul>
  <li>Barrowman equations for lift coefficients (optional)</li>
  <li>Drag coefficients can be easily imported from other sources (e.g. CFD simulations)</li>
</ul>
</details>

<details>
<summary>Parachutes with external trigger functions</summary>
<ul>
  <li>Test the exact code that will fly</li>
  <li>Sensor data can be augmented with noise</li>
</ul>
</details>

<details>
<summary>Solid motors models</summary>
<ul>
  <li>Burn rate and mass variation properties from thrust curve</li>
  <li>CSV and ENG file support</li>
</ul>
</details>

<details>
<summary>Monte Carlo simulations</summary>
<ul>
  <li>Dispersion analysis</li>
  <li>Global sensitivity analysis</li>
</ul>
</details>

<details>
<summary>Flexible and modular</summary>
<ul>
  <li>Straightforward engineering analysis (e.g. apogee and lifting off speed as a function of mass)</li>
  <li>Non-standard flights (e.g. parachute drop test from helicopter)</li>
  <li>Multi-stage rockets</li>
  <li>Custom continuous and discrete control laws</li>
  <li>Create new classes (e.g. other types of motors)</li>
</ul>
</details>

### Documentation

Check out documentation details using the links below:
  - [User Guide](https://rocketpyalpha.readthedocs.io/en/latest/user/index.html)
  - [Code Documentation](https://rocketpyalpha.readthedocs.io/en/latest/reference/index.html)
  - [Development Guide](https://rocketpyalpha.readthedocs.io/en/latest/development/index.html)

## Join Our Community!
RocketPy is growing fast! Many unviersity groups and rocket hobbyist have already started using it. The number of stars and forks for this repository is skyrocketing. And this is all thanks to a great community of users, engineers, developers, marketing specialists, and everyone interested in helping.

If you want to be a part of this and make RocketPy your own, join our [Discord](https://discord.gg/b6xYnNh) server today!

## Previewing

You can preview RocketPy's main functionalities by browsing through a sample notebook either in [Google Colab](https://colab.research.google.com/github/giovaniceotto/rocketpy/blob/master/docs/notebooks/getting_started_colab.ipynb) or in [MyBinder](https://mybinder.org/v2/gh/giovaniceotto/RocketPy/master?filepath=docs%2Fnotebooks%2Fgetting_started.ipynb)!

Then, you can read the *Getting Started* section to get your own copy!

## Getting Started

These instructions will get you a copy of RocketPy up and running on your local machine.

### Prerequisites

The following is needed in order to run RocketPy:

 - Python >= 3.0
 - Numpy >= 1.0
 - Scipy >= 1.0
 - Matplotlib >= 3.0
 - requests
 - netCDF4 >= 1.4 (optional, requires Cython)
 
All of these packages, with the exception of netCDF4, should be automatically
installed when RocketPy is installed using either pip or conda.

However, in case the user wants to install these packages manually, they can do
so by following the instructions bellow:

The first 4 prerequisites come with Anaconda, but Scipy might need
updating. The nedCDF4 package can be installed if there is interest in
importing weather data from netCDF files. To update Scipy and install
netCDF4 using Conda, the following code is used:

```
$ conda install "scipy>=1.0"
$ conda install -c anaconda "netcdf4>=1.4"
```

Alternatively, if you only have Python 3.X installed, the packages needed can be installed using pip:

```
$ pip install "numpy>=1.0"
$ pip install "scipy>=1.0"
$ pip install "matplotlib>=3.0"
$ pip install "netCDF4>=1.4"
$ pip install "requests"
```

Although [Jupyter Notebooks](http://jupyter.org/) are by no means required to run RocketPy, they are strongly recommend. They already come with Anaconda builds, but can also be installed separately using pip:

```
$ pip install jupyter
```

### Installation

To get a copy of RocketPy using pip, just run:

```
$ pip install rocketpy
```

Alternatively, the package can also be installed using conda:

```
$ conda install -c conda-forge rocketpy
```

If you want to downloaded it from source, you may do so either by:

- Downloading it from [RocketPy's GitHub](https://github.com/giovaniceotto/RocketPy) page
    - Unzip the folder and you are ready to go
- Or cloning it to a desired directory using git:
    - ```$ git clone https://github.com/giovaniceotto/RocketPy.git```

The RockeyPy library can then be installed by running:

```
$ python setup.py install 
```

### Documentations

You can find RocketPy's documentation at [Read the Docs](https://rocketpyalpha.readthedocs.io/en/latest/).

### Running Your First Simulation

In order to run your first rocket trajectory simulation using RocketPy, you can start a Jupyter Notebook and navigate to the **_nbks_** folder. Open **_Getting Started - Examples.ipynb_** and you are ready to go.

Otherwise, you may want to create your own script or your own notebook using RocketPy. To do this, let's see how to use RocketPy's four main classes:

- Environment - Keeps data related to weather.
- SolidMotor - Keeps data related to solid motors. Hybrid motor support is coming in the next weeks.
- Rocket - Keeps data related to a rocket.
- Flight - Runs the simulation and keeps the results.

A typical workflow starts with importing these classes from RocketPy:

```python
from rocketpy import Environment, Rocket, SolidMotor, Flight
```

Then create an Environment object. To learn more about it, you can use:

```python
help(Environment)
```

A sample code is:

```python
Env = Environment(
    railLength=5.2,
    latitude=32.990254,
    longitude=-106.974998,
    elevation=1400,
    date=(2020, 3, 4, 12) # Tomorrow's date in year, month, day, hour UTC format
) 

Env.setAtmosphericModel(type='Forecast', file='GFS')
```

This can be followed up by starting a Solid Motor object. To get help on it, just use:

```python
help(SolidMotor)
```

A sample Motor object can be created by the following code:

```python
Pro75M1670 = SolidMotor(
    thrustSource="../data/motors/Cesaroni_M1670.eng",
    burnOut=3.9,
    grainNumber=5,
    grainSeparation=5/1000,
    grainDensity=1815,
    grainOuterRadius=33/1000,
    grainInitialInnerRadius=15/1000,
    grainInitialHeight=120/1000,
    nozzleRadius=33/1000,
    throatRadius=11/1000,
    interpolationMethod='linear'
)
```

With a Solid Motor defined, you are ready to create your Rocket object. As you may have guessed, to get help on it, use:

```python
help(Rocket)
```

A sample code to create a Rocket is:

```python
Calisto = Rocket(
    motor=Pro75M1670,
    radius=127/2000,
    mass=19.197-2.956,
    inertiaI=6.60,
    inertiaZ=0.0351,
    distanceRocketNozzle=-1.255,
    distanceRocketPropellant=-0.85704,
    powerOffDrag='../data/calisto/powerOffDragCurve.csv',
    powerOnDrag='../data/calisto/powerOnDragCurve.csv'
)

Calisto.setRailButtons([0.2, -0.5])

NoseCone = Calisto.addNose(length=0.55829, kind="vonKarman", distanceToCM=0.71971)

FinSet = Calisto.addFins(4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956)

Tail = Calisto.addTail(topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656)
```

You may want to add parachutes to your rocket as well:

```python
def drogueTrigger(p, y):
    return True if y[5] < 0 else False

def mainTrigger(p, y):
    return True if y[5] < 0 and y[2] < 800 else False

Main = Calisto.addParachute('Main',
                            CdS=10.0,
                            trigger=mainTrigger, 
                            samplingRate=105,
                            lag=1.5,
                            noise=(0, 8.3, 0.5))

Drogue = Calisto.addParachute('Drogue',
                              CdS=1.0,
                              trigger=drogueTrigger, 
                              samplingRate=105,
                              lag=1.5,
                              noise=(0, 8.3, 0.5))
```

Finally, you can create a Flight object to simulate your trajectory. To get help on the Flight class, use:

```python
help(Flight)
```

To actually create a Flight object, use:

```python
TestFlight = Flight(rocket=Calisto, environment=Env, inclination=85, heading=0)
```

Once the TestFlight object is created, your simulation is done! Use the following code to get a summary of the results:

```python
TestFlight.info()
```

To seel all available results, use:

```python
TestFlight.allInfo()
```

## Built With

* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [netCDF4](https://github.com/Unidata/netcdf4-python)

## Contributing

Please read [CONTRIBUTING.md](https://github.com/giovaniceotto/RocketPy/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us. - **_Still working on this!_**

## Authors and Contributors

This package was originally created by [Giovani Ceotto](https://github.com/giovaniceotto/) as part of his work at Projeto Jupiter. [Rodrigo Schmitt](https://github.com/rodrigo-schmitt/) was one of the first contributors.

Later, [Guilherme Fernandes](https://github.com/Gui-FernandesBR/) and [Lucas Azevedo](https://github.com/lucasfourier/) joined the team to work on the expansion and sustainability of this project.

Since then, the team has been growing fast and our contributors are what makes us special!

![GitHub Contributors Image](https://contrib.rocks/image?repo=Projeto-Jupiter/RocketPy)

See a [detailed list of contributors](https://github.com/giovaniceotto/RocketPy/contributors) who are actively working on RocketPy.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/giovaniceotto/RocketPy/blob/master/LICENSE) file for details

## Release Notes
Want to know which bugs have been fixed and new features of each version? Check out the [release notes](https://github.com/giovaniceotto/RocketPy/releases).

