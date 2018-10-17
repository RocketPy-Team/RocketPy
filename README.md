# RocketPy
RocketPy is a trajectory simulation for High-Power Rocketry built by [Projeto Jupiter](https://www.facebook.com/ProjetoJupiter/). The code is written as a [Python](http://www.python.org) library and allows for a complete 6 degrees of freedom simulation of a rocket's flight trajectory, including high fidelity variable mass effects as well as descent under parachutes. Weather conditions, such as wind profile, can be imported from sophisticated datasets, allowing for realistic scenarios. Furthermore, the implementation facilitates complex simulations, such as multi-stage rockets, design and trajectory optimization and dispersion analysis.

## Previewing

You can preview RocketPy's main functionalities by browsing through a [sample notebook](https://mybinder.org/v2/gh/giovaniceotto/RocketPy/master?filepath=nbks%2FCalisto.ipynb)!

Then, you can read the *Getting Started* section to get your own copy!

## Getting Started

These instructions will get you a copy of RocketPy up and running on your local machine for development and testing purposes.

### Prerequisites

The following is needed in order to run RocketPy:

 - Python >= 3.0
 - Numpy >= 1.0
 - Scipy >= 1.0
 - Matplotlib >= 3.0
 - netCDF4 >= 1.4 (optional, requires Cython)
 
The first 4 prerequisites come with Anaconda, but Scipy might need updating. The nedCDF4 package can be installed if there is interest in importing weather data from netCDF files. To update Scipy and install netCDF4 using Conda, the following code is used:

```
$ conda install scipy>=1.0
$ conda install -c anaconda netcdf4>=1.4
```

Alternatively, if you only have Python 3.X installed, the four packages needed can be installed using pip:

```
$ pip install "numpy>=1.0"
$ pip install "scipy>=1.0"
$ pip install "matplotlib>=3.0"
$ pip install "netCDF4>=1.4"
```

Although [Jupyter Notebooks](http://jupyter.org/) are by no means required to run RocketPy, they are strongly recommend. They already come with Anaconda builds, but can also be installed separately using pip:

```
$ pip install jupyter
```

### Downloading

To get a copy of RocketPy, you currently have two options:

- Download it from [RocketPy's GitHub](https://github.com/giovaniceotto/RocketPy) page
    - Unzip the folder and you are ready to go
- Or clone it to a desired directory using git:
    - ```$ git clone https://github.com/giovaniceotto/RocketPy.git```

The repository comes with the following content:

- Files
  - README.md
  - LICENSE.md
- Folders
  - data - Input data for the simulation, such as motor thrust curves and wind profiles.
  - disp - Example of dispersion analysis, but needs to be refined.
  - docs - Documentation available about the physics models used.
  - nbks - Main python library, some example notebooks and other random files which will soon be cleaned up.

The main Python library is kept under the **_nkbs_** folder and is currently named **_rocketpyAlpha.py_**. Keep in mind that the files are still being organized for a proper release.

### Running Your First Simulation

In order to run your first rocket trajectory simulation using RocketPy, you can start a Jupyter Notebook and navigate to the **_nbks_** folder. Open **_Calisto.ipynb_** and you are ready to go.

Otherwise, you may want to create your own script or your own notebook using RocketPy. To do this, let's see how to use RocketPy's four main classes:

- Environment - Keeps data related to weather.
- Motor - Keeps data related to solid motors.
- Rocket - Keeps data related to a rocket.
- Flight - Runs the simulation and process the results.

A typical workflow starts with importing these classes from RocketPy:

```
>> from rocketpyAlpha import *
```

Then create an Environment object. To learn more about it, you can use:

```
>> help(Environment)
```

A sample code is:

```
>> Env = Environment(railLength=5.2,
                     gravity=9.8,
                     windData="../data/weather/SpacePort.nc",
                     location=(32.990254, -106.974998),
                     date=(2016, 6, 20, 18))
```

This can be followed up by starting a Motor object. To get help on it, just use:

```
>> help(Motor)
```

A sample Motor object can be created by the following code:

```
>> Cesaroni_M1670 = Motor(thrustSource="../data/motors/Cesaroni_M1670.eng",
                          burnOut=3.9,
                          grainNumber=5,
                          grainSeparation=5/1000,
                          grainDensity=1815,
                          grainOuterRadius=33/1000,
                          grainInitialInnerRadius=15/1000,
                          grainInitialHeight=120/1000,
                          nozzleRadius=33/1000,
                          throatRadius=11/1000)
```

With a Motor defined, you are ready to create your Rocket object. As you may have guessed, to get help on it, use:

```
>> help(Rocket)
```

A sample code to create a Rocket is:

```
>> Calisto = Rocket(motor=Cesaroni_M1670,
                 radius=127/2000,
                 mass=19.197-2.956,
                 inertiaI=6.60,
                 inertiaZ=0.0351,
                 distanceRocketNozzle=1.255,
                 distanceRocketPropellant=0.85704,
                 powerOffDrag='../data/calisto/powerOffDragCurve.csv',
                 powerOnDrag='../data/calisto/powerOnDragCurve.csv')

>> Calisto.addNose(length=0.55829, kind="vonKarman", distanceToCM=0.71971)

>> Calisto.addFins(4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956)

>> Calisto.addTail(topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656)

>> Calisto.addParachute('Drogue',
                     CdS=1.0,
                     trigger=lambda p, y: return y[5] < 0,
                     samplingRate=1,
                     lag=1.5)

>> Calisto.addParachute('Main',
                     CdS=10.0,
                     trigger=lambda p, y: return (y[2] < 500 and y[5] < 0), 
                     samplingRate=1,
                     lag=1.5)
```

Finally, you can create a Flight object to simulate your trajectory. To get help on the Flight class, use:

```
>> help(Flight)
```

To actually create a Flight object, use:

```
>> TestFlight = Flight(rocket=Calisto, environment=Env, inclination=85, heading=0, maxStepSize=0.01, maxTime=600)
```

Once the TestFlight object is created, your simulation is done! Use the following code to get a summary of the results:

```
>> TestFlight.info()
```

To seel all available results, use:
```
>> TestFlight.allInfo()
```

To summarize, the complete code would be:

```
from rocketpyAlpha import *

Env = Environment(railLength=5.2,
                     gravity=9.8,
                     windData="../data/weather/SpacePort.nc",
                     location=(32.990254, -106.974998),
                     date=(2016, 6, 20, 18))
                     
Cesaroni_M1670 = Motor(thrustSource="../data/motors/Cesaroni_M1670.eng",
                          burnOut=3.9,
                          grainNumber=5,
                          grainSeparation=5/1000,
                          grainDensity=1815,
                          grainOuterRadius=33/1000,
                          grainInitialInnerRadius=15/1000,
                          grainInitialHeight=120/1000,
                          nozzleRadius=33/1000,
                          throatRadius=11/1000)

Calisto = Rocket(motor=Cesaroni_M1670,
                 radius=127/2000,
                 mass=19.197-2.956,
                 inertiaI=6.60,
                 inertiaZ=0.0351,
                 distanceRocketNozzle=1.255,
                 distanceRocketPropellant=0.85704,
                 powerOffDrag='../data/calisto/powerOffDragCurve.csv',
                 powerOnDrag='../data/calisto/powerOnDragCurve.csv')

Calisto.addNose(length=0.55829, kind="vonKarman", distanceToCM=0.71971)

Calisto.addFins(4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956)

Calisto.addTail(topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656)

Calisto.addParachute('Drogue',
                     CdS=1.0,
                     trigger=lambda p, y: return y[5] < 0,
                     samplingRate=1,
                     lag=1.5)

Calisto.addParachute('Main',
                     CdS=10.0,
                     trigger=lambda p, y: return (y[2] < 500 and y[5] < 0), 
                     samplingRate=1,
                     lag=1.5)

TestFlight = Flight(rocket=Calisto, environment=Env, inclination=85, heading=0, maxStepSize=0.01, maxTime=600)

TestFlight.info()

TestFlight.allInfo()
```

## Built With

* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [netCDF4](https://github.com/Unidata/netcdf4-python)

## Contributing

Please read [CONTRIBUTING.md](https://github.com/giovaniceotto/RocketPy/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us. - **_Still working on this!_**

## Versioning

**_Still working on this!_**

## Authors

* **Giovani Hidalgo Ceotto**

See also the list of [contributors](https://github.com/giovaniceotto/RocketPy/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/giovaniceotto/RocketPy/blob/master/LICENSE) file for details

## Acknowledgments

**_Still working on this!_**
