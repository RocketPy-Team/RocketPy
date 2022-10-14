""" Create functions that you be imported by monte carlo simulations """

import numpy as np
import math
from numpy.random import choice, normal, uniform
import simplekml
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def flight_settings(analysis_parameters, total_number):
    i = 0
    while i < total_number:
        # Generate a flight setting
        flight_setting = {}
        for parameter_key, parameter_value in analysis_parameters.items():
            if type(parameter_value) is tuple:
                flight_setting[parameter_key] = normal(*parameter_value)
            else:
                flight_setting[parameter_key] = choice(parameter_value)

        # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
        # if flight_setting["lag_rec"] < 0 or flight_setting["lag_se"] < 0:
        #     continue # TODO: get back on the road

        # Update counter
        i += 1
        # Yield a flight setting
        yield flight_setting


def invertedHaversine(lat0, lon0, distance, bearing, eRadius=6.3781e6):
    """Returns a tuple with new latitude and longitude coordinates considering
    a displacement of a given distance in a given direction (bearing compass)
    starting from a point defined by (lat0, lon0). This is the opposite of
    Haversine function.

    Parameters
    ----------
    lat0 : float
        Origin latitude coordinate, in degrees.
    lon0 : float
        Origin longitude coordinate, in degrees.
    distance : float
        Distance from the origin point, in meters.
    bearing : float
        Azimuth (or bearing compass) from the origin point, in degrees.
    eRadius : float, optional
        Earth radius, in meters. Default value is 6.3781e6.
        See the supplement.calculateEarthRadius() function for more accuracy.

    Returns
    -------
    lat1 : float
        New latitude coordinate, in degrees.
    lon1 : float
        New longitude coordinate, in degrees.

    """

    # Convert coordinates to radians
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    # Apply inverted Haversine formula
    lat1_rad = math.asin(
        math.sin(lat0_rad) * math.cos(distance / eRadius)
        + math.cos(lat0_rad)
        * math.sin(distance / eRadius)
        * math.cos(np.deg2rad(bearing))
    )

    lon1_rad = lon0_rad + math.atan2(
        math.sin(np.deg2rad(bearing))
        * math.sin(distance / eRadius)
        * math.cos(lat0_rad),
        math.cos(distance / eRadius) - math.sin(lat0_rad) * math.sin(lat1_rad),
    )

    # Convert back to degrees and then return
    lat1_deg = np.rad2deg(lat1_rad)
    lon1_deg = np.rad2deg(lon1_rad)

    return [lat1_deg, lon1_deg]


def createEllipses(dispersion_results):
    """A function to create apogee and impact ellipses from the dispersion
    results.
    Parameters
    ----------
    dispersion_results : dict
        A dictionary containing the results of the dispersion analysis.
    """
    # Retrieve dispersion data por apogee and impact XY position
    apogeeX = np.array(dispersion_results["apogeeX"])
    apogeeY = np.array(dispersion_results["apogeeY"])
    impactX = np.array(dispersion_results["impactX"])
    impactY = np.array(dispersion_results["impactY"])
    # Define function to calculate eigen values
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    # Calculate error ellipses for impact
    impactCov = np.cov(impactX, impactY)
    impactVals, impactVecs = eigsorted(impactCov)
    impactTheta = np.degrees(np.arctan2(*impactVecs[:, 0][::-1]))
    impactW, impactH = 2 * np.sqrt(impactVals)
    # Draw error ellipses for impact
    impact_ellipses = []
    for j in [1, 2, 3]:
        impactEll = Ellipse(
            xy=(np.mean(impactX), np.mean(impactY)),
            width=impactW * j,
            height=impactH * j,
            angle=impactTheta,
            color="black",
        )
        impactEll.set_facecolor((0, 0, 1, 0.2))
        impact_ellipses.append(impactEll)
    # Calculate error ellipses for apogee
    apogeeCov = np.cov(apogeeX, apogeeY)
    apogeeVals, apogeeVecs = eigsorted(apogeeCov)
    apogeeTheta = np.degrees(np.arctan2(*apogeeVecs[:, 0][::-1]))
    apogeeW, apogeeH = 2 * np.sqrt(apogeeVals)
    apogee_ellipses = []
    # Draw error ellipses for apogee
    for j in [1, 2, 3]:
        apogeeEll = Ellipse(
            xy=(np.mean(apogeeX), np.mean(apogeeY)),
            width=apogeeW * j,
            height=apogeeH * j,
            angle=apogeeTheta,
            color="black",
        )
        apogeeEll.set_facecolor((0, 1, 0, 0.2))
        apogee_ellipses.append(apogeeEll)
    return impact_ellipses, apogee_ellipses


# Checked!
def prepareEllipses(ellipses, origin_lat, origin_lon, resolution=100):
    """Generate a list of latitude and longitude points for each ellipse in
    ellipses.
    Parameters
    ----------
    ellipses : list
        List of matplotlib.patches.Ellipse objects.
    origin_lat : float
        Latitude of the origin of the coordinate system.
    origin_lon : float
        Longitude of the origin of the coordinate system.
    resolution : int, optional
        Number of points to generate for each ellipse, by default 100
    """
    outputs = []

    for ell in ellipses:
        # Get ellipse path points
        center = ell.get_center()
        width = ell.get_width()
        height = ell.get_height()
        angle = np.deg2rad(ell.get_angle())
        points = []

        for i in range(resolution):
            x = width / 2 * math.cos(2 * np.pi * i / resolution)
            y = height / 2 * math.sin(2 * np.pi * i / resolution)
            x_rot = center[0] + x * math.cos(angle) - y * math.sin(angle)
            y_rot = center[1] + x * math.sin(angle) + y * math.cos(angle)
            points.append((x_rot, y_rot))
        points = np.array(points)

        # Convert path points to lat/lon
        lat_lon_points = []
        for point in points:
            x = point[0]
            y = point[1]

            # Convert to distance and bearing
            d = math.sqrt(x**2 + y**2)
            bearing = math.atan2(
                x, y
            )  # math.atan2 returns the angle in the range [-pi, pi]

            lat_lon_points.append(
                invertedHaversine(
                    lat0=origin_lat,
                    lon0=origin_lon,
                    distance=d,
                    bearing=np.rad2deg(bearing),
                    eRadius=6.3781e6,
                )
            )
        # Export string
        outputs.append(lat_lon_points)
    return outputs


def exportEllipsesToKML(
    dispersion_results,
    filename,
    origin_lat,
    origin_lon,
    type="all",
    resolution=100,
    color="ff0000ff",
):
    """Generates a KML file with the ellipses on the impact point.
    Parameters
    ----------
    dispersion_results : dict
        Contains dispersion results from the Monte Carlo simulation.
    filename : String
        Name to the KML exported file.
    origin_lat : float
        Latitude coordinate of Ellipses' geometric center, in degrees.
    origin_lon : float
        Latitude coordinate of Ellipses' geometric center, in degrees.
    type : String
        Type of ellipses to be exported. Options are: 'all', 'impact' and
        'apogee'. Default is 'all', it exports both apogee and impact
        ellipses.
    resolution : int
        Number of points to be used to draw the ellipse. Default is 100.
    color : String
        Color of the ellipse. Default is 'ff0000ff', which is red.
    """

    impact_ellipses, apogee_ellipses = createEllipses(dispersion_results)
    outputs = []

    if type == "all" or type == "impact":
        outputs = prepareEllipses(
            impact_ellipses, origin_lat, origin_lon, resolution=resolution
        )

    if type == "all" or type == "apogee":
        outputs = prepareEllipses(
            apogee_ellipses, origin_lat, origin_lon, resolution=resolution
        )

    # Prepare data to KML file
    kml_data = []
    for i in range(len(outputs)):
        temp = []
        for j in range(len(outputs[i])):
            temp.append((outputs[i][j][1], outputs[i][j][0]))  # log, lat
        kml_data.append(temp)

    print()
    print(kml_data)
    print()

    # Export to KML
    kml = simplekml.Kml()

    for i in range(len(outputs)):
        if (type == "all" and i < 3) or (type == "impact"):
            ellName = "Impact σ" + str(i + 1)
        elif type == "all" and i >= 3:
            ellName = "Apogee σ" + str(i - 2)
        else:
            ellName = "Apogee σ" + str(i + 1)

        mult_ell = kml.newmultigeometry(name=ellName)
        mult_ell.newpolygon(
            outerboundaryis=kml_data[i],
            name="Ellipse " + str(i),
        )
        # Setting ellipse style
        mult_ell.tessellate = 1
        mult_ell.visibility = 1
        mult_ell.style.linestyle.color = color
        mult_ell.style.linestyle.width = 3
        mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
            100, simplekml.Color.blue
        )
        mult_ell.style.polystyle.fill = 1

    kml.save(filename)
    return None


def meanLateralWindSpeed(dispersion_results):
    """_summary_
    Parameters
    ----------
    dispersion_results : _type_
        _description_
    """
    print(
        f'Lateral Surface Wind Speed -Mean Value: {np.mean(dispersion_results["lateralWind"]):0.3f} m/s'
    )
    print(
        f'Lateral Surface Wind Speed - Std. Dev.: {np.std(dispersion_results["lateralWind"]):0.3f} m/s'
    )
    return None


def plotLateralWindSpeed(dispersion_results):
    """_summary_
    Parameters
    ----------
    dispersion_results : _type_
        _description_
    """
    meanLateralWindSpeed(dispersion_results)
    plt.figure()
    plt.hist(
        dispersion_results["lateralWind"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Lateral Surface Wind Speed")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Number of Occurrences")
    plt.show()
    return None


def meanFrontalWindSpeed(dispersion_results):
    """_summary_
    Parameters
    ----------
    dispersion_results : _type_
        _description_
    """
    print(
        f'Frontal Surface Wind Speed -Mean Value: {np.mean(dispersion_results["frontalWind"]):0.3f} m/s'
    )
    print(
        f'Frontal Surface Wind Speed - Std. Dev.: {np.std(dispersion_results["frontalWind"]):0.3f} m/s'
    )
    return None


def plotFrontalWindSpeed(dispersion_results):
    """_summary_
    Parameters
    ----------
    dispersion_results : _type_
        _description_
    """
    meanFrontalWindSpeed(dispersion_results)
    plt.figure()
    plt.hist(
        dispersion_results["frontalWind"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Frontal Surface Wind Speed")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Number of Occurrences")
    plt.show()
    return None


def info(dispersion_results):
    """_summary_
    Returns
    -------
    None
    """
    dispersion_results = dispersion_results
    meanApogeeAltitude(dispersion_results)
    meanOutOfRailVelocity(dispersion_results)
    meanStaticMargin(dispersion_results)
    meanLateralWindSpeed(dispersion_results)
    meanFrontalWindSpeed(dispersion_results)
    printMeanOutOfRailTime(dispersion_results)
    meanApogeeTime(dispersion_results)
    meanApogeeXPosition(dispersion_results)
    meanApogeeYPosition(dispersion_results)
    meanImpactTime(dispersion_results)
    meanImpactVelocity(dispersion_results)
    meanImpactXPosition(dispersion_results)
    meanImpactYPosition(dispersion_results)
    meanMaximumVelocity(dispersion_results)
    meanNumberOfParachuteEvents(dispersion_results)
    meanDrogueFullyInflatedTime(dispersion_results)
    meanDrogueFullyVelocity(dispersion_results)
    meanDrogueTriggerTime(dispersion_results)
    return None


def allInfo(dispersion_results):
    # plotEllipses(dispersion_results, image, actual_landing_point)
    plotApogeeAltitude(dispersion_results)
    plotOutOfRailVelocity(dispersion_results)
    plotStaticMargin(dispersion_results)
    plotLateralWindSpeed(dispersion_results)
    plotFrontalWindSpeed(dispersion_results)
    plotOutOfRailTime(dispersion_results)
    plotApogeeTime(dispersion_results)
    plotApogeeXPosition(dispersion_results)
    plotApogeeYPosition(dispersion_results)
    plotImpactTime(dispersion_results)
    plotImpactVelocity(dispersion_results)
    plotImpactXPosition(dispersion_results)
    plotImpactYPosition(dispersion_results)
    plotMaximumVelocity(dispersion_results)
    plotNumberOfParachuteEvents(dispersion_results)
    plotDrogueFullyInflatedTime(dispersion_results)
    plotDrogueFullyVelocity(dispersion_results)
    plotDrogueTriggerTime(dispersion_results)
    return None


def outOfRailTime(dispersion_results):
    """Calculate the time of the rocket's departure from the rail, in seconds.

    Returns
    -------
    _type_
        _description_
    """
    mean_out_of_rail_time = (
        np.mean(dispersion_results["outOfRailTime"])
        if dispersion_results["outOfRailTime"]
        else None
    )
    std_out_of_rail_time = (
        np.std(dispersion_results["outOfRailTime"])
        if dispersion_results["outOfRailTime"]
        else None
    )
    return mean_out_of_rail_time, std_out_of_rail_time


def printMeanOutOfRailTime(dispersion_results):
    """Prints out the mean and std. dev. of the "outOfRailTime" parameter.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    mean_out_of_rail_time, std_out_of_rail_time = outOfRailTime(dispersion_results)
    print(f"Out of Rail Time -Mean Value: {mean_out_of_rail_time:0.3f} s")
    print(f"Out of Rail Time - Std. Dev.: {std_out_of_rail_time:0.3f} s")

    return None


def plotOutOfRailTime(dispersion_results):
    """Plot the out of rail time distribution

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    plt.figure()
    plt.hist(
        dispersion_results["outOfRailTime"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Out of Rail Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanOutOfRailVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Out of Rail Velocity -Mean Value: {np.mean(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
    )
    print(
        f'Out of Rail Velocity - Std. Dev.: {np.std(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
    )

    return None


def plotOutOfRailVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanOutOfRailVelocity(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["outOfRailVelocity"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Out of Rail Velocity")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanApogeeTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Impact Time -Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
    )
    print(f'Impact Time - Std. Dev.: {np.std(dispersion_results["impactTime"]):0.3f} s')

    return None


def plotApogeeTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanApogeeTime(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["impactTime"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Impact Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanApogeeAltitude(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Apogee Altitude -Mean Value: {np.mean(dispersion_results["apogeeAltitude"]):0.3f} m'
    )
    print(
        f'Apogee Altitude - Std. Dev.: {np.std(dispersion_results["apogeeAltitude"]):0.3f} m'
    )

    return None


def plotApogeeAltitude(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanApogeeAltitude(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["apogeeAltitude"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Apogee Altitude")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanApogeeXPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Apogee X Position -Mean Value: {np.mean(dispersion_results["apogeeX"]):0.3f} m'
    )
    print(
        f'Apogee X Position - Std. Dev.: {np.std(dispersion_results["apogeeX"]):0.3f} m'
    )

    return None


def plotApogeeXPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanApogeeAltitude(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["apogeeX"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Apogee X Position")
    plt.xlabel("Apogee X Position (m)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanApogeeYPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Apogee Y Position -Mean Value: {np.mean(dispersion_results["apogeeY"]):0.3f} m'
    )
    print(
        f'Apogee Y Position - Std. Dev.: {np.std(dispersion_results["apogeeY"]):0.3f} m'
    )

    return None


def plotApogeeYPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanApogeeAltitude(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["apogeeY"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Apogee Y Position")
    plt.xlabel("Apogee Y Position (m)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanImpactTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Impact Time -Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
    )
    print(f'Impact Time - Std. Dev.: {np.std(dispersion_results["impactTime"]):0.3f} s')

    return None


def plotImpactTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanImpactTime(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["impactTime"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Impact Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanImpactXPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Impact X Position -Mean Value: {np.mean(dispersion_results["impactX"]):0.3f} m'
    )
    print(
        f'Impact X Position - Std. Dev.: {np.std(dispersion_results["impactX"]):0.3f} m'
    )

    return None


def plotImpactXPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanImpactXPosition(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["impactX"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Impact X Position")
    plt.xlabel("Impact X Position (m)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanImpactYPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Impact Y Position -Mean Value: {np.mean(dispersion_results["impactY"]):0.3f} m'
    )
    print(
        f'Impact Y Position - Std. Dev.: {np.std(dispersion_results["impactY"]):0.3f} m'
    )

    return None


def plotImpactYPosition(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanImpactYPosition(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["impactY"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Impact Y Position")
    plt.xlabel("Impact Y Position (m)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanImpactVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Impact Velocity -Mean Value: {np.mean(dispersion_results["impactVelocity"]):0.3f} m/s'
    )
    print(
        f'Impact Velocity - Std. Dev.: {np.std(dispersion_results["impactVelocity"]):0.3f} m/s'
    )

    return None


def plotImpactVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanImpactVelocity(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["impactVelocity"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Impact Velocity")
    plt.xlim(-35, 0)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanStaticMargin(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Initial Static Margin -    Mean Value: {np.mean(dispersion_results["initialStaticMargin"]):0.3f} c'
    )
    print(
        f'Initial Static Margin -     Std. Dev.: {np.std(dispersion_results["initialStaticMargin"]):0.3f} c'
    )

    print(
        f'Out of Rail Static Margin -Mean Value: {np.mean(dispersion_results["outOfRailStaticMargin"]):0.3f} c'
    )
    print(
        f'Out of Rail Static Margin - Std. Dev.: {np.std(dispersion_results["outOfRailStaticMargin"]):0.3f} c'
    )

    print(
        f'Final Static Margin -      Mean Value: {np.mean(dispersion_results["finalStaticMargin"]):0.3f} c'
    )
    print(
        f'Final Static Margin -       Std. Dev.: {np.std(dispersion_results["finalStaticMargin"]):0.3f} c'
    )

    return None


def plotStaticMargin(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanStaticMargin(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["initialStaticMargin"],
        label="Initial",
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.hist(
        dispersion_results["outOfRailStaticMargin"],
        label="Out of Rail",
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.hist(
        dispersion_results["finalStaticMargin"],
        label="Final",
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.legend()
    plt.title("Static Margin")
    plt.xlabel("Static Margin (c)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanMaximumVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Maximum Velocity -Mean Value: {np.mean(dispersion_results["maxVelocity"]):0.3f} m/s'
    )
    print(
        f'Maximum Velocity - Std. Dev.: {np.std(dispersion_results["maxVelocity"]):0.3f} m/s'
    )

    return None


def plotMaximumVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanMaximumVelocity(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["maxVelocity"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Maximum Velocity")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanNumberOfParachuteEvents(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Number of Parachute Events -Mean Value: {np.mean(dispersion_results["numberOfEvents"]):0.3f} s'
    )
    print(
        f'Number of Parachute Events - Std. Dev.: {np.std(dispersion_results["numberOfEvents"]):0.3f} s'
    )

    return None


def plotNumberOfParachuteEvents(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanNumberOfParachuteEvents(dispersion_results)

    plt.figure()
    plt.hist(dispersion_results["numberOfEvents"])
    plt.title("Parachute Events")
    plt.xlabel("Number of Parachute Events")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanDrogueTriggerTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Drogue Trigger Time -Mean Value: {np.mean(dispersion_results["drogueTriggerTime"]):0.3f} s'
    )
    print(
        f'Drogue Trigger Time - Std. Dev.: {np.std(dispersion_results["drogueTriggerTime"]):0.3f} s'
    )

    return None


def plotDrogueTriggerTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanDrogueTriggerTime(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["drogueTriggerTime"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Drogue Trigger Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanDrogueFullyInflatedTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Drogue Fully Inflated Time -Mean Value: {np.mean(dispersion_results["drogueInflatedTime"]):0.3f} s'
    )
    print(
        f'Drogue Fully Inflated Time - Std. Dev.: {np.std(dispersion_results["drogueInflatedTime"]):0.3f} s'
    )

    return None


def plotDrogueFullyInflatedTime(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanDrogueFullyInflatedTime(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["drogueInflatedTime"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Drogue Fully Inflated Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None


def meanDrogueFullyVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print(
        f'Drogue Parachute Fully Inflated Velocity -Mean Value: {np.mean(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
    )
    print(
        f'Drogue Parachute Fully Inflated Velocity - Std. Dev.: {np.std(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
    )

    return None


def plotDrogueFullyVelocity(dispersion_results):
    """_summary_

    Parameters
    ----------
    dispersion_results : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanDrogueFullyVelocity(dispersion_results)

    plt.figure()
    plt.hist(
        dispersion_results["drogueInflatedVelocity"],
        bins=int(len(dispersion_results.keys()) ** 0.5),
    )
    plt.title("Drogue Parachute Fully Inflated Velocity")
    plt.xlabel("Velocity m/s)")
    plt.ylabel("Number of Occurrences")
    plt.show()

    return None
