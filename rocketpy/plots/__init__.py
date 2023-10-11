import numpy as np
from matplotlib.patches import Polygon


def _generate_nozzle(motor, translate=(0, 0), csys=1):
    nozzle_radius = motor.nozzle_radius
    nozzle_position = motor.nozzle_position
    try:
        throat_radius = motor.throat_radius
    except AttributeError:
        # Liquid motors don't have throat radius, let's estimate it
        throat_radius = 0.01

    # calculate length between throat and nozzle outlet using 15º angle
    major_axis = (nozzle_radius - throat_radius) / np.tan(np.deg2rad(15))
    # calculate minor axis considering a 45º angle
    minor_axis = (nozzle_radius - throat_radius) / np.tan(np.deg2rad(45))

    # calculate x and y coordinates of the nozzle
    x = csys * np.array(
        [0, 0, major_axis, major_axis + minor_axis, major_axis + minor_axis]
    )
    y = csys * np.array([0, nozzle_radius, throat_radius, nozzle_radius, 0])
    # we need to draw the other half of the nozzle
    x = np.concatenate([x, x[::-1]])
    y = np.concatenate([y, -y[::-1]])
    # now we need to sum the position and the translate
    x = x + nozzle_position + translate[0]
    y = y + translate[1]

    patch = Polygon(
        np.column_stack([x, y]),
        label="Nozzle",
        facecolor="black",
        edgecolor="black",
    )
    motor.nozzle_length = major_axis + minor_axis
    return patch
