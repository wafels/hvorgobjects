#
# Generates the positions of planets as seen from the Earth
#
# Allows us to label planets in the LASCO C2, C3 field of view
# for example, 2000/05/15 11:18
# See http://sci.esa.int/soho/18976-lasco-c3-image-showing-the-positions-of-the-planets/
# 
#
import json
import os

import numpy as np

from astropy.coordinates import get_body
from astropy.time import Time
import astropy.units as u

from sunpy.coordinates import frames

#
directory = os.path.expanduser('~/hvp/hvorgobjects/output/json')

# Days we want to calculate positions for
# transit_of_venus_2012 = Time('2012-06-05 00:00:00')
# n_days = 2

multiple_planets = Time('2000-05-11 00:00:00')
n_days = 1

# Which times?
initial_time = multiple_planets

# Time step
dt = 30*u.minute

# duration
duration = 1 * u.day

# Where are we looking from - the observer
observer_name = 'earth'

# Bodies
body_names = ('mercury', 'saturn', 'jupiter', 'venus')


# Format the output time as requested.
def format_time_output(t):
    """
    Format the time output.

    Parameters
    ----------
    t : `~astropy.time.Time`
        The time that is to be formatted.

    Returns
    -------
    format_time_output
        The input time in the requested format.

    """
    # Returns as an integer number of seconds
    return int(np.rint(t.unix * 1000))


def format_time_for_filename(t):
    """
    Format the input time for the output filename


    Parameters
    ----------
    t : `~astropy.time.Time`
        The time that is to be formatted.

    Returns
    -------
    format_time_for_filename
        The input time in the requested format.

    """
    return None


def file_name_format(observer_name, body_name, t, file_type='json'):
    """
    The file name format that has been decided upon for the output.

    Parameters
    ----------
    observer_name : `~str`
        The name of the observer.

    body_name : `~str`
        The body that is observed.

    t : `~astropy.time.Time`
        A time that signifies the time range of information in the file.

    file_type : '~str`
        The file type that will be written.

    Returns
    -------
    file_name_format
        The filename in the requested format
    """
    return '{:s}_{:s}_{:n}.{:s}'.format(observer_name, body_name, format_time_for_filename(t), file_type)


def distance_format(d, scale=1*u.au):
    """
    Returns the distances in the requested format

    Parameters
    ----------
    d : `~astropy.unit`
        A distance in units convertible to kilometers.

    scale : `~astropy.unit`
        The scale the input unit is measured in.

    Returns
    -------
    distance_format : `~float`
        A floating point number that is implicitly measured in the input scale.
    """
    return (d / scale).decompose().value


# Go through each of the bodies
for body_name in body_names:

    # Pick the next start time
    for n in range(0, n_days):
        start_time = initial_time + n*u.day

        # Reset the positions directory for the new start time
        positions = dict()
        positions[observer_name] = dict()
        positions[observer_name][body_name] = dict()

        # Reset the counter to the new start time
        t = start_time

        # Calculate the positions in the duration
        while t - start_time < duration:
            # The location of the observer
            observer_location = get_body(observer_name, t)

            # The location of the body
            this_body = get_body(body_name, t)

            # The position of the body as seen from the observer location
            position = this_body.transform_to(observer_location).transform_to(frames.Helioprojective)

            # time index is unix time stamp in milliseconds - cast to ints
            t_index = format_time_output(t)
            positions[observer_name][body_name][t_index] = dict()

            # store the positions of the
            positions[observer_name][body_name][t_index]["x"] = position.Tx.value
            positions[observer_name][body_name][t_index]["y"] = position.Ty.value

            # location of the body in HCC
            body_hcc = this_body.transform_to(frames.Heliocentric)

            # location of the observer in HCC
            observer_hcc = observer_location.transform_to(frames.Heliocentric)

            # Distance from the observer to the body
            distance_observer_to_body = np.sqrt((body_hcc.x - observer_hcc.x)**2 + (body_hcc.y - observer_hcc.y)**2 + (body_hcc.z - observer_hcc.z)**2)
            positions[observer_name][body_name][t_index]["distance_observer_to_body_au"] = distance_format(distance_observer_to_body)

            # Distance of the body from the Sun
            distance_body_to_sun = np.sqrt(body_hcc.x**2 + body_hcc.y**2 + body_hcc.z**2)
            positions[observer_name][body_name][t_index]["distance_body_to_sun_au"] = distance_format(distance_body_to_sun)

            # Is the body behind the plane of the Sun?
            positions[observer_name][body_name][t_index]["behind_plane_of_sun"] = str(body_hcc.z.value < 0)

            # Move the counter forward
            t += dt

        # Open the JSON file and dump out the positional information
        file_path = os.path.join(directory, file_name_format(observer_name, body_name, start_time))
        f = open(file_path, 'w')
        json.dump(positions, f)
        f.close()
