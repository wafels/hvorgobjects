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
initial_time = Time('2000-05-10 00:00:00')
n_days = 7

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
    """
    return int(np.rint(t.unix * 1000))


def file_name_format(observer_name, body_name, t):
    """

    observer_name :
    body_name :
    t :

    Returns
    -------

    """
    return '{:s}_{:s}_{:n}.json'.format(observer_name, body_name, format_time_output(t))

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

            # Move the counter forward
            t += dt

        # Open the JSON file and dump out the positional information
        file_path = os.path.join(directory, file_name_format(observer_name, body_name, start_time))
        f = open(file_path, 'w')
        json.dump(positions, f)
        f.close()
