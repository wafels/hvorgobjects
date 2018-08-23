#
# Generates the positions of planets as seen from the Earth
#
# Allows us to label planets in the LASCO C2, C3 field of view
# for example, 2000/05/15 11:18
# See http://sci.esa.int/soho/18976-lasco-c3-image-showing-the-positions-of-the-planets/
# 
#
import json

import numpy as np

from astropy.coordinates import get_body
from astropy.time import Time
import astropy.units as u

from sunpy.coordinates import frames



# Time we are interested in
start_time = Time('2000-05-10 00:00:00')
end_time = Time('2000-05-11 00:00:00')

# Time step
dt = 30*u.minute

# Where are we looking from - the observer
observer_name = 'earth'

# Bodies
body_names = ('mercury', 'saturn', 'jupiter', 'venus')


#
def format_time_output(t):
    """
    Format the time output
    """
    return int(np.rint(t.unix * 1000))


for body_name in body_names:

    for nday in range(0, 1):
        day_start = start_time + nday*u.day
        start_time_unix = format_time_output(day_start)

        positions = dict()
        positions[body_name] = dict()
        positions[body_name][observer_name] = dict()

        t = day_start
        while t - day_start < 1 * u.day:
            # The location of the observer
            observer_location = get_body(observer_name, t)

            # The location of the body
            this_body = get_body(body_name, t)

            # The position of the body as seen from the observer location
            position = this_body.transform_to(observer_location).transform_to(frames.Helioprojective)

            # time index is unix time stamp in milliseconds - cast to ints
            t_index = format_time_output(t)
            positions[body_name][observer_name][t_index] = dict()

            # todo  -  Positions should be saved as float
            positions[body_name][observer_name][t_index]["x"] = position.Tx.value
            positions[body_name][observer_name][t_index]["y"] = position.Ty.value

            t = t + dt

        f = open('{:s}_{:s}_{:n}.json'.format(observer_name, body_name, start_time_unix), 'w')
        json.dump(positions, f)
        f.close()
