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
from copy import deepcopy

import numpy as np

from astropy.coordinates import get_body
from astropy.time import Time
import astropy.units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord

from sunpy.coordinates import frames
from sunpy import coordinates

# Where to store the data
root = os.path.expanduser('~/hvp/hvorgobjects/output/json')

# Where are we looking from - the observer
observer_name = 'earth'

# Bodies
body_names = ('mercury', 'venus', 'jupiter', 'saturn', 'uranus', 'neptune')

# Look for transits that start in this time range.
search_time_range = [Time('2000-01-01 00:00:00'), Time('2001-01-01 00:00:00')]
time_step = 1 * u.day

# Time step
transit_time_step = 30*u.minute

# Write a file only when the bisy has an angular separation from the Sun
# less than the maximum below
maximum_angular_separation = 10 * u.deg


# Create the storage directories
class Directory:
    def __init__(self, observer_name, body_names, root="~"):
        self.observer_name = observer_name.lower()
        self.body_names = [body_name.lower() for body_name in body_names]
        self.root = root
        self.directories = dict()
        self.directories[observer_name] = dict()

        self.observer_path = os.path.join(os.path.expanduser(self.root), self.observer_name)
        if not os.path.isdir(self.observer_path):
            os.makedirs(self.observer_path, exist_ok=True)

        for body_name in self.body_names:
            path = os.path.join(self.observer_path, body_name)
            os.makedirs(path, exist_ok=True)
            self.directories[self.observer_name][body_name] = path

    def get(self, this_observer_name, this_body_name):
        return self.directories[this_observer_name.lower()][this_body_name.lower()]


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
    # Returns as an integer number of milliseconds
    return int(np.rint(t.unix * 1000))


def body_coordinate_file_name_format(observer_name, body_name, t0, t1, file_type='json'):
    """
    The file name format that has been decided upon for the body coordinate
    output.

    Parameters
    ----------
    observer_name : `~str`
        The name of the observer.

    body_name : `~str`
        The body that is observed.

    t0 : `~astropy.time.Time`
        A time that signifies the start time of the transit.

    t1 : `~astropy.time.Time`
        A time that signifies the end time of the transit.

    file_type : '~str`
        The file type that will be written.

    Returns
    -------
    file_name_format
        The filename string in the requested format.
    """
    tc0 = deepcopy(t0)
    tc0.out_subfmt = 'date'

    tc1 = deepcopy(t1)
    tc1.out_subfmt = 'date'
    return '{:s}_{:s}_{:s}_{:s}.{:s}'.format(observer_name, body_name, str(tc0), str(tc1), file_type)


def transit_meta_data_file_name_format(observer_name, body_name, file_type='json'):
    return '{:s}_{:s}_dictionary.{:s}'.format(observer_name, body_name, file_type)


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


class PlanetaryGeometry:
    def __init__(self, observer, body_name, t):
        """Calculate properties of the geometry of the observer, the body and
        the Sun."""
        self.observer = observer
        self.body_name = body_name
        self.t = t

        # Position of the Sun
        self.sun = get_body('sun', self.t)

        # Position of the body
        self.body = get_body(body_name, self.t)

        # The location of the observer in HPC co-ordinates.
        self.observer_hpc = coordinates.Helioprojective(observer=observer, obstime=observer.obstime)

        # The location of the body as seen by the observer, following the
        # SunPy example
        self.body_hpc = self.body.transform_to(frames.Helioprojective).transform_to(self.observer_hpc)

        # The body and the observer in HCC for ease of distance calculation.
        self._body_hcc = self.body.transform_to(frames.Heliocentric)
        self._observer_hcc = self.observer.transform_to(frames.Heliocentric)

    # Angular separation of the Sun and the body
    def separation(self):
        """Returns the angular separation of the Sun and the body."""
        return self.sun.separation(self.body)

    # Is the body close to the Sun in an angular sense.
    def is_close(self, angular_limit=10*u.deg):
        """Returns True if the body is close to the Sun in an angular sense,
        when compared to the angular limit."""
        return np.abs(self.separation()) < angular_limit

    def distance_observer_to_body(self):
        """Distance from the observer to the body in AU."""
        return np.sqrt((self._observer_hcc.x - self._body_hcc.x)**2 + (self._observer_hcc.y - self._body_hcc.y) ** 2 + (self._observer_hcc.z - self._body_hcc.z)**2).to(u.au)

    def distance_sun_to_body(self):
        """Distance from the Sun to the body in AU."""
        return np.sqrt(self._body_hcc.x ** 2 + self._body_hcc.y ** 2 + self._body_hcc.z ** 2).to(u.au)

    def light_travel_time(self):
        """The time in seconds it takes for light to travel from the body to
        the observer."""
        return (self.distance_observer_to_body().to(u.m) / c.to(u.m/u.s)).to(u.s)

    def behind_the_plane_of_the_sun(self):
        """Returns True if the body is behind the plane of the Sun."""
        return self._body_hcc.z.value < 0


def find_transit_start_time(observer_name, body_name, test_start_time, search_limit=None):
    """Find the start time of the transit of the body as seen from the
    observer."""

    # Define the observer
    observer = get_observer(observer_name, test_start_time)

    # Calculate the geometry of the observer and body at the test_start_time
    pg = PlanetaryGeometry(observer, body_name, test_start_time)

    # The test start time is already in transit.  Go backwards in time to
    # find the start.
    if pg.is_close():
        t = deepcopy(test_start_time)
        found_transit_start_time = False
        while not found_transit_start_time:
            t -= time_step
            observer = get_observer(observer_name, t)
            pg = PlanetaryGeometry(observer, body_name, t)
            if not pg.is_close():
                found_transit_start_time = True
    else:
        # The test start time is not in transit.  Go forward in time to
        # find the start.
        t = deepcopy(test_start_time)
        found_transit_start_time = False
        while not found_transit_start_time:
            t += time_step
            observer = get_observer(observer_name, t)
            pg = PlanetaryGeometry(observer, body_name, t)
            if pg.is_close():
                found_transit_start_time = True
    if search_limit is None:
        return t
    elif t <= search_limit:
        return t
    else:
        return None


def find_transit_end_time(observer_name, body_name, test_time):
    """Find the end of the transit of the body as seen from the observer."""
    observer = get_observer(observer_name, test_time)
    pg = PlanetaryGeometry(observer, body_name, test_time)
    if not pg.is_close():
        raise ValueError('The input time is not one for which the body is transiting.')
    else:
        t = deepcopy(test_time)
        found_transit_end_time = False
        while not found_transit_end_time:
            t += time_step
            observer = get_observer(observer_name, t)
            pg = PlanetaryGeometry(observer, body_name, t)
            if not pg.is_close():
                found_transit_end_time = True
    return t


def get_observer(observer_name, t):
    """ Get the location of the observer given the name of the observer."""
    # TODO understand LASCO and helioviewer image processing steps
    if observer_name.lower() == 'soho':
        earth = get_body('earth', t).transform_to(frames.Heliocentric)
        return SkyCoord(earth.x, earth.y, 0.99*earth.z, frame=frames.Heliocentric, obstime=t)
    else:
        return get_body(observer_name, t)


# Create the storage directories
sd = Directory(observer_name, body_names)


# Go through each of the bodies
for body_name in body_names:

    # Set up where we are going to store the transit filenames
    transit_filenames = dict()

    # Some information for the user
    print('{:s} - looking for transits in the time range {:s} to {:s}.'.format(body_name, str(search_time_range[0]), str(search_time_range[1])))

    # Set the transit start to be just outside the search time range
    transit_start_time = search_time_range[0] - time_step

    # Search for transits in the time range
    while transit_start_time <= search_time_range[1]:

        # Update the transit start time
        search_limit = search_time_range[1]
        test_transit_start_time = deepcopy(transit_start_time)
        transit_start_time = find_transit_start_time(observer_name, body_name, test_transit_start_time, search_limit=search_limit)
        if transit_start_time is None:
            print('{:s} - no transit start time after {:s} and before the end of search time range {:s}.'.format(body_name, str(test_transit_start_time), str(search_limit)))
            # This will cause an exit from the while loop and start a transit
            # search for the next body.
            transit_start_time = search_time_range[1] + time_step
        else:
            print('{:s} - transit start time = {:s}'.format(body_name, str(transit_start_time)))

        # Found a transit start time within the search time range
        if transit_start_time <= search_time_range[1]:
            transit_end_time = find_transit_end_time(observer_name, body_name, transit_start_time + time_step)
            print('{:s} - transit end time = {:s}'.format(body_name, str(transit_end_time)))
            print('{:s} - calculating transit between {:s} and {:s}.'.format(body_name, str(transit_start_time), str(transit_end_time)))

            # Storage for position of the body as seen by the observer
            positions = dict()
            positions[observer_name] = dict()
            positions[observer_name][body_name] = dict()

            # Start at the transit start time
            transit_time = deepcopy(transit_start_time)

            # Set the flag to record the first time that light from the body
            # reaches the observer
            record_first_time_that_photons_reach_observer = True

            # Go through the entire transit
            while transit_time <= transit_end_time:

                # Get the location of the observer
                observer = get_observer(observer_name, transit_time)

                # Calculate the geometry
                pg = PlanetaryGeometry(observer, body_name, transit_time)

                # Add in the light travel time
                time_that_photons_reach_observer = transit_time + pg.light_travel_time()

                if record_first_time_that_photons_reach_observer:
                    first_time_that_photons_reach_observer = deepcopy(time_that_photons_reach_observer)
                    record_first_time_that_photons_reach_observer = False

                # Convert the time to that used for output.
                t_index = format_time_output(time_that_photons_reach_observer)

                # Create the data to be saved
                positions[observer_name][body_name][t_index] = dict()

                # Distance between the observer and the body
                positions[observer_name][body_name][t_index]["distance_observer_to_body_au"] = distance_format(pg.distance_observer_to_body().to(u.au))

                # Distance between the body and the Sun.
                positions[observer_name][body_name][t_index]["distance_body_to_sun_au"] = distance_format(pg.distance_sun_to_body().to(u.au))

                # Is the body behind the plane of the Sun?
                positions[observer_name][body_name][t_index]["behind_plane_of_sun"] = str(pg.behind_the_plane_of_the_sun())

                # Store the positions of the body
                positions[observer_name][body_name][t_index]["x"] = pg.body_hpc.Tx.value
                positions[observer_name][body_name][t_index]["y"] = pg.body_hpc.Ty.value

                # Advance the time during the transit
                transit_time += transit_time_step

            # The last time that photons reach the observatory for this
            # transit.
            last_time_that_photons_reach_observer = deepcopy(time_that_photons_reach_observer)

            # Save the transit data
            filename_transit_start_time = first_time_that_photons_reach_observer
            filename_transit_end_time = last_time_that_photons_reach_observer
            filename = body_coordinate_file_name_format(observer_name, body_name, filename_transit_start_time, filename_transit_end_time)
            storage_directory = sd.get(observer_name, body_name)
            file_path = os.path.join(storage_directory, filename)
            f = open(file_path, 'w')
            json.dump(positions, f)
            f.close()

            # Store the filename for this transit
            transit_filenames[filename] = dict()
            transit_filenames[filename]['start'] = format_time_output(filename_transit_start_time)
            transit_filenames[filename]['end'] = format_time_output(filename_transit_end_time)

            # Update the initial search time
            transit_start_time = deepcopy(transit_end_time) + transit_time_step

    # Save the filenames for the transits for this observer and body
    filename = transit_meta_data_file_name_format(observer_name, body_name)
    storage_directory = sd.get(observer_name, body_name)
    file_path = os.path.join(storage_directory, filename)
    f = open(file_path, 'w')
    json.dump(transit_filenames, f)
    f.close()

