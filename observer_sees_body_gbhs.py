#
# Generates the positions of solar system objects as seen from the Earth
#
# Allows us to label objects in the LASCO C2, C3 field of view
# for example, 2000/05/15 11:18
# See http://sci.esa.int/soho/18976-lasco-c3-image-showing-the-positions-of-the-planets/
#
# TODO understand LASCO and helioviewer image processing steps
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

import heliopy.spice as spice
import heliopy.data.spice as spicedata


# Where to store the data
root = os.path.expanduser('~/hvp/hvorgobjects/output/json')
# root = os.path.expanduser('~/Desktop')

# Supported solar system objects
solar_system_objects = ('sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')

# Supported spacecraft
spice_spacecraft = ('psp', 'stereo_a', 'stereo_b', 'soho')


# Significant times.
# Calculate positions up to this time
calculation_end_time = Time('2025-12-31 23:59:59')

# The start times for the spacecraft should be determined from the time range
# over which the SPICE kernels are valid.  Until that is implemented, these
# times are sufficient to allow calculation to proceed
soho_start_time = Time('1996-02-01 00:00:00')
stereo_start_time = Time('2007-01-01 00:00:00')
psp_start_time = Time('2018-09-01 00:00:00')

# Contact with STEREO B was lost around this time.
stereo_b_end_time = Time('2014-10-01 23:59:59')


# 1a - Planets as seen from SOHO - done 2019/03/21
#observer_name = 'soho'
#body_names = ('mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
#search_time_range = [soho_start_time, calculation_end_time]
#search_time_range = [Time('2000-01-01 00:00:00'), Time('2000-12-31 23:59:59')]
#search_time_range = [Time('2016-01-01 00:00:00'), Time('2016-12-31 23:59:59')]

# 1b - PSP as seen from SOHO.
#observer_name = 'soho'
#body_names = ('psp',)
#search_time_range = [psp_start_time, calculation_end_time]

# 1c - STEREO A as seen from SOHO
#observer_name = 'soho'
#body_names = ('stereo_a',)
#search_time_range = [stereo_start_time, calculation_end_time]

# 1d - STEREO B as seen from SOHO
#observer_name = 'soho'
#body_names = ('stereo_b',)
#search_time_range = [stereo_start_time, calculation_end_time]

# 2a - Planets as seen from STEREO-A
#observer_name = 'stereo_a'
#body_names = ('venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
#search_time_range = [stereo_start_time, calculation_end_time]

# 2b - PSP as seen from STEREO-A
#observer_name = 'stereo_a'
#body_names = ('psp',)
#search_time_range = [psp_start_time, calculation_end_time]

# 2c - STEREO-B as seen from STEREO-A
#observer_name = 'stereo_a'
#body_names = ('stereo_b',)
#search_time_range = [stereo_start_time, stereo_b_end_time]

# 3a - Planets as seen from STEREO-B
#observer_name = 'stereo_b'
#body_names = ('mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
#search_time_range = [stereo_start_time, stereo_b_end_time]

# 3b - STEREO-A as seen from STEREO-B
#observer_name = 'stereo_b'
#body_names = ('stereo_a',)
#search_time_range = [stereo_start_time, stereo_b_end_time]

# Test 1: mercury as seen from STEREO A
#observer_name = 'stereo_a'
#body_names = ('mercury',)
#search_time_range = [Time('2012-01-01 00:00:00'), Time('2012-12-31 23:59:59')]


# Test 2: Planets as seen from SOHO for a time range when a lot of planets
#         are in the field of view.
observer_name = 'soho'
body_names = ('mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
search_time_range = [Time('2000-01-01 00:00:00'), Time('2000-12-31 23:59:59')]
#body_names = ('psp',)  # ('mercury', 'venus', 'jupiter', 'saturn', 'uranus', 'neptune')
#search_time_range = [Time('2018-09-01 00:00:00'), Time('2025-06-30 00:00:00')]


search_time_step = 1 * u.day

# Time step
transit_time_step = 30*u.minute

# Write a file only when the body has an angular separation from the Sun
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


class SpacecraftKernel:
    """

    """
    def __init__(self):
        self.loaded_kernels = []
        self.setup_has_been_run = False

    def load(self, body_name):
        if body_name not in self.loaded_kernels:
            if not self.setup_has_been_run:
                spice.setup_spice()
                self.setup_has_been_run = True

            if body_name == 'psp':
                kernels = spicedata.get_kernel('psp')
                kernels += spicedata.get_kernel('psp_pred')
                spice.furnish(kernels)
                self.loaded_kernels.append(body_name)

            if body_name == 'stereo_a':
                kernels = spicedata.get_kernel('stereo_a')
                kernels += spicedata.get_kernel('stereo_a_pred')
                spice.furnish(kernels)
                self.loaded_kernels.append(body_name)

            if body_name == 'stereo_b':
                kernels = spicedata.get_kernel('stereo_b')
                kernels += spicedata.get_kernel('stereo_b_pred')
                spice.furnish(kernels)
                self.loaded_kernels.append(body_name)

            if body_name == 'soho':
                kernels = spicedata.get_kernel('soho')
                spice.furnish(kernels)
                self.loaded_kernels.append(body_name)


# Load in the the spacecraft kernel loader and checker
spacecraft_kernels = SpacecraftKernel()


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


def speed_format(v, unit=u.km/u.s):
    """
    Returns the speeds in the requested format

    Parameters
    ----------
    v : `~astropy.unit`
        A speed in units convertible to kilometers per second.

    unit : `~astropy.unit`
        The scale the input unit is measured in.

    Returns
    -------
    speed_format : `~float`
        A floating point number that is implicitly measured in the input units.
    """
    if v is None:
        return None
    else:
        return v.to(unit).value


def get_spice_target(body_name):
    """
    Return the HelioPy target object

    Parameters
    ----------
    body_name : `~str`
        The name of the SPICE target object.

    Returns
    -------
    `~heliopy.space.Trajectory`

    """
    # Parker Solar Probe (also sometimes referred to as Solar Probe Plus or SPP)
    if body_name == 'psp':
        spacecraft_kernels.load('psp')
        target = spice.Trajectory('SPP')
    elif body_name == 'stereo_a':
        spacecraft_kernels.load('stereo_a')
        target = spice.Trajectory('STEREO AHEAD')
    elif body_name == 'stereo_b':
        spacecraft_kernels.load('stereo_b')
        target = spice.Trajectory('STEREO BEHIND')
    elif body_name == 'soho':
        spacecraft_kernels.load('soho')
        target = spice.Trajectory('SOHO')
    else:
        target = None

    return target


def get_position(body_name, time):
    """
    Get the position of one of the supported bodies.

    Parameters
    ----------
    body_name :

    time :

    Returns
    -------
    `~astropy.coordinate.SkyCoord`
    """
    _body_name = body_name.lower()

    # Check if the body is one of the supported spacecraft
    if _body_name in spice_spacecraft:
        spice_target = get_spice_target(_body_name)
        if _body_name == 'soho':
            # Use the SPICE kernels if available, otherwise estimate the
            # position of SOHO.
            try:
                coordinate = spice_target.coordinate(time)
            except:  # SpiceyError:
                earth = get_body('earth', time).transform_to(frames.Heliocentric)
                coordinate = SkyCoord(earth.x, earth.y, 0.99 * earth.z, frame=frames.Heliocentric, obstime=time, observer=earth)
        else:
            coordinate = spice_target.coordinate(time)
    # Check if the body is one of the supported solar system objects
    elif _body_name in solar_system_objects:
        coordinate = get_body(_body_name, observer=observer, time=time)
    else:
        raise ValueError('The body name is not recognized.')
    return coordinate


def get_position_heliographic_stonyhurst(body_name, observer, time):
    """
    Get the position of one of the supported bodies.

    Parameters
    ----------
    body_name :


    observer :


    time :

    Returns
    -------
    `~astropy.coordinate.SkyCoord`
    """
    _body_name = body_name.lower()

    # Check if the body is one of the supported spacecraft
    if _body_name in spice_spacecraft:
        raise ValueError('Light travel time corrected locations of spacecraft not yet supported.')
        """
        spice_target = get_spice_target(_body_name)
        if _body_name == 'soho':
            # Use the SPICE kernels if available, otherwise estimate the
            # position of SOHO.
            try:
                coordinate = spice_target.coordinate(time)
            except:  # SpiceyError:
                earth = get_body('earth', time).transform_to(frames.Heliocentric)
                coordinate = SkyCoord(earth.x, earth.y, 0.99 * earth.z, frame=frames.Heliocentric, obstime=time, observer=earth)
        else:
            coordinate = spice_target.coordinate(time)
        """
    # Check if the body is one of the supported solar system objects
    elif _body_name in solar_system_objects:
        coordinate = get_body_heliographic_stonyhurst(_body_name, observer=observer, time=time)
    else:
        raise ValueError('The body name is not recognized.')
    return coordinate


def get_speed(body_name, time):
    """
    Get the position of one of the supported bodies.

    :param body_name:
    :param time:
    :return:
    """
    _body_name = body_name.lower()

    # Check if the body is one of the supported spacecraft
    if _body_name in spice_spacecraft:
        spice_target = get_spice_target(_body_name)
        speed = spice_target.speed(time)
    else:
        speed = None
    return speed


class PlanetaryGeometry:
    def __init__(self, observer, body_name, t):
        """
        Calculate properties of the geometry of the observer, the body and the Sun.

        Parameters
        ----------
        observer :


        body_name :


        t :

        """
        # Position of the observer
        self.observer = observer
        self.observer_hpc = self.observer.transform_to(frames.Helioprojective)

        # The body that we are observing
        self.body_name = body_name

        # The time at which the positions of the observer, body and Sun are calculated.
        self.t = t

        # Position of the Sun as seen from the location of the observer in Helioprojective Cartesian
        self.sun = (get_position('sun', self.t)).transform_to(self.observer_hpc)

        # Position of the body as seen from the location of the observer in Helioprojective Cartesian
        self.body = (get_position_heliographic_stonyhurst(self.body_name, observer, self.t)).transform_to(self.observer_hpc)

    # Angular separation of the Sun and the body
    def separation(self):
        """
        Returns the angular separation of the Sun and the body as seen by the observer.
        """
        return (self.sun.separation(self.body)).to(u.deg)

    # Is the body close to the Sun in an angular sense.
    def is_close(self, angular_limit=10*u.deg, distance_limit=0.25*u.au):
        """
        Returns True if the body is close to the Sun.
        """
        if body_name == "psp":
            close = self.distance_sun_to_body() < distance_limit
        else:
            close = np.abs(self.separation().to(u.deg)) < angular_limit
        return close

    def _distance(self, a, b):
        p = a.transform_to(frames.Heliocentric)
        q = b.transform_to(frames.Heliocentric)
        distance = np.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)
        return distance

    def distance_observer_to_body(self):
        """
        Distance from the observer to the body in AU.
        """
        return (self.observer.separation_3d(self.body)).to(u.au)

    def distance_sun_to_body(self):
        """
        Distance from the Sun to the body in AU.
        """
        return (self.sun.separation_3d(self.body)).to(u.au)

    def distance_sun_to_observer(self):
        """
        Distance from the Sun to the observer in AU.
        """
        return (self._distance(self.sun, self.observer)).to(u.au)

    def light_travel_time(self):
        """
        The time in seconds it takes for light to travel from the body to
        the observer.
        """
        return (self.distance_observer_to_body().to(u.m) / c.to(u.m/u.s)).to(u.s)

    def behind_the_plane_of_the_sun(self):
        """
        Returns True if the body is behind the plane of the Sun.
        """
        return (self.body.transform_to(frames.Heliocentric(observer=self.observer))).z.value < 0


def find_transit_start_time(observer_name, body_name, test_start_time, search_limit=None):
    """
    Find the start time of the transit of the body as seen from the observer.

    Parameters
    ----------

    :param observer_name:
    :param body_name:
    :param test_start_time:
    :param search_limit:

    Returns
    -------
    """

    # Define the observer
    observer = get_position(observer_name, test_start_time)

    # Calculate the geometry of the observer and body at the test_start_time
    pg = PlanetaryGeometry(observer, body_name, test_start_time)

    # The test start time is already in transit.  Go backwards in time to
    # find the start.
    if pg.is_close():
        t = deepcopy(test_start_time)
        found_transit_start_time = False
        while not found_transit_start_time:
            t -= search_time_step
            observer = get_position(observer_name, t)
            pg = PlanetaryGeometry(observer, body_name, t)
            # When the body has stepped out of being in transit
            if not pg.is_close():
                # Take a step back to when the body is in transit
                t += search_time_step
                found_transit_start_time = True
    else:
        # The test start time is not in transit.  Go forward in time to
        # find the start.
        t = deepcopy(test_start_time)
        found_transit_start_time = False
        while not found_transit_start_time:
            t += search_time_step
            observer = get_position(observer_name, t)
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
    """
    Find the end of the transit of the body as seen from the observer.

    Parameters
    ----------
    :param observer_name:
    :param body_name:
    :param test_time:

    Returns
    -------
    """
    observer = get_position(observer_name, test_time)
    pg = PlanetaryGeometry(observer, body_name, test_time)
    if not pg.is_close():
        raise ValueError('The input time is not one for which the body is transiting.')
    else:
        t = deepcopy(test_time)
        found_transit_end_time = False
        while not found_transit_end_time:
            t += search_time_step
            observer = get_position(observer_name, t)
            pg = PlanetaryGeometry(observer, body_name, t)
            if not pg.is_close():
                found_transit_end_time = True
    return t


# Create the storage directories
sd = Directory(observer_name, body_names, root=root)


# Go through each of the bodies
for body_name in body_names:

    # Set up where we are going to store the transit filenames
    transit_filenames = dict()

    # Some information for the user
    print('{:s} - looking for transits in the time range {:s} to {:s} as seen from {:s}.'.format(body_name, str(search_time_range[0]), str(search_time_range[1]), observer_name))

    # Set the transit start to be just outside the search time range
    transit_start_time = search_time_range[0] - search_time_step

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
            transit_start_time = search_time_range[1] + search_time_step
        else:
            print('{:s} - transit start time = {:s}'.format(body_name, str(transit_start_time)))

        # Found a transit start time within the search time range
        if transit_start_time <= search_time_range[1]:
            transit_end_time = find_transit_end_time(observer_name, body_name, transit_start_time)
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
                observer = get_position(observer_name, transit_time)

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

                # Distance between the observer and the sun
                positions[observer_name][body_name][t_index]["distance_sun_to_observer_au"] = distance_format(pg.distance_sun_to_observer().to(u.au))

                # Distance between the body and the Sun.
                positions[observer_name][body_name][t_index]["distance_sun_to_body_au"] = distance_format(pg.distance_sun_to_body().to(u.au))

                # Is the body behind the plane of the Sun?
                positions[observer_name][body_name][t_index]["behind_plane_of_sun"] = str(pg.behind_the_plane_of_the_sun())

                # Store the positions of the body
                positions[observer_name][body_name][t_index]["x"] = pg.body.Tx.value
                positions[observer_name][body_name][t_index]["y"] = pg.body.Ty.value

                # Store the velocity of the body
                if body_name in spice_spacecraft:
                    positions[observer_name][body_name][t_index]["speedkms"] = speed_format(get_speed(body_name, transit_time).to(u.km/u.s))

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

