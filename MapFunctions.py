# Author: Jack Adams
# Date Started: 18/06/3
# Last Updated: 18/06/3

# This file contains the functions which will generate intensity values for
# different types of Magic depending on the terrain of the Map.

import numpy as np
from scipy.stats import skewnorm
import h5py as h5


class RegionMap:
    """
    This class houses the arrays which contain information about the terrain
    across the map, as well as the arrays holding the intensity of the
    different types of Magic across that Map. It has methods which will find
    the next state of the Magic across the Map too.
    """

    def __init__(self):
        self.terrain = None
        self.magics = None

    def save_map(self, filename, centre):
        """"""

        # First prepare the filename so it can be used as an h5py file and then
        # create the file handle.
        filename = filename + '.h5'
        h5handle = h5.File(filename, 'w')

        # Now store the magic and centre data.
        h5handle.create_dataset('magic_arrays', data=self.magics)
        h5handle.create_dataset('centre_location', data=centre)

        # Lastly close the file handle.
        h5handle.close()

    def load_map(self, filename):
        """"""

        # First prepare the filename for reading and then open the file handle.
        filename = filename + '.h5'
        h5handle = h5.File(filename, 'r')

        # Now extract the data into the map.
        self.magics = h5handle['magic_arrays'][:]
        centre = h5handle['centre_location'][:]

        # Lastly close the file handle.
        h5handle.close()

        # Extract addtional information about the height and width of the Map,
        # as well as the current time step.
        [a, b, c, d] = self.magics.shape
        time = a
        height = c
        width = d

        return height, width, time

    def initialise_map(self, height, width):
        """
        Creates the arrays within the Map which are needed to store the values
        for the Magic intensities.

        :param height: The number of points in the Map from north to south.
        :param width: The number of points in the Map from east to west.
        """

        self.terrain = np.zeros([1, height, width])
        self.magics = np.zeros([1, 12, height, width])

    def print_region(self, time, ystart, ystop, xstart, xstop):
        """"""

    def create_next_time(self, height, width):
        """
        Extend the Magic arrays so that they can hold information about Magic
        in the next time step.

        :param height: The number of points in the Map from north to south.
        :param width: The number of points in the Map from east to west.
        """

        self.magics = np.append(self.magics, np.zeros([1, 12, height, width]),
                                axis=0)

    def find_magic(self, start, stop, height, width, centre):
        """
        This is the master-method for finding how the Magic changes over time.
        It will be run for a number of time steps between the inputs start and
        stop. In each, it will first set the BCs up, then step through and find
        the Magic values according to a variety of methods which differ
        between types of Magic.

        :param start: The starting time step.
        :param stop: The final time step.
        :param height: The number of points in the Map from north to south.
        :param width: The number of points in the Map from east to west.
        """

        for time in range(start, stop):

            print('Time = {}'.format(time))
            # To start with, place the boundary conditions for all the
            # types of Magic which require it.
            self.initialise_BCs(height, width, time)

            # For the arrays of Magic, each call their respective generation
            # functions.
            for i in range(height):
                for j in range(width):
                    # First deal with the Light, Dark and Shadow Magics which
                    # come from the epicentre specified in the following calls.
                    self.gen_light_value(width, centre, time, i, j)
                    self.gen_dark_value(width, centre, time, i, j)
                    self.calculate_shadow(time, i, j)

                    # After those, find if there are any points which massive
                    # bursts of Waxing Magic.
                    self.gen_waxing_burst(time, i, j)

                    # Next, find the heat and cold Magics which emanate from
                    # the north and south respectively.
                    if i is not 0:
                        self.gen_heat_and_fire(height, width, time, i, j)

                    if i != height-1:
                        self.gen_cold_and_ice(height, width, time, i, j)

                    # Then find the Serc and Romond Magics which have complex
                    # functions to do with their terrain/location.
                    if j is not 0:
                        self.gen_wind_and_water(height, width, time, i, j)

                    # And lastly find the other types of Magic; Dren and Vaelf.
                    if j is not 0:
                        self.gen_remainder(height, width, time, i, j)

                    for k in range(12):
                        if self.magics[time, k, i, j] < 0:
                            self.magics[time, k, i, j] = 0
                        elif self.magics[time, k, i, j] > 4:
                            self.magics[time, k, i, j] = 4

            # Lastly create the next time step.
            self.create_next_time(height, width)

    def initialise_BCs(self, height, width, time):
        """
        Put in place values for the boundary conditions of Magical values
        across the borders of the Map.

        :param height: The number of points in the Map from north to south.
        :param width: The number of points in the Map from east to west.
        :param time: The value of time since the map started its weather
                     tracking.
        """

        # Start with the Heat and Cold BCs along the top and bottom edges since
        # the Light, Dark, and Shadow Magics don't need BCs.
        for i in range(width):
            self.magics[time, 4, 0, i] = np.round(skewnorm.rvs(1, loc=3,
                                                               scale=0.5))
            self.magics[time, 5, height-1, i] = np.round(skewnorm.rvs(1, loc=3,
                                                                      scale=0.5))

            # Now do the Talon and Izeth BCs, which are along the top and
            # bottom boundaries as well.
            self.magics[time, 6, 0, i] = np.round(skewnorm.rvs(1, loc=3,
                                                               scale=0.5))
            self.magics[time, 7, height-1] = np.round(skewnorm.rvs(1, loc=3,
                                                                   scale=0.5))

        # Now do the BCs for Dren, Romond, Serc, and Vaelf, all of which flow
        # from the left boundary.
        for i in range(height):
            self.magics[time, 8, i, 0] = np.round(skewnorm.rvs(1, loc=0.6,
                                                               scale=0.3))
            self.magics[time, 9, i, 0] = np.round(skewnorm.rvs(1, loc=3,
                                                               scale=0.5))
            self.magics[time, 10, i, 0] = np.round(skewnorm.rvs(1, loc=0.6,
                                                                scale=0.3))
            self.magics[time, 11, i, 0] = np.round(skewnorm.rvs(1, loc=3,
                                                                scale=0.5))

    def gen_light_value(self, width, centre, time, y, x):
        """
        Generates a value for the intensity of Light Magic depending on how far
        away it is from the epicentre of Light.

        :param width: The number of points in the x-dimension of the Map.
        :param centre: The y-x coordinates of the Light's epicentre.
        :param time: The value of time since the Map started its weather
                     tracking.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        # Assuming that the epicentre of light is located at roughly (25, 10),
        # find the number of points away the point of interest is, then shift
        # the master time by that many hours.
        distance = np.round(np.sqrt((centre[0] - y) ** 2
                                    + (centre[1] - x) ** 2))
        phase = 15 * np.pi / 180
        master_time = time % 24
        local_time = (master_time + distance) % 24

        if local_time < 12:
            skew_value = (6 - local_time) / 4
            scale_value = 0.2 + abs((6 - local_time) / 30)
        else:
            skew_value = (local_time - 18) / 4
            scale_value = 0.2 + abs((local_time - 18) / 30)

        # At Light epicentre, the exponential function should be roughly 1, so
        # the behaviour is determined by the scaled cos function. As the
        # distance increases, however, the exponential will cause the decay of
        # Light at a rate suited so at a distance of 60, Light is now a third
        # of its original intensity.

        value = (np.exp((-0.6931 / width) * distance) * (1.8 -
                 1.6 * np.cos(phase * local_time)) +
                 skewnorm.rvs(skew_value, loc=0, scale=scale_value))

        self.magics[time, 0, y, x] = value.round()

    def gen_dark_value(self, width, centre, time, y, x):
        """
        Generates a value for the intensity of Dark Magic depending on how far
        away it is from the epicentre of Light.

        :param width: The number of points in the x-dimension of the Map.
        :param centre: The y-x coordinates of the Light's epicentre.
        :param time: The value of time since the Map started its weather
                     tracking.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        # Assuming that the epicentre of light is located at roughly (25, 10),
        # find the number of points away the point of interest is, then shift
        # the master time by that many hours.
        distance = np.sqrt((centre[0] - y) ** 2 + (centre[1] - x) ** 2)
        phase = 15 * np.pi / 180
        master_time = time % 24
        local_time = (master_time + distance) % 24

        if local_time < 12:
            skew_value = -(6 - local_time) / 4
            scale_value = 0.2 + abs((6 - local_time) / 30)
        else:
            skew_value = -(local_time - 18) / 4
            scale_value = 0.2 + abs((local_time - 18) / 30)

        # At Light epicentre, the exponential function should be roughly 1, so
        # the behaviour is determined by the scaled cos function. As the
        # distance increases, however, the exponential will cause the decay of
        # Light at a rate suited so at a distance of 60, Light is now a half
        # of its original intensity.

        value = (np.exp((-0.6931 / width) * distance) * (1.8 -
                 np.cos(phase * local_time - np.pi)) +
                 skewnorm.rvs(skew_value, loc=0, scale=scale_value))

        self.magics[time, 1, y, x] = value.round()

    def calculate_shadow(self, time, y, x):
        """
        Using the values for the amount of Light and Dark Magic across the
        entire Map, this method will find the amount of Shadow Magic present
        and adjust the Light and Dark Magics accordingly.

        :param time: The value of time since the Map started its weather
                     tracking.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        if self.magics[time, 0, y, x] < self.magics[time, 1, y, x]:
            self.magics[time, 2, y, x] = self.magics[time, 0, y, x]
        else:
            self.magics[time, 2, y, x] = self.magics[time, 1, y, x]
        self.magics[time, 0, y, x] = (self.magics[time, 0, y, x] -
                                      self.magics[time, 2, y, x])
        self.magics[time, 1, y, x] = (self.magics[time, 1, y, x] -
                                      self.magics[time, 2, y, x])

    def gen_waxing_burst(self, time, y, x):
        """
        Generates points where Waxing Magic is extremely strong.

        :param time: The value of time since the Map started its weather
                     tracking.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        self.magics[time, 3, y, x] = np.round(skewnorm.rvs(4, loc=0, scale=1.4))

        if self.magics[time, 3, y, x] < 4:
            self.magics[time, 3, y, x] = 0

    def gen_heat_and_fire(self, height, width, time, y, x):
        """
        Determines the value of both Heat and Fire Magic at a point given the
        values of the points nearby in the previous time step.

        :param height: The number of points in the y-dimension of the Map.
        :param width: The number of points in the x-dimension of the Map.
        :param time: The value of time since the Map started its weather
                     tracking.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        if time == 0:
            # Want higher end to be at 3.3;     exp(1.1939) = 3.3
            # Want lowest end to be at 0.7;     exp(-0.3567) = 0.7
            # Therefore use (1.1939 + 0.3567) / width
            decay_loc = np.exp(1.1939 - ((1.5506 * y) / width))

            # Want higher end to be at 0.5;     exp(-0.6931) = 0.5
            # Want lowest end to be at 0.25;    exp(-1.3863) = 0.25
            # Therefore use (-1.3863 + 0.6931) / height
            decay_scale = np.exp(-0.6931 - ((0.6932 * y) / width))

            self.magics[time, 4, y, x] = np.round(skewnorm.rvs(0, loc=decay_loc,
                                                               scale=decay_scale))
            self.magics[time, 6, y, x] = np.round(skewnorm.rvs(0, loc=decay_loc,
                                                               scale=decay_scale))
        else:
            average1 = self.find_TB_average(width, time-1, 4, y, x)
            average2 = self.find_TB_average(width, time-1, 6, y, x)

            decay_loc = np.exp(1.1939 - ((1.5506 * y) / width))
            skew_value1 = 3 * (average1 - decay_loc)
            skew_value2 = 3 * (average2 - decay_loc)
            decay_scale = np.exp(-0.6931 - ((0.6932 * y) / width))

            seasonal_shift = 0.7 + 0.3 * np.cos(8.7266*(10**-4) * (time%7200))

            self.magics[time, 4, y, x] = np.round(skewnorm.rvs(skew_value1,
                                                               loc=decay_loc * seasonal_shift,
                                                               scale=decay_scale))
            self.magics[time, 6, y, x] = np.round(skewnorm.rvs(skew_value2,
                                                               loc=decay_loc,
                                                               scale=decay_scale))

    def gen_cold_and_ice(self, height, width, time, y, x):
        """
        Determines the value of both Cold and Ice Magic at a point given the
        values of the points nearby in the previous time step.

        :param height: The number of points in the y-dimension of the Map.
        :param width: The number of points in the x-dimension of the Map.
        :param time: The value of time since the Map started its weather
                     tracking.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        if time == 0:
            # Want higher end to be at 3.3;     exp(1.1939) = 3.3
            # Want lowest end to be at 0.7;     exp(-0.3567) = 0.7
            # Therefore use (1.1939 + 0.3567) / width
            growth_loc = np.exp(-0.3567 + ((1.5506 * y) / height))

            # Want higher end to be at 0.5;     exp(-0.6931) = 0.5
            # Want lowest end to be at 0.25;    exp(-1.3863) = 0.25
            # Therefore use (-1.3863 + 0.6931) / height
            growth_scale = np.exp(-1.3863 + ((0.6932 * y) / height))

            self.magics[time, 5, y, x] = np.round(skewnorm.rvs(0, loc=growth_loc,
                                                               scale=growth_scale))
            self.magics[time, 7, y, x] = np.round(skewnorm.rvs(0, loc=growth_loc,
                                                               scale=growth_scale))
        else:
            average1 = self.find_BT_average(width, time-1, 4, y, x)
            average2 = self.find_BT_average(width, time-1, 6, y, x)

            growth_loc = np.exp(-0.3567 + ((1.5506 * y) / height))
            skew_value1 = 3 * (average1 - growth_loc)
            skew_value2 = 3 * (average2 - growth_loc)
            growth_scale = np.exp(-1.3863 + ((0.6932 * y) / height))

            seasonal_shift = 0.7 + 0.3 * np.cos(np.pi + 8.7266 * (10**-4)
                                                * (time % 7200))

            self.magics[time, 5, y, x] = np.round(skewnorm.rvs(skew_value1,
                                                               loc=growth_loc * seasonal_shift,
                                                               scale=growth_scale))
            self.magics[time, 7, y, x] = np.round(skewnorm.rvs(skew_value2,
                                                               loc=growth_loc,
                                                               scale=growth_scale))

    def gen_wind_and_water(self, height, width, time, y, x):
        """
        Determines the values for the Magic of Serc and Romond given the values
        in the previous time step.

        :param height: The number of points in the y-dimension of the Map.
        :param width: The number of points in the x-dimension of the Map.
        :param time: The current time.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        if time == 0:
            # Want higher end to be at 3.3;     exp(1.1939) = 3.3
            # Want lowest end to be at 0.7;     exp(-0.3567) = 0.7
            # Therefore use (1.1939 + 0.3567) / width
            decay_loc = np.exp(1.1939 - ((1.5506 * x) / width))
            growth_loc = np.exp(-0.3567 + ((1.5506 * x) / width))

            # Want higher end to be at 0.5;     exp(-0.6931) = 0.5
            # Want lowest end to be at 0.25;    exp(-1.3863) = 0.25
            # Therefore use (-1.3863 + 0.6931) / height
            decay_scale = np.exp(-0.6931 - ((0.6932 * x) / width))
            growth_scale = np.exp(-1.3863 + ((0.6932 * x) / width))

            self.magics[time, 9, y, x] = np.round(skewnorm.rvs(0,
                                                               loc=decay_loc,
                                                               scale=decay_scale))
            self.magics[time, 10, y, x] = np.round(skewnorm.rvs(0,
                                                                loc=growth_loc,
                                                                scale=growth_scale))
        else:
            log_avg = self.find_LR_average(height, time-1, 9, y, x)
            exp_avg = self.find_LR_average(height, time-1, 10, y, x)

            decay_loc = np.exp(1.1939 - ((1.5506 * x) / width))
            growth_loc = np.exp(-0.3567 + ((1.5506 * x) / width))
            decay_scale = np.exp(-0.6931 - ((0.6932 * x) / width))
            growth_scale = np.exp(-1.3863 + ((0.6932 * x) / width))
            decay_skew = 3 * (log_avg - decay_loc)
            growth_skew = 3 * (exp_avg - growth_loc)

            self.magics[time, 9, y, x] = np.round(skewnorm.rvs(decay_skew,
                                                               loc=decay_loc,
                                                               scale=decay_scale))
            self.magics[time, 10, y, x] = np.round(skewnorm.rvs(growth_skew,
                                                                loc=growth_loc,
                                                                scale=growth_scale))

    def gen_remainder(self, height, width, time, y, x):
        """
        Finds the values for the Dren and Vaelf Magics depending on the Magic
        values from the previous time step.

        :param height: The number of points in the y-dimension of the Map.
        :param width: The number of points in the x-dimension of the Map.
        :param time: The current time.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        """

        if time == 0:
            # Want higher end to be at 3.3;     exp(1.1939) = 3.3
            # Want lowest end to be at 0.7;     exp(-0.3567) = 0.7
            # Therefore use (1.1939 + 0.3567) / width
            decay_loc = np.exp(1.1939 - ((1.5506 * x) / width))
            growth_loc = np.exp(-0.3567 + ((1.5506 * x) / width))

            # Want higher end to be at 0.5;     exp(-0.6931) = 0.5
            # Want lowest end to be at 0.25;    exp(-1.3863) = 0.25
            # Therefore use (-1.3863 + 0.6931) / height
            decay_scale = np.exp(-0.6931 - ((0.6932 * x) / width))
            growth_scale = np.exp(-1.3863 + ((0.6932 * x) / width))

            self.magics[time, 8, y, x] = np.round(skewnorm.rvs(0,
                                                               loc=growth_loc,
                                                               scale=growth_scale))
            self.magics[time, 11, y, x] = np.round(skewnorm.rvs(0,
                                                                loc=decay_loc,
                                                                scale=decay_scale))
        else:
            log_avg = self.find_LR_average(height, time-1, 8, y, x)
            exp_avg = self.find_LR_average(height, time-1, 11, y, x)

            decay_loc = np.exp(1.1939 - ((1.5506 * x) / width))
            growth_loc = np.exp(-0.3567 + ((1.5506 * x) / width))
            decay_scale = np.exp(-0.6931 - ((0.6932 * x) / width))
            growth_scale = np.exp(-1.3863 + ((0.6932 * x) / width))
            decay_skew = 3 * (log_avg - decay_loc)
            growth_skew = 3 * (exp_avg - growth_loc)

            self.magics[time, 8, y, x] = np.round(skewnorm.rvs(growth_skew,
                                                               loc=growth_loc,
                                                               scale=growth_scale))
            self.magics[time, 11, y, x] = np.round(skewnorm.rvs(decay_skew,
                                                                loc=decay_loc,
                                                                scale=decay_scale))

    def find_LR_average(self, height, time, magic, y, x):
        """
        Finds the average of the three values just east (one due east, the
        others north- and south-east respectively) of the point of interest.
        Edge cases are treated by just considering the due east point and one
        other which is available.

        :param height: The number of points in the y-dimension of the Map.
        :param time: The current time.
        :param magic: The array of Magic being inspected.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        :return: The average value of the inspected points.
        """

        if y == 0:
            average = (self.magics[time, magic, y, x-1] +
                       self.magics[time, magic, y+1, x-1]) / 2
        elif y == height-1:
            average = (self.magics[time, magic, y-1, x-1] +
                       self.magics[time, magic, y, x-1]) / 2
        else:
            average = (self.magics[time, magic, y-1, x-1] +
                       self.magics[time, magic, y, x-1] +
                       self.magics[time, magic, y+1, x-1]) / 3

        return average

    def find_TB_average(self, width, time, magic, y, x):
        """
        Finds the average of the three values just north (one due north, the
        others north-east and north-est respectively) of the point of interest.
        Edge cases are treated by just considering the due north point and one
        other which is available.

        :param width: The number of points in the y-dimension of the Map.
        :param time: The current time.
        :param magic: The array of Magic being inspected.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        :return: The average value of the inspected points.
        """

        if x == 0:
            average = (self.magics[time, magic, y-1, x] +
                       self.magics[time, magic, y-1, x+1]) / 2
        elif x == width-1:
            average = (self.magics[time, magic, y-1, x-1] +
                       self.magics[time, magic, y-1, x]) / 2
        else:
            average = (self.magics[time, magic, y-1, x-1] +
                       self.magics[time, magic, y-1, x] +
                       self.magics[time, magic, y-1, x+1]) / 3

        return average

    def find_BT_average(self, width, time, magic, y, x):
        """
        Finds the average of the three values just south (one due south, the
        others south-east and south-west respectively) of the point of
        interest. Edge cases are treated by just considering the due south
        point and one other which is available.

        :param width: The number of points in the y-dimension of the Map.
        :param time: The current time.
        :param magic: The array of Magic being inspected.
        :param y: The y-location of the given point.
        :param x: The x-location of the given point.
        :return: The average value of the inspected points.
        """

        if x == 0:
            average = (self.magics[time, magic, y+1, x] +
                       self.magics[time, magic, y+1, x+1]) / 2
        elif x == width-1:
            average = (self.magics[time, magic, y+1, x-1] +
                       self.magics[time, magic, y+1, x]) / 2
        else:
            average = (self.magics[time, magic, y+1, x-1] +
                       self.magics[time, magic, y+1, x] +
                       self.magics[time, magic, y+1, x+1]) / 3

        return average
