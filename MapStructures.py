# Author: Jack Adams
# Date Started: 18/05/17
# Last Updated: 18/05/19

# This file contains the definitions of the map and point structures. It also
# contains all of the methods which act on those structures.

import numpy as np
import scipy as sp


class Map:
    """
    This class will be the map which holds an array the size of the map for
    each type of Magic required and also an array for the diffusion constants
    associated spacially with each type of Magic.
    """

    Light = None
    Dark = None
    DifLight = None
    DifDark = None

    def prepare_map_arrays(self, map_width):
        """
        This method will generate the set of points corresponding to locations
        on the map. The map will have a buffer of three additional points
        around each side to allow the boundary conditions to be initialised
        outside the region of interest.

        :param map: The map to be initialised
        :param map_width: An integer corresponding to the desired number of
                          points across the map.
        """

        empty_array = np.zeros([map_width + 6, map_width + 6])

        self.Light = empty_array
        self.Dark = empty_array
        self.DifLight = empty_array
        self.DifDark = empty_array

    def create_next_time_step(self):
        """ Increases the third dimension of each Magic array by one. """

        sizes = self.Light.shape

        if len(sizes) == 2:
            empty_array = np.zeros([1, sizes[0], sizes[1]])
            self.Light = np.array([self.Light])
            self.Light = np.append(self.Light, empty_array, axis=0)
            self.Dark = np.array([self.Dark])
            self.Dark = np.append(self.Dark, empty_array, axis=0)

        elif len(sizes) == 3:
            self.Light = np.append(self.Light, np.zeros([1, sizes[1], sizes[2]]),
                                   axis=0)
            self.Dark = np.append(self.Dark, np.zeros([1, sizes[1], sizes[2]]),
                                   axis=0)

    def calculate_next_values(self):


    def calculate_inner_values(self):


    def calculate_inner_buffer(self):


    def calculate_outer_buffer(self):


    def initialise_BCs(self):
