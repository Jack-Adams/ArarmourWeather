# Author: Jack Adams
# Date Started: 18/05/17
# Last Updated: 18/05/19

# This file contains the definitions of the map and point structures. It also
# contains all of the methods which act on those structures.

import numpy as np
import scipy as sp
import h5py as h5


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
    LDPressure = None

    def prepare_map_arrays(self, map_width):
        """
        This method will generate the set of points corresponding to locations
        on the map. The map will have a buffer of three additional points
        around each side to allow the boundary conditions to be initialised
        outside the region of interest.

        :param map_width: An integer corresponding to the desired number of
                          points across the map.
        """

        self.Light = np.zeros([map_width + 6, map_width + 6])
        self.Dark = np.zeros([map_width + 6, map_width + 6])
        self.DifLight = np.zeros([map_width + 6, map_width + 6])
        self.DifDark = np.zeros([map_width + 6, map_width + 6])
        self.LDPressure = np.zeros([map_width + 6, map_width + 6])

    def initialise_values(self, magic_value=-1, dif_value=-1):
        """
        Seed the created arrays with random values unless a specific starting
        value is provided

        :param magic_value: A specific starting value given to start the arrays
                            at.
        :param dif_value: A specific starting value given to start the
                          diffusion value of each array at.
        """

        if magic_value == -1:
            for i in range(self.Light.shape[0]):
                for j in range(self.Light.shape[1]):
                    self.Light[i, j] = 100
                    self.Dark[i, j] = 100
                    self.LDPressure[i, j] = self.Light[i, j] + self.Dark[i, j]

        elif magic_value != -1:
            for i in range(self.Light.shape[0]):
                for j in range(self.Light.shape[1]):
                    self.Light[i, j] = magic_value
                    self.Dark[i, j] = magic_value
                    self.LDPressure[i, j] = self.Light[i, j] + self.Dark[i, j]

        if dif_value == -1:
            for i in range(self.Light.shape[0]):
                for j in range(self.Light.shape[1]):
                    self.DifLight[i, j] = 1
                    self.DifDark[i, j] = 1

        elif dif_value != -1:
            for i in range(self.Light.shape[0]):
                for j in range(self.Light.shape[1]):
                    self.DifLight[i, j] = dif_value
                    self.DifDark[i, j] = dif_value

    def create_next_time_step(self):
        """ Increases the third dimension of each Magic array by one. """

        sizes = self.Light.shape

        if len(sizes) == 2:
            empty_array = np.zeros([1, sizes[0], sizes[1]])
            self.Light = np.array([self.Light])
            self.Light = np.append(self.Light, empty_array, axis=0)
            self.Dark = np.array([self.Dark])
            self.Dark = np.append(self.Dark, empty_array, axis=0)
            self.LDPressure = np.array([self.LDPressure])
            self.LDPressure = np.append(self.LDPressure, empty_array, axis=0)

        elif len(sizes) == 3:
            self.Light = np.append(self.Light, np.zeros([1, sizes[1], sizes[2]]),
                                   axis=0)
            self.Dark = np.append(self.Dark, np.zeros([1, sizes[1], sizes[2]]),
                                   axis=0)
            self.LDPressure = np.append(self.LDPressure, np.zeros([1, sizes[1],
                                                                   sizes[2]]),
                                        axis=0)

    def calculate_next_time_step(self, map_width):
        """
        This method is called to do most of the work. This method looks at the
        previous time step, looping through them. It calls various finite
        difference schemes depending on which point it is looking at. Lastly,
        it will using special functions to create specific boundary conditions
        along the outside of the map.
        """

        w = map_width + 5

        for i in range (1, w-1):
            for j in range(1, w-1):
                if ((i == -1 && j == -1) || (i == 1 && j == -1) || 
                     (i == -1 && j == 1) || (i == 1 && j == 1)):
                     self.Light

    def calculate_roi_values(self, magic_field, dif_field, pres_field, tstep,
                             map_width):
        """
        This method is used to calculate the values of the given type of Magic
        across the region of interest, defined by the size of map_width, for
        the given time.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param dif_field: The map array which is the Magical pressure.
        :param tstep: The value of the next time step in the Magic array.
        :param map_width: The number of points wide the region of interest is.
        """

        for i in range(3, map_width + 2):
            for j in range(3, map_width + 2):
                magic_field[tstep, i, j] = (magic_field[tstep - 1 , i, j]
                                            + self.roi_magic_stencil(magic_field,
                                                                dif_field,
                                                                tstep, i, j)
                                            + self.roi_pressure_stencil(pres_field,
                                                                   tstep, i, j))

    def calculate_inner_buffer_values(self, magic_field, dif_field, pres_field,
                                      tstep, map_width):
        """
        This method is used to find the values of the given type of Magic
        in the inner buffer region, the one-point-thick layer around the
        region of interest.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param dif_field: The map array which is the Magical pressure.
        :param tstep: The value of the next time step in the Magic array.
        :param map_width: The number of points wide the region of interest is.
        """

        w = map_width + 3

        for i in range(2, w):
            magic_field[tstep, i, 2] = (magic_field[tstep - 1, i, 2] +
                                        self.inner_buffer_magic_stencil(magic_field,
                                            dif_field, tstep, i, 2) +
                                        self.inner_buffer_pres_stencil(pres_field,
                                            tstep, i, 2))

            magic_field[tstep, 2, i] = (magic_field[tstep - 1, 2, i] +
                                        self.inner_buffer_magic_stencil(magic_field,
                                            dif_field, tstep, 2, i) +
                                        self.inner_buffer_pres_stencil(pres_field,
                                            tstep, 2, i))

            magic_field[tstep, i, w] = (magic_field[tstep - 1, i, w] +
                                        self.inner_buffer_magic_stencil(magic_field,
                                            dif_field, tstep, i, w) +
                                        self.inner_buffer_pres_stencil(pres_field,
                                            tstep, i, w))

            magic_field[tstep, w, i] = (magic_field[tstep - 1, w, i] +
                                        self.inner_buffer_magic_stencil(magic_field,
                                            dif_field, tstep, w, i) +
                                        self.inner_buffer_pres_stencil(pres_field,
                                            tstep, w, i))


    def roi_magic_stencil(self, magic_field, dif_field, tstep, x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        magnitude = (1/45 * dif_field[x-3, y] * magic_field[tstep-1, x-3, y] -
                     3/10 * dif_field[x-2, y] * magic_field[tstep-1, x-2, y] +
                     3 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] +
                     3 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                     3/10 * dif_field[x+2, y] * magic_field[tstep-1, x+2, y] +
                     1/45 * dif_field[x+3, y] * magic_field[tstep-1, x+3, y] +
                     1/45 * dif_field[x, y-3] * magic_field[tstep-1, x, y-3] -
                     3/10 * dif_field[x, y-2] * magic_field[tstep-1, x, y-2] +
                     3 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                     3 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1] -
                     3/10 * dif_field[x, y+2] * magic_field[tstep-1, x, y+2] +
                     1/45 * dif_field[x, y+3] * magic_field[tstep-1, x, y+3] -
                     98/9 * dif_field[x, y] * magic_field[tstep-1, x, y])

        return magnitude

    def roi_pres_stencil(self, pres_field, tstep, x, y):
        """
        Calculates the magnitude of the force pushing all Magics away from a
        point due to the high 'pressure' or presence of lots of Magic at that
        point.

        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to the total pressure of Magic.
        """

        beta = 1
        magnitude = (1 / 45 * pres_field[tstep-1, x - 3, y] -
                     3 / 10 * pres_field[tstep-1, x - 2, y] +
                     3 * pres_field[tstep-1, x - 1, y] +
                     3 * pres_field[tstep-1, x + 1, y] -
                     3 / 10 * pres_field[tstep-1, x + 2, y] +
                     1 / 45 * pres_field[tstep-1, x + 3, y] +
                     1 / 45 * pres_field[tstep-1, x, y - 3] -
                     3 / 10 * pres_field[tstep-1, x, y - 2] +
                     3 * pres_field[tstep-1, x, y - 1] +
                     3 * pres_field[tstep-1, x, y + 1] -
                     3 / 10 * pres_field[tstep-1, x, y + 2] +
                     1 / 45 * pres_field[tstep-1, x, y + 3] -
                     98 / 9 * pres_field[tstep-1, x, y]) * beta

        return magnitude

    #def inner_buffer_magic_stencil(self, magic_field, dif_field, tstep, x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at any point along the inner buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        magnitude = (-1/6 * dif_field[x-2, y] * magic_field[tstep-1, x-2, y] +
                     8/3 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                     10 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                     8/3 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                     1/6 * dif_field[x+2, y] * magic_field[tstep-1, x+2, y] -
                     1/6 * dif_field[x, y-2] * magic_field[tstep-1, x, y-2] +
                     8/3 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                     8/3 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1] -
                     1/6 * dif_field[x, y+2] * magic_field[tstep-1, x, y+2])

        return magnitude


    #def inner_buffer_pres_stencil(self, pres_field, tstep, x, y):
        """
        Calculates the magnitude of the force pushing all Magics away from a
        point due to the high 'pressure' or presence of lots of Magic at that
        point for points in the inner buffer region.

        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to the total pressure of Magic.
        """

        beta = 1
        magnitude = (-1/6 * pres_field[tstep-1, x-2, y] +
                     8/3 * pres_field[tstep-1, x-1, y] -
                     10 * pres_field[tstep-1, x, y] +
                     8/3 * pres_field[tstep-1, x+1, y] -
                     1/6 * pres_field[tstep-1, x+2, y] -
                     1/6 * pres_field[tstep-1, x, y-2] +
                     8/3 * pres_field[tstep-1, x, y-1] +
                     8/3 * pres_field[tstep-1, x, y+1] -
                     1/6 * pres_field[tstep-1, x, y+2]) * beta

        return magnitude

    #def outer_horz_magic_stencil(self, magic_field, dif_field, tstep, x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at points along the top or bottom of the outer
        buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        magnitude = (-1/6 * dif_field[x-2, y] * magic_field[tstep-1, x-2, y] +
                     8/3 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                     9 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                     8/3 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                     1/6 * dif_field[x+2, y] * magic_field[tstep-1, x+2, y] -
                     2 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                     2 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1]):

        return magnitude

    #def outer_horz_pres_stencil(self, pres_field, tstep, x, y):
        """
        Calculates the magnitude of the force pushing all Magics away from a
        point due to the high 'pressure' or presence of lots of Magic at that
        point for points along the top or bottom of the outer buffer region.

        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to the total pressure of Magic.
        """

        beta = 1
        magnitude = (-1/6 * pres_field[tstep-1, x-2, y] +
                     8/3 * pres_field[tstep-1, x-1, y] -
                     9 * pres_field[tstep-1, x, y] +
                     8/3 * pres_field[tstep-1, x+1, y] -
                     1/6 * pres_field[tstep-1, x+2, y] +
                     2 * pres_field[tstep-1, x, y-1] +
                     2 * pres_field[tstep-1, x, y+1]) * beta

        return magnitude

    #def outer_vert_magic_stencil(self, magic_field, dif_field, tstep, x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at points along the left or right of the outer
        buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        magnitude = (2 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                     9 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                     2 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                     1/6 * dif_field[x, y-2] * magic_field[tstep-1, x, y-2] +
                     8/3 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                     8/3 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1] -
                     1/6 * dif_field[x, y+2] * magic_field[tstep-1, x, y+2])

        return magnitude


    #def outer_vert_pres_stencil(self, pres_field, tstep, x, y):
        """
        Calculates the magnitude of the force pushing all Magics away from a
        point due to the high 'pressure' or presence of lots of Magic at that
        point for points along the left or right of the outer buffer region.

        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to the total pressure of Magic.
        """

        beta = 1
        magnitude = (2 * pres_field[tstep-1, x-1, y] -
                     9 * pres_field[tstep-1, x, y] +
                     2 * pres_field[tstep-1, x+1, y] -
                     1/6 * pres_field[tstep-1, x, y-2] +
                     8/3 * pres_field[tstep-1, x, y-1] +
                     8/3 * pres_field[tstep-1, x, y+1] -
                     1/6 * pres_field[tstep-1, x, y+2]) * beta

        return magnitude


    #def outer_corner_magic_stencil(self, magic_field, dif_field, tstep, x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at any points in the corners of the outer buffer
        region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        magnitude = (2 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] +
                     2 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] -
                     8 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                     2 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] +
                     2 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1])

        return magnitude


    #def outer_corner_pres_stencil(selfpres_field, tstep, x, y):
        """
        Calculates the magnitude of the force pushing all Magics away from a
        point due to the high 'pressure' or presence of lots of Magic at that
        point for points in the corners of the outer buffer region.

        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to the total pressure of Magic.
        """

        beta = 1
        magnitude = (2 * pres_field[tstep-1, x-1, y] +
                     2 * pres_field[tstep-1, x, y-1] -
                     2 * pres_field[tstep-1, x, y] +
                     2 * pres_field[tstep-1, x+1, y] +
                     2 * pres_field[tstep-1, x, y+1]
                     ) * beta

        return magnitude


    #def initialise_BCs(self):
