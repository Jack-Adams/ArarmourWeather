# Author: Jack Adams
# Date Started: 18/05/17
# Last Updated: 18/06/1

# This file contains the definitions of the map and point structures. It also
# contains all of the methods which act on those structures.

import numpy as np
from scipy.stats import skewnorm
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

    def calculate_next_time_step(self, magic_field, dif_field, pres_field,
                                 tstep, map_width):
        """
        This method is called to do most of the work. This method looks at the
        previous time step, looping through them. It calls various finite
        difference schemes depending on which point it is looking at. Lastly,
        it will using special functions to create specific boundary conditions
        along the outside of the map.

        :param tstep: The time step for which these values will be calculated.
        :param map_width: The width of the generated map.
        """

        w = map_width + 4

        for i in range(1, w+1):
            for j in range(1, w+1):
                if ((i == 1 and j == 1) or (i == 1 and j == w) or
                        (i == w and j == 1) or (i == w and j == w)):
                    self.outer_corner_stencil(magic_field, dif_field,
                                              pres_field, tstep, i, j)
                elif i == 1 or i == w:
                    self.outer_vert_stencil(magic_field, dif_field,
                                            pres_field, tstep, i, j)
                elif j == 1 or j == w:
                    self.outer_horz_stencil(magic_field, dif_field,
                                            pres_field, tstep, i, j)
                elif i == 2 or j == 2 or i == w-1 or j == w-1:
                    self.inner_stencil(magic_field, dif_field,
                                       pres_field, tstep, i, j)
                else:
                    self.roi_stencil(magic_field, dif_field, pres_field,
                                     tstep, i, j)

    def update_pressure(self, magic_field1, magic_field2, pres_field,
                        tstep, map_width):
        """
        Takes two paired Magic field and finds the Magical pressure at each
        point across the map.

        :param magic_field1: The first Magic field.
        :param magic_field2: The second Magic field.
        :param pres_field: The pressure field associated with the given Magic
                           fields.
        :param tstep: The current time.
        :param map_width: The number of points across the region of interest.
        """

        w = map_width + 6

        for i in range(w):
            for j in range(w):
                pres_field[tstep, i, j] = (magic_field1[tstep, i, j] +
                                          magic_field2[tstep, i, j])

    def roi_stencil(self, magic_field, dif_field, pres_field, tstep, x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        beta = 0.02

        magic = (1/45 * dif_field[x-3, y] * magic_field[tstep-1, x-3, y] -
                 3/10 * dif_field[x-2, y] * magic_field[tstep-1, x-2, y] +
                 3 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                 98/9 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                 3 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                 3/10 * dif_field[x+2, y] * magic_field[tstep-1, x+2, y] +
                 1/45 * dif_field[x+3, y] * magic_field[tstep-1, x+3, y] +
                 1/45 * dif_field[x, y-3] * magic_field[tstep-1, x, y-3] -
                 3/10 * dif_field[x, y-2] * magic_field[tstep-1, x, y-2] +
                 3 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                 3 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1] -
                 3/10 * dif_field[x, y+2] * magic_field[tstep-1, x, y+2] +
                 1/45 * dif_field[x, y+3] * magic_field[tstep-1, x, y+3])

        pressure = (1/45 * pres_field[tstep-1, x-3, y] -
                    3/10 * pres_field[tstep-1, x-2, y] +
                    3 * pres_field[tstep-1, x-1, y] -
                    98/9 * pres_field[tstep-1, x, y] +
                    3 * pres_field[tstep-1, x+1, y] -
                    3/10 * pres_field[tstep-1, x+2, y] +
                    1/45 * pres_field[tstep-1, x+3, y] +
                    1/45 * pres_field[tstep-1, x, y-3] -
                    3/10 * pres_field[tstep-1, x, y-2] +
                    3 * pres_field[tstep-1, x, y-1] +
                    3 * pres_field[tstep-1, x, y+1] -
                    3/10 * pres_field[tstep-1, x, y+2] +
                    1/45 * pres_field[tstep-1, x, y+3])

        magic_field[tstep, x, y] = magic_field[tstep - 1, x, y] + magic \
            + pressure * beta

    def inner_stencil(self, magic_field, dif_field, pres_field, tstep,
                             x, y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at any point along the inner buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        beta = 0.02

        magic = (-1/6 * dif_field[x-2, y] * magic_field[tstep-1, x-2, y] +
                 8/3 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                 10 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                 8/3 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                 1/6 * dif_field[x+2, y] * magic_field[tstep-1, x+2, y] -
                 1/6 * dif_field[x, y-2] * magic_field[tstep-1, x, y-2] +
                 8/3 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                 8/3 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1] -
                 1/6 * dif_field[x, y+2] * magic_field[tstep-1, x, y+2])

        pressure = (-1/6 * pres_field[tstep-1, x-2, y] +
                    8/3 * pres_field[tstep-1, x-1, y] -
                    10 * pres_field[tstep-1, x, y] +
                    8/3 * pres_field[tstep-1, x+1, y] -
                    1/6 * pres_field[tstep-1, x+2, y] -
                    1/6 * pres_field[tstep-1, x, y-2] +
                    8/3 * pres_field[tstep-1, x, y-1] +
                    8/3 * pres_field[tstep-1, x, y+1] -
                    1/6 * pres_field[tstep-1, x, y+2])

        magic_field[tstep, x, y] = magic_field[tstep - 1, x, y] + magic \
            + pressure * beta

    def outer_horz_stencil(self, magic_field, dif_field, pres_field, tstep, x,
                           y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at points along the top or bottom of the outer
        buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        beta = 0.02

        magic = (-1/6 * dif_field[x-2, y] * magic_field[tstep-1, x-2, y] +
                 8/3 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                 9 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                 8/3 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                 1/6 * dif_field[x+2, y] * magic_field[tstep-1, x+2, y] +
                 2 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                 2 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1])

        pressure = (-1/6 * pres_field[tstep-1, x-2, y] +
                    8/3 * pres_field[tstep-1, x-1, y] -
                    9 * pres_field[tstep-1, x, y] +
                    8/3 * pres_field[tstep-1, x+1, y] -
                    1/6 * pres_field[tstep-1, x+2, y] +
                    2 * pres_field[tstep-1, x, y-1] +
                    2 * pres_field[tstep-1, x, y+1])

        magic_field[tstep, x, y] = magic_field[tstep - 1, x, y] + magic \
            + pressure * beta

    def outer_vert_stencil(self, magic_field, dif_field, pres_field, tstep, x,
                           y):
        """
        Calculates the change in Magic due to diffusion of Magic into or from
        surrounding points at points along the left or right of the outer
        buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        :return: A value for the change in Magic in the next time step at the
                 point of interest due to diffusion of this type of Magic.
        """

        beta = 0.02

        magic = (2 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] -
                 9 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                 2 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] -
                 1/6 * dif_field[x, y-2] * magic_field[tstep-1, x, y-2] +
                 8/3 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] +
                 8/3 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1] -
                 1/6 * dif_field[x, y+2] * magic_field[tstep-1, x, y+2])

        pressure = (2 * pres_field[tstep-1, x-1, y] -
                    9 * pres_field[tstep-1, x, y] +
                    2 * pres_field[tstep-1, x+1, y] -
                    1/6 * pres_field[tstep-1, x, y-2] +
                    8/3 * pres_field[tstep-1, x, y-1] +
                    8/3 * pres_field[tstep-1, x, y+1] -
                    1/6 * pres_field[tstep-1, x, y+2])

        magic_field[tstep, x, y] = magic_field[tstep - 1, x, y] + magic\
            + pressure * beta

    def outer_corner_stencil(self, magic_field, dif_field, pres_field, tstep,
                             x, y):
        """
        Calculates the change in Magic due to diffusion and pressure of Magic
        into or from surrounding points at any points in the corners of the
        outer buffer region.

        :param magic_field: One of the arrays of Magic which is stored in the
                            map.
        :param dif_field: The diffusion field associated with that Magic field.
        :param pres_field: The array which corresponds to the sum of Magics
                           at a point.
        :param tstep: The value of the next time step in the Magic array.
        :param x: The x-position of the point of interest.
        :param y: The y-position of the point of interest.
        """

        beta = 0.02

        magic = (2 * dif_field[x-1, y] * magic_field[tstep-1, x-1, y] +
                 2 * dif_field[x, y-1] * magic_field[tstep-1, x, y-1] -
                 8 * dif_field[x, y] * magic_field[tstep-1, x, y] +
                 2 * dif_field[x+1, y] * magic_field[tstep-1, x+1, y] +
                 2 * dif_field[x, y+1] * magic_field[tstep-1, x, y+1])

        pressure = (2 * pres_field[tstep-1, x-1, y] +
                    2 * pres_field[tstep-1, x, y-1] -
                    8 * pres_field[tstep-1, x, y] +
                    2 * pres_field[tstep-1, x+1, y] +
                    2 * pres_field[tstep-1, x, y+1])

        magic_field[tstep, x, y] = magic_field[tstep - 1, x, y] + magic\
            + pressure * beta

    def generate_BCs(self, magic_field, tstep, map_width):
        """
        For the entire outer edge of the map, this method will generate the
        boundary conditions for the current time step which are required to
        find the values of the Magic field in the next time step. This is done
        by using a finite difference scheme to find the first order change in
        the Magic field values which have been calculated and then predict what
        the Magic field value around the outside should be.

        :param magic_field: The Magic field which needs its BCs calculated.
        :param tstep: The current time step.
        :param map_width: The number of points across the region of interest.
        """

        w = map_width + 5

        for i in range(1, w):
            magic_field[tstep, i, 0] = skewnorm.rvs(0, loc=magic_field[
                                                         tstep-1, i, 0],
                                                     scale=10)

            magic_field[tstep, i, w] = skewnorm.rvs((50/(100.1-magic_field[
                                            tstep-1, i, w])),
                                                     loc=magic_field[
                                                         tstep-1, i, w],
                                                     scale=15)

            magic_field[tstep, 0, i] = skewnorm.rvs((50/(100.1-magic_field[
                                            tstep-1, 0, i])),
                                                     loc=magic_field[
                                                         tstep-1, 0, i],
                                                     scale=15)

            magic_field[tstep, w, i] = skewnorm.rvs((50/(100.1-magic_field[
                                            tstep-1, 0, i])),
                                                     loc=magic_field[
                                                         tstep-1, 0, i],
                                                     scale=15)

    def LD_forcing_functions(self, light_field, dark_field, tstep, map_width):
        """
        Enforces the generation and consumption of Magic which drives the
        systems.

        :param light_field: The Magic Field corresponding to Light Magic.
        :param dark_field: The Magic Field corresponding to Dark Magic.
        :param tstep: The current time step.
        :param map_width: The number of points in the region of interest.
        """

        w = map_width + 5

        light_field[tstep, 11, 7] = (light_field[tstep, 11, 7] +
                                     skewnorm.rvs(4, loc=200, scale=80))
        consumption = light_field[tstep, 17, 10] * 1/5
        light_field[tstep, 11, 15] = light_field[tstep, 11, 15] - consumption
        dark_field[tstep, 11, 15] = dark_field[tstep, 11, 15] + consumption
