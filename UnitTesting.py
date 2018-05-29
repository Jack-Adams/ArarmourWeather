import unittest as test
import MapStructures as MS
import numpy as np
import scipy as sp


class TestMapSetup(test.TestCase):

    def test_map(self):
        test_map = MS.Map()
        self.assertEqual(test_map.Light, None)
        self.assertEqual(test_map.Dark, None)
        self.assertEqual(test_map.DifLight, None)
        self.assertEqual(test_map.DifDark, None)

    def test_prepare_map_arrays(self):
        test_map = MS.Map()
        test_map.prepare_map_arrays(1)
        self.assertEqual(test_map.Light.shape[0], 7)
        self.assertEqual(test_map.Light.shape[1], 7)
        self.assertEqual(test_map.Dark.shape[0], 7)
        self.assertEqual(test_map.Dark.shape[1], 7)
        self.assertEqual(test_map.DifLight.shape[0], 7)
        self.assertEqual(test_map.DifLight.shape[1], 7)
        self.assertEqual(test_map.DifDark.shape[0], 7)
        self.assertEqual(test_map.DifDark.shape[1], 7)
        self.assertEqual(test_map.LDPressure.shape[0], 7)
        self.assertEqual(test_map.LDPressure.shape[1], 7)

    def test_initialise_values(self):
        test_map = MS.Map()
        test_map.prepare_map_arrays(1)
        test_map.initialise_values()
        self.assertEqual(test_map.Light[3, 3], 100)
        self.assertEqual(test_map.Dark[3, 3], 100)
        self.assertEqual(test_map.DifLight[3, 3], 1)
        self.assertEqual(test_map.DifDark[3, 3], 1)
        self.assertEqual(test_map.LDPressure[3, 3], 200)

        test_map.initialise_values(50, 3)
        self.assertEqual(test_map.Light[3, 3], 50)
        self.assertEqual(test_map.Dark[3, 3], 50)
        self.assertEqual(test_map.DifLight[3, 3], 3)
        self.assertEqual(test_map.DifDark[3, 3], 3)
        self.assertEqual(test_map.LDPressure[3, 3], 100)

    def test_create_next_time_step(self):
        test_map = MS.Map()
        test_map.prepare_map_arrays(1)
        test_map.create_next_time_step()
        test_sizes1 = np.array([2, 7, 7])
        for i in range(len(test_sizes1)):
            self.assertEqual(test_map.Light.shape[i], test_sizes1[i])
            self.assertEqual(test_map.Dark.shape[i], test_sizes1[i])

        test_map.create_next_time_step()
        test_sizes2 = np.array([3, 7, 7])
        for i in range(len(test_sizes2)):
            self.assertEqual(test_map.Light.shape[i], test_sizes2[i])
            self.assertEqual(test_map.Dark.shape[i], test_sizes2[i])

        test_map.create_next_time_step()
        test_map.create_next_time_step()
        test_map.create_next_time_step()
        test_sizes3 = np.array([6, 7, 7])
        for i in range(len(test_sizes2)):
            self.assertEqual(test_map.Light.shape[i], test_sizes3[i])
            self.assertEqual(test_map.Dark.shape[i], test_sizes3[i])


class TestFiniteDifferenceSchemes(test.TestCase):

    #def test_roi_stencil(self):


    #def test_inner_buffer_stencil(self):


    #def test_outer_horz_stencil(self):


    def test_outer_vert_stencil(self):
        # First, set up the map and the initial conditions for the tests.
        test_map = MS.Map()
        test_map.Light = 100 * np.ones([1, 9, 9])
        test_map.Light = np.append(test_map.Light, np.zeros([1, 9, 9]), axis=0)
        test_map.DifLight = 0.05 * np.ones([9, 9])
        test_map.LDPressure = 200 * np.ones([1, 9, 9])
        test_map.LDPressure = np.append(test_map.LDPressure,
                                        np.zeros([1, 9, 9]), axis=0)
        time_step = 1
        test_map.Light[time_step - 1, 4, 4] = 150

        # Next, calculate the values for the stencil.
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 4, 4)
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 3, 4)
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 5, 4)
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 4, 2)
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 4, 3)
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 4, 5)
        test_map.outer_vert_stencil(test_map.Light, test_map.DifLight,
                                    test_map.LDPressure, time_step, 4, 6)

        # Lastly test the outputs of the stencil.
        self.assertAlmostEqual(test_map.Light[1, 4, 4], 127.5)
        self.assertAlmostEqual(test_map.Light[1, 3, 4], 105)
        self.assertAlmostEqual(test_map.Light[1, 5, 4], 105)
        self.assertAlmostEqual(test_map.Light[1, 4, 2], 99.583)
        self.assertAlmostEqual(test_map.Light[1, 4, 3], 106.667)
        self.assertAlmostEqual(test_map.Light[1, 4, 5], 106.667)
        self.assertAlmostEqual(test_map.Light[1, 4, 6], 99.583)

    def test_outer_corner_stencil(self):
        # First, set up the map and the initial conditions for the tests.
        test_map = MS.Map()
        test_map.Light = 100 * np.ones([1, 9, 9])
        test_map.Light = np.append(test_map.Light, np.zeros([1, 9, 9]), axis=0)
        test_map.DifLight = 0.05 * np.ones([9, 9])
        test_map.LDPressure = 200 * np.ones([1, 9, 9])
        test_map.LDPressure = np.append(test_map.LDPressure,
                                        np.zeros([1, 9, 9]), axis=0)
        time_step = 1
        test_map.Light[time_step - 1, 4, 4] = 150

        # Next, calculate the values for the stencil.
        test_map.outer_corner_stencil(test_map.Light, test_map.DifLight,
                                      test_map.LDPressure, time_step, 4, 4)
        test_map.outer_corner_stencil(test_map.Light, test_map.DifLight,
                                      test_map.LDPressure, time_step, 3, 4)
        test_map.outer_corner_stencil(test_map.Light, test_map.DifLight,
                                      test_map.LDPressure, time_step, 5, 4)
        test_map.outer_corner_stencil(test_map.Light, test_map.DifLight,
                                      test_map.LDPressure, time_step, 4, 3)
        test_map.outer_corner_stencil(test_map.Light, test_map.DifLight,
                                      test_map.LDPressure, time_step, 4, 5)

        # Lastly test the outputs of the stencil.
        self.assertEqual(test_map.Light[1, 4, 4], 130)
        self.assertEqual(test_map.Light[1, 3, 4], 105)
        self.assertEqual(test_map.Light[1, 5, 4], 105)
        self.assertEqual(test_map.Light[1, 4, 3], 105)
        self.assertEqual(test_map.Light[1, 4, 5], 105)


if __name__ == '__main__':
    test.main()