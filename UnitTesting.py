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

    def test_roi_magic_stencil(self):
        test_map = MS.Map()
        magic_field = 100 * np.ones([1, 9, 9])
        magic_field = np.append(magic_field, np.zeros([1, 9, 9]), axis=0)
        dif_field = 0.2 * np.ones([9, 9])
        time_step = 1
        y = 4
        x = 3
        magic_field[time_step - 1, 4, 4] = 150
        value = test_map.roi_magic_stencil(magic_field, dif_field, time_step,
                                           x, y)
        self.assertEqual(np.round(value, 1), 30.0)

        x = 4
        value = test_map.roi_magic_stencil(magic_field, dif_field, time_step,
                                           x, y)
        self.assertEqual(np.round(value, 1), -108.9)

    def test_roi_pres_stencil(self):
        test_map = MS.Map()
        pres_field = 100 * np.ones([1, 9, 9])
        pres_field = np.append(pres_field, np.zeros([1, 9, 9]), axis=0)
        time_step = 1
        y = 4
        x = 3
        pres_field[time_step - 1, 4, 4] = 150
        value = test_map.roi_pres_stencil(pres_field, time_step,
                                              x, y)
        self.assertEqual(np.round(value, 1), 30.0)

        x = 4
        value = test_map.roi_pres_stencil(pres_field, time_step,
                                              x, y)
        self.assertEqual(np.round(value, 1), -108.9)

if __name__ == '__main__':
    test.main()