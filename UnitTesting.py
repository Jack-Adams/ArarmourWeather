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

    # def test_prepare_map_arrays(self):
    #    test_map = MS.Map()
    #    test_map.prepare_map_arrays(1)
    #    for i in test_map.Light[0]:
    #        for j in test_map.Light[0]:
    #            self.assertTrue(test_map.Light[i, j] == 0)

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


if __name__ == '__main__':
    test.main()