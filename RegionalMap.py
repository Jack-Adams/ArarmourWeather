# Author: Jack Adams
# Date Started: 18/06/3
# Last Updated: 18/06/3

# This file can run the functions associated with the Map to load, save, and
# step through time for the regional map of Aramour.

import numpy as np
from MapFunctions import RegionMap

# First set up the Map.
Aramour = RegionMap()
height = 50
width = 70
centre = np.array([25, 10])
Aramour.initialise_map(height, width)

Aramour.find_magic(0, 1000, height, width, centre)

print("Done!")

Aramour.save_map('C:/Users/Jack/Desktop/seasonal_test', centre)

test = RegionMap()
test.load_map('C:/Users/Jack/Desktop/test')
print('done!')