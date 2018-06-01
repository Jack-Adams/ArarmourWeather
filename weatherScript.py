# Author: Jack Adams
# Date Started: 18/05/27
# Last Updated: 18/06/1

# This script is used to run the finite element schemes over time.

import MapStructures

aramour = MapStructures.Map()
width = 15
aramour.prepare_map_arrays(width)
aramour.initialise_values(100, 0.05)
aramour.create_next_time_step()
aramour.Light[0, 11, 11] = 150
aramour.LDPressure[0, 11, 11] = 250
tstep = 1

aramour.calculate_next_time_step(aramour.Light, aramour.DifLight,
                                 aramour.LDPressure, tstep, width)
print(aramour.Light[1])
