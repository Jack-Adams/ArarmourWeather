# Author: Jack Adams
# Date Started: 18/05/27
# Last Updated: 18/05/27

# This script is used to run the finite element schemes over time.

import MapStructures

aramour = MapStructures.Map()
aramour.prepare_map_arrays(5)
aramour.initialise_values(100, 0.005)
aramour.create_next_time_step()
aramour.Light[0, 5, 5] = 150
aramour.LDPressure[0, 5, 5] = 250
tstep = 1

aramour.calculate_next_time_step(aramour.Light, aramour.DifLight,
                                 aramour.LDPressure, tstep, 5)
print(aramour.Light[1])
