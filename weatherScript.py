# Author: Jack Adams
# Date Started: 18/05/27
# Last Updated: 18/06/1

# This script is used to run the finite element schemes over time.

import MapStructures
import numpy as np

aramour = MapStructures.Map()
width = 15
aramour.prepare_map_arrays(width)
aramour.initialise_values(100, 0.04)
aramour.create_next_time_step()
aramour.Light[0, 11, 11] = 150
aramour.LDPressure[0, 11, 11] = 250

for tstep in range(1,100):
    aramour.calculate_next_time_step(aramour.Light, aramour.DifLight,
                                     aramour.LDPressure, tstep, width)
    aramour.calculate_next_time_step(aramour.Dark, aramour.DifDark,
                                     aramour.LDPressure, tstep, width)
    aramour.generate_BCs(aramour.Light, tstep, width)
    aramour.generate_BCs(aramour.Dark, tstep, width)
    aramour.LD_forcing_functions(aramour.Light, aramour.Dark, tstep, width)
    aramour.update_pressure(aramour.Light, aramour.Dark, aramour.LDPressure,
                            tstep, width)
    aramour.create_next_time_step()
print("Done!")
