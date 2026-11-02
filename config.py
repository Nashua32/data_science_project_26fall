SEED = 42
DT = 1  # Sampling Time [min]
H = 20  # Horizon [steps]
NY, NU = 6, 2
QB_bounds = (180.0, 360.0)                      # dm3/min
Tr_bounds = (75 + 273.15, 100 + 273.15)         # K (75-100 degC)

# How to pick DT
# from simulation.estimate_time_constants 
# fastest tau: 0.138 min (8.3 s)
# slowest tau: 7.20 min
# longest settling time (t95): 21.4 min

# Horizon should longer than the longest settling time, so > 21.4 min so settled state is visible for mpc
# we pick H = 20 steps, not too many so not too computationally expensive
# so DT = 21.4 / 20 = 1.07 min ~ 1 min