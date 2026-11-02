import numpy as np
from scipy.stats import qmc

from config import QB_bounds,Tr_bounds, NU     

U_LO, U_HI = [QB_bounds[0],Tr_bounds[0]], [QB_bounds[1],Tr_bounds[1]]


def draw_dwell_times(rng, n_steps, hold_range, log_uniform=True):
    h_min, h_max = hold_range
    dwell, total = [], 0
    while total < n_steps:
        if log_uniform:
            h = int(round(np.exp(rng.uniform(np.log(h_min), np.log(h_max)))))
        else:
            h = int(rng.integers(h_min, h_max, endpoint=True))
        dwell.append(h)
        total += h
    return np.array(dwell)


def make_aprbs(rng, n_steps, hold_range, dwelltime_same, log_uniform=True):
    if dwelltime_same==True:                                            # inputs with same dwell times, sampled on same hypercube
        dwell = draw_dwell_times(rng, n_steps, hold_range, log_uniform)
        unit = qmc.LatinHypercube(d=NU, seed=rng).random(len(dwell))
        amp = qmc.scale(unit, U_LO, U_HI)
        U = np.repeat(amp, dwell, axis=0)[:n_steps]
    else:                                                               # pick inputs with different dwell times, different hypercubes
        cols = []
        for i in range(NU):
            dwell = draw_dwell_times(rng, n_steps, hold_range, log_uniform)
            unit = qmc.LatinHypercube(d=1, seed=rng).random(len(dwell))
            amp = qmc.scale(unit, U_LO[i], U_HI[i])
            cols.append(np.repeat(amp, dwell)[:n_steps]) # repeat each amplitude dwell times amount
        U =  np.column_stack(cols)
    return U

