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


def design_points(rng, n_stages):
    unit = qmc.LatinHypercube(d=NU, seed=rng).random(n_stages)
    return qmc.scale(unit, U_LO, U_HI)


def make_aprbs(rng, n_steps, hold_range, ramp_steps=0, log_uniform=True):
    """APRBS-like input signal (n_steps x NU).
    - dwell time of every stage: random in hold_range (samples)
    - amplitudes: Latin hypercube design points in [U_LO, U_HI]
    - ramp_steps > 0: linear ramp (in samples) from the previous to the new design point
      before each dwell phase; ramp_steps = 0 gives the classic APRBS with jumps."""
    dwell = draw_dwell_times(rng, n_steps, hold_range, log_uniform)
    d = design_points(rng, len(dwell))
    u, prev = [], d[0]
    for h, d_s in zip(dwell, d):
        if ramp_steps > 0:
            alpha = np.arange(1, ramp_steps + 1)[:, None] / ramp_steps
            u.append(prev + alpha * (d_s - prev))       # ramp phase
        u.append(np.tile(d_s, (h, 1)))                  # dwell phase
        prev = d_s
    return np.vstack(u)[:n_steps]