import numpy as np
from scipy.integrate import solve_ivp
from model import wo_reactor_odes

# Integrates ODE system for 1 step
def step(y0, u, dt):
    sol = solve_ivp(lambda t, c: wo_reactor_odes(t, c, u), (0, dt), y0,method="BDF", rtol=1e-8, atol=1e-10)
    return sol.y[:, -1]