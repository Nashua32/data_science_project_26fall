import numpy as np
from scipy.spatial.distance import pdist


def standardize(a):
    """Zero mean, unit variance per column (constant columns are only centered)."""
    s = a.std(axis=0)
    s[s == 0] = 1.0
    return (a - a.mean(axis=0)) / s


def build_regressors(y, U, n_a, n_b, t):
    """Regressor rows for the target y[t+1]:
    [Y_t ... Y_{t-n_a+1}, U_t ... U_{t-n_b+1}]  (n_a past states, n_b past inputs)."""
    cols = [y[t - j] for j in range(n_a)] + [U[t - j] for j in range(n_b)]
    return np.hstack(cols)


def lipschitz_index(y, U, n_a, n_b, t, p_frac=0.01, eps=1e-12):
    """Lipschitz index q(n) of He and Asada for one lag combination (n_a, n_b)."""
    X = build_regressors(y, U, n_a, n_b, t)
    T = y[t + 1]
    dx = pdist(X)                       # distances of all regressor pairs
    dy = pdist(T)                       # distances of all target pairs
    ok = dx > eps                       # skip identical regressors
    q = dy[ok] / dx[ok]                 # Lipschitz quotients
    p = max(1, int(p_frac * len(t)))    # number of largest quotients used
    q_top = np.partition(q, -p)[-p:]
    n = X.shape[1]                      # number of regressors
    return np.exp(np.mean(np.log(np.sqrt(n) * q_top + 1e-300)))


def lipschitz_grid(y, U, n_max=6, p_frac=0.01, max_points=3000, seed=0):
    """Index for all combinations n_a, n_b = 1 ... n_max.
    y : states incl. start state, shape (N+1, NY);  U : inputs, shape (N, NU).
    Returns Q with Q[n_a-1, n_b-1]."""
    y, U = standardize(y), standardize(U)          # remove the influence of units
    t = np.arange(n_max - 1, len(U))               # same target rows for every combination
    if len(t) > max_points:                        # limit the number of pairs
        t = np.sort(np.random.default_rng(seed).choice(t, max_points, replace=False))
    Q = np.zeros((n_max, n_max))
    for a in range(1, n_max + 1):
        for b in range(1, n_max + 1):
            Q[a - 1, b - 1] = lipschitz_index(y, U, a, b, t, p_frac)
    return Q


import sys
sys.path.append("src/models")
d = np.load("data/aprbs_data.npz")
y = d["y"]
U = d["U"]

Q = lipschitz_grid(y, U, n_max=6)

import matplotlib.pyplot as plt
plt.figure()
for b in range(Q.shape[1]):
    plt.plot(range(1, Q.shape[0] + 1), Q[:, b], marker='o', label=f'n_b={b+1}')
plt.xlabel('n_a (past states)'); plt.ylabel('Lipschitz index'); plt.yscale('log')
plt.legend(); plt.show()