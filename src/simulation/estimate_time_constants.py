"""Estimate the time constants of the Williams-Otto CSTR.

Two independent methods at several operating points (grid over QB and Tr):
  1. Linearization: eigenvalues of the Jacobian at the steady state -> tau_i = -1/Re(lambda_i)
  2. Step responses of the nonlinear model -> time to 63 % (t63) and settling time (t95)
Time unit: minutes.
"""
import sys;from pathlib import Path;sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from model import wo_reactor_odes
from config import QB_bounds, Tr_bounds

STATE_NAMES = ["cA", "cB", "cC", "cP", "cE", "cG"]
STEP_FRACTION = 0.10      # input step = 10 % of the input range
MIN_CHANGE = 1e-5         # states that change less than this [mol/dm3] are ignored
PLOT = True


def rhs(c, u):
    return np.array(wo_reactor_odes(0.0, c, u))


def steady_state(u, c_init=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)):
    """Long integration, then polish with fsolve."""
    sol = solve_ivp(lambda t, c: wo_reactor_odes(t, c, u), (0, 120), c_init,
                    method="BDF", rtol=1e-10, atol=1e-12)
    c = fsolve(lambda c: rhs(c, u), sol.y[:, -1], xtol=1e-10)
    if np.max(np.abs(rhs(c, u))) > 1e-9 or c.min() < -1e-9:
        raise RuntimeError(f"No valid steady state for u={u}")
    return c


def jacobian(c, u, eps=1e-6):
    """Central finite-difference Jacobian of the right-hand side."""
    J = np.zeros((len(c), len(c)))
    for j in range(len(c)):
        d = np.zeros(len(c)); d[j] = eps * max(1.0, abs(c[j]))
        J[:, j] = (rhs(c + d, u) - rhs(c - d, u)) / (2 * d[j])
    return J


def eigen_time_constants(c, u):
    ev = np.linalg.eigvals(jacobian(c, u))
    if np.any(ev.real >= 0):
        print(f"  WARNING: eigenvalue with Re >= 0 at u={u} (not asymptotically stable)")
    ev = ev[ev.real < 0]
    return np.sort(-1.0 / ev.real)


def step_response(c_ss, u_new, t_end=60.0, n=3000):
    t = np.linspace(0, t_end, n)
    sol = solve_ivp(lambda t, c: wo_reactor_odes(t, c, u_new), (0, t_end), c_ss,
                    method="BDF", t_eval=t, rtol=1e-9, atol=1e-12)
    return t, sol.y.T


def response_times(t, y):
    """Per state: t63 (first time 63.2 % of the final change is reached) and
    t95 (last time the response is outside +-5 % of the final change)."""
    t63, t95 = np.full(y.shape[1], np.nan), np.full(y.shape[1], np.nan)
    dy = y - y[0]
    for i in range(y.shape[1]):
        d_end = dy[-1, i]
        if abs(d_end) < MIN_CHANGE:
            continue
        frac = dy[:, i] / d_end
        t63[i] = t[np.argmax(frac >= 0.632)]
        outside = np.where(np.abs(frac - 1.0) > 0.05)[0]
        t95[i] = t[outside[-1] + 1] if len(outside) else 0.0
    return t63, t95


if __name__ == "__main__":
    QB_pts = np.linspace(*QB_bounds, 3)
    Tr_pts = np.linspace(*Tr_bounds, 3)
    dQB = STEP_FRACTION * (QB_bounds[1] - QB_bounds[0])
    dTr = STEP_FRACTION * (Tr_bounds[1] - Tr_bounds[0])

    all_tau, all_t95 = [], []
    print(f"{'QB':>6} {'Tr[C]':>6} | {'tau_min':>8} {'tau_max':>8} | {'t95(QB step)':>12} {'t95(Tr step)':>12}   [min]")
    for QB in QB_pts:
        for Tr in Tr_pts:
            u0 = (QB, Tr)
            c_ss = steady_state(u0)
            tau = eigen_time_constants(c_ss, u0)
            t_q = np.linspace(0, 60.0, 3000)
            _, y_q = step_response(c_ss, (QB + dQB, Tr))
            _, y_T = step_response(c_ss, (QB, Tr + dTr))
            t95_q = np.nanmax(response_times(t_q, y_q)[1])
            t95_T = np.nanmax(response_times(t_q, y_T)[1])
            all_tau += list(tau); all_t95 += [t95_q, t95_T]
            print(f"{QB:6.0f} {Tr - 273.15:6.1f} | {tau.min():8.3f} {tau.max():8.3f} | {t95_q:12.2f} {t95_T:12.2f}")

    tau_fast, tau_slow, t95_max = min(all_tau), max(all_tau), np.nanmax(all_t95)
    print(f"\nfastest tau: {tau_fast:.3f} min ({tau_fast * 60:.1f} s)")
    print(f"slowest tau: {tau_slow:.2f} min,  longest settling time (t95): {t95_max:.1f} min")
    print(f"rule of thumb: DT ~ tau_fast/10 ... tau_fast/3 = {tau_fast * 60 / 10:.1f} ... {tau_fast * 60 / 3:.1f} s")
    print(f"               longest hold time ~ 3-4 * tau_slow = {3 * tau_slow:.0f} ... {4 * tau_slow:.0f} min")

    if PLOT:
        QB_m, Tr_m = QB_pts[1], Tr_pts[1]
        u0 = (QB_m, Tr_m); c_ss = steady_state(u0)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, (u_new, title) in zip(ax, [((QB_m + dQB, Tr_m), "Step in QB"),
                                          ((QB_m, Tr_m + dTr), "Step in Tr")]):
            t, y = step_response(c_ss, u_new, t_end=40.0, n=2000)
            for i, name in enumerate(STATE_NAMES):
                a.plot(t, y[:, i] - y[0, i], label=name)
            a.set_xlabel("Time [min]"); a.set_ylabel("Deviation from steady state [mol/dm3]")
            a.set_title(title + " (middle operating point)"); a.legend(ncol=3)
        plt.tight_layout(); plt.show()