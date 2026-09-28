# src/simulation/william_otto.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class WilliamsOttoReactor:
    DEFAULT_PARAMS = {
        "k10": 9.9594e6,   # dm3/(mol*min)
        "k20": 8.66124e9,  # dm3/(mol*min)
        "k30": 9.9594e6,   # dm3/(mol*min)
        "E1": 6666.7,       # K
        "E2": 8333.3,       # K
        "E3": 11111.0,      # K
        "cA0": 10.0,         # mol/dm3
        "cB0": 10.0,         # mol/dm3
        "V": 2105.0,          # dm3
        "QA": 112.35,        # dm3/min
    }

    QB_bounds = (180.0, 360.0)                     # dm3/min
    Tr_bounds = (75 + 273.15, 100 + 273.15)         # K (source: 75-100 degC)

    def __init__(self, params=None):
        self.p = params if params is not None else self.DEFAULT_PARAMS.copy()

    def reaction_rates(self, c, Tr):
        """Computes reaction rates r1, r2, r3."""
        cA, cB, cC, cP, cE, cG = c
        k1 = self.p["k10"] * np.exp(-self.p["E1"] / Tr)
        k2 = self.p["k20"] * np.exp(-self.p["E2"] / Tr)
        k3 = self.p["k30"] * np.exp(-self.p["E3"] / Tr)

        r1 = k1 * cA * cB
        r2 = k2 * cB * cC
        r3 = k3 * cC * cP
        return r1, r2, r3

    def odes(self, t, c, u_func):
        """ODE system dc/dt for Williams-Otto CSTR.

        Inputs:
            t : float
                current time 
            c : array of 6 floats [cA, cB, cC, cP, cE, cG]
                current concentrations in mol/dm3
            u : callable, u(t) -> (QB, Tr)
                QB : float, flow rate of B in dm3/min
                Tr : float, reactor temperature in K
            p : dict
                plant parameters (params)
    
        Output:
            list of 6 floats [dcA, dcB, dcC, dcP, dcE, dcG]
            time derivatives of the concentrations (mol/(dm3*min))
        """
        cA, cB, cC, cP, cE, cG = c
        QB, Tr = u_func(t)

        QA = self.p["QA"]
        Qr = QA + QB
        V = self.p["V"]

        r1, r2, r3 = self.reaction_rates(c, Tr)

        dcA = (QA * self.p["cA0"] - Qr * cA) / V - r1
        dcB = (QB * self.p["cB0"] - Qr * cB) / V - r1 - r2
        dcC = (-Qr * cC) / V + r1 - r2 - r3
        dcP = (-Qr * cP) / V + r2 - r3
        dcE = (-Qr * cE) / V + r2
        dcG = (-Qr * cG) / V + r3

        return [dcA, dcB, dcC, dcP, dcE, dcG]

    def find_steady_state(self, QB, Tr, c_guess=None):
        """Finds static steady-state concentration vector for given inputs."""
        if c_guess is None:
            c_guess = [5.0, 5.0, 1.0, 1.0, 1.0, 1.0]

        def residual(c):
            return self.odes(0, c, lambda t: (QB, Tr))

        c_ss, _, ier, _ = fsolve(residual, c_guess, full_output=True)
        if ier != 1:
            raise RuntimeError(f"Steady-state solver failed for QB={QB}, Tr={Tr}")
        return c_ss

    def simulate(self, c0, u_trajectory, t_eval):
        """Simulates dynamic response over time using Numerical Integration.
        
        Inputs:
            c0           : Initial concentrations [6,]
            u_trajectory : Callable u_trajectory(t) -> (QB, Tr)
            t_eval       : Array of time steps to evaluate
        Returns:
            c_traj       : Concentation matrix of shape (len(t_eval), 6)
        """
        sol = solve_ivp(
            fun=lambda t, c: self.odes(t, c, u_trajectory),
            t_span=(t_eval[0], t_eval[-1]),
            y0=c0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        return sol.y.T  # Transpose to shape (N_samples, 6)