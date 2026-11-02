# ============================================================
# Import Modules
import numpy as np

# ============================================================
# Physical Model
p = {
    "k10": 9.9594e6,      # dm3/(mol*min)
    "k20": 8.66124e9,     # dm3/(mol*min)
    "k30": 9.9594e6,      # dm3/(mol*min)
    "E1": 6666.7,         # K
    "E2": 8333.3,         # K
    "E3": 11111.0,        # K
    "cA0": 10.0,          # mol/dm3
    "cB0": 10.0,          # mol/dm3
    "V": 2105.0,          # dm3
    "QA": 112.35,         # dm3/min
}



def reaction_rates(c, Tr):
    """
    Computes the three reaction rates r1, r2, r3.

    Inputs:
        c  : array of 6 floats [cA, cB, cC, cP, cE, cG] in mol/dm3
        Tr : float, reactor temperature in K

    Output:
        r1, r2, r3 : tuple of 3 floats, reaction rates in mol/(dm3*min)
    """
    cA, cB, cC, cP, cE, cG = c

    k1 = p["k10"] * np.exp(-p["E1"] / Tr)
    k2 = p["k20"] * np.exp(-p["E2"] / Tr)
    k3 = p["k30"] * np.exp(-p["E3"] / Tr)

    r1 = k1 * cA * cB
    r2 = k2 * cB * cC
    r3 = k3 * cC * cP

    return r1, r2, r3


def wo_reactor_odes(t, c, u):
    """
    ODE system of the Williams-Otto CSTR.

    Inputs:
        t : float, current time
        c : array of 6 floats [cA, cB, cC, cP, cE, cG] in mol/dm3
        u : (QB, Tr)
            QB : float, flow rate of B in dm3/min
            Tr : float, reactor temperature in K

    Output:
        list of 6 floats [dcA, dcB, dcC, dcP, dcE, dcG] in mol/(dm3*min)
    """
    cA, cB, cC, cP, cE, cG = c
    QB, Tr = u

    QA = p["QA"]
    Qr = QA + QB
    V = p["V"]

    r1, r2, r3 = reaction_rates(c, Tr)

    dcA = (QA * p["cA0"] - Qr * cA) / V - r1
    dcB = (QB * p["cB0"] - Qr * cB) / V - r1 - r2
    dcC = (-Qr * cC) / V + r1 - r2 - r3
    dcP = (-Qr * cP) / V + r2 - r3
    dcE = (-Qr * cE) / V + r2
    dcG = (-Qr * cG) / V + r3

    return [dcA, dcB, dcC, dcP, dcE, dcG]
