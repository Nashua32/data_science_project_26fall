"""
Comments: basic RF performs a lot worse than LinReg (probably because it averages out a lot more peaks), but still better than the trivial model

with 200 steps, it took like 30 minutes to train the GBR

TODO: save the models
    - maybe leave out GBR
"""

import sys
sys.path.append("src/models")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from helper_functions import make_windows_multistep

# load data
d = np.load("data/aprbs_data.npz")
U = d["U"]
y = d["y"]

H = 200      # Predict 200 time steps ahead
NY = 6
X, w = make_windows_multistep(y, U, L=0, H=H) # X = [y(t) | u(t) … u(t+49)], w = [y(t+1) … y(t+50)]

# Split into training and testing sets
n_tr = int(0.8 * len(X))
Xtrain, Xtest = X[:n_tr - H], X[n_tr:]
wtrain, wtest = w[:n_tr - H], w[n_tr:]

# Preprocess the data
scaler = StandardScaler()
Xtrain_s, Xtest_s = scaler.fit_transform(Xtrain), scaler.transform(Xtest)

# Fit Models
models = {
    "LinReg": LinearRegression(),
    "RF": RandomForestRegressor(n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Tuned_RF": RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,          # Allow deeper trees to capture sharper dynamics
        min_samples_leaf=2,    # Smaller leaves reduce heavy smoothing/averaging
        n_jobs=-1, 
        random_state=42
    ),
    "GBR": MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=50, 
            max_depth=4, 
            learning_rate=0.1, 
            random_state=42
        )
    )
}

preds = {"Trivial": np.tile(Xtest[:, :NY], H)}         # repeat the current state as a trivial prediction   

for name, m in models.items():
    print(f"Training {name}...")
    preds[name] = m.fit(Xtrain_s, wtrain).predict(Xtest_s)

# NRMSE per step
rng = wtest.reshape(-1, NY).max(0) - wtest.reshape(-1, NY).min(0)
def nrmse_k(p):
    e = (wtest - p).reshape(-1, H, NY)
    return (np.sqrt((e**2).mean(axis=0)) / rng).mean(axis=1)

# Plot NRMSE curves for all models side-by-side
plt.figure(figsize=(10, 5))
for name, p in preds.items():
    plt.plot(range(1, H + 1), nrmse_k(p), label=name, linewidth=2)
plt.xlabel("Horizon step k")
plt.ylabel("avg NRMSE")
plt.title("Multi-Step NRMSE Comparison: Linear Regression vs. Random Forest")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


L = 0
names = ["cA", "cB", "cC", "cP", "cE", "cG"]

def plot_window(i, j, past=30):
    """i: index in testset, j: state (0-5)."""
    t0 = L + n_tr + i                                   # start time of plot
    tt = np.arange(max(t0 - past, 0), t0 + H + 1)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                           gridspec_kw={"height_ratios": [3, 1.5]})
    ax[0].plot(tt, y[tt, j], "k", label="real state")
    for name, p in preds.items():
        ax[0].plot(t0 + np.arange(1, H + 1), p[i].reshape(H, NY)[:, j], ".", label=f"{name} pred")
    ax[0].set_ylabel(names[j]); ax[0].legend(); ax[0].grid()
    ax[1].step(tt[:-1], U[tt[:-1]], where="post")
    ax[1].set_ylabel("Inputs"); ax[1].set_xlabel("Time Step"); ax[1].legend(["QB", "Tr"]); ax[1].grid()
    for a in ax: a.axvline(t0, ls="--", c="gray")       # prediction start line
    plt.tight_layout(); plt.show()

plot_window(i=50, j=3)