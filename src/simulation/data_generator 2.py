import sys;from pathlib import Path;sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from sampling import make_aprbs
from config import SEED,DT,H
from simulator import step

# Hyperparameters for data generation
N_steps=10000
hold_range=(1,H)
dwelltime_same=False
rng = np.random.default_rng(SEED)

# Generate APRBS inputs
U = make_aprbs(rng, n_steps=N_steps, hold_range=hold_range, dwelltime_same=dwelltime_same, log_uniform=True)

# Simulate system for generated inputs
y0 = [10,10,0,0,0,0]
y = [y0]
for i in range(N_steps):
    y.append(step(y0,U[i,:],DT))
    y0 = y[-1]

# Plot inputs and outputs
plt.figure()
plt.plot(range(N_steps),U)
plt.xlabel('Steps in DT'), plt.ylabel('LHS Inputs'), plt.legend(['QB','Tr'])
plt.figure()
plt.plot(range(N_steps+1),y)
plt.xlabel('Steps in DT'), plt.ylabel('Simulated Outputs'), plt.legend(['cA','cB','cC','cP','cE','cG'])
plt.show()

# Save data to file
fname = f"data/aprbs_N{N_steps}_hold{hold_range[0]}-{hold_range[1]}_dt{DT:g}min_seed{SEED}.npz"
np.savez(fname, U=U, y=y, N_steps=N_steps, hold_range=np.array(hold_range), DT=DT, seed=SEED, y0=y0)