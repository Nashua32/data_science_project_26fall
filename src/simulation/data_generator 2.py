import sys
sys.path.append("src/simulation")
import numpy as np
import matplotlib.pyplot as plt
from sampling import make_aprbs
from config import SEED,DT
from simulator import step

N_steps=20000
rng = np.random.default_rng(SEED)
U = make_aprbs(rng, n_steps=N_steps, hold_range=(1, 250))

y0 = [10,10,0,0,0,0]
y = [y0]

for i in range(N_steps):
    y.append(step(y0,U[i,:],DT))
    y0 = y[-1]


plt.figure()
plt.plot(range(N_steps),U)
plt.xlabel('Steps in DT'), plt.ylabel('LHS Inputs'), plt.legend(['QB','Tr'])

plt.figure()
plt.plot(range(N_steps+1),y)
plt.xlabel('Steps in DT'), plt.ylabel('Simulated Outputs'), plt.legend(['cA','cB','cC','cP','cE','cG'])

plt.show()

np.savez("data/aprbs_data.npz", U=U, y=y)