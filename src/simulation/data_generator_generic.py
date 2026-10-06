#src/simulation/data_generator_generic.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.simulation.william_otto_reactor_model import WilliamsOttoReactor

reactor = WilliamsOttoReactor() #create rector

t_end = 60*24*365*4                         #Simulation time in min (currently 4 years)
dt = 10                                     #timestep for simulation
t_eval = np.arange(0.0, t_end + dt, dt)     #evaluation times

Tr_mean = 400           #mean of sine function in °K
Tr_amplitude = 20.0     #amplitude of sine function in °K
Tr_period = 525600      #periode of sine function in min

def temperature_input(t):
    """sin function with its highest point in summer and lowest point in winter """
    return Tr_mean + Tr_amplitude * np.sin(
        2.0 * np.pi * t / Tr_period
        -1/2 * np.pi
    )

Qb_mean = 224           #mean of QB in dm3/min
Qb_amplitude1 = 9.0     #amplitude of sine function 1 in dm3/min
Qb_amplitude2 = 5.0     #amplitude of sine function 2 in dm3/min
Qb_amplitude3 = 8.0     #amplitude of sine function 3 in dm3/min
Qb_amplitude4 = 7.0     #amplitude of sine function 4 in dm3/min
Qb_period1 = 5*60*24    #periode of sine function 1 in min
Qb_period2 = 13*60*24   #periode of sine function 2 in min
Qb_period3 = 67*60*24    #periode of sine function 3 in min
Qb_period4 = 122*60*24   #periode of sine function 4 in min

def QB_input(t):
    """
    some kind of fluctuations with the Qb input
    two sin functions with different frequencies to simulate some randomness
    """
    return (Qb_mean +
            Qb_amplitude1 * np.sin(
                2.0 * np.pi * t / Qb_period1)
            + Qb_amplitude2 * np.sin(
                2.0 * np.pi * t / Qb_period2)
            + Qb_amplitude3 * np.sin(
            2.0 * np.pi * t / Qb_period3)
            + Qb_amplitude4 * np.sin(
            2.0 * np.pi * t / Qb_period4)
            )


def u_trajectory(t):
    """combine massflow B and temperature Tr in one function"""
    QB = QB_input(t)
    Tr = temperature_input(t)

    return QB, Tr

#find an initial steady state

QB0 = QB_input(0.0)
Tr0 = temperature_input(0.0)

c0 = reactor.find_steady_state(
    QB=QB0,
    Tr=Tr0
)

print("Initial steady-state concentrations:")
print(c0)

#simulate using steady state start conditions and defined trajectorys

c_traj = reactor.simulate(
    c0=c0,
    u_trajectory=u_trajectory,
    t_eval=t_eval
)

#variables for plotting and the final data csv

species = ["A", "B", "C", "P", "E", "G"]
QB_traj = np.array([QB_input(t) for t in t_eval])
Tr_traj = np.array([temperature_input(t) for t in t_eval])

#saving the data in a csv

df = pd.DataFrame({
    "time_min": t_eval,
    "cA": c_traj[:, 0],
    "cB": c_traj[:, 1],
    "cC": c_traj[:, 2],
    "cP": c_traj[:, 3],
    "cE": c_traj[:, 4],
    "cG": c_traj[:, 5],
    "QB": QB_traj,
    "Tr": Tr_traj,
})
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"
file_path = data_dir / "data_generic.csv"
df.to_csv(file_path, index=False)

#plotting inputs

#Reactor temperature over time
plt.figure()
plt.plot(t_eval,Tr_traj)
plt.xlabel("Time [min]")
plt.ylabel("Reactor temperature [K]")
plt.grid()
plt.show()

#flow rate B over time
plt.figure()
plt.step(
    t_eval,
    QB_traj,
    where="post"
)
plt.xlabel("Time [min]")
plt.ylabel("B flow rate QB [dm3/min]")
plt.grid()
plt.show()

#plotting results

plt.figure()

for i, name in enumerate(species):
    plt.plot(
        t_eval,
        c_traj[:, i],
        label=name
    )

plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/dm3]")
plt.legend()
plt.grid()

plt.show()
