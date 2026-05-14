import matplotlib.pyplot as plt
import numpy as np

from simulation.simulation import Conductor, ElectricField, IonGenerator, IonSimulation

field = ElectricField((160,90),0.1,(16,9))
cond1 = Conductor(-1,(15,25),(40,50))

cond2 = Conductor(1,(135,145),(40,50))

field.add(cond1)
field.add(cond2)
field.compute()

gen = IonGenerator(1.67e-27, 1.6e-19, np.array([0,0.5]),(70,90),(15,20))
n_ions = 10
sim = IonSimulation(field,gen,n_ions)
for i in range(20):
    print(f"====== {i} ======")
    print(sim.x_vel)
    print(sim.y_vel)
    sim.step(1e-5)


X = np.linspace(0,160,num=160)*0.1
Y = np.linspace(0,90,num=90)*0.1
skip = 5

fig, ax = plt.subplots(figsize=(16,9))

ax.pcolormesh(
    X, Y,
    field.V[1:-1,1:-1].T,
    cmap='RdBu_r', shading='nearest'
)

ax.quiver(X[::skip], Y[::skip],
              field.Ex[::skip, ::skip].T, field.Ey[::skip, ::skip].T,
              alpha=0.6, scale=None, width=0.003)


plt.show()
