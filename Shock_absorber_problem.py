from math import sqrt, exp, pi
import numpy as np
import matplotlib.pyplot as plt

# Given data
m = 2.0e3  # Kg
k = 60.0e3  # N/m
c = 20.0e3  # Ns/m
v0 = 10  # m/s

t = []
x = []
z = []
cc = []
for k in np.arange(55000, 100001, 5000):
    Cc = 2*sqrt(m*k)
    zeta = c/Cc
    Wn = sqrt(k/m)
    Wd = Wn*sqrt(1-zeta**2)

    tmax = pi/(2*Wd)
    xmax = exp(-zeta*Wn*tmax)*v0/Wd
    t.append(tmax)
    x.append(xmax)
    z.append(zeta)
    cc.append(Cc)

k = [i for i in np.arange(55000, 100001, 5000)]
# Plot results
plt.title('Shock Absorber Optimization')
plt.plot(k, t, label='t max')
plt.plot(k, x, label='x max')
plt.plot(k, z, label='Zeta')
# plt.plot(k, cc, label='crit. Damping')
plt.xlim(min(k), max(k))
plt.legend()
plt.xlabel('Stiffness K (N/m)')
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.grid(True)
plt.show()
