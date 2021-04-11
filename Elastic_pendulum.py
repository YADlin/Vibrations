import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# Equation of motion


def G(y, t):
    r_d, th_d, r, th = y[0], y[1], y[2], y[3]

    r_dd = (l0 + r)*th_d**2 - k/m * r + g * cos(th)
    th_dd = -(2/(l0 + r))*r_d*th_d - g/(l0 + r)*sin(th)

    return np.array([r_dd, th_dd, r_d, th_d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)

    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


# Parameters
m = 2.0
k = 100.0
g = 9.81
dof = 2
l0 = 1.0
dt = 0.01


time = np.arange(0.0, 5.0, dt)

# Initial condition
y = np.array([0.0, 0.0, 0.0, 0.1])


# Empty list for plotting
R = []
TH = []

# Solving loop
for t in time:
    y = y + RK4_step(y, t, dt)

    R.append(y[2])
    TH.append(y[3])

# Plot
plt.plot(time, R, label='r')
plt.plot(time, TH, label='Theta')
plt.legend()
plt.show()
