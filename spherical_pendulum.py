# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from math import sin, cos, pi, tan


def G(y, t):
    th_d, phi_d, th, phi = y[0], y[1], y[2], y[3]

    th_dd = phi_d*phi_d*sin(th)*cos(th) - g*sin(th)/l
    phi_dd = -2.0*th_d*phi_d/tan(th)

    return np.array([th_dd, phi_dd, th_d, phi_d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)
    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def Total_Energy(y):
    T = 0.5*m*l*l*(y[0]**2 + (y[1]**2)*sin(y[2])**2)
    V = -m*g*l*cos(y[2])
    return T + V


# Variables
m = 3.0
l = 1.5
a1, a2 = 1, 0
g = 9.81

t = 0.0
dt = 0.01
time = np.arange(t, 5.0, dt)

# initial state
# [ Velocity vector, displacement vector]=4x1 vector
y = np.array([0, 1, a1, a2])


# placeholders
THETA = []
PHI = []
ToEn = []
# Time stepping solution
for t in time:

    y = y + RK4_step(y,  t, dt)
    THETA.append(y[2])
    PHI.append(y[3])
    ToEn.append(Total_Energy(y))

# Plot the results
plt.plot(time, THETA, label="theta")
plt.plot(time, PHI, label="phi")
# plt.plot(time, ToEn, label="Total Energy")
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
