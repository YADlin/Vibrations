import numpy as np
from math import sin, cos
from scipy.linalg import eigh
from numpy.linalg import inv
import matplotlib.pyplot as plt


def forcing_func(F0, w, t):
    return F0 * cos(w*t)


def F(y, t):
    return dt*np.dot(inv_A, (f - np.dot(B, y)))


def RK4_step(y, t, dt):
    k1 = F(y, t)
    k2 = F(y + 0.5*k1*dt, t + 0.5 * dt)
    k3 = F(y + 0.5*k2*dt, t + 0.5 * dt)
    k4 = F(y + k3 * dt, t + dt)
    return dt*(1/6)*(k1 + 2 * k2 + 2 * k3 + k4)


# Setup parameters
F0 = 5.0  # N
w = 1  # rads/s

m1, m2 = 1.0, 1.0  # Kg
k1, k2 = 1000.0, 1000.0  # N/m
dof = 2

dt = 0.02
tf = 360.0
time = np.arange(0.0, tf, dt)

ini_vel1, ini_vel2 = 1.0, -1.0
ini_pos1, ini_pos2 = 0.0, 0.0

# Setup matrices
K = np.array([[k1+k2, -k2], [-k2, k2]])
M = np.array([[m1, 0], [0, m2]])
I = np.eye(dof)

A = np.zeros((2*dof, 2*dof))
B = np.zeros((2*dof, 2*dof))
y = np.zeros((2*dof, 1))
f = np.zeros((2*dof, 1))

A[0:dof, 0:dof] = M
A[dof:, dof:] = I
B[0:dof, dof:] = K
B[dof:, 0:dof] = -I

# Find natural frquency and mode shapes
evals, evecs = eigh(K, M)
frequencies = np.sqrt(evals)
print('Natural frequencies: ', frequencies)
print('Mode shapes: ', evecs[:, 0], evecs[:, 1])

inv_A = inv(A)
force = []
X1 = []
X2 = []

# initial values
y[0], y[1], y[2], y[3] = ini_vel1, ini_vel2, ini_pos1, ini_pos2

# Numerically integrate the EOMs
for t in time:
    f[1] = forcing_func(F0, w, t)
    y = y + RK4_step(y, t, dt)
    force.extend(f[1])
    X1.extend(y[dof])
    X2.extend(y[dof+1])

# Plot results
plt.plot(time, X1, label="X1")
plt.plot(time, X2, label="X2")

plt.grid(True)
plt.legend()

plt.ylabel("Displacement (m)")
plt.xlabel("time (s)")
plt.title("Response curve")
plt.show()

plt.plot(time, force, label="Forcing function")

plt.legend()
plt.ylabel("Forcing function (N)")
plt.xlabel("time (s)")
plt.title("force input")
plt.grid(True)
plt.show()
