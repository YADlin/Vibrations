import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Setup parameters
F0 = 5.0  # N
w = 47.75747487  # rads/s
m = 1.0  # Kg
k = 1000.0  # N/m
dof = 3
time_step = 1.0e-4
end_time = 4.0

# Setup matrices
K = np.array([[3*k, -k, -k], [-k, k, 0], [-k, 0, k]])
M = np.array([[2*m, 0, 0], [0, m, 0], [0, 0, m]])
I = np.eye(dof)

A = np.zeros((2*dof, 2*dof))
B = np.zeros((2*dof, 2*dof))
Y = np.zeros((2*dof, 1))
F = np.zeros((2*dof, 1))

A[0:dof, 0:dof] = M
A[dof:, dof:] = I
B[0:dof, dof:] = K
B[dof:, 0:dof] = -I

# Find natural frquency and mode shapes
evals, evecs = eigh(K, M)
frequencies = np.sqrt(evals)
print(frequencies)
print(evecs)

inv_A = inv(A)
force = []
X1 = []
X2 = []
X3 = []

# Numerically integrate the EOMs
for t in np.arange(0, end_time, time_step):
    F[1] = F0*sin(w*t)
    Y_new = Y + time_step*np.dot(inv_A, (F - np.dot(B, Y)))
    Y = Y_new
    force.extend(F[1])
    X1.extend(Y[dof])
    X2.extend(Y[dof+1])
    X3.extend(Y[dof+2])
t = [i for i in np.arange(0, end_time, time_step)]

# Plot results
plt.plot(t, X1, label="X1")
plt.plot(t, X2, label="X2")
plt.plot(t, X3, label="X3")
plt.grid(True)
plt.legend()
plt.ylabel("Displacement (m)")
plt.xlabel("time (s)")
plt.title("Response curve")
plt.show()
plt.plot(t, force, label="Forcing function")
plt.legend()
plt.ylabel("Forcing function (N)")
plt.xlabel("time (s)")
plt.title("force input")
plt.grid(True)
plt.show()
