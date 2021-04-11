# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Variables
m = 2.0
k = 2.0
c = 1.0     # critical damping = 2*sqrt(m*k) = 4
F0 = 1.0
dt = 0.001
omega = 1.0   # forcing frequency
time = np.arange(0.0, 40.0, dt)

# initial state
y = np.array([0, 0])     # [ Velocity, displacement]
A = np.array([[m, 0], [0, 1]])
B = np.array([[c, k], [-1, 0]])
F = np.array([0.0, 0.0])

# placeholders
Y = []
Force = []
ToEn = []
# Time stepping solution
for t in time:
    if t < 15:
        F[0] = F0*np.cos(omega*t)
    else:
        F[0] = 0
    y = y + dt*np.dot(inv(A), (F - np.dot(B, y)))
    Y.append(y[1])
    Force.append(F[0])
    T = 0.5*m*y[0]**2
    V = 0.5*k*y[1]**2
    ToEn.append(T+V)

# Plot the results
t = [i for i in time]
plt.plot(t, Y, label="Displacement")
plt.plot(t, Force, label="Force")
# plt.plot(t,ToEn, label="Total Energy")
plt.grid(True)
plt.legend(loc="upper_right")
plt.show()
print("Critical damping:", 2*np.sqrt(m*k))
print("Natural frequency:", np.sqrt(k/m))
