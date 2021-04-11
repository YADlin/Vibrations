# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


def F(t):
    F = np.array([0.0, 0.0])
    if t < 15:
        F[0] = F0*np.cos(omega*t)
    else:
        F[0] = 0
    return F


def G(y, t):
    return np.dot(inv_A, (F(t) - np.dot(B, y)))


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)
    # return dt*G(y, t)
    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def Total_Energy(y):
    T = 0.5*m*y[0]**2
    V = 0.5*k*y[1]**2
    return T + V


# Variables
m = 2.0
k = 2.0
c = 0.0     # critical damping = 2*sqrt(m*k) = 4
F0 = 0.0
dt = 0.1
omega = 1.0   # forcing frequency
time = np.arange(0.0, 40.0, dt)

# initial state
y = np.array([0, 1])     # [ Velocity, displacement]
A = np.array([[m, 0], [0, 1]])
B = np.array([[c, k], [-1, 0]])
inv_A = inv(A)

# placeholders
Y = []
Force = []
ToEn = []
# Time stepping solution
for t in time:

    y = y + RK4_step(y,  t, dt)
    Y.append(y[1])
    Force.append(F(t)[0])
    ToEn.append(Total_Energy(y))
    print(Total_Energy(y))

# Plot the results
plt.plot(time, Y, label="Displacement")
plt.plot(time, Force, label="Force")
plt.plot(time, ToEn, label="Total Energy")
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
print("Critical damping:", 2*np.sqrt(m*k))
print("Natural frequency:", np.sqrt(k/m))
