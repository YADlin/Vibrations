# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from math import sin, cos, pi
# EOM


def G(y, t):
    a1d, a2d = y[0], y[1]
    a1, a2 = y[2], y[3]
    m11, m12 = (m1 + m2)*l1, m2*l2*cos(a1-a2)
    m21, m22 = l1*cos(a1-a2), l2
    m = np.array([[m11, m12], [m21, m22]])

    f1 = -m2*l2*(a2d**2)*sin(a1-a2) - (m1+m2)*g*sin(a1)
    f2 = l1*a1d**2*sin(a1-a2)-g*sin(a2)
    f = np.array([f1, f2])

    accel = np.dot(inv(m), f)
    return np.array([accel[0], accel[1], a1d, a2d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)
    # return dt*G(y, t)
    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def Total_Energy(y):
    T = 0.5*(m1+m2)*(l1**2)*(y[0]**2) + 0.5*m2 * \
        y[1]**2 + m2*l1*l2*y[0]*y[1]*cos(y[2]-y[3])
    V = -(m1 + m2)*g*l1*cos(y[2]) - m2*g*l2*cos(y[3])
    return T + V


# Variables
m1, m2 = 2.0, 1.0
l1, l2 = 1.0, 2.0
g = 9.81
dt = 0.1
time = np.arange(0.0, 10.0, dt)

# initial state
# [ Velocity vector, displacement vector]=4x1 vector
y = np.array([0, 0, 0, 1.0])


# placeholders
Y1 = []
Y2 = []
ToEn = []
# Time stepping solution
for t in time:

    y = y + RK4_step(y,  t, dt)
    Y1.append(y[2])
    Y2.append(y[3])
    ToEn.append(Total_Energy(y))

# Plot the results
# plt.plot(time, Y1, label="mass1")
# plt.plot(time, Y2, label="mass2")
plt.plot(time, ToEn, label="Total Energy")
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
# print("Critical damping:", 2*np.sqrt(m*k))
# print("Natural frequency:", np.sqrt(k/m))
