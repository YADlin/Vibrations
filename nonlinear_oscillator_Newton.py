import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# functions


def G(y, t):
    x1_d, x2_d, x1, x2 = y[0], y[1], y[2], y[3]
    r1, er1, r2, er2 = r((x1, x2))

    D1 = mag(r1) - l1
    D2 = mag(r2) - l2

    F = -k1*D1*er1 - k2*D2*er2

    x1_dd = np.dot(F, e1)
    x2_dd = np.dot(F, e2)

    return np.array([x1_dd, x2_dd, x1_d, x2_d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)

    return (1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def mag(r):
    return np.sqrt(r[0]**2 + r[1]**2)


# def G(y, t):
#     x1_d, x2_d, x1, x2 = y[0], y[1], y[2], y[3]
#     x1_dd = np.dot(F1 + F2, e1)
#     x2_dd = np.dot(F1 + F2, e1)

#     return np.array([x1_dd, x2_dd, x1_d, x2_d])


def r(P):
    r1 = (P[0] - P1[0], P[1] - P1[1])
    r2 = (P[0] - P2[0], P[1] - P2[1])
    er1, er2 = r1/mag(r1), r2/mag(r2)

    return r1, er1, r2, er2


# parameters
m = 1.0
k1 = 1.0
k2 = 1.0
g = 9.81
l1 = 100.0
l2 = 100.0

dt = 0.01
time = np.arange(0.0, 10.0, dt)

# prelimenary parameters
e1, e2 = np.array([1, 0]), np.array([0, 1])
P1, P2 = (-l1, 0), (0, -l2)

# initial condition
y = np.array([0.0, 0.0, 1.1, 0.1])

# value holders for plot
X = []
Y = []

for t in time:

    y = y + dt*RK4_step(y, t, dt)
    X.append(y[2])
    Y.append(y[3])

plt.plot(time, X, label='X')
plt.plot(time, Y, label='Y')
plt.legend()
plt.show()
