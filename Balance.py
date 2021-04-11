import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import sin, cos, pi


# Functions
def forcing_func(th):
    return (-m2*g*l*sin(th))


def F(inv_A, B, y, t):
    return dt*np.dot(inv_A, (f - np.dot(B, y)))


def RK4_step(inv_A, B, y, t, dt):
    k1 = F(inv_A, B, y, t)
    k2 = F(inv_A, B, y + 0.5*k1*dt, t + 0.5 * dt)
    k3 = F(inv_A, B, y + 0.5*k2*dt, t + 0.5 * dt)
    k4 = F(inv_A, B, y + k3 * dt, t + dt)
    return dt*(1/6)*(k1 + 2 * k2 + 2 * k3 + k4)


def update_mat(A, B, M, C):
    A[0:dof, 0:dof] = M
    B[0:dof, 0:dof] = C
    inv_A = inv(A)
    return B, inv_A


# parameters
m1 = 10.0 
m2 = 100000.0
g = 9.81
l = 2.0
dt = 0.05
dof = 2

# Initial condition
X = []
theta = []
time = np.arange(0.0, 10.0, dt)
y = np.zeros((2*dof, 1))


ini_vel_x, ini_vel_th = 15.0, 12.0
ini_pos_x, ini_pos_th = 0.0, 1.0

y[0], y[1], y[2], y[3] = ini_vel_x, ini_vel_th, ini_pos_x, ini_pos_th


# Setup matrices
M = np.array([[m1 + m2, m2*l*cos(y[3])], [m2*l*cos(y[3]), 2*m2*l*l]])
C = np.array([[0.0, -m2*l*sin(y[3])*y[1]], [0.0, 0.0]])
I = np.eye(dof)

A = np.zeros((2*dof, 2*dof))
B = np.zeros((2*dof, 2*dof))
f = np.zeros((2*dof, 1))
A[dof:, dof:] = I
B[dof:, 0:dof] = -I

# solving
for t in time:

    # updating matrics
    M[0, 1] = m2*l*cos(y[3])
    M[1, 0] = m2*l*cos(y[3])
    C[0, 1] = -m2*l*sin(y[3])*y[1]
    B, inv_A = update_mat(A, B, M, C)

    # print(A)
    th = y[3]
    f[1] = forcing_func(th)
    y = y + RK4_step(inv_A, B, y, t, dt)
    X.append(y[2])
    theta.append(y[3])

# plotting
plt.plot(time, X, label='X position of thr cart')
plt.plot(time, theta, label='theta of the pendulum')
plt.xlabel('time (s)')
plt.ylabel('x, theta')
plt.legend()
plt.grid(True)
plt.show()
