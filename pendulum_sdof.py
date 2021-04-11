import numpy as np
import matplotlib.pyplot as plt

# Define the equation of motion


def F(y, t):
    return np.array([y[1], -(g/l)*np.sin(y[0])])


def RK4_step(y, t, dt):
    k1 = F(y, t)
    k2 = F(y + 0.5*k1*dt, t + 0.5 * dt)
    k3 = F(y + 0.5*k2*dt, t + 0.5 * dt)
    k4 = F(y + k3 * dt, t + dt)
    return dt*(1/6)*(k1 + 2 * k2 + 2 * k3 + k4)


def Total_Energy(y):
    T = 0.5*m*(l**2)*y[1]**2
    V = m*g*l*(1-np.cos(y[0]))
    return T + V


# Parameters
m = 1
g = 9.81
l = 1.0
dt = 0.001
omega = np.sqrt(g/l)
time = np.arange(0.0, 10.0, dt)
theta = []
theta_l = []
ToEn = []
ToEn_l = []

# Initial conditions
vel = 0.0
th = 130.0
theta0 = np.pi*th/180
# [angular displacement, angular velocity]
y = np.array([np.pi*th/180, np.pi*vel/180])

# Solution to the non linear problem
for t in time:
    y = y + RK4_step(y, t, dt)
    theta.append(y[0])
    ToEn.append(Total_Energy(y))

# Solution to the linear problem
for t in time:
    theta_l.append(theta0*np.cos(omega*t))

# Plot the results
plt.plot(time, theta, label="theta non linear")
plt.plot(time, theta_l, label="theta linear")
plt.plot(time, ToEn, label="Total Energy")
s = "(Initial Angle = " + str(th) + " degrees)"
plt.title('Pendulum Motion :' + s)
plt.grid(True)
plt.xlabel('Time(s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.show()
