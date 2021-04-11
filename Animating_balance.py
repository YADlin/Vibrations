import numpy as np
from numpy.linalg import inv
import pygame
from pygame.locals import *
import sys
from math import sin, cos, pi


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


def update(cart_point, end_point):
    cart.update(cart_point)
    pendulm.update(cart_point, end_point)


def render():
    screen.fill(WHITE)

    cart.render()
    pendulm.render()


class Cart():

    def __init__(self, position, color, width, height):
        self.pos = position
        self.color = color
        self.w = width
        self.h = height

        self.left = self.pos[0] - self.w/2
        self.top = self.pos[1] - self.h/2

    def render(self):
        pygame.draw.rect(screen, self.color,
                         (self.left, self.top, self.w, self.h))

    def update(self, position):
        self.pos = position
        self.left = self.pos[0] - self.w/2
        self.top = self.pos[1] - self.h/2


class Pend():

    def __init__(self, rot_point, position, color, length, thickness):
        self.rp = rot_point
        self.end = position
        self.color = color
        self.l = length
        self.t = thickness

    def render(self):
        pygame.draw.line(screen, self.color, self.rp, self.end, self.t)
        pygame.draw.circle(screen, self.color, (int(
            self.end[0]), int(self.end[1])), 4*self.t)

    def update(self, rot_point, position):
        self.end = position
        self.rp = rot_point


w, h = 1200, 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((w, h))
screen.fill(WHITE)
clock = pygame.time.Clock()

# parameters
m1 = 100.0
m2 = 100.0
g = 9.81
l = 2.0
dt = 0.09
dof = 2
scale = 100
t = 0

# Initial condition
y = np.zeros((2*dof, 1))


ini_vel_x, ini_vel_th = 0.0, 0.5
ini_pos_x, ini_pos_th = 0.0, 0.0

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

running = True

center = (600, 240)

'''for visualization it is better to keep the height of masses same and change the its lenght to display a difference in mass'''


cart = Cart(center, RED, 70, 50)
pendulm = Pend(center, (0, 0), BLACK, 60, 5)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    cart_point = (center[0] + scale*y[2], center[1])
    end_point = (center[0] + scale*(y[2]+(l*sin(y[3]))),
                 center[1] + (scale*l*cos(y[3])))
    update(cart_point, end_point)
    render()
    t += dt
   # updating matrics
    M[0, 1] = m2*l*cos(y[3])
    M[1, 0] = m2*l*cos(y[3])
    C[0, 1] = -m2*l*sin(y[3])*y[1]
    B, inv_A = update_mat(A, B, M, C)

    # print(A)
    th = y[3]
    f[1] = forcing_func(th)
    y = y + RK4_step(inv_A, B, y, t, dt)

    clock.tick(60)
    pygame.display.update()
