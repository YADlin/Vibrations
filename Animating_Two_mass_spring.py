import numpy as np
from numpy.linalg import inv
import pygame
from pygame.locals import *
import sys
from math import sin, cos, pi
from spring import spring


def f(t):
    return np.array([0, F0 * sin(w*t), 0, 0])


def F(y, t):
    return dt*np.dot(inv_A, (f(t) - np.dot(B, y)))


def RK4_step(y, t, dt):
    k1 = F(y, t)
    k2 = F(y + 0.5*k1*dt, t + 0.5 * dt)
    k3 = F(y + 0.5*k2*dt, t + 0.5 * dt)
    k4 = F(y + k3 * dt, t + dt)
    return dt*(1/6)*(k1 + 2 * k2 + 2 * k3 + k4)


def update(point1, point2, point3):
    mass1.update(point2)
    mass2.update(point3)

    s1.update(point1, point2)
    s2.update(point2, point3)


def render():
    screen.fill(WHITE)

    render_statics()

    s1.render()
    s2.render()

    mass1.render()
    mass2.render()


def render_statics():
    # render floor and wall
    pygame.draw.line(
        screen, BLACK, (30, point1[1] + 45), (10e3, point1[1] + 45), 10)
    pygame.draw.line(screen, BLACK, (30, point1[1]-50), (30, point1[1]+50), 10)

    pygame.draw.line(
        screen, GRAY, (point1[0]+offset1, point1[1]+55), (point1[0]+offset1, point1[1]+70), 3)
    pygame.draw.line(
        screen, GRAY, (point1[0]+offset2, point1[1]+55), (point1[0]+offset2, point1[1]+70), 3)


class Mass():

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


class Spring():
    def __init__(self, color, start, end, nodes, width, lead1, lead2):
        self.start = start
        self.end = end
        self.nodes = nodes
        self.width = width
        self.lead1 = lead1
        self.lead2 = lead2
        self.weight = 3
        self.color = color

    def update(self, start, end):
        self.start = start
        self.end = end

        self.x, self.y, self.p1, self.p2 = spring(
            self.start, self.end, self.nodes, self.width, self.lead1, self.lead2)
        self.p1 = (int(self.p1[0]), int(self.p1[1]))
        self.p2 = (int(self.p2[0]), int(self.p2[1]))

    def render(self):
        pygame.draw.line(screen, self.color, self.start, self.p1, self.weight)
        prev_point = self.p1

        for point in zip(self.x, self.y):
            pygame.draw.line(screen, self.color,
                             prev_point, point, self.weight)
            prev_point = point

        pygame.draw.line(screen, self.color, self.p2, self.end, self.weight)


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
F0 = 10.0  # N
w = 1.0  # rads/s

m1, m2 = 2.0, 1.0  # Kg
k1, k2 = 3.0, 2.0  # N/m
dof = 2
scale = 100

t = 0.0
dt = 0.2
ini_vel1, ini_vel2 = 0, 0
ini_pos1, ini_pos2 = 1, 0
y = 100*np.array([ini_vel1, ini_vel2, ini_pos1, ini_pos2])

K = np.array([[k1+k2, -k2], [-k2, k2]])
M = np.array([[m1, 0], [0, m2]])
I = np.eye(dof)

A = np.zeros((2*dof, 2*dof))
B = np.zeros((2*dof, 2*dof))


A[0:dof, 0:dof] = M
A[dof:, dof:] = I
B[0:dof, dof:] = K
B[dof:, 0:dof] = -I

inv_A = inv(A)

running = True

point1 = (35, 300)
offset1 = 300
offset2 = 550

'''for visualization it is better to keep the height of masses same and change the its lenght to display a difference in mass'''

mass1 = Mass((150, 100), RED, 120, 80)
mass2 = Mass((270, 100), BLUE, 80, 80)

s1 = Spring(BLACK, (0, 0), (0, 0), 20, 50, 20, 70)
s2 = Spring(BLACK, (0, 0), (0, 0), 15, 30, 70, 50)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    point2 = (point1[0] + offset1 + y[2], point1[1])
    point3 = (point1[0] + offset2 + y[3], point1[1])
    update(point1, point2, point3)
    render()
    t += dt
    y = y + RK4_step(y, t, dt)

    clock.tick(60)
    pygame.display.update()
