import numpy as np
from numpy.linalg import inv, norm
import pygame
from pygame.locals import *
import sys
from math import sin, cos, pi, tan
from spring import spring

############# functions ##############


def G(y, t):
    x1_d, x2_d, x1, x2 = y[0], y[1], y[2], y[3]
    r1 = np.array([l1+x1, x2])
    r2 = np.array([x1, x2+l2])

    er1 = r1/norm(r1)
    er2 = r2/norm(r2)

    D1 = norm(r1) - l1
    D2 = norm(r2) - l2

    F = -k1*D1*er1 - k2*D2*er2

    x1_dd = np.dot(F, e1)/m
    x2_dd = np.dot(F, e2)/m

    return np.array([x1_dd, x2_dd, x1_d, x2_d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)

    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def r(P):
    r1 = np.array([P[0] - P1[0], P[1] - P1[1]])
    r2 = np.array([P[0] - P2[0], P[1] - P2[1]])
    er1, er2 = r1/norm(r1), r2/norm(r2)

    return r1, er1, r2, er2


def location(x1, x2):
    x = scale*x1 + eqilb[0]
    y = -scale*x2 + eqilb[1]

    return (int(x), int(y))

############# update and render ##################


def update(position):
    mass.update(position)

    s1.update(P1, position)
    s2.update(P2, position)


def render(position):
    if prev_point:
        pygame.draw.line(trace, LT_BLUE, prev_point, position, 2)

    screen.fill(WHITE)
    if is_tracing:
        screen.blit(trace, (0, 0))

    f1.render()
    f2.render()

    s1.render()
    s2.render()
    mass.render()
    return position

    ############# Mass and Spring Class ##############


class Mass():

    def __init__(self, color, position, diameter):
        self.color = color
        self.pos = position
        self.dia = diameter

    def render(self):
        pygame.draw.circle(screen, self.color, self.pos, self.dia)

    def update(self, position):
        self.pos = position


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


############# Display Parameters ##################
w, h = 1024, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LT_BLUE = (100, 100, 100)
BLUE = (0, 0, 255)
is_tracing = True

screen = pygame.display.set_mode((w, h))
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
m = 8.0
k1 = 10.0
k2 = 10.0
g = 9.81
l1 = 1.0
l2 = 1.0
t = 0.0
dt = 0.05

running = True
prev_point = None
scale = 200

# prelimenary parameters
eqilb = (500, 200)
e1 = np.array([1, 0])
e2 = np.array([0, 1])
P1 = (eqilb[0]-scale*l1, eqilb[1])
P2 = (eqilb[0], eqilb[1]+scale*l2)

# initial condition
y = np.array([0.0, 0.0, 1.0, 0.3])

# Defining mass and spring
mass = Mass(BLUE, eqilb, 15)
f1 = Mass(BLACK, (int(P1[0]), int(P1[1])), 8)
f2 = Mass(BLACK, (int(P2[0]), int(P2[1])), 8)
s1 = Spring(BLACK, P1, eqilb, 15, 30, 50, 50)
s2 = Spring(BLACK, P2, eqilb, 15, 30, 50, 50)

pygame.font.init()
myfont = pygame.font.SysFont('Comic San MS', 38)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_t:
                is_tracing = not(is_tracing)
            if event.key == K_c:
                trace.fill(WHITE)

    pos = location(y[2], y[3])

    update(pos)
    prev_point = render(pos)

    time_string = 'Time: {} seconds'.format(round(t, 1))
    text = myfont.render(time_string, False, (0, 0, 0))
    screen.blit(text, (10, 10))

    t += dt
    y = y + RK4_step(y, t, dt)

    clock.tick(60)
    pygame.display.update()
