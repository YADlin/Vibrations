import numpy as np
from numpy.linalg import inv
import pygame
from pygame.locals import *
import sys
from math import sin, cos, pi, tan
from spring import spring


def G(y, t):
    r_d, th_d, r, th = y[0], y[1], y[2], y[3]

    r_dd = (l0 + r)*th_d**2 - k/m * r + g * cos(th)
    th_dd = -(2/(l0 + r))*r_d*th_d - g/(l0 + r)*sin(th)

    return np.array([r_dd, th_dd, r_d, th_d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)

    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def update(position):
    bob.update(position)
    s.update(pointO, position)


def render(position):
    if prev_point:
        pygame.draw.line(trace, LT_BLUE, prev_point, position, 2)

    screen.fill(WHITE)
    if is_tracing:
        screen.blit(trace, (0, 0))
    s.render()
    bob.render()

    return position


def position(r, th):
    l = l0 + r
    xp = scale*l*sin(th) + pointO[0]
    yp = scale*l*cos(th) + pointO[1]

    return (int(xp), int(yp))


class Bob():

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


w, h = 1024, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LT_BLUE = (100, 100, 100)
BLUE = (0, 0, 255)
pointO = (w//2, 100)
is_tracing = True

screen = pygame.display.set_mode((w, h))
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
running = True
m = 6.0
k = 100.0
l0 = 3.5
r_d, th_d = 0.0, 0.0
r, th = 1, 1
g = 9.81
prev_point = None
scale = 100
t = 0.0
dt = 0.02
y = np.array([r_d, th_d, r, th])

pygame.font.init()
myfont = pygame.font.SysFont('Comic San MS', 38)

start = position(y[2]+pointO[0], y[3]+pointO[1])
bob = Bob(RED, start, 25)
s = Spring(BLACK, pointO, start, 15, 50, 50, 50)

pos = start
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_t:
                is_tracing = not(is_tracing)
            if event.key == K_c:
                trace.fill(WHITE)

    pos = position(y[2], y[3])

    update(pos)
    prev_point = render(pos)

    time_string = 'Time: {} seconds'.format(round(t, 1))
    text = myfont.render(time_string, False, (0, 0, 0))
    screen.blit(text, (10, 10))

    t += dt
    y = y + RK4_step(y, t, dt)

    clock.tick(60)
    pygame.display.update()
