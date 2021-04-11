import numpy as np
from numpy.linalg import inv
import pygame
from pygame.locals import *
import sys
from math import sin, cos, pi


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


def update(a1, a2):
    scale = 100
    x1 = l1*scale*sin(a1) + offset[0]
    y1 = l1*scale*cos(a1) + offset[1]
    x2 = x1 + l2*scale*sin(a2)
    y2 = y1 + l2*scale*cos(a2)

    return (x1, y1), (x2, y2)


def render(point1, point2):
    scale = 10
    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])

    if prev_point:
        xp, yp = prev_point[0], prev_point[1]
        pygame.draw.line(trace, LT_BLUE, (xp, yp), (x2, y2), 3)

    screen.fill(WHITE)
    screen.blit(trace, (0, 0))

    pygame.draw.line(screen, BLACK, offset, (x1, y1), 5)
    pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 5)
    pygame.draw.circle(screen, BLACK, offset, 8)
    pygame.draw.circle(screen, RED, (x1, y1), int(m1*scale))
    pygame.draw.circle(screen, BLUE, (x2, y2), int(m2*scale))

    return (x2, y2)


w, h = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LT_BLUE = (230, 230, 255)
BLUE = (0, 0, 255)
offset = (400, 200)

screen = pygame.display.set_mode((w, h))
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
running = True
m1, m2 = 3.0, 2.0
l1, l2 = 1.5, 2
a1, a2 = 2.0, 0.5
g = 9.81
prev_point = None

t = 0.0
dt = 1/60
y = np.array([0, 0, a1, a2])

pygame.font.init()
myfont = pygame.font.SysFont('Comic San MS', 38)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    point1, point2 = update(y[2], y[3])
    prev_point = render(point1, point2)

    time_string = 'Time: {} seconds'.format(round(t, 1))
    text = myfont.render(time_string, False, (0, 0, 0))
    screen.blit(text, (10, 10))

    t += dt
    y = y + RK4_step(y, t, dt)

    clock.tick(60)
    pygame.display.update()
