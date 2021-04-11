import numpy as np
from numpy.linalg import inv
import pygame
from pygame.locals import *
import sys
from math import sin, cos, pi, tan


def G(y, t):
    th_d, phi_d, th, phi = y[0], y[1], y[2], y[3]

    th_dd = phi_d**2 * sin(th) * cos(th) - g/l * sin(th)
    phi_dd = -2.0 * th_d * phi_d/tan(th)

    return np.array([th_dd, phi_dd, th_d, phi_d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + np.dot(k1, (dt/2)), t+(dt/2))
    k3 = G(y + np.dot(k2, (dt/2)), t+(dt/2))
    k4 = G(y + np.dot(k3, dt), t+dt)

    return dt*(1/6)*(k1 + 2*k2 + 2 * k3 + k4)


def update(a1, a2):
    # here only x, y coordinates of the sherical pendulum is considered since pygame allows only 2D
    x = l*scale*sin(a1)*cos(a2) + offset[0]
    y = l*scale*cos(a1) + offset[1]
    z = l*scale*sin(a1)*sin(a2)

    return (int(x), int(y), int(z))


def render(point):
    x, y, z = int(point[0]), int(point[1]), int(point[2])
    z_scale = (2 - z/(l*scale))*5.0

    if prev_point:
        pygame.draw.line(trace, LT_BLUE, prev_point, (x, y), int(z_scale*0.2))

    screen.fill(WHITE)
    if is_tracing:
        screen.blit(trace, (0, 0))

    pygame.draw.line(screen, BLACK, offset, (x, y), int(z_scale*0.3))
    pygame.draw.circle(screen, BLACK, offset, 8)
    pygame.draw.circle(screen, RED, (x, y), int(m*z_scale))

    return (x, y)


w, h = 1024, 768
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LT_BLUE = (230, 230, 255)
BLUE = (0, 0, 255)
offset = (w//2, h//3)
is_tracing = True

screen = pygame.display.set_mode((w, h))
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
running = True
m = 3.0
l = 4.5
theta_D0, phi_D0 = 0.0, 0.5
theta0, phi0 = 1.5, 0.0
g = 9.81
prev_point = None
scale = 100
t = 0.0
dt = 0.02
y = np.array([theta_D0, phi_D0, theta0, phi0])

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

    point = update(y[2], y[3])
    prev_point = render(point)

    time_string = 'Time: {} seconds'.format(round(t, 1))
    text = myfont.render(time_string, False, (0, 0, 0))
    screen.blit(text, (10, 10))

    t += dt
    y = y + RK4_step(y, t, dt)

    clock.tick(60)
    pygame.display.update()
