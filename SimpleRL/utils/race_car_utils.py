#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:39:19 2021

@author: hemerson
"""

import numpy as np
import math

# TESTING
import matplotlib.pyplot as plt

# Try using this to generate a race track:
# https://github.com/juangallostra/procedural-tracks

# Used to compute the convex hull
# https://startupnextdoor.com/computing-convex-hull-in-python/

# Used for BSpline interpolation and evaluation:
# https://github.com/XuejiaoYuan/BSpline

def generate_track():
    pass

import pygame, sys
import random 

# Screen dimensions
WIDTH = 800 
HEIGHT = 600

###
# Drawing
###
TITLE = 'Procedural Race Track'

START_TILE_HEIGHT = 10
START_TILE_WIDTH = 10

# Colors
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
BLUE = [0, 0, 255]
GRASS_GREEN = [58, 156, 53]
GREY = [186, 182, 168]

KERB_PLACEMENT_X_CORRECTION = 5
KERB_PLACEMENT_Y_CORRECTION = 4
KERB_POINT_ANGLE_OFFSET = 5
STEP_TO_NEXT_KERB_POINT = 4

CHECKPOINT_POINT_ANGLE_OFFSET = 3
CHECKPOINT_MARGIN = 5

TRACK_POINT_ANGLE_OFFSET = 3

###
# Track parameters
###

# Boundaries for the numbers of points that will be randomly 
# generated to define the initial polygon used to build the track
MIN_POINTS = 20
MAX_POINTS = 30

SPLINE_POINTS = 1000

# Margin between screen limits and any of the points that shape the
# initial polygon
MARGIN = 50
# minimum distance between points that form the track skeleton
MIN_DISTANCE = 100 # 20
# Maximum midpoint displacement for points placed after obtaining the initial polygon
MAX_DISPLACEMENT = 80
# Track difficulty
DIFFICULTY = 0.1
# min distance between two points that are part of thr track skeleton
DISTANCE_BETWEEN_POINTS = 20
# Maximum corner allowed angle
MAX_ANGLE = 90

# Angle boundaries used to determine the corners that will have a kerb
MIN_KERB_ANGLE = 20
MAX_KERB_ANGLE = 90

TRACK_WIDTH = 40

FULL_CORNER_NUM_POINTS = 17

###
# Game parameters
###
N_CHECKPOINTS = 10


####
## logical functions
####
def random_points(min_p=MIN_POINTS, max_p=MAX_POINTS, margin=MARGIN, min_distance=MIN_DISTANCE):
    
    # get number of points
    pointCount = random.randrange(min_p, max_p + 1, 1)
    
    points = []
    for i in range(pointCount):
        
        # get x and y of points
        x = random.randrange(margin, WIDTH - margin + 1, 1)
        y = random.randrange(margin, HEIGHT -margin + 1, 1)
        
        # remove any points which are less than the minimum distance
        distances = list(filter(lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
        
        # append x and y 
        if len(distances) == 0:
            points.append((x, y))
            
    return np.array(points)

def convex_hull(points):
    
    hull_points = []
    
    # set the starting point
    start = points[0, :]
    
    # set the minimum x
    min_x = start[0]
    
    # cycle through all the points
    for p in points[1:, :]:
        if p[0] < min_x:
            min_x = p[0]
            start = p
    
    # set the starting point
    point = start    
    hull_points.append(start)
    
    # cycle through points from start to finish
    far_point = None
    while not (far_point==start).all():        
        
        p1 = None
        for p in points:
            
            # if point is current point
            if (p==point).all():
                continue
            
            # if not set p1 as point
            else:
                p1 = p
                break
        
        # set the far point as p1
        far_point = p1
        
        # cycle through the points again
        for p2 in points:
            
            # p2 is current point or equal to p1
            if (p2==point).all() or (p2==p1).all():
                continue
            
            # get the direction between p1 and p2
            else:
                direction = get_orientation(point, far_point, p2)
                if direction > 0:                    
                    far_point = p2
                    
        hull_points.append(far_point)
        point = far_point
        
    return np.array(hull_points[:-1])

def shape_track(track_points, difficulty=DIFFICULTY, max_displacement=MAX_DISPLACEMENT, margin=MARGIN, track_width=TRACK_WIDTH):
    
    # create a list of zero pairs twice as long as hull points
    track_set = [[0,0] for i in range(len(track_points) * 2)] 
        
    # create track vectors and displacements
    for i in range(len(track_points)):
        
        # get a random displacement
        displacement = math.pow(random.random(), difficulty) * max_displacement
        
        # multiply magnitude my random unit vector
        disp = [displacement * i for i in make_rand_vector(2)]
        
        # set first index to track point        
        track_set[i * 2] = track_points[i]
        
        # set second index to current track point + mean of current and next track point + magnitude of dispalcement
        track_set[i * 2 + 1][0] = int((track_points[i][0] + track_points[(i + 1) % len(track_points)][0]) / 2 + disp[0])
        track_set[i * 2 + 1][1] = int((track_points[i][1] + track_points[(i + 1) % len(track_points)][1]) / 2 + disp[1])
       
    
    # ensure angles are suitable and points are sufficiently far apart
    for i in range(3):
        track_set = fix_angles(track_set)
        track_set = push_points_apart(track_set)
        
    # ensure all the points are within the screen limits
    final_set = []
    for point in track_set:
        
        # if outside x dimension including track width
        if point[0] - track_width < margin:
            point[0] = margin + track_width            
        elif point[0] + track_width > (WIDTH - margin):
            point[0] = (WIDTH - margin) - track_width
        
        # if outside y dimension including track width        
        if point[1] - track_width < margin:
            point[1] = margin + track_width
        elif point[1] + track_width > HEIGHT - margin:
            point[1] = (HEIGHT - margin) - track_width
            
        final_set.append(point)
     
    # make it a complete loop
    final_set.append(final_set[0])
        
    return final_set

def make_rand_vector(dims):
    
    # create a random dims-dimensional vector
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def fix_angles(points, max_angle=MAX_ANGLE):
    
    for i in range(len(points)):
        
        # set the previous point
        if i > 0:
            prev_point = i - 1
        
        # set the previous point to the end of the list
        else:
            prev_point = len(points)-1
        
        # set the next point
        next_point = (i+1) % len(points)
        
        # get the difference between current point and previous point
        px = points[i][0] - points[prev_point][0]
        py = points[i][1] - points[prev_point][1]
        
        # get the distance
        pl = math.sqrt(px*px + py*py)
        
        # get norm of difference
        px /= pl
        py /= pl
        
        # compute the difference between the current point and the next point
        nx = -(points[i][0] - points[next_point][0])
        ny = -(points[i][1] - points[next_point][1])
        
        # get the distance
        nl = math.sqrt(nx*nx + ny*ny)
        
        # norm the distances
        nx /= nl
        ny /= nl  
        
        # calculate the angle of the corner
        a = math.atan2(px * ny - py * nx, px * nx + py * ny)
        
        # if angle is suitable continue
        if (abs(math.degrees(a)) <= max_angle):
            continue
        
        # get the difference beyond the max angle
        diff = math.radians(max_angle * math.copysign(1,a)) - a
        
        # recalculate a new position which fits constraint
        c = math.cos(diff)
        s = math.sin(diff)
        new_x = (nx * c - ny * s) * nl
        new_y = (nx * s + ny * c) * nl
        
        # update the points
        points[next_point][0] = int(points[i][0] + new_x)
        points[next_point][1] = int(points[i][1] + new_y)
        
    return points


def push_points_apart(points, distance=DISTANCE_BETWEEN_POINTS):
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            
            # get distance between 2 points 
            p_distance =  math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            
            # if beyond the max distance
            if p_distance < distance:
                
                # get coord differences
                dx = points[j][0] - points[i][0]  
                dy = points[j][1] - points[i][1]  
                dl = math.sqrt(dx*dx + dy*dy)  
                
                # norm the distances
                dx /= dl  
                dy /= dl
                
                # get the difference and update the coords
                dif = distance - dl
                dx *= dif  
                dy *= dif  
                
                # updated coords in array
                points[j][0] = int(points[j][0] + dx)
                points[j][1] = int(points[j][1] + dy)  
                points[i][0] = int(points[i][0] - dx)  
                points[i][1] = int(points[i][1] - dy) 
                    
    return points

def get_checkpoints(track_points, n_checkpoints=N_CHECKPOINTS):
    # get step between checkpoints
    checkpoint_step = len(track_points) // n_checkpoints
    # get checkpoint track points
    checkpoints = []
    for i in range(N_CHECKPOINTS):
        index = i * checkpoint_step
        checkpoints.append(track_points[index])
    return checkpoints

####
## drawing functions
####


def draw_track(surface, color, points):
    
    radius = TRACK_WIDTH // 2
    
    # draw track
    chunk_dimensions = (radius * 2, radius * 2)
    for point in points:
        blit_pos = (point[0] - radius, point[1] - radius)
        track_chunk = pygame.Surface(chunk_dimensions, pygame.SRCALPHA)
        pygame.draw.circle(track_chunk, color, (radius, radius), radius)
        surface.blit(track_chunk, blit_pos)
        
    starting_grid = draw_starting_grid(radius*2)
    
    # rotate and place starting grid
    offset = TRACK_POINT_ANGLE_OFFSET
    vec_p = [points[offset][1] - points[0][1], -(points[offset][0] - points[0][0])]
    n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
    
    # compute angle
    angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
    rot_grid = pygame.transform.rotate(starting_grid, -angle)
    start_pos = (points[0][0] - math.copysign(1, n_vec_p[0])*n_vec_p[0] * radius, points[0][1] - math.copysign(1, n_vec_p[1])*n_vec_p[1] * radius)    
    surface.blit(rot_grid, start_pos)

def draw_starting_grid(track_width):    
    starting_grid = pygame.Surface((track_width, START_TILE_HEIGHT ), pygame.SRCALPHA)
    return starting_grid

def draw_checkpoint(track_surface, points, checkpoint, checkpoint_idx):
    # given the main point of a checkpoint, compute and draw the checkpoint box
    margin = CHECKPOINT_MARGIN
    radius = TRACK_WIDTH // 2 + margin
    offset = CHECKPOINT_POINT_ANGLE_OFFSET
    check_index = points.index(checkpoint)
    vec_p = [points[check_index + offset][1] - points[check_index][1], -(points[check_index+offset][0] - points[check_index][0])]
    n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
    # compute angle
    angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
    
    # set checkpoint colour for the start
    if checkpoint_idx == 0: colour = RED
    else: colour = BLUE
    
    # draw checkpoint    
    checkpoint = draw_rectangle((radius*2, 5), colour, line_thickness=1, fill=False)    
    rot_checkpoint = pygame.transform.rotate(checkpoint, -angle)
    check_pos = (points[check_index][0] - math.copysign(1, n_vec_p[0])*n_vec_p[0] * radius, points[check_index][1] - math.copysign(1, n_vec_p[1])*n_vec_p[1] * radius)    
    track_surface.blit(rot_checkpoint, check_pos)

def draw_rectangle(dimensions, color, line_thickness=1, fill=False):
    filled = line_thickness
    if fill:
        filled = 0
    rect_surf = pygame.Surface(dimensions, pygame.SRCALPHA)
    pygame.draw.rect(rect_surf, color, (0, 0, dimensions[0], dimensions[1]), filled)
    return rect_surf

def get_orientation(origin, p1, p2):
    '''
    Returns the orientation of the Point p1 with regards to Point p2 using origin.
    Negative if p1 is clockwise of p2.
    :param p1:
    :param p2:
    :return: integer
    '''
    difference = (
        ((p2[0] - origin[0]) * (p1[1] - origin[1]))
        - ((p1[0] - origin[0]) * (p2[1] - origin[1]))
    )

    return difference    


def smooth_track(track_points):    
    
    # get x and y in the appropriate format
    x, y = zip(*track_points)
    x, y = list(x), list(y)
    
    k, H = 3, len(x) - 1
    
    print('D_n: {}'.format(len(x)))
    print('H: {}'.format(H))
    print('k: {}'.format(k))    
        
    # combine the lists
    combined_list = [x, y]
        
    p_centripetal = centripetal(len(x), combined_list)
    knot = knot_vector(p_centripetal, k, len(x))
    P_control = curve_approximation(D=combined_list, N=len(x), H=H, k=k, param=p_centripetal, knot=knot)

    # P_control = combined_list
    
    p_piece = np.linspace(0, 1, SPLINE_POINTS)
    p_centripetal_new = centripetal(H, P_control)
    knot_new = knot_vector(p_centripetal_new, k, H)
    P_piece = curve(P_control, H, k, p_piece, knot_new) 
    
    return [(P_piece[0][idx], P_piece[1][idx]) for idx in range(len(P_piece[0]))]      
    

####
## Main function
####
def main(draw_checkpoints_in_track=True):
    
    # initialise pygame
    pygame.init()
    
    # set the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    background_color = GRASS_GREEN
    screen.fill(background_color)
        
    # set the seed
    # random.seed(11)

    # generate a set of random points
    points = random_points()
    
    # calculate the convex hull of the random points
    hull_points = convex_hull(points)    
    
    # get the points of the track
    track_points = shape_track(hull_points)
    
    # smooth the track points
    f_points = smooth_track(track_points)
    
    """    
    x, y = zip(*f_points)    
    plt.plot(y, x)
    x_1, y_1 = zip(*track_points)  
    plt.scatter(y_1, x_1)
    plt.show()        
    sladhjsasanda
    """
            
    # draw the actual track (road, kerbs, starting grid)
    draw_track(screen, GREY, f_points)
    
    # draw checkpoints
    checkpoints = get_checkpoints(f_points)

    if draw_checkpoints_in_track:
        for checkpoint_idx, checkpoint in enumerate(checkpoints):
            draw_checkpoint(screen, f_points, checkpoint, checkpoint_idx)
    
    pygame.display.set_caption(TITLE)
    while True: # main loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()
        
        
def curve(P, N, k, param, knot):
    '''
    Calculate B-spline curve.
    :param P: Control points
    :param N: the number of control points
    :param k: degree
    :param param: parameters
    :param knot: knot vector
    :return: data point on the b-spline curve
    '''
    Nik = np.zeros((len(param), N))

    for i in range(len(param)):
        for j in range(N):
            Nik[i][j] = BaseFunction(j, k+1, param[i], knot)
    Nik[len(param)-1][N - 1] = 1
    D = []
    for i in range(len(P)):
        D.append(np.dot(Nik, P[i]).tolist())
    return D

def BaseFunction(i, k, u, knot):
    '''
    Calculate base function using Cox-deBoor function.
    :param i: index of base function
    :param k: order (degree + 1)
    :param u: parameter
    :param knot: knot vector
    :return: base function
    '''
    Nik_u = 0
    if k == 1:
        if u >= knot[i] and u < knot[i + 1]:
            Nik_u = 1.0
        else:
            Nik_u = 0.0
    else:
        length1 = knot[i + k - 1] - knot[i]
        length2 = knot[i + k] - knot[i + 1]
        if not length1 and not length2:
            Nik_u = 0
        elif not length1:
            Nik_u = (knot[i + k] - u) / length2 * BaseFunction(i + 1, k - 1, u, knot)
        elif not length2:
            Nik_u = (u - knot[i]) / length1 * BaseFunction(i, k - 1, u, knot)
        else:
            Nik_u = (u - knot[i]) / length1 * BaseFunction(i, k - 1, u, knot) + \
                    (knot[i + k] - u) / length2 * BaseFunction(i + 1, k - 1, u, knot)
    return Nik_u

def centripetal(n, P):
    '''
    Calculate parameters using the centripetal method.
    :param n: the number of data points
    :param P: data points
    :return: parameters
    '''
    a = 0.5
    parameters = np.zeros((1, n))
    for i in range(1, n):
        dis = 0
        
        for j in range(len(P)):
            dis = dis + (P[j][i]-P[j][i-1]) ** 2
            
        dis = np.sqrt(dis)
        parameters[0][i] = parameters[0][i-1] + np.power(dis, a)
        
    for i in range(1, n):
        parameters[0][i] = parameters[0][i] / parameters[0][n-1]
        
    return parameters[0]

def knot_vector(param, k, N):
    '''
    Generate knot vector.
    :param param: parameters
    :param k: degree
    :param N: the number of data points
    :return: knot vector
    '''
    m = N + k
    knot = np.zeros((1, m+1))
    for i in range(k + 1):
        knot[0][i] = 0
        
    for i in range(m - k, m + 1):
        knot[0][i] = 1
        
    for i in range(k + 1, m - k):
        for j in range(i - k, i):
            knot[0][i] = knot[0][i] + param[j]
        knot[0][i] = knot[0][i] / k
        
    return knot[0]

def curve_approximation(D, N, H, k, param, knot):
    '''
    Given a set of N data points, D0, D1, ..., Dn, a degree k,
    and a number H, where N > H > k >= 1, find a B-spline curve
    of degree k defined by H control points that satisfies the
    following conditions:
        1. this curve contains the first and last data points;
        2. this curve approximates the data polygon in the sense
        of least square;
    :param D: data points (N x 2)
    :param H: the number of control points
    :param k: degree
    :param param: parameters
    :param knot: knot vector
    :return: control points (H x 2)
    '''
    P = []
    if H >= N or H <= k:
        print('Parameter H is out of range')
        return P

    for idx in range(len(D)):
        P_ = np.zeros((1, H))
        P_[0][0] = D[idx][0]
        P_[0][H-1] = D[idx][N-1]
        Qk = np.zeros((N - 2, 1))
        Nik = np.zeros((N, H))
        
        for i in range(N):
            for j in range(H):
                Nik[i][j] = BaseFunction(j, k + 1, param[i], knot)
        # print(Nik)

        for j in range(1, N - 1):
            Qk[j - 1] = D[idx][j] - Nik[j][0] * P_[0][0] - Nik[j][H - 1] * P_[0][H - 1]

        N_part = Nik[1: N - 1, 1: H - 1]
        Q = np.dot(N_part.transpose(), Qk)
        M = np.dot(np.transpose(N_part), N_part)
        P_[0][1:H - 1] = np.dot(np.linalg.inv(M), Q).transpose()
        P.append(P_.tolist()[0])

    return P
        
if __name__ == '__main__':
    main(draw_checkpoints_in_track=True)

