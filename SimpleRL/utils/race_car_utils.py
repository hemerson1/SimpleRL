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

def generate_track():
    pass

import pygame, sys
import random 

# TODO: remove scipy import
from scipy import interpolate

  
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
MIN_DISTANCE = 20
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

def shape_track(track_points, difficulty=DIFFICULTY, max_displacement=MAX_DISPLACEMENT, margin=MARGIN):
    
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
        
        
    for i in range(3):
        track_set = fix_angles(track_set)
        track_set = push_points_apart(track_set)
        
    # push any point outside screen limits back again
    final_set = []
    for point in track_set:
        if point[0] < margin:
            point[0] = margin
        elif point[0] > (WIDTH - margin):
            point[0] = WIDTH - margin
        if point[1] < margin:
            point[1] = margin
        elif point[1] > HEIGHT - margin:
            point[1] = HEIGHT - margin
        final_set.append(point)
        
    return final_set

def make_rand_vector(dims):
    
    # create a random dims-dimensional vector
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def fix_angles(points, max_angle=MAX_ANGLE):
    for i in range(len(points)):
        if i > 0:
            prev_point = i - 1
        else:
            prev_point = len(points)-1
        next_point = (i+1) % len(points)
        px = points[i][0] - points[prev_point][0]
        py = points[i][1] - points[prev_point][1]
        pl = math.sqrt(px*px + py*py)
        px /= pl
        py /= pl
        nx = -(points[i][0] - points[next_point][0])
        ny = -(points[i][1] - points[next_point][1])
        nl = math.sqrt(nx*nx + ny*ny)
        nx /= nl
        ny /= nl  
        a = math.atan2(px * ny - py * nx, px * nx + py * ny)
        if (abs(math.degrees(a)) <= max_angle):
            continue
        diff = math.radians(max_angle * math.copysign(1,a)) - a
        c = math.cos(diff)
        s = math.sin(diff)
        new_x = (nx * c - ny * s) * nl
        new_y = (nx * s + ny * c) * nl
        points[next_point][0] = int(points[i][0] + new_x)
        points[next_point][1] = int(points[i][1] + new_y)
    return points


def push_points_apart(points, distance=DISTANCE_BETWEEN_POINTS):
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p_distance =  math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            if p_distance < distance:
                dx = points[j][0] - points[i][0];  
                dy = points[j][1] - points[i][1];  
                dl = math.sqrt(dx*dx + dy*dy);  
                dx /= dl;  
                dy /= dl;  
                dif = distance - dl;  
                dx *= dif;  
                dy *= dif;  
                points[j][0] = int(points[j][0] + dx);  
                points[j][1] = int(points[j][1] + dy);  
                points[i][0] = int(points[i][0] - dx);  
                points[i][1] = int(points[i][1] - dy);  
    return points

def get_corners_with_kerb(points, min_kerb_angle=MIN_KERB_ANGLE, max_kerb_angle=MAX_KERB_ANGLE):
    require_kerb = []
    for i in range(len(points)):
        if i > 0:
            prev_point = i - 1
        else:
            prev_point = len(points)-1
        next_point = (i+1) % len(points)
        px = points[prev_point][0] - points[i][0]
        py = points[prev_point][1] - points[i][1]
        pl = math.sqrt(px*px + py*py)
        px /= pl
        py /= pl
        nx = points[next_point][0] - points[i][0]
        ny = points[next_point][1] - points[i][1]
        nl = math.sqrt(nx*nx + ny*ny)
        nx /= nl
        ny /= nl 
        a = math.atan(px * ny - py * nx)
        if (min_kerb_angle <= abs(math.degrees(a)) <= max_kerb_angle):
            require_kerb.append(points[i])
    return require_kerb

def smooth_track(track_points):
    x = np.array([p[0] for p in track_points])
    y = np.array([p[1] for p in track_points])

    # append the starting x,y coordinates
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    
    # TODO: replace this interpolate function with a homemade versio

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # evaluate the spline fits for # points evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, SPLINE_POINTS), tck)
    return [(int(xi[i]), int(yi[i])) for i in range(len(xi))]

def get_full_corners(track_points, corners):
    # get full range of points that conform the corner
    offset = FULL_CORNER_NUM_POINTS
    corners_in_track = get_corners_from_kp(track_points, corners)
    # for each corner keypoint in smoothed track, 
    # get the set of points that make the corner.
    # This are the offset previous and offset next points
    f_corners = []
    for corner in corners_in_track:
        # get kp index
        i = track_points.index(corner)
        # build temp list to get set of points
        tmp_track_points = track_points + track_points + track_points
        f_corner = tmp_track_points[i+len(track_points)-1-offset:i+len(track_points)-1+offset]
        f_corners.append(f_corner)
    return f_corners

def get_corners_from_kp(complete_track, corner_kps):
    # for each detected corner find closest point in final track (smoothed track)
    return [find_closest_point(complete_track, corner) for corner in corner_kps]

def find_closest_point(points, keypoint):
    min_dist = None
    closest_point = None
    for p in points:
        dist = math.hypot(p[0]-keypoint[0], p[1]-keypoint[1])
        if min_dist is None or dist < min_dist:
            min_dist = dist
            closest_point = p
    return closest_point

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


def draw_track(surface, color, points, corners):
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
    
    random.seed(0)

    # generate a set of random points
    points = random_points()
    
    # calculate the convex hull of the random points
    hull_points = convex_hull(points)
    
    track_points = shape_track(hull_points)
    corner_points = get_corners_with_kerb(track_points)
    f_points = smooth_track(track_points)
    # get complete corners from keypoints
    corners = get_full_corners(f_points, corner_points)
    # draw the actual track (road, kerbs, starting grid)
    draw_track(screen, GREY, f_points, corners)
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

if __name__ == '__main__':
    main(draw_checkpoints_in_track=True)

