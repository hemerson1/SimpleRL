#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:12:23 2021

@author: hemerson
"""

import numpy as np

def generate_scenario(GRID_SIZE=8, PLAYER_PARAM=1, ENEMY_PARAM=2, TERRAIN_PARAM=3, EMPTY_PARAM=0): 
    
    # TODO: update the parameter information to be consistent with the style  
    
    """
    creates a random grid with players, enemies and terrain fro the laser_tag game.
    
    Parameters:
    ----------
    GRID_SIZE
        how large a grid should the game be set in
        
    Return:
    ------
     map_grid 
         A 2d numpy array with values corresponding to player, enemy
         and terrain obstacles         
    """
    
    PERCENT_TERRAIN = 0.3 # what is the percentage coverage of the grid with terrain
   
    # generate a numbered grid and get an array of the edge elements
    array_grid = np.arange(GRID_SIZE * GRID_SIZE).reshape(GRID_SIZE, GRID_SIZE)
    array_edges = np.concatenate([array_grid[0,:-1], array_grid[:-1,-1], array_grid[-1,::-1], array_grid[-2:0:-1,0]])
    
    # randomly select an edge element and assign position to player and opposite position to enemys
    start_num = array_edges[np.random.choice(len(array_edges), size=1, replace=False)]
    player_pos = np.array([start_num[0] / GRID_SIZE, start_num[0] % GRID_SIZE], dtype=np.int32)
    enemy_pos = np.absolute(player_pos - np.array([GRID_SIZE - 1, GRID_SIZE - 1]))
    
    # get the positions which are not at the edge and make PERCENT_TERRAIN of tiles terrain
    array_centre = np.setdiff1d(array_grid.flatten(), array_edges)
    num_terrain_squares = int(len(array_centre) * PERCENT_TERRAIN)
    terrain_squares = array_centre[np.random.choice(len(array_centre), size=num_terrain_squares, replace=False)]
    
    get_pos = lambda x: np.array([x/GRID_SIZE, x % GRID_SIZE], dtype=np.int32)
    terrain_pos = get_pos(terrain_squares)
    
    # create a grid and assign numbers representing player, enemy and terrain
    # 0, 1, 2, 3 = vacant, player, enemy, terrain
    map_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    map_grid[player_pos[0], player_pos[1]] = PLAYER_PARAM
    map_grid[enemy_pos[0], enemy_pos[1]] = ENEMY_PARAM
    map_grid[terrain_pos[0, :], terrain_pos[1, :]] = TERRAIN_PARAM
    
    return map_grid        