#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 21:57:14 2021

@author: hemerson
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from SimpleRL.envs.base import environment_base 
from SimpleRL.utils.laser_tag_utils import generate_scenario

class laser_tag_env(environment_base):
    
    # TODO: update the notation
    # TODO: add code for adversary -> planning based method
    
    def __init__(self, render=False, seed=None, mode="default"):
        
        # Ensure the mode for the environment is valid
        valid_mode = ["default", "adversarial"]
        mode_error = "mode is not valid for this environment, " \
            + "please select one of the following {} ".format(valid_mode)
        assert mode in valid_mode, mode_error
        
        self.render = render 
        self.seed = seed 
        self.mode = mode
        self.ACTION_DIM = 1 # how many actions are made each turn?
        self.ACTION_NUM = np.array([8], dtype=np.int32) # how many values are there per action?  
        
        # Set environmetal parameters
        np.random.seed(self.seed)  
        self.SHOT_RANGE = 5 # how many squares will a bullet travel?
        self.GRID_SIZE= 8
        self.POSITIVE_REWARD = +1
        self.NEGATIVE_REWARD = -1
        self.PLAYER_PARAM = 1
        self.ENEMY_PARAM = 2
        self.TERRAIN_PARAM = 3
        self.EMPTY_PARAM = 0
        self.DISPLAY_PAUSE = 0.25
        
        # Initialise the environment
        self.bullet_path = None # for displaying the bullets path
        self.bullet_hits = None # for displaying successful shots
        
        # Reset the environment parameters
        self.reset()
        
        # Intialise the display
        if self.render: 
            self.image, self.figure, self.axis = self.init_display(self.grid_map)        
    
    def reset(self):
        
        # reset the map parameters
        self.current_player = self.PLAYER_PARAM
        self.opposing_player = self.ENEMY_PARAM
        self.game_outcome = None
        self.grid_map = generate_scenario(
                GRID_SIZE = self.GRID_SIZE
                )
        
        return self.grid_map.flatten()
        
    def step(self, action=None):
        
        # check the action input is valid (i.e. np.int32 of valid range)        
        self.check_discrete_input(action)
        
        # 8 actions
        # 0, 1, 2, 3 = move_left, move_up, move_right, move_down
        # 4, 5, 6, 7 = shoot_left, shoot_up, shoot_right, shoot_down
        moves = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        reward, done, info = 0, False, {"outcome" : None}    
        
        # reset the bullet path and the hit arrays
        self.bullet_path = np.empty((0, 2), dtype=np.int32)
        self.bullet_hits = np.empty((0, 2), dtype=np.int32)
        
        # get the player's location
        player_pos = np.asarray(np.where(self.grid_map == self.current_player)).flatten() 
        
        # if the action is a move
        if action < 4:
            
            # get the player move direction
            chosen_move = moves[action[0], :]
            
            # get the grid value of the move           
            final_player_pos = player_pos + chosen_move      
            
            # assign the new player position
            row, col = final_player_pos          
            
            # check the square is within the grid
            if (row >= 0 and row < self.GRID_SIZE) and (col >= 0 and col < self.GRID_SIZE):
            
                # check the square is empty
                if self.grid_map[row, col] == 0:
                    self.grid_map[row, col] = self.current_player
                    self.grid_map[player_pos[0], player_pos[1]] = 0
                
        # if the action is a shot
        else:
                        
            # get the player shot direction
            chosen_move = moves[action[0] - 4, :]
                        
            for i in range(self.SHOT_RANGE):
                
                # get the current bullet position
                bullet_vec = chosen_move * (i + 1)
                bullet_pos = player_pos + bullet_vec                
                row, col = bullet_pos
                
                # is the bullet out of bounds?
                if (row < 0 or row >= self.GRID_SIZE) or (col < 0 or col >= self.GRID_SIZE):
                    break
                
                # has the bullet hit terrain?
                if self.grid_map[row, col] == 3:
                    break
                
                # has the bullet hit the enemy?
                if self.grid_map[row, col] == self.opposing_player:      
                    
                    # remove the enemy
                    self.grid_map[row, col] = 0
                    self.bullet_hits = np.append(self.bullet_hits, bullet_pos.reshape(1, -1), axis=0)
                    
                    # set the parameters to end the game
                    reward = self.POSITIVE_REWARD
                    done = True
                    info["outcome"] = self.current_player
                    break
                
                # add the bullet path to the array for displaying
                self.bullet_path = np.append(self.bullet_path, bullet_pos.reshape(1, -1), axis=0)
        
        # switch the current player to the opposite player 
        temp_opposing_player = self.opposing_player 
        self.opposing_player = self.current_player
        self.current_player = temp_opposing_player
        
        # display the map
        if self.render:
            self.display()
                 
        return self.grid_map.flatten(), reward, done, info
    
    def init_display(self, grid_map):
        
        # define the colour map for the grid
        cmap = colors.ListedColormap(['#5ba01b', '#3a87e2', '#c7484e', '#5c5c5c'])
        
        figure, axis = plt.subplots(1,1)
        image = axis.imshow(grid_map, cmap=cmap)
        
        # get the player and enemy positions
        player_row, player_col = np.asarray(np.where(self.grid_map == self.PLAYER_PARAM)).flatten() 
        enemy_row, enemy_col =  np.asarray(np.where(self.grid_map == self.ENEMY_PARAM)).flatten()         
        
        # label their positions
        axis.text(player_col, player_row, 'P', ha="center", va="center", color="white")
        axis.text(enemy_col, enemy_row, 'E', ha="center", va="center", color="white")            
        
        # adjust plot to figure area
        figure.tight_layout()        
        
        return image, figure, axis
    
    def display(self):   
        
        # update the image
        new_array = self.grid_map
        self.image.set_data(new_array)
        
        # remove all the previous text labels
        self.axis.texts = []
        
        # get the player and label their positions
        player_pos = np.where(self.grid_map == self.PLAYER_PARAM)
        if len(player_pos[0]) > 0:        
            player_row, player_col = np.asarray(player_pos).flatten() 
            self.axis.text(player_col, player_row, 'P', ha="center", va="center", color="white")
                        
        # get the enemy and label their positions
        enemy_pos = np.where(self.grid_map == self.ENEMY_PARAM)        
        if len(enemy_pos[0]) > 0:
            enemy_row, enemy_col = np.asarray(enemy_pos).flatten()  
            self.axis.text(enemy_col, enemy_row, 'E', ha="center", va="center", color="white")
        
        # mark the bullet's path on the grid            
        for sqr in range(self.bullet_path.shape[0]):
            self.axis.text(self.bullet_path[sqr, 1], self.bullet_path[sqr, 0], '*', ha="center", va="center", color="black")
                
        # mark a hit on the grid
        if self.bullet_hits.shape[0] > 0:
            self.axis.text(self.bullet_hits[0, 1], self.bullet_hits[0, 0], 'X', ha="center", va="center", color="black")
        
        # draw the new image
        self.figure.canvas.draw_idle()
        plt.pause(self.DISPLAY_PAUSE)   
        
    def close_display(self):
        plt.close()
        
if __name__ == "__main__":       
    
    # intialise the environment
    env = laser_tag_env(render=True)
    
    # reset the state
    state, done = env.reset(), False
    counter = 0
    
    # run the training loop
    while not done and counter < 100:
        
        action = env.sample_discrete_action()            
        next_state, reward, done, info = env.step(action=action)
        
        # print the winner
        if done: 
            print('Player {} wins'.format(info["outcome"]))
            env.close_display()
            
        state = next_state
        counter += 1
            