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
        self.ACTION_NUM = np.array([25], dtype=np.int32) # how many values are there per action?  
        
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
        
    def step(self, player_action=None):
        
        # check the action input is valid (i.e. np.int32 of valid range)        
        self.check_discrete_input(player_action)        
        
        # update the grid according to the player move
        reward, done, info = self.update_grid(action=player_action)
        
        # display the map
        if self.render:
            self.display()
          
        # check if the episode has terminated
        if not done:
        
            if self.mode == "default":
                computer_action = self.get_computer_action()
                reward, done, info = self.update_grid(action=computer_action)
                
            elif self.mode == "adversarial":
                
                #TODO: add adversarial implementation in which two seperate agents can
                #      be trained                
                pass      
            
        # display the map
        if self.render:
            self.display()
                 
        return self.grid_map.flatten(), reward, done, info
    
    def update_grid(self, action):
        
        # TODO: add compatibility for multi-discrete action agents
        
        # 25 actions --------        
        #      NO|LM|UM|RM|DM
        #   NO|00|01|02|03|04
        #   LS|05|06|07|08|09
        #   US|10|11|12|13|14
        #   RS|15|15|16|17|18
        #   DS|20|21|22|23|24
        # -------------------
        
        # 0, 1, 2, 3, 4 = no_move, move_left, move_up, move_right, move_down
        move_action = action % 5
        
        # 0, 1, 2, 3, 4 = no_shot, shoot_left, shoot_up, shoot_right, shoot_down
        shot_action = np.floor_divide(action, 5)
        
        move_direction = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        reward, done, info = 0, False, {"outcome" : None}    
        
        # reset the bullet path and the hit arrays
        self.bullet_path = np.empty((0, 2), dtype=np.int32)
        self.bullet_hits = np.empty((0, 2), dtype=np.int32)
        
        # get the player's location
        player_pos = np.asarray(np.where(self.grid_map == self.current_player)).flatten() 
        
        # if the action is a move
        if move_action > 0:
            
            # get the player move direction
            chosen_move = move_direction[move_action[0] - 1, :]
            
            # get the grid value of the move           
            final_player_pos = player_pos + chosen_move   
                        
            valid_move = self.is_move_valid(final_player_pos)
            
            # Update the grid if the move is valid
            if valid_move:
                
                # assign players to correct square
                self.grid_map[final_player_pos[0], final_player_pos[1]] = self.current_player
                self.grid_map[player_pos[0], player_pos[1]] = 0
                
                # update the player position for the shot
                player_pos = final_player_pos            
                
        # if the action is a shot
        if shot_action > 0:
                        
            # get the player shot direction
            chosen_move = move_direction[shot_action[0] - 1, :]
            
            # get the outcome of the shot
            reward, done, info = self.get_shot_trajectory(chosen_move, player_pos, self.grid_map)
                                
        # switch the current player to the opposite player 
        temp_opposing_player = self.opposing_player 
        self.opposing_player = self.current_player
        self.current_player = temp_opposing_player
        
        return reward, done, info    
    
    def get_shot_trajectory(self, chosen_move, current_player_pos, grid_map):
        
        # the default outcomes
        reward, done, info = 0, False, {"outcome" : None}  
        
        for i in range(self.SHOT_RANGE):
            
            # get the current bullet position
            bullet_vec = chosen_move * (i + 1)
            bullet_pos = current_player_pos + bullet_vec                
            row, col = bullet_pos
            
            # is the bullet out of bounds?
            if (row < 0 or row >= self.GRID_SIZE) or (col < 0 or col >= self.GRID_SIZE):
                break
            
            # has the bullet hit terrain?
            if grid_map[row, col] == 3:
                break
            
            # has the bullet hit the enemy?
            if grid_map[row, col] == self.opposing_player:      
                
                # remove the enemy
                grid_map[row, col] = 0
                self.bullet_hits = np.append(self.bullet_hits, bullet_pos.reshape(1, -1), axis=0)
                
                # set the parameters to end the game
                reward = self.POSITIVE_REWARD
                done = True
                info["outcome"] = self.current_player
                break
            
            # add the bullet path to the array for displaying
            self.bullet_path = np.append(self.bullet_path, bullet_pos.reshape(1, -1), axis=0)
            
        return reward, done, info
    
    def is_move_valid(self, final_player_pos, grid_map):   
        
        # assign the new player position
        row, col = final_player_pos          
        
        # check the square is within the grid
        if (row >= 0 and row < self.GRID_SIZE) and (col >= 0 and col < self.GRID_SIZE):
        
            # check the square is empty
            if grid_map[row, col] == 0:
                return True
                
        return False     
    
    def get_computer_action(self):
        
        # Want some sort of planning algorithm to make decisions        
        # Planning horizon could represent the agent's difficulty maybe?
        
        # get the computer's and the player's location
        computer_pos = np.asarray(np.where(self.grid_map == self.current_player)).flatten() 
        player_pos =  np.asarray(np.where(self.grid_map == self.opposing_player)).flatten()    
        
        # get the possible shot direction
        directions = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        
        # STEP 1: Check if a shot is possible
        
        # cycle through possible shots
        for shot_direction in range(directions.shape[0]):
            
            #  check if the game could conclude with a shot            
            _, done, _ = self.get_shot_trajectory(directions[shot_direction, :], computer_pos, self.grid_map)
            
            # select the concluding action
            if done: 
                move_action = 0 
                shot_action = shot_direction + 1
                return np.array([move_action + shot_action * 5], dtype=np.int32)
            
        # STEP 2: Check if a step shot is possible
        
        valid_computer_moves = []
        
        # cycle through possible moves
        for move_direction in range(directions.shape[0]):
            
            # get the possible moves
            chosen_move = directions[move_direction, :]        
            final_computer_pos =  computer_pos  + chosen_move
            
            # Check if the move is valid
            valid_move = self.is_move_valid(final_computer_pos, self.grid_map)
            
            #  check if the game could conclude with a move-shot  
            if valid_move:
                
                valid_computer_moves.append(chosen_move)
                
                for shot_direction in range(directions.shape[0]):
                    _, done, _ = self.get_shot_trajectory(directions[shot_direction, :], computer_pos)
                    
                    # select the concluding action
                    if done: 
                        move_action = move_direction + 1 
                        shot_action = shot_direction + 1
                        return np.array([move_action + shot_action * 5], dtype=np.int32)
                    
        # STEP 3: Check if any moves would result in the possibility of an enemy step shot
        
        # only cycle through valid moves
        valid_computer_moves = np.array(valid_computer_moves, dtype=np.int32)
        
        # initialise the safe computer moves
        safe_computer_moves = []
        
        # cycle through current player moves
        for computer_move_direction in range(valid_computer_moves.shape[0]):
            
            computer_move_safe = True
            
            # get the possible moves
            comp_chosen_move = directions[computer_move_direction, :]        
            final_computer_pos =  computer_pos + comp_chosen_move
            
            # update a temporary grid
            temp_grid_map = self.grid_map            
            temp_grid_map[final_computer_pos[0], final_computer_pos[1]] = self.current_player
            temp_grid_map[computer_pos[0], computer_pos[1]] = 0            
            
            # cycle through the player moves                            
            for player_move_direction in range(directions.shape[0]):
                
                # get the possible moves
                play_chosen_move = directions[player_move_direction, :]        
                final_player_pos =  player_pos + play_chosen_move
                
                # Check if the move is valid
                valid_player_move = self.is_move_valid(final_player_pos, temp_grid_map)
                
                if valid_player_move:
                    
                    # cycle through player shots
                    for player_shot_direction in range(directions.shape[0]):
                        
                        #  check if the game could conclude with a shot            
                        _, done, _ = self.get_shot_trajectory(directions[player_shot_direction, :], final_player_pos, temp_grid_map)
                        
                        # mark this computer move as unsafes
                        if done: 
                            computer_move_safe = False
                            break
                        
                if not computer_move_safe:                    
                    break
                
            # update the safe moves
            if computer_move_safe:
                safe_computer_moves.append(comp_chosen_move)        
        
        # STEP 4: Take the shortest path to the player excluding those that would result in a step shot
        
        safe_computer_moves = np.array(safe_computer_moves, np.int32)
            
        return np.random.randint(7, size=(1,))
            
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
        next_state, reward, done, info = env.step(player_action=action)
        
        # print the winner
        if done: 
            print('Player {} wins'.format(info["outcome"]))
            
        state = next_state
        counter += 1
    
    # close the display
    env.close_display()
    
            