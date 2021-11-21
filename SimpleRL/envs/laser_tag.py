#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 21:57:14 2021

@author: hemerson
"""

""" 
laser_tag_env - a simple grid environment for 1 vs 1 laser tag 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from SimpleRL.envs.base import environment_base 
from SimpleRL.utils.laser_tag_utils import generate_scenario, shortest_path

class laser_tag_env(environment_base):

    # TODO: remove inefficiency in the code (i.e. repeated expressions, improve speed)
    
    def __init__(self, render=False, seed=None, action_mode="default", enemy_mode="default", lives=1):
        
        # Get assertion errors
        
        # Ensure the enemy mode for the environment is valid
        valid_action_mode = ["default", "single"]
        action_mode_error = "action_mode is not valid for this environment, " \
            + "please select one of the following {} ".format(valid_action_mode)
        assert action_mode in valid_action_mode, action_mode_error
        
        # Ensure the enemy mode for the environment is valid
        valid_enemy_mode = ["default", "adversarial"]
        enemy_mode_error = "enemy_mode is not valid for this environment, " \
            + "please select one of the following {} ".format(valid_enemy_mode)
        assert enemy_mode in valid_enemy_mode, enemy_mode_error
        
        self.render = render 
        self.seed = seed 
        self.enemy_mode = enemy_mode
        self.action_mode = action_mode
        self.lives = lives
        
        # create a 1D action space
        if self.action_mode == "single":
            self.action_dim = 1 # how many actions are made each turn?
            self.action_num = np.array([25], dtype=np.int32) # how many values are there per action?  
         
        # create the default multi-dimensional action space
        elif self.action_mode == "default":
            self.action_dim = 2
            self.action_num = np.array([5, 5], dtype=np.int32) 
        
        # Set environmetal parameters
        np.random.seed(self.seed)  
        self.shot_range = 5 # how many squares will a bullet travel?
        self.grid_size = 8
        self.positive_reward = +1
        self.negative_reward = -1
        self.player_param = 1
        self.enemy_param = 2
        self.terrain_param = 3
        self.empty_param = 0
        self.display_pause = 0.25
        
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
        self.current_player = self.player_param
        self.player_lives = self.lives
        self.opposing_player = self.enemy_param
        self.enemy_lives = self.lives
        
        self.game_outcome = None
        self.grid_map = generate_scenario(
                grid_size = self.grid_size
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
            
            # get the computer action
            if self.enemy_mode == "default":
                computer_action = self.get_computer_action()                  
                reward, done, info = self.update_grid(action=computer_action)  
                
                # display the map
                if self.render:
                    self.display()                
            
            # get the action of another network
            elif self.enemy_mode == "adversarial":       
                pass
                 
        return self.grid_map.flatten(), reward, done, info
    
    def sample_discrete_action(self):
        return np.random.randint(self.action_num - 1, size=self.action_dim)
    
    def update_grid(self, action):
        
        # 25 actions --------        
        #      NO|LM|UM|RM|DM
        #   NO|00|01|02|03|04
        #   LS|05|06|07|08|09
        #   US|10|11|12|13|14
        #   RS|15|15|16|17|18
        #   DS|20|21|22|23|24
        # -------------------
        
        # 0, 1, 2, 3, 4 = no_move, move_left, move_up, move_right, move_down
        # 0, 1, 2, 3, 4 = no_shot, shoot_left, shoot_up, shoot_right, shoot_down
              
        if self.action_mode == "single":
            move_action = action % 5        
            shot_action = np.floor_divide(action, 5)
        
        elif self.action_mode == "default":
            move_action = action[0].reshape(1,)
            shot_action = action[1].reshape(1,)
        
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
                        
            valid_move = self.is_move_valid(final_player_pos, self.grid_map)
            
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
            reward, done, info = self.get_shot_trajectory(chosen_move, player_pos, self.grid_map, self.current_player)
            
            # if the player has been shot remove them
            if done:
                
                done = False
                
                # update the lives
                if self.current_player == self.player_param:
                    self.enemy_lives -= 1            
                else:
                    self.player_lives -= 1
                
                # end the game when lives have run out
                if self.player_lives == 0 or self.enemy_lives == 0:                
                
                    # get the enemy position
                    enemy_pos = np.asarray(np.where(self.grid_map == self.opposing_player)).flatten() 
                    
                    # remove the enemy and update the display
                    self.grid_map[enemy_pos[0], enemy_pos[1]] = 0
                    self.bullet_hits = np.append(self.bullet_hits, enemy_pos.reshape(1, -1), axis=0)
                    
                    done = True
                                                
        # switch the current player to the opposite player 
        temp_opposing_player = self.opposing_player 
        self.opposing_player = self.current_player
        self.current_player = temp_opposing_player
                
        return reward, done, info    
    
    def get_shot_trajectory(self, chosen_move, current_player_pos, grid_map, current_player):
                
        # get the player value
        if current_player == self.player_param:
            opposing_player = self.enemy_param
        else:
            opposing_player = self.player_param
        
        # the default outcomes
        reward, done, info = 0, False, {"outcome" : None}  
        
        for i in range(self.shot_range):
            
            # get the current bullet position
            bullet_vec = chosen_move * (i + 1)
            bullet_pos = current_player_pos + bullet_vec                
            row, col = bullet_pos
            
            # is the bullet out of bounds?
            if (row < 0 or row >= self.grid_size) or (col < 0 or col >= self.grid_size):
                break
            
            # has the bullet hit terrain?
            if grid_map[row, col] == self.terrain_param:
                break
            
            # has the bullet hit the enemy?
            if grid_map[row, col] == opposing_player: 
                
                # set the parameters to end the game
                reward = self.positive_reward
                done = True
                info["outcome"] = current_player
                break
            
            # add the bullet path to the array for displaying
            self.bullet_path = np.append(self.bullet_path, bullet_pos.reshape(1, -1), axis=0)
            
        return reward, done, info
    
    def is_move_valid(self, final_player_pos, grid_map):   
        
        # assign the new player position
        row, col = final_player_pos          
        
        # check the square is within the grid
        if (row >= 0 and row < self.grid_size) and (col >= 0 and col < self.grid_size):
        
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
        # left, up, right, down
        directions = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        
        # STEP 1: Check if a shot is possible
        
        # cycle through possible shots
        for shot_direction in range(directions.shape[0]):
            
            #  check if the game could conclude with a shot            
            _, done, _ = self.get_shot_trajectory(directions[shot_direction, :], computer_pos, self.grid_map, self.current_player)
            
            # select the concluding action
            if done: 
                
                move_action = 0 
                shot_action = shot_direction + 1
                
                if self.action_mode == "single":
                    return np.array([move_action + shot_action * 5], dtype=np.int32)
                
                elif self.action_mode == "default":
                    return np.array([move_action, shot_action], dtype=np.int32)
        
        # STEP 2: Check if a step shot is possible
        
        valid_computer_moves = []
        
        # cycle through possible moves
        for move_direction in range(directions.shape[0]):
            
            # get the possible moves
            chosen_move = directions[move_direction, :]        
            final_computer_pos =  computer_pos + chosen_move
            
            # Check if the move is valid
            valid_move = self.is_move_valid(final_computer_pos, self.grid_map)
            
            #  check if the game could conclude with a move-shot  
            if valid_move:
                
                valid_computer_moves.append(chosen_move)
                
                # update a temporary grid
                temp_grid_map = np.copy(self.grid_map)
                temp_grid_map[final_computer_pos[0], final_computer_pos[1]] = self.current_player
                temp_grid_map[computer_pos[0], computer_pos[1]] = 0   
                
                for shot_direction in range(directions.shape[0]):
                    _, done, _ = self.get_shot_trajectory(directions[shot_direction, :], final_computer_pos, temp_grid_map, self.current_player)
                    
                    # select the concluding action
                    if done: 
                        move_action = move_direction + 1 
                        shot_action = shot_direction + 1
                        
                        if self.action_mode == "single":
                            return np.array([move_action + shot_action * 5], dtype=np.int32)
                        
                        elif self.action_mode == "default":
                            return np.array([move_action, shot_action], dtype=np.int32)
       
        # STEP 3: Check if any moves would result in the possibility of an enemy step shot
        
        # only cycle through valid moves
        valid_computer_moves = np.array(valid_computer_moves, dtype=np.int32)
        
        # initialise the safe computer moves
        unsafe_computer_moves = []
        safe_computer_moves = []
        
        # cycle through current player moves
        for computer_move_direction in range(valid_computer_moves.shape[0]):
            
            computer_move_safe = True
            
            # get the possible moves
            comp_chosen_move = valid_computer_moves[computer_move_direction, :]                
            final_computer_pos = computer_pos + comp_chosen_move
            
            # update a temporary grid
            temp_grid_map = np.copy(self.grid_map)
            temp_grid_map[final_computer_pos[0], final_computer_pos[1]] = self.current_player
            temp_grid_map[computer_pos[0], computer_pos[1]] = 0   
            
            # update temp player
            temp_player = self.opposing_player
            
            # cycle through the player moves                            
            for player_move_direction in range(directions.shape[0]):
                
                # get the possible moves
                play_chosen_move = directions[player_move_direction, :]        
                final_player_pos =  player_pos + play_chosen_move
                
                # Check if the move is valid
                valid_player_move = self.is_move_valid(final_player_pos, temp_grid_map)
                
                if valid_player_move:
                    
                    # update the temporary grid again
                    player_temp_grid_map = np.copy(temp_grid_map)
                    player_temp_grid_map[final_player_pos[0], final_player_pos[1]] = temp_player
                    player_temp_grid_map[player_pos[0], player_pos[1]] = 0   
                    
                    # cycle through player shots
                    for player_shot_direction in range(directions.shape[0]):
                        
                        #  check if the game could conclude with a shot            
                        _, done, _ = self.get_shot_trajectory(directions[player_shot_direction, :], final_player_pos, player_temp_grid_map, temp_player)
                        
                        # mark this computer move as unsafes
                        if done:              
                            computer_move_safe = False
                            unsafe_computer_moves.append(comp_chosen_move)
                            break
                
                # break the loop if an unsafe move is found
                if not computer_move_safe:                    
                    break
            
            # make a list of safe moves
            if computer_move_safe:
                safe_computer_moves.append(comp_chosen_move)
       
        # STEP 4: Take the shortest path to the player excluding those that would result in a step shot from opponent
        
        unsafe_computer_moves = np.array(unsafe_computer_moves, np.int32)
        safe_computer_moves = np.array(safe_computer_moves, np.int32)
        
        # if there is only one safe move take it
        if safe_computer_moves.shape[0] == 1:            
            move_action = np.where(np.all(directions == safe_computer_moves, axis=1))[0][0] + 1 
            
            if self.action_mode == "single":
                return np.array([move_action], dtype=np.int32)
            
            elif self.action_mode == "default":
                return np.array([move_action, 0], dtype=np.int32)   
        
        # if there are no safe moves, do not move
        if safe_computer_moves.shape[0] == 0:     
            
            if self.action_mode == "single":
                return np.array([0], dtype=np.int32)
            
            elif self.action_mode == "default":
                return np.array([0, 0], dtype=np.int32)   
                    
        # Remove the dangerous moves from the scope of the search algorithm
        temp_grid_map = np.copy(self.grid_map)
        for moves in range(unsafe_computer_moves.shape[0]):
            
            # get the possible moves
            chosen_move = unsafe_computer_moves[moves, :]        
            final_computer_pos = computer_pos  + chosen_move
            
            # Block out the dangerous move from the search
            temp_grid_map[final_computer_pos[0], final_computer_pos[1]] = 3
        
        # get the shortest path
        path = shortest_path(temp_grid_map, start=computer_pos, goal=player_pos)
        
        # get the selected move
        selected_move = np.array(path[-2], dtype=np.int32) - computer_pos        
        move_action = np.where(np.all(directions == selected_move, axis=1))[0][0] + 1   
        
        if self.action_mode == "single":
            return np.array([move_action], dtype=np.int32)
        
        elif self.action_mode == "default":
            return np.array([move_action, 0], dtype=np.int32)

            
    def init_display(self, grid_map):
        
        # define the colour map for the grid
        cmap = colors.ListedColormap(['#5ba01b', '#3a87e2', '#c7484e', '#5c5c5c'])
        
        figure, axis = plt.subplots(1,1)
        image = axis.imshow(grid_map, cmap=cmap)
        
        # get the player and enemy positions
        player_row, player_col = np.asarray(np.where(self.grid_map == self.player_param)).flatten() 
        enemy_row, enemy_col =  np.asarray(np.where(self.grid_map == self.enemy_param)).flatten()         
        
        # label their positions
        axis.text(player_col, player_row, 'P', ha="center", va="center", color="white")
        axis.text(enemy_col, enemy_row, 'E', ha="center", va="center", color="white")            
        
        # adjust plot to figure area
        figure.tight_layout()        
        
        return image, figure, axis
    
    def display(self):   
        
        # update the image
        new_array = np.copy(self.grid_map)
        self.image.set_data(new_array)
        
        # remove all the previous text labels
        self.axis.texts = []
        
        # get the player and label their positions
        player_pos = np.where(self.grid_map == self.player_param)
        if len(player_pos[0]) > 0:        
            player_row, player_col = np.asarray(player_pos).flatten() 
            self.axis.text(player_col, player_row, self.player_lives, ha="center", va="center", color="white")
                        
        # get the enemy and label their positions
        enemy_pos = np.where(self.grid_map == self.enemy_param)        
        if len(enemy_pos[0]) > 0:
            enemy_row, enemy_col = np.asarray(enemy_pos).flatten()  
            self.axis.text(enemy_col, enemy_row, self.enemy_lives, ha="center", va="center", color="white")
        
        # mark the bullet's path on the grid            
        for sqr in range(self.bullet_path.shape[0]):
            self.axis.text(self.bullet_path[sqr, 1], self.bullet_path[sqr, 0], '*', ha="center", va="center", color="black")
                
        # mark a hit on the grid
        if self.bullet_hits.shape[0] > 0:
            self.axis.text(self.bullet_hits[0, 1], self.bullet_hits[0, 0], 'X', ha="center", va="center", color="black")
        
        # draw the new image
        self.figure.canvas.draw_idle()
        plt.pause(self.display_pause)   
        
    def close_display(self):
        plt.close()
        
if __name__ == "__main__":    
    
    seed_range = 1
    enemy_mode = "adversarial"
    
    for seed in range(seed_range):
    
        # intialise the environment
        env = laser_tag_env(seed=seed, 
                            render=True,
                            action_mode="default",
                            enemy_mode=enemy_mode,
                            lives=1)
        
        # reset the state
        state, done = env.reset(), False
        counter = 0
        
        # run the training loop
        while not done and counter < 100:
            
            action = env.sample_discrete_action()            
            next_state, reward, done, info = env.step(player_action=action)
            
            # print the winner
            if done: 
                print('Seed {} - Player {} wins'.format(seed, info["outcome"]))
                
            state = next_state
            counter += 1
            
            # get an action from the opposing network in adversarial mode
            if enemy_mode == "adversarial" and not done:
                
                action = env.sample_discrete_action()            
                next_state, reward, done, info = env.step(player_action=action)
                
                # print the winner
                if done: 
                    print('Seed {} - Player {} wins'.format(seed, info["outcome"]))
                    
                state = next_state
                counter += 1                
        
        # close the display
        env.close_display()
    
            