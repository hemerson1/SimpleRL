#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:15:32 2021

@author: hemerson
"""

""" 
race_car_env - a race car track where the agent is timed for 1 lap

"""

import numpy as np
import random
import pygame
import os
import math

from SimpleRL.envs import environment_base 
from SimpleRL.utils import generate_track
from SimpleRL.utils import init_video, save_frames, create_video

# Testing
import matplotlib.pyplot as plt

class race_car_env(environment_base):
    
    # TODO: remove inefficiency in the code (i.e. repeated expressions, improve speed)
    
    def __init__(self, render=False, seed=None, render_mode="default"):
        
        # Get assertion errors
        
        # Ensure the render_mode for the environment is valid
        valid_render = ["default", "video"]
        render_error = "render_mode is not valid for this environment, " \
            + "please select one of the following {} ".format(valid_render)
        assert render_mode in valid_render, render_error
        
        # Display the input settings 
        print('\Race Car Settings')
        print('--------------------')
        print('Render: {}'.format(render))        
        print('Seed: {}'.format(seed))
        print('Render Mode: {}'.format(render_mode))   
        print('--------------------')
        
        self.render = render 
        self.render_mode = render_mode
        self.seed = seed 
        
        # Set environmetal parameters
        np.random.seed(self.seed)  
        random.seed(self.seed)
        self.environment_name = 'Race_car'
        self.action_dim = 1
        self.action_num = np.array([4], dtype=np.int32)
        
        self.height = 600
        self.width = 800
        self.track_width = 40
        
        # Reset the environment parameters
        self.reset()
        
        # Intialise the display
        if self.render: 
            
            # Check if there is an available display
            try: os.environ["DISPLAY"]
            
            # Configure a dummy display
            except: os.environ["SDL_VIDEODRIVER"] = "dummy"
            
            if self.render_mode == 'video':
                self.frame_count, self.image_folder, self.video_folder = init_video(environment_name=self.environment_name)
            
            # set the screen dimensions
            self.window_width = self.width
            self.window_height = self.height 
            
            # get the screen
            self._init_display() 
            
            # set the fps
            self.fps = 5
            
            # get the font 
            self.font = pygame.font.Font(None, 32)
            
            # create the colours
            self.white = (255, 255, 255)
            self.black = (0, 0, 0)    
            self.blue = (102, 178, 255)
            self.red = (255, 51, 51)
            self.grass_green = (58, 156, 53)            
            self.grey = (186, 182, 168)
                  
    
    def reset(self):
        
        # generate the new track points and checkpoints
        self.track_points, self.checkpoints = generate_track(height=self.height, width=self.width, track_width=self.track_width)        
        
        # Cycle through the points and get the location of the edges of the track
        # create this into a surface and calculate the point at which feelers overlap the edge of the surface. 
        
        inside_track_points = []
        outside_track_points = []
        radius = self.track_width // 2
        
        final_index = len(self.track_points) - 1
        for idx, point in enumerate(self.track_points):
            
            # get the next and current point
            current_point = point
            next_point = self.track_points[(idx + 1) % final_index]
            
            # calculate an angle between the two points (in radians)
            angle = math.atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
            
            outside_track_x = current_point[0] + radius * math.sin(angle)
            outside_track_y = current_point[1] - radius * math.cos(angle)   
            outside_track_points.append((outside_track_x, outside_track_y))
            
            inside_track_x = current_point[0] - radius * math.sin(angle)
            inside_track_y = current_point[1] + radius * math.cos(angle)
            inside_track_points.append((inside_track_x, inside_track_y))                  
            
        # TODO: place player on track
        
        # TODO: process the player state and return it
        # How will the players state be detected? 
        # Probably should be feelers for distance on front half of car
        
        # Could try and get perimeter of track and then calculate point of intersection relative to player position?
        
        
    def step(self, player_action=None):
        
       # process the action
        
        # display the map
        if self.render:
            self._display()      
             
        # state, reward, done, info
        return self.grid_map.flatten(), reward, done, info

            
    def _init_display(self):
        
        # quit any previous games
        pygame.display.quit()
        
        # initialise pygame
        pygame.init()    
        
        # set the environment name
        pygame.display.set_caption("Race Car Environment")
        
        # initialise the clock
        self.clock = pygame.time.Clock()
        
        # create the screen
        self.screen = pygame.display.set_mode([self.window_height, self.window_width])
        
    
    def _display(self):   
        
        # quit the game
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.display.quit()
                        
        # set the background colour to green
        self.screen.fill(self.green)        
        
        # TODO: fill in with specific code
        
        # save frames to the folder
        if self.render_mode == "video":            
            self.frame_count = save_frames(screen=self.screen, image_folder=self.image_folder, frame_count=self.frame_count)
          
        # update the display
        pygame.display.update()
        
        # update the frame rate
        self.clock.tick(self.fps)
        
    def _close_display(self):
        
        # shut the display window
        pygame.display.quit()
        
        # create a video
        if self.render_mode == 'video':    
            create_video(image_folder=self.image_folder, video_folder=self.video_folder, fps=self.fps)
        
        
if __name__ == "__main__": 
        
    seed_range = 10
    
    # track the player wins out of max
    total_reward = 0
    
    for seed in range(seed_range):
    
        # intialise the environment
        env = race_car_env(seed=seed, render=False)
        
        jaskdbjaksdh
        
        # reset the state
        state, done = env.reset(), False
        counter = 0
        
        # run the training loop
        while not done and counter < 100:
            
            action = np.random.randint(0, env.action_dim, env.action_num)     
            next_state, reward, done, info = env.step(player_action=action)
            
            if reward > 0:
                total_reward += reward
            
            # print the winner
            if done: 
                print('Seed {} - Player {} wins'.format(seed, info["outcome"]))
                
            state = next_state
            counter += 1
        
    