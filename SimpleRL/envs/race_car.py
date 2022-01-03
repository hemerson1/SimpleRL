#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:15:32 2021

@author: hemerson
"""

""" 
race_car_env - a race car track where the agent is timed for 1 lap.

"""

import numpy as np
import random
import pygame
import os
import math

from SimpleRL.envs import environment_base 
from SimpleRL.utils import generate_track, draw_map, simulate_car
from SimpleRL.utils import init_video, save_frames, create_video

# Testing
import matplotlib.pyplot as plt

class race_car_env(environment_base):
    
    # TODO: remove inefficiency in the code (i.e. repeated expressions, improve speed)
    # TODO: fix the gap in the track edge
    # TODO: add inner layer to detection
    # TODO: add multiple 'feelers' for detection
    # TODO: add checkpoint detection
    
    def __init__(self, render=False, seed=None, render_mode="default", driver_mode="human"):
        
        # Get assertion errors
        
        # Ensure the render_mode for the environment is valid
        valid_render = ["default", "video"]
        render_error = "render_mode is not valid for this environment, " \
            + "please select one of the following {} ".format(valid_render)
        assert render_mode in valid_render, render_error
        
        # Ensure the driver_mode for the environment is valid
        valid_driver= ["default", "human"]
        driver_error = "driver_mode is not valid for this environment, " \
            + "please select one of the following {} ".format(valid_driver)
        assert driver_mode in valid_driver, driver_error
        
        # Display the input settings 
        print('\nRace Car Settings')
        print('--------------------')
        print('Render: {}'.format(render))        
        print('Seed: {}'.format(seed))
        print('Render Mode: {}'.format(render_mode))   
        print('Driver Mode: {}'.format(driver_mode))  
        print('--------------------')
        
        self.render = render 
        self.render_mode = render_mode
        self.seed = seed 
        self.driver_mode = driver_mode
        
        # set render to true if human driver
        if self.driver_mode == "human":
            self.render = True
        
        # Set environmetal parameters
        np.random.seed(self.seed)  
        random.seed(self.seed)
        self.environment_name = 'Race_car'
        self.action_dim = 1
        self.action_num = np.array([4], dtype=np.int32)
        
        self.height, self.width = 600, 800
        self.track_width = 60 
        self.fps = 30        
        
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
            
            # set map render parameters
            self.checkpoint_margin = 5
            self.checkpoint_angle_offset = 3
            
            # get the screen
            self._init_display() 
            
            # get the font 
            self.font = pygame.font.Font(None, 32)
            
            # create the colours
            self.white = (255, 255, 255)
            self.black = (0, 0, 0)    
            self.blue = (102, 178, 255)
            self.red = (255, 51, 51)
            self.grass_green = (58, 156, 53)            
            self.grey = (186, 182, 168)
            self.yellow = (255, 233, 0)
            
    
    def reset(self):
        
        # generate the new track points and checkpoints
        self.track_points, self.checkpoints = generate_track(height=self.height, width=self.width, track_width=self.track_width)        
        
        # Cycle through the points and get the location of the edges of the track
        # create this into a surface and calculate the point at which feelers overlap the edge of the surface. 
        
        inside_track_points = []
        outside_track_points = []
        radius = self.track_width // 2
        final_angle = None
        
        final_index = len(self.track_points) - 1 
        for idx, point in enumerate(self.track_points):
            
            # get the next and current point
            current_point = point
            next_point = self.track_points[(idx + 30) % final_index]
            prev_point = self.track_points[(idx - 30) % final_index]
            
            # calculate an angle between the two points (in radians)
            angle = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
                        
            # add the outside track coordinates
            outside_track_x = current_point[0] + radius * math.sin(angle)
            outside_track_y = current_point[1] - radius * math.cos(angle)   
            outside_track_points.append((outside_track_x, outside_track_y))  
            
            # add the inside track coordinates
            inside_track_x = current_point[0] - radius * math.sin(angle)
            inside_track_y = current_point[1] + radius * math.cos(angle)
            inside_track_points.append((inside_track_x, inside_track_y))   
            
            # record the angle between the final point and the first 
            if idx == final_index:
                final_angle = angle
         
        # get the start point of the car
        start_point = self.checkpoints[0]
        start_angle = final_angle
        
        # initialise the car
        self.car = simulate_car(fps=self.fps, starting_position=start_point, starting_angle=start_angle)
        
        # set the inside and outside track points
        self.inside_track_points = inside_track_points
        self.outside_track_points = outside_track_points
        
        
        """
        x, y = zip(*self.track_points)
        x_i, y_i = zip(*inside_track_points)
        x_o, y_o = zip(*outside_track_points)
        
        plt.plot(y, x)
        plt.plot(y_i, x_i)
        plt.plot(y_o, x_o)
        plt.scatter(start_point[1], start_point[0])
        
        plt.show()
        """       
        
        # TODO: process the player state and return it
        # How will the players state be detected? 
        # Probably should be feelers for distance on front half of car
        
        # Could try and get perimeter of track and then calculate point of intersection relative to player position?
        
        
    def step(self, player_action=None):
        
        # process the action
        self.car.process_action(action=player_action)
        
        self.car.get_sensor_range(screen=self.screen, 
                                  outside_track_points=self.outside_track_points, 
                                  inside_track_points=self.inside_track_points,
                                  track_points=self.track_points
                                  )
        
        # display the map
        if self.render:
            self._display() 
        
        # if there is an AI driver
        if self.driver_mode == "default":            
            # state, reward, done, info
            return self.grid_map.flatten(), reward, done, info
        
        elif self.driver_mode == "human":            
            done = False
            return done

            
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
        self.screen = pygame.display.set_mode([self.window_width, self.window_height])
        
    
    def _display(self):   
        
        # quit the game
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.display.quit()
                
        # draw the map
        self.screen = draw_map(f_points=self.track_points, checkpoints=self.checkpoints, screen=self.screen,
                               track_width=self.track_width, checkpoint_margin=self.checkpoint_margin, 
                               checkpoint_angle_offset=self.checkpoint_angle_offset, track_colour=self.grey, 
                               checkpoint_colour=self.blue, start_colour=self.red, background_colour=self.grass_green)           
        
        # render the car
        self.screen = self.car.render_car(screen=self.screen, car_colour=self.yellow)        
        
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
        
    seed_range = 1
    driver_mode = "human"
    render = True
    
    # track the player wins out of max
    total_reward = 0
    
    for seed in range(seed_range):
    
        # intialise the environment
        env = race_car_env(seed=seed, render=False, driver_mode=driver_mode)
        
        # reset the state
        done, counter = False, 0
        if driver_mode == "default":
            state = env.reset
        
        # run the training loop
        while not done:
            
            if driver_mode == "human":
                
                # get the human action
                action = [False] * 4
                keys = pygame.key.get_pressed()                
                if keys[pygame.K_DOWN]:
                    action[0] = True
                if keys[pygame.K_UP]:
                    action[1] = True
                if keys[pygame.K_LEFT]:
                    action[2] = True
                if keys[pygame.K_RIGHT]:                    
                    action[3] = True
                            
                done = env.step(player_action=action)
                
            elif driver_mode == "default":
                next_state, reward, done, info = env.step(player_action=action)
            
                if reward > 0:
                    total_reward += reward
                    
                state = next_state
                
            counter += 1            
            #if counter >= 1000:
            #    done = True
            #    env._close_display()
        
    