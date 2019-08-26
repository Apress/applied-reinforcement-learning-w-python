#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:59:09 2019

@author: tawehbeysolow
"""

import numpy as np
from skimage import transform 
from collections import deque 
from vizdoom import *             
          
class Memory():
    
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size=batch_size,
                                replace=True)
        
        return [self.buffer[i] for i in index]

def create_environment(filepath='/Users/tawehbeysolow/Desktop/applied_rl_python/chapter3/'):
    game = DoomGame()    
    game.load_config(filepath+'basic.cfg')
    game.set_doom_scenario_path(filepath+'basic.wad')    
    game.init()
    
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions
 

def preprocess_frame(frame):
    cropped_frame = frame[30:-10,30:-30]    
    normalized_frame = cropped_frame/float(255)
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame

def stack_frames(stacked_frames, state, new_episode, stack_size=4):
    
    frame = preprocess_frame(state)
    
    if new_episode == True:
        
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        for i in range(4):
            stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames