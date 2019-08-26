#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:00:57 2019

@author: tawehbeysolow
"""

import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from algorithms.actor_critic_utilities import train_model
from neural_networks.models import ActorCriticModel

#Parameters
environment = gym_super_mario_bros.make('SuperMarioBros-v0')
environment = BinarySpaceToDiscreteSpaceEnv(environment, SIMPLE_MOVEMENT)
observation = environment.reset()
learning_rate = 1e-4
gamma = 0.96
epsilon = 0.9
n_episodes = 10000
n_steps = 2048
max_steps = int(1e7)
_lambda = 0.95
value_coefficient = 0.5
entropy_coefficient = 0.01
max_grad_norm = 0.5
log_interval = 10

def play_super_mario(model, environment=environment):
     
    observations = environment.reset()
    score, n_step, done = 0, 0, False
    scores = []
    
    for _ in range(100):
        
        while done:
            
            actions, values = model.step(observations)        
            observations, rewards, done, info = environment.step(actions)
            score += rewards    
            environment.render()        
            n_step += 1
            scores.append(score)
                        
        print('Step: %s \nScore: %s '%(n_step, score))
        environment.reset()
    
    print(np.mean(scores))

if __name__ == '__main__':
    
    model = train_model(policy_model=ActorCriticModel,
                        environment=environment, 
                        n_steps=n_steps, 
                        max_steps=max_steps, 
                        gamma=gamma, 
                        _lambda=_lambda,
                        value_coefficient=value_coefficient, 
                        entropy_coefficient=entropy_coefficient, 
                        learning_rate=learning_rate, 
                        max_grad_norm=max_grad_norm, 
                        log_interval=log_interval)
    
    play_super_mario(model=model,
                     environment=environment)