#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 23:18:17 2019

@author: tawehbeysolow
"""

import gym

def cartpole():
    environment = gym.make('CartPole-v1')
    environment.reset()
    for _ in range(1000):
        environment.render()
        action = environment.action_space.sample()
        observation, reward, done, info = environment.step(action)
        print("Step {}:".format(_))
        print("action: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
    
if __name__ == '__main__':
    
    cartpole()