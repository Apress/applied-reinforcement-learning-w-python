#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:58:25 2019

@author: tawehbeysolow
"""

import os, time, gym, numpy as np

#Parameters
learning_rate = 1e-2
gamma = 0.96
epsilon = 0.9
n_episodes = 10000
max_steps = 100
environment = gym.make('FrozenLake-v0')
Q_matrix = np.zeros((environment.observation_space.n, environment.action_space.n))

def choose_action(state):
    '''
    To be used after Q table has been updated, returns an action
    
    Parameters:
        
        state - int - the current state of the agent 
        
    :return: int
    '''   
    return np.argmax(Q_matrix[state, :])

def exploit_explore(prior_state, epsilon=epsilon, Q_matrix=Q_matrix):    
    '''
    One half of the exploit-explore paradigm that we will utilize 
    
    Parameters 
        
        prior_state - int  - the prior state of the environment at a given iteration
        epsilon - float - parameter that we use to determine whether we will try a new or current best action 
        
    :return: int
    '''
    
    if np.random.uniform(0, 1) < epsilon:
        return environment.action_space.sample()
    else:
        return np.argmax(Q_matrix[prior_state, :])
    
    
def update_q_matrix(prior_state, observation , reward, action):
    '''
    Algorithm that updates the values in the Q table to reflect knowledge acquired by the agent 
    
    Parameters 
    
        prior_state - int  - the prior state of the environment before the current timestemp
        observation - int  - the current state of the environment
        reward - int - the reward yielded from the environment after an action 
        action - int - the action suggested by the epsilon greedy algorithm 
        
    :return: None
    '''
    
    prediction = Q_matrix[prior_state, action]
    actual_label = reward + gamma * np.max(Q_matrix[observation, :])
    Q_matrix[prior_state, action] = Q_matrix[prior_state, action] + learning_rate*(actual_label - prediction)
    
    
def populate_q_matrix(render=False, n_episodes=n_episodes):
    '''
    Directly implementing Q Learning (Greedy Epsilon) on the Frozen Lake Game
    This function populations the empty Q matrix 
    Parameters 
    
        prior_state - int  - the prior state of the environment before the current timestemp
        observation - int  - the current state of the environment
        reward - int - the reward yielded from the environment after an action 
        action - int - the action suggested by the epsilon greedy algorithm 
        
    :return: None
    '''    
    
    for episode in range(n_episodes):
        prior_state = environment.reset()
        _ = 0
        
        while _ < max_steps:
            
            if render == True: environment.render()
            action = exploit_explore(prior_state)  
            observation, reward, done, info = environment.step(action)      
            
            update_q_matrix(prior_state=prior_state, 
                            observation=observation, 
                            reward=reward, 
                            action=action)
            
            prior_state = observation
            _ += 1
            
            if done:
                break
                            

def play_frozen_lake(n_episodes):
    
    '''
    Directly implementing Q Learning (Greedy Epsilon) on the Frozen Lake Game
    This function uses the already populated Q Matrix and displays the game being used
    
    Parameters 
    
        prior_state - int  - the prior state of the environment before the current timestemp
        observation - int  - the current state of the environment
        reward - int - the reward yielded from the environment after an action 
        action - int - the action suggested by the epsilon greedy algorithm 
        
    :return: None
    '''        
        
    for episode in range(n_episodes):
        print('Episode: %s'%episode+1)
        prior_state = environment.reset()
        done = False

        while not done: 
            environment.render()
            action = choose_action(prior_state)
            observation, reward, done, info = environment.step(action)
            prior_state = observation
            if reward == 0:
                time.sleep(0.5)
            else:
                print('You have won on episode %s!'%(episode+1))
                time.sleep(5)
                os.system('clear')

            if done and reward == -1:
                print('You have lost this episode... :-/')
                time.sleep(5)
                os.system('clear')
                break
                
        
        
if __name__ == '__main__':
    
    
    populate_q_matrix(render=False)
    play_frozen_lake(n_episodes=10)
