#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:50:58 2019

@author: tawehbeysolow
"""

import gym, numpy as np, matplotlib.pyplot as plt
from neural_networks.policy_gradient_utilities import PolicyGradient

#Parameters 
n_units = 5
gamma = .99
batch_size = 50
learning_rate = 1e-3
n_episodes = 10000
render = False
goal = 190
n_layers = 2
n_classes = 2
environment = gym.make('CartPole-v1')
environment_dimension = len(environment.reset())
            
def calculate_discounted_reward(reward, gamma=gamma):
    output = [reward[i] * gamma**i for i in range(0, len(reward))]
    return output[::-1]

def score_model(model, n_tests, render=render):
    scores = []    
    for _ in range(n_tests):
        environment.reset()
        observation = environment.reset()
        reward_sum = 0
        while True:
            if render:
                environment.render()
                
            state = np.reshape(observation, [1, environment_dimension])
            predict = model.predict([state])[0]
            action = np.argmax(predict)
            observation, reward, done, _ = environment.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
        
    environment.close()
    return np.mean(scores)

def cart_pole_game(environment, policy_model, model_predictions):
    loss = []
    n_episode, reward_sum, score, episode_done = 0, 0, 0, False
    n_actions = environment.action_space.n
    observation = environment.reset()
    
    states = np.empty(0).reshape(0, environment_dimension)
    actions = np.empty(0).reshape(0, 1)
    rewards = np.empty(0).reshape(0, 1)
    discounted_rewards = np.empty(0).reshape(0, 1)
    
    while n_episode < n_episodes: 
         
        state = np.reshape(observation, [1, environment_dimension])        
        prediction = model_predictions.predict([state])[0]
        action = np.random.choice(range(environment.action_space.n), p=prediction)
        states = np.vstack([states, state])
        actions = np.vstack([actions, action])
        
        observation, reward, episode_done, info = environment.step(action)
        reward_sum += reward
        rewards = np.vstack([rewards, reward])

        if episode_done == True:
            
            discounted_reward = calculate_discounted_reward(rewards)
            discounted_rewards = np.vstack([discounted_rewards, discounted_reward])
            rewards = np.empty(0).reshape(0, 1)
            
            if (n_episode + 1) % batch_size == 0:
                
                discounted_rewards -= discounted_rewards.mean()
                discounted_rewards /= discounted_rewards.std()
                discounted_rewards = discounted_rewards.squeeze()
                actions = actions.squeeze().astype(int)
                
                train_actions = np.zeros([len(actions), n_actions])
                train_actions[np.arange(len(actions)), actions] = 1
                
                error = policy_model.train_on_batch([states, discounted_rewards], train_actions)
                loss.append(error)
                
                states = np.empty(0).reshape(0, environment_dimension)
                actions = np.empty(0).reshape(0, 1)
                discounted_rewards = np.empty(0).reshape(0, 1)
                                
                score = score_model(model=model_predictions, n_tests=10)
                
                print('''\nEpisode: %s \nAverage Reward: %s  \nScore: %s \nError: %s'''
                      )%(n_episode+1, reward_sum/float(batch_size), score, np.mean(loss[-batch_size:]))
    
                if score >= goal: 
                    break 
                
                reward_sum = 0
                
            n_episode += 1
            observation = environment.reset()
            
    plt.title('Policy Gradient Error plot over %s Episodes'%(n_episode+1))
    plt.xlabel('N batches')
    plt.ylabel('Error Rate')
    plt.plot(loss)
    plt.show()
    
if __name__ == '__main__':
        
    
    mlp_model = PolicyGradient(n_units=n_units, 
                              n_layers=n_layers, 
                              n_columns=environment_dimension, 
                              n_outputs=n_classes, 
                              learning_rate=learning_rate, 
                              hidden_activation='selu', 
                              output_activation='softmax',
                              loss_function='log_likelihood')
        
    policy_model, model_predictions = mlp_model.create_policy_model(input_shape=(environment_dimension, ))
    
    policy_model.summary()
    
    cart_pole_game(environment=environment, 
                   policy_model=policy_model, 
                   model_predictions=model_predictions)
    