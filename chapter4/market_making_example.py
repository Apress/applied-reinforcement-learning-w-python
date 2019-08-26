#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:00:05 2019

@author: tawehbeysolow
"""

import random, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tgym.envs import SpreadTrading
from tgym.gens.deterministic import WavySignal
from neural_networks.market_making_models import DeepQNetworkMM, Memory
from chapter2.cart_pole_example import calculate_discounted_reward
from neural_networks.policy_gradient_utilities import PolicyGradient
from tgym.gens.csvstream import CSVStreamer

#Parameters
np.random.seed(2018)
n_episodes = 1
trading_fee = .2
time_fee = 0
history_length = 2
memory_size = 2000
gamma = 0.96
epsilon_min = 0.01
batch_size = 64
action_size = len(SpreadTrading._actions)
learning_rate = 1e-2
n_layers = 4
n_units = 500
n_classes = 3
goal = 190
max_steps = 1000
explore_start = 1.0
explore_stop = 0.01
decay_rate = 1e-4
_lambda = 0.95
value_coefficient = 0.5
entropy_coefficient = 0.01
max_grad_norm = 0.5
log_interval = 10
hold =  np.array([1, 0, 0])
buy = np.array([0, 1, 0])
sell = np.array([0, 0, 1])
possible_actions = [hold, buy, sell]

#Classes and variables
generator = CSVStreamer(filename='/Users/tawehbeysolow/Downloads/amazon_order_book_data2.csv')
#generator = WavySignal(period_1=25, period_2=50, epsilon=-0.5)

memory = Memory(max_size=memory_size)

environment = SpreadTrading(spread_coefficients=[1],
                            data_generator=generator,
                            trading_fee=trading_fee,
                            time_fee=time_fee,
                            history_length=history_length)

state_size = len(environment.reset())


def baseline_model(n_actions, info, random=False):
    
    if random == True:
        action = np.random.choice(range(n_actions), p=np.repeat(1/float(n_actions), 3))
        action = possible_actions[action]

    else:
        
        if len(info) == 0:
            action = np.random.choice(range(n_actions), p=np.repeat(1/float(n_actions), 3))
            action = possible_actions[action]
        
        elif info['action'] == 'sell':
            action = buy
        
        else:   
            action = sell
            
    return action
        

def score_model(model, n_tests):
    scores = []    
    for _ in range(n_tests):
        environment.reset()
        observation = environment.reset()
        reward_sum = 0
        while True:
            ''
            #environment.render()
                
            predict = model.predict([observation.reshape(1, 8)])[0]
            action = possible_actions[np.argmax(predict)]
            observation, reward, done, _ = environment.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
        
    return np.mean(scores)


def exploit_explore(session, model, explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
        
    else:                
        Qs = session.run(model.output_layer, feed_dict = {model.input_matrix: state.reshape((1, 8))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability


def train_model(environment, dql=None, pg=None, baseline=None):
    scores = []
    done = False
    error_rate, step = 0, 0
    info = {}
    n_episode, reward_sum, score, episode_done = 0, 0, 0, False
    n_actions = len(SpreadTrading._actions)
    observation = environment.reset()
    states = np.empty(0).reshape(0, state_size)
    actions = np.empty(0).reshape(0, len(SpreadTrading._actions))
    rewards = np.empty(0).reshape(0, 1)
    discounted_rewards = np.empty(0).reshape(0, 1)
    observation = environment.reset()
    
    if baseline == True:
        
        
        for episode in range(n_episodes):
                                
            for _ in range(100):
                action = baseline_model(n_actions=n_actions,
                                        info=info)
                
                state, reward, done, info = environment.step(action)
                reward_sum += reward
                            
                next_state = np.zeros((state_size,), dtype=np.int)
                step = max_steps                                    
                scores.append(reward_sum)                    
                memory.add((state, action, reward, next_state, done))
           
            print('Episode: {}'.format(episode),
                     'Total reward: {}'.format(reward_sum))
                
            reward_sum = 0
            
        environment.reset()
            
        print(np.mean(scores))
        plt.hist(scores)
        plt.xlabel('Distribution of Scores')
        plt.ylabel('Relative Frequency')
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
                
    
    elif dql == True:
        
        loss = []
        
        model = DeepQNetworkMM(n_units=n_units, 
                               n_classes=n_classes, 
                               state_size=state_size, 
                               action_size=action_size, 
                               learning_rate=learning_rate)

        #tf.summary.scalar('Loss', model.error_rate)
        

        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            decay_step = 0
    
            for episode in range(n_episodes):
                
                current_step, reward_sum = 0, []
                state = np.reshape(observation, [1, state_size])    
    
                while current_step < max_steps:
                    
                    current_step += 1; decay_step += 1
                    
                    action, explore_probability = exploit_explore(session=sess,
                                                                  model=model,
                                                                  explore_start=explore_start, 
                                                                  explore_stop=explore_stop, 
                                                                  decay_rate=decay_rate, 
                                                                  decay_step=decay_step, 
                                                                  state=state, 
                                                                  actions=possible_actions)
                    
                    state, reward, done, info = environment.step(action)
                    reward_sum.append(reward)
                    
                    if current_step >= max_steps:
                        done = True
                                                
                    if done == True:
                        
                        next_state = np.zeros((state_size,), dtype=np.int)
                        step = max_steps                    
                        total_reward = np.sum(reward_sum)                    
                        scores.append(total_reward)                    
                        memory.add((state, action, reward, next_state, done))
                       
                        print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Loss: {}'.format(error_rate),
                                  'Explore P: {:.4f}'.format(explore_probability))
                        
                        loss.append(error_rate)
    
                    elif done != True:
                        
                        next_state = environment.reset()
                        state = next_state
                        memory.add((state, action, reward, next_state, done))
    
                    batch = memory.sample(batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch]) 
                    next_states = np.array([each[3] for each in batch])
                    dones = np.array([each[4] for each in batch])
    
                    target_Qs_batch = []
                    
                    Qs_next_state = sess.run(model.predicted_Q, feed_dict={model.input_matrix: next_states, model.actions: actions})
                    
                    for i in range(0, len(batch)):
                        terminal = dones[i]
    
                        if terminal:
                            target_Qs_batch.append(rewards[i])
                            
                        else:
                            target = rewards[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                            
                        
                    targets = np.array([each for each in target_Qs_batch])
    
                    error_rate, _ = sess.run([model.error_rate, model.optimizer], 
                                              feed_dict={model.input_matrix: states,
                                                         model.target_Q: targets,
                                                         model.actions: actions})
            if episode == n_episodes - 1:
                
                market_making(model=model,
                              environment=environment,
                              sess=sess,
                              state=state,
                              dpl=True)
    
    elif pg == True:
        
        loss = []
            
        mlp_model = PolicyGradient(n_units=n_units, 
                                  n_layers=n_layers, 
                                  n_columns=8, 
                                  n_outputs=n_classes, 
                                  learning_rate=learning_rate, 
                                  hidden_activation='selu', 
                                  output_activation='softmax',
                                  loss_function='categorical_crossentropy')
        
        policy_model, model_predictions = mlp_model.create_policy_model(input_shape=(len(observation), ))
        
        policy_model.summary()   
       
        while n_episode < n_episodes: 
         
            state = observation.reshape(1, 8)    
            prediction = model_predictions.predict([state])[0]
            action = np.random.choice(range(len(SpreadTrading._actions)), p=prediction)
            action = possible_actions[action]
            states = np.vstack([states, state])
            actions = np.vstack([actions, action])
            
            observation, reward, episode_done, info = environment.step(action)
            reward_sum += reward
            rewards = np.vstack([rewards, reward])
            step += 1
            
            if step == max_steps: 
                episode_done = True
    
            if episode_done == True:
                
                discounted_reward = calculate_discounted_reward(rewards, gamma=gamma)
                discounted_rewards = np.vstack([discounted_rewards, discounted_reward])
                
                discounted_rewards -= discounted_rewards.mean()
                discounted_rewards /= discounted_rewards.std()
                discounted_rewards = discounted_rewards.squeeze()
                actions = actions.squeeze().astype(int)
                
                #train_actions = np.zeros([len(actions), n_actions])
                #train_actions[np.arange(len(actions)), actions] = 1
                
                error = policy_model.train_on_batch([states, discounted_rewards], actions)
                loss.append(error)
                
                states = np.empty(0).reshape(0, 8)
                actions = np.empty(0).reshape(0, 3)
                rewards = np.empty(0).reshape(0, 1)
                discounted_rewards = np.empty(0).reshape(0, 1)
                                
                score = score_model(model=model_predictions, n_tests=10)
                
                print('''\nEpisode: %s \nAverage Reward: %s  \nScore: %s \nError: %s'''
                      )%(n_episode+1, reward_sum/float(batch_size), score, np.mean(loss[-batch_size:]))
    
                if score >= goal: 
                    break 
                
                reward_sum = 0
                    
                n_episode += 1
                observation = environment.reset()
                
            if n_episode == n_episodes - 1:
                
                market_making(model=model_predictions,
                              environment=environment,
                              sess=None,
                              state=state,
                              pg=True)
             
    if baseline != True:
        
        plt.title('Policy Gradient Error plot over %s Episodes'%(n_episode+1))
        plt.xlabel('N batches')
        plt.ylabel('Error Rate')
        plt.plot(loss)
        plt.show()
        plt.waitforbuttonpress()
        return model
        
def market_making(model, environment, sess, state, dpl=None, pg=None):    
    
    scores = []
    total_reward = 0
    environment.reset()
    
    for _ in range(1000):
                
        for __ in range(100):
            
            state = np.reshape(state, [1, state_size])  
            
            if dpl == True:
                Q_matrix = sess.run(model.output_layer, feed_dict = {model.input_matrix: state.reshape((1, 8))})
                choice = np.argmax(Q_matrix)
                action = possible_actions[int(choice)]

            elif pg == True:
                state = np.reshape(state, [1, 8])
                predict = model.predict([state])[0]
                action = np.argmax(predict)
                action = possible_actions[int(action)]
                                
            state, reward, done, info = environment.step(action)
            total_reward += reward

                
        print('Episode: {}'.format(_),
              'Total reward: {}'.format(total_reward))
        scores.append(total_reward)
        state = environment.reset()

    print(np.mean(scores))
    plt.hist(scores)
    plt.xlabel('Distribution of Scores')
    plt.ylabel('Relative Frequency')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    
if __name__ == '__main__':
    
    
    train_model(environment=environment, dql=True)
    
        
    
    
    

