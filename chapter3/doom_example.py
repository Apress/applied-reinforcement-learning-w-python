#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:50:31 2019

@author: tawehbeysolow
"""

import warnings, random, time, tensorflow as tf, numpy as np, matplotlib.pyplot as plt  
from neural_networks.models import DeepQNetwork
from algorithms.dql_utilities import create_environment, stack_frames, Memory
from chapter3.frozen_lake_example import exploit_explore
from collections import deque 

#Parameters
stack_size = 4
gamma = 0.95
memory_size = int(1e7)
train = True
episode_render = False
n_units = 500
n_classes = 3
learning_rate = 2e-4
stride = 4 
kernel = 8
n_filters = 3
n_episodes = 1
max_steps = 100
batch_size = 64 
environment, possible_actions = create_environment()
state_size = [84, 84, 4]
action_size = 3 #environment.get_avaiable_buttons_size()
explore_start = 1.0
explore_stop = 0.01
decay_rate = 1e-4
pretrain_length = batch_size
warnings.filterwarnings('ignore')
#writer = tf.summary.FileWriter("/tensorboard/dqn/1")
write_op = tf.summary.merge_all()

def exploit_explore(session, model, explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
        
    else:
        Qs = session.run(model.output, feed_dict = {model.input_matrix: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability

def train_model(model, environment):
    tf.summary.scalar('Loss', model.error_rate)
    saver = tf.train.Saver()
    stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 
    memory = Memory(max_size=memory_size)
    scores = []
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        environment.init()

        for episode in range(n_episodes):
            step, reward_sum = 0, []
            environment.new_episode()
            state = environment.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1; decay_step += 1
                
                action, explore_probability = exploit_explore(session=sess,
                                                              model=model,
                                                              explore_start=explore_start, 
                                                              explore_stop=explore_stop, 
                                                              decay_rate=decay_rate, 
                                                              decay_step=decay_step, 
                                                              state=state, 
                                                              actions=possible_actions)
                    
                reward = environment.make_action(action)
                done = environment.is_episode_finished()
                reward_sum.append(reward)

                if done:
                    
                    next_state = np.zeros((84,84), dtype=np.int)
                    
                    next_state, stacked_frames = stack_frames(stacked_frames=stacked_frames, 
                                                              state=next_state, 
                                                              new_episode=False)
                    step = max_steps
                    
                    total_reward = np.sum(reward_sum)
                    
                    scores.append(total_reward)
                    
                    
                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))

                else:
                    next_state = environment.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state


                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch], ndmin=3)
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch]) 
                next_states = np.array([each[3] for each in batch], ndmin=3)
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
                '''                
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={model.inputs_: states,
                                                   model.target_Q: targets,
                                                   model.actions_: actions})

                writer.add_summary(summary, episode)
                writer.flush()
              

            if episode % 5 == 0:
                #saver.save(sess, filepath+'/models/model.ckpt')
                #print("Model Saved")
                '''
    
    plt.plot(scores)
    plt.title('DQN Performance During Training')
    plt.xlabel('N Episodes')
    plt.ylabel('Score Value')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    return model
  
    
def play_doom(model, environment):
    
    stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 
    scores = []
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        totalScore = 0
        
        for _ in range(100):
            
            done = False
            environment.new_episode()
            
            state = environment.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
                
            while not environment.is_episode_finished():
                
                Q_matrix = sess.run(model.output, feed_dict = {model.input_matrix: state.reshape((1, *state.shape))})
                choice = np.argmax(Q_matrix)
                action = possible_actions[int(choice)]
                
                environment.make_action(action)
                done = environment.is_episode_finished()
                score = environment.get_total_reward()
                scores.append(score)
                time.sleep(0.01)
                
                if done:
                    break  
                                        
            score = environment.get_total_reward()
            print("Score: ", score)
            
        environment.close()
        
    plt.plot(scores)
    plt.title('DQN Performance After Training')
    plt.xlabel('N Episodes')
    plt.ylabel('Score Value')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    
if __name__ == '__main__':
    
    
    model = DeepQNetwork(n_units=n_units, 
                         n_classes=n_classes, 
                         n_filters=n_filters, 
                         stride=stride, 
                         kernel=kernel, 
                         state_size=state_size, 
                         action_size=action_size, 
                         learning_rate=learning_rate)
    
    trained_model = train_model(model=model,
                                environment=environment)
    
    play_doom(model=trained_model,
              environment=environment)