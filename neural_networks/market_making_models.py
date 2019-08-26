#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:42:23 2019

@author: tawehbeysolow
"""

import tensorflow as tf, numpy as np
from collections import deque 


activation_dictionary = {'elu': tf.nn.elu,
                         'relu': tf.nn.relu, 
                         'selu': tf.nn.selu, 
                         'sigmoid': tf.nn.sigmoid,
                         'softmax': tf.nn.softmax,
                          None: None}

def fully_connected_layer(inputs, units, activation, gain=np.sqrt(2)):
    
    return tf.layers.dense(inputs=inputs, 
                           units=units, 
                           activation=activation_dictionary[activation],
                           kernel_initializer=tf.orthogonal_initializer(gain))
    
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


class DeepQNetworkMM():
    
    def __init__(self, n_units, n_classes, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.n_classes = n_classes
        
        self.input_matrix = tf.placeholder(tf.float32, [None, state_size])
        self.actions = tf.placeholder(tf.float32, [None, n_classes])
        self.target_Q = tf.placeholder(tf.float32, [None])
        
        
        self.layer1 = fully_connected_layer(inputs=self.input_matrix,
                                                 units=self.n_units,
                                                 activation='selu')
        
        self.hidden_layer = fully_connected_layer(inputs=self.layer1,
                                                  units=self.n_units,
                                                  activation='selu')
        
        self.output_layer = fully_connected_layer(inputs=self.hidden_layer,
                                                  units=n_classes,
                                                  activation=None)
        
        self.predicted_Q = tf.reduce_sum(tf.multiply(self.output_layer, self.actions), axis=1)
        
        self.error_rate = tf.reduce_mean(tf.square(self.target_Q - self.predicted_Q))
        
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.error_rate)
    
