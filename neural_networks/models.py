#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:49:13 2019

@author: tawehbeysolow
"""

import tensorflow as tf, numpy as np
from baselines.common.distributions import make_pdtype
        
activation_dictionary = {'elu': tf.nn.elu,
                         'relu': tf.nn.relu, 
                         'selu': tf.nn.selu, 
                         'sigmoid': tf.nn.sigmoid,
                         'softmax': tf.nn.softmax,
                          None: None}
            
def normalized_columns_initializer(standard_deviation=1.0):
  def initializer(shape, dtype=None, partition_info=None):
    output = np.random.randn(*shape).astype(np.float32)
    output *= standard_deviation/float(np.sqrt(np.square(output).sum(axis=0, keepdims=True)))
    return tf.constant(output)
  return initializer

def linear_operation(x, size, name, initializer=None, bias_init=0):
  with tf.variable_scope(name):
    weights = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer)
    biases = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, weights) + biases

def convolution_layer(inputs, dimensions, filters, kernel_size, strides, gain=np.sqrt(2), activation='relu'):
    
    if dimensions == 3:
    
        return tf.layers.conv1d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                kernel_initializer=tf.orthogonal_initializer(gain),
                                strides=(strides),
                                activation=activation_dictionary[activation])
    elif dimensions == 4:
        
        return tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                kernel_initializer=tf.orthogonal_initializer(gain),
                                strides=(strides),
                                activation=activation_dictionary[activation])


def fully_connected_layer(inputs, units, activation, gain=np.sqrt(2)):
    return tf.layers.dense(inputs=inputs, 
                           units=units, 
                           activation=activation_dictionary[activation],
                           kernel_initializer=tf.orthogonal_initializer(gain))

def lstm_layer(input, size, actions, apply_softmax=False):
      input = tf.expand_dims(input, [0])
      lstm = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True)
      state_size = lstm.state_size
      step_size = tf.shape(input)[:1]
      cell_init = np.zeros((1, state_size.c), np.float32)
      hidden_init = np.zeros((1, state_size.h), np.float32)
      initial_state = [cell_init, hidden_init]
      cell_state = tf.placeholder(tf.float32, [1, state_size.c])
      hidden_state = tf.placeholder(tf.float32, [1, state_size.h])
      input_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
      
      _outputs, states = tf.nn.dynamic_rnn(cell=lstm,
                                           inupts=input,
                                           initial_state=input_state,
                                           sequence_length=step_size,
                                           time_major=False)
      _cell_state, _hidden_state = states
      output = tf.reshape(_outputs, [-1, size])
      output_state = [_cell_state[:1, :], _hidden_state[:1, :]]
      output = linear_operation(output, actions, "logits", normalized_columns_initializer(0.01))
      output = tf.nn.softmax(output, dim=-1)
      return output, _cell_state, _hidden_state

def create_weights_biases(n_layers, n_units, n_columns, n_outputs):
    '''
    Creates dictionaries of variable length for differing neural network models
    
    Arguments 
    
    n_layers - int - number of layers 
    n_units - int - number of neurons within each individual layer
    n_columns - int - number of columns within dataset
    
    :return: dict (int), dict (int)
    '''
    weights, biases = {}, {}
    for i in range(n_layers):
        if i == 0: 
            weights['layer'+str(i)] = tf.Variable(tf.random_normal([n_columns, n_units]))
            biases['layer'+str(i)] = tf.Variable(tf.random_normal([n_columns]))
        elif i != 0 and i != n_layers-1:
            weights['layer'+str(i)] = tf.Variable(tf.random_normal([n_units, n_units]))
            biases['layer'+str(i)] = tf.Variable(tf.random_normal([n_units]))
        elif i != 0 and i == n_layers-1:
            weights['output_layer'] = tf.Variable(tf.random_normal([n_units, n_outputs]))
            biases['output_layer'] = tf.Variable(tf.random_normal([n_outputs]))
            
    return weights, biases

def create_input_output(input_dtype, output_dtype, n_columns, n_outputs):
    '''
    Create placeholder variables for tensorflow graph
    
    '''   
    X = tf.placeholder(shape=(None, n_columns), dtype=input_dtype)
    Y = tf.placeholder(shape=(None, n_outputs), dtype=output_dtype)
    return X, Y


class DeepQNetwork():
    
    def __init__(self, n_units, n_classes, n_filters, stride, kernel, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.stride = stride
        self.kernel = kernel
        
        self.input_matrix = tf.placeholder(tf.float32, [None, state_size])
        self.actions = tf.placeholder(tf.float32, [None, n_classes])
        self.target_Q = tf.placeholder(tf.float32, [None])
            
        
        self.network1 = convolution_layer(inputs=self.input_matrix, 
                                     filters=self.n_filters, 
                                     kernel_size=self.kernel, 
                                     strides=self.stride,
                                     dimensions=4,
                                     activation='elu')
        
        self.network1 = tf.layers.batch_normalization(self.network1,
                                                 training=True,
                                                 epsilon=1e-5)    

        self.network2 = convolution_layer(inputs=self.network1, 
                                     filters=self.n_filters*2, 
                                     kernel_size=int(self.kernel/2), 
                                     strides=int(self.stride/2), 
                                     dimensions=4,
                                     activation='elu')
     
        self.network2 = tf.layers.batch_normalization(inputs=self.network2,
                                                 training=True,
                                                 epsilon=1e-5)

        self.network3 = convolution_layer(inputs=self.network2, 
                                     filters=self.n_filters*4, 
                                     kernel_size=int(self.kernel/2), 
                                     strides=int(self.stride/2),
                                     dimensions=4,
                                     activation='elu')
     
        self.network3 = tf.layers.batch_normalization(inputs=self.network3,
                                                      training=True,
                                                      epsilon=1e-5)

        self.network3 = tf.layers.flatten(inputs=self.network3)
        
        self.output = fully_connected_layer(inputs=self.network3, 
                                            units=self.n_units,
                                            activation='elu')
        
        self.output = fully_connected_layer(inputs=self.output,
                                            units=n_classes,
                                            activation=None)
        
        self.predicted_Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        
        self.error_rate = tf.reduce_mean(tf.square(self.target_Q - self.predicted_Q))
        
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.error_rate)
 
    
class ActorCriticModel():
    
    def __init__(self, session, environment, action_space, n_batches, n_steps, reuse=False):
        
        session.run(tf.global_variables_initializer())
        self.distribution_type = make_pdtype(action_space)
        height, weight, channel = environment.shape
        inputs_ = tf.placeholder(tf.float32, [height, weight, channel], name='inputs')
        scaled_images = tf.cast(inputs_, tf.float32)/float(255)
        
        with tf.variable_scope('model', reuse=reuse):

            layer1 = tf.layers.batch_normalization(convolution_layer(inputs=scaled_images, 
                                                                     filters=32, 
                                                                     kernel_size=8, 
                                                                     strides=4,
                                                                     dimensions=3))
                        
            layer2 = tf.layers.batch_normalization(convolution_layer(inputs=tf.nn.relu(layer1), 
                                                                     filters=64, 
                                                                     kernel_size=4, 
                                                                     strides=2,
                                                                     dimensions=3))
            
            layer3 = tf.layers.batch_normalization(convolution_layer(inputs=tf.nn.relu(layer2), 
                                                                     filters=64, 
                                                                     kernel_size=3, 
                                                                     strides=1,
                                                                     dimensions=3))
            
            layer3 = tf.layers.flatten(inputs=layer3)
            output_layer = fully_connected_layer(inputs=layer3, units=512, activation='softmax')
            self.distribution, self.logits = self.distribution_type.pdfromlatent(output_layer, init_scale=0.01)
            value_function = fully_connected_layer(output_layer, units=1, activation=None)[:, 0]
            
        self.initial_state = None
        sampled_action = self.distribution.sample()
        
        def step(current_state, *_args, **_kwargs):
            action, value = session.run([sampled_action, value_function], {inputs_: current_state})
            return action, value

        def value(current_state, *_args, **_kwargs):
            return session.run(value_function, {inputs_: current_state})

        def select_action(current_state, *_args, **_kwargs):
            return session.run(sampled_action, {inputs_: current_state})

        self.inputs_ = inputs_
        self.value_function = value_function
        self.step = step
        self.value = value
        self.select_action = select_action
    
        
class A3CModel():
    
    def __init__(self, s_size, a_size, scope, trainer):
        
        with tf.variable_scope(scope):

            self.input_layer = tf.placeholder(shape=[None, s_size], 
                                         dtype=tf.float32)
            
            self.input_layer = tf.reshape(self.input_layer, 
                                          shape=[-1,84,84,1])
            
            self.layer1 = tf.layers.batch_normalization(convolution_layer(inputs=input_layer, 
                                                                     filters=32, 
                                                                     kernel_size=8, 
                                                                     strides=4,
                                                                     dimensions=3))
                        
            self.layer2 = tf.layers.batch_normalization(convolution_layer(inputs=tf.nn.relu(layer1), 
                                                                     filters=64, 
                                                                     kernel_size=4, 
                                                                     strides=2,
                                                                     dimensions=3))
            
            layer3 = tf.layers.flatten(inputs=layer3)
            
            output_layer = fully_connected_layer(inputs=layer3, 
                                                 units=512, 
                                                 activation='softmax')
            
            outputs, cell_state, hidden_state = lstm_layer(input=hidden, 
                                                           size=s_size, 
                                                           actions=a_size, 
                                                           apply_softmax=False)
                    
            self.state_out = (cell_state[:1, :], hidden_state[:1, :])
            ouptut_layer = tf.reshape(outputs, [-1, 256])
            
            self.policy = slim.fully_connected(input=output_layer, 
                                               n_units=a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            
            self.value = slim.fully_connected(input=rnn_out,
                                              n_units=1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))



        