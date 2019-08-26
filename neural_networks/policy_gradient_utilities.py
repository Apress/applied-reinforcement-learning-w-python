#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:22:27 2019

@author: tawehbeysolow
"""

import keras.layers as layers
from keras import backend
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform

class PolicyGradient():
    
    def __init__(self, n_units, n_layers, n_columns, n_outputs, learning_rate, hidden_activation, output_activation, loss_function):
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_outputs = n_outputs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def create_policy_model(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        advantages = layers.Input(shape=[1])
        
        hidden_layer = layers.Dense(units=self.n_units, 
                                    activation=self.hidden_activation,
                                    use_bias=False,
                                    kernel_initializer=glorot_uniform(seed=42))(input_layer)
        
        output_layer = layers.Dense(units=self.n_outputs, 
                                    activation=self.output_activation,
                                    use_bias=False,
                                    kernel_initializer=glorot_uniform(seed=42))(hidden_layer)
        
        def log_likelihood_loss(actual_labels, predicted_labels):
            log_likelihood = backend.log(actual_labels * (actual_labels - predicted_labels) + 
                                  (1 - actual_labels) * (actual_labels + predicted_labels))
            return backend.mean(log_likelihood * advantages, keepdims=True)
        
        if self.loss_function == 'log_likelihood':
            self.loss_function = log_likelihood_loss
        else:
            self.loss_function = 'categorical_crossentropy'
                
        policy_model = Model(inputs=[input_layer, advantages], outputs=output_layer)
        policy_model.compile(loss=self.loss_function, optimizer=Adam(self.learning_rate))
        model_prediction = Model(input=[input_layer], outputs=output_layer)
        return policy_model, model_prediction
