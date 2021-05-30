# MountainCar-v0 Deep Q Learning : Brain file
# Include input, output and learning rate of training

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:10:52 2021

@author: tranxuandien
pip uninstall Keras
"""

# Import Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

# Building the Brain class
class Brain():
    # Not creating XLA devices, tf_xla_enable_xla_devices not set fix
    #os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    def __init__(self,numInputs, numOutputs, lr):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.learningRate = lr
        
        # Create a neural network
        # Initalizing network
        self.model = Sequential()
        # Input layers with neuron = 32, activation = 'relu'
        self.model.add(Dense(units = 32, activation='relu', input_shape = (self.numInputs, )))
        # Hidden layers with neuron = 16 (<32 so will better)
        self.model.add(Dense(units = 16, activation='relu'))
        # Output layers with neuron = numOutput and activation = None
        # activation = None mean activation = linear regression
        # In deep Q learning the activation of output should be linear requression
        # sometime it work better than softmax
        self.model.add(Dense(units = self.numOutputs))
        # Add optimizer
        self.model.compile(optimizer=Adam(lr = self.learningRate), loss = 'mean_squared_error')

if __name__ == "__main__":
    pass