# Snake: Deep Q Learning : Brain file
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
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Building the Brain class
class Brain():
    def __init__(self,inputShape, lr=0.005):
        # inputShape is 3D: a list of frame(image2D) of game
        self.inputShape = inputShape 
        self.learningRate = lr
        self.numOutputs = 4 # up=0, down=1, left=3, right=2 
        
        # Create a neural network
        # Initalizing network
        self.model = Sequential()
        # Input layers with convolution network
        # First add input layer
        self.model.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu', input_shape = self.inputShape))
        # Applying Max pooling
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        # Add one more convolution layer
        self.model.add(Conv2D(filters = 64, kernel_size=(3,3), activation='relu'))
        # Flatten it
        self.model.add(Flatten())
        
        # Hidden layers with neuron = 256
        self.model.add(Dense(units = 256, activation='relu'))
        
        # Output layers with neuron = numOutput and activation = None
        # activation = None mean activation = linear regression
        # In deep Q learning the activation of output should be linear requression
        # sometime it work better than softmax
        self.model.add(Dense(self.numOutputs))
        # Add optimizer
        self.model.compile(optimizer=Adam(lr = self.learningRate), loss = 'mean_squared_error')
    
    # Building a method to load a model
    def loadModel(self,filePath):
        self.model = load_model(filePath)
        return self.model

if __name__ == "__main__":
    pass