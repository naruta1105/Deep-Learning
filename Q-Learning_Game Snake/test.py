#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:19:19 2021

@author: tranxuandien
"""

# Import Libraries
from environment import Environment
from brain import Brain
import numpy as np

# Defining the parameters
waitTime = 75
nLastState = 4
filepathToOpen = 'model.h5'

# Initializing Enviroment, Brain
env = Environment(waitTime)
# input_shape : width = env.nColumns, height = env.nRows, numberOfFrame= nLastState
# output = 4 (up=0, down=1, left=3, right=2)
brain = Brain((env.nColumns, env.nRows, nLastState))
model = brain.loadModel(filepathToOpen)

# Building a function that reset currentState and nextState
def resetState():
    currentState = np.zeros((1,env.nColumns, env.nRows, nLastState))
    # we assign screenmap of game to every frame of currentState
    for i in range(nLastState):
        currentState[0,:,:,i] = env.screenMap
    return currentState, currentState

# Starting the main loop
nCollected = 0 # apple eaten in 1 game
maxNCollected = 0 # max Apple eaten
totNCollected = 0 # total Apple eaten every 100 game
scores = list() # list of total Apple every 100 game

while True:
    # Reseting the environment and starting play the game
    env.reset()
    currentState, nextState = resetState()
    gameOver = False
    
    while not gameOver:
        # Taking a action
        qvalues = model.predict(currentState)[0]
        action = np.argmax(qvalues) # return index of max qvalues
        
        # Updating the Enviroment
        frame , _, gameOver = env.step(action)
        # convert frame from 2D to 4D(same as State)
        frame = np.reshape(frame,(1,env.nColumns, env.nRows, 1))
        # append frame to nextStage, index = 3(nLastState)
        nextState = np.append(nextState, frame, axis = 3)
        # delete the oldest frame: index=0, axis =3 (nLastState)
        nextState = np.delete(nextState, 0, axis = 3)
        
        # Updating current state
        currentState = nextState