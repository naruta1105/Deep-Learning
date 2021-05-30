# Snake Deep Q Learning : Train file

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:10:54 2021

@author: tranxuandien
pip install tensorflow
pip install keras
pip install matplotlib
"""

# Import Libraries
from environment import Environment
from dqn import Dqn
from brain import Brain
import numpy as np
import matplotlib.pyplot as plt

# Setting the parameters:
learningRate = 0.0001
maxMemory = 60000 
gamma = 0.9
batchSize = 32
nLastState = 4 # How many frame stack on top -> to see where snake going

epsilon = 1. # completely random. we will you epsilon greedy
epsilonDecayRate = 0.0002 # this time we subtract 
minEpsilon = 0.05

filepathToSave = 'model1.h5'

# Initializing Enviroment, Brain and Experience Replay Memory
env = Environment(0)
# input_shape : width = env.nColumns, height = env.nRows, numberOfFrame= nLastState
# output = 4 (up=0, down=1, left=3, right=2)
brain = Brain((env.nColumns, env.nRows, nLastState),learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

# Building a function that reset currentState and nextState
def resetState():
    currentState = np.zeros((1,env.nColumns, env.nRows, nLastState))
    # we assign screenmap of game to every frame of currentState
    for i in range(nLastState):
        currentState[0,:,:,i] = env.screenMap
    return currentState, currentState



# Starting the main loop
epoch = 0
nCollected = 0 # apple eaten in 1 game
maxNCollected = 0 # max Apple eaten
totNCollected = 0 # total Apple eaten every 100 game
scores = list() # list of total Apple every 100 game

while True:
    epoch += 1
    
    # Reseting the environment and starting play the game
    env.reset()
    currentState, nextState = resetState()
    gameOver = False
    
    while not gameOver:
        # Taking a action
        if np.random.rand() <= epsilon :
            action = np.random.randint(0,4)
        else:
            qvalues = model.predict(currentState)[0]
            action = np.argmax(qvalues) # return index of max qvalues
        
        # Updating the Enviroment
        frame , reward, gameOver = env.step(action)
        # convert frame from 2D to 4D(same as State)
        frame = np.reshape(frame,(1,env.nColumns, env.nRows, 1))
        # append frame to nextStage, index = 3(nLastState)
        nextState = np.append(nextState, frame, axis = 3)
        # delete the oldest frame: index=0, axis =3 (nLastState)
        nextState = np.delete(nextState, 0, axis = 3)
        
        #Remembering the experience, training AI and update currentState 
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        #print(f'Begin Train {number_training}')
        model.train_on_batch(inputs, targets)
        #print(f'End Traing {number_training}')
        
        # Updating the score and current state
        if env.collected :
            nCollected += 1
        currentState = nextState
        
    #Lowering the epsilon and save the model
    epsilon = epsilon - epsilonDecayRate
    epsilon = max(epsilon, minEpsilon)
    
    if nCollected > maxNCollected and nCollected > 2:
        model.save(filepathToSave)
        maxNCollected = nCollected
    
    # Display result
    totNCollected += nCollected
    nCollected = 0
    
    if epoch % 100 == 0 and epoch != 0:
        scores.append(totNCollected/100)
        totNCollected = 0
        
        plt.plot(scores)
        plt.xlabel('Epoch / 100')
        plt.ylabel('Average Collected')
        plt.show()
    print('Epoch : '+str(epoch)+' Current Best: '+str(maxNCollected) +' Epsilon : {:.5f}'.format(epsilon))