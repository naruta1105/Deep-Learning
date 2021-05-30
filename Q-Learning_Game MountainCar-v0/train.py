# MountainCar-v0 Deep Q Learning : Train file

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:10:54 2021

@author: tranxuandien

pip install gym

https://gym.openai.com/docs/
pip install --upgrade pyglet

pip install tensorflow
pip install keras
pip install matplotlib
conda install --update tensorflow -c anaconda
conda install -c intel mkl
conda list
"""

# Import Libraries
from dqn import Dqn
from brain import Brain
import gym
import numpy as np
import matplotlib.pyplot as plt


# Setting the parameters:
learningRate = 0.001
maxMemory = 5000 # 5000 is too low. game have 200 move => 5000 = 25 game => too low for AI remember, so sometime it will forgotten how to to play game 
gamma = 0.9
batchSize = 32
epsilon = 1. # completely random. we will you epsilon greedy
epsilonDecayRate = 0.995 #that will reduce epsilon every game

# Initializing Enviroment, Brain and Experience Replay Memory
env = gym.make('MountainCar-v0')
# read from document of MountainCar-v0
# input = 2 (position and velocity)
# output = 3 (left = 0, stay = 1, right = 2)
brain = Brain(2,3,learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

# Starting the main loop
epoch = 0
currentStage = np.zeros((1,2)) # 1 row, 2 column for 2 input
nextStage = currentStage
totReward = 0 # total Reward up to now
rewards = list()
number_training = 0
while True:
    epoch += 1
    
    # Starting play the game
    env.reset()
    currentStage = np.zeros((1,2))
    nextStage = currentStage
    gameOver = False
    while not gameOver:
        
        # Taking a action
        if np.random.rand() <= epsilon :
            action = np.random.randint(0,3)
        else:
            qvalues = model.predict(currentStage)[0]
            action = np.argmax(qvalues) # return index of max qvalues
        
        # Updating the Enviroment
        nextStage[0], reward, gameOver, _ = env.step(action)
        env.render()
        totReward += reward
        
        #Remembering the experience, training AI and update currentStage 
        DQN.remember([currentStage, action, reward, nextStage], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        print(f'Begin Train {number_training}')
        model.train_on_batch(inputs, targets)
        print(f'End Traing {number_training}')
        number_training += 1
        
        currentStage = nextStage
        
    #Lowering the epsilon and display result
    epsilon = epsilon*epsilonDecayRate
    
    print('Epoch : '+str(epoch)+' Epsilon : {:.5f}'.format(epsilon)+ ' Total Reward: {:.2f}'.format(totReward))
    
    rewards.append(totReward)
    totReward = 0
    
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Rewards')
    plt.show()
    
env.close()