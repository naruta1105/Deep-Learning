# Snake Deep Q Learning : Experience Replay Memory file

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:10:53 2021

@author: tranxuandien
"""

# Import Libraries
import numpy as np

class Dqn():
     def __init__(self,maxMemory, discount): # discount = gamma in Q-learning
         self.maxMemory = maxMemory
         self.discount = discount
         self.memory = list()
     
     # Remembering new experience
     # transition = currentStage, currentAction, currentreward, nestStage
     def remember(self,transition, gameOver):
         self.memory.append([transition,gameOver])
         if len(self.memory)>self.maxMemory :
             del self.memory[0]
             
     # Getting batches of input and targets
     # numInputs = currentStage.shape[1] (1 because keras do that)
     def getBatch(self, model, batchSize):
        lenMemory = len(self.memory)
        # numInputs = self.memory[0][0][0].shape[1] -> in convolution networl, we don't know numInput
        numOutputs = model.output_shape[-1]
        
        #Initializing the inputs and targets:
        # inputs = np.zeros((min(batchSize,lenMemory),numInputs))
        # shape[1] = width, shape[2]= height, shape[3] = number of frame
        inputs = np.zeros((min(batchSize,lenMemory),self.memory[0][0][0].shape[1],
                           self.memory[0][0][0].shape[2],self.memory[0][0][0].shape[3]))
        targets = np.zeros((min(batchSize,lenMemory),numOutputs))
        
        # Extracting the transitions from random experience
        randomIndex = np.random.randint(0, lenMemory, 
                                        size = min(batchSize, lenMemory))
        for i, inx in enumerate(randomIndex):
            currentStage, action, reward, nextStage = self.memory[inx][0]
            gameOver = self.memory[inx][1]
            
            # Updating inputs and targets
            inputs[i] = currentStage
            targets[i] = model.predict(currentStage)[0]
            # Applying Q Learning
            if gameOver : # if lose, penalty it
                targets[i][action] = reward
            else :
                targets[i][action] = reward + self.discount*np.max(model.predict(nextStage)[0])
        
        return inputs, targets
    
if __name__ == "__main__":
    pass