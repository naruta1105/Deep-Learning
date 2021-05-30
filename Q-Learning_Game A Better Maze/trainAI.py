#A Better Maze: Q-Learning - Training file

# Import Libraries
from environment import Environment
import numpy as np

# Defining the parametrer
gamma = 0.9
alpha = 0.75
nEpouchs = 10000 # how many time find the ways

# Enviroment and Q-Table initialization
env = Environment()
rewards = env.rewardBoard
QTable = rewards.copy()

#Preparing the Q-Learning process 1
possibleState = list()
for i in range(rewards.shape[0]): # rows
    if sum(abs(rewards[i])) != 0: # if i is not a block because block can not go other tile
        possibleState.append(i) # list of possible tile for start
     
#Preparing the Q-Learning process 2
def maximun(qvalues):
    inx = 0
    maxQValue = - np.inf
    for i in range(len(qvalues)):
        if (qvalues[i]>maxQValue) and qvalues[i] != 0 : # !=0 mean can go to tile
            maxQValue = qvalues[i]
            inx = i
    return inx, maxQValue

# Starting the Q Learning process
for epouch in range(nEpouchs):
    print('\rEpouch: '+str(epouch+1),end = '')
    
    startingPos = np.random.choice(possibleState)
    
    # Getting all the playable actions
    possibleAction = list()
    for i in range(rewards.shape[1]): #colums
        if rewards[startingPos][i] != 0: #choose possible tile
            possibleAction.append(i)
            
    #Playing random action
    action = np.random.choice(possibleAction)
    
    reward = rewards[startingPos][action]
    
    #Updating the Q value
    _, maxQValue = maximun(QTable[action])
    
    TD = reward + gamma*maxQValue - QTable[startingPos][action]
    
    QTable[startingPos][action] = QTable[startingPos][action] + alpha*TD
    
#Displaying the results
currentPos = env.startingPos
while True:
    action, _ = maximun(QTable[currentPos])
    env.movePlayer(action)
    currentPos = action











    