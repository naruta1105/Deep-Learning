# MountainCar-v0 Enviroment

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:12:19 2021

@author: tranxuandien

pip install gym

https://gym.openai.com/docs/
pip install --upgrade pyglet
"""

import gym

env = gym.make('MountainCar-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
