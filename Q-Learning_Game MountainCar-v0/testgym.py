#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:12:19 2021

@author: tranxuandien

https://gym.openai.com/docs/
pip install --upgrade pyglet
"""

import gym
#env = gym.make('CartPole-v0')
#env = gym.make('')
#env = gym.make('MountainCar-v0')
env = gym.make('MsPacman-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

import tensorflow as tf
import os
os.environ['MKLDNN_VERBOSE'] = '1'
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
   print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled()) 
   
   
import numpy as np
import tensorflow as tf

x_in = np.array([[
  [[2], [1], [2], [0], [1]],
  [[1], [3], [2], [2], [3]],
  [[1], [1], [3], [3], [0]],
  [[2], [2], [0], [1], [1]],
  [[0], [0], [3], [1], [2]], ]])
kernel_in = np.array([
 [ [[2, 0.1]], [[3, 0.2]] ],
 [ [[0, 0.3]],[[1, 0.4]] ], ])
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')




conda install tensorflow tensorflow-base=2.3.0=mkl_py37h7075554_0
conda install -c intel tensorflow


