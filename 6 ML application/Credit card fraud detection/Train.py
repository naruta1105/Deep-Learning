#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:56:06 2020

@author: tranxuandien
"""

### Importing Librabry
import pandas as pd
import numpy as np

np.random.seed(2)

### Import Dataset
datasets= pd.read_csv('Dataset/creditcard.csv')
## Data exploration
datasets.shape
datasets.head()
datasets.columns
datasets.describe()

# Feature Sclaing
from sklearn.preprocessing import StandardScaler
datasets['normalizedAmount'] =  StandardScaler().fit_transform(datasets['Amount'].values.reshape(-1, 1))
datasets = datasets.drop(['Amount'], axis = 1)

datasets = datasets.drop(['Time'], axis = 1)

X = datasets.iloc[:, datasets.columns != 'Class']
y = datasets.iloc[:, datasets.columns == 'Class']

### Spliting train and test data
## 5.Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train.shape
X_test.shape

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

## Building the ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout
### Initializing the ANN
ann = Sequential()
### Adding the input layer and the first hidden layer
#unit = number of neutron you want. activation='relu' mean activation function is rectifier
# input_dim = 29 because data have 29 columns
ann.add(Dense(units=16, input_dim = 29,  activation='relu'))
### Adding the second hidden layer
ann.add(Dense(units=24, activation='relu'))
### Adding a Dropout (bỏ bớt các weight và neutron ko xài đến)
ann.add(Dropout(0.5))
### Adding the third hidden layer
ann.add(Dense(units=20, activation='relu'))
### Adding the fourth hidden layer
ann.add(Dense(units=24, activation='relu'))
### Adding the output layer
# output  = [0,1] => number of neutron = 1 (dimension of output). If output = [A,B,C] => number of neutron = 3 because there is no connection of A,B or C
# output is binary so we use activation function threshold or sigmoid.
# use sigmoid for not only predict 0 or 1 but also the probability that 0 or 1 sigmoid
ann.add(Dense(units=1, activation='sigmoid'))

### for short
ann = Sequential([
    Dense(units=16, input_dim = 29,  activation='relu'),
    Dense(units=24, activation='relu'),
    Dropout(0.5),
    Dense(units=20, activation='relu'),
    Dense(units=24, activation='relu'),
    Dense(units=1, activation='sigmoid')
    ])

ann.summary()


## Training the ANN
### Compiling the ANN
# optimizer để xác định phương thức update lại weight cho mạng neutron
# optimizer = 'adam' mean using stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
# loss để xác định phương thức so sánh y^ và y thực tế
# loss = 'binary_crossentropy' dùng cho các bài toán về binary. còn về phân loại ta có thể dùng 'CategoricalCrossentropy' (output layer dùng activation = softmax - khi đó tổng % các output = 100  )
ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])
### Training the ANN on the Training set
# batch_size = 32 mean after every 32 prediction, we will compare them with 32 real result at once (not one by one)
# epochs mean how many times are going to be presenting or presenting dataset and update the weight as we go
ann.fit(X_train, y_train, batch_size = 15, epochs = 5)

score = ann.evaluate(X_test, y_test, batch_size = 1)
print(score)

y_pred = ann.predict(X_test)
y_test = pd.DataFrame(y_test)
## Making the Confusion Matrix (kiểm tra % đúng sai của dự đoán)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print(cm)



accuracy_score(y_test, y_pred)
# The precision ability of the classifier not to label as positive a sample that is negative.
# true positives / (true positive + false positives)
precision_score(y_test, y_pred)
# The recall is  ability of the classifier to find all the positive samples.
# (True positive/ (True positive + False Negative))
recall_score(y_test, y_pred)
# the Specificity = (True Negatives/(False Positive + True Negatives)
# F1 = 2 * (precision * recall) / (precision + recall)
f1_score(y_test, y_pred)