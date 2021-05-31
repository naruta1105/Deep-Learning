#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:52:31 2020

@author: tranxuandien
"""
#### Importing Libraries ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('datasets/churn_data.csv') # Users who were 60 days enrolled, churn in the next 30


#### EDA ####

# Viewing the Data
dataset.head(5)
dataset.columns
dataset.describe() # Distribution of Numerical Variables

# Cleaning Data
dataset[dataset.credit_score < 300]
dataset = dataset[dataset.credit_score >= 300]

# Removing NaN
dataset.isna().any()
dataset.isna().sum()
dataset = dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])


## Histograms
dataset2 = dataset.drop(columns = ['user', 'churn'])
# size of each figure (X=15, y = 12)
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    # there will be 6 row x 5 column of figure
    plt.subplot(6, 5, i)
    f = plt.gca()
    # disable value of y axis
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
# A rectangle (left, bottom, right, top) in the normalized figure coordinate 
# that the whole subplots area (including labels) will fit into. Default is (0, 0, 1, 1).
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Pie Plots
# use for only binary cause (not number)
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    
    # normalize = True mean use percen
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    # autopct='%1.1f%%' = how to display distribution
    plt.pie(values, labels = index, autopct='%1.1f%%')
    # no x or y axis show
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


## Exploring Uneven Features
# check features which have very small diiferent in value with response(churn)
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.cancelled_loan == 1].churn.value_counts()
dataset[dataset2.received_loan == 1].churn.value_counts()
dataset[dataset2.rejected_loan == 1].churn.value_counts()
dataset[dataset2.left_for_one_month == 1].churn.value_counts()



## Correlation with Response Variable
## Quan hệ tương quan giưa các cột thông số với response (ta loại bỏ các
## cột chứa thông tin khác số)
dataset2.drop(columns = ['housing', 'payment_type',
                         'registered_phones', 'zodiac_sign']
    ).corrwith(dataset.churn).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 15, rot = 45, color = list('rgbkymc'),
              grid = True)


## Correlation Matrix
sn.set(style="white")
# Compute the correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr()
# Generate a mask for the upper triangle
# Che nửa trên của ma trận quan hệ
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Removing Correlated Fields
# thực tế cho thấy app_web_user phụ thuộc vào app_downloađe và web_users
# vì train ko được dùng cột phụ thuộc nên ta bỏ
dataset = dataset.drop(columns = ['app_web_user'])

## Note: Although there are somewhat correlated fields, they are not colinear
## These feature are not functions of each other, so they won't break the model
## But these feature won't help much either. Feature Selection should remove them.

dataset.to_csv('new_churn_data.csv', index = False)