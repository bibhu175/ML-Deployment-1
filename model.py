# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:17:33 2019

@author: BIBHUTI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace = True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace = True)

X = dataset.iloc[:,:3]

# convering words into integer value
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 
                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12,'zero':0,0:0}
    return word_dict[word]

X['experience']= X['experience'].apply(lambda x: convert_to_int(x))


y = dataset.iloc[:,-1]

# splitting training set and test set
# since we have a small data set so we will train our data with all available data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fitting model with training data
regressor.fit(X, y)

# saving model to disc
pickle.dump(regressor,open('model.pkl', 'wb'))


# loading the model to compare the result
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6]]))