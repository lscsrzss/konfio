# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:08:59 2019

@author: Luis Rodriguez
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


data_set=pd.read_csv('model_dataset_vf.csv')
X = data_set.drop(columns='class')
Y = data_set['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(6,6,6),solver='adam',max_iter=100000)

mlp.fit(X_train,Y_train)
Y_pred=mlp.predict(X_test)

print('The accuracy of the Neural Network Classifier Model on test set is: ',mlp.score(X_test,Y_test),'%','\n')

print(classification_report(Y_test,Y_pred))


