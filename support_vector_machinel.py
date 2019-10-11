# -*- coding: utf-8 -*-
"""
Created on Thu Oct  10 23:33:03 2019

@author: Luis Rodriguez
"""

from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split


data_set=pd.read_csv('model_dataset_vf.csv')
print(data_set.describe())

#print(data_set)


X = data_set.drop(columns='class')
Y = data_set['class']

# We will use the 80% of the data for the training set and 20% for the testing set
# Since the accuracy of the model increased 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

clf=SVC(kernel='linear').fit(X_train,Y_train)

print(clf.score(X_test,Y_test))
