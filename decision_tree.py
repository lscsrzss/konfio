# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:42:46 2019

@author: Luis Rodriguez
"""

# Decision Tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


data_set=pd.read_csv('model_dataset_vf.csv')
X = data_set.drop(columns='class')
Y = data_set['class']

X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = 0.2,random_state = 0)


tree = DecisionTreeClassifier(max_depth=6, random_state=10)
tree.fit(X_train, Y_train)
Y_pred = tree.predict(X_test)


dtc_score=tree.score(X_test, Y_test)*100

print('The accuracy of the decision tree classifier model on test set is: ',dtc_score,'%','\n')

print(classification_report(Y_test,Y_pred))







