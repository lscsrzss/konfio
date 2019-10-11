# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:16:35 2019

@author: Luis Rodriguez
"""

# Binary Classification Model

# Training and Testing Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold




data_set=pd.read_csv('model_dataset_vf.csv')
print(data_set.describe())

#print(data_set)


X = data_set.drop(columns='class')
Y = data_set['class']




# We will use the 80% of the data for the training set and 20% for the testing set
# Since the accuracy of the model increased 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

X_train.fillna(0)
X_test.fillna(0)
Y_train.fillna(0)
X_test.fillna(0)


# Logistic Regression for Binary Classification

logistic_regresion_model = linear_model.LogisticRegression()
logistic_regresion_model.fit(X_train,Y_train)

# Testing the model

Y_pred = logistic_regresion_model.predict(X_test)
lrm_score=logistic_regresion_model.score(X_test, Y_test)*100

print('The accuracy of the logistic regression classifier model on test set is: ',lrm_score,'%')

confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)

  









