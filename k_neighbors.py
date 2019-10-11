# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:12:57 2019

@author: Luis Rodriguez
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

data_set=pd.read_csv('model_dataset_vf.csv')
X = data_set.drop(columns='class')
Y = data_set['class']


X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = 0.2,random_state = 0)


neighbors=np.arange(1,600)
train_ex=np.empty(len(neighbors))
test_ex=np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    train_ex[i]=knn.score(X_train,Y_train)
    test_ex[i]=knn.score(X_test,Y_test)
    
plt.title('Knn Accuracy Variation')
plt.plot(neighbors,test_ex,label='Test Accuracy')   
plt.plot(neighbors,train_ex,label='Train Accuracy')   
plt.legend()
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')

a=test_ex.mean()*100

print('The accuracy of the K Nearest Neighbors model on test set is: ',a,'%')

print(classification_report(Y_test,Y_pred))
