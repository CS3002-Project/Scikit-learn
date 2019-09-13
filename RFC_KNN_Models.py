#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#Using dataset from subject 13 with ideal placement of sensors.
data = pd.read_csv("subject13_ideal.log",delim_whitespace=True,header=None)

X = data.drop(119, axis = 1)
Y = data[119]

#Splitting dataset into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[ ]:


#Standardize features by removing the mean and scaling to unit variance
def standardise_dataset(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train, X_test

def train_model(model, X_train, y_train):
    clf = model
    clf.fit(X_train, y_train)
    return clf

def eval_model(clf, X_test, y_test):
    pred_clf = clf.predict(X_test)
    print(classification_report(y_test, pred_clf))
    

clf = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators = 20)

X_train, X_test = standardise_dataset(X_train, X_test)

clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)
print(classification_report(y_test, pred_clf))

knn_clf = train_model(KNeighborsClassifier(), X_train, y_train)
eval_model(knn_clf, X_test, y_test)

rf_clf = train_model(RandomForestClassifier(n_estimators = 20), X_train, y_train)
eval_model(rf_clf, X_test, y_test)

# rfc.fit(X_train,y_train)
# pred_rfc = rfc.predict(X_test)
# print(classification_report(y_test, pred_rfc))






# In[ ]:




