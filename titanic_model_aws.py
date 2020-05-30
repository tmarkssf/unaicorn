#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('titanic_train.csv')

def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age
    
    
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train=pd.concat([train, sex, embark], axis=1)
train.drop('PassengerId', axis=1, inplace=True)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

logreg_model = LogisticRegression(max_iter=500)
logreg_model.fit(X_train, y_train)

prediction = logreg_model.predict(X_test)

pickle.dump(logreg_model, open('titanic_model_aws.sav', 'wb'))

