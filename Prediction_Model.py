#!/usr/bin/env python
# coding: utf-8


get_ipython().system('pip install scikit-learn')


pip install mixed-naive-bayes


import sklearn as sl
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Loading Data
df = pd.read_csv(r'C:\test\Xtern\XTern 2024 Artificial Intelegence Data Set - Xtern_TrainData.csv')
dd = pd.read_csv(r'C:\test\Xtern\XTern 2024 Artificial Intelegence Data Set - Menu.csv')

# Data Preparation
df['Order'] = df['Order'].apply(str)
dd['Item'] = dd['Item'].apply(str)
df_row = df.shape[0]
dd_row = dd.shape[0]
price = []
pr = 0
calorie = []
cal = 0
for i in range(df_row):
    for j in range(dd_row):
        if df['Order'][i] == dd['Item'][j]:
            pr = dd['Price'][j]
            cal = dd['Calories'][j]
    price.append(pr)
    calorie.append(cal)
df['Price'] = price
df['Calories'] = calorie
df.to_csv(r'C:\test\Xtern\Experiment.csv',index = False)
data = pd.read_csv(r'C:\test\Xtern\Experiment.csv')

# Convert String to categorical variable
le = LabelEncoder()
data['Year'] = le.fit_transform(data['Year'])
data['Major'] = le.fit_transform(data['Major'])
data['University'] = le.fit_transform(data['University'])
data

# Training & Testing Set
x = data[['Year', 'Major', 'University', 'Time', 'Price', 'Calories']]
y = data['Order']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=0)

# Training
clf=MultinomialNB()
clf.fit(x_train,y_train)

#Predict
y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))
