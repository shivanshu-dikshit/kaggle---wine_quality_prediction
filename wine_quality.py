# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing the essential libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#importing the data and doing the essential preprocessing
data = pd.read_csv("wineQualityReds.csv")
print(data.describe())
print(data.shape)

#diviing the data into dependent vaiables and independent variables
y = data.quality
X = data.drop('quality', axis = 1)

#diving the data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#creating the model and fitting the data to the model 
model = RandomForestRegressor(n_estimators=500, min_samples_split=2, max_depth=None).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))

