import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#linear regression
#y = mx+c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

#Polynomial regression
# y = m1x^2 + c
#x^2 is changed to x^0,x^1,x^2
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)
lr = LinearRegression()
lr.fit(X_poly, y)

#Support vector regressor
'''
scale the data
It uses the hyperplane as a regression line 
It tries to cover every point inside the support vector
classifiers
The points on the line and outside the line are called support vectors
The error cost is distance of every point outside the classifier(slack variables) with the classifier lines 
'''
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
y_pred = regressor.predict(X_test)

'''
Decision Tree
'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
y_pred = regressor.predict(X_test)


'''
Random Forest 
It takes the average of all the decision tree 
'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
regressor.predict(X_test)


#Gradient Boost
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X_train,y_train)
y_pred = ada.predict(X_test)

#XGBoost
from xgboost import XGBRegressor
classifier = XGBRegressor()
classifier.fit(X_train, y_train)