
# Importing the libraries
import numpy as np
import pandas as pd
import os
os.chdir("E:\Work & Files\Projects\KrishiNet\Model")
from DataPreProcessing import x_train, x_test, y_train, y_test

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_regressor = PolynomialFeatures(degree=3)
x_train_poly = poly_regressor.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_train_poly, y_train)

# Predicting the Test set results
x_test_poly = poly_regressor.transform(x_test)
y_pred = regressor.predict(x_test_poly)
np.concatenate((y_test, y_pred), 1)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
