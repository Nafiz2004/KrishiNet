
# Importing the libraries
import numpy as np
import pandas as pd
import os
os.chdir("E:\Work & Files\Projects\KrishiNet\Model")
from DataPreProcessing import x_train, x_test, y_train, y_test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_train[:, -2:] = x_scaler.fit_transform(x_train[:, -2:])
y_train[:, -2:] = y_scaler.fit_transform(y_train[:, -2:])



# Training the Multiple Linear Regression model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(x_train, y_train)

# Predicting the Test set results
x_test[:, -2:] = x_scaler.transform(x_test[:, -2:])

y_pred = regressor.predict(x_test)
y_pred.reshape(-1, 1)
y_pred[:, -2:] = y_scaler.inverse_transform(y_pred[:, -2:])
np.concatenate((y_test, y_pred), 1)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
