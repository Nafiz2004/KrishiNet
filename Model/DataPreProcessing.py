# System path
import os
os.chdir("E:\Work & Files\Projects\KrishiNet\Datasets\Production")

# Importing libries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('MinorCropProduction_2022-24.csv')
x = dataset.iloc[:, 1:7].values
y = (dataset.iloc[:, 8:9].values) * 1000

print(x)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = "Unknown")
x[:, 2:3] = imputer.fit_transform(x[:, 2:3])

print(x)

# Train Test split
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, train_size = 0.8, random_state = 1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


# Encoding x axis
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as OHE
ct = ColumnTransformer(transformers = [('encoder', OHE(handle_unknown='ignore'), slice(0, 4))], remainder = 'passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

x_train = x_train.toarray()
x_test = x_test.toarray()


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train[:, 157:] = scaler.fit_transform(x_train[:, 157:])
# x_test[:, 157:] = scaler.transform(x_test[:, 157:])

print(x_train)
print(x_test)
