# Data Preprocessing Tools

# Importing the libraries
# Numpy
from math import remainder
from tkinter import Label
import numpy as np
# Matplot Library
import matplotlib.pyplot as plt
# Pandas
import pandas as pd

# Importing the Dataset
dataSet = pd.read_csv("Data.csv")
# Matrix Variable
x = dataSet.iloc[:, :-1].values
# Dependent variable
y = dataSet.iloc[:, -1].values

# Taking care of missing Data
# Scikit Learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding Categorical Data

# Encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columnTr = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnTr.fit_transform(x))

# Encoding Dependent Variable
from sklearn.preprocessing import LabelEncoder

labelEn = LabelEncoder()
y = labelEn.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Feature Scaling (If needed)
# Standardisation & Normalisation (Standardisation can do both so PREFER)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_test)

