# Regression

# Multiple Linear Regression
# y = b0 + b1*x1 + b2*x2 ... + bn*D1(Dummy variable only 1)

# Building a model with good variables
# 1. All-in
# 2. Backward Elimination
# 3. Forward Selection
# 4. Bidirectionla Elimination (Stepwise Regression) 
# 5. Score Comparion

# Backward Elimination
# Step 1: Select a significance level to stay (e.g SL = 0.05)
# Step 2: Fit the full model with all possible predictors
# step 3: Consider the predictor with highest P-value. if P > SL, go to STEP 4, otherwise go to FIN
# Step 4: Remove the predictor
# Step 5: Fit model without this variable*

# Forward Elimination
# Step 1: Select a significance level to enter the model (e.g SL = 0.05)
# Step 2: Fit all simple regression models y = Xn. Select the one with the lowest P-value
# Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s)
# Step 4: Consider the predictor with the lowest P-value. If P < SL, go Step 3, else go to FIN

# Bidirectional Elimination
# Step 1: Select a significance level to enter and to stay in the model
# e.g SLENTER = 0.05, SLSTAY = 0.05
# Step 2: Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter)
# Step 3: Perform ALL steps of Backward Elimination (old variables must have P < SLSTAY to stay)
# Step 4: No new variables can enter and no old variables can exit

# All possible Models
# Step 1: Select a criterion of goodness of fit (e.g Akaike criterion)
# Step 2: Construct ALL Possible Regression Models: 2n - 1 total combinations
# Step 3: Select the one with the best criterion

# Importing Libraries
from argparse import ONE_OR_MORE
from math import remainder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting dataset to train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# .reshape used to make the data vertical with the len for row and 1 for column

# Making a single prediction (e.g R&D spend = 160000, administration spend = 130000, marketing spend = 300000, state = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

