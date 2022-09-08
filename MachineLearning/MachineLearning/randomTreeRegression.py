# Random Forest Regression

# A version of Ensemble Learning

# STEP 1: Pick at random K data points from the Training set.
# STEP 2: Build the Decision Tree associated to  these K data points.
# STEP 3: Choose the number Ntree of trees you want to build and repeat STEPS 1 & 2.
# STEP 4: For a new data point, make each one of your Ntree trees predict the value of Y to for the 
# data point in question, and assign the new data point the average across all of the predicted Y values.

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

# Predicting a new result
print(regressor.predict([[6.5]]))

# Visualising the Random Forest Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()