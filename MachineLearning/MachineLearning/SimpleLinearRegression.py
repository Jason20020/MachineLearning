# Regressions

# Simple Linear Regression
# y(Dependent Var) = b0(const) + b1(Coeficcient) * x(Indeoendent Var)

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Simple Linear Regression model on Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the Training set Results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')

plt.show()

# Visualising the Test set Results
plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')

plt.show()

# Making a single prediction for a year
print(regressor.predict([[12]]))

# Getting the final linear regression equation
coefficient = regressor.coef_
constantValue = regressor.intercept_

# Salary = constantValue + Experience * coefficient