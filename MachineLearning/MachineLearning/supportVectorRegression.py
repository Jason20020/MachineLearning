# Support Vector Regression (SVR)

# It will be a tube instead of a line to make sure all the values will include (Insensitive Tube)
# Slack Variables are outside the tube svi and svi*

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1) # Reshape it to 2D-Array 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x) # The fit_transform method will have a record of the x, therefore need to create a new method
sc_2 = StandardScaler()
y = sc_2.fit_transform(y)

# Training the SVR Model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Predicting a new result (transform it to x scaler and reverse result back to y)
pred = regressor.predict(sc.transform([[6.5]]))
print(sc_2.inverse_transform([pred]))

# Visualising the SVR results
plt.scatter(sc.inverse_transform(x), sc_2.inverse_transform(y), color = 'red')
pred = regressor.predict(x)
pred = pred.reshape(len(pred), 1)
plt.plot(sc.inverse_transform(x), sc_2.inverse_transform(pred), color = 'blue')
plt.title('Truth of Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
x_grid = np.arange(min(sc.inverse_transform(x)), max(sc.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc.inverse_transform(x), sc_2.inverse_transform(y), color = 'red')
pred_2 = regressor.predict(sc.transform(x_grid))
pred_2 = pred_2.reshape(len(pred_2), 1)
plt.plot(x_grid, sc_2.inverse_transform(pred_2), color = 'blue')
plt.title('Truth of Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()