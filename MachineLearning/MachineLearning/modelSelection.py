# Model Selection in Regression

# Multiple Linear Regression

# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('BigData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print('Multiple Linear Regression')
print(r2_score(y_test, y_pred))



# Polynomial Linear Regression

# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('BigData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Polynomial Linear Regression model on Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)

# Predicting the Test set results
y_pred = regressor.predict(poly_reg.transform(x_test))
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print('Polynomial Linear Regression')
print(r2_score(y_test, y_pred))



# Support Vector Regression(SVR)

# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('BigData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = y_train.reshape(len(y_train), 1)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)

# Predicting the Test set results
pred = regressor.predict(sc_x.transform(x_test))
y_pred = sc_y.inverse_transform(pred.reshape(len(pred), 1))
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print('Support Vector Regression')
print(r2_score(y_test, y_pred))



# Decision Tree Regression

# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('BigData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Decision Tree Regression model on Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print('Decision Tree Regression')
print(r2_score(y_test, y_pred))



# Random Forest Regression

# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('BigData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Decision Tree Regression model on Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print('Random Forest Regression')
print(r2_score(y_test, y_pred))