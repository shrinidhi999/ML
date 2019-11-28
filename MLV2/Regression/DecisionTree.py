# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('.\Datasets\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values
# Fitting the Regression Model to the dataset

# Create your regressor here
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X,y)

# Predicting a new result
test = np.array([6.5,7])
test = test.reshape(-1, 1)

y_pred = dt.predict(test)

# Visualising the Regression results
x_g = np.arange(min(X), max(X), 0.001)
x_g = x_g.reshape(-1,1)
plt.scatter(X, y, color = 'red')
plt.plot(x_g, dt.predict(x_g), color = 'blue')
plt.title('DT')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()