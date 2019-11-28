import pandas as pd
import numpy as np


ds = pd.read_csv(r'D:\ML\ML_Rev\Datasets\Position_Salaries.csv')
X = ds.iloc[:,1:2].values
y = ds.iloc[:,2:3].values

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_poly,y)

from matplotlib import pyplot as plt
x_grid = np.arange(min(X),max(X),0.1).reshape(-1,1)

plt.scatter(X,y,color='red')
plt.plot(x_grid, lr.predict(poly.fit_transform(x_grid)), color='blue' )
plt.show()

print(np.array([[6.5]]).reshape(-1,1))

t = poly.fit_transform(np.array([6.5]).reshape(-1,1))
sal = lr.predict(np.array([6.5]))