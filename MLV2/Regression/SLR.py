import pandas as pd
import numpy as np


ds = pd.read_csv('.\Datasets\Salary_Data.csv')
X = ds.iloc[:,0:1].values
y = ds.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
y_train = np.array(y_train)
y_test = np.array(y_test)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from matplotlib import pyplot as plt
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.show()

from matplotlib import pyplot as plt
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.show()
