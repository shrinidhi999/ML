import pandas as pd
import numpy as np


ds = pd.read_csv('.\Datasets\Startups.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,4:5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,3] = LabelEncoder().fit_transform(X[:,3])
X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()

#remove dummy var
X= X[:,1:]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
y_train = np.array(y_train)
y_test = np.array(y_test)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

import statsmodels.formula.api as sm

# Appending to accomodate b0x0
X = np.append(arr=np.ones((50,1)).astype(int), values=X , axis=1)

sl= 0.05
x_opt = X[:,:]

# def BwdElimination(x_opt,sl):
#
#     for i in range(len(x_opt[0])):
#         ols = sm.OLS(y,x_opt).fit()
#         max_p = max(ols.pvalues).astype(float)
#         if max_p > sl:
#             for j in range(len(x_opt[0])):
#                 if (ols.pvalues[j].astype(float) == max_p):
#                     x_opt = np.delete(x_opt, j, 1)
#
#         print(ols.summary())
#     return x_opt
#
#
#
# x_final = BwdElimination(x_opt,sl)


x_opt = X
ols = sm.OLS(endog=y, exog=x_opt).fit()
print(ols.summary())

x_opt = X[:,[0,1,3,4,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
print(ols.summary())

x_opt = X[:,[0,3,4,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
print(ols.summary())


x_opt = X[:,[0,3,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
print(ols.summary())

x_opt = X[:,[0,3]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
print(ols.summary())

print(max(ols.pvalues.astype(float)))

# Test
# ds = pd.read_csv('.\Datasets\Startups.csv')
# X = ds.iloc[:,:-1].values
# y = ds.iloc[:,4:5].values
#
# from sklearn.model_selection import train_test_split
# X_train1,X_test1,y_train1,y_test1 = train_test_split(X[:,0:1],y,test_size=0.2,random_state=0)
#
# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(X_train1, y_train1)
#
# y_pred1 = lr.predict(X_test1)