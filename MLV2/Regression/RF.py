# Regression Template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'.\Datasets\train.csv')
X = dataset.iloc[:, [4,6,7,8,9,11,14]].values
X1 = dataset.iloc[:, [4,6,7,8,9,11,14]].values
y = dataset.iloc[:, 16:18].values


#considering dollar as neutral currency
for i in y:
    if i[0].upper() == 'AUD':
        i[1] = i[1]*.69

    if i[0].upper() == 'EUR':
        i[1] = i[1]*1.14

    if i[0].upper() == 'HKD':
        i[1] = i[1]*.13

    if i[0].upper() == 'INR':
        i[1] = i[1]*.014

    if i[0].upper() == 'KRW':
        i[1] = i[1]*.00087

y= y[:,1:2]


# Fitting the Regression Model to the dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,2] = LabelEncoder().fit_transform(X[:,2])
X[:,5] = LabelEncoder().fit_transform(X[:,5])
X = OneHotEncoder(categorical_features=[2,5]).fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Create your regressor here
from sklearn.ensemble import RandomForestRegressor
dt = RandomForestRegressor(n_estimators=1, random_state=0)
dt.fit(X_train,y_train)

# Predicting results
y_pred = dt.predict(X_test)
from sklearn import metrics

print(f"RMSE : {np.sqrt(metrics.mean_absolute_error(y_test,y_pred))}")


# Importing the test dataset
dataset_test = pd.read_csv(r'.\Datasets\test.csv')
X_test_set = dataset.iloc[:, [4,6,7,8,9,11,14]].values
y_test_set = dataset.iloc[:, 16:17].values

X_test_set[:,2] = LabelEncoder().fit_transform(X_test_set[:,2])
X_test_set[:,5] = LabelEncoder().fit_transform(X_test_set[:,5])
X_test_set = OneHotEncoder(categorical_features=[2,5]).fit_transform(X_test_set).toarray()

y_pred_test = dt.predict(X_test_set)
y_pred_test = y_pred_test.reshape(-1,1)
print(y_test_set[0][0])

for i in range(len(y_test_set)):
    if y_test_set[i][0].upper() == 'AUD':
        y_pred_test[i] = y_pred_test[i] / .69

    if y_test_set[i][0].upper() == 'EUR':
        y_pred_test[i] = y_pred_test[i] / 1.14

    if y_test_set[i][0].upper() == 'HKD':
        y_pred_test[i] = y_pred_test[i] / .13

    if y_test_set[i][0].upper() == 'INR':
        y_pred_test[i] = y_pred_test[i] / .014

    if y_test_set[i][0].upper() == 'KRW':
        y_pred_test[i] = y_pred_test[i] / .00087

y_df = pd.DataFrame.from_records(y_pred_test)

import os
path = r'C:\Users\srm\PycharmProjects\ML'
y_df.to_csv(os.path.join(path,r'res.csv'),index=False)