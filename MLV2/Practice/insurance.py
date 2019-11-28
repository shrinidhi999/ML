import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\insurance.csv')

print(dataset.describe())
print(dataset.info())

print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')


dataset['sex'] = dataset['sex'].astype('category')
dataset['smoker'] = dataset['smoker'].astype('category')
dataset['region'] = dataset['region'].astype('category')

print(dataset.isnull().sum())

import phik
from phik import resources, report

corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

corr = dataset.phik_matrix()['charges'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col]<0.1]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


# corr_mat = dataset.drop('rating',axis=1).corr(method='spearman').abs()
# to_drop = [col for col in corr_mat.columns if any((corr_mat[col] > 0.7)&(corr_mat[col] < 1))]

col = corr.index
#print(corr.iloc[1,2])
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.8:
            print(f"{col[i]} -{col[j]}")

dataset = pd.get_dummies(dataset)



X = dataset.drop(['charges'],axis=1).values
y = dataset['charges'].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y,random_state = 7,test_size=0.2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score

clf_lr = LinearRegression()
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### Linear Regression #######')
print('Score : %.4f' % clf_lr.score(test_x, test_y))
print(accuracies)
mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
rmse = mean_squared_error(test_y, y_pred)**0.5
r2 = r2_score(test_y, y_pred)
print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


from sklearn.ensemble import RandomForestRegressor

clf_lr = RandomForestRegressor(n_estimators=10)
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 10,verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### Forest #######')
print('Score : %.4f' % clf_lr.score(test_x, test_y))
print(accuracies)

mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
rmse = mean_squared_error(test_y, y_pred)**0.5
r2 = r2_score(test_y, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)
