import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

dataset = pd.read_csv('.\Datasets\kc_house_data.csv')
dataset = dataset.drop(['date','id'],axis=1)
print(dataset.describe())
print(dataset.info())

# plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
# p=sns.heatmap(dataset.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

print(dataset.isnull().sum())
print(dataset.shape)
print(dataset.corr()['price'].sort_values())

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X = dataset.drop(['price'],axis=1).values
# X = sc.fit_transform(X)

y=dataset['price'].values
y=y.reshape(-1,1)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y,random_state = 7,test_size=0.2)

#
# from sklearn.decomposition import PCA
# pca= PCA(n_components=None)
# train_x = pca.fit_transform(train_x)
# test_x = pca.transform(test_x)
# e_var = pca.explained_variance_ratio_
# e_var =e_var.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

clf_lr = Lasso(alpha=1e-10)
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### Lasso Regression #######')
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


from sklearn.model_selection import GridSearchCV

las = Lasso()
params={'alpha':[1e-10,1e-5,1e-3,1e-2,1,5,10,20]}
las_reg = GridSearchCV(las,params,scoring='neg_mean_squared_error',cv=5)
las_reg.fit(train_x,train_y)

print(las_reg.best_params_)
print(las_reg.best_score_ )
