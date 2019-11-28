import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

dataset = pd.read_csv('.\Datasets\Diamonds.csv', usecols= range(1,11))
print(dataset.info())

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

chi_features = SelectKBest(chi2,k=2)


# plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
# p=sns.heatmap(dataset.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

print(dataset.isnull().sum())
print(dataset.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['cut'] = le.fit_transform(dataset['cut'])
dataset['color'] = le.fit_transform(dataset['color'])
dataset['clarity'] = le.fit_transform(dataset['clarity'])

dataset= pd.get_dummies(dataset,columns=['cut'],drop_first=True)
dataset= pd.get_dummies(dataset,columns=['color'],drop_first=True)
dataset= pd.get_dummies(dataset,columns=['clarity'],drop_first=True)

dataset['vol'] =dataset['x'] * dataset['y']*dataset['z']

x = dataset.drop(["price",'x','y','z'],axis=1).values
y = dataset['price'].values
y=y.reshape(-1,1)

best_cat = chi_features.fit_transform()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x = sc.fit_transform(x)

from sklearn.decomposition import PCA
pca= PCA(n_components=None)
x = pca.fit_transform(x)
e_var = pca.explained_variance_ratio_
e_var =e_var.reshape(-1,1)


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 2,test_size=0.3)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score
#
# clf_lr = LinearRegression()
# clf_lr.fit(train_x , train_y)
# accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)
# y_pred = clf_lr.predict(test_x)
# print('')
# print('####### Linear Regression #######')
# print('Score : %.4f' % clf_lr.score(test_x, test_y))
# print(accuracies)
#
# mse = mean_squared_error(test_y, y_pred)
# mae = mean_absolute_error(test_y, y_pred)
# rmse = mean_squared_error(test_y, y_pred)**0.5
# r2 = r2_score(test_y, y_pred)
#
# print('')
# print('MSE    : %0.2f ' % mse)
# print('MAE    : %0.2f ' % mae)
# print('RMSE   : %0.2f ' % rmse)
# print('R2     : %0.2f ' % r2)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
params = [{'n_estimators':[1,10,15], 'max_depth':[1,2,3,4]}]
vc = GridSearchCV(estimator=RandomForestRegressor(n_estimators=1),param_grid=params,
                  verbose=1,cv=10,n_jobs=-1)
res = vc.fit(train_x,train_y)
print(res.best_score_)
print(res.best_params_)

clf_lr = RandomForestRegressor(n_estimators=10)
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)
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
