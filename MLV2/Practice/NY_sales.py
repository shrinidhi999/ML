import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

dataset = pd.read_csv('.\Datasets\\nyc.csv')
print(dataset.info())

dataset = dataset.drop(['Unnamed: 0'],axis=1)

dataset['TAX CLASS AT TIME OF SALE'] = dataset['TAX CLASS AT TIME OF SALE'].astype('category')
dataset['TAX CLASS AT PRESENT'] = dataset['TAX CLASS AT PRESENT'].astype('category')
dataset['LAND SQUARE FEET'] = pd.to_numeric(dataset['LAND SQUARE FEET'], errors='coerce')
dataset['GROSS SQUARE FEET']= pd.to_numeric(dataset['GROSS SQUARE FEET'], errors='coerce')
#df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
dataset['SALE PRICE'] = pd.to_numeric(dataset['SALE PRICE'], errors='coerce')
dataset['BOROUGH'] = dataset['BOROUGH'].astype('category')

print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')
print(sum(dataset.duplicated(dataset.columns)))


print(dataset.columns[dataset.isnull().any()])
print(dataset.isnull().sum())
dataset['LAND SQUARE FEET'] = dataset['LAND SQUARE FEET'].fillna(dataset['LAND SQUARE FEET'].mean())
dataset['GROSS SQUARE FEET'] = dataset['GROSS SQUARE FEET'].fillna(dataset['GROSS SQUARE FEET'].mean())

# plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
# p=sns.heatmap(dataset.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

# sns.heatmap(dataset.corr())

print(dataset.corr()['SALE PRICE'].sort_values())

dataset = dataset[(dataset['TOTAL UNITS'] > 0) & (dataset['TOTAL UNITS'] !=2261)]

dataset = dataset.drop(['NEIGHBORHOOD', 'ADDRESS','APARTMENT NUMBER','BUILDING CLASS AT PRESENT','BUILDING CLASS AT TIME OF SALE','SALE DATE'],axis=1)


test=dataset[dataset['SALE PRICE'].isna()]
data=dataset[~dataset['SALE PRICE'].isna()]

data = pd.get_dummies(data,columns=['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE','EASE-MENT'],drop_first=True)
print(data.info())
X = data.drop(['SALE PRICE'],axis=1).values
y = data['SALE PRICE'].values
y=y.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

# from sklearn.decomposition import PCA
# pca= PCA(n_components=57)
# X = pca.fit_transform(X)
# e_var = pca.explained_variance_ratio_
# e_var =e_var.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train ,X_test, Y_train , Y_test = train_test_split(X , y , test_size = 0.3 , random_state =34)

print(X_train.shape , Y_train.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

linreg = LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred_lin = linreg.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test,Y_pred_lin)))

from sklearn.model_selection import cross_val_score
clf_lr = LinearRegression()
clf_lr.fit(X_train , Y_train)
accuracies = cross_val_score(estimator = clf_lr, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = clf_lr.predict(X_test)
print('')
print('####### Linear Regression #######')
print('Score : %.4f' % clf_lr.score(X_test, Y_test))
print(accuracies)

mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


