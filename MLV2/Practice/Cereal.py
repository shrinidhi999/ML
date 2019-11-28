import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\cereal.csv')

print(dataset.info())
dataset.drop(['name'],axis=1,inplace=True)
dataset['mfr'] = dataset['mfr'].astype('category')
dataset['type'] = dataset['type'].astype('category')


print(dataset.isnull().sum())
print(dataset.describe(include='all'))


#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')


#
#import phik
#from phik import resources, report
#
#corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
#
#
#corr = dataset.phik_matrix()['rating'].abs()
#print(abs(corr).sort_values())
#to_drop_1 = [col for col in corr.index if corr[col]<0.2]
#dataset.drop(to_drop_1, axis=1, inplace=True)
#
#corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
#
#
## corr_mat = dataset.drop('rating',axis=1).corr(method='spearman').abs()
## to_drop = [col for col in corr_mat.columns if any((corr_mat[col] > 0.7)&(corr_mat[col] < 1))]
#
#col = corr.index
##print(corr.iloc[1,2])
#for i in range(len(col)):
#    for j in range(i+1, len(col)):
#        if corr.iloc[i,j] >= 0.8:
#            print(f"{col[i]} -{col[j]}")
#
#dataset.drop(['potass','weight'],inplace=True,axis=1)
dataset.skew(axis = 0, skipna = True)

dataset = pd.get_dummies(dataset)

#Dropping after RFECV results
#dataset.drop(['type_H'],inplace=True,axis=1)

X = dataset.drop('rating',axis=1)
y = dataset['rating']


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y,random_state = 7,test_size=0.3)


from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestRegressor() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='neg_mean_squared_error')   #5-fold cross-validation
rfecv = rfecv.fit(train_x, train_y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', train_x.columns[rfecv.support_])

cl = [col for col in train_x.columns if col not in train_x.columns[rfecv.support_]]

# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score


clf_lr = RandomForestRegressor()
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 10,verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### Forest Regression #######')
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


