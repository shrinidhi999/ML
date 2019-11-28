import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

dataset_old = pd.read_csv('D:\ML\ML_Rev\Datasets\playstore.csv')







#mask = dataset_old['Reviews'] =='3.0M'
#dataset_old.loc[mask, 'Reviews'] = 0
#dataset_old['Reviews'] = dataset_old['Reviews'].astype('int64')
#dataset_old.drop(['App','Size','Current Ver','Last Updated'], axis=1, inplace=True)

#dataset_old.drop(['Size','Current Ver','Android Ver','Last Updated','App'], axis=1, inplace=True)
#dataset_old['Size'] = dataset_old['Size'].astype('float64')
#dataset_old['Installs'] = dataset_old['Installs'].astype('float64')
#dataset_old['Price'] = dataset_old['Price'].astype('float64')

#Taking care of Missing Data in Dataset
print(dataset_old.isnull().sum())
dataset_old.dropna(inplace=True)

#Duplicate check
print(sum(dataset_old.duplicated(dataset_old.columns)))
dataset_old = dataset_old.drop_duplicates(dataset_old.columns, keep='last')

#for col in dataset_old.select_dtypes(include=['category','object']).columns:
#    dataset_old.loc[dataset_old[col].value_counts()[dataset_old[col]].values < 100, col] = np.nan

#dataset.info()

# Correlation check 
# using heat map - phi_k

#corr = dataset_old.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
#
#
#corr = dataset_old.phik_matrix()['Rating'].abs()
##print(abs(corr).sort_values())
#to_drop_1 = [col for col in corr.index if corr[col]<0.1]
#dataset_old.drop(to_drop_1, axis=1, inplace=True)
#
#corr = dataset_old.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
#
#
# corr_mat = dataset.drop('rating',axis=1).corr(method='spearman').abs()
# to_drop = [col for col in corr_mat.columns if any((corr_mat[col] > 0.7)&(corr_mat[col] < 1))]
#
#col = corr.index
#for i in range(len(col)):
#    for j in range(i+1, len(col)):
#        if corr.iloc[i,j] >= 0.9:
#            print(f"{col[i]} -{col[j]}")
#
#dataset_old.drop(['Genres','Type'],inplace=True,axis=1)
#
#
#dataset_old.drop(['Genres','Type','Content Rating','Genres','Reviews','Type'],inplace=True,axis=1)
#
#mask = (dataset_old['Category'] != 'Family') & (dataset_old['Category'] != 'Games') & (dataset_old['Category'] != 'Tools')
#dataset_old.loc[mask, 'Category'] = 'Other'
#
#
#mask = (dataset_old['Installs'] != '1,000,000+') & (dataset_old['Installs'] != '10,000,000+') & (dataset_old['Installs'] != '100,000+')
#dataset_old.loc[mask, 'Installs'] = 'Other'
#
#
#mask = (dataset_old['Price'] != '$2.99') & (dataset_old['Price'] != '$0.99') & (dataset_old['Price'] != '0')
#dataset_old.loc[mask, 'Price'] = 'Other'


#Encoding categorical data
dataset_old = pd.get_dummies(dataset_old)

dataset = dataset_old[~dataset_old['Rating'].isnull()]



X = dataset.drop('Rating',axis=1)
y = dataset.Rating

#Splitting the Dataset into Training set and Test Set
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=42)



# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
clf_lr = RandomForestRegressor()
clf_lr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_lr, X = X_train, y = y_train, cv = 5,verbose = 1)
print('')
print('####### Forest Regression #######')
print('Score : %.4f' % clf_lr.score(X_test, y_test))
print(accuracies.mean())

y_pred = clf_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)
