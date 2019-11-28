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

dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\playstore.csv')

dataset = dataset[~dataset['Rating'].isnull()]
dataset.drop(['App'], axis=1, inplace=True)

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')
dataset = dataset.reset_index()


#Category
dataset['Category'].value_counts()
c = dataset[dataset['Category']=='1.9']
dataset.drop(dataset.index[8641], inplace=True)
dataset.drop(['index'], axis=1, inplace=True)

#Reviews
dataset['Reviews'].value_counts()
dataset['Reviews'].str.isnumeric().sum()
dataset['Reviews'] = dataset['Reviews'].astype('int64')

#Size
dataset['Size'].value_counts()
dataset['Size'] = dataset['Size'].str.replace('k', 'e+3')
dataset['Size'] = dataset['Size'].str.replace('M', 'e+6')
dataset['Size'] = dataset['Size'].replace('Varies with device', np.nan)
dataset['Size'].str.isnumeric().sum()
dataset['Size'] = dataset['Size'].astype('float64')

#Installs
dataset['Installs'].value_counts()
dataset['Installs'] = dataset['Installs'].str.replace('+','')
dataset['Installs'] = dataset['Installs'].str.replace(',','')
dataset['Installs'].str.isnumeric().sum()
dataset['Installs'] = dataset['Size'].astype('float64')

#Type
dataset['Type'].value_counts()

#Price
dataset['Price'].value_counts()
dataset['Price'] = dataset['Price'].str.replace('$','')
dataset['Price'].str.isnumeric().sum()
dataset['Price'] = dataset['Price'].astype('float64')

#Content Rating
dataset['Content Rating'].value_counts()

#Genres
dataset['Genres'].value_counts()
dataset['Genres'] = dataset['Genres'].apply(lambda x: x.split(';')[0])

#Last Updated
from datetime import date,datetime
#
dataset['Last Updated'] = pd.to_datetime(dataset['Last Updated'])
dataset['Last Updated'] = dataset['Last Updated'].apply(lambda x:date.today()-datetime.date(x))
dataset['Last Updated'] = dataset['Last Updated'].dt.days
dataset['Last Updated'] = dataset['Last Updated'].astype('float64')

#Current Version
dataset['Current Ver'].value_counts()
import re
dataset['Current Ver']=dataset['Current Ver'].replace(np.nan,'Varies with device')
dataset['Current Ver']=dataset['Current Ver'].apply(lambda x: 'Varies with device' if x=='Varies with device'  
                else  re.findall('^[0-9]\.[0-9]|[\d]|\W*',str(x))[0] )
dataset['Current Ver']=dataset['Current Ver'].replace('Varies with device',1.0)
dataset['Current Ver']=dataset['Current Ver'].replace('','1.0')
dataset['Current Ver']=dataset['Current Ver'].astype('float64')


#Android Ver
dataset['Android Ver'].value_counts()
dataset['Android Ver']= dataset['Android Ver'].apply(lambda x:str(x).split(' ')[0])
dataset['Android Ver']= dataset['Android Ver'].str.replace('W','')
dataset['Android Ver']= dataset['Android Ver'].apply(lambda x: 'Varies with device' if x=='Varies'  
                else  re.findall('^[0-9]\.[0-9]|[\d]|\W*',str(x))[0] )
dataset['Android Ver']= dataset['Android Ver'].replace('Varies with device','1.0')
dataset['Android Ver']= dataset['Android Ver'].replace('','1.0')
dataset['Android Ver']= dataset['Android Ver'].astype('float64')
dataset['Android Ver']= dataset['Android Ver'].replace(np.nan, dataset['Android Ver'].median())

#corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
#
#
#corr = dataset.phik_matrix()['Rating'].abs()
#print(abs(corr).sort_values())
#to_drop_1 = [col for col in corr.index if corr[col]<0.09]
#dataset.drop(to_drop_1, axis=1, inplace=True)

#corr = dataset.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


# corr_mat = dataset.drop('rating',axis=1).corr(method='spearman').abs()
# to_drop = [col for col in corr_mat.columns if any((corr_mat[col] > 0.7)&(corr_mat[col] < 1))]
#
#col = corr.index
#for i in range(len(col)):
#    for j in range(i+1, len(col)):
#        if corr.iloc[i,j] >= 0.9:
#            print(f"{col[i]} -{col[j]}")

dataset.drop(['Category','Size'],inplace=True,axis=1)

#Remove outliers - 
from scipy import stats
numeric_cols = dataset.select_dtypes(include=['int64','float64'])
z = np.abs(stats.zscore(numeric_cols))
to_drop_rows=[]

for i in range(numeric_cols.shape[0]):
    for j in range(numeric_cols.shape[1]):
        if z[i,j] >= 3:
#            print(f"{i} -{j}")
            to_drop_rows.append(i)
            numeric_cols.iloc[i,j] = numeric_cols.iloc[:,j].median()

# drop or replace by mean
#dataset = dataset.drop([to_drop_rows], axis=0)
dataset.update(numeric_cols) 


#For Categorical vars - remove/replace low freq vars
for col in dataset.select_dtypes(include=['category','object']).columns:
    dataset.loc[dataset[col].value_counts()[dataset[col]].values < 20, col] = np.nan

print(dataset.isnull().sum())
dataset['Installs'] = dataset['Installs'].replace(np.nan, dataset['Installs'].median())
dataset = dataset.dropna()

#Encoding categorical data
dataset = pd.get_dummies(dataset)    
dataset.info()


X = dataset.drop('Rating',axis=1).values
y = dataset['Rating'].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y,random_state = 7,test_size=0.3)

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


from sklearn.tree import DecisionTreeRegressor
clf_lr = DecisionTreeRegressor()
clf_lr.fit(train_x , train_y)
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)
y_pred = clf_lr.predict(test_x)
print('')
print('####### DecisionTreeRegressor #######')
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
accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)
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


