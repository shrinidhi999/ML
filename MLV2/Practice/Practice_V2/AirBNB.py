import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale =1, color_codes=True) 


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\AirBNB.csv')
dataset.drop(['id','name','host_id','host_name'], axis=1, inplace=True)

dataset['last_review'] = pd.to_datetime(dataset['last_review'], infer_datetime_format=True)

dataset.isnull().sum()
dataset['reviews_per_month'].fillna(0, inplace=True)
earliest = min(dataset['last_review'])
dataset['last_review'].fillna(earliest, inplace=True)
dataset['last_review'] = dataset['last_review'].apply(lambda x: x.toordinal() - earliest.toordinal())

print(sum(dataset.duplicated(dataset.columns)))


neigh_val_cnt = dataset['neighbourhood'].value_counts()

dataset['neighbourhood'] = dataset['neighbourhood'].apply(lambda x: x if neigh_val_cnt[x] > 1500 else 'Other')


dataset.skew(axis = 0, skipna = True)
dataset['minimum_nights'] = np.cbrt(dataset['minimum_nights'])
dataset['price'] = 1+ dataset['price']
dataset['price'] = np.log(dataset['price'])

num_cols =  dataset.select_dtypes(include=['float64','int64']).columns

sc = StandardScaler()
dataset[num_cols] = sc.fit_transform(dataset[num_cols])

import phik
from phik import resources, report

corr = dataset.phik_matrix()
corr = corr['price'].abs()
print(corr.sort_values())

to_drop_1 = [col for col in corr.index if corr[col]<0.1]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

dataset.drop('neighbourhood_group', axis=1, inplace=True)
    
X= dataset.drop('price', axis=1)
y = dataset['price']            
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)



num_cols =  X_train.select_dtypes(include=['float64','int64']).columns
scaled_x_train = X_train
scaled_x_test = x_test

sc = StandardScaler()
scaled_x_train[num_cols] = sc.fit_transform(scaled_x_train[num_cols])

scaled_x_test[num_cols] = sc.transform(scaled_x_test[num_cols])

scaled_x_train = pd.get_dummies(scaled_x_train, drop_first=True)
scaled_x_test = pd.get_dummies(scaled_x_test, drop_first=True)

# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'neg_mean_squared_error'
results = []

# Standardize the dataset 
# create pipeline 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('lr', LinearRegression())) 
model1 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('Lasso', Lasso())) 
model2 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('ElasticNet', ElasticNet())) 
model3 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('KNN', KNeighborsRegressor())) 
model4 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('DecisionTreeRegressor', DecisionTreeRegressor())) 
model5 = Pipeline(estimators) 

#estimators = [] 
#estimators.append(('standardize', StandardScaler())) 
#estimators.append(('SVR', SVR())) 
#model6 = Pipeline(estimators) 

models =[]
models.append(('LR', model1))
models.append(('Lasso', model2))
models.append(('ElasticNet', model3))
models.append(('KNN', model4))
models.append(('DecisionTreeRegressor', model5))
#models.append(('SVR', model6))

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, scaled_x_train, y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    


# KNN Algorithm tuning  
param_grid = {"n_neighbors":[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsRegressor() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(scaled_x_train, y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
    

model = KNeighborsRegressor(n_neighbors=11)
model.fit(scaled_x_train, y_train)

pred = model.predict(scaled_x_test)
print(r2_score(y_test, pred))

 
    
### ENSEMBLE #############33

# create pipeline 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('ADA', AdaBoostRegressor())) 
model1 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('GDB', GradientBoostingRegressor())) 
model2 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('RF', RandomForestRegressor())) 
model3 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('ET', ExtraTreesRegressor())) 
model4 = Pipeline(estimators) 


models =[]
models.append(('ADA', model1))
models.append(('GDB', model2))
models.append(('RF', model3))
models.append(('ET', model4))

results=[]

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, scaled_x_train, y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)    
   

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
rfe_model = RandomForestRegressor() 
rfecv = RFECV(estimator=rfe_model, step=1, cv=10, scoring='neg_mean_squared_error')   #5-fold cross-validation
rfecv = rfecv.fit(scaled_x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', scaled_x_train.columns[rfecv.support_])

cl = [col for col in scaled_x_train.columns if col not in scaled_x_train.columns[rfecv.support_]]

scaled_x_train.drop(cl, inplace=True, axis=1)
scaled_x_test.drop(cl, inplace=True, axis=1)


### Scoring#########
model = GradientBoostingRegressor()
model.fit(scaled_x_train, y_train)

pred = model.predict(scaled_x_test)
print(r2_score(y_test, pred))
