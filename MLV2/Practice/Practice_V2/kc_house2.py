import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import ElasticNet 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True) 


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\kc_house_data.csv')
dataset = dataset.drop(['view','yr_renovated','id'], axis=1)

dataset.isnull().sum()
#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

import phik
from phik import resources, report

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

corr = corr['price'].abs()

print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col] < 0.2]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()

col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.8:
            print(f"{col[i]} -{col[j]}- {corr.iloc[i,j]}")

dataset.drop(['sqft_above','sqft_basement','bathrooms'],inplace=True,axis=1)
            
dataset.info()            


X= dataset.drop('price', axis=1)
y= dataset['price']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size =0.2, random_state =0)

#Remove outliers -  try remove outliers using np.log
#from scipy import stats
#numeric_cols = X_train.select_dtypes(include=['int64','float64'])
#z = np.abs(stats.zscore(numeric_cols))
#to_drop_rows=[]
#
###method 2
#for j in range(numeric_cols.shape[1]):
#    mean = numeric_cols.iloc[:,j].mean()
#    for i in range(numeric_cols.shape[0]):        
#        if z[i,j] >= 3:
#            print(f"{i} -{j}")
#            numeric_cols.iloc[i,j] = median #or mean()
#
#X_train.update(numeric_cols)
#
#numeric_cols = X_validation.select_dtypes(include=['int64','float64'])
#z = np.abs(stats.zscore(numeric_cols))
#to_drop_rows=[]
#
###method 2
#for j in range(numeric_cols.shape[1]):
#    mean = numeric_cols.iloc[:,j].mean()
#    for i in range(numeric_cols.shape[0]):        
#        if z[i,j] >= 3:
#            print(f"{i} -{j}")
#            numeric_cols.iloc[i,j] = median #or mean()
#
#X_validation.update(numeric_cols)
    
X_train.skew()
# sns.distplot(X_train['bedrooms'])
# plt.show()
#sns.distplot(X_train['sqft_lot15'])
# plt.show()
#sns.distplot(X_train['sqft_lot'])
# plt.show()
#sns.distplot(X_train['waterfront'])
# plt.show()

#X_train['bedrooms'] = np.cbrt(X_train['bedrooms'])
#X_validation['bedrooms'] = np.cbrt(X_validation['bedrooms'])


# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'neg_mean_squared_error'

# Spot-Check Algorithms 
models = [] 
models.append(('LR', LinearRegression())) 
models.append(('LASSO', Lasso())) 
models.append(('EN', ElasticNet())) 
models.append(('KNN', KNeighborsRegressor())) 
models.append(('CART', DecisionTreeRegressor())) 
# models.append(('SVR', SVR()))

results = [] 
names = [] 
for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)          



sc = StandardScaler()
sc.fit(X_train)
rescale_X = sc.transform(X_train)

from sklearn.feature_selection import RFECV
rfe = RFECV(estimator = LinearRegression(), step=1, scoring= scoring, cv=10)
rfe.fit(rescale_X, Y_train)

print(X_train.columns[rfe.support_])

cl = [col for col in X_train.columns if col not in X_train.columns[rfe.support_]]
X_train.drop(cl, axis=1, inplace=True)
X_validation.drop(cl, axis=1, inplace=True)



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

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('SVR', SVR())) 
model6 = Pipeline(estimators) 

models =[]
models.append(('LR', model1))
models.append(('Lasso', model2))
models.append(('ElasticNet', model3))
models.append(('KNN', model4))
models.append(('DecisionTreeRegressor', model5))
#models.append(('SVR', model6))

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)

# KNN Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 

param_grid = {"n_neighbors":[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsRegressor() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))     
 

model = KNeighborsRegressor(n_neighbors = 11)         
model.fit(rescaledX, Y_train)

rescaledX_validation = scaler.transform(X_validation) 
y_pred = model.predict(rescaledX_validation)

print('')
print('####### KNeighborsRegressor #######')
print('Score : %.4f' % model.score(rescaledX_validation, Y_validation))


mse = mean_squared_error(Y_validation, y_pred)
mae = mean_absolute_error(Y_validation, y_pred)
rmse = mean_squared_error(Y_validation, y_pred)**0.5
r2 = r2_score(Y_validation, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)    

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

for name, model in models: 
kfold = KFold(n_splits=num_folds, random_state=seed) 
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
results.append(cv_results) 
names.append(name) 
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
print(msg) 

# AdaBoostRegressor tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"n_estimators":[50,100,150,200,250,300,350,400,500,600]}
model = AdaBoostRegressor(random_state=seed) 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))     
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))
    

model = AdaBoostRegressor(n_estimators = 50, random_state=seed)         
model.fit(rescaledX, Y_train)

rescaledX_validation = scaler.transform(X_validation) 
y_pred = model.predict(rescaledX_validation)

print('')
print('####### AdaBoostRegressor #######')
print('Score : %.4f' % model.score(rescaledX_validation, Y_validation))


mse = mean_squared_error(Y_validation, y_pred)
mae = mean_absolute_error(Y_validation, y_pred)
rmse = mean_squared_error(Y_validation, y_pred)**0.5
r2 = r2_score(Y_validation, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)  