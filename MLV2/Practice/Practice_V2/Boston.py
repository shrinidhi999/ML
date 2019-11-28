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
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True) 


dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\housing.csv')


dataset.shape
dataset.info()
dataset.sample(5)

pd.set_option('precision',1)
dataset.describe()

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

print((dataset.isnull().sum()/ len(dataset)) * 100)
for col in dataset.columns:
    t = dataset[col].median()
    dataset[col].fillna(t,inplace=True)
    


# correlation 
pd.set_option('precision', 2)
corr = dataset.corr(method='pearson').abs()

col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] > 0.7:
            print(f"{col[i]} -{col[j]} - {corr.iloc[i,j]}")


# box and whisker plots 
#dataset.plot(kind='box', subplots=True, layout=(5,3), sharex=False, sharey=False, fontsize=8) 
#plt.show()
#
## histograms 
#dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) 
#plt.show()
#
## scatter plot matrix
#scatter_matrix(dataset) 
#plt.show()
#
#corr = dataset.corr(method='pearson').abs()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


# Split-out validation dataset 
array = dataset.values 
X = array[:,0:13] 
Y = array[:,13] 
validation_size = 0.20 
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

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
models.append(('SVR', SVR()))

results = [] 
names = [] 
for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    
##################################################################33

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
models.append(('SVR', model6))

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
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
    
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
    
    
# GDB Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"n_estimators":[50,100,150,200,250,300,350,400,500,600]}
model = GradientBoostingRegressor(random_state=seed) 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))


scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
model = GradientBoostingRegressor(n_estimators = 300, random_state=seed)         
model.fit(rescaledX, Y_train)

rescaledX_validation = scaler.transform(X_validation) 
pred = model.predict(rescaledX_validation)
print(mean_squared_error(Y_validation, pred))
