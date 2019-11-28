import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix  
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True) 


dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\sonar.csv', header=None)# -*- coding: utf-8 -*-

dataset.shape
dataset.info()
dataset.sample(5)
dataset.describe()
dataset.groupby(60).size()
print(dataset.groupby(dataset.columns[60]).size())

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

print((dataset.isnull().sum()/ len(dataset)) * 100)


# histograms 
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) 
plt.show()

# density 
dataset.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1) 
plt.show()

corr = dataset.corr(method='pearson').abs()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
    
col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] > 0.7:
            print(f"{i} - {col[i]} -{col[j]} - {corr.iloc[i,j]}")


# Split-out validation dataset 
array = dataset.values 
X = array[:,0:60].astype(float) 
Y = array[:,60] 
validation_size = 0.20 
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'accuracy'


# Spot-Check Algorithms 
models = [] 
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))

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
estimators.append(('lr', LogisticRegression())) 
model1 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis())) 
model2 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('KNeighborsClassifier', KNeighborsClassifier())) 
model3 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('DecisionTreeClassifier', DecisionTreeClassifier())) 
model4 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('GaussianNB', GaussianNB())) 
model5 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('SVC', SVC())) 
model6 = Pipeline(estimators) 

models =[]
models.append(('LR', model1)) 
models.append(('LDA', model2)) 
models.append(('KNN', model3))
models.append(('CART', model4)) 
models.append(('NB', model5)) 
models.append(('SVM', model6))   

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
model = KNeighborsClassifier() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#Best: 0.849398 using {'n_neighbors': 1}    

# SVM Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
model = SVC() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))    

#Best: 0.867470 using {'C': 1.5, 'kernel': 'rbf'}        
    
    ########### ENSEMBLE #########333
# ensembles 
ensembles = [] 
ensembles.append(('AB', AdaBoostClassifier())) 
ensembles.append(('GBM', GradientBoostingClassifier())) 
ensembles.append(('RF', RandomForestClassifier())) 
ensembles.append(('ET', ExtraTreesClassifier())) 

for name, model in ensembles: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)    
    

# GDB Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"n_estimators":[50,100,150,200,250,300,350,400,500,600]}
model = GradientBoostingClassifier(random_state=seed) 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))

#Best: 0.843373 using {'n_estimators': 150}    
    

### Prediction #########    
scalar = StandardScaler().fit(X_train)
scaled_X = scalar.transform(X_train)
model = SVC(C=1.5, kernel='rbf')
model.fit(scaled_X, Y_train)

scaled_X_validation = scalar.transform(X_validation)
pred = model.predict(scaled_X_validation)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")



scalar = StandardScaler().fit(X_train)
scaled_X = scalar.transform(X_train)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(scaled_X, Y_train)

scaled_X_validation = scalar.transform(X_validation)
pred = model.predict(scaled_X_validation)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")



scalar = StandardScaler().fit(X_train)
scaled_X = scalar.transform(X_train)
model = GradientBoostingClassifier(n_estimators=150)
model.fit(scaled_X, Y_train)

scaled_X_validation = scalar.transform(X_validation)
pred = model.predict(scaled_X_validation)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")


