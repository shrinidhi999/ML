import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True) 

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\pulsar_stars.csv')
dataset['target_class'].value_counts()


dataset.isnull().sum()
print(sum(dataset.duplicated(dataset.columns)))

dataset.skew(axis = 0, skipna = True)
dataset[' Excess kurtosis of the integrated profile'] = np.cbrt(dataset[' Excess kurtosis of the integrated profile'])

dataset[' Skewness of the integrated profile'] = np.cbrt(dataset[' Skewness of the integrated profile'])

dataset[' Mean of the DM-SNR curve'] = np.cbrt(dataset[' Mean of the DM-SNR curve'])
dataset[' Skewness of the DM-SNR curve'] = np.cbrt(dataset[' Skewness of the DM-SNR curve'])


num_cols =  dataset.select_dtypes(include=['float64','int64']).columns

sc = StandardScaler()
dataset[num_cols] = sc.fit_transform(dataset[num_cols])


import phik
from phik import resources, report

corr = dataset.phik_matrix()
corr = corr['target_class'].abs()
print(corr.sort_values())

to_drop_1 = [col for col in corr.index if corr[col]<0.1]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  

corr = dataset.phik_matrix()
col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.8:
            print(f"{col[i]} -{col[j]}- {corr.iloc[i,j]}")

#Excess kurtosis of the DM-SNR curve 
#Mean of the DM-SNR curve           



X = dataset.drop('target_class', axis=1)
y = dataset['target_class']

validation_size = 0.20 
seed = 7 
np.random.seed(seed)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed, stratify=y)

scaled_X_train = X_train
num_cols = scaled_X_train.select_dtypes(include=['float64']).columns

sc = StandardScaler()
scaled_X_train[num_cols] = sc.fit_transform(scaled_X_train[num_cols])


scaled_X_test = X_validation
num_cols = scaled_X_test.select_dtypes(include=['float64']).columns

scaled_X_test[num_cols] = sc.fit_transform(scaled_X_test[num_cols])

Y_train = pd.get_dummies(Y_train, drop_first=True)
Y_validation = pd.get_dummies(Y_validation, drop_first=True)

# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'accuracy'

# Spot-Check Algorithms 
models = [] 
models.append(('LR', LogisticRegression())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 

results = [] 
names = [] 
for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, scaled_X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)


   
# Standardize the dataset 
# create pipeline 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('lr', LogisticRegression(C= 2.0,penalty= 'l1',solver='liblinear'))) 
model1 = Pipeline(estimators) 


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


models =[]
models.append(('LR', model1)) 
models.append(('KNN', model3))
models.append(('CART', model4)) 
models.append(('NB', model5)) 

results = [] 

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, scaled_X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)

     

# LR Algorithm tuning 
param_grid = {"C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], "penalty" : ['l1', 'l2'], "solver" : ['liblinear', 'warn']}
model = LogisticRegression() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(scaled_X_train, Y_train)  
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

#Best: 0.978698 using {'C': 1.3, 'penalty': 'l1', 'solver': 'warn'}

model = LogisticRegression(C=1.3, penalty='l1', solver='warn')
model.fit(scaled_X_train, Y_train)

pred = model.predict(scaled_X_test)
print(accuracy_score(Y_validation, pred))


########### ENSEMBLE #########333
# ensembles 

# Standardize the dataset 
# create pipeline 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('rf', RandomForestClassifier())) 
model1 = Pipeline(estimators) 


estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('er', ExtraTreesClassifier())) 
model3 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('ada', AdaBoostClassifier())) 
model4 = Pipeline(estimators) 

estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('gb', GradientBoostingClassifier())) 
model5 = Pipeline(estimators) 


models =[]
models.append(('RF', model1)) 
models.append(('ER', model3))
models.append(('ADA', model4)) 
models.append(('GB', model5)) 

results = [] 

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, scaled_X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)

    