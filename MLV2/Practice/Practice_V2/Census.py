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


dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\Census.csv')

sns.countplot(dataset['income'])


dataset.shape
dataset.info()
dataset.sample(5)
dataset.describe()
dataset.groupby('income').size()

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

print((dataset.isnull().sum()/ len(dataset)) * 100)

cnt1 = dataset['workclass'].value_counts()
dataset['workclass'] = dataset['workclass'].apply(lambda x: 'other' if x == '?' else x)

cnt1 = dataset['education'].value_counts()
dataset['education'] = dataset['education'].apply(lambda x: 'other' if cnt1[x] < 1000 else x)

dataset.rename(columns = {'marital.status': 'status'}, inplace=True)
cnt1 = dataset['status'].value_counts()
#dataset['status'] = dataset['status'].apply(lambda x: 'other' if cnt1[x] < 1000 else x)
dataset['status'] = dataset['status'].apply(lambda x: 1 if (x == 'Married-civ-spouse') | (x == 'Married-spouse-absent') | (x == 'Married-AF-spouse') else 0)
dataset['status'] = dataset['status'].apply(lambda x: int(x))


dataset['occupation'] = dataset['occupation'].apply(lambda x: 'other' if x == '?' else x)
cnt1 = dataset['occupation'].value_counts()
dataset['occupation'] = dataset['occupation'].apply(lambda x: 'other' if cnt1[x] < 1000 else x)

cnt1 = dataset['relationship'].value_counts()

cnt1 = dataset['race'].value_counts()

dataset.rename(columns = {'native.country': 'native'}, inplace=True)
dataset['native'] = dataset['native'].apply(lambda x: 'other' if x == '?' else x)
cnt1 = dataset['native'].value_counts()
dataset['native'] = dataset['native'].apply(lambda x: 'other' if cnt1[x] <= 100 else x)

dataset['income'].value_counts()
#dataset['capital.loss'].value_counts()
#
#num_col = dataset.select_dtypes(include = ['int64','float64'])
#sns.distplot(num_col['capital.loss'])
#sns.barplot(x='sex', y='income', data=dataset)
#sns.pairplot(data = num_col, diag_kind='hist', kind='reg')

import phik
from phik import resources, report

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

corr = corr['income'].abs()

print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col] < 0.2]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True) 

col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.7:
            print(f"{col[i]} -{col[j]}- {corr.iloc[i,j]}")
            

dataset.drop(['education', 'sex'], inplace=True, axis=1)            


# Split-out validation dataset 
X = dataset.drop('income', axis=1)
Y = dataset['income']
validation_size = 0.20 
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y)

X_train = pd.get_dummies(X_train,drop_first=True)
X_validation = pd.get_dummies(X_validation, drop_first=True)

#
## Variance check - if variance less than threshold remove them
#from sklearn.feature_selection import VarianceThreshold
#constant_filter = VarianceThreshold(threshold=0.2)
#constant_filter.fit(X_train)
#print(X_train.columns[constant_filter.get_support()])
#constant_columns = [column for column in X_train.columns if column not in X_train.columns[constant_filter.get_support()]]
#X_train.drop(labels=constant_columns, axis=1, inplace=True)
#X_validation.drop(labels=constant_columns, axis=1, inplace=True)


#X_train.skew()
#X_train['capital.gain'] = np.cbrt(X_train['capital.gain'])
#X_train['capital.loss'] = np.cbrt(X_train['capital.loss'])

#Remove outliers -  try remove outliers using np.log
#from scipy import stats
#numeric_cols = X_train.select_dtypes(include=['int64','float64'])
#z = np.abs(stats.zscore(numeric_cols))
#to_drop_rows=[]
#
###method 2
#for j in range(numeric_cols.shape[1]):
#    median = numeric_cols.iloc[:,j].median()
#    for i in range(numeric_cols.shape[0]):        
#        if z[i,j] >= 3:
#            print(f"{i} -{j}")
#            numeric_cols.iloc[i,j] = median #or mean()
#            
#X_train.update(numeric_cols)         
#




# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'accuracy'
#
#sc = StandardScaler()
#sc.fit(X_train)
#rescaled_X = sc.transform(X_train)
#rescaled_XV = sc.transform(X_validation)

#
#from sklearn.feature_selection import RFECV
#model = LinearDiscriminantAnalysis()
#rfecv = RFECV(estimator=model, step=1,cv=10, scoring=scoring)
#rfecv.fit(X_train, Y_train)
#
#print(rfecv.n_features_)
#print(X_train.columns[rfecv.support_])
#
#cl = [col for col in X_train.columns if col not in X_train.columns[rfecv.support_]]
#X_train.drop(cl, axis=1, inplace=True)
#X_validation.drop(cl, axis=1, inplace=True)

#from sklearn.decomposition import PCA
#
#scaler = StandardScaler().fit(X_train) 
#rescaledX = scaler.transform(X_train) 
#rescaled_X_test = scaler.transform(X_validation) 
#pca = PCA(0.90)
#pca.fit(rescaledX)
#rescaledX = pca.transform(rescaledX)
#rescaled_X_test = pca.transform(rescaled_X_test)

# Spot-Check Algorithms 
models = [] 
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
#models.append(('SVM', SVC()))

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
estimators.append(('lr', LogisticRegression(C= 2.0,penalty= 'l1',solver='liblinear'))) 
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
#
#estimators = [] 
#estimators.append(('standardize', StandardScaler())) 
#estimators.append(('SVC', SVC())) 
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


scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], "penalty" : ['l1', 'l2'], "solver" : ['liblinear', 'warn']}
model = LogisticRegression() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

rescaled_X_test = scaler.transform(X_validation) 
model = LogisticRegression(C= 0.1,penalty= 'l1',solver='liblinear')
model.fit(rescaledX, Y_train)

pred = model.predict(rescaled_X_test)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")

#Accurancy: 0.8492624462200369


########### ENSEMBLE #########333
# ensembles 

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

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
param_grid = {"n_estimators":[50,100,150,200,250,300,350]}
model = GradientBoostingClassifier(random_state=seed) 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))     
#Best: 0.867071 using {'n_estimators': 350}

model = GradientBoostingClassifier(n_estimators=350)
model.fit(rescaledX, Y_train)

pred = model.predict(rescaled_X_test)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")

#Accurancy: 0.873540258143823


          