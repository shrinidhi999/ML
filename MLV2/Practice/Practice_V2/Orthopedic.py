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


dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\Orthopedic.csv')
dataset['class'].value_counts()

dataset.shape
dataset.info()
dataset.sample(5)
dataset.describe()
dataset.groupby('class').size()

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

dataset.skew()

# Split-out validation dataset 
X = dataset.drop('class', axis=1)
Y = dataset['class']
validation_size = 0.20 
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y)

#X_train['degree_spondylolisthesis'] = np.cbrt(X_train['degree_spondylolisthesis'])
#X_validation['degree_spondylolisthesis'] = np.cbrt(X_validation['degree_spondylolisthesis'])
#
#X_train['degree_spondylolisthesis'] = X_train['degree_spondylolisthesis'].fillna(X_train['degree_spondylolisthesis'].median())
#
#
#X_validation['degree_spondylolisthesis'] = X_validation['degree_spondylolisthesis'].fillna(X_validation['degree_spondylolisthesis'].median())
#
#X_train.skew()
#X_validation.skew()
#
##Remove outliers -  try remove outliers using np.log
#from scipy import stats
#numeric_cols = X_train.select_dtypes(include=['int64','float64'])
#z = np.abs(stats.zscore(numeric_cols))
#
#for j in range(numeric_cols.shape[1]):
#    median = numeric_cols.iloc[:,j].median()
#    for i in range(numeric_cols.shape[0]):        
#        if z[i,j] >= 3:
#            print(f"{i} -{j}")
#            numeric_cols.iloc[i,j] = median #or mean()
#            
#X_train.update(numeric_cols)    
#
#
#numeric_cols = X_validation.select_dtypes(include=['int64','float64'])
#z = np.abs(stats.zscore(numeric_cols))
#
#for j in range(numeric_cols.shape[1]):
#    median = numeric_cols.iloc[:,j].median()
#    for i in range(numeric_cols.shape[0]):        
#        if z[i,j] >= 3:
#            print(f"{i} -{j}")
#            numeric_cols.iloc[i,j] = median #or mean()
#            
#X_validation.update(numeric_cols)     



# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'accuracy'

chk = X_train[X_train['degree_spondylolisthesis'] ==np.nan]
np.where(np.isnan(X_train['degree_spondylolisthesis']))

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

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = LogisticRegression() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, Y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])

cl = [col for col in X_train.columns if col not in X_train.columns[rfecv.support_]]

X_train.drop(cl, axis=1, inplace=True)
X_validation.drop(cl, axis=1, inplace=True)


# LR Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], "penalty" : ['l1', 'l2'], "solver" : ['liblinear', 'warn']}
model = LogisticRegression() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

rescaled_X_test = scaler.transform(X_validation) 
model = LogisticRegression(C= 2.0,penalty= 'l1',solver='liblinear')
model.fit(rescaledX, Y_train)

pred = model.predict(rescaled_X_test)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")

#Accurancy: 0.9354838709677419


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
param_grid = {"n_estimators":[50,100,150,200,250,300,350]}
model = GradientBoostingClassifier(random_state=seed) 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1) 
grid_result = grid.fit(rescaledX, Y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))     
#Best: 0.818548 using {'n_estimators': 100}

model = GradientBoostingClassifier(n_estimators=100)
model.fit(rescaledX, Y_train)

pred = model.predict(rescaled_X_test)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")

#Accurancy: 0.8225806451612904 
      