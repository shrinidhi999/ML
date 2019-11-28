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
from sklearn import model_selection
import seaborn as sns
import phik
from phik import resources, report

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True) 


dataset = pd.read_csv('D:\ML\ML_Rev\Datasets\\bank.csv', delimiter=';')

dataset.shape
dataset.info()
dataset.sample(5)
dataset.describe()
dataset.groupby('y').size()

#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')

#Taking care of Missing Data in Dataset
print(dataset.isnull().sum())

################################################################
dataset['duration'] = np.cbrt(dataset['duration'])
dataset['campaign'] = np.log(dataset['campaign'])
dataset['pdays'] = np.cbrt(dataset['pdays'])
dataset['previous'] = np.cbrt(dataset['previous'])

corr = dataset.phik_matrix()
plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


corr = corr['y'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col] < 0.1]
dataset.drop(to_drop_1, axis=1, inplace=True)

corr = dataset.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True) 

col = corr.index
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.7:
            print(f"{col[i]} -{col[j]}- {corr.iloc[i,j]}")

#dataset.drop(['poutcome','cons.conf.idx','cons.price.idx', 'nr.employed','contact','month','euribor3m'],inplace=True,axis=1)            

################################################################
X = dataset.drop('y', axis=1)
y = dataset['y']

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7, stratify=y)

X_train.skew(axis = 0, skipna = True)
X_train['duration'] = np.cbrt(X_train['duration'])
X_train['pdays'] = np.cbrt(X_train['pdays'])
X_train['previous'] = np.cbrt(X_train['previous'])

X_test['duration'] = np.cbrt(X_test['duration'])
X_test['pdays'] = np.cbrt(X_test['pdays'])
X_test['previous'] = np.cbrt(X_test['previous'])


#Remove outliers -  try remove outliers using np.log
from scipy import stats
numeric_cols = X_train.select_dtypes(include=['int64','float64'])
z = np.abs(stats.zscore(numeric_cols))

for j in range(numeric_cols.shape[1]):
    median = numeric_cols.iloc[:,j].median()
    for i in range(numeric_cols.shape[0]):        
        if z[i,j] >= 3:
            print(f"{i} -{j}")
            numeric_cols.iloc[i,j] = median #or mean()
            
X_train.update(numeric_cols)    

numeric_cols = X_test.select_dtypes(include=['int64','float64'])
z = np.abs(stats.zscore(numeric_cols))

for j in range(numeric_cols.shape[1]):
    median = numeric_cols.iloc[:,j].median()
    for i in range(numeric_cols.shape[0]):        
        if z[i,j] >= 3:
            print(f"{i} -{j}")
            numeric_cols.iloc[i,j] = median #or mean()
            
X_test.update(numeric_cols)     

X_train.info()

## To replace categorical values with less than 10 freq

cnt1 = X_train['job'].value_counts()
cnt_indx = cnt1[cnt1 < 1000].index
X_train.loc[X_train['job'].isin(cnt_indx), 'job'] = 'others'   

cnt2 = X_test['job'].value_counts()
cnt_indx = cnt2[cnt2 < 400].index
X_test.loc[X_test['job'].isin(cnt_indx), 'job'] = 'others'   

def age_cal(x):
    if (x >= 10) & (x < 20):
        return 1
    if (x >= 20) & (x < 30):
        return 2
    if (x >= 30) & (x < 40):
        return 3
    if (x >= 40) & (x < 50):
        return 4
    if (x >= 50) & (x < 60):
        return 5
    if (x >= 60) & (x < 70):
        return 6
    if (x >= 70) & (x < 80):
        return 7
    
X_train['age'] = X_train['age'].apply(lambda x: age_cal(int(x)))
X_test['age'] = X_test['age'].apply(lambda x: age_cal(int(x)))


# Variance check - if variance less than threshold remove them
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0.5)
n_cols = dataset.select_dtypes(include=['int64','float64'])
constant_filter.fit(n_cols)
print(n_cols.columns[constant_filter.get_support()])
X_train.drop('previous', axis=1, inplace=True)
X_test.drop('previous', axis=1, inplace=True)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = LogisticRegression() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])

cl = [col for col in X_train.columns if col not in X_train.columns[rfecv.support_]]

X_train.drop(cl, axis=1, inplace=True)
X_test.drop(cl, axis=1, inplace=True)

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
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    
    

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
#
#estimators = [] 
#estimators.append(('standardize', StandardScaler())) 
#estimators.append(('SVC', SVC())) 
#model6 = Pipeline(estimators) 

models =[]
models.append(('LR', model1)) 
models.append(('LDA', model2)) 
models.append(('KNN', model3))
models.append(('CART', model4)) 
models.append(('NB', model5)) 
#models.append(('SVM', model6))   

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
        

# LR Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], "penalty" : ['l1', 'l2'], "solver" : ['liblinear', 'warn']}
model = LogisticRegression() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

rescaled_X_test = scaler.transform(X_test) 
model = LogisticRegression(C= 0.3,penalty= 'l2',solver='liblinear')
model.fit(rescaledX, y_train)

pred = model.predict(rescaled_X_test)

print(f"Accurancy: {accuracy_score(y_test,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,y_test)} ")
print(f"Accurancy: {classification_report(pred,y_test)} ")

#Accurancy: 0.9086861491135757

  ########### ENSEMBLE #########333
# ensembles 
ensembles = [] 
ensembles.append(('AB', AdaBoostClassifier())) 
ensembles.append(('GBM', GradientBoostingClassifier())) 
ensembles.append(('RF', RandomForestClassifier())) 
ensembles.append(('ET', ExtraTreesClassifier())) 

for name, model in ensembles: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    
    
# GDB Algorithm tuning 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
param_grid = {"n_estimators":[50,100,150,200,250,300,350]}
model = GradientBoostingClassifier(random_state=seed) 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1) 
grid_result = grid.fit(rescaledX, y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))     
#Best: 0.903480 using {'n_estimators': 50}
model = GradientBoostingClassifier(n_estimators=50)
model.fit(rescaledX, y_train)

pred = model.predict(rescaled_X_test)

print(f"Accurancy: {accuracy_score(y_test,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,y_test)} ")
print(f"Accurancy: {classification_report(pred,y_test)} ")

#Accurancy: 0.9043956933538412 
