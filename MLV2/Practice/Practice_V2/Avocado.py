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



dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\avocado.csv')
dataset['type'].value_counts()

dataset.info()

dataset.groupby(['year','type']).size()
dataset.groupby(['year', 'type']).size()

dataset.drop(['year', 'Date','Unnamed: 0', 'region'],axis=1, inplace=True)

dataset.isnull().sum()
print(sum(dataset.duplicated(dataset.columns)))

dataset.skew(axis = 0, skipna = True)
dataset['Total Volume'] = np.log(dataset['Total Volume'])
dataset['4046'] = np.cbrt(dataset['4046'])
dataset['4225'] = np.cbrt(dataset['4225'])
dataset['4770'] = np.cbrt(dataset['4770'])
dataset['Total Bags'] = np.cbrt(dataset['Total Bags'])
dataset['Small Bags'] = np.cbrt(dataset['Small Bags'])
dataset['Large Bags'] = np.cbrt(dataset['Large Bags'])
dataset['XLarge Bags'] = np.cbrt(dataset['XLarge Bags'])


X = dataset.drop('type', axis=1)
y = dataset['type']

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

for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, scaled_X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    
model = KNeighborsClassifier()
model.fit(scaled_X_train, Y_train)

pred = model.predict(scaled_X_test)

print(f"Accurancy: {accuracy_score(Y_validation,pred)} ")
print(f"Accurancy: {confusion_matrix(pred,Y_validation)} ")
print(f"Accurancy: {classification_report(pred,Y_validation)} ")    