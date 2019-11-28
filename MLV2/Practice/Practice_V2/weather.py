import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True) 

from sklearn.preprocessing import Imputer
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import StandardScaler
import phik
from phik import resources, report
from warnings import simplefilter
from scipy.stats import mstats

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\weatherAUS.csv')

print(sum(dataset.duplicated(dataset.columns)))

miss = dataset.isnull().sum() * 100 / len(dataset)
miss[miss == miss[1]].index[0]
miss_col=[]
for i in miss.index:
    if miss[i] > 30:
        c = miss[miss == miss[i]].index[0]
        miss_col.append(i)
  

dataset.info()

dataset = dataset.drop(['Date','Location','Pressure9am', 'Temp9am', 'Temp3pm'], axis=1)

dataset = dataset.drop(miss_col, axis=1)

X = dataset.drop('RainTomorrow', axis=1)
y = dataset['RainTomorrow']

validation_size = 0.20 
seed = 7 
np.random.rand(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, stratify=y, random_state = seed)

le_num = Imputer(strategy='median')
num_cols = X_train.select_dtypes(include=['int64','float64']).columns

X_train[num_cols] = le_num.fit_transform(X_train[num_cols])
X_train.isnull().sum() * 100 / len(dataset)

X_train['WindGustDir'].value_counts() 
X_train['WindDir9am'].value_counts()
X_train['WindDir3pm'].value_counts()


cat_cols = X_train.select_dtypes(include=['object']).columns
cat_mode={}
for i in cat_cols:
    cat_mode[i] = X_train[i].mode()

for i in cat_cols:
    X_train[i].fillna(cat_mode[i][0], inplace=True)
    
X_train.isnull().sum() * 100 / len(dataset)

X_train.info()
print(sum(X_train.duplicated(X_train.columns)))
X_train = X_train.drop_duplicates(X_train.columns, keep='last')

X_train.skew()

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

for feature in X_train[num_cols].keys():
    sk = X_train[feature].skew()
    if sk > 2 or sk < -2:
        X_train[feature] = pd.Series(mstats.winsorize(X_train[feature], limits=[0.05, 0.05]) )

scaled_X_train = X_train        
sc = StandardScaler()
scaled_X_train[num_cols] = sc.fit_transform(scaled_X_train[num_cols])

for i in y_train.index:
    if i not in scaled_X_train.index:
        y_train.drop(i, inplace=True)


dataset_train = scaled_X_train.copy()
dataset_train['RainTomorrow'] = y_train

corr = dataset_train.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


corr = corr['RainTomorrow'].abs()
print(abs(corr).sort_values())
to_drop_1 = [col for col in corr.index if corr[col]<0.1]
dataset_train.drop(to_drop_1, axis=1, inplace=True)

corr = dataset_train.phik_matrix()
#plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.
#p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

col = corr.index
corr_target = corr['RainTomorrow'].abs()
for i in range(len(col)):
    for j in range(i+1, len(col)):
        if corr.iloc[i,j] >= 0.7:
            if corr_target[col[i]] > corr_target[col[j]]:
                print(col[j])
                dataset_train.drop(col[j], axis=1, inplace=True)
            else:
                print(col[i])
                dataset_train.drop(col[i], axis=1, inplace=True)



scaled_X_train = dataset_train.drop('RainTomorrow', axis=1)
y_train = dataset_train['RainTomorrow']

scaled_X_train = pd.get_dummies(scaled_X_train, drop_first=True)

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
    cv_results = cross_val_score(model, scaled_X_train, y_train, cv=kfold, scoring=scoring) 
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
    cv_results = cross_val_score(model, scaled_X_train, y_train, cv=kfold, scoring=scoring) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)