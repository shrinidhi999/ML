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
from sklearn.preprocessing import Imputer
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale =1, color_codes=True) 


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv(r'D:\ML\ML_Rev\Datasets\cal_housing.csv')
dataset.info()

dataset['ocean_proximity'].value_counts()

seed =7
np.random.seed(seed)
X = dataset.drop('median_house_value', axis=1)
y = dataset['median_house_value']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
#-----------------------------------------------------------------

X_train.isnull().sum()
imp = Imputer(strategy='median')

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

X_train[num_cols] = imp.fit_transform(X_train[num_cols])

print(sum(X_train.duplicated(X_train.columns)))
#----------------------------------------------------------------------

#----------------------------------------------------------------------
from scipy.stats import mstats
X_train[num_cols].skew()
#
#df['population'].hist()
#
#df['population'] = pd.Series(mstats.winsorize(df['population'], limits=[0.05, 0.05]) )
#df['population'].hist()

for feature in X_train[num_cols].keys():
    sk = X_train[feature].skew()
    if sk > 2 or sk < -2:
        print(feature)
        X_train[feature] = pd.Series(mstats.winsorize(X_train[feature], limits=[0.05, 0.05]) )

#----------------------------------------------------------------------


outliers = []
lista = []
dict_qua = {}
# For each feature find the data points with extreme high or low values
for feature in X_train[num_cols].keys():
    
   # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(X_train[feature],25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(X_train[feature],75)
    
    dict_qua[feature] = [Q1, Q3]
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3-Q1)   
   
    lista = X_train[~((X_train[feature] >= Q1 - step) & (X_train[feature] <= Q3 + step))].index.tolist()
    outliers.append(lista)
    
    
# Detecting outliers that appear in more than one product
seen = {}
dupes = []

for lista in outliers:
    for index in lista:
        if index not in seen:
            seen[index] = 1
        else:
            if seen[index] == 1:
                dupes.append(index)
            seen[index] += 1
dupes = sorted(dupes)
dupes

con_dataset = X_train.copy()
con_dataset['price'] = y_train
# Removing outliers  
con_dataset = con_dataset.drop(dupes, axis=0).reset_index(drop=True)


X_train = con_dataset.drop('price', axis=1)
y_train = con_dataset['price']

#-------------------------------------------------------    

X_train.skew()

#X_train['total_rooms'] = np.log(X_train['total_rooms'])
#X_train['total_bedrooms']= np.log(X_train['total_bedrooms'])
#X_train['population']= np.log(X_train['population'])
#X_train['households']= np.log(X_train['households'])    

#-------------------------------------------------------    

sc = StandardScaler()
X_train[num_cols] = sc.fit_transform(X_train[num_cols])

#-------------------------------------------------------    
import phik
from phik import resources, report

con_dataset = X_train.copy()
con_dataset['price'] = y_train

corr = con_dataset.phik_matrix()
corr = corr['price'].abs()
print(corr.sort_values())

to_drop_1 = [col for col in corr.index if corr[col]< 0.2]
con_dataset.drop(to_drop_1, axis=1, inplace=True)


X_train = con_dataset.drop('price', axis=1)
y_train = con_dataset['price']

#-------------------------------------------------------    

X_train = pd.get_dummies(X_train, drop_first=True)

#-------------------------------------------------------    
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
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
    val = np.sqrt(-cv_results.mean())
    msg = "%s: %f  (%f)" % (name, val, cv_results.std()) 
    print(cv_results)
    print(msg)
    


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
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
    val = np.sqrt(-cv_results.mean())
    msg = "%s: %f  (%f)" % (name, val, cv_results.std()) 
    print(cv_results)
    print(msg)    
   

# RandomForestRegressor Algorithm tuning  
param_grid = [
        {'n_estimators':[3,	10,	30],'max_features':	[2,	4,	6,	8]},				           {'bootstrap':[False],'n_estimators':[3,	10],'max_features':	[2,	3,	4]}]
model = RandomForestRegressor() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(X_train, y_train)    


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
    

model = RandomForestRegressor(max_features=4, n_estimators=30)
model.fit(X_train, y_train)
#49082.37984421823

#Prep Test set-----------------------------------------------------------------

#NULL checks
num_cols = x_test.select_dtypes(include=['int64', 'float64']).columns

x_test[num_cols] = imp.transform(x_test[num_cols])

print(sum(x_test.duplicated(x_test.columns)))
    
#-----Outlier removal--------------------------------------------------------------

outliers = []
lista = []
# For each feature find the data points with extreme high or low values
for feature in x_test[num_cols].keys():
    
   # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = dict_qua[feature][0]
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = dict_qua[feature][1]
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3-Q1)   
   
    lista = x_test[~((x_test[feature] >= Q1 - step) & (x_test[feature] <= Q3 + step))].index.tolist()
    outliers.append(lista)
    
    
# Detecting outliers that appear in more than one product
seen = {}
dupes = []

for lista in outliers:
    for index in lista:
        if index not in seen:
            seen[index] = 1
        else:
            if seen[index] == 1:
                dupes.append(index)
            seen[index] += 1
dupes = sorted(dupes)
dupes

con_dataset = x_test.copy()
con_dataset['price'] = y_test
# Removing outliers  
con_dataset = con_dataset.drop(dupes, axis=0).reset_index(drop=True)


x_test = con_dataset.drop('price', axis=1)
y_test = con_dataset['price']

#-------------------------------------------------------    

x_test.skew()

#X_train['total_rooms'] = np.log(X_train['total_rooms'])
#X_train['total_bedrooms']= np.log(X_train['total_bedrooms'])
#X_train['population']= np.log(X_train['population'])
#X_train['households']= np.log(X_train['households'])    

#-------------------------------------------------------    

sc = StandardScaler()
x_test[num_cols] = sc.fit_transform(x_test[num_cols])

#-------------------------------------------------------    

con_dataset = x_test.copy()
con_dataset['price'] = y_test

to_drop_1 = [col for col in corr.index if corr[col]< 0.2]
con_dataset.drop(to_drop_1, axis=1, inplace=True)


x_test = con_dataset.drop('price', axis=1)
y_test = con_dataset['price']

#-------------------------------------------------------    

x_test = pd.get_dummies(x_test, drop_first=True)

#-------------------------------------------------------    


model = RandomForestRegressor(max_features=4, n_estimators=30)
model.fit(X_train, y_train)

pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))
print(r2_score(y_test, pred))
